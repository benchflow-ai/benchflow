#!/usr/bin/env python3
"""Hill-climb a skills folder against a held-out test split, on BenchFlow.

The loop of "Automating eval design and hillclimbing with Claude" (claude.dev,
2026-09-28), built from BenchFlow's public primitives:

- ``bf.Evaluation`` runs a split with the skills deployed (``skills_dir``,
  ``skill_mode="with-skill"``), once per trial, as normal BenchFlow jobs;
- ``bf.load_job`` reads the trials back; ``Job.solve_rates`` gives pass@1, and
  unscored trials (infrastructure errors) are left out of every score;
- ``bf.run(bf.RolloutConfig(...))`` runs the optimizer as a sandboxed rollout
  whose only uploads are the train split's material (hillclimb_proposer.py);
- ``bf.Budget`` caps each job's spend.

The statistics (bootstrap intervals, the noise gate) are in hillclimb_stats.py
and the HTML report in hillclimb_report.py. From a BenchFlow checkout::

    uv run python docs/examples/hillclimb/hillclimb.py --tasks-dir tasks/ \\
        --skills skills/ --out runs/demo --model claude-haiku-4-5 \\
        --proposer-model claude-opus-4-8 --trials 5 --min-gain 0.15

See README.md for the recipe, the safeguards and the outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import benchflow as bf
from hillclimb_proposer import ProposerSettings, mounted, run_optimizer, write_workspace
from hillclimb_report import write_report
from hillclimb_stats import bootstrap, noise_gate, paired_delta

SPLITS = ("train", "test")


@dataclass
class Settings:
    tasks_dir: Path
    skills: Path
    out: Path
    include: list[str] = field(default_factory=list)
    agent: str = "claude-agent-acp"
    model: str | None = "claude-haiku-4-5"
    agent_env: dict[str, str] = field(default_factory=dict)
    sandbox: str = "docker"
    concurrency: int = 4
    trials: int = 3
    rounds: int = 5
    min_gain: float = 0.1
    test_frac: float = 0.3
    seed: int = 0
    split_file: Path | None = None
    stall_rounds: int = 3
    max_cost_usd: float | None = None
    max_infra_error_rate: float = 0.25
    retry_attempts: int = 2
    force: bool = False
    skip_controls: bool = False
    bootstrap_samples: int = 2000
    proposer: ProposerSettings = field(default_factory=ProposerSettings)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def climb(s: Settings) -> dict:
    """Run the climb; return the hillclimb.json record (also written to s.out)."""
    rec = Record(s)
    tasks = find_tasks(s)
    train, test = split_tasks(s, sorted(tasks))
    if not s.skip_controls:  # graders first: the oracle must pass, doing nothing must not
        broken = await check_graders(s, tasks, rec)
        train, test = [t for t in train if t not in broken], [t for t in test if t not in broken]
    rec.doc["split"].update(train=train, test=test)

    baseline = current = await evaluate(s, rec, copy_skills(s.skills, rec.version("v000")), train, test, "baseline")
    rec.doc["baseline"] = baseline.doc
    if breach := infra_breach(s, baseline):
        return rec.finish("stopped", "infra", breach, baseline, current)
    gate = noise_gate(baseline.values["train"], baseline.values["test"], min_gain=s.min_gain,
                      trials=s.trials, samples=s.bootstrap_samples, seed=s.seed)
    rec.doc["noise_gate"] = {**gate, "forced": s.force and not gate["passed"]}
    if not gate["passed"] and not s.force:
        return rec.finish("refused", "noise_gate", gate["message"], baseline, current)

    stall = 0
    for r in range(1, s.rounds + 1):
        if not rec.can_afford(baseline):
            return rec.finish("stopped", "budget", "--max-cost-usd would be exceeded", baseline, current)
        cand = await propose(s, rec, r, current, tasks, train, test)
        entry = {"round": r, "base_version": current.version, "candidate": cand}
        if cand["status"] == "ok":
            ev = await evaluate(s, rec, rec.version(cand["version"]), train, test, cand["id"])
            keep, reasons, entry["train_delta"], entry["test_delta"] = decide(s, current, ev)
            entry.update(evaluation=ev.doc, decision="keep" if keep else "revert", reasons=reasons)
            current = ev if keep else current
        else:
            keep, entry["decision"], entry["reasons"] = False, "invalid", [cand["error"]]
        stall = 0 if keep else stall + 1
        entry.update(version_after=current.version, train_after=current.doc["train"]["score"],
                     test_after=current.doc["test"]["score"])
        rec.doc["rounds"].append(entry)
        rec.save()
        if stall >= s.stall_rounds:  # stalled: sort what is left by root cause
            rec.doc["analysis"] = await analyze(s, rec, current, tasks, train, test)
            return rec.finish("finished", "stalled", f"nothing kept in {stall} rounds", baseline, current)
    return rec.finish("finished", "rounds", f"ran all {s.rounds} rounds", baseline, current)


def decide(s: Settings, current: Scores, ev: Scores) -> tuple[bool, list[str], dict, dict]:
    """The post's rule: keep only if train gains >= min_gain and test improves;
    revert a flat or lower test score (overfitting), any regression, or a rise
    in trials that ended without a score."""
    d = {split: paired_delta(current.values[split], ev.values[split],
                             samples=s.bootstrap_samples, seed=s.seed) for split in SPLITS}
    reasons = [
        f"{split}: trials without a score rose from {current.infra_count(split)} to {ev.infra_count(split)}"
        for split in SPLITS
        if ev.infra_count(split) - current.infra_count(split) > max(2, 0.1 * ev.attempted(split))
    ]
    train, test = d["train"]["value"], d["test"]["value"]
    if train is None or test is None:
        return False, [*reasons, "no task was scored on both sides"], d["train"], d["test"]
    if train < 0:
        reasons.append(f"train regressed ({train:+.3f})")
    elif train < s.min_gain:
        reasons.append(f"train gain {train:+.3f} is below --min-gain {s.min_gain:g}")
    if test < 0:
        reasons.append(f"test regressed ({test:+.3f})")
    elif test == 0:
        reasons.append("test is flat" + (": overfitting suspected" if train >= s.min_gain else ""))
    ok = [f"train {train:+.3f} (>= --min-gain {s.min_gain:g}) and test {test:+.3f}"]
    return not reasons, reasons or ok, d["train"], d["test"]


# ---------------------------------------------------------------------------
# Evaluations: normal BenchFlow jobs, read back with bf.load_job
# ---------------------------------------------------------------------------


@dataclass
class Scores:
    """One skills version on both splits."""

    id: str
    version: str
    values: dict[str, dict[str, list[float]]]  # split -> task -> [1.0 passed / 0.0 failed]
    rows: dict[str, list[dict]]  # split -> one row per trial, run or not
    doc: dict  # what hillclimb.json keeps

    def infra_count(self, split: str) -> int:
        return sum(1 for r in self.rows[split] if r["reward"] is None)

    def attempted(self, split: str) -> int:
        return len(self.rows[split])


async def run_jobs(s: Settings, configs: list[tuple[Path, bf.EvaluationConfig]]) -> None:
    """One Evaluation per (folder, config). On Docker they run one after another:
    until fix/parallel-runs lands, parallel Docker evaluations in one process
    prune each other's just-created containers."""
    jobs = [bf.Evaluation(s.tasks_dir, jobs_dir, config=cfg, job_name="job") for jobs_dir, cfg in configs]
    if s.sandbox == "docker":
        for job in jobs:
            await job.run()
    else:
        await asyncio.gather(*(job.run() for job in jobs))


def config(s: Settings, names: list[str], **overrides) -> bf.EvaluationConfig:
    share = s.concurrency if s.sandbox == "docker" else max(1, s.concurrency // (2 * s.trials))
    return bf.EvaluationConfig(
        agent=s.agent, model=s.model, agent_env=dict(s.agent_env), environment=s.sandbox,
        concurrency=share, include_tasks=set(names),
        retry=bf.RetryConfig(max_retries=s.retry_attempts), **overrides,
    )


async def evaluate(s: Settings, rec: Record, skills: Path, train: list[str], test: list[str], eval_id: str) -> Scores:
    """Run both splits ``s.trials`` times with ``skills`` deployed, and read them back."""
    root = s.out / "evals" / eval_id
    budget = rec.remaining()
    deploy = {"skills_dir": str(skills), "skill_mode": "with-skill",
              "budget": bf.Budget(max_cost_usd=budget) if budget else None}
    await run_jobs(s, [(root / split / f"trial-{k:02d}", config(s, names, **deploy))
                       for split, names in (("train", train), ("test", test))
                       for k in range(1, s.trials + 1)])
    values, rows, doc = {}, {}, {"id": eval_id, "version": skills.parent.name}
    for split, names in (("train", train), ("test", test)):
        job = bf.load_job(root / split)
        rows[split] = trial_rows(job, names, s.trials)
        values[split] = {t: [float(r["passed"]) for r in rows[split] if r["task"] == t and r["reward"] is not None]
                         for t in names}
        rates = job.solve_rates(ks=[1])
        errors = [r for r in rows[split] if r["reward"] is None]
        doc[split] = {
            "job_dir": str(root / split),
            "score": bootstrap(values[split], samples=s.bootstrap_samples, seed=s.seed),
            "pass_at_1": rates.get(1).pass_at_k if rates.get(1) else None,
            "tasks": len(names),
            "trials": len(rows[split]),
            "infra_errors": len(errors),
            "infra_categories": {c: sum(1 for r in errors if r["error"] == c)
                                 for c in sorted({r["error"] for r in errors})},
            "cost_usd": job.cost_usd,
            "per_task": [{"task": t, "solved": values[split][t], "trials": s.trials} for t in names],
        }
        rec.spend(agent=job.cost_usd or 0.0)
    return Scores(eval_id, doc["version"], values, rows, doc)


def trial_rows(job: bf.Job, names: list[str], trials: int) -> list[dict]:
    """One row per (task, trial): a scored trial, an unscored one, or one that never ran."""
    rows, seen = [], set()
    for t in job.agents():
        k = int(t.path.parent.parent.name.split("-")[1])  # .../trial-NN/job/<task>__<id>
        scored = t.assessment == "scored"
        error = None if scored else (t.result.error_category or t.result.verifier_error_category or "unscored")
        rows.append({"task": t.task_name, "trial": k, "reward": t.reward if scored else None,
                     "passed": bool(t.passed) if scored else None, "error": error, "path": str(t.path)})
        seen.add((t.task_name, k))
    rows += [{"task": n, "trial": k, "reward": None, "passed": None, "error": "not run", "path": None}
             for n in names for k in range(1, trials + 1) if (n, k) not in seen]
    return sorted(rows, key=lambda r: (r["task"], r["trial"]))


def infra_breach(s: Settings, ev: Scores) -> str | None:
    errors = sum(ev.infra_count(split) for split in SPLITS)
    total = sum(ev.attempted(split) for split in SPLITS)
    if total and errors / total > s.max_infra_error_rate:
        return (f"{errors} of {total} trials of {ev.id} ended without a score, above "
                f"--max-infra-error-rate {s.max_infra_error_rate:g}: fix the plumbing first")
    return None


async def check_graders(s: Settings, tasks: dict[str, Path], rec: Record) -> set[str]:
    """Run the task's own solution (oracle) and an agent that does nothing (nop)."""
    names = sorted(tasks)
    root = s.out / "controls"
    await run_jobs(s, [(root / agent, config(s, names, agent=agent, model=None)) for agent in ("oracle", "nop")])
    results = {}
    for agent in ("oracle", "nop"):
        results[agent] = {t.task_name: t.reward if t.assessment == "scored" else None
                          for t in bf.load_job(root / agent).trials}
    rows = []
    for name in names:
        oracle, nop = results["oracle"].get(name), results["nop"].get(name)
        flags = [f for f, bad in (("oracle_fails", oracle != 1.0), ("nop_passes", nop == 1.0)) if bad]
        rows.append({"task": name, "oracle": oracle, "nop": nop, "flags": flags})
    broken = {r["task"] for r in rows if r["flags"]}
    rec.doc["controls"] = {"tasks": rows, "excluded": sorted(broken)}
    rec.warn(*(f"grader check: {r['task']} {', '.join(r['flags'])} (excluded)" for r in rows if r["flags"]))
    return broken


# ---------------------------------------------------------------------------
# The optimizer's rounds (the sandboxing itself is in hillclimb_proposer.py)
# ---------------------------------------------------------------------------


async def propose(s: Settings, rec: Record, r: int, current: Scores, tasks: dict[str, Path],
                  train: list[str], test: list[str]) -> dict:
    cid, version = f"r{r:02d}", f"v{r:03d}"
    work = s.out / "proposer" / cid
    uploads = write_workspace(
        work / "workspace", skills=rec.version(current.version),
        train_tasks={t: tasks[t] for t in train},
        failures=[row for row in current.rows["train"] if row["passed"] is False][: s.proposer.max_failures],
        infra=[], scores=rec.scores_for_optimizer(current), history=rec.history())
    seen = check_mounts(s, rec, uploads, tasks, test, work)
    out = await run_optimizer("propose", uploads, settings=s.proposer, task_dir=work / "task", jobs_dir=work)
    rec.spend(proposer=out.get("cost_usd") or 0.0)
    proposal = out.get("output") or {}
    cand = {"id": cid, "version": None, "status": out["status"], "error": out["error"],
            "rollout_dir": out.get("rollout_dir"), "cost_usd": out.get("cost_usd"), "mounted": seen,
            **{k: proposal.get(k) for k in ("root_cause", "change", "rationale")}}
    if out["status"] != "ok":
        return cand
    new = copy_skills(Path(out["surface"]), rec.version(version))
    cand.update(version=version, diff=diff_dirs(rec.version(current.version), rec.version(version)))
    problems = [f"{p.parent.name} has no SKILL.md frontmatter" for p in new.glob("*/SKILL.md")
                if not p.read_text().startswith("---")]
    pasted = pasted_text(cand["diff"], work / "workspace" / "evidence" / "train")
    cand["pasted"] = pasted
    if not cand["diff"].strip() or problems or pasted:
        why = "the optimizer changed nothing" if not cand["diff"].strip() else "; ".join(
            problems or [f"the patch pastes text from {p['source']}" for p in pasted])
        cand.update(status="invalid", error=why)
    return cand


async def analyze(s: Settings, rec: Record, current: Scores, tasks: dict[str, Path],
                  train: list[str], test: list[str]) -> dict:
    work = s.out / "proposer" / "analysis"
    rows = current.rows["train"]
    uploads = write_workspace(
        work / "workspace", skills=rec.version(current.version),
        train_tasks={t: tasks[t] for t in train},
        failures=[r for r in rows if r["passed"] is False][:48],
        infra=[r for r in rows if r["reward"] is None][:48],
        scores=rec.scores_for_optimizer(current), history=rec.history())
    seen = check_mounts(s, rec, uploads, tasks, test, work)
    out = await run_optimizer("analyze", uploads, settings=s.proposer, task_dir=work / "task", jobs_dir=work)
    rec.spend(proposer=out.get("cost_usd") or 0.0)
    data = out.get("output") or {}
    failures = [f for f in data.get("failures") or []
                if isinstance(f, dict) and f.get("category") in CATEGORIES and f.get("id")]
    return {"status": out["status"], "error": out["error"], "mounted": seen,
            "summary": data.get("summary"), "failures": failures,
            "counts": {c: sum(f["category"] == c for f in failures) for c in CATEGORIES},
            "recommendations": [str(x) for x in data.get("recommendations") or []]}


def check_mounts(s: Settings, rec: Record, uploads: dict[str, str], tasks: dict[str, Path],
                 test: list[str], work: Path) -> dict:
    """Record what the optimizer receives; refuse to run it if any test task is in it."""
    seen = mounted(uploads, test_instructions={t: bf.Task(tasks[t]).instruction for t in test},
                   open_network=s.proposer.open_network, manifest=work / "mounted.json")
    if seen["test_tasks_in_paths"]:
        raise RuntimeError(f"test tasks in the optimizer's uploads: {seen['test_tasks_in_paths']}")
    rec.warn(*(f"{work.name}: test task {t}'s instruction appears in train material"
               for t in seen["test_instructions_in_files"]))
    return seen


CATEGORIES = ("ambiguous_task", "grader_bug", "infrastructure", "capability_gap")


# ---------------------------------------------------------------------------
# Small helpers: tasks, split, skills folders, diffs, pasted text
# ---------------------------------------------------------------------------


def find_tasks(s: Settings) -> dict[str, Path]:
    found = {p.name: p.resolve() for p in sorted(s.tasks_dir.iterdir())
             if (p / "task.md").is_file() or (p / "task.toml").is_file()}
    if s.include:
        missing = sorted(set(s.include) - set(found))
        if missing:
            raise SystemExit(f"--include names tasks not in {s.tasks_dir}: {', '.join(missing)}")
        found = {n: p for n, p in found.items() if n in s.include}
    return found


def split_tasks(s: Settings, names: list[str]) -> tuple[list[str], list[str]]:
    """A split file, or a seeded random split with ``round(n * test_frac)`` test tasks."""
    if s.split_file:
        raw = json.loads(Path(s.split_file).read_text())
        return sorted(raw["train"]), sorted(raw["test"])
    shuffled = list(names)
    random.Random(s.seed).shuffle(shuffled)
    n_test = min(max(round(len(names) * s.test_frac), 1), len(names) - 1)
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


def copy_skills(src: Path, dest: Path) -> Path:
    """A skills version: regular files only (symlinks are never uploaded)."""
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest, ignore=lambda d, ns: [n for n in ns if (Path(d) / n).is_symlink()])
    return dest


def diff_dirs(old: Path, new: Path) -> str:
    def files(root: Path) -> dict[str, list[str]]:
        return {p.relative_to(root).as_posix(): p.read_text(errors="replace").splitlines(keepends=True)
                for p in sorted(root.rglob("*")) if p.is_file()}

    a, b = files(old), files(new)
    return "".join("".join(difflib.unified_diff(a.get(f, []), b.get(f, []), f"a/{f}", f"b/{f}"))
                   for f in sorted(set(a) | set(b)))


def pasted_text(diff: str, train_material: Path, words: int = 12) -> list[dict]:
    """Runs of ``words`` words the patch copies from a train instruction or grader output."""
    def runs(text: str) -> set[tuple[str, ...]]:
        w = text.lower().split()
        return {tuple(w[i : i + words]) for i in range(len(w) - words + 1)}

    added = runs("\n".join(ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")))
    hits = []
    for path in sorted(train_material.rglob("*")):
        if path.is_file() and (path.name == "instruction.md" or "verifier" in path.parts):
            if common := added & runs(path.read_text(errors="replace")):
                hits.append({"source": path.relative_to(train_material).as_posix(), "words": " ".join(min(common))})
    return hits


# ---------------------------------------------------------------------------
# hillclimb.json
# ---------------------------------------------------------------------------


class Record:
    """The run's record: rewritten, with report.html, after every phase."""

    def __init__(self, s: Settings) -> None:
        if (s.out / "hillclimb.json").exists():
            raise SystemExit(f"{s.out} already holds a run; pass a new --out")
        s.out.mkdir(parents=True, exist_ok=True)
        self.s, self.agent_usd, self.proposer_usd, self.last_proposer_usd = s, 0.0, 0.0, 0.0
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.doc: dict = {
            "kind": "hillclimb-demo", "schema_version": 1, "status": "running",
            "created_at": now, "updated_at": now,
            "settings": {"tasks_dir": str(s.tasks_dir), "skills": str(s.skills), "agent": s.agent,
                         "model": s.model, "sandbox": s.sandbox, "trials": s.trials, "rounds": s.rounds,
                         "min_gain": s.min_gain, "stall_rounds": s.stall_rounds, "seed": s.seed,
                         "max_cost_usd": s.max_cost_usd, "force": s.force,
                         "proposer": {"agent": s.proposer.agent, "model": s.proposer.model,
                                      "sandbox": s.proposer.sandbox,
                                      "open_network": s.proposer.open_network}},
            "split": {"seed": None if s.split_file else s.seed, "test_frac": s.test_frac,
                      "file": str(s.split_file) if s.split_file else None, "train": [], "test": []},
            "controls": None, "baseline": None, "noise_gate": None, "rounds": [],
            "analysis": None, "best": None, "stop": None, "cost": {}, "warnings": [],
        }
        self.save()

    def version(self, name: str) -> Path:
        return self.s.out / "surfaces" / name

    def spend(self, agent: float = 0.0, proposer: float = 0.0) -> None:
        self.agent_usd += agent
        self.proposer_usd += proposer
        self.last_proposer_usd = proposer or self.last_proposer_usd

    def remaining(self) -> float | None:
        cap = self.s.max_cost_usd
        return None if cap is None else max(cap - self.agent_usd - self.proposer_usd, 0.01)

    def can_afford(self, baseline: Scores) -> bool:
        """A round costs about what the baseline and the last optimizer run cost."""
        cap = self.s.max_cost_usd
        cost = sum(baseline.doc[split]["cost_usd"] or 0.0 for split in SPLITS) + self.last_proposer_usd
        return cap is None or self.agent_usd + self.proposer_usd + cost <= cap

    def warn(self, *messages: str) -> None:
        self.doc["warnings"] += [m for m in messages if m not in self.doc["warnings"]]

    def scores_for_optimizer(self, current: Scores) -> dict:
        """Train in detail; the test split as aggregate scores only."""
        def agg(split_doc: dict) -> dict:
            return {"score": split_doc["score"]["value"], "ci95": split_doc["score"]["ci"], "tasks": split_doc["tasks"]}

        return {
            "note": "Test scores are aggregates over held-out tasks you cannot see.",
            "baseline": {split: agg(self.doc["baseline"][split]) for split in SPLITS},
            "current": {"version": current.version, "test": agg(current.doc["test"]),
                        "train": {**agg(current.doc["train"]), "per_task": current.doc["train"]["per_task"]}},
            "rounds": [{"round": e["round"], "decision": e["decision"], "reasons": e["reasons"],
                        "train_delta": (e.get("train_delta") or {}).get("value"),
                        "test_delta": (e.get("test_delta") or {}).get("value")} for e in self.doc["rounds"]],
        }

    def history(self) -> list[dict]:
        return [{"id": e["candidate"]["id"], "decision": e["decision"], "reasons": e["reasons"],
                 "change": e["candidate"].get("change"), "diff": e["candidate"].get("diff", "")}
                for e in self.doc["rounds"]]

    def save(self) -> None:
        self.doc["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.doc["cost"] = {"agent_usd": round(self.agent_usd, 6), "proposer_usd": round(self.proposer_usd, 6),
                            "total_usd": round(self.agent_usd + self.proposer_usd, 6),
                            "max_cost_usd": self.s.max_cost_usd}
        (self.s.out / "hillclimb.json").write_text(json.dumps(self.doc, indent=2) + "\n")
        write_report(self.doc, self.s.out / "report.html")

    def finish(self, status: str, reason: str, detail: str, baseline: Scores, best: Scores) -> dict:
        """The verdict: the best version against the baseline on the test split."""
        d = paired_delta(baseline.values["test"], best.values["test"], samples=self.s.bootstrap_samples, seed=self.s.seed)
        gated = bool(self.doc["noise_gate"] and self.doc["noise_gate"]["passed"])
        exceeds = best is not baseline and bool(d["ci"]) and d["ci"][0] > 0
        if best is baseline:
            text = "No patch was kept; the baseline stands."
        else:
            text = (f"Best version {best.version}: test {best.doc['test']['score']['value']:.3f} against the "
                    f"baseline's {baseline.doc['test']['score']['value']:.3f}, a change of {d['value']:+.3f} "
                    f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]. " + (
                        "The gain exceeds noise: its 95% interval is above zero. Recommend merging." if exceeds and gated
                        else "Its interval is above zero, but the run was not gated on noise (--force)." if exceeds
                        else "The gain is within noise: its 95% interval includes zero. Recommend against merging."))
        self.doc["best"] = {"version": best.version, "train": best.doc["train"]["score"],
                            "test": best.doc["test"]["score"], "test_delta": d,
                            "verdict": {"exceeds_noise": exceeds, "recommend_merge": exceeds and gated, "text": text}}
        self.doc.update(status=status, stop={"reason": reason, "detail": detail})
        self.save()
        return self.doc


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--skills", type=Path, required=True, help="the skills folder to improve")
    p.add_argument("--out", type=Path, required=True, help="a new run folder")
    p.add_argument("--include", action="append", default=[], help="only these tasks; repeatable")
    p.add_argument("--agent", default="claude-agent-acp")
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--sandbox", default="docker")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--min-gain", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-file", type=Path)
    p.add_argument("--stall-rounds", type=int, default=3)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--max-infra-error-rate", type=float, default=0.25)
    p.add_argument("--force", action="store_true", help="climb although the noise gate refuses")
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--proposer-agent", default="claude-agent-acp")
    p.add_argument("--proposer-model")
    p.add_argument("--proposer-timeout", type=int, default=1800)
    a = p.parse_args()
    s = Settings(
        tasks_dir=a.tasks_dir, skills=a.skills, out=a.out, include=a.include, agent=a.agent, model=a.model,
        sandbox=a.sandbox, concurrency=a.concurrency, trials=a.trials, rounds=a.rounds, min_gain=a.min_gain,
        test_frac=a.test_frac, seed=a.seed, split_file=a.split_file, stall_rounds=a.stall_rounds,
        max_cost_usd=a.max_cost_usd, max_infra_error_rate=a.max_infra_error_rate, force=a.force,
        skip_controls=a.skip_controls,
        proposer=ProposerSettings(agent=a.proposer_agent, model=a.proposer_model, sandbox=a.sandbox,
                                  timeout_sec=a.proposer_timeout),
    )
    doc = asyncio.run(climb(s))
    print(f"{doc['status']}: {doc['stop']['detail']}")
    print(doc["best"]["verdict"]["text"] if doc["best"] else "")
    print(f"record: {s.out / 'hillclimb.json'}\nreport: {s.out / 'report.html'}")
    raise SystemExit({"refused": 2, "stopped": 1}.get(doc["status"], 0))


if __name__ == "__main__":
    main()

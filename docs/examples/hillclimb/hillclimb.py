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
- ``bf.Budget`` caps each job's spend and sandbox-seconds.

The statistics (bootstrap intervals, the noise gate) are in hillclimb_stats.py,
the cost of each rollout (from Claude Code's session log) in hillclimb_cost.py
and the HTML report in hillclimb_report.py. From a BenchFlow checkout::

    uv run python docs/examples/hillclimb/hillclimb.py --tasks-dir tasks/ \\
        --skills skills/ --out runs/demo --model claude-haiku-4-5-20251001 \\
        --proposer-model claude-opus-5-5 --trials 5 --min-gain 0.15

See README.md for the recipe, the safeguards and the outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from hillclimb_cost import CAPTURE, scrub, summary, trial_cost
from hillclimb_pool import QUOTA, NoHeadroom, Pool
from hillclimb_proposer import ProposerSettings, mounted, run_optimizer, write_workspace
from hillclimb_report import write_report
from hillclimb_stats import bootstrap, noise_gate, paired_delta

import benchflow as bf

SPLITS = ("train", "test")
# Each evaluation trial copies Claude Code's session log into its folder
# (hillclimb_cost.CAPTURE), through a task setup command.
SESSION_CAPTURE = {
    "sandbox": {"setup_commands": [{"command": CAPTURE, "timeout_sec": 60}]}
}


@dataclass
class Settings:
    tasks_dir: Path
    skills: Path
    out: Path
    include: list[str] = field(default_factory=list)
    agent: str = "claude-agent-acp"
    model: str | None = "claude-haiku-4-5-20251001"
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
    # Caps that bind when no USD is known (a subscription login): the rollouts
    # that call a model (evaluation trials and optimizer runs), and sandbox
    # wall-clock seconds (the grader checks included).
    max_rollouts: int | None = None
    max_sandbox_seconds: float | None = None
    session_logs: bool = True  # copy Claude Code's session log into each trial
    # Claude subscriptions to spread the jobs over (hillclimb_pool); a trial
    # that ends on an account's usage limit runs again on another, this often.
    pool: Pool | None = None
    quota_retries: int = 2
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
    if (
        not s.skip_controls
    ):  # graders first: the oracle must pass, doing nothing must not
        train, test = await check_graders(s, rec, tasks, train, test)
    rec.doc["split"].update(train=train, test=test)
    if over := rec.over_cap(rollouts=(len(train) + len(test)) * s.trials):
        return rec.finish("stopped", "budget", f"the baseline alone: {over}")

    v0 = copy_skills(s.skills, rec.version("v000"))
    rec.baseline = rec.current = await evaluate(s, rec, v0, train, test, "baseline")
    rec.doc["baseline"] = rec.baseline.doc
    if breach := infra_breach(s, rec.baseline):
        return rec.finish("stopped", "infra", breach)
    gate = check_noise(s, rec)
    if not gate["passed"] and not s.force:
        return rec.finish("refused", "noise_gate", gate["message"])

    stall = 0
    for r in range(1, s.rounds + 1):
        if over := rec.over_cap(**rec.round_estimate()):
            return rec.finish("stopped", "budget", f"round {r}: {over}")
        entry = await one_round(s, rec, r, tasks, train, test)
        stall = 0 if entry["decision"] == "keep" else stall + 1
        if stall >= s.stall_rounds:  # stalled: sort what is left by root cause
            rec.doc["analysis"] = await analyze(s, rec, tasks, train, test)
            return rec.finish("finished", "stalled", f"nothing kept in {stall} rounds")
    return rec.finish("finished", "rounds", f"ran all {s.rounds} rounds")


async def one_round(
    s: Settings,
    rec: Record,
    r: int,
    tasks: dict[str, Path],
    train: list[str],
    test: list[str],
) -> dict:
    """Propose one patch from the train failures, evaluate it, keep or revert it."""
    current = rec.current
    cand = await propose(s, rec, r, current, tasks, train, test)
    entry = {"round": r, "base_version": current.version, "candidate": cand}
    if cand["status"] == "ok":
        skills = rec.version(cand["version"])
        ev = await evaluate(s, rec, skills, train, test, cand["id"])
        keep, reasons, entry["train_delta"], entry["test_delta"] = decide(
            s, current, ev
        )
        entry.update(evaluation=ev.doc, reasons=reasons)
        entry["decision"] = "keep" if keep else "revert"
        rec.current = ev if keep else current
    else:
        entry.update(decision="invalid", reasons=[cand["error"]])
    after = rec.current
    entry["version_after"] = after.version
    entry["train_after"] = after.doc["train"]["score"]
    entry["test_after"] = after.doc["test"]["score"]
    rec.doc["rounds"].append(entry)
    rec.save()
    return entry


def check_noise(s: Settings, rec: Record) -> dict:
    """The noise gate on the baseline (hillclimb_stats.noise_gate), recorded."""
    base = rec.baseline.values
    gate = noise_gate(
        base["train"],
        base["test"],
        min_gain=s.min_gain,
        trials=s.trials,
        samples=s.bootstrap_samples,
        seed=s.seed,
    )
    rec.doc["noise_gate"] = {**gate, "forced": s.force and not gate["passed"]}
    return gate


def decide(
    s: Settings, current: Scores, ev: Scores
) -> tuple[bool, list[str], dict, dict]:
    """The post's rule: keep only if train gains >= min_gain and test improves;
    revert a flat or lower test score (overfitting), any regression, or a rise
    in trials that ended without a score."""
    d = {
        split: paired_delta(
            current.values[split],
            ev.values[split],
            samples=s.bootstrap_samples,
            seed=s.seed,
        )
        for split in SPLITS
    }
    reasons = [
        f"{split}: trials without a score rose from {current.infra_count(split)} to {ev.infra_count(split)}"
        for split in SPLITS
        if ev.infra_count(split) - current.infra_count(split)
        > max(2, 0.1 * ev.attempted(split))
    ]
    train, test = d["train"]["value"], d["test"]["value"]
    if train is None or test is None:
        return (
            False,
            [*reasons, "no task was scored on both sides"],
            d["train"],
            d["test"],
        )
    if train < 0:
        reasons.append(f"train regressed ({train:+.3f})")
    elif train < s.min_gain:
        reasons.append(f"train gain {train:+.3f} is below --min-gain {s.min_gain:g}")
    if test < 0:
        reasons.append(f"test regressed ({test:+.3f})")
    elif test == 0:
        reasons.append(
            "test is flat" + (": overfitting suspected" if train >= s.min_gain else "")
        )
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
    values: dict[
        str, dict[str, list[float]]
    ]  # split -> task -> [1.0 passed / 0.0 failed]
    rows: dict[str, list[dict]]  # split -> one row per trial, run or not
    doc: dict  # what hillclimb.json keeps

    def infra_count(self, split: str) -> int:
        return sum(1 for r in self.rows[split] if r["reward"] is None)

    def attempted(self, split: str) -> int:
        return len(self.rows[split])


async def run_jobs(
    s: Settings, rec: Record, configs: list[tuple[Path, bf.EvaluationConfig]]
) -> list[dict]:
    """One Evaluation per (folder, config), all at once. With an account pool
    (hillclimb_pool), an agent job first leases an account and runs on its
    token. Per job: its folder, its sandbox-seconds as its budget counted them
    (retried attempts included; None without a budget), its account's name,
    and whether it ended on that account's usage limit."""

    async def one(jobs_dir: Path, cfg: bf.EvaluationConfig) -> dict:
        run = {"job": jobs_dir, "spent": None, "account": None, "quota": False}
        account, pooled = None, s.pool and cfg.agent not in ("oracle", "nop")
        if pooled:
            label = jobs_dir.relative_to(s.out).as_posix()
            try:
                account = await s.pool.lease(
                    label, kind="agent", rollouts=len(cfg.include_tasks)
                )
            except NoHeadroom as exc:
                rec.warn(f"{label} not run: no Claude account has headroom ({exc})")
                return run
            token = {"CLAUDE_CODE_OAUTH_TOKEN": account.token}
            cfg = replace(cfg, agent_env={**cfg.agent_env, **token})
            run["account"] = account.name
        try:
            result = await bf.Evaluation(
                s.tasks_dir, jobs_dir, config=cfg, job_name="job"
            ).run()
            budget = getattr(result, "budget", None)
            run["spent"] = budget and float(budget["spent"]["sandbox_seconds"])
            run["quota"] = bool(pooled and quota_failed(jobs_dir))
        finally:
            if account:
                await s.pool.release(
                    account,
                    kind="agent",
                    rollouts=len(cfg.include_tasks),
                    quota=run["quota"],
                )
        return run

    return list(await asyncio.gather(*(one(d, c) for d, c in configs)))


def quota_failed(jobs_dir: Path) -> list[str]:
    """The tasks of a job whose last attempt ended on the account's usage limit."""
    try:
        job = bf.load_job(jobs_dir / "job")
    except FileNotFoundError:
        return []
    return sorted(
        t.task_name for t in job.agents() if QUOTA.search(t.result.error or "")
    )


def sandbox_seconds(runs: list[dict], rows: list[dict]) -> float:
    """The larger of what the jobs' budgets counted (retried attempts included;
    only with a budget) and the final trials' own wall-clock."""
    spent = sum(r["spent"] for r in runs if r["spent"] is not None)
    return max(spent, sum(r["seconds"] for r in rows))


def session_capture(s: Settings, rec: Record, names: list[str]) -> bool:
    """Whether the evaluations get SESSION_CAPTURE: an override replaces a
    task's own setup commands, so not when a task declares some."""
    own = [n for n in names if bf.Task(s.tasks_dir / n).config.sandbox.setup_commands]
    if s.session_logs and own:
        rec.warn(
            f"no session logs (so no cost under a subscription): {', '.join(own)} "
            "declare their own setup commands, which the capture would replace"
        )
    return s.session_logs and not own


def config(s: Settings, names: list[str], **overrides) -> bf.EvaluationConfig:
    return bf.EvaluationConfig(
        **{
            "agent": s.agent,
            "model": s.model,
            "agent_env": dict(s.agent_env),
            "environment": s.sandbox,
            "concurrency": max(1, s.concurrency // (2 * s.trials)),
            "include_tasks": set(names),
            "retry": bf.RetryConfig(max_retries=s.retry_attempts),
            **overrides,
        }
    )


async def evaluate(
    s: Settings,
    rec: Record,
    skills: Path,
    train: list[str],
    test: list[str],
    eval_id: str,
) -> Scores:
    """Run both splits ``s.trials`` times with ``skills`` deployed, and read them back."""
    root = s.out / "evals" / eval_id
    planned = (len(train) + len(test)) * s.trials
    deploy = {
        "skills_dir": str(skills),
        "skill_mode": "with-skill",
        "config_override": SESSION_CAPTURE
        if session_capture(s, rec, train + test)
        else None,
    }

    def jobs(folders: list[tuple[Path, list[str]]]) -> list:
        return [
            (d, config(s, names, budget=rec.job_budget(len(names) / planned), **deploy))
            for d, names in folders
        ]

    runs = await run_jobs(
        s,
        rec,
        jobs(
            [
                (root / split / f"trial-{k:02d}", names)
                for split, names in (("train", train), ("test", test))
                for k in range(1, s.trials + 1)
            ]
        ),
    )
    # A trial that ended on an account's usage limit runs again on another
    # account, in trial-NN/retry-N/ (trial_rows keeps its last attempt).
    last = runs
    for n in range(1, s.quota_retries + 1):
        again = [
            (trial / f"retry-{n}", tasks)
            for r in last
            if r["quota"] and (tasks := quota_failed(r["job"]))
            for trial in [r["job"] if n == 1 else r["job"].parent]
        ]
        if not again:
            break
        last = await run_jobs(s, rec, jobs(again))
        runs += last
    values, rows, doc = {}, {}, {"id": eval_id, "version": skills.name}
    for split, names in (("train", train), ("test", test)):
        try:
            job = bf.load_job(root / split)
        except FileNotFoundError:  # every trial was cancelled (budget)
            job = None
        secrets = s.pool.secrets() if s.pool else ()
        rows[split] = trial_rows(job, names, s.trials, secrets)
        ran = [r for r in rows[split] if r["path"]]
        cost = summary([r["cost"] for r in ran])
        mine = [r for r in runs if (root / split) in r["job"].parents]
        seconds = sandbox_seconds(mine, ran)
        account = {r["job"]: r["account"] for r in mine}  # a trial's job folder's
        for row in ran:
            rec.charge(account.get(Path(row["path"]).parent.parent), row["cost"]["usd"])
        rec.spend(
            agent=cost["usd"] or 0.0,
            rollouts=len(job.agents()) if job else 0,  # quota-retried attempts too
            seconds=seconds,
            costs=[r["cost"] for r in ran],
        )
        values[split] = {
            t: [
                float(r["passed"])
                for r in rows[split]
                if r["task"] == t and r["reward"] is not None
            ]
            for t in names
        }
        rates = job.solve_rates(ks=[1]) if job else None
        errors = [r for r in rows[split] if r["reward"] is None]
        doc[split] = {
            "job_dir": str(root / split),
            "score": bootstrap(values[split], samples=s.bootstrap_samples, seed=s.seed),
            "pass_at_1": rates.get(1).pass_at_k if rates and rates.get(1) else None,
            "tasks": len(names),
            "trials": len(rows[split]),
            "infra_errors": len(errors),
            "infra_categories": {
                c: sum(1 for r in errors if r["error"] == c)
                for c in sorted({r["error"] for r in errors})
            },
            "cost_usd": cost["usd"],
            "cost_source": cost["source"],
            "cost_sources": cost["sources"],
            "sandbox_seconds": round(seconds, 1),
            # Which account ran each job (its name, never its token).
            "jobs": [
                {
                    "job": r["job"].relative_to(root / split).as_posix(),
                    "account": r["account"],
                    "quota": r["quota"],
                }
                for r in mine
            ],
            "per_task": [
                {"task": t, "solved": values[split][t], "trials": s.trials}
                for t in names
            ],
        }
    return Scores(eval_id, doc["version"], values, rows, doc)


def trial_rows(
    job: bf.Job | None, names: list[str], trials: int, secrets: Iterable[str] = ()
) -> list[dict]:
    """One row per (task, trial): a scored trial, an unscored one, or one that
    never ran; with what it cost (hillclimb_cost) and its sandbox wall-clock.
    A trial that ran again after a usage limit (trial-NN/retry-N/) keeps its
    scored attempt, else its last one."""
    best: dict[tuple[str, int], dict] = {}
    for t in job.agents() if job else []:
        # .../trial-NN/job/<task>__<id> or .../trial-NN/retry-N/job/<task>__<id>
        k = next(int(p[6:]) for p in t.path.parts if p.startswith("trial-"))
        scored = t.assessment == "scored"
        error = (
            None
            if scored
            else "usage limit"
            if QUOTA.search(t.result.error or "")
            else (
                t.result.error_category
                or t.result.verifier_error_category
                or "unscored"
            )
        )
        scrub(t.path, secrets)
        row = {
            "task": t.task_name,
            "trial": k,
            "reward": t.reward if scored else None,
            "passed": bool(t.passed) if scored else None,
            "error": error,
            "path": str(t.path),
            "cost": trial_cost(t.path, t.cost_usd),
            "seconds": float(t.timing.get("total") or t.duration_sec or 0.0),
        }
        old = best.get((t.task_name, k))
        if old is None or (scored, row["path"]) > (
            old["reward"] is not None,
            old["path"],
        ):
            best[(t.task_name, k)] = row
    rows, seen = list(best.values()), set(best)
    rows += [
        {
            "task": n,
            "trial": k,
            "reward": None,
            "passed": None,
            "error": "not run",
            "path": None,
            "cost": None,
            "seconds": 0.0,
        }
        for n in names
        for k in range(1, trials + 1)
        if (n, k) not in seen
    ]
    return sorted(rows, key=lambda r: (r["task"], r["trial"]))


def infra_breach(s: Settings, ev: Scores) -> str | None:
    errors = sum(ev.infra_count(split) for split in SPLITS)
    total = sum(ev.attempted(split) for split in SPLITS)
    if total and errors / total > s.max_infra_error_rate:
        return (
            f"{errors} of {total} trials of {ev.id} ended without a score, above "
            f"--max-infra-error-rate {s.max_infra_error_rate:g}: fix the plumbing first"
        )
    return None


async def check_graders(
    s: Settings, rec: Record, tasks: dict[str, Path], train: list[str], test: list[str]
) -> tuple[list[str], list[str]]:
    """Run the task's own solution (oracle) and an agent that does nothing (nop)."""
    names = sorted(tasks)
    root = s.out / "controls"
    # Each control job's share of the caps: two control runs, then the
    # baseline's trials, each about one control run's worth.
    share = 1 / (2 + s.trials)
    runs = await run_jobs(
        s,
        rec,
        [
            (
                root / agent,
                config(s, names, agent=agent, model=None, budget=rec.job_budget(share)),
            )
            for agent in ("oracle", "nop")
        ],
    )
    results = {}
    for run, agent in zip(runs, ("oracle", "nop"), strict=True):
        trials = bf.load_job(root / agent).trials
        timing = [{"seconds": float(t.timing.get("total") or 0.0)} for t in trials]
        rec.spend(seconds=sandbox_seconds([run], timing))
        results[agent] = {
            t.task_name: t.reward if t.assessment == "scored" else None for t in trials
        }
    rows = []
    for name in names:
        oracle, nop = results["oracle"].get(name), results["nop"].get(name)
        flags = [
            f
            for f, bad in (("oracle_fails", oracle != 1.0), ("nop_passes", nop == 1.0))
            if bad
        ]
        rows.append({"task": name, "oracle": oracle, "nop": nop, "flags": flags})
    broken = {r["task"] for r in rows if r["flags"]}
    rec.doc["controls"] = {"tasks": rows, "excluded": sorted(broken)}
    rec.warn(
        *(
            f"grader check: {r['task']} {', '.join(r['flags'])} (excluded)"
            for r in rows
            if r["flags"]
        )
    )
    return [t for t in train if t not in broken], [t for t in test if t not in broken]


# ---------------------------------------------------------------------------
# The optimizer's rounds (the sandboxing itself is in hillclimb_proposer.py)
# ---------------------------------------------------------------------------


async def propose(
    s: Settings,
    rec: Record,
    r: int,
    current: Scores,
    tasks: dict[str, Path],
    train: list[str],
    test: list[str],
) -> dict:
    cid, version = f"r{r:02d}", f"v{r:03d}"
    work = s.out / "proposer" / cid
    uploads = write_workspace(
        work / "workspace",
        skills=rec.version(current.version),
        train_tasks={t: tasks[t] for t in train},
        failures=[row for row in current.rows["train"] if row["passed"] is False][
            : s.proposer.max_failures
        ],
        infra=[],
        scores=rec.scores_for_optimizer(current),
        history=rec.history(),
    )
    seen = check_mounts(s, rec, uploads, tasks, test, work)
    out = await optimize(s, rec, "propose", uploads, work)
    proposal = out.get("output") or {}
    cand = {
        "id": cid,
        "version": None,
        "status": out["status"],
        "error": out["error"],
        "rollout_dir": out.get("rollout_dir"),
        "cost_usd": out.get("cost_usd"),
        "cost_source": out.get("cost_source"),
        "sandbox_seconds": out.get("sandbox_seconds"),
        "accounts": out["accounts"],
        "mounted": seen,
        **{k: proposal.get(k) for k in ("root_cause", "change", "rationale")},
    }
    if out["status"] != "ok":
        return cand
    new = copy_skills(Path(out["surface"]), rec.version(version))
    cand.update(
        version=version,
        diff=diff_dirs(rec.version(current.version), rec.version(version)),
    )
    problems = [
        f"{p.parent.name} has no SKILL.md frontmatter"
        for p in new.glob("*/SKILL.md")
        if not p.read_text().startswith("---")
    ]
    pasted = pasted_text(cand["diff"], work / "workspace" / "evidence" / "train")
    cand["pasted"] = pasted
    if not cand["diff"].strip() or problems or pasted:
        why = (
            "the optimizer changed nothing"
            if not cand["diff"].strip()
            else "; ".join(
                problems
                or [f"the patch pastes text from {p['source']}" for p in pasted]
            )
        )
        cand.update(status="invalid", error=why)
    return cand


async def analyze(
    s: Settings,
    rec: Record,
    tasks: dict[str, Path],
    train: list[str],
    test: list[str],
) -> dict:
    work = s.out / "proposer" / "analysis"
    current = rec.current
    rows = current.rows["train"]
    if over := rec.over_cap(
        rollouts=1, usd=rec.last_proposer_usd, seconds=rec.last_proposer_seconds
    ):
        return {
            "status": "skipped",
            "error": f"not run: {over}",
            "mounted": None,
            "summary": None,
            "failures": [],
            "counts": dict.fromkeys(CATEGORIES, 0),
            "recommendations": [],
        }
    uploads = write_workspace(
        work / "workspace",
        skills=rec.version(current.version),
        train_tasks={t: tasks[t] for t in train},
        failures=[r for r in rows if r["passed"] is False][:48],
        infra=[r for r in rows if r["reward"] is None][:48],
        scores=rec.scores_for_optimizer(current),
        history=rec.history(),
    )
    seen = check_mounts(s, rec, uploads, tasks, test, work)
    out = await optimize(s, rec, "analyze", uploads, work)
    data = out.get("output") or {}
    failures = [
        f
        for f in data.get("failures") or []
        if isinstance(f, dict) and f.get("category") in CATEGORIES and f.get("id")
    ]
    return {
        "status": out["status"],
        "error": out["error"],
        "mounted": seen,
        "accounts": out["accounts"],
        "summary": data.get("summary"),
        "failures": failures,
        "counts": {c: sum(f["category"] == c for f in failures) for c in CATEGORIES},
        "recommendations": [str(x) for x in data.get("recommendations") or []],
    }


async def optimize(
    s: Settings, rec: Record, mode: str, uploads: dict[str, str], work: Path
) -> dict:
    """One optimizer rollout (hillclimb_proposer.run_optimizer), counted. With an
    account pool it runs on a leased account, and a run that ends on the
    account's usage limit runs again on another."""
    settings = replace(s.proposer, timeout_sec=rec.proposer_timeout())
    accounts: list[str] = []
    for _ in range(1 + (s.quota_retries if s.pool else 0)):
        account = None
        if s.pool:
            try:
                account = await s.pool.lease(
                    f"proposer/{work.name}", kind="optimizer", rollouts=1
                )
            except NoHeadroom as exc:
                return {
                    "status": "failed",
                    "error": f"no Claude account has headroom ({exc})",
                    "accounts": accounts,
                }
            token = {"CLAUDE_CODE_OAUTH_TOKEN": account.token}
            settings = replace(settings, agent_env={**s.proposer.agent_env, **token})
            accounts.append(account.name)
        quota = False
        try:
            out = await run_optimizer(
                mode, uploads, settings=settings, task_dir=work / "task", jobs_dir=work
            )
            quota = bool(QUOTA.search(out.get("error") or ""))
        finally:
            if account:
                await s.pool.release(account, kind="optimizer", rollouts=1, quota=quota)
        rec.spend_optimizer(out)
        rec.charge(account and account.name, out.get("cost_usd"))
        if not quota:
            break
    return {**out, "accounts": accounts}


def check_mounts(
    s: Settings,
    rec: Record,
    uploads: dict[str, str],
    tasks: dict[str, Path],
    test: list[str],
    work: Path,
) -> dict:
    """Record what the optimizer receives; refuse to run it if any test task is in it."""
    seen = mounted(
        uploads,
        test_instructions={t: bf.Task(tasks[t]).instruction for t in test},
        open_network=s.proposer.open_network,
        manifest=work / "mounted.json",
    )
    seen["manifest"] = (work / "mounted.json").relative_to(s.out).as_posix()
    if seen["test_tasks_in_paths"]:
        raise RuntimeError(
            f"test tasks in the optimizer's uploads: {seen['test_tasks_in_paths']}"
        )
    rec.warn(
        *(
            f"{work.name}: test task {t}'s instruction appears in train material"
            for t in seen["test_instructions_in_files"]
        )
    )
    return seen


CATEGORIES = ("ambiguous_task", "grader_bug", "infrastructure", "capability_gap")


# ---------------------------------------------------------------------------
# Small helpers: tasks, split, skills folders, diffs, pasted text
# ---------------------------------------------------------------------------


def find_tasks(s: Settings) -> dict[str, Path]:
    found = {
        p.name: p.resolve()
        for p in sorted(s.tasks_dir.iterdir())
        if (p / "task.md").is_file() or (p / "task.toml").is_file()
    }
    if s.include:
        missing = sorted(set(s.include) - set(found))
        if missing:
            raise SystemExit(
                f"--include names tasks not in {s.tasks_dir}: {', '.join(missing)}"
            )
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
    shutil.copytree(
        src, dest, ignore=lambda d, ns: [n for n in ns if (Path(d) / n).is_symlink()]
    )
    return dest


def diff_dirs(old: Path, new: Path) -> str:
    def files(root: Path) -> dict[str, list[str]]:
        return {
            p.relative_to(root).as_posix(): p.read_text(errors="replace").splitlines(
                keepends=True
            )
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    a, b = files(old), files(new)
    return "".join(
        "".join(difflib.unified_diff(a.get(f, []), b.get(f, []), f"a/{f}", f"b/{f}"))
        for f in sorted(set(a) | set(b))
    )


def pasted_text(diff: str, train_material: Path, words: int = 12) -> list[dict]:
    """Runs of ``words`` words the patch copies from a train instruction or grader output."""

    def runs(text: str) -> set[tuple[str, ...]]:
        w = text.lower().split()
        return {tuple(w[i : i + words]) for i in range(len(w) - words + 1)}

    added = runs(
        "\n".join(
            ln[1:]
            for ln in diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")
        )
    )
    hits = []
    for path in sorted(train_material.rglob("*")):
        read = path.is_file() and (
            path.name == "instruction.md" or "verifier" in path.parts
        )
        if read and (common := added & runs(path.read_text(errors="replace"))):
            hits.append(
                {
                    "source": path.relative_to(train_material).as_posix(),
                    "words": " ".join(min(common)),
                }
            )
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
        self.s, self.agent_usd, self.proposer_usd, self.last_proposer_usd = (
            s,
            0.0,
            0.0,
            0.0,
        )
        self.rollouts, self.sandbox_seconds, self.last_proposer_seconds = 0, 0.0, 0.0
        self.costs: list[dict] = []  # hillclimb_cost.trial_cost of every rollout
        self.account_usd: dict[str, float] = {}  # per account of the pool
        self.baseline: Scores | None = None  # set once the baseline has run
        self.current: Scores | None = None  # the version rounds build on
        now = datetime.now(UTC).isoformat(timespec="seconds")
        model_env = s.proposer.agent_env.get("ANTHROPIC_MODEL")  # --proposer-model-env
        self.doc: dict = {
            "kind": "hillclimb-demo",
            "schema_version": 1,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "settings": {
                "tasks_dir": str(s.tasks_dir),
                "skills": str(s.skills),
                "agent": s.agent,
                "model": s.model,
                "sandbox": s.sandbox,
                "trials": s.trials,
                "rounds": s.rounds,
                "min_gain": s.min_gain,
                "stall_rounds": s.stall_rounds,
                "seed": s.seed,
                "max_cost_usd": s.max_cost_usd,
                "max_rollouts": s.max_rollouts,
                "max_sandbox_seconds": s.max_sandbox_seconds,
                "session_logs": s.session_logs,
                "oauth_pool": sorted(s.pool.accounts) if s.pool else None,
                "force": s.force,
                "proposer": {
                    "agent": s.proposer.agent,
                    "model": s.proposer.model or model_env,
                    "model_via": "ANTHROPIC_MODEL"
                    if model_env and not s.proposer.model
                    else "acp",
                    "sandbox": s.proposer.sandbox,
                    "open_network": s.proposer.open_network,
                },
            },
            "split": {
                "seed": None if s.split_file else s.seed,
                "test_frac": s.test_frac,
                "file": str(s.split_file) if s.split_file else None,
                "train": [],
                "test": [],
            },
            "controls": None,
            "baseline": None,
            "noise_gate": None,
            "rounds": [],
            "analysis": None,
            "accounts": None,
            "best": None,
            "stop": None,
            "cost": {},
            "warnings": [],
        }
        self.save()

    def version(self, name: str) -> Path:
        return self.s.out / "surfaces" / name

    def spend(
        self,
        agent: float = 0.0,
        rollouts: int = 0,
        seconds: float = 0.0,
        costs: list[dict] | None = None,
    ) -> None:
        self.agent_usd += agent
        self.rollouts += rollouts
        self.sandbox_seconds += seconds
        self.costs += costs or []

    def charge(self, account: str | None, usd: float | None) -> None:
        """Add a rollout's USD to the account of the pool it ran on."""
        if account and usd is not None:
            self.account_usd[account] = self.account_usd.get(account, 0.0) + usd

    def spend_optimizer(self, out: dict) -> None:
        """Count one optimizer rollout (run_optimizer's outcome)."""
        if "cost_source" not in out:  # bf.run raised: no trial folder to price
            self.spend(rollouts=1)
            return
        usd, seconds = out["cost_usd"] or 0.0, out["sandbox_seconds"] or 0.0
        self.proposer_usd += usd
        self.last_proposer_usd = usd or self.last_proposer_usd
        self.last_proposer_seconds = seconds or self.last_proposer_seconds
        self.spend(
            rollouts=1,
            seconds=seconds,
            costs=[
                {
                    "usd": out["cost_usd"],
                    "source": out["cost_source"],
                    "models": out["cost_models"],
                }
            ],
        )

    def job_budget(self, share: float) -> bf.Budget | None:
        """A job's ``bf.Budget``: ``share`` of what is left under --max-cost-usd
        and --max-sandbox-seconds, so the jobs of one step together stay under
        both. (The rollout cap is checked before each step.)"""
        s, caps = self.s, {}
        if s.max_cost_usd is not None:
            left = s.max_cost_usd - self.agent_usd - self.proposer_usd
            caps["max_cost_usd"] = max(left * share, 0.01)
        if s.max_sandbox_seconds is not None:
            left = s.max_sandbox_seconds - self.sandbox_seconds
            caps["max_sandbox_seconds"] = max(left * share, 1.0)
        return bf.Budget(**caps) if caps else None

    def proposer_timeout(self) -> int:
        """The optimizer's time limit, within what is left of --max-sandbox-seconds."""
        timeout, cap = self.s.proposer.timeout_sec, self.s.max_sandbox_seconds
        if cap is None:
            return timeout
        return max(60, min(timeout, int(cap - self.sandbox_seconds)))

    def round_estimate(self) -> dict:
        """A round costs about what the baseline and the last optimizer run did."""
        base = self.baseline
        return {
            "rollouts": sum(base.attempted(split) for split in SPLITS) + 1,
            "usd": sum(base.doc[split]["cost_usd"] or 0.0 for split in SPLITS)
            + self.last_proposer_usd,
            "seconds": sum(base.doc[split]["sandbox_seconds"] for split in SPLITS)
            + self.last_proposer_seconds,
        }

    def over_cap(
        self, rollouts: int = 0, usd: float = 0.0, seconds: float = 0.0
    ) -> str | None:
        """Which cap the next step (``rollouts``, ``usd``, ``seconds``) would pass."""
        s = self.s
        for flag, cap, need, unit in (
            ("--max-rollouts", s.max_rollouts, self.rollouts + rollouts, "rollouts"),
            (
                "--max-cost-usd",
                s.max_cost_usd,
                self.agent_usd + self.proposer_usd + usd,
                "USD",
            ),
            (
                "--max-sandbox-seconds",
                s.max_sandbox_seconds,
                self.sandbox_seconds + seconds,
                "sandbox-seconds",
            ),
        ):
            if cap is not None and need > cap:
                return f"{flag} {cap:g} would be exceeded ({need:,.2f} {unit} with the next step)"
        return None

    def warn(self, *messages: str) -> None:
        self.doc["warnings"] += [m for m in messages if m not in self.doc["warnings"]]

    def scores_for_optimizer(self, current: Scores) -> dict:
        """Train in detail; the test split as aggregate scores only."""

        def agg(split_doc: dict) -> dict:
            return {
                "score": split_doc["score"]["value"],
                "ci95": split_doc["score"]["ci"],
                "tasks": split_doc["tasks"],
            }

        return {
            "note": "Test scores are aggregates over held-out tasks you cannot see.",
            "baseline": {split: agg(self.doc["baseline"][split]) for split in SPLITS},
            "current": {
                "version": current.version,
                "test": agg(current.doc["test"]),
                "train": {
                    **agg(current.doc["train"]),
                    "per_task": current.doc["train"]["per_task"],
                },
            },
            "rounds": [
                {
                    "round": e["round"],
                    "decision": e["decision"],
                    "reasons": e["reasons"],
                    "train_delta": (e.get("train_delta") or {}).get("value"),
                    "test_delta": (e.get("test_delta") or {}).get("value"),
                }
                for e in self.doc["rounds"]
            ],
        }

    def history(self) -> list[dict]:
        return [
            {
                "id": e["candidate"]["id"],
                "decision": e["decision"],
                "reasons": e["reasons"],
                "change": e["candidate"].get("change"),
                "diff": e["candidate"].get("diff", ""),
            }
            for e in self.doc["rounds"]
        ]

    def save(self) -> None:
        self.doc["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        costs = summary(self.costs)
        self.doc["cost"] = {
            "agent_usd": round(self.agent_usd, 6),
            "proposer_usd": round(self.proposer_usd, 6),
            "total_usd": round(self.agent_usd + self.proposer_usd, 6),
            "max_cost_usd": self.s.max_cost_usd,
            # Where the USD came from (hillclimb_cost): trials per source.
            "source": costs["source"],
            "sources": costs["sources"],
            "context_1m": costs["context_1m"],
            "rollouts": self.rollouts,
            "max_rollouts": self.s.max_rollouts,
            "sandbox_seconds": round(self.sandbox_seconds, 1),
            "max_sandbox_seconds": self.s.max_sandbox_seconds,
        }
        # Per account of the pool: jobs, rollouts, USD, usage-limit hits, last probe.
        self.doc["accounts"] = (
            {
                name: {**use, "usd": round(self.account_usd.get(name, 0.0), 6)}
                for name, use in self.s.pool.usage().items()
            }
            if self.s.pool
            else None
        )
        (self.s.out / "hillclimb.json").write_text(
            json.dumps(self.doc, indent=2) + "\n"
        )
        write_report(self.doc, self.s.out / "report.html")

    def finish(self, status: str, reason: str, detail: str) -> dict:
        """The verdict: the best version (the current one) against the baseline on test."""
        baseline, best = self.baseline, self.current
        if baseline is None:  # stopped before the baseline ran: nothing to compare
            self.doc.update(status=status, stop={"reason": reason, "detail": detail})
            self.save()
            return self.doc
        d = paired_delta(
            baseline.values["test"],
            best.values["test"],
            samples=self.s.bootstrap_samples,
            seed=self.s.seed,
        )
        gated = bool(self.doc["noise_gate"] and self.doc["noise_gate"]["passed"])
        exceeds = best is not baseline and bool(d["ci"]) and d["ci"][0] > 0
        if best is baseline:
            text = "No patch was kept; the baseline stands."
        else:
            interval = f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d["ci"] else ""
            text = (
                f"Best version {best.version}: test {best.doc['test']['score']['value']:.3f} against the "
                f"baseline's {baseline.doc['test']['score']['value']:.3f}, a change of {d['value']:+.3f}{interval}. "
                + (
                    "The gain exceeds noise: its 95% interval is above zero. Recommend merging."
                    if exceeds and gated
                    else "Its interval is above zero, but the run was not gated on noise (--force)."
                    if exceeds
                    else "The gain is within noise: its 95% interval includes zero. Recommend against merging."
                )
            )
        self.doc["best"] = {
            "version": best.version,
            "train": best.doc["train"]["score"],
            "test": best.doc["test"]["score"],
            "test_delta": d,
            "verdict": {
                "exceeds_noise": exceeds,
                "recommend_merge": exceeds and gated,
                "text": text,
            },
        }
        self.doc.update(status=status, stop={"reason": reason, "detail": detail})
        self.save()
        return self.doc


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument(
        "--skills", type=Path, required=True, help="the skills folder to improve"
    )
    p.add_argument("--out", type=Path, required=True, help="a new run folder")
    p.add_argument(
        "--include", action="append", default=[], help="only these tasks; repeatable"
    )
    p.add_argument("--agent", default="claude-agent-acp")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
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
    p.add_argument(
        "--max-rollouts",
        type=int,
        help="cap on rollouts that call a model: evaluation trials and optimizer runs",
    )
    p.add_argument(
        "--max-sandbox-seconds",
        type=float,
        help="cap on sandbox wall-clock seconds, the grader checks included",
    )
    p.add_argument(
        "--no-session-logs",
        action="store_true",
        help="do not copy Claude Code's session log into each trial (no cost from it)",
    )
    p.add_argument(
        "--oauth-pool",
        type=Path,
        help="a file of CC_OAUTH_<NAME>=<token> lines: run each job on the Claude "
        "subscription with the most headroom (hillclimb_pool)",
    )
    p.add_argument("--max-infra-error-rate", type=float, default=0.25)
    p.add_argument(
        "--force", action="store_true", help="climb although the noise gate refuses"
    )
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--proposer-agent", default="claude-agent-acp")
    p.add_argument("--proposer-model")
    p.add_argument(
        "--proposer-model-env",
        action="store_true",
        help="give Claude Code the optimizer's model as ANTHROPIC_MODEL instead of "
        "through ACP (whose model picker can map claude-opus-5-5 to its 1M-context row)",
    )
    p.add_argument("--proposer-timeout", type=int, default=1800)
    a = p.parse_args()
    via_env = a.proposer_model_env and a.proposer_model
    if a.oauth_pool and any(
        os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    ):
        raise SystemExit(
            "--oauth-pool: unset ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN, which "
            "Claude Code would use instead of the pool's subscriptions"
        )
    s = Settings(
        tasks_dir=a.tasks_dir,
        skills=a.skills,
        out=a.out,
        include=a.include,
        agent=a.agent,
        model=a.model,
        sandbox=a.sandbox,
        concurrency=a.concurrency,
        trials=a.trials,
        rounds=a.rounds,
        min_gain=a.min_gain,
        test_frac=a.test_frac,
        seed=a.seed,
        split_file=a.split_file,
        stall_rounds=a.stall_rounds,
        max_cost_usd=a.max_cost_usd,
        max_rollouts=a.max_rollouts,
        max_sandbox_seconds=a.max_sandbox_seconds,
        session_logs=not a.no_session_logs,
        pool=Pool.from_file(a.oauth_pool) if a.oauth_pool else None,
        max_infra_error_rate=a.max_infra_error_rate,
        force=a.force,
        skip_controls=a.skip_controls,
        proposer=ProposerSettings(
            agent=a.proposer_agent,
            model=None if via_env else a.proposer_model,
            agent_env={"ANTHROPIC_MODEL": a.proposer_model} if via_env else {},
            sandbox=a.sandbox,
            timeout_sec=a.proposer_timeout,
        ),
    )
    doc = asyncio.run(climb(s))
    print(f"{doc['status']}: {doc['stop']['detail']}")
    print(doc["best"]["verdict"]["text"] if doc["best"] else "")
    print(f"record: {s.out / 'hillclimb.json'}\nreport: {s.out / 'report.html'}")
    raise SystemExit({"refused": 2, "stopped": 1}.get(doc["status"], 0))


if __name__ == "__main__":
    main()

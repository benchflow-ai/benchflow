"""The optimizer of the hill-climb demo: one sandboxed BenchFlow rollout per round.

Each round writes a small task folder and runs it with ``bf.run(RolloutConfig)``,
like any other task. The optimizer agent gets exactly two uploads
(``RolloutConfig.uploads``) and nothing else:

- ``/hillclimb`` (read-only, root-owned before the agent starts): the train
  split's failed trials (verdict, trajectory, grader output), the train tasks'
  instructions, the scores, and earlier rounds' proposals;
- ``/app/surface``: the current skills folder, which it edits in place.

The test split is never in them: this module is only ever given train trials
and train task folders, and test results reach it only as aggregate scores in
``scores.json``. The task sets ``allow_internet: false``, so the optimizer
cannot fetch the tasks from anywhere else either. Before each upload the
demo checks every uploaded path and file for each test task's name and
instruction text, and records what was mounted with a sha256 manifest.

The task's ``tests/test.sh`` copies ``/app/surface`` and the optimizer's
``/app/proposal.json`` to ``/logs/verifier``, where BenchFlow collects them
into the trial folder, and checks their shape, so the rollout's reward only
means "a well-formed proposal exists". In analysis mode (after a stall) the
optimizer sorts the remaining train failures by root cause instead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

import benchflow as bf

EVIDENCE = "/hillclimb"
SURFACE = "/app/surface"
CATEGORIES = ("ambiguous_task", "grader_bug", "infrastructure", "capability_gap")


@dataclass
class ProposerSettings:
    agent: str = "claude-agent-acp"
    model: str | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    sandbox: str = "docker"
    timeout_sec: int = 1800
    max_failures: int = 24
    # With it the sandbox could fetch public copies of the test tasks; the demo's
    # Docker test needs it only to reach its scripted model on the host.
    open_network: bool = False
    # Appended to the brief: a scripted model's marker in tests.
    extra_instructions: str = ""


PROPOSE = """You are improving the skills an AI agent receives when it solves benchmark tasks. The goal is a higher score: more tasks solved. Make ONE targeted change this round, aimed at the root cause of a group of failures.

Edit in place (the only output that counts): `/app/surface/skills/`, a skills folder. Each subfolder is one skill with a `SKILL.md` (YAML frontmatter with `name` and `description`, then instructions) and optional scripts. The agent gets these skills on every task.

Read (read-only):
- `/hillclimb/train/failures/<task>/trial-NN/`: failed runs on training tasks: `verdict.json`, `trajectory/acp_trajectory.jsonl` (the agent's actions), `verifier/` (the grader's output).
- `/hillclimb/train/tasks/<task>/instruction.md`: the training tasks' instructions.
- `/hillclimb/scores.json`: train and test scores with 95% intervals, and past rounds' decisions.
- `/hillclimb/history/`: earlier proposals, their diffs, and whether they were kept. Do not repeat a reverted idea.

1. Group the failures by root cause (a missing procedure, a wrong assumption, a misused tool, a skipped check).
2. Pick the cause whose fix could lift the most tasks, including tasks you have not seen.
3. Make one small, general change for it: a procedure, a check, a piece of domain knowledge, a pitfall.
4. Do not paste task instructions, expected outputs, test names or answers into the skills. Held-out tasks you cannot see decide whether the change is kept; content specific to one training task will not transfer, will be detected, and the change will be reverted.
5. Every skill folder needs a `SKILL.md` that starts with YAML frontmatter.

Then write `/app/proposal.json`: {"root_cause": "...", "change": "one sentence", "rationale": "why it fixes the cause and should transfer", "evidence": ["<task>/trial-NN"]}
"""

ANALYZE = """The hill-climb on an AI agent's skills has stalled: recent changes did not improve the score. Do not edit anything. Read every remaining failure on the training tasks and sort it by root cause.

Read: `/hillclimb/train/failures/<task>/trial-NN/` (failed runs), `/hillclimb/train/infra/<task>/trial-NN/verdict.json` (runs that ended without a score), `/hillclimb/train/tasks/`, `/hillclimb/scores.json`, `/hillclimb/history/`, and the current skills in `/app/surface/`.

Categories, one per failure:
- `ambiguous_task`: the instruction allows the agent's answer, or does not say what the grader requires;
- `grader_bug`: the grader rejects a correct solution, or checks something the task does not ask for;
- `infrastructure`: the sandbox, tools, network, provider or time limit failed, not the agent;
- `capability_gap`: the agent had what it needed and still got it wrong.

Write `/app/analysis.json`: {"summary": "three to five sentences on what limits the score now", "failures": [{"id": "<task>/trial-NN", "category": "capability_gap", "explanation": "..."}], "recommendations": ["..."]}
"""

TASK_MD = """---
schema_version: '1.3'
verifier:
  type: test-script
  timeout_sec: 120.0
agent:
  timeout_sec: {timeout}
sandbox:
  docker_image: {image}
  workdir: /app{network}
  cpus: 1
  memory_mb: 2048
  storage_mb: 4096
---

"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
cp /app/{output} /logs/verifier/{output} 2>/dev/null
rm -rf /logs/verifier/surface && cp -R /app/surface /logs/verifier/surface 2>/dev/null
if python3 "$(dirname "$0")/validate.py" /app/{output}; then r=1; else r=0; fi
echo $r > /logs/verifier/reward.txt
"""

VALIDATE_PY = """import json, sys
data = json.load(open(sys.argv[1]))
keys = ("root_cause", "change", "rationale") if "proposal" in sys.argv[1] else ("summary",)
sys.exit(0 if all(isinstance(data.get(k), str) and data[k].strip() for k in keys) else 1)
"""


def _copy_head_tail(src: Path, dest: Path, limit: int = 400_000) -> None:
    """Copy a text file, keeping its head and tail when it is long."""
    if not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    if len(data) > limit:
        data = data[: limit // 4] + b"\n[... cut ...]\n" + data[-(limit * 3 // 4) :]
    dest.write_bytes(data)


def write_workspace(
    root: Path,
    *,
    skills: Path,
    train_tasks: dict[str, Path],
    failures: list[dict],
    infra: list[dict],
    scores: dict,
    history: list[dict],
) -> dict[str, str]:
    """Write what the optimizer may see; return the upload map for RolloutConfig.

    Only the train split goes in: ``train_tasks``, ``failures`` and ``infra``
    are train records, and ``scores`` holds the test split as aggregates.
    """
    shutil.rmtree(root, ignore_errors=True)
    evidence, surface = root / "evidence", root / "surface"
    shutil.copytree(skills, surface / "skills", symlinks=False)
    (evidence / "history").mkdir(parents=True)
    (evidence / "scores.json").write_text(json.dumps(scores, indent=2))
    for name, task_dir in train_tasks.items():
        dest = evidence / "train" / "tasks" / name / "instruction.md"
        dest.parent.mkdir(parents=True)
        dest.write_text(bf.Task(task_dir).instruction.strip() + "\n")
    for kind, rows in (("failures", failures), ("infra", infra)):
        for row in rows:
            dest = evidence / "train" / kind / row["task"] / f"trial-{row['trial']:02d}"
            dest.mkdir(parents=True)
            verdict = {k: row[k] for k in ("task", "trial", "reward", "passed", "error")}
            (dest / "verdict.json").write_text(json.dumps(verdict, indent=2))
            if not row["path"]:  # a trial that never ran has no folder
                continue
            trial = Path(row["path"])
            _copy_head_tail(
                trial / "trajectory" / "acp_trajectory.jsonl",
                dest / "trajectory" / "acp_trajectory.jsonl",
            )
            for name in ("test-stdout.txt", "test-stderr.txt", "reward.txt", "ctrf.json"):
                _copy_head_tail(trial / "verifier" / name, dest / "verifier" / name, 200_000)
    for item in history:
        (evidence / "history" / f"{item['id']}.json").write_text(json.dumps(item, indent=2))
    return {str(evidence): EVIDENCE, str(surface): SURFACE}


def mounted(
    uploads: dict[str, str],
    *,
    test_instructions: dict[str, str],
    open_network: bool,
    manifest: Path,
) -> dict:
    """Record every uploaded file (sha256) and check the uploads for the test split.

    Returns what ``hillclimb.json`` keeps: the mounts, the train tasks and
    failures they hold, and the test tasks found in them by name (in any
    path) or by instruction text (in any file). Both lists are empty unless
    something is wrong.
    """

    def squash(text: str) -> str:
        return " ".join(text.split())

    needles = {t: squash(s) for t, s in test_instructions.items() if len(squash(s)) >= 40}
    files, mounts, in_paths, in_text = [], [], set(), set()
    for host, target in uploads.items():
        count = size = 0
        for path in sorted(Path(host).rglob("*")):
            rel = path.relative_to(host)
            in_paths.update(p for p in rel.parts if p in test_instructions)
            if not path.is_file():
                continue
            data = path.read_bytes()
            count, size = count + 1, size + len(data)
            files.append(
                {"mount": target, "path": rel.as_posix(), "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
            )
            text = squash(data.decode("utf-8", "replace"))
            in_text.update(t for t, needle in needles.items() if needle in text)
        mounts.append({"sandbox_path": target, "read_only": target == EVIDENCE,
                       "files": count, "bytes": size})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"files": files}, indent=2))
    train = Path(next(h for h, t in uploads.items() if t == EVIDENCE)) / "train"
    return {
        "mounts": mounts,
        "manifest": str(manifest),
        "network": "open" if open_network else "none",
        "train_tasks": sorted(p.name for p in (train / "tasks").glob("*")),
        "failures": sorted(f"{p.parent.name}/{p.name}" for p in (train / "failures").glob("*/*")),
        "test_tasks": len(test_instructions),
        "test_tasks_in_paths": sorted(in_paths),
        "test_instructions_in_files": sorted(in_text),
    }


def _read_only(root: Path, on: bool) -> None:
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222 if on else mode | stat.S_IWUSR)


async def _lock_evidence(sandbox) -> None:
    """Pre-agent hook: the evidence becomes root-owned and read-only."""
    result = await sandbox.exec(
        f"chown -R 0:0 {EVIDENCE} && chmod -R a-w,a+rX {EVIDENCE}", user="root", timeout_sec=120
    )
    if result.return_code != 0:
        raise RuntimeError(f"could not lock {EVIDENCE}: {result.stderr}")


async def run_optimizer(
    mode: str,
    uploads: dict[str, str],
    *,
    settings: ProposerSettings,
    task_dir: Path,
    jobs_dir: Path,
) -> dict:
    """Run one optimizer rollout; return its outcome and what it left behind."""
    output = "proposal.json" if mode == "propose" else "analysis.json"
    shutil.rmtree(task_dir, ignore_errors=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.md").write_text(
        TASK_MD.format(
            timeout=float(settings.timeout_sec),
            image=bf.ReviewerConfig().image,  # the rubric reviewer's pinned Python image
            network="" if settings.open_network else "\n  allow_internet: false",
        )
        + (PROPOSE if mode == "propose" else ANALYZE)
        + settings.extra_instructions
    )
    (task_dir / "tests" / "test.sh").write_text(TEST_SH.format(output=output))
    (task_dir / "tests" / "validate.py").write_text(VALIDATE_PY)
    evidence = Path(next(h for h, t in uploads.items() if t == EVIDENCE))
    _read_only(evidence, True)  # a backend that keeps modes uploads it read-only too
    try:
        result = await bf.run(
            bf.RolloutConfig(
                task_path=task_dir,
                agent=settings.agent,
                model=settings.model,
                agent_env=dict(settings.agent_env),
                environment=settings.sandbox,
                jobs_dir=jobs_dir,
                job_name="job",
                timeout=settings.timeout_sec,
                uploads=uploads,
                pre_agent_hooks=[_lock_evidence],
            )
        )
    except Exception as exc:
        return {"status": "failed", "error": f"optimizer rollout failed: {exc}"}
    finally:
        _read_only(evidence, False)
    out = {
        "status": "failed",
        "error": result.error,
        "rollout_dir": str(result.rollout_dir) if result.rollout_dir else None,
        "cost_usd": result.cost_usd,
    }
    verifier = Path(result.rollout_dir or jobs_dir) / "verifier"
    try:
        data = json.loads((verifier / output).read_text())
    except (OSError, ValueError):
        out["error"] = f"the optimizer left no readable {output} ({result.error})"
        return out
    surface = verifier / "surface" / "skills"
    if mode == "propose" and not surface.is_dir():
        out["error"] = "the edited skills folder was not collected"
        return out
    return {**out, "status": "ok", "error": None, "output": data, "surface": str(surface)}

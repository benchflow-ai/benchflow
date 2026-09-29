"""The proposer: a sandboxed BenchFlow rollout that edits the surface.

Each round an agent (Claude Code by default) runs one ordinary rollout of a
generated task, built here on the host the way the rubric reviewer's wrapper
task is (:mod:`benchflow.review.wrapper`):

- a digest-pinned base image, ``allow_internet: false`` (the sandbox-local
  model proxy and the agent-UID egress firewall), no task Dockerfile;
- ``RolloutConfig.uploads`` puts two host folders into the sandbox after it
  starts: the **evidence** at ``/hillclimb`` (locked read-only and root-owned
  before the agent starts) and the **current surface** at ``/app/surface``,
  which the agent edits in place;
- the instruction is the brief below; the wrapper's ``tests/test.sh`` copies
  ``/app/surface`` and ``/app/proposal.json`` to ``/logs/verifier`` and
  checks their shape, so the rollout's reward means only "a well-formed
  proposal exists".

The evidence holds the train split's failed trials (verdict, trajectory,
verifier output), the train tasks' instructions, the scores, and earlier
rounds' proposals. The test split is **never** in it: its trials,
instructions and task names stay on the host, and only its aggregate scores
with confidence intervals reach ``scores.json``. The sandbox has no network,
so the proposer cannot fetch the tasks from anywhere else either. That is the
structural form of the post's rule that the optimizer never sees the test set.

In analysis mode (after a stall) the same rollout classifies the remaining
train failures by root cause and writes ``/app/analysis.json`` instead.
"""

from __future__ import annotations

import json
import logging
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.hillclimbing.evaluate import SplitRun, TrialRecord
from benchflow.hillclimbing.record import ANALYSIS_CATEGORIES
from benchflow.hillclimbing.surface import SurfaceSpec

logger = logging.getLogger(__name__)

Mode = Literal["propose", "analyze"]

EVIDENCE_MOUNT = "/hillclimb"
WORKDIR = "/app"
SURFACE_MOUNT = f"{WORKDIR}/surface"
PROPOSAL_FILE = "proposal.json"
ANALYSIS_FILE = "analysis.json"
DEFAULT_TIMEOUT_SEC = 1800
VERIFIER_TIMEOUT_SEC = 120
TRAJECTORY_LIMIT = 400_000
TEXT_LIMIT = 200_000
_VERIFIER_FILES = (
    "test-stdout.txt",
    "test-stderr.txt",
    "reward.txt",
    "reward.json",
    "reward-details.json",
    "ctrf.json",
)


def default_image() -> str:
    from benchflow.review.options import REVIEWER_IMAGE

    return REVIEWER_IMAGE


@dataclass
class ProposerSettings:
    """How the proposer runs. The agent needs its own credentials in ``agent_env``
    (or the host login) like any rollout."""

    agent: str = "claude-agent-acp"
    model: str | None = None
    reasoning_effort: str | None = None
    environment: str = "docker"
    agent_env: dict[str, str] = field(default_factory=dict)
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    image: str = field(default_factory=default_image)
    open_network: bool = False
    max_failures: int = 24
    # Appended to the brief. For scripted models in tests (a fake provider's
    # script marker); not a CLI option.
    extra_instructions: str | None = None


@dataclass
class ProposerOutcome:
    """What one proposer rollout left behind."""

    status: Literal["ok", "failed"]
    error: str | None = None
    rollout_dir: Path | None = None
    workspace_dir: Path | None = None
    reward: float | None = None
    cost_usd: float | None = None
    surface_dir: Path | None = None
    output: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------

_SURFACE_TEXT = {
    "skills": (
        f"- `{SURFACE_MOUNT}/skills/`: a skills folder. Each subfolder is one skill "
        "with a `SKILL.md` (YAML frontmatter with `name` and `description`, then "
        "instructions) and optional scripts or references. The agent under test "
        "gets these skills on every task."
    ),
    "prompt": (
        f"- `{SURFACE_MOUNT}/prompt.md`: text placed before every task's prompt "
        "for the agent under test."
    ),
}

_OBJECTIVE_TEXT = {
    "score": (
        "The goal is a higher score: more tasks solved by the agent under test."
    ),
    "cost": (
        "The goal is a lower cost per task at the same score: the agent under test "
        "should reach the same results with fewer tokens and turns (less "
        "exploration, fewer retries, shorter outputs)."
    ),
}

PROPOSE_BRIEF = """You are improving the files an AI agent receives when it solves benchmark tasks. {objective} Make ONE targeted change this round, aimed at the root cause of a group of failures.

What you can edit (edit the files in place; this is the only output that counts):
{surface}

What you can read (read-only):
- `{evidence}/train/failures/<task>/trial-NN/`: failed runs on training tasks. `verdict.json` has the reward and errors, `trajectory/acp_trajectory.jsonl` the agent's actions, `verifier/` the grader's output.
- `{evidence}/train/tasks/<task>/instruction.md`: the training tasks' instructions.
- `{evidence}/scores.json`: the current train and test scores with 95% confidence intervals, and past rounds' decisions.
- `{evidence}/history/`: earlier proposals, their diffs and whether they were kept. Do not repeat a reverted idea.

How to work:
1. Read the failures and group them by root cause (a missing procedure, a wrong assumption, a tool the agent misuses, a check it skips).
2. Pick the one cause whose fix could lift the most tasks, including tasks you have not seen.
3. Make one small change to the files above that fixes that cause in general: a procedure, a check, a piece of domain knowledge, a pitfall to avoid.
4. Do not paste task instructions, expected outputs, test names, file contents or answers from the failures into the files. A held-out set of tasks you cannot see decides whether your change is kept; content specific to one training task will not transfer, will be detected, and the change will be reverted.
5. Keep skills well-formed: every skill folder needs a `SKILL.md` that starts with YAML frontmatter.

When you are done, write `{proposal}` as a JSON object:
{{"root_cause": "the cause you targeted, in one or two sentences", "change": "what you changed, in one sentence", "rationale": "why this fixes the cause and should transfer to unseen tasks", "evidence": ["<task>/trial-NN", "..."], "expected_effect": "which failures should now pass"}}
"""

ANALYZE_BRIEF = """The hill-climb on the files an AI agent receives has stalled: the last rounds' changes did not improve the score. Your job now is analysis, not editing. Read every remaining failure on the training tasks and sort it by root cause.

What you can read (read-only):
- `{evidence}/train/failures/<task>/trial-NN/`: failed runs. `verdict.json` has the reward and errors, `trajectory/acp_trajectory.jsonl` the agent's actions, `verifier/` the grader's output.
- `{evidence}/train/infra/<task>/trial-NN/verdict.json`: runs that ended without a score (infrastructure errors).
- `{evidence}/train/tasks/<task>/instruction.md`: the training tasks' instructions.
- `{evidence}/scores.json` and `{evidence}/history/`: the scores and every change that was tried.
- `{surface_path}`: the current files the agent receives.

Categories, one per failure:
- `ambiguous_task`: the instruction allows the agent's answer, or does not say what the grader requires.
- `grader_bug`: the grader rejects a correct solution, or checks something the task does not ask for.
- `infrastructure`: the sandbox, tools, network, provider or time limit failed, not the agent.
- `capability_gap`: the agent had what it needed and still got it wrong.

Write `{analysis}` as a JSON object:
{{"summary": "three to five sentences on what limits the score now", "failures": [{{"id": "<task>/trial-NN", "category": "capability_gap", "explanation": "one or two sentences citing the evidence"}}], "recommendations": ["what a person should do next: fix a task, fix a grader, add trials, ..."]}}
Cover every failure and infrastructure error listed above.
"""


def brief(
    mode: Mode,
    specs: Sequence[SurfaceSpec],
    objective: str,
    extra: str | None = None,
) -> str:
    if mode == "propose":
        text = PROPOSE_BRIEF.format(
            objective=_OBJECTIVE_TEXT[objective],
            surface="\n".join(_SURFACE_TEXT[s.kind] for s in specs),
            evidence=EVIDENCE_MOUNT,
            proposal=f"{WORKDIR}/{PROPOSAL_FILE}",
        )
    else:
        text = ANALYZE_BRIEF.format(
            evidence=EVIDENCE_MOUNT,
            surface_path=f"{SURFACE_MOUNT}/",
            analysis=f"{WORKDIR}/{ANALYSIS_FILE}",
        )
    return text + (f"\n{extra.strip()}\n" if extra else "")


# ---------------------------------------------------------------------------
# The evidence workspace
# ---------------------------------------------------------------------------


def _copy_limited(src: Path, dest: Path, limit: int) -> None:
    """Copy a text file, keeping the head and tail when it is over ``limit`` bytes."""
    if not src.is_file() or src.is_symlink():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    if size <= limit:
        shutil.copyfile(src, dest)
        return
    head_n, tail_n = limit // 4, limit - limit // 4
    with src.open("rb") as handle:
        head = handle.read(head_n)
        handle.seek(max(size - tail_n, 0))
        tail = handle.read()
    head = head[: head.rfind(b"\n") + 1] or head
    nl = tail.find(b"\n")
    tail = tail[nl + 1 :] if nl != -1 else tail
    marker = f"\n[... {size - len(head) - len(tail)} bytes omitted ...]\n".encode()
    dest.write_bytes(head + marker + tail)


def select_failures(records: Sequence[TrialRecord], limit: int) -> list[TrialRecord]:
    """Up to ``limit`` records, one per task first, then more per task."""
    by_task: dict[str, list[TrialRecord]] = {}
    for r in sorted(records, key=lambda r: (r.task, r.trial)):
        by_task.setdefault(r.task, []).append(r)
    chosen: list[TrialRecord] = []
    depth = 0
    while len(chosen) < limit and any(len(v) > depth for v in by_task.values()):
        for task in sorted(by_task):
            if depth < len(by_task[task]) and len(chosen) < limit:
                chosen.append(by_task[task][depth])
        depth += 1
    return chosen


def _verdict(record: TrialRecord) -> dict[str, Any]:
    return {
        "task": record.task,
        "trial": record.trial,
        "reward": record.reward,
        "passed": record.passed,
        "error_category": record.category,
        "error": record.error,
        "n_tool_calls": record.n_tool_calls,
        "cost_usd": record.cost_usd,
    }


def _write_trial(record: TrialRecord, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verdict.json").write_text(json.dumps(_verdict(record), indent=2) + "\n")
    if not record.path:
        return
    trial_dir = Path(record.path)
    _copy_limited(
        trial_dir / "trajectory" / "acp_trajectory.jsonl",
        dest / "trajectory" / "acp_trajectory.jsonl",
        TRAJECTORY_LIMIT,
    )
    for name in _VERIFIER_FILES:
        _copy_limited(trial_dir / "verifier" / name, dest / "verifier" / name, TEXT_LIMIT)


def _strip_write_bits(root: Path) -> None:
    bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~bits)


def remove_tree(root: Path) -> None:
    """Remove a workspace whose files were made read-only."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    shutil.rmtree(root)


def instruction_text(task_dir: Path) -> str:
    from benchflow.rollout._setup import _read_task_instruction

    try:
        return _read_task_instruction(task_dir)
    except Exception as exc:  # a task we could not read is still listed
        return f"(could not read this task's instruction: {exc})"


@dataclass
class Workspace:
    root: Path
    evidence: Path
    surface: Path
    uploads: dict[str, str]
    failures: list[TrialRecord]
    infra: list[TrialRecord]


def build_workspace(
    root: Path,
    *,
    mode: Mode,
    version_dir: Path,
    specs: Sequence[SurfaceSpec],
    train: SplitRun,
    train_dirs: Mapping[str, Path],
    scores: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    objective: str,
    max_failures: int,
    extra_instructions: str | None = None,
) -> Workspace:
    """Assemble what the proposer may see; nothing else is uploaded.

    ``train`` must be the train split's run: this function never receives
    the test split's records, task folders or per-task scores, so it cannot
    copy them. ``scores`` carries the test split as aggregates only.
    """
    if train.split != "train":
        raise ValueError("the proposer only reads the train split")
    if root.exists():
        remove_tree(root)
    evidence = root / "evidence"
    surface = root / "surface"
    evidence.mkdir(parents=True)
    for spec in specs:
        src = version_dir / spec.name
        if spec.kind == "skills":
            shutil.copytree(src, surface / spec.name)
        else:
            surface.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, surface / spec.name)
    surface.mkdir(parents=True, exist_ok=True)

    (evidence / "BRIEF.md").write_text(
        brief(mode, specs, objective, extra_instructions)
    )
    (evidence / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    tasks_dir = evidence / "train" / "tasks"
    for name in train.tasks:
        target = tasks_dir / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "instruction.md").write_text(
            instruction_text(train_dirs[name]) + "\n"
        )
    failures = select_failures([r for r in train.records if r.failed], max_failures)
    for record in failures:
        _write_trial(
            record,
            evidence / "train" / "failures" / record.task / f"trial-{record.trial:02d}",
        )
    infra: list[TrialRecord] = []
    if mode == "analyze":
        infra = select_failures(train.infra_records, max_failures)
        for record in infra:
            dest = evidence / "train" / "infra" / record.task / f"trial-{record.trial:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "verdict.json").write_text(
                json.dumps(_verdict(record), indent=2) + "\n"
            )
    hist = evidence / "history"
    hist.mkdir()
    for item in history:
        cid = str(item.get("id"))
        (hist / f"{cid}.json").write_text(
            json.dumps({k: v for k, v in item.items() if k != "diff"}, indent=2) + "\n"
        )
        if item.get("diff"):
            (hist / f"{cid}.diff").write_text(str(item["diff"]))
    _strip_write_bits(evidence)
    return Workspace(
        root=root,
        evidence=evidence,
        surface=surface,
        uploads={str(evidence): EVIDENCE_MOUNT, str(surface): SURFACE_MOUNT},
        failures=failures,
        infra=infra,
    )


def leaked_test_names(workspace: Workspace, test_names: Sequence[str]) -> list[str]:
    """Test task names that appear as a path in the uploaded tree (should be none).

    A defense-in-depth check the climb runs before every upload; the
    workspace is built from train data only, so a hit is a bug.
    """
    names = set(test_names)
    hits = set()
    for upload in workspace.uploads:
        for path in Path(upload).rglob("*"):
            hits.update(part for part in path.relative_to(upload).parts if part in names)
    return sorted(hits)


def _squash(text: str) -> str:
    return " ".join(text.split())


def exposure(
    workspace: Workspace,
    *,
    test_instructions: Mapping[str, str],
    open_network: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    """Exactly what one proposer sandbox received, checked against the test split.

    Writes ``manifest_path`` (every uploaded file: mount, path, size, sha256)
    and returns the summary recorded in ``hillclimb.json``: the mounts, the
    train tasks and failures they hold, and which test tasks were found in
    them, by name in any path or by their full instruction text in any file
    (both lists are empty unless something is wrong).
    """
    import hashlib

    files: list[dict[str, Any]] = []
    mounts = []
    names = set(test_instructions)
    in_paths: set[str] = set()
    in_text: set[str] = set()
    needles = {
        task: _squash(text)
        for task, text in test_instructions.items()
        if len(_squash(text)) >= 40
    }
    for host, target in workspace.uploads.items():
        root = Path(host)
        count = size = 0
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            in_paths.update(part for part in rel.parts if part in names)
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
            count += 1
            size += len(data)
            files.append(
                {
                    "mount": target,
                    "path": rel.as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            text = _squash(data.decode("utf-8", errors="replace"))
            in_text.update(task for task, needle in needles.items() if needle in text)
        mounts.append(
            {
                "sandbox_path": target,
                "read_only": target == EVIDENCE_MOUNT,
                "files": count,
                "bytes": size,
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"files": files}, indent=2) + "\n")
    evidence = workspace.evidence / "train"
    return {
        "mounts": mounts,
        "network": "open" if open_network else "none",
        "train_tasks": sorted(
            p.name for p in (evidence / "tasks").iterdir() if p.is_dir()
        )
        if (evidence / "tasks").is_dir()
        else [],
        "failures": [f"{r.task}/trial-{r.trial:02d}" for r in workspace.failures],
        "infra_errors": [f"{r.task}/trial-{r.trial:02d}" for r in workspace.infra],
        "test_tasks": len(names),
        "test_tasks_in_paths": sorted(in_paths),
        "test_instructions_in_files": sorted(in_text),
    }


# ---------------------------------------------------------------------------
# The wrapper task
# ---------------------------------------------------------------------------

_FRONTMATTER = """---
schema_version: '1.3'
metadata:
  category: hillclimb-{mode}
verifier:
  type: test-script
  timeout_sec: {verifier_timeout}
agent:
  timeout_sec: {agent_timeout}
sandbox:
  docker_image: {image}
  workdir: {workdir}{network_line}
  cpus: 1
  memory_mb: 2048
  storage_mb: 4096
---

"""

_TEST_SCRIPT = """#!/bin/bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /logs/verifier
cp {workdir}/{output} /logs/verifier/{output} 2>/dev/null || true
rm -rf /logs/verifier/surface
if [ -d {surface} ]; then cp -R {surface} /logs/verifier/surface; fi
if python3 "$DIR/validate.py" {mode} {workdir}; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

_VALIDATOR = '''"""Shape check of the proposer's output (stdlib only).

Reward 1 means a well-formed output exists, nothing about its quality.
"""

import json
import sys
from pathlib import Path

CATEGORIES = {categories!r}


def text(value):
    return isinstance(value, str) and value.strip() != ""


def main():
    mode, workdir = sys.argv[1], Path(sys.argv[2])
    name = "proposal.json" if mode == "propose" else "analysis.json"
    path = workdir / name
    problems = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{{name}} missing or not JSON: {{exc}}")
        return 1
    if not isinstance(data, dict):
        print(f"{{name}} must be a JSON object")
        return 1
    if mode == "propose":
        for key in ("root_cause", "change", "rationale"):
            if not text(data.get(key)):
                problems.append(f"{{key}} must be a non-empty string")
        if not (workdir / "surface").is_dir():
            problems.append("surface/ is missing")
    else:
        if not text(data.get("summary")):
            problems.append("summary must be a non-empty string")
        failures = data.get("failures")
        if not isinstance(failures, list):
            problems.append("failures must be a list")
            failures = []
        for i, item in enumerate(failures):
            if not isinstance(item, dict):
                problems.append(f"failures[{{i}}] must be an object")
                continue
            if item.get("category") not in CATEGORIES:
                problems.append(f"failures[{{i}}].category must be one of {{sorted(CATEGORIES)}}")
            if not text(item.get("id")) or not text(item.get("explanation")):
                problems.append(f"failures[{{i}}] needs id and explanation")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def assemble_task(dest: Path, *, mode: Mode, settings: ProposerSettings, instruction: str) -> Path:
    """Write the wrapper task (``task.md``, ``tests/``) under ``dest``."""
    if dest.exists():
        shutil.rmtree(dest)
    tests = dest / "tests"
    tests.mkdir(parents=True)
    frontmatter = _FRONTMATTER.format(
        mode=mode,
        verifier_timeout=float(VERIFIER_TIMEOUT_SEC),
        agent_timeout=float(settings.timeout_sec),
        image=settings.image,
        workdir=WORKDIR,
        network_line="" if settings.open_network else "\n  allow_internet: false",
    )
    (dest / "task.md").write_text(frontmatter + instruction)
    output = PROPOSAL_FILE if mode == "propose" else ANALYSIS_FILE
    test_sh = tests / "test.sh"
    test_sh.write_text(
        _TEST_SCRIPT.format(
            workdir=WORKDIR, output=output, surface=SURFACE_MOUNT, mode=mode
        )
    )
    test_sh.chmod(0o755)
    (tests / "validate.py").write_text(
        _VALIDATOR.format(categories=tuple(ANALYSIS_CATEGORIES))
    )
    return dest


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


async def run_rollout(config: Any) -> Any:
    """``bf.run``; a seam the tests replace with a scripted sandbox."""
    from benchflow import run

    return await run(config)


async def _lock_evidence(sandbox: Any) -> None:
    """Make the uploaded evidence root-owned and read-only before the agent starts."""
    result = await sandbox.exec(
        f"chown -R 0:0 {EVIDENCE_MOUNT} && chmod -R a-w,a+rX {EVIDENCE_MOUNT}",
        user="root",
        timeout_sec=120,
    )
    if getattr(result, "return_code", 0) != 0:
        raise RuntimeError(
            "failed to lock the hillclimb evidence read-only: "
            f"{(getattr(result, 'stderr', '') or '')[:300]}"
        )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _find_leaf(jobs_dir: Path) -> Path | None:
    leaves = sorted(p.parent for p in jobs_dir.rglob("result.json"))
    return leaves[-1] if leaves else None


async def run_proposer(
    workspace: Workspace,
    *,
    mode: Mode,
    settings: ProposerSettings,
    task_dir: Path,
    jobs_dir: Path,
    runner: Callable[[Any], Any] | None = None,
) -> ProposerOutcome:
    """Run one proposer rollout over ``workspace`` and collect what it wrote."""
    from benchflow.rollout import RolloutConfig

    instruction = (workspace.evidence / "BRIEF.md").read_text()
    assemble_task(task_dir, mode=mode, settings=settings, instruction=instruction)
    config = RolloutConfig(
        task_path=task_dir,
        agent=settings.agent,
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        agent_env=dict(settings.agent_env),
        environment=settings.environment,
        jobs_dir=jobs_dir,
        job_name="job",
        timeout=settings.timeout_sec,
        uploads=dict(workspace.uploads),
        pre_agent_hooks=[_lock_evidence],
    )
    outcome = ProposerOutcome(status="failed", workspace_dir=workspace.root)
    try:
        result = await (runner or run_rollout)(config)
    except Exception as exc:
        logger.error("Proposer rollout failed", exc_info=True)
        outcome.error = f"proposer rollout raised: {exc}"
        outcome.rollout_dir = _find_leaf(jobs_dir)
        return outcome
    leaf = getattr(result, "rollout_dir", None) or _find_leaf(jobs_dir)
    outcome.rollout_dir = Path(leaf) if leaf else None
    outcome.reward = getattr(result, "reward", None)
    outcome.cost_usd = getattr(result, "cost_usd", None)
    if outcome.rollout_dir is None:
        outcome.error = f"the proposer rollout left no trial folder ({getattr(result, 'error', None)})"
        return outcome
    verifier = outcome.rollout_dir / "verifier"
    name = PROPOSAL_FILE if mode == "propose" else ANALYSIS_FILE
    output = _read_json(verifier / name)
    if not isinstance(output, dict):
        agent_error = getattr(result, "error", None)
        outcome.error = f"the proposer left no readable {name}" + (
            f" (agent error: {agent_error})" if agent_error else ""
        )
        return outcome
    outcome.output = output
    if mode == "propose":
        surface = verifier / "surface"
        if not surface.is_dir() or surface.is_symlink():
            outcome.error = "the proposer's surface folder was not collected"
            return outcome
        outcome.surface_dir = surface
    outcome.status = "ok"
    return outcome

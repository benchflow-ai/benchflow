"""Branch an agent run from Python in one call: ``bf.branch`` / ``await bf.abranch``.

This is the Python face of ``bench eval branch`` and runs the same driver
(:func:`benchflow.branch_run.run_branch_trial`): run the task up to a
checkpoint, snapshot it, run every child from the snapshot with its own
prompt, score each child with the task's verifier, then (by default) restore
the parent, finish it and verify it. The job folder is the same as the CLI's
(``tree.json``, per-child ``observation.json``, ``summary.json``).

>>> import benchflow as bf
>>> bf.BranchResult.__name__
'BranchResult'
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from benchflow.branch_run import (
    BranchPlan,
    BranchPlanError,
    BranchTrialOutcome,
    ChildSpec,
    load_checkpoint_source,
    parse_child_spec,
    run_branch_trial,
    write_branch_job,
)
from benchflow.checkpoints import CheckpointPolicy
from benchflow.models import RolloutResult

__all__ = [
    "BranchChildResult",
    "BranchPlanError",
    "BranchResult",
    "ChildSpec",
    "abranch",
    "branch",
]

ChildrenArg = Mapping[str, str | None] | Sequence[ChildSpec | str]


@dataclass(frozen=True)
class BranchChildResult:
    """One child of a branched run, as recorded in ``tree.json``.

    ``reward`` is None when the child was not scored (never a silent 0);
    ``reward_source`` says where a reward came from (``verifier`` or
    ``runner_return``). ``parent_label`` names the child a nested fork was
    made from (None for children of the checkpoint).
    """

    label: str
    status: str
    reward: float | None
    reward_source: str | None
    path: str | None
    fork_id: str | None
    parent_label: str | None = None
    # tokens, usd (None unless the provider reported one), sandbox_seconds.
    cost: dict[str, Any] | None = None
    # reward - V of the child's own fork; None when either is missing.
    advantage: float | None = None

    def to_record(self) -> dict[str, Any]:
        """The child as a flat dict."""
        return {
            "label": self.label,
            "status": self.status,
            "reward": self.reward,
            "reward_source": self.reward_source,
            "path": self.path,
            "fork_id": self.fork_id,
            "parent_label": self.parent_label,
            "advantage": self.advantage,
        }


@dataclass
class BranchResult:
    """What one branched run produced.

    ``value`` is V(checkpoint), the mean reward of the checkpoint's children.
    ``parent`` is the parent rollout's :class:`RolloutResult` (its
    ``branches`` block is in ``parent.rollout_dir / "result.json"``);
    ``parent_reward`` is None when the parent was discarded. ``error`` is set
    when the trial failed; the children that ran are still listed.
    """

    task: str
    job_dir: Path
    rollout_dir: Path | None
    value: float | None
    children: list[BranchChildResult]
    parent: RolloutResult | None
    parent_reward: float | None
    fork_status: str | None
    parent_restore: str | None
    kept_snapshot: str | None = None
    error: str | None = None
    timing_sec: dict[str, Any] | None = field(default=None, repr=False)
    # Trial totals (tokens, usd, usd_known, sandbox_seconds) and one row per
    # fork (id, from, children, value, cost); see docs/branching.md.
    cost: dict[str, Any] | None = field(default=None, repr=False)
    forks: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        """True when the fork completed and the trial raised no error."""
        return self.error is None and self.fork_status == "completed"

    def child(self, label: str) -> BranchChildResult:
        """The child with this label (KeyError naming the labels otherwise)."""
        for each in self.children:
            if each.label == label:
                return each
        raise KeyError(
            f"no child {label!r}; children: {[c.label for c in self.children]}"
        )

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per child, with the task and V(checkpoint)."""
        return [
            {"task": self.task, "value": self.value, **c.to_record()}
            for c in self.children
        ]

    def to_csv(self, path: str | Path) -> Path:
        """Write one CSV row per child and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        fields = ["task", "value", *BranchChildResult.__dataclass_fields__]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        return path

    def to_jsonl(self, path: str | Path) -> Path:
        """Write one JSON line per child and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in self.to_records()))
        return path


def _children(children: ChildrenArg) -> list[ChildSpec]:
    if isinstance(children, Mapping):
        specs: list[ChildSpec] = []
        for label, prompt in children.items():
            if prompt is not None and not isinstance(prompt, str):
                raise TypeError(
                    f"child {label!r}: the prompt must be a str or None, "
                    f"got {type(prompt).__name__}"
                )
            if prompt is not None and not prompt.strip():
                raise BranchPlanError(
                    f"child {label!r}: the prompt is empty (use None for the "
                    "parent's remaining prompts)"
                )
            specs.append(ChildSpec(label=str(label), prompt=prompt))
        return specs
    specs = []
    for item in children:
        if isinstance(item, ChildSpec):
            specs.append(item)
        elif isinstance(item, str):
            specs.append(parse_child_spec(item))
        else:
            raise TypeError(
                f"children items are ChildSpec or 'label=NAME[,prompt=TEXT]' "
                f"strings, got {type(item).__name__}"
            )
    return specs


def _result(outcome: BranchTrialOutcome, job_dir: Path) -> BranchResult:
    rollout_dir = Path(outcome.rollout_dir) if outcome.rollout_dir else None
    fork_values = {f.get("id"): f.get("value") for f in outcome.forks}

    def advantage(child: dict[str, Any]) -> float | None:
        value = fork_values.get(
            child.get("fork_id"),
            outcome.value if child.get("parent_label") is None else None,
        )
        reward = child.get("reward")
        if isinstance(reward, int | float) and isinstance(value, int | float):
            return round(reward - value, 6)
        return None

    parent = None
    if rollout_dir is not None and (rollout_dir / "result.json").is_file():
        parent = RolloutResult.load(rollout_dir)
    return BranchResult(
        task=outcome.task,
        job_dir=job_dir,
        rollout_dir=rollout_dir,
        value=outcome.value,
        children=[
            BranchChildResult(
                label=str(c["label"]),
                status=str(c["status"]),
                reward=c.get("reward"),
                reward_source=c.get("reward_source"),
                path=c.get("path"),
                fork_id=c.get("fork_id"),
                parent_label=c.get("parent_label"),
                cost=c.get("cost"),
                advantage=advantage(c),
            )
            for c in outcome.children
        ],
        parent=parent,
        parent_reward=outcome.parent_reward,
        fork_status=outcome.fork_status,
        parent_restore=outcome.parent_restore,
        kept_snapshot=outcome.kept_snapshot,
        error=outcome.error,
        timing_sec=outcome.timing_sec,
        cost=outcome.cost,
        forks=list(outcome.forks),
    )


# CLI flag -> Python keyword, for BranchPlanError messages raised to Python
# callers (the plan checks are shared with bench eval branch).
_PYTHON_NAMES = {
    "checkpoint-after-prompt": "checkpoint_after=",
    "checkpoints": "checkpoints=",
    "checkpoint": "checkpoint=",
    "child-retries": "child_retries=",
    "child": "children",
    "concurrency": "concurrency=",
    "from-checkpoint": "from_checkpoint=",
    "fork": "fork=",
    "isolate-children": "isolate_children=True",
    "parent": "parent=",
    "prompt": "prompts=",
    "resume-session": "resume_session=True",
    "retain-snapshots": "retain_snapshots=True",
    "sandbox": "sandbox=",
    "snapshot-layers": "snapshot_layers=",
    "stop-on-child-failure": "continue_after_child_failure=False",
    "tasks-dir": "task_path",
}
_FLAG_NAMES = sorted(_PYTHON_NAMES, key=lambda name: -len(name))
_FLAG = re.compile(r"--(" + "|".join(_FLAG_NAMES) + r")(?![a-z-])")


def _python_names(message: str) -> str:
    return _FLAG.sub(lambda m: _PYTHON_NAMES[m.group(1)], message)


def _plan(
    task_path: str | Path | None,
    *,
    agent: str | None,
    children: ChildrenArg,
    model: str | None,
    prompts: Iterable[str] | None,
    checkpoint_after: int | None,
    sandbox: str | None,
    parent: Literal["continue", "discard"] | None,
    snapshot_layers: Iterable[str],
    concurrency: int,
    isolate_children: bool,
    resume_session: bool,
    child_retries: int,
    continue_after_child_failure: bool,
    retain_snapshots: bool,
    from_checkpoint: str | Path | None,
    fork: str | None,
    checkpoints: CheckpointPolicy | None,
    jobs_dir: str | Path,
    job_name: str | None,
    agent_env: Mapping[str, str] | None,
    reasoning_effort: str | None,
    checkpoint: str | None = None,
    harness: str = "acp",
) -> BranchPlan:
    try:
        return _build_plan(
            task_path,
            agent=agent,
            children=children,
            model=model,
            prompts=prompts,
            checkpoint_after=checkpoint_after,
            sandbox=sandbox,
            parent=parent,
            snapshot_layers=snapshot_layers,
            concurrency=concurrency,
            isolate_children=isolate_children,
            resume_session=resume_session,
            child_retries=child_retries,
            continue_after_child_failure=continue_after_child_failure,
            retain_snapshots=retain_snapshots,
            from_checkpoint=from_checkpoint,
            fork=fork,
            checkpoints=checkpoints,
            jobs_dir=jobs_dir,
            job_name=job_name,
            agent_env=agent_env,
            reasoning_effort=reasoning_effort,
            checkpoint=checkpoint,
            harness=harness,
        )
    except BranchPlanError as exc:
        raise BranchPlanError(_python_names(str(exc))) from None


def _build_plan(
    task_path: str | Path | None,
    *,
    agent: str | None,
    children: ChildrenArg,
    model: str | None,
    prompts: Iterable[str] | None,
    checkpoint_after: int | None,
    sandbox: str | None,
    parent: Literal["continue", "discard"] | None,
    snapshot_layers: Iterable[str],
    concurrency: int,
    isolate_children: bool,
    resume_session: bool,
    child_retries: int,
    continue_after_child_failure: bool,
    retain_snapshots: bool,
    from_checkpoint: str | Path | None,
    fork: str | None,
    checkpoints: CheckpointPolicy | None,
    jobs_dir: str | Path,
    job_name: str | None,
    agent_env: Mapping[str, str] | None,
    reasoning_effort: str | None,
    checkpoint: str | None,
    harness: str = "acp",
) -> BranchPlan:
    from benchflow._utils.config import normalize_agent_name
    from benchflow.rollout import RolloutConfig
    from benchflow.runtime import check_host, check_rollout_config

    if checkpoint is not None and fork is not None:
        raise BranchPlanError("give checkpoint= or fork=, not both")
    fork = checkpoint if checkpoint is not None else fork
    source = (
        load_checkpoint_source(Path(from_checkpoint), fork)
        if from_checkpoint is not None
        else None
    )
    if fork is not None and source is None:
        raise BranchPlanError("checkpoint= (or fork=) needs from_checkpoint=")
    # Defaults from the checkpoint's trial, as bench eval branch does.
    recorded: dict[str, Any] = {}
    if source is not None:
        try:
            recorded = json.loads((source.trial_dir / "config.json").read_text())
        except (OSError, ValueError):
            recorded = {}
    if task_path is None:
        recorded_task = recorded.get("task_path")
        if not recorded_task:
            raise BranchPlanError(
                "task_path is required (it defaults to the task of "
                "from_checkpoint= when that trial recorded it)"
            )
        if not Path(str(recorded_task)).is_dir():
            raise BranchPlanError(
                f"task_path is required: the checkpoint's task {recorded_task} "
                "is not a directory here"
            )
        task_path = Path(str(recorded_task))
    if agent is None:
        if not recorded.get("agent"):
            raise BranchPlanError("agent= is required")
        agent = str(recorded["agent"])
        model = model or recorded.get("model")
    sandbox = sandbox or (source.provider if source else "docker")
    requested = RolloutConfig(
        task_path=Path(task_path), agent=agent, environment=sandbox, harness=harness
    )
    check_rollout_config(requested)
    check_host([requested])
    agent = normalize_agent_name(agent)
    plan = BranchPlan(
        task_paths=[Path(task_path)],
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
        harness=requested.harness,
        sandbox=sandbox,
        children=_children(children),
        checkpoint_after=checkpoint_after
        if checkpoint_after is not None
        else (0 if agent == "oracle" or source is not None else 1),
        snapshot_layers=frozenset(snapshot_layers),
        parent_mode=parent or ("discard" if source is not None else "continue"),
        retain_snapshots=retain_snapshots,
        prompts=list(prompts) if prompts is not None else None,
        agent_env=dict(agent_env or {}),
        jobs_dir=Path(jobs_dir),
        job_name=job_name or f"branch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        source=source,
        checkpoints=checkpoints,
        resume_session=resume_session,
        child_retries=child_retries,
        continue_after_child_failure=continue_after_child_failure,
        concurrency=concurrency,
        isolate_children=isolate_children,
    )
    plan.validate()
    return plan


async def abranch(
    task_path: str | Path | None = None,
    *,
    agent: str | None = None,
    children: ChildrenArg,
    model: str | None = None,
    prompts: Iterable[str] | None = None,
    checkpoint_after: int | None = None,
    sandbox: str | None = None,
    parent: Literal["continue", "discard"] | None = None,
    snapshot_layers: Iterable[str] = ("sandbox",),
    concurrency: int = 1,
    isolate_children: bool = False,
    resume_session: bool = False,
    child_retries: int = 1,
    continue_after_child_failure: bool = True,
    retain_snapshots: bool = False,
    from_checkpoint: str | Path | None = None,
    fork: str | None = None,
    checkpoint: str | None = None,
    checkpoints: CheckpointPolicy | None = None,
    jobs_dir: str | Path = "jobs",
    job_name: str | None = None,
    agent_env: Mapping[str, str] | None = None,
    reasoning_effort: str | None = None,
    harness: str = "acp",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> BranchResult:
    """Branch one task's run at a checkpoint into labelled, verifier-scored children.

    ``children`` maps each label to its own prompt (``None``: the parent's
    remaining prompts, or the task's prompts when the checkpoint is after the
    last one), or is a list of :class:`ChildSpec` / ``"label=...,prompt=..."``
    strings, which also allow nested forks (``parent=LABEL``). At least two
    children fork from the checkpoint.

    ``prompts`` are the parent's prompts (``"@instruction"`` is the task
    instruction; default: the task's own), and ``checkpoint_after`` is how many
    of them run before the fork (default 1; 0 for the oracle). Children start
    with a fresh agent session unless ``resume_session=True``; only files (and
    declared environment state with ``snapshot_layers={"sandbox",
    "environment"}``) are restored, so child prompts must stand on their own.
    ``parent="discard"`` skips the parent's restore and verification.
    ``concurrency > 1`` or ``isolate_children`` runs each child in its own
    sandbox. ``from_checkpoint`` (a trial folder with a kept snapshot, see
    ``retain_snapshots``) branches again from a checkpoint an earlier trial
    kept; ``checkpoint`` picks one (a fork id or an automatic checkpoint
    ``"prompt:N"``; ``fork`` is the older name), and ``task_path``,
    ``agent`` and ``model`` default to that trial's. ``on_event`` receives
    progress dicts (``event``: ``trial_started``, ``checkpoint_reached``,
    ``child_finished`` with ``label`` and ``reward``, ``fork_finished``,
    ``parent_finished``, ``trial_finished``). Invalid requests raise
    :class:`BranchPlanError` (a ``ValueError``, naming these keywords)
    before anything starts; a trial that fails later is returned with
    ``error`` set.
    """
    plan = _plan(
        task_path,
        agent=agent,
        children=children,
        model=model,
        prompts=prompts,
        checkpoint_after=checkpoint_after,
        sandbox=sandbox,
        parent=parent,
        snapshot_layers=snapshot_layers,
        concurrency=concurrency,
        isolate_children=isolate_children,
        resume_session=resume_session,
        child_retries=child_retries,
        continue_after_child_failure=continue_after_child_failure,
        retain_snapshots=retain_snapshots,
        from_checkpoint=from_checkpoint,
        fork=fork,
        checkpoints=checkpoints,
        jobs_dir=jobs_dir,
        job_name=job_name,
        agent_env=agent_env,
        reasoning_effort=reasoning_effort,
        checkpoint=checkpoint,
        harness=harness,
    )
    task = plan.task_paths[0]

    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            on_event({"event": event, "task": task.name, **fields})

    emit(
        "trial_started",
        agent=plan.agent,
        sandbox=plan.sandbox,
        children=[c.label for c in plan.children],
    )
    outcome = await run_branch_trial(plan, task, on_event=on_event)
    job_dir = write_branch_job(plan, [outcome])
    result = _result(outcome, job_dir)
    emit(
        "trial_finished",
        value=result.value,
        error=result.error,
        job_dir=str(job_dir),
    )
    return result


def branch(
    task_path: str | Path | None = None,
    *,
    agent: str | None = None,
    children: ChildrenArg,
    model: str | None = None,
    prompts: Iterable[str] | None = None,
    checkpoint_after: int | None = None,
    sandbox: str | None = None,
    parent: Literal["continue", "discard"] | None = None,
    snapshot_layers: Iterable[str] = ("sandbox",),
    concurrency: int = 1,
    isolate_children: bool = False,
    resume_session: bool = False,
    child_retries: int = 1,
    continue_after_child_failure: bool = True,
    retain_snapshots: bool = False,
    from_checkpoint: str | Path | None = None,
    fork: str | None = None,
    checkpoint: str | None = None,
    checkpoints: CheckpointPolicy | None = None,
    jobs_dir: str | Path = "jobs",
    job_name: str | None = None,
    agent_env: Mapping[str, str] | None = None,
    reasoning_effort: str | None = None,
    harness: str = "acp",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> BranchResult:
    """Blocking form of :func:`abranch`; takes exactly the same arguments.

    Example (runs a real sandbox, so not a doctest)::

        result = bf.branch(
            "tests/examples/hello-world-task",
            agent="claude-agent-acp",
            model="claude-haiku-4-5",
            prompts=["Create draft.txt containing: Hello world", "@instruction"],
            children={
                "baseline": None,  # the parent's remaining prompts
                "hint": "Rename draft.txt to hello.txt and stop.",
            },
            sandbox="daytona",
        )
        print(result.value, [(c.label, c.reward) for c in result.children])
    """
    from benchflow.batch import run_blocking

    return run_blocking(
        lambda: abranch(
            task_path,
            agent=agent,
            children=children,
            model=model,
            prompts=prompts,
            checkpoint_after=checkpoint_after,
            sandbox=sandbox,
            parent=parent,
            snapshot_layers=snapshot_layers,
            concurrency=concurrency,
            isolate_children=isolate_children,
            resume_session=resume_session,
            child_retries=child_retries,
            continue_after_child_failure=continue_after_child_failure,
            retain_snapshots=retain_snapshots,
            from_checkpoint=from_checkpoint,
            fork=fork,
            checkpoints=checkpoints,
            jobs_dir=jobs_dir,
            job_name=job_name,
            agent_env=agent_env,
            reasoning_effort=reasoning_effort,
            checkpoint=checkpoint,
            harness=harness,
            on_event=on_event,
        )
    )

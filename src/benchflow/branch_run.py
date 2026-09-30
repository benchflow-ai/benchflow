"""``bench eval branch``: branch an agent run from the command line.

One *branch trial* per task drives the rollout lifecycle that
``docs/composed-checkpoints.md`` describes by hand::

    setup -> start -> install_agent
          -> connect -> execute(prompts[:N])            (the checkpoint)
          -> branch(children)                           (tree.json)
          -> connect -> execute(prompts[N:]) -> verify  (--parent continue)
          -> finalize                                   (result.json)

Each child is a fresh agent session started from the checkpoint (with
``--resume-session``, the parent's conversation reloaded through ACP
``session/load``), sent its own prompt and scored by the task's verifier. With ``--parent discard`` the parent
restore after the last child is skipped (``restore_parent=False``) and the
parent is finalized unverified.

A trial can also start from a checkpoint an earlier trial kept
(``--retain-snapshots``): the new rollout starts the task's sandbox, replaces it
with one restored from the kept snapshot, installs the agent again (credential
files never enter a snapshot) and branches from there. Only the sandbox layer
can be reused this way; declared environment state is not exportable.

The job folder has the normal layout (one ``<task>__<id>`` folder per trial,
``results.jsonl``) plus a ``summary.json`` whose counts are over children.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.checkpoints import CheckpointPolicy, after_prompt
from benchflow.errors import UserError
from benchflow.review.persistence import write_json_atomic

logger = logging.getLogger(__name__)

ParentMode = Literal["continue", "discard"]
_CHILD_KEYS = ("label", "parent", "prompt", "prompt-file")
_MAX_LABEL = 200


class BranchPlanError(ValueError, UserError):
    """The branch request is inconsistent; nothing was started."""


@dataclass(frozen=True)
class ChildSpec:
    """One ``--child`` option: a label and the child's own prompt.

    ``prompt`` None means the child runs the parent's remaining prompts, or the
    task's prompts when the checkpoint is after the last one. ``parent`` names
    another child: this child is then forked from that child's state once the
    parent child has run its prompts (a nested fork); a nested child without a
    prompt runs the task's prompts.
    """

    label: str
    prompt: str | None = None
    parent: str | None = None


def parse_child_spec(spec: str, *, base_dir: Path | None = None) -> ChildSpec:
    """Parse ``label=NAME[,parent=LABEL][,prompt=TEXT|,prompt-file=PATH]``.

    ``prompt=`` takes the rest of the option, so the prompt may contain commas;
    it must come last. ``prompt-file=`` is read relative to ``base_dir``.
    """
    fields: dict[str, str] = {}
    rest = spec
    while rest:
        key, sep, value = rest.partition("=")
        key = key.strip()
        if not sep or key not in _CHILD_KEYS:
            raise BranchPlanError(
                f"--child {spec!r}: expected "
                "label=NAME[,parent=LABEL][,prompt=TEXT|,prompt-file=PATH]"
            )
        if key in fields:
            raise BranchPlanError(f"--child {spec!r}: {key} given twice")
        if key == "prompt":
            fields[key], rest = value, ""
        else:
            fields[key], _, rest = value.partition(",")
    label = fields.get("label", "").strip()
    if not label or len(label) > _MAX_LABEL:
        raise BranchPlanError(
            f"--child {spec!r}: label=NAME is required (1-{_MAX_LABEL} characters)"
        )
    if "prompt" in fields and "prompt-file" in fields:
        raise BranchPlanError(f"--child {spec!r}: give prompt or prompt-file, not both")
    prompt = fields.get("prompt")
    if "prompt-file" in fields:
        path = Path(fields["prompt-file"]).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        try:
            prompt = path.read_text()
        except OSError as exc:
            raise BranchPlanError(
                f"--child {spec!r}: cannot read {path}: {exc}"
            ) from None
    if prompt is not None and not prompt.strip():
        raise BranchPlanError(f"--child {spec!r}: the prompt is empty")
    parent = fields.get("parent", "").strip() or None
    return ChildSpec(label=label, prompt=prompt, parent=parent)


def parse_snapshot_layers(value: str) -> frozenset[str]:
    layers = frozenset(part.strip() for part in value.split(",") if part.strip())
    unknown = layers - {"environment", "sandbox"}
    if not layers or unknown:
        raise BranchPlanError(
            f"--snapshot-layers {value!r}: use sandbox, environment or both "
            "(comma-separated)"
        )
    return layers


@dataclass(frozen=True)
class CheckpointSource:
    """A sandbox snapshot an earlier trial kept, to branch from again."""

    trial_dir: Path
    fork_id: str
    provider: str
    ref: str
    task_name: str
    # The checkpoint's agent session id (only automatic checkpoints record it).
    session_id: str | None = None
    # Events of the source trial's trajectory the checkpoint holds (the
    # exported prefix of forks from it); None for checkpoints recorded before this field existed.
    prefix_events: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "trial": self.trial_dir.name,
            "trial_path": str(self.trial_dir.resolve()),
            "fork_id": self.fork_id,
            "provider": self.provider,
            "ref": self.ref,
            "prefix_events": self.prefix_events,
        }


def load_checkpoint_source(trial_dir: Path, fork_id: str | None) -> CheckpointSource:
    """Find a kept sandbox snapshot of an earlier trial.

    Two sources: a fork of ``tree.json`` that kept its snapshot
    (``--retain-snapshots``), and the automatic checkpoints of
    ``checkpoints.json`` (``--checkpoints``; selected as ``prompt:N``).
    ``fork_id`` picks one (a fork id or ``prompt:N``); by default the last
    kept fork, else the last kept checkpoint. Refuses deleted snapshots and
    environment-layer forks, which cannot be re-imported.
    """
    from benchflow.checkpoints import load_checkpoints

    try:
        config = json.loads((trial_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchPlanError(
            f"--from-checkpoint {trial_dir}: not a trial folder (needs "
            f"config.json): {exc}"
        ) from None
    task_name = Path(str(config.get("task_path", ""))).name
    try:
        tree = json.loads((trial_dir / "tree.json").read_text())
    except (OSError, json.JSONDecodeError):
        tree = {}
    forks = [fork for fork in tree.get("forks", []) if isinstance(fork, dict)]
    rows = load_checkpoints(trial_dir)
    if fork_id is not None and fork_id.startswith("prompt:"):
        forks = []
        rows = [row for row in rows if row.get("id") == fork_id]
        if not rows:
            raise BranchPlanError(
                f"--checkpoint {fork_id}: no such checkpoint in {trial_dir}"
            )
    elif fork_id is not None:
        forks = [fork for fork in forks if fork.get("id") == fork_id]
        rows = []
        if not forks:
            raise BranchPlanError(
                f"--fork {fork_id}: no such fork in {trial_dir / 'tree.json'}"
            )
    kept = [
        fork
        for fork in forks
        if (fork.get("snapshot") or {}).get("retention") == "kept"
        and (fork.get("snapshot") or {}).get("sandbox")
        # A fork that reused a checkpoint points at that checkpoint's image,
        # which checkpoints.json governs (and may delete later).
        and not (fork.get("snapshot") or {}).get("reused")
    ]
    kept_rows = [row for row in rows if row.get("status") == "kept" and row.get("ref")]
    if not kept and kept_rows:
        row = kept_rows[-1]
        return CheckpointSource(
            trial_dir=trial_dir,
            fork_id=str(row["id"]),
            provider=str(row["provider"]),
            ref=str(row["ref"]),
            task_name=task_name,
            session_id=row.get("agent_session_id"),
            prefix_events=row.get("trajectory_events")
            if isinstance(row.get("trajectory_events"), int)
            else None,
        )
    if not kept:
        if fork_id is not None and rows:
            raise BranchPlanError(
                f"--checkpoint {fork_id}: that snapshot is {rows[0].get('status')}, "
                "not kept"
            )
        raise BranchPlanError(
            f"--from-checkpoint {trial_dir}: no kept sandbox snapshot. A trial "
            "can be branched again when it was made with --retain-snapshots "
            "(bench eval branch) or --checkpoints (bench eval run or branch), "
            "and its snapshot has not been deleted since."
        )
    fork = kept[-1]
    snapshot = fork["snapshot"]
    if "environment" in snapshot.get("captured_layers", []):
        raise BranchPlanError(
            f"--from-checkpoint {trial_dir}: fork {fork['id']} also captured "
            "declared environment state, which cannot be re-imported; only "
            "sandbox-only checkpoints can be branched again."
        )
    return CheckpointSource(
        trial_dir=trial_dir,
        fork_id=str(fork["id"]),
        provider=str(snapshot["sandbox"]["provider"]),
        ref=str(snapshot["sandbox"]["ref"]),
        task_name=task_name,
        prefix_events=_fork_prefix_events(tree, fork),
    )


def _fork_prefix_events(tree: dict[str, Any], fork: dict[str, Any]) -> int | None:
    """Trajectory events up to a fork's parent node (its step id is
    ``step-<i>-<type>``); None when the node or its step is unknown."""
    import re

    node = next(
        (
            n
            for n in tree.get("nodes") or []
            if isinstance(n, dict) and n.get("id") == fork.get("parent_node")
        ),
        None,
    )
    if node is None:
        return None
    if node.get("step_id") is None:
        return 0
    match = re.match(r"^step-(\d+)-", str(node.get("step_id")))
    return int(match.group(1)) + 1 if match else None


@dataclass
class BranchPlan:
    """A validated ``bench eval branch`` request."""

    task_paths: list[Path]
    agent: str
    model: str | None
    sandbox: str
    children: list[ChildSpec]
    checkpoint_after: int
    snapshot_layers: frozenset[str] = frozenset({"sandbox"})
    parent_mode: ParentMode = "continue"
    retain_snapshots: bool = False
    prompts: list[str] | None = None
    reasoning_effort: str | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    jobs_dir: Path = Path("jobs")
    job_name: str = "branch"
    source: CheckpointSource | None = None
    checkpoints: CheckpointPolicy | None = None
    resume_session: bool = False
    # A child that failed before its agent did anything is retried this many
    # times; one failed child does not stop its siblings.
    child_retries: int = 1
    continue_after_child_failure: bool = True
    concurrency: int = 1
    isolate_children: bool = False

    @property
    def is_oracle(self) -> bool:
        return self.agent == "oracle"

    @property
    def isolated(self) -> bool:
        """Children in their own sandboxes: asked for, parallel, or nested."""
        return (
            self.isolate_children
            or self.concurrency > 1
            or any(child.parent is not None for child in self.children)
        )

    def kids(self, parent: str | None) -> list[ChildSpec]:
        """The children forked from ``parent`` (None: the checkpoint)."""
        return [child for child in self.children if child.parent == parent]

    def validate(self) -> None:
        if len(self.kids(None)) < 2:
            raise BranchPlanError("give at least two top-level --child options")
        labels = [child.label for child in self.children]
        if len(set(labels)) != len(labels):
            raise BranchPlanError(f"child labels must be unique, got {labels}")
        for parent in {c.parent for c in self.children if c.parent is not None}:
            if parent not in labels:
                raise BranchPlanError(f"--child: unknown parent {parent!r}")
            if len(self.kids(parent)) < 2:
                raise BranchPlanError(
                    f"a nested fork needs at least two children under {parent!r}"
                )
        if self.concurrency < 1:
            raise BranchPlanError("--concurrency must be at least 1")
        if self.child_retries < 0:
            raise BranchPlanError("--child-retries must be 0 or more")
        if self.resume_session and self.source is not None:
            if self.is_oracle or not self.source.session_id:
                raise BranchPlanError(
                    "--resume-session with --from-checkpoint needs a checkpoint that "
                    "recorded its agent session id (automatic --checkpoints do; "
                    "--retain-snapshots forks and oracle runs do not); this one did not"
                )
        elif self.resume_session and (self.is_oracle or self.checkpoint_after == 0):
            raise BranchPlanError(
                "--resume-session needs an agent conversation before the fork: "
                "an agent (not the oracle) and --checkpoint-after-prompt 1 or more"
            )
        if self.isolated and self.snapshot_layers != frozenset({"sandbox"}):
            raise BranchPlanError(
                "--concurrency, --isolate-children and nested children use the "
                "sandbox layer only (--snapshot-layers sandbox)"
            )
        if not self.task_paths:
            raise BranchPlanError("no task selected")
        if self.checkpoint_after < 0:
            raise BranchPlanError("--checkpoint-after-prompt must be 0 or more")
        if self.is_oracle:
            if any(child.prompt is not None for child in self.children):
                raise BranchPlanError(
                    "the oracle takes no prompts: every oracle child runs the "
                    "task's solve.sh, so drop prompt= from --child"
                )
            if self.prompts:
                raise BranchPlanError("the oracle takes no --prompt")
            if self.checkpoint_after > 1:
                raise BranchPlanError(
                    "the oracle has one step (solve.sh): "
                    "--checkpoint-after-prompt must be 0 or 1"
                )
        if self.prompts is not None and self.checkpoint_after > len(self.prompts):
            raise BranchPlanError(
                f"--checkpoint-after-prompt {self.checkpoint_after} is after the "
                f"last of {len(self.prompts)} --prompt"
            )
        if self.source is not None:
            if self.snapshot_layers != frozenset({"sandbox"}):
                raise BranchPlanError(
                    "--from-checkpoint reuses a sandbox snapshot: use "
                    "--snapshot-layers sandbox"
                )
            if self.source.provider != self.sandbox:
                raise BranchPlanError(
                    f"the kept checkpoint is a {self.source.provider} snapshot; "
                    f"run with --sandbox {self.source.provider}"
                )
            if self.checkpoint_after:
                raise BranchPlanError(
                    "--from-checkpoint branches at the kept checkpoint; "
                    "--checkpoint-after-prompt does not apply"
                )
            if len(self.task_paths) != 1:
                raise BranchPlanError("--from-checkpoint branches exactly one task")


@dataclass
class BranchTrialOutcome:
    """What one branch trial produced, for summary.json and the console."""

    task: str
    rollout_dir: str | None = None
    value: float | None = None
    fork_status: str | None = None
    parent_restore: str | None = None
    parent_reward: float | None = None
    children: list[dict[str, Any]] = field(default_factory=list)
    retention: str | None = None
    kept_snapshot: str | None = None
    timing_sec: dict[str, Any] | None = None
    error: str | None = None
    source: dict[str, str] | None = None
    # One row per fork (id, from, children, value, cost) and trial totals.
    forks: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] | None = None


INSTRUCTION = "@instruction"


def child_prompts(
    plan: BranchPlan,
    resolved: list[str],
    prompt_prefix: str | None = None,
    specs: list[ChildSpec] | None = None,
) -> list[list[str]]:
    """The prompt list each child is sent, from the parent's resolved prompts.

    ``resolved`` already carries the task's ``prompt_prefix``; a child's own
    prompt gets the same prefix, so every arm sees the same harness policy.
    ``specs`` defaults to the top-level children; nested children without a
    prompt run the task's prompts.
    """
    from benchflow.rollout._setup import _apply_prompt_prefix

    specs = plan.kids(None) if specs is None else specs
    nested = any(spec.parent is not None for spec in specs)
    remaining = resolved if nested else resolved[plan.checkpoint_after :] or resolved
    return [
        _apply_prompt_prefix([child.prompt], prompt_prefix)
        if child.prompt
        else list(remaining)
        for child in specs
    ]


def child_requests(
    plan: BranchPlan, resolved: list[str], specs: list[ChildSpec] | None = None
) -> list[str]:
    """tree.json's ``intervention.requested`` per child: short, no prompt text.

    A child's own prompt is named by length and digest (its text is in the
    child's trajectory); tree.json stays an allowlisted display record.
    """
    specs = plan.kids(None) if specs is None else specs
    if plan.is_oracle:
        return ["oracle solve.sh"] * len(specs)
    nested = any(spec.parent is not None for spec in specs)
    remaining = [] if nested else resolved[plan.checkpoint_after :]
    default = (
        f"parent's remaining prompts ({len(remaining)})"
        if remaining
        else f"the task's prompts ({len(resolved)})"
    )
    requests = []
    for child in specs:
        if child.prompt is None:
            requests.append(default)
            continue
        digest = hashlib.sha256(child.prompt.encode()).hexdigest()[:12]
        requests.append(f"own prompt ({len(child.prompt)} characters, sha256:{digest})")
    return requests


async def _oracle_turn(rollout: Any, node: Any = None) -> None:
    """Run the task's solve.sh as one step, filling ``node`` when given."""
    from benchflow.rollout import _run_oracle
    from benchflow.trajectories.tree import Step

    await rollout._env.exec(
        "git config --global --add safe.directory "
        f"{shlex.quote(rollout._agent_cwd)} 2>/dev/null || true",
        user="root",
        timeout_sec=10,
    )
    events, rollout._agent_name = await _run_oracle(
        rollout._env, rollout._config.task_path, rollout._timeout, sandbox_user=None
    )
    rollout._trajectory.extend(events)
    step = Step(
        id=f"step-{len(rollout._trajectory) - 1}-oracle",
        data={"event": events[0], "event_type": "oracle", "n_tool_calls": 0},
    )
    if node is not None:
        rollout._cursor = rollout._tree.populate(node, step)
    else:
        rollout._cursor = rollout._tree.advance(rollout._cursor, step)


async def _run_prompts(rollout: Any, prompts: list[str], *, oracle: bool, node=None):
    if oracle:
        await _oracle_turn(rollout, node)
        return
    await rollout.connect()
    await rollout.execute(prompts, node=node)


async def _run_parent_prompts(
    rollout: Any, prompts: list[str], *, first_number: int, oracle: bool
) -> None:
    """The parent's prompts, one execute each, each followed by its opt-in
    checkpoint (``--checkpoints``; numbered across the whole run)."""
    if not prompts:
        return
    if oracle:
        await _oracle_turn(rollout)
        await after_prompt(rollout, first_number)
        return
    await rollout.connect()
    for offset, prompt in enumerate(prompts):
        await rollout.execute([prompt])
        await after_prompt(rollout, first_number + offset)


def _make_runner(
    rollout: Any,
    plan: BranchPlan,
    prompts: list[list[str]],
    *,
    specs: list[ChildSpec] | None = None,
    resolved: list[str] | None = None,
    prefix: str | None = None,
    emit: Callable[..., None] | None = None,
):
    """The CLI's child runner: send the child's prompts, fork its own
    children (nested ``--child ...,parent=LABEL``) from its state, verify."""
    emit = emit or (lambda event, **fields: None)
    from benchflow.rollout_branch import BranchChild

    specs = plan.kids(None) if specs is None else specs

    async def run_child(node, *, child: BranchChild) -> float | None:
        target = child.rollout if child.rollout is not None else rollout
        rewards = None
        try:
            await _run_prompts(
                target, prompts[child.index], oracle=plan.is_oracle, node=node
            )
            kids = plan.kids(specs[child.index].label)
            if kids:
                await target.branch(
                    len(kids),
                    _make_runner(
                        target,
                        plan,
                        child_prompts(plan, resolved or [], prefix, kids),
                        specs=kids,
                        resolved=resolved,
                        prefix=prefix,
                        emit=emit,
                    ),
                    snapshot_layers={"sandbox"},
                    child_labels=[kid.label for kid in kids],
                    child_requests=list(child_requests(plan, resolved or [], kids)),
                    isolate_children=True,
                    concurrency=plan.concurrency,
                    resume_session=plan.resume_session,
                    child_retries=plan.child_retries,
                    continue_after_child_failure=plan.continue_after_child_failure,
                )
            rewards = await target.verify()
        finally:
            await target.disconnect()
        # None (no canonical reward) marks the child unscored, never 0.
        reward = (rewards or {}).get("reward")
        emit(
            "child_finished",
            label=specs[child.index].label,
            fork_id=child.fork_id,
            reward=reward,
        )
        return reward

    return run_child


def _fork_outcome(outcome: BranchTrialOutcome, rollout: Any) -> None:
    forks = getattr(rollout, "_branch_forks", None) or []
    if not forks:
        return
    # The trial's own fork comes first; nested forks follow it.
    fork = forks[0]
    outcome.value = fork.get("value")
    outcome.fork_status = fork.get("status")
    outcome.parent_restore = fork.get("parent_restore")
    outcome.timing_sec = fork.get("timing_sec")
    snapshot = fork.get("snapshot") or {}
    outcome.retention = snapshot.get("retention")
    # A reused image (a checkpoint) is not the fork's own kept snapshot.
    if (
        outcome.retention == "kept"
        and snapshot.get("sandbox")
        and not snapshot.get("reused")
    ):
        outcome.kept_snapshot = snapshot["sandbox"].get("ref")
    from benchflow.branch_lineage import cost_totals

    outcome.cost = cost_totals(forks)
    # Every child of every fork; parent_label names the child a nested fork
    # was made from (None for the trial's own fork).
    label_of = {
        child["node_id"]: child["intervention"]["label"]
        for each in forks
        for child in each.get("children", [])
    }
    outcome.forks = [
        {
            "id": each["id"],
            "from": "checkpoint"
            if each is fork
            else label_of.get(each.get("rollout")) or each.get("rollout"),
            "children": len(each.get("children", [])),
            "value": each.get("value"),
            "status": each.get("status"),
            "cost": each.get("cost"),
        }
        for each in forks
    ]
    outcome.children = [
        {
            "label": child["intervention"]["label"],
            "node_id": child["node_id"],
            "status": child["status"],
            "reward": child["reward"],
            "reward_source": child["reward_source"],
            "path": child["artifacts"]["path"],
            "fork_id": each["id"],
            "parent_label": label_of.get(each.get("rollout"))
            if each is not fork
            else None,
            "cost": child.get("cost"),
        }
        for each in forks
        for child in each.get("children", [])
    ]


def _snapshot_at_fork(rollout: Any, plan: BranchPlan, source_image: Any) -> Any:
    """An existing image of the sandbox at the fork point, or None."""
    from benchflow.checkpoints import load_checkpoints
    from benchflow.sandbox.protocol import SandboxImage

    if plan.snapshot_layers != frozenset({"sandbox"}):
        return None
    if plan.checkpoint_after == 0:
        return source_image
    cursor = getattr(getattr(rollout, "_cursor", None), "id", None)
    for row in reversed(load_checkpoints(rollout._rollout_dir)):
        if (
            row.get("after_prompt") == plan.checkpoint_after
            and row.get("status") == "kept"
            and row.get("ref")
            and row.get("node_id") == cursor
        ):
            return SandboxImage(provider=row["provider"], ref=row["ref"])
    return None


async def run_branch_trial(
    plan: BranchPlan,
    task_path: Path,
    *,
    rollout_factory: Callable[[Any], Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> BranchTrialOutcome:
    """Run one task's branch trial; never raises for a trial-level failure.

    ``on_event`` receives progress events as dicts with ``event`` (
    ``checkpoint_reached``, ``child_finished``, ``fork_finished``,
    ``parent_finished``) and ``task``; a callback that raises is logged and
    ignored."""

    def emit(event: str, **fields: Any) -> None:
        if on_event is None:
            return
        try:
            on_event({"event": event, "task": task_path.name, **fields})
        except Exception:
            logger.warning("on_event callback failed for %s", event, exc_info=True)

    from benchflow.evaluation import _environment_manifest_from_task_document
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.rollout_branch import restore_sandbox_with_services
    from benchflow.sandbox.protocol import SandboxImage

    config = RolloutConfig(
        task_path=task_path,
        agent=plan.agent,
        model=plan.model,
        reasoning_effort=plan.reasoning_effort,
        # '@instruction' is the task instruction (None in RolloutConfig).
        prompts=[None if p == INSTRUCTION else p for p in plan.prompts]
        if plan.prompts
        else None,
        environment=plan.sandbox,
        agent_env=dict(plan.agent_env) or None,
        jobs_dir=plan.jobs_dir,
        job_name=plan.job_name,
        checkpoints=plan.checkpoints,
        # As in `bench eval run`: a task.md-declared environment plane starts
        # its services and restarts them after every branch restore.
        environment_manifest=_environment_manifest_from_task_document(task_path),
    )
    rollout = (rollout_factory or Rollout)(config)
    outcome = BranchTrialOutcome(task=task_path.name)
    if plan.source is not None:
        outcome.source = plan.source.to_record()
    try:
        await rollout.setup()
        outcome.rollout_dir = str(rollout._rollout_dir)
        source_image = (
            SandboxImage(provider=plan.source.provider, ref=plan.source.ref)
            if plan.source is not None
            else None
        )
        fast = False
        if source_image is not None:
            # Create the sandbox straight from the kept checkpoint where the
            # provider can (Daytona): no fresh sandbox to throw away, and
            # start() skips uploads and setup commands the snapshot has.
            start_from = getattr(rollout._env, "start_from_snapshot", None)
            fast = bool(start_from is not None and start_from(source_image))
            if fast:
                rollout._from_branch_snapshot = True
        await rollout.start()
        if source_image is not None and not fast:
            # Replace the fresh sandbox with the kept checkpoint, then install
            # the agent again: credential files never enter a snapshot.
            await restore_sandbox_with_services(rollout, source_image)
        if source_image is not None:
            # The kept snapshot holds the source trial's pre-agent verifier
            # baseline; install_agent keeps it (and says so).
            rollout._from_branch_snapshot = True
        await rollout.install_agent()
        if plan.source is not None:
            write_json_atomic(
                rollout._require_rollout_dir() / "checkpoint_source.json",
                {
                    **plan.source.to_record(),
                    "snapshot_start": getattr(rollout, "_snapshot_start", None),
                },
            )
        resolved = [str(prompt) for prompt in rollout._resolved_prompts]
        task = getattr(rollout, "_task", None)
        prefix = task.config.agent.prompt_prefix if task is not None else None
        await _run_parent_prompts(
            rollout,
            resolved[: plan.checkpoint_after],
            first_number=1,
            oracle=plan.is_oracle,
        )
        emit("checkpoint_reached", after_prompt=plan.checkpoint_after)
        await rollout.branch(
            len(plan.kids(None)),
            _make_runner(
                rollout,
                plan,
                child_prompts(plan, resolved, prefix),
                resolved=resolved,
                prefix=prefix,
                emit=emit,
            ),
            snapshot_layers=set(plan.snapshot_layers),
            child_labels=[child.label for child in plan.kids(None)],
            child_requests=list(child_requests(plan, resolved)),
            retain_snapshots=plan.retain_snapshots,
            restore_parent=plan.parent_mode == "continue",
            isolate_children=plan.isolated,
            concurrency=plan.concurrency,
            resume_session=plan.resume_session,
            child_retries=plan.child_retries,
            continue_after_child_failure=plan.continue_after_child_failure,
            # An image of this exact state already exists: the kept checkpoint
            # the trial started from, or the automatic checkpoint just taken.
            reuse_snapshot=_snapshot_at_fork(rollout, plan, source_image),
            # A kept checkpoint's recorded session (checked by validate()).
            resume_session_id=plan.source.session_id
            if plan.source is not None and plan.resume_session
            else None,
        )
        _fork_outcome(outcome, rollout)
        emit("fork_finished", value=outcome.value, status=outcome.fork_status)
        if plan.parent_mode == "continue":
            await _run_parent_prompts(
                rollout,
                resolved[plan.checkpoint_after :],
                first_number=plan.checkpoint_after + 1,
                oracle=plan.is_oracle,
            )
            rewards = await rollout.verify()
            outcome.parent_reward = (rewards or {}).get("reward")
            emit("parent_finished", reward=outcome.parent_reward)
    except Exception as exc:
        logger.exception("Branch trial for %s failed", task_path.name)
        outcome.error = f"{type(exc).__name__}: {exc}"
        _fork_outcome(outcome, rollout)
    finally:
        try:
            if getattr(rollout, "_rollout_dir", None) is not None:
                await rollout.finalize()
            else:
                await rollout.cleanup()
        except Exception as exc:
            logger.exception("Branch trial cleanup for %s failed", task_path.name)
            outcome.error = outcome.error or f"cleanup: {type(exc).__name__}: {exc}"
    return outcome


def _job_cost(costs: list[dict[str, Any] | None]) -> dict[str, Any]:
    known = [cost for cost in costs if cost]
    usd_known = bool(known) and all(cost.get("usd_known") for cost in known)
    return {
        "tokens": sum(cost.get("tokens") or 0 for cost in known),
        "usd": round(sum(cost.get("usd") or 0.0 for cost in known), 6)
        if usd_known
        else None,
        "usd_known": usd_known,
        "sandbox_seconds": round(
            sum(cost.get("sandbox_seconds") or 0.0 for cost in known), 3
        ),
    }


def write_branch_job(plan: BranchPlan, outcomes: list[BranchTrialOutcome]) -> Path:
    """Write the job's summary.json and results.jsonl; return the job dir.

    ``total``/``passed``/``score`` count children (the unit of a branch job),
    so ``bench eval list`` shows how many children reached reward 1.0.
    """
    from benchflow.trajectories.results import write_job_results_jsonl

    job_dir = plan.jobs_dir / plan.job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    children = [child for outcome in outcomes for child in outcome.children]
    rewards = [child["reward"] for child in children if child["reward"] is not None]
    summary = {
        "kind": "benchflow-branch-job",
        "job_name": plan.job_name,
        "agent": plan.agent,
        "model": plan.model,
        "sandbox": plan.sandbox,
        "checkpoint_after_prompt": plan.checkpoint_after,
        "snapshot_layers": sorted(plan.snapshot_layers),
        "parent": plan.parent_mode,
        "children_mode": {"isolated": plan.isolated, "concurrency": plan.concurrency},
        # Labels, and whether each child had its own prompt; the prompts
        # themselves are in each child's trajectory.
        "children_requested": [
            {
                "label": child.label,
                "own_prompt": child.prompt is not None,
                "parent": child.parent,
            }
            for child in plan.children
        ],
        "total": len(children),
        "passed": sum(reward >= 1.0 for reward in rewards),
        "scored": len(rewards),
        "score": round(sum(rewards) / len(children), 4) if children else 0.0,
        "cost": _job_cost([outcome.cost for outcome in outcomes]),
        "trials": [asdict(outcome) for outcome in outcomes],
    }
    write_json_atomic(job_dir / "summary.json", summary)
    try:
        write_job_results_jsonl(job_dir)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Job-level results.jsonl aggregation failed: %s", exc)
    return job_dir

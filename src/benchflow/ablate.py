"""Stage-level ablation — run a task once, branch a stage, compare the arms.

The library half of ``bench eval ablate`` (rollout-branching RFC §5). The
branch machinery underneath is already complete: a rollout captures a stage
boundary (RFC §3.2), :meth:`~benchflow.rollout.Rollout.branch_at_stage` forks
that recorded world into one child per
:class:`~benchflow.branch_delta.BranchDelta` (RFC §3.3), and every child leaves
lineage artifacts (RFC §3.4). What was missing is the user-facing shape of the
experiment: *arms*.

An **arm** is one delta plus the name a reader recognizes it by
(``with-skill``, ``no-skill``, ``inject:<file>``). This module parses arm specs
into deltas, drives the parent rollout to the requested boundary, forks it once
into all arms, and turns the per-arm rewards into an :class:`AblationReport` —
a deterministic ``ablation.json`` plus a one-line, observation-only verdict per
arm.

Everything decidable from the request alone is decided *before* the parent
rollout runs (:func:`parse_arms`, :func:`validate_arms_for_stage`): an ablation
costs a full task run before the branch, so a request the branch engine would
reject at fork time must not cost that run first. The engine keeps its own
gates — these are a pre-flight mirror, never a replacement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.branch_delta import BranchDelta
from benchflow.branch_skill import (
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_LAYER,
    SKILL_DELTA_STAGE,
)
from benchflow.branch_stage import (
    BRANCH_STAGES,
    MARKED_STAGES,
    STAGE_ENV_READY,
    validate_stage,
)
from benchflow.rollout_branch import CHILD_WALL_CLOCK_KEY
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL

if TYPE_CHECKING:
    from benchflow.trajectories.tree import RolloutNode, RolloutTree

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The report an ablation writes into its output directory.
REPORT_FILENAME = "ablation.json"

#: The parent rollout's job name under ``--out-dir`` — fixed, not stamped, so
#: the run directory of an ablation is derivable from its output directory.
PARENT_JOB_NAME = "ablation"

#: Arm spec prefix for a plan-injection arm: ``inject:<path-to-file>``.
INJECT_PREFIX = "inject:"

ARM_KIND_SKILL_MODE = "skill-mode"
ARM_KIND_INJECT = "inject"

#: Reward at or above which a rollout counts as a pass — the framework's
#: binary convention (``bench eval metrics`` and ``bench review`` both read
#: 1.0 as passing).
PASS_REWARD = 1.0

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

#: Reference name used when an arm has no counterpart arm to compare against.
REFERENCE_PARENT = "parent"

#: The stages this command can capture on its own — everything the lifecycle
#: reaches without an explicit ``Rollout.mark_stage()`` call.
CAPTURABLE_STAGES: tuple[str, ...] = tuple(
    stage for stage in BRANCH_STAGES if stage not in MARKED_STAGES
)

_SKILL_ARM_NAMES = (SKILL_MODE_WITH_SKILL, SKILL_MODE_NO_SKILL)
_ARM_SPEC_HELP = (
    f"supported arms are {SKILL_MODE_WITH_SKILL!r}, {SKILL_MODE_NO_SKILL!r}, "
    f"and '{INJECT_PREFIX}<path-to-file>'"
)


class AblationError(Exception):
    """Base class for ablation failures the CLI reports without a traceback."""


class AblationSpecError(AblationError, ValueError):
    """The requested ablation cannot run as asked (fail closed, before any run).

    Raised for a malformed arm spec, an arm the branch engine could not execute
    at the requested stage, a stage this command cannot capture, or a
    ``--tasks-dir`` that does not name exactly one task.
    """


class AblationRunError(AblationError, RuntimeError):
    """The parent rollout never reached the stage boundary to branch from."""


# Arms


@dataclass(frozen=True)
class AblationArm:
    """One arm of the ablation: a recognizable name and the delta it runs under.

    ``name`` is the spec exactly as the caller wrote it — it is the row label
    in the table, the key in ``ablation.json``, and the reference other arms
    are compared against, so it survives the round trip verbatim. ``source``
    records where an injection arm read its text from; the text itself is
    never recorded (provenance hashes it, per #908).
    """

    name: str
    kind: str
    delta: BranchDelta
    source: str | None = None


def parse_arm(spec: str) -> AblationArm:
    """Parse one arm spec into an :class:`AblationArm` (fail closed).

    Three v1 kinds, each mapping onto exactly one executable
    :class:`BranchDelta` field: ``with-skill`` / ``no-skill`` become
    ``skill_mode`` (the child re-runs installation as a fresh rollout from the
    ``env-ready`` snapshot), and ``inject:<path>`` reads the file and becomes
    ``injected_prompt`` (the child's user-visible continuation prompt) — which
    at ``--at-stage env-ready`` is *also* delivered by a fresh rollout, since
    every child of that boundary installs the agent for itself. An
    unknown kind, an empty spec, or an injection file that is missing or blank
    raises :class:`AblationSpecError` — a silently dropped arm would publish an
    ablation table with a missing comparison.
    """
    name = spec.strip()
    if not name:
        raise AblationSpecError(f"empty arm in --arms — {_ARM_SPEC_HELP}")
    if name in _SKILL_ARM_NAMES:
        return AblationArm(
            name=name, kind=ARM_KIND_SKILL_MODE, delta=BranchDelta(skill_mode=name)
        )
    if name.startswith(INJECT_PREFIX):
        raw = name[len(INJECT_PREFIX) :].strip()
        if not raw:
            raise AblationSpecError(
                f"arm {name!r} names no file — the injection arm is "
                f"'{INJECT_PREFIX}<path-to-file>' (e.g. "
                f"'{INJECT_PREFIX}oracle-plan.md')"
            )
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AblationSpecError(
                f"arm {name!r} cannot read its injection file {raw!r}: {exc}"
            ) from None
        if not text.strip():
            raise AblationSpecError(
                f"arm {name!r} reads an empty injection file {raw!r} — an "
                "injection arm must carry the text it injects"
            )
        return AblationArm(
            name=name,
            kind=ARM_KIND_INJECT,
            delta=BranchDelta(injected_prompt=text),
            source=str(path),
        )
    raise AblationSpecError(f"unknown arm {name!r} — {_ARM_SPEC_HELP}")


def parse_arms(spec: str) -> list[AblationArm]:
    """Parse a comma-separated ``--arms`` value into ordered arms.

    Two whole-request rules beyond the per-arm ones. A fork needs at least two
    children (the branch engine's own ``n >= 2``), so a single arm is rejected
    here rather than after a full parent run; and duplicate arm names are
    rejected because the report keys arms by name, which would make the
    attribution ambiguous.
    """
    if not spec.strip():
        raise AblationSpecError(
            "--arms is empty — pass at least two arms, e.g. "
            f"'{SKILL_MODE_WITH_SKILL},{SKILL_MODE_NO_SKILL}'"
        )
    arms = [parse_arm(entry) for entry in spec.split(",")]
    seen: set[str] = set()
    for arm in arms:
        if arm.name in seen:
            raise AblationSpecError(
                f"duplicate arm {arm.name!r} in --arms — each arm is one "
                "controlled change, and the report keys arms by name"
            )
        seen.add(arm.name)
    if len(arms) < 2:
        raise AblationSpecError(
            f"--arms needs at least two arms to compare, got one "
            f"({arms[0].name!r}) — a branch forks into >= 2 children. Pair it "
            f"with a counterpart arm, e.g. "
            f"'{SKILL_MODE_WITH_SKILL},{SKILL_MODE_NO_SKILL}'"
        )
    return arms


def validate_arms_for_stage(
    arms: Sequence[AblationArm],
    stage: str,
    *,
    snapshot_layers: frozenset[str] | set[str] | None = None,
) -> str:
    """Reject a stage/arm combination before the parent rollout runs.

    Three pre-flight gates, each mirroring a rule that lives elsewhere:

    * ``post-research`` is a mid-``execute()`` cut point only an explicit
      ``Rollout.mark_stage()`` can record (RFC §3.2). This command drives the
      lifecycle, not the agent's planning, so it cannot mark it — saying so up
      front beats running the whole task and dying at the fork.
    * a ``skill_mode`` arm executes only at ``env-ready``, because skills are
      deployed by ``install_agent()`` (RFC §3.3). The branch engine fails
      closed on this too; here it costs nothing instead of a full run.
    * an ``env-ready`` ablation runs *every* arm as a fresh rollout — that
      boundary precedes ``install_agent()``, so each child installs the agent
      for itself — which needs the container layer in the stage snapshot. The
      engine raises :class:`~benchflow.rollout_branch.BranchChildExecutionNotSupported`
      for this; the mirror here is checked only when the caller passes the
      layers it will request.

    Returns the validated stage, so callers can use this as their one gate.
    """
    validate_stage(stage, field="--at-stage")
    if stage in MARKED_STAGES:
        raise AblationSpecError(
            f"--at-stage {stage!r} cannot be captured by this command: it is a "
            "mid-execute() cut point only an explicit Rollout.mark_stage() call "
            "can record. Mark it through the Python API, or ablate one of "
            f"{list(CAPTURABLE_STAGES)!r}"
        )
    for arm in arms:
        if arm.kind == ARM_KIND_SKILL_MODE and stage != SKILL_DELTA_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {SKILL_DELTA_STAGE!r}, not "
                f"{stage!r}: skills are deployed by install_agent(), which has "
                f"already run by {stage!r}, so the arm would measure nothing"
            )
    if (
        stage == FRESH_CHILD_STAGE
        and snapshot_layers is not None
        and FRESH_CHILD_LAYER not in snapshot_layers
    ):
        raise AblationSpecError(
            f"--at-stage {FRESH_CHILD_STAGE!r} needs the {FRESH_CHILD_LAYER!r} "
            f"snapshot layer, got {sorted(frozenset(snapshot_layers))!r}: that "
            "boundary precedes install_agent(), so every arm re-installs the "
            "agent for itself and the container layer is what rolls one arm's "
            "installation back before the next one runs"
        )
    return stage


def resolve_ablation_task(tasks_dir: Path) -> Path:
    """The one task an ablation runs, resolved from ``--tasks-dir``.

    An ablation compares arms *within* one task — the arms are the axis, the
    task is fixed — so a directory holding several tasks is a request this
    command cannot answer, not a batch to expand. It fails closed naming the
    tasks it found.
    """
    from benchflow.task.discovery import is_task_dir, resolve_task_collection_root

    path = Path(tasks_dir)
    if not path.is_dir():
        raise AblationSpecError(f"--tasks-dir {str(path)!r} is not a directory")
    root = resolve_task_collection_root(path)
    if is_task_dir(root):
        return root
    tasks = sorted(
        child for child in root.iterdir() if child.is_dir() and is_task_dir(child)
    )
    if not tasks:
        raise AblationSpecError(
            f"no task found under --tasks-dir {str(path)!r} — a task directory "
            "carries a task.md or task.toml"
        )
    if len(tasks) > 1:
        names = [task.name for task in tasks]
        raise AblationSpecError(
            f"--tasks-dir {str(path)!r} holds {len(names)} tasks ({names!r}) — "
            "an ablation compares arms within one task; point --tasks-dir at "
            "the task directory itself"
        )
    return tasks[0]


# Request / report


@dataclass(frozen=True)
class AblationRequest:
    """One ablation: a task, a stage boundary, and the arms to fork into."""

    task_path: Path
    arms: Sequence[AblationArm]
    agent: str
    stage: str = STAGE_ENV_READY
    model: str | None = None
    sandbox: str = "docker"
    out_dir: Path = Path("jobs")
    # The layers the stage snapshot composes (RFC §3.1). The container layer is
    # mandatory for a skills arm and sufficient on its own; the environment
    # layer needs an Environment plane, which this command does not bind, so
    # requesting it by default would fail closed at capture time.
    snapshot_layers: frozenset[str] = frozenset({SKILL_DELTA_LAYER})


@dataclass
class ArmOutcome:
    """What one arm did: its reward, its cost, and its recorded delta."""

    name: str
    kind: str
    delta: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    reward: float | None = None
    wall_clock_sec: float | None = None
    delta_execution: str | None = None
    node_id: str | None = None
    artifacts: str | None = None
    error: str | None = None
    reference: str | None = None
    verdict: str = ""

    @property
    def status(self) -> str:
        """``pass`` / ``fail`` / ``error`` / ``skipped`` — derived in one place.

        ``skipped`` is an arm that never ran because an earlier arm errored:
        the branch engine runs children sequentially and propagates a child
        failure, so the arms after it have no world to report on. They are
        reported as skipped, never as a zero reward.
        """
        if self.reward is not None:
            return STATUS_PASS if self.reward >= PASS_REWARD else STATUS_FAIL
        if self.error is not None:
            return STATUS_ERROR
        return STATUS_SKIPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "delta": self.delta,
            "delta_execution": self.delta_execution,
            "source": self.source,
            "reward": self.reward,
            "status": self.status,
            "wall_clock_sec": self.wall_clock_sec,
            "node": self.node_id,
            "artifacts": self.artifacts,
            "error": self.error,
            "reference": self.reference,
            "verdict": self.verdict,
        }


@dataclass
class AblationReport:
    """The ablation's result: the parent's own run plus one row per arm."""

    task_id: str
    task_path: str
    stage: str
    snapshot_layers: list[str]
    agent: str
    model: str | None
    sandbox: str
    arms: list[ArmOutcome]
    parent_reward: float | None = None
    parent_error: str | None = None
    parent_run_dir: str | None = None
    value: float | None = None
    error: str | None = None

    @property
    def has_errors(self) -> bool:
        """Whether any arm failed to produce a reward.

        The *parent*'s own error is deliberately not counted: attributing a
        failed run is the reason this command exists (RFC §1), so a parent that
        failed while every arm scored is a complete ablation, not a failed one.
        """
        return self.error is not None or any(
            arm.status in (STATUS_ERROR, STATUS_SKIPPED) for arm in self.arms
        )

    def to_dict(self) -> dict[str, Any]:
        """The ``ablation.json`` payload.

        Deterministic given the same rewards: arms keep request order, every
        delta is the engine's own content-addressed provenance dict, and no
        wall-clock *timestamp* is recorded anywhere. Per-arm
        ``wall_clock_sec`` is a measured duration — the one field that varies
        between identical runs, carried because an ablation's cost per arm is
        part of its result.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "task": {"id": self.task_id, "path": self.task_path},
            "stage": self.stage,
            "snapshot_layers": sorted(self.snapshot_layers),
            "agent": self.agent,
            "model": self.model,
            "sandbox": self.sandbox,
            "parent": {
                "reward": self.parent_reward,
                "error": self.parent_error,
                "run_dir": self.parent_run_dir,
            },
            "value": self.value,
            "error": self.error,
            "arms": [arm.to_dict() for arm in self.arms],
        }


def write_ablation_report(report: AblationReport, out_dir: Path) -> Path:
    """Write ``<out_dir>/ablation.json`` and return its path.

    Deterministic like every other branch artifact: sorted keys, indented,
    trailing newline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


# Attribution


def _score(reward: float) -> str:
    return f"{reward:.2f}"


def _reference_for(
    arm: ArmOutcome, by_name: dict[str, ArmOutcome], parent_reward: float | None
) -> tuple[str | None, float | None]:
    """The single arm (or the parent) this arm's verdict compares against.

    The two skills arms are each other's counterpart — that pair *is* the
    ablation. Any other arm compares against the parent's own linear reward,
    the only other observation the run produced; when neither exists the
    verdict states the reward and claims nothing.
    """
    counterpart = {
        SKILL_MODE_WITH_SKILL: SKILL_MODE_NO_SKILL,
        SKILL_MODE_NO_SKILL: SKILL_MODE_WITH_SKILL,
    }.get(arm.name)
    if counterpart is not None:
        other = by_name.get(counterpart)
        if other is not None and other.reward is not None:
            return counterpart, other.reward
    if parent_reward is not None:
        return REFERENCE_PARENT, parent_reward
    return None, None


def attribute(
    outcomes: Sequence[ArmOutcome], *, parent_reward: float | None, stage: str
) -> None:
    """Fill in each arm's reference and one-line verdict, in place.

    The verdict is an observation, not a causal claim: it names the two rewards
    it compares, the stage they were forked from, and the fact that it rests on
    one run per arm. Nothing is inferred about boundaries that were not forked
    — localizing a failure to a stage takes a second ablation at a second
    boundary (RFC §5, T3).
    """
    by_name = {outcome.name: outcome for outcome in outcomes}
    for arm in outcomes:
        reward = arm.reward
        if arm.status == STATUS_ERROR:
            arm.verdict = "errored before scoring — no reward to attribute"
            continue
        if reward is None:
            arm.verdict = "not run — an earlier arm errored"
            continue
        ref_name, ref_reward = _reference_for(arm, by_name, parent_reward)
        arm.reference = ref_name
        if ref_name is None or ref_reward is None:
            arm.verdict = (
                f"scores {_score(reward)} at {stage} — no counterpart arm or "
                "parent reward to compare against"
            )
        elif reward == ref_reward:
            arm.verdict = (
                f"matches {ref_name} at {stage} (both {_score(reward)}) — no "
                "difference in this comparison"
            )
        elif (reward >= PASS_REWARD) != (ref_reward >= PASS_REWARD):
            passes = reward >= PASS_REWARD
            arm.verdict = (
                f"{'passes' if passes else 'fails'} ({_score(reward)}) where "
                f"{ref_name} {'fails' if passes else 'passes'} "
                f"({_score(ref_reward)}) at {stage} — this delta decides the "
                f"outcome when applied at {stage} (1 run per arm)"
            )
        else:
            direction = "higher" if reward > ref_reward else "lower"
            arm.verdict = (
                f"scores {_score(reward)} vs {ref_name} {_score(ref_reward)} at "
                f"{stage} — {direction} reward in this single comparison"
            )


# Execution


def _branch_children(tree: RolloutTree) -> list[RolloutNode]:
    """The branch children of this run, in fork order.

    A branch child is the only node the engine records a ``delta`` on (at fork
    time, RFC §3.4), and ``RolloutTree.nodes()`` yields pre-order from the root
    — so this is fork order without reading engine private state or guessing at
    positions.
    """
    return [node for node in tree.nodes() if "delta" in node.state]


def _outcomes(
    arms: Sequence[AblationArm],
    children: Sequence[RolloutNode],
    *,
    run_dir: Path | None,
    branch_error: str | None,
) -> list[ArmOutcome]:
    """Pair each arm with the child node the engine forked for it.

    Children are created one at a time and each runs before the next is
    attached, so a failing arm leaves a node with no recorded reward and the
    arms after it leave no node at all: the arm that raised carries the branch
    error, the rest report as skipped.

    An arm whose child ran but was never scored (the engine recorded
    :data:`~benchflow.branch.UNSCORED_KEY` on the node) is an arm *error*
    carrying the engine's reason — never a reward. A missing score is not an
    observation, and an ablation that reports one as ``0.00`` invents its own
    evidence.
    """
    from benchflow.branch import UNSCORED_KEY
    from benchflow.branch_lineage import branch_child_dir

    outcomes: list[ArmOutcome] = []
    for index, arm in enumerate(arms):
        node = children[index] if index < len(children) else None
        outcome = ArmOutcome(
            name=arm.name,
            kind=arm.kind,
            delta=arm.delta.provenance_dict(),
            source=arm.source,
        )
        if node is not None:
            outcome.node_id = node.id
            outcome.delta_execution = node.state.get("delta_execution")
            outcome.wall_clock_sec = node.state.get(CHILD_WALL_CLOCK_KEY)
            if run_dir is not None and node.parent is not None:
                outcome.artifacts = str(
                    branch_child_dir(run_dir, node.parent.id, node.id)
                )
            reward = node.state.get("reward")
            if reward is None:
                outcome.error = (
                    node.state.get(UNSCORED_KEY)
                    or branch_error
                    or "the branch ended without a reward"
                )
            else:
                outcome.reward = float(reward)
        outcomes.append(outcome)
    return outcomes


async def _run_parent(rollout: Any, stage: str) -> tuple[float | None, str | None]:
    """Drive the parent past ``stage`` and score it; return ``(reward, error)``.

    Not ``Rollout.run()``: that cleans the sandbox up on the way out, and a
    torn-down sandbox has nothing left to branch. The phases below are the
    linear lifecycle in order, with the boundary captured by the rollout's own
    ``snapshot_stages`` policy as it passes.

    A failure *after* the boundary is recorded, not raised: the snapshot the
    arms fork from was already taken, and attributing a failed run is the
    reason this command exists (RFC §1).
    """
    from benchflow._utils.text import describe_exception

    try:
        await rollout.setup()
        await rollout.start()
    except Exception as exc:
        raise AblationRunError(
            f"the parent rollout failed before the {stage!r} boundary, so there "
            f"is nothing to branch: {describe_exception(exc)}"
        ) from exc
    try:
        await rollout.install_agent()
        await rollout.connect()
        await rollout.execute()
        rewards = await rollout.verify()
    except Exception as exc:
        error = describe_exception(exc)
        logger.warning(
            "ablation parent run failed after the %r boundary (%s) — the arms "
            "still fork from the recorded snapshot",
            stage,
            error,
        )
        return None, error
    if not rewards:
        return None, None
    reward = rewards.get("reward")
    return (None if reward is None else float(reward)), None


async def run_ablation(request: AblationRequest) -> AblationReport:
    """Run the task once, fork the requested stage into the arms, score them.

    The whole command in one call: validate (again — the library is the
    contract, the CLI one caller), run the parent to the stage boundary, fork
    it once into ``len(arms)`` children carrying the arms' deltas, and read the
    per-arm rewards back off the tree the engine grew. The parent's sandbox is
    always cleaned up, and a branch failure becomes reported arm errors rather
    than an exception — the arms that did run keep their rewards.
    """
    from benchflow._utils.text import describe_exception
    from benchflow.evaluation import effective_model
    from benchflow.rollout import Rollout, RolloutConfig

    stage = validate_arms_for_stage(
        request.arms, request.stage, snapshot_layers=request.snapshot_layers
    )
    if request.agent == "oracle":
        raise AblationSpecError(
            "bench eval ablate needs an ACP agent: every branch child connects "
            "an agent session over the restored snapshot, and the oracle path "
            "(solve.sh) has no session to fork"
        )
    task_path = Path(request.task_path)
    model = effective_model(request.agent, request.model)
    config = RolloutConfig.from_legacy(
        task_path=task_path,
        agent=request.agent,
        model=model,
        environment=request.sandbox,
        jobs_dir=Path(request.out_dir),
        job_name=PARENT_JOB_NAME,
        rollout_name=task_path.name,
        snapshot_stages={stage},
        snapshot_layers=request.snapshot_layers,
    )
    rollout = Rollout(config)
    report = AblationReport(
        task_id=task_path.name,
        task_path=str(task_path),
        stage=stage,
        snapshot_layers=sorted(request.snapshot_layers),
        agent=request.agent,
        model=model,
        sandbox=request.sandbox,
        arms=[],
    )
    branch_error: str | None = None
    try:
        report.parent_reward, report.parent_error = await _run_parent(rollout, stage)
        try:
            report.value = await rollout.branch_at_stage(
                stage, len(request.arms), deltas=[arm.delta for arm in request.arms]
            )
        except Exception as exc:
            branch_error = describe_exception(exc)
            logger.error("ablation branch at %r failed: %s", stage, branch_error)
    finally:
        await rollout.cleanup()

    run_dir = getattr(rollout, "_rollout_dir", None)
    report.parent_run_dir = None if run_dir is None else str(run_dir)
    report.arms = _outcomes(
        request.arms,
        _branch_children(rollout.tree),
        run_dir=run_dir,
        branch_error=branch_error,
    )
    if branch_error is not None and all(arm.error is None for arm in report.arms):
        # The branch failed before it forked anything (an uncaptured stage, a
        # capability gap): no arm can carry the error, so the report does.
        report.error = branch_error
    attribute(report.arms, parent_reward=report.parent_reward, stage=stage)
    try:
        materialized = rollout.result
    except Exception:
        logger.warning(
            "ablation parent result artifacts failed to build — the arms' "
            "rewards are unaffected",
            exc_info=True,
        )
    else:
        if materialized is None:
            logger.info("ablation parent reached no terminal result to materialize")
    return report

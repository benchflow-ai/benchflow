"""Branch reporting — the ablation report model and its attribution.

Everything that turns a finished (or half-finished) fork into something a
reader can hold: the report half of ``bench eval ablate`` that used to live
inside :mod:`benchflow.ablate` — the :class:`ArmOutcome` /
:class:`AblationReport` model, the pairing of arms with the child nodes the
engine forked for them (:func:`outcomes_for_arms`), the per-test mining from
each child's own verifier artifacts, and the attribution that turns rewards
into one-line observational verdicts (:func:`attribute`,
:func:`sub_test_attribution`). The fork's own lineage artifacts are adjacent
but deliberately elsewhere: the serializers live in
:mod:`benchflow.branch_lineage`, and the engine's failure-isolated lineage
write stays in :mod:`benchflow.rollout_branch`, whose module namespace is the
patch seam the lineage-isolation tests pin.

The split follows the data's direction: :mod:`benchflow.ablate` owns the
*request* side (driving the parent and the fork, with arm specs and
pre-flight validation in :mod:`benchflow.ablate_arms`); this module owns the
*result* side, and nothing here reaches back — ``ablate`` imports from
``branch_report``, never the reverse, and the engine-facing lineage writer
knows nothing about arms.

Determinism is the report's contract: arms keep request order, every delta is
the engine's own content-addressed provenance dict, test names sort, and no
wall-clock *timestamp* is recorded anywhere (per-arm ``wall_clock_sec`` is a
measured duration — part of an arm's cost, not a stamp).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.branch import UNSCORED_KEY
from benchflow.branch_transaction import CHILD_WALL_CLOCK_KEY
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchflow.ablate_arms import AblationArm
    from benchflow.environment.manifest import ManifestBinding
    from benchflow.trajectories.tree import RolloutNode, RolloutTree

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The report an ablation writes into its output directory.
REPORT_FILENAME = "ablation.json"

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


# The ablation report model


def environment_stamp(binding: ManifestBinding | None) -> dict[str, Any] | None:
    """The report's record of a bound environment, deterministic by design.

    Name, the ref exactly as the caller wrote it (flag value, arm spec, or
    ``task.md`` declaration — never a machine-local resolved path), the
    manifest's ``sha256`` content address, and the image(s) it names. This is
    the report-side answer to "which environment did this ablation compare
    arms in", readable without the registry.
    """
    if binding is None:
        return None
    manifest = binding.manifest
    return {
        "name": manifest.name,
        "ref": binding.ref,
        "env_hash": binding.env_hash,
        "image": manifest.image,
        "base_image": manifest.base_image,
    }


@dataclass
class ArmOutcome:
    """What one arm did: its reward, its cost, and its recorded delta.

    ``tests`` is the arm's per-test outcome map (``{test name -> status}``)
    mined from its own verifier artifacts, or ``None`` when that verifier
    reported none. ``None`` and ``{}`` are not the same thing here and the
    distinction is load-bearing: a missing map means *not observed*, and no
    sub-test claim may be made about that arm.

    ``environment`` is the :func:`environment_stamp` of the manifest this
    arm's delta swapped in — set only for an ``env:`` arm, absent for every
    arm that inherits the parent's bound environment (which the report stamps
    once at the top level).
    """

    name: str
    kind: str
    delta: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    reward: float | None = None
    wall_clock_sec: float | None = None
    delta_execution: str | None = None
    node_id: str | None = None
    artifacts: str | None = None
    tests: dict[str, str] | None = None
    error: str | None = None
    reference: str | None = None
    verdict: str = ""
    environment: dict[str, Any] | None = None

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
            "tests": None if self.tests is None else dict(sorted(self.tests.items())),
            "error": self.error,
            "reference": self.reference,
            "verdict": self.verdict,
            "environment": self.environment,
        }


@dataclass
class AblationReport:
    """The ablation's result: the parent's own run plus one row per arm.

    ``environment`` is the :func:`environment_stamp` of the world the parent
    (and therefore every arm that carries no environment delta) ran in —
    ``None`` when no manifest was bound. ``stage_snapshot`` is the branched
    stage's recorded snapshot refs (the committed sandbox image ref and the
    environment snapshot id, plus the layers captured), the handles a reader
    needs to restore that world and re-branch it by hand later — annotated by
    :func:`~benchflow.ablate.retain_stage_snapshot` with their lifetime:
    ``ephemeral: true`` with ``exported: null`` when cleanup destroyed the
    image (the default), or ``ephemeral: false`` with the exported tar's path
    and sha256 under ``--keep-snapshots``.
    """

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
    environment: dict[str, Any] | None = None
    stage_snapshot: dict[str, Any] | None = None

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
        delta is the engine's own content-addressed provenance dict, test names
        sort, and no wall-clock *timestamp* is recorded anywhere. Per-arm
        ``wall_clock_sec`` is a measured duration — the one field that varies
        between identical runs, carried because an ablation's cost per arm is
        part of its result.

        ``test_attribution`` is derived, not stored: it is a reading of the
        arms' own ``tests`` maps, so the section can never disagree with the
        rows it summarizes.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "task": {"id": self.task_id, "path": self.task_path},
            "stage": self.stage,
            "snapshot_layers": sorted(self.snapshot_layers),
            "stage_snapshot": self.stage_snapshot,
            "agent": self.agent,
            "model": self.model,
            "sandbox": self.sandbox,
            "environment": self.environment,
            "parent": {
                "reward": self.parent_reward,
                "error": self.parent_error,
                "run_dir": self.parent_run_dir,
            },
            "value": self.value,
            "error": self.error,
            "arms": [arm.to_dict() for arm in self.arms],
            "test_attribution": sub_test_attribution(self.arms),
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


# Reading the fork back: arms -> child nodes -> outcomes


def branch_children_of(tree: RolloutTree) -> list[RolloutNode]:
    """The branch children of this run, in fork order.

    A branch child is the only node the engine records a ``delta`` on (at fork
    time, RFC §3.4), and ``RolloutTree.nodes()`` yields pre-order from the root
    — so this is fork order without reading engine private state or guessing at
    positions.
    """
    return [node for node in tree.nodes() if "delta" in node.state]


def _child_test_outcomes(child_dir: Path) -> dict[str, str] | None:
    """One branch child's per-test outcomes, whichever way the engine ran it.

    The child kinds keep their verifier output in different places — a
    fresh-rollout child in its own run directory, an in-place child in the
    ``mounted/`` archive of what it wrote to the parent's shared bind mounts —
    and :func:`~benchflow.branch_artifacts.child_artifact_roots` is the one
    place that knows both. Each candidate is read with the CLI's existing CTRF
    reader (one parser, no branch-specific copy), and the first root that
    reports outcomes wins; ``None`` when neither does, which is a real
    observation ("this arm's verifier emitted no per-test data") and keeps the
    arm out of every sub-test comparison.

    Without the ``mounted/`` fallback an in-place ablation (``--at-stage
    pre-verify`` / ``post-verify``) fell back to scalar-only attribution while
    the per-test data sat on disk one directory down — the exact difference
    the sub-test section exists to expose.
    """
    from benchflow.branch_artifacts import child_artifact_roots
    from benchflow.cli._failure_evidence import artifact_test_outcomes

    for root in child_artifact_roots(child_dir):
        outcomes = artifact_test_outcomes(root)
        if outcomes is not None:
            return outcomes
    return None


def outcomes_for_arms(
    arms: Sequence[AblationArm],
    children: Sequence[RolloutNode],
    *,
    run_dir: Path | None,
    branch_error: str | None,
    environment_stamps: dict[str, dict[str, Any]] | None = None,
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

    Each child's per-test outcomes are mined from its own verifier artifacts
    here, where the child directory is known — see :func:`_child_test_outcomes`
    for the two places a child's verifier output can live. Report-time reads
    only: the engine stays file-free, and a child whose verifier emitted no
    CTRF report keeps ``tests = None``.
    """
    from benchflow.branch_lineage import branch_child_dir

    outcomes: list[ArmOutcome] = []
    for index, arm in enumerate(arms):
        node = children[index] if index < len(children) else None
        outcome = ArmOutcome(
            name=arm.name,
            kind=arm.kind,
            delta=arm.delta.provenance_dict(),
            source=arm.source,
            # The swapped-in environment an env: arm declares — request-level
            # provenance, stamped whether or not the arm got to run.
            environment=(environment_stamps or {}).get(arm.name),
        )
        if node is not None:
            outcome.node_id = node.id
            outcome.delta_execution = node.state.get("delta_execution")
            outcome.wall_clock_sec = node.state.get(CHILD_WALL_CLOCK_KEY)
            if run_dir is not None and node.parent is not None:
                child_dir = branch_child_dir(run_dir, node.parent.id, node.id)
                outcome.artifacts = str(child_dir)
                outcome.tests = _child_test_outcomes(child_dir)
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


# Attribution


#: How many differing test names a one-line verdict spells out before it rolls
#: the rest up as ``(+N more)`` — a verdict is a line, not a list.
_VERDICT_TEST_NAMES = 3


def _score(reward: float) -> str:
    return f"{reward:.2f}"


def _tested(arms: Sequence[ArmOutcome]) -> list[ArmOutcome]:
    """The arms whose verifier actually reported per-test outcomes.

    An arm with ``tests is None`` was not observed at this granularity, so it
    is excluded from every sub-test comparison rather than being treated as an
    arm where nothing ran — otherwise one CTRF-less arm would make every test
    of its counterpart look like a difference.
    """
    return [arm for arm in arms if arm.tests is not None]


def _name_list(names: Sequence[str]) -> str:
    """Up to :data:`_VERDICT_TEST_NAMES` names, the rest rolled up as a count."""
    shown = ", ".join(names[:_VERDICT_TEST_NAMES])
    extra = len(names) - _VERDICT_TEST_NAMES
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def differing_tests(arms: Sequence[ArmOutcome]) -> list[dict[str, Any]]:
    """The tests whose outcome is not identical across the arms that reported
    one, sorted by test name.

    This is the sub-outcome an ablation's scalar reward can hide: two arms can
    both score 0.00 while one passes ``test_a`` and fails ``test_b`` and the
    other does the reverse. Each entry is ``{"test": name, "outcomes": {arm ->
    status}}``, with ``None`` for an arm whose report does not name that test
    at all — a real difference (the test was reported for one arm and not the
    other), stated as a missing observation rather than invented as a failure.

    Comparison needs at least two observed arms; with fewer there is nothing to
    differ *from* and the result is empty.
    """
    tested = _tested(arms)
    if len(tested) < 2:
        return []
    differences: list[dict[str, Any]] = []
    for name in sorted({test for arm in tested for test in arm.tests or {}}):
        outcomes = {arm.name: (arm.tests or {}).get(name) for arm in tested}
        if len(set(outcomes.values())) > 1:
            differences.append(
                {"test": name, "outcomes": dict(sorted(outcomes.items()))}
            )
    return differences


def _scalar_tie(arms: Sequence[ArmOutcome]) -> bool:
    """Whether every arm that produced a reward produced the *same* reward.

    The condition under which the per-arm verdicts read "no difference" — and
    therefore the condition the per-test section exists to qualify.
    """
    rewards = [arm.reward for arm in arms if arm.reward is not None]
    return len(rewards) >= 2 and len(set(rewards)) == 1


def sub_test_attribution(arms: Sequence[ArmOutcome]) -> dict[str, Any]:
    """The report's ``test_attribution`` section: which sub-outcomes differ.

    Always present, and always honest about its own coverage: it names the
    arms that reported per-test outcomes *and* the arms that did not, so a
    reader can tell "the tests tie" from "no test data was mined". Only the
    differing tests are listed with their per-arm outcomes; tying tests are
    noise for attribution and appear as names only. Deterministic — sorted
    names throughout, no wall-clock anywhere.
    """
    tested = _tested(arms)
    differing = differing_tests(arms)
    differing_names = [entry["test"] for entry in differing]
    differing_set = set(differing_names)
    tying = [
        name
        for name in sorted({test for arm in tested for test in arm.tests or {}})
        if name not in differing_set
    ]
    scalar_tie = _scalar_tie(arms)
    if len(tested) < 2:
        summary = (
            f"scalar-only attribution — {len(tested)} of {len(arms)} arms "
            "reported per-test outcomes, so no sub-test comparison was made"
        )
    elif not differing:
        summary = (
            f"{len(tested)} arms reported per-test outcomes and all "
            f"{len(tying)} tie — no sub-test difference in this comparison"
        )
    elif scalar_tie:
        summary = (
            f"scalar rewards tie, but {len(differing_names)} sub-test "
            f"outcome(s) differ: {_name_list(differing_names)}"
        )
    else:
        summary = (
            f"{len(differing_names)} sub-test outcome(s) differ: "
            f"{_name_list(differing_names)}"
        )
    return {
        "arms_with_tests": [arm.name for arm in tested],
        "arms_without_tests": [arm.name for arm in arms if arm.tests is None],
        "differing_tests": differing,
        "tying_tests": tying,
        "scalar_tie": scalar_tie,
        "summary": summary,
    }


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

    One qualification the scalar cannot make on its own: when two arms tie on
    reward but their verifiers disagree per test, the verdict says so and names
    the tests instead of reading "no difference in this comparison". A binary
    reward can net two opposite-signed sub-outcomes to exactly zero, and a tool
    that printed "no difference" there would be true about the reward and false
    about the behavior. The unqualified wording survives only where the
    sub-tests were observed and tie, or were never observed at all.
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
            other = by_name.get(ref_name)
            differing = (
                [entry["test"] for entry in differing_tests([arm, other])]
                if other is not None
                else []
            )
            tail = (
                f"scalar rewards tie, but {len(differing)} sub-test outcome(s) "
                f"differ: {_name_list(differing)}"
                if differing
                else "no difference in this comparison"
            )
            arm.verdict = (
                f"matches {ref_name} at {stage} (both {_score(reward)}) — {tail}"
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

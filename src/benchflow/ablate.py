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

Attribution runs at two granularities, because a binary reward is a lossy
summary of what an arm did: the scalar comparison, and the per-test outcomes
mined from each arm's own verifier report (:func:`differing_tests`,
:func:`sub_test_attribution`). A measured skills ablation that scored 0.00/0.00
had *both* its sub-tests flip in opposite directions — attributing on the
scalar alone would have reported "no difference" about a large, reproducible
behavioral one.

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
    from benchflow.environment.manifest import EnvironmentManifest
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

#: Arm spec prefix for a config-override arm: ``config:<inline-or-@file>`` —
#: the same dual "value or ref" form as the run-level ``--config-override``.
CONFIG_PREFIX = "config:"

ARM_KIND_SKILL_MODE = "skill-mode"
ARM_KIND_INJECT = "inject"
ARM_KIND_CONFIG = "config-override"

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
    f"'{INJECT_PREFIX}<path-to-file>', and "
    f"'{CONFIG_PREFIX}<inline-json-or-@file>'"
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

    Four kinds, each mapping onto exactly one executable
    :class:`BranchDelta` field: ``with-skill`` / ``no-skill`` become
    ``skill_mode`` (the child re-runs installation as a fresh rollout from the
    ``env-ready`` snapshot), ``inject:<path>`` reads the file and becomes
    ``injected_prompt`` (the child's user-visible continuation prompt) — which
    at ``--at-stage env-ready`` is *also* delivered by a fresh rollout, since
    every child of that boundary installs the agent for itself — and
    ``config:<inline-or-@file>`` parses through the run-level overlay loader
    and becomes ``config_override`` (the child runs fresh under the parent's
    config with the allowlisted patch deep-merged on top, #790). An
    unknown kind, an empty spec, an injection file that is missing or blank,
    or a config patch that is unparsable or touches a non-allowlisted section
    raises :class:`AblationSpecError` — a silently dropped arm would publish an
    ablation table with a missing comparison, and a scorer-touching patch must
    die here, before the parent run costs anything.
    """
    name = spec.strip()
    if not name:
        raise AblationSpecError(f"empty arm in --arms — {_ARM_SPEC_HELP}")
    if name in _SKILL_ARM_NAMES:
        return AblationArm(
            name=name, kind=ARM_KIND_SKILL_MODE, delta=BranchDelta(skill_mode=name)
        )
    if name.startswith(CONFIG_PREFIX):
        return _parse_config_arm(name)
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


def _parse_config_arm(name: str) -> AblationArm:
    """Parse a ``config:<inline-or-@file>`` arm through the #790 loader.

    The value after the prefix is exactly what ``--config-override`` accepts —
    inline JSON/YAML/TOML or an ``@file`` ref — parsed by the same
    :func:`~benchflow._utils.config_override.load_config_override` so the two
    surfaces cannot drift. The allowlist runs *here*, at parse time: an arm
    that patches the scorer must fail before the parent run costs anything,
    exactly as the branch engine would fail it at fork time.
    """
    from benchflow._utils.config_override import load_config_override, validate_overlay

    raw = name[len(CONFIG_PREFIX) :].strip()
    if not raw:
        raise AblationSpecError(
            f"arm {name!r} carries no patch — the config arm is "
            f"'{CONFIG_PREFIX}<inline-json-or-@file>' (e.g. "
            f"""'{CONFIG_PREFIX}{{"agent": {{"timeout_sec": 60}}}}' or """
            f"'{CONFIG_PREFIX}@overlay.yaml')"
        )
    try:
        overlay = load_config_override(raw)
    except (ValueError, OSError) as exc:
        raise AblationSpecError(
            f"arm {name!r} cannot load its config patch: {exc}"
        ) from None
    if not overlay:
        raise AblationSpecError(
            f"arm {name!r} parses to an empty patch — a config arm must "
            "change at least one allowlisted section"
        )
    try:
        validate_overlay(overlay)
    except ValueError as exc:
        raise AblationSpecError(f"arm {name!r} is not executable: {exc}") from None
    return AblationArm(
        name=name,
        kind=ARM_KIND_CONFIG,
        delta=BranchDelta(config_override=overlay),
        source=raw[1:] if raw.startswith("@") else None,
    )


def _split_arm_specs(spec: str) -> list[str]:
    """Split ``--arms`` on commas, ignoring commas nested in JSON braces.

    A ``config:`` arm may carry inline JSON (``config:{"agent": {"a": 1,
    "b": 2}}``) whose commas are content, not separators. Depth counting over
    ``{}``/``[]`` keeps every historical spec splitting exactly as before —
    no other arm kind can contain a brace.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in spec:
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


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
    arms = [parse_arm(entry) for entry in _split_arm_specs(spec)]
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

    Pre-flight gates, each mirroring a rule that lives elsewhere:

    * ``post-research`` is a mid-``execute()`` cut point only an explicit
      ``Rollout.mark_stage()`` can record (RFC §3.2). This command drives the
      lifecycle, not the agent's planning, so it cannot mark it — saying so up
      front beats running the whole task and dying at the fork.
    * a ``skill_mode`` arm executes only at ``env-ready``, because skills are
      deployed by ``install_agent()`` (RFC §3.3). The branch engine fails
      closed on this too; here it costs nothing instead of a full run.
    * a ``config:`` arm executes only at ``env-ready`` for the same shape of
      reason: the overlay is applied by the child's own ``setup()``, which
      only a fresh child of that boundary re-runs.
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
        if arm.kind == ARM_KIND_CONFIG and stage != FRESH_CHILD_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {FRESH_CHILD_STAGE!r}, not "
                f"{stage!r}: the config patch is applied by the child's own "
                f"setup(), and by {stage!r} the parent's config has already "
                "been consumed — the arm would record the override and run "
                "without it"
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


def resolve_ablation_environment_manifest(
    task_path: Path,
) -> EnvironmentManifest | None:
    """The environment the task declares — the same one ``bench eval`` binds.

    A stateful task declares its world in ``task.md``
    (``benchflow.environment.manifest``: the image, the services the framework
    starts, the readiness probes). ``bench eval`` resolves that declaration per
    task before building the rollout config
    (:func:`~benchflow.environment.manifest.manifest_from_task_document`); an
    ablation that skipped it would run the parent — and therefore every arm
    forked from it — in a *different* environment than the run it is meant to
    explain, and would report the comparison as if it were the same one.

    Resolution failures are fatal here rather than degrading to ``None``: an
    ablation whose declared environment could not be built is not an ablation
    that ran without services, and it fails before the parent run costs
    anything.
    """
    from benchflow._utils.text import describe_exception
    from benchflow.environment.manifest import manifest_from_task_document

    try:
        return manifest_from_task_document(task_path)
    except Exception as exc:
        raise AblationSpecError(
            f"{task_path.name} declares an environment manifest in its task.md "
            f"that could not be resolved: {describe_exception(exc)}. Every arm "
            "forks the parent's environment, so this ablation would compare "
            "arms in a world the task says is the wrong one"
        ) from exc


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
    """What one arm did: its reward, its cost, and its recorded delta.

    ``tests`` is the arm's per-test outcome map (``{test name -> status}``)
    mined from its own verifier artifacts, or ``None`` when that verifier
    reported none. ``None`` and ``{}`` are not the same thing here and the
    distinction is load-bearing: a missing map means *not observed*, and no
    sub-test claim may be made about that arm.
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


# Execution


def _branch_children(tree: RolloutTree) -> list[RolloutNode]:
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

    Each child's per-test outcomes are mined from its own verifier artifacts
    here, where the child directory is known — see :func:`_child_test_outcomes`
    for the two places a child's verifier output can live. Report-time reads
    only: the engine stays file-free, and a child whose verifier emitted no
    CTRF report keeps ``tests = None``.
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
    environment_manifest = resolve_ablation_environment_manifest(task_path)
    config = RolloutConfig.from_legacy(
        task_path=task_path,
        agent=request.agent,
        model=model,
        environment=request.sandbox,
        # The task's own declared world, resolved exactly as a normal
        # evaluation resolves it. Every arm forks the parent's snapshot and
        # (at ``env-ready``) re-runs from the parent's config, so binding it
        # here binds it for the whole experiment — see
        # :func:`resolve_ablation_environment_manifest`.
        environment_manifest=environment_manifest,
        jobs_dir=Path(request.out_dir),
        job_name=PARENT_JOB_NAME,
        rollout_name=task_path.name,
        # Stated, not defaulted: the arms fork the parent's own ``env-ready``
        # image, and a with-skill parent bakes its pack into that image — a
        # ``no-skill`` arm would restore the pack and still be labelled
        # no-skill. The branch engine refuses that fork; pinning the parent
        # here is what keeps every ablation on the side of the gate that runs.
        skill_mode=SKILL_MODE_NO_SKILL,
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

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

import asyncio
import logging
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.branch_delta import BranchDelta
from benchflow.branch_report import (  # noqa: F401  (report API re-exported here)
    PASS_REWARD,
    REFERENCE_PARENT,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    AblationReport,
    ArmOutcome,
    attribute,
    branch_children_of,
    differing_tests,
    environment_stamp,
    outcomes_for_arms,
    sub_test_attribution,
    write_ablation_report,
)
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
    STAGE_POST_RESEARCH,
    validate_stage,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL

if TYPE_CHECKING:
    from benchflow.environment.manifest import ManifestBinding

logger = logging.getLogger(__name__)

#: The parent rollout's job name under ``--out-dir`` — fixed, not stamped, so
#: the run directory of an ablation is derivable from its output directory.
PARENT_JOB_NAME = "ablation"

#: Arm spec prefix for a plan-injection arm: ``inject:<path-to-file>``.
INJECT_PREFIX = "inject:"

#: Arm spec prefix for a config-override arm: ``config:<inline-or-@file>`` —
#: the same dual "value or ref" form as the run-level ``--config-override``.
CONFIG_PREFIX = "config:"

#: Arm spec prefix for an environment arm: ``env:<registry-ref-or-path>``
#: (the ``env0@prod`` vs ``env0@outage`` tool-outage pattern).
ENV_PREFIX = "env:"

ARM_KIND_SKILL_MODE = "skill-mode"
ARM_KIND_INJECT = "inject"
ARM_KIND_CONFIG = "config-override"
ARM_KIND_ENV = "environment-ref"

#: The stages this command can capture on its own — everything the lifecycle
#: reaches without an explicit ``Rollout.mark_stage()`` call.
CAPTURABLE_STAGES: tuple[str, ...] = tuple(
    stage for stage in BRANCH_STAGES if stage not in MARKED_STAGES
)

_SKILL_ARM_NAMES = (SKILL_MODE_WITH_SKILL, SKILL_MODE_NO_SKILL)
_ARM_SPEC_HELP = (
    f"supported arms are {SKILL_MODE_WITH_SKILL!r}, {SKILL_MODE_NO_SKILL!r}, "
    f"'{INJECT_PREFIX}<path-to-file>', "
    f"'{CONFIG_PREFIX}<inline-json-or-@file>', and "
    f"'{ENV_PREFIX}<registry-ref>'"
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

    Five kinds, each mapping onto exactly one executable
    :class:`BranchDelta` field: ``with-skill`` / ``no-skill`` become
    ``skill_mode`` (the child re-runs installation as a fresh rollout from the
    ``env-ready`` snapshot), ``inject:<path>`` reads the file and becomes
    ``injected_prompt`` (the child's user-visible continuation prompt) — which
    at ``--at-stage env-ready`` is *also* delivered by a fresh rollout, since
    every child of that boundary installs the agent for itself —
    ``config:<inline-or-@file>`` parses through the run-level overlay loader
    and becomes ``config_override`` (the child runs fresh under the parent's
    config with the allowlisted patch deep-merged on top, #790), and
    ``env:<registry-ref>`` becomes ``environment_ref`` (the service-level
    environment swap; resolution against the parent's manifest happens in
    :func:`run_ablation`, where the parent's manifest is known). An
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
    if name.startswith(ENV_PREFIX):
        ref = name[len(ENV_PREFIX) :].strip()
        if not ref:
            raise AblationSpecError(
                f"arm {name!r} names no environment — the environment arm is "
                f"'{ENV_PREFIX}<registry-ref>' (e.g. '{ENV_PREFIX}env0@outage') "
                "or a manifest file path"
            )
        return AblationArm(
            name=name, kind=ARM_KIND_ENV, delta=BranchDelta(environment_ref=ref)
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
    no other arm kind can contain a brace. Inside a JSON string literal,
    braces, brackets and commas are content too, so the walk tracks quote
    state (with ``\\``-escape handling) and ignores structure until the
    string closes. Commas in a ``config:@<path>`` file path remain
    unrepresentable — that grammar limit is documented on ``--arms``.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    for char in spec:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            current.append(char)
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char in "{[":
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
    research_end_marker: str | None = None,
) -> str:
    """Reject a stage/arm combination before the parent rollout runs.

    Pre-flight gates, each mirroring a rule that lives elsewhere:

    * ``post-research`` is a mid-``execute()`` cut point only an explicit
      ``Rollout.mark_stage()`` can record (RFC §3.2). This command drives the
      lifecycle, not the agent's planning, so on its own it cannot mark it —
      ``--mark-research-end-on <workspace-path>`` supplies the concrete
      trigger (the engine marks the stage the first time that file exists in
      the workspace; see :func:`watch_research_end`), and without it the
      rejection says so up front, which beats running the whole task and
      dying at the fork. The marker is meaningless for any other stage and
      fails closed there.
    * a ``skill_mode`` arm executes only at ``env-ready``, because skills are
      deployed by ``install_agent()`` (RFC §3.3). The branch engine fails
      closed on this too; here it costs nothing instead of a full run.
    * a ``config:`` arm executes only at ``env-ready`` for the same shape of
      reason: the overlay is applied by the child's own ``setup()``, which
      only a fresh child of that boundary re-runs.
    * an ``env:`` arm executes only at ``env-ready`` likewise: the child
      manifest's services are provisioned over the restored snapshot by the
      fresh child. (Its *content* gates — resolvable ref, same image,
      framework-started services — need the parent's manifest and run in
      :func:`run_ablation`, still before the parent run.)
    * an ``env-ready`` ablation runs *every* arm as a fresh rollout — that
      boundary precedes ``install_agent()``, so each child installs the agent
      for itself — which needs the container layer in the stage snapshot. The
      engine raises :class:`~benchflow.rollout_branch.BranchChildExecutionNotSupported`
      for this; the mirror here is checked only when the caller passes the
      layers it will request.

    Returns the validated stage, so callers can use this as their one gate.
    """
    validate_stage(stage, field="--at-stage")
    if research_end_marker is not None and stage != STAGE_POST_RESEARCH:
        raise AblationSpecError(
            f"--mark-research-end-on only applies to --at-stage "
            f"{STAGE_POST_RESEARCH!r}, not {stage!r}: the marker file's first "
            "appearance is what defines the research-end boundary, and no "
            "other stage is captured from it"
        )
    if stage in MARKED_STAGES and research_end_marker is None:
        raise AblationSpecError(
            f"--at-stage {stage!r} cannot be captured by this command without "
            "a research-end trigger: it is a mid-execute() cut point only an "
            "explicit Rollout.mark_stage() call can record. Pass "
            "--mark-research-end-on <workspace-path> (e.g. /app/PLAN.md) so "
            "the engine marks the stage the first time that file appears "
            "during the agent's run; or drive the run through the SDK — "
            f"await rollout.mark_stage({stage!r}) at the cut point during the "
            f"agent's run, then await rollout.branch_at_stage({stage!r}, n, "
            "deltas=[...]) — or ablate one of "
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
        if arm.kind == ARM_KIND_ENV and stage != FRESH_CHILD_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {FRESH_CHILD_STAGE!r}, not "
                f"{stage!r}: the child manifest's services are provisioned "
                "over the restored snapshot by the fresh child rollout, and "
                f"by {stage!r} the parent's provisioned services survive the "
                "fork — the arm would record the swap and run without it"
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


def resolve_ablation_environment_binding(
    task_path: Path, *, explicit: Path | str | None = None
) -> ManifestBinding | None:
    """The environment the ablation binds — flag first, then the task's own.

    ``explicit`` is the ``--environment-manifest`` value (a manifest path or a
    ``name@version`` registry spec): when given it wins outright and the
    task-declared manifest is not even resolved — the same precedence ``bench
    eval run`` applies (an explicit run-level manifest suppresses
    ``manifest_from_task_document``). Otherwise the task's own ``task.md``
    declaration is resolved
    (:func:`~benchflow.environment.manifest.manifest_binding_from_task_document`),
    exactly as a normal evaluation resolves it; a stateful task ablated
    without it would run the parent — and therefore every arm forked from it —
    in a *different* environment than the run it is meant to explain.

    The returned :class:`~benchflow.environment.manifest.ManifestBinding`
    keeps the ref verbatim and the manifest's content address, which is what
    :func:`environment_stamp` writes into ``ablation.json``. Resolution
    failures are fatal rather than degrading to ``None``: an ablation whose
    declared environment could not be built is not an ablation that ran
    without services, and it fails before the parent run costs anything.
    """
    from benchflow._utils.text import describe_exception
    from benchflow.environment.manifest import (
        load_manifest_binding,
        manifest_binding_from_task_document,
    )

    if explicit is not None:
        try:
            return load_manifest_binding(explicit)
        except Exception as exc:
            raise AblationSpecError(
                f"--environment-manifest {str(explicit)!r} does not resolve to "
                f"an environment manifest: {describe_exception(exc)}"
            ) from exc
    try:
        return manifest_binding_from_task_document(task_path)
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
    # Agent reasoning/thinking effort (``--reasoning-effort``) — the same
    # normalized control ``bench eval run`` resolves, threaded through the
    # canonical plan so parent and child configs record the effort the arms
    # actually ran under. ``None`` = the agent's own default.
    reasoning_effort: str | None = None
    sandbox: str = "docker"
    out_dir: Path = Path("jobs")
    # Explicit environment binding (``--environment-manifest``): a manifest
    # path or ``name@version`` registry spec. Beats the task-declared
    # manifest — the same precedence as ``bench eval run``. ``None`` = bind
    # whatever the task declares (or nothing).
    environment_manifest: Path | str | None = None
    # The research-end trigger (``--mark-research-end-on``): a workspace path
    # (absolute, or relative to the agent's cwd) whose first appearance marks
    # ``post-research`` during the parent's run — the FrontierPhysics
    # convention is the agent materializing its plan as ``/app/PLAN.md``.
    # Required for ``stage='post-research'``; meaningless (fail closed) for
    # any other stage. ``None`` = no trigger.
    mark_research_end_on: str | None = None
    # The layers the stage snapshot composes (RFC §3.1). The container layer is
    # mandatory for a skills arm and sufficient on its own; the environment
    # layer needs an Environment plane, which this command does not bind, so
    # requesting it by default would fail closed at capture time.
    snapshot_layers: frozenset[str] = frozenset({SKILL_DELTA_LAYER})
    # Durable snapshot retention (RFC §3.6): export the branched stage's
    # committed sandbox image (``docker save``) to
    # ``<out_dir>/snapshots/<ref>.tar`` before cleanup destroys it, and record
    # the tar's path + sha256 in the report. Without it the snapshot dies with
    # the rollout and the report marks its handle ephemeral instead.
    keep_snapshots: bool = False


# Execution


def resolve_canonical_parent_config(
    request: AblationRequest,
    *,
    stage: str,
    environment_manifest: Any = None,
) -> Any:
    """The parent's RolloutConfig: the canonical eval plan + the ablation axis.

    An ablation's parent is a *normal evaluation run* of the task — the arms
    fork its world, so any control the parent dropped is dropped for every arm
    too. The request is therefore resolved through the same two stages ``bench
    eval run`` uses: :func:`~benchflow.eval_plan.build_eval_plan` (normalized
    agent/model/effort/sandbox/usage settings, fail-closed validation) and
    :func:`~benchflow.evaluation.task_rollout_config` (task digest, dataset and
    source identity, prompts, the task-declared environment fallback). A
    hand-rolled reduced config here is the PR #1046 review finding: the real
    E2E parent and child configs published ``task_digest: null`` and
    ``reasoning_effort: null``.

    Overlaid on top — the only fields the ablation owns:

    * the stage-capture request (``snapshot_stages={stage}`` plus the request's
      layers), which is *why* this rollout exists;
    * ``skill_mode='no-skill'`` — stated, not defaulted: the arms fork the
      parent's own ``env-ready`` image, and a with-skill parent bakes its pack
      into that image, so a ``no-skill`` arm would restore the pack and still
      be labelled no-skill. The branch engine refuses that fork; pinning the
      parent here keeps every ablation on the side of the gate that runs;
    * out-dir / job naming (``<out_dir>/ablation/<task>``, so the run
      directory is derivable from the output directory);
    * the resolved environment binding (explicit flag beats the task's own
      declaration — resolved fail-closed by
      :func:`resolve_ablation_environment_binding` before this is called).

    Plan-validation failures re-raise as :class:`AblationSpecError`: they are
    request defects, decidable before the parent run costs anything.
    """
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
    from benchflow.evaluation import task_rollout_config

    task_path = Path(request.task_path)
    try:
        plan = build_eval_plan(
            EvalCreateRequest(
                tasks_dir=task_path,
                agent=request.agent,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
                environment=request.sandbox,
                jobs_dir=str(request.out_dir),
                # One parent rollout at a time — the truthful value for a
                # single-task experiment, not the batch default.
                concurrency=1,
            )
        )
    except EvalPlanError as exc:
        raise AblationSpecError(str(exc)) from exc
    return task_rollout_config(
        plan.make_eval_config(),
        task_path,
        job_name=PARENT_JOB_NAME,
        jobs_dir=plan.output_jobs_dir,
        rollout_name=task_path.name,
        environment_manifest=environment_manifest,
        skill_mode=SKILL_MODE_NO_SKILL,
        snapshot_stages={stage},
        snapshot_layers=request.snapshot_layers,
    )


#: How often the research-end watcher polls the workspace for the marker file.
RESEARCH_END_POLL_SEC = 2.0


def _resolve_marker_path(rollout: Any, marker: str) -> str:
    """An absolute in-sandbox path for the research-end marker file."""
    if marker.startswith("/"):
        return marker
    cwd = getattr(rollout, "_agent_cwd", None) or "/app"
    return f"{str(cwd).rstrip('/')}/{marker}"


async def _research_marker_exists(rollout: Any, path: str) -> bool:
    """One cheap ``test -e`` for the marker file in the parent's sandbox."""
    sandbox = getattr(rollout, "env", None)
    if sandbox is None:
        return False
    result = await sandbox.exec(f"test -e {shlex.quote(path)}", timeout_sec=10)
    return getattr(result, "return_code", 1) == 0


async def watch_research_end(
    rollout: Any, marker: str, *, poll_interval: float = RESEARCH_END_POLL_SEC
) -> bool:
    """Mark ``post-research`` the first time ``marker`` exists in the workspace.

    The concrete trigger behind ``--mark-research-end-on`` (RFC §3.2): a
    research-style agent materializes its plan as a workspace file (the
    FrontierPhysics convention is ``/app/PLAN.md``), so the file's first
    appearance *is* the research→execution boundary. The engine has no
    per-LLM-exchange hook from outside the agent process, so the check runs
    on a cheap wall-clock poll (one sandbox ``test -e`` per
    ``poll_interval``) concurrent with ``execute()``, plus a final check when
    the agent quiesces — **the capture therefore lands within one poll of the
    file appearing, and the snapshot may include up to that much post-plan
    agent work.** That cadence bound is the documented tradeoff of marking
    from outside the agent; the exchange index recorded with the mark
    (``capture_stage``) is exact for the moment the capture actually ran.

    Transient poll failures (a raced exec, teardown) keep polling; a
    ``mark_stage()`` failure propagates — a capture the run was told to take
    and could not is the same fail-closed rule the lifecycle's own boundaries
    apply. Returns True once the stage is marked; cancellation is the normal
    end when the agent finishes before the marker ever appears.
    """
    path = _resolve_marker_path(rollout, marker)
    while True:
        try:
            found = await _research_marker_exists(rollout, path)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("research-end marker poll failed", exc_info=True)
            found = False
        if found:
            await rollout.mark_stage(STAGE_POST_RESEARCH)
            return True
        await asyncio.sleep(poll_interval)


async def _execute_parent(rollout: Any, research_end_marker: str | None) -> None:
    """``execute()``, with the research-end watcher beside it when requested.

    The watcher runs as a sibling task for the duration of the agent's run and
    is cancelled when the agent quiesces; a marker that appeared between the
    watcher's last poll and quiescence is caught by one final check, so the
    trigger is decided by the file's presence, not by poll timing. A watcher
    failure (its ``mark_stage`` raising) is logged and leaves the stage
    uncaptured — the branch then fails closed on the missing stage rather
    than this masking the agent's own outcome.
    """
    if research_end_marker is None:
        await rollout.execute()
        return
    watcher = asyncio.create_task(watch_research_end(rollout, research_end_marker))
    marked = False
    watcher_failed = False
    try:
        await rollout.execute()
    finally:
        if not watcher.done():
            watcher.cancel()
        try:
            marked = await watcher
        except asyncio.CancelledError:
            marked = False
        except Exception:
            watcher_failed = True
            logger.warning(
                "research-end watcher failed — 'post-research' was not "
                "captured mid-run",
                exc_info=True,
            )
    if marked or watcher_failed:
        return
    if await _research_marker_exists(
        rollout, _resolve_marker_path(rollout, research_end_marker)
    ):
        await rollout.mark_stage(STAGE_POST_RESEARCH)


async def _run_parent(
    rollout: Any, stage: str, *, research_end_marker: str | None = None
) -> tuple[float | None, str | None]:
    """Drive the parent past ``stage`` and score it; return ``(reward, error)``.

    Not ``Rollout.run()``: that cleans the sandbox up on the way out, and a
    torn-down sandbox has nothing left to branch. The phases below are the
    linear lifecycle in order, with the boundary captured by the rollout's own
    ``snapshot_stages`` policy as it passes — except ``post-research``, which
    is marked by the research-end watcher (:func:`watch_research_end`) when
    ``research_end_marker`` names the trigger file.

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
        await _execute_parent(rollout, research_end_marker)
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


def _snapshot_tar_name(ref: str) -> str:
    """A filesystem-safe tar basename for a snapshot image ref."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-") or "snapshot"


def _file_sha256(path: Path) -> str:
    """Streaming ``sha256:``-prefixed digest of a file (tars can be large)."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


async def export_stage_snapshot(
    sandbox: Any, *, sandbox_ref: str, out_dir: Path
) -> dict[str, Any]:
    """``docker save`` the branched stage's sandbox image into ``out_dir``.

    The durable half of ``--keep-snapshots`` (RFC §3.6): the tar lands at
    ``<out_dir>/snapshots/<ref>.tar`` and the returned record carries its
    path and content sha256 — enough for a reader to verify and
    ``docker load`` the world later. Raises when the sandbox backend cannot
    export (the caller records the failure; the report stays truthful).
    """
    export = getattr(sandbox, "export_image", None)
    if export is None:
        backend = "no sandbox attached" if sandbox is None else type(sandbox).__name__
        raise AblationError(
            f"--keep-snapshots: the sandbox backend ({backend}) does not "
            "support exporting snapshot images (export_image); the docker "
            "backend does"
        )
    snapshots_dir = Path(out_dir) / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    tar_path = snapshots_dir / f"{_snapshot_tar_name(sandbox_ref)}.tar"
    await export(sandbox_ref, tar_path)
    return {"path": str(tar_path), "sha256": _file_sha256(tar_path)}


async def retain_stage_snapshot(
    report: AblationReport, *, sandbox: Any, keep_snapshots: bool, out_dir: Path
) -> None:
    """Make the report truthful about the stage snapshot's lifetime (RFC §3.6).

    Must run **before** ``rollout.cleanup()`` — cleanup's
    ``compose down --rmi all`` is what destroys the committed ``bf-snap-*``
    image, and a report serialized afterwards once published a handle
    ``docker image inspect`` could no longer resolve. With
    ``keep_snapshots`` the image is exported to a tar and the entry records
    ``ephemeral: false`` plus the tar's path and sha256; without it (or when
    the export fails — recorded as ``export_error``) the entry records
    ``ephemeral: true, exported: null`` so a reader knows the ref no longer
    resolves. Never raises: the arms' rewards must survive a failed export.
    """
    snap = report.stage_snapshot
    if snap is None:
        return
    snap["ephemeral"] = True
    snap["exported"] = None
    sandbox_ref = snap.get("sandbox_ref")
    if not keep_snapshots:
        return
    if sandbox_ref is None:
        snap["export_error"] = (
            "--keep-snapshots: the branched stage recorded no sandbox-layer "
            "image to export"
        )
        return
    try:
        exported = await export_stage_snapshot(
            sandbox, sandbox_ref=sandbox_ref, out_dir=out_dir
        )
    except Exception as exc:
        from benchflow._utils.text import describe_exception

        snap["export_error"] = describe_exception(exc)
        logger.error(
            "--keep-snapshots could not export %s — the report records the "
            "snapshot as ephemeral: %s",
            sandbox_ref,
            snap["export_error"],
        )
    else:
        snap["ephemeral"] = False
        snap["exported"] = exported


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
    from benchflow.rollout import Rollout

    stage = validate_arms_for_stage(
        request.arms,
        request.stage,
        snapshot_layers=request.snapshot_layers,
        research_end_marker=request.mark_research_end_on,
    )
    if request.agent == "oracle":
        raise AblationSpecError(
            "bench eval ablate needs an ACP agent: every branch child connects "
            "an agent session over the restored snapshot, and the oracle path "
            "(solve.sh) has no session to fork"
        )
    task_path = Path(request.task_path)
    environment_binding = resolve_ablation_environment_binding(
        task_path, explicit=request.environment_manifest
    )
    environment_manifest = (
        None if environment_binding is None else environment_binding.manifest
    )
    # Pre-flight the env arms' content gates now that the parent's manifest is
    # known: an unresolvable ref, an image-changing manifest, or an
    # entrypoint-owned lifecycle is decidable from the request alone, and the
    # branch engine would reject it only after a full parent run. An arm that
    # passes the gate is stamped with the environment it swaps in, so its
    # report row names the world it ran against.
    from benchflow.branch_skill import resolve_environment_ref_delta
    from benchflow.environment.manifest import load_manifest_binding

    environment_stamps: dict[str, dict[str, Any]] = {}
    for arm in request.arms:
        if arm.kind == ARM_KIND_ENV and arm.delta.environment_ref is not None:
            try:
                resolve_environment_ref_delta(
                    environment_manifest,
                    arm.delta.environment_ref,
                    subject=f"arm {arm.name!r}",
                )
            except NotImplementedError as exc:
                raise AblationSpecError(str(exc)) from exc
            stamp = environment_stamp(load_manifest_binding(arm.delta.environment_ref))
            if stamp is not None:
                environment_stamps[arm.name] = stamp
    # The canonical evaluation configuration with the ablation axis overlaid —
    # the bound world (an explicit ``--environment-manifest`` when given, else
    # the task's own declaration) rides along: every arm forks the parent's
    # snapshot and (at ``env-ready``) re-runs from the parent's config, so
    # binding it here binds it for the whole experiment.
    config = resolve_canonical_parent_config(
        request, stage=stage, environment_manifest=environment_manifest
    )
    rollout = Rollout(config)
    report = AblationReport(
        task_id=task_path.name,
        task_path=str(task_path),
        stage=stage,
        snapshot_layers=sorted(request.snapshot_layers),
        agent=config.agent,
        model=config.model,
        sandbox=request.sandbox,
        arms=[],
        environment=environment_stamp(environment_binding),
    )
    branch_error: str | None = None
    try:
        report.parent_reward, report.parent_error = await _run_parent(
            rollout, stage, research_end_marker=request.mark_research_end_on
        )
        if request.mark_research_end_on is not None and stage not in getattr(
            rollout, "_stage_snapshots", {}
        ):
            # The trigger never fired: branching would only raise
            # BranchStageNotCaptured, so say what actually happened — the
            # marker file, not the stage machinery, is what was missing.
            branch_error = (
                f"the research-end marker {request.mark_research_end_on!r} "
                "never appeared in the workspace during the parent run, so "
                f"{stage!r} was never captured and there is nothing to branch"
            )
            logger.error("ablation branch at %r failed: %s", stage, branch_error)
        else:
            try:
                report.value = await rollout.branch_at_stage(
                    stage,
                    len(request.arms),
                    deltas=[arm.delta for arm in request.arms],
                )
            except Exception as exc:
                branch_error = describe_exception(exc)
                logger.error("ablation branch at %r failed: %s", stage, branch_error)
    finally:
        # The branched stage's recorded snapshot refs — the committed sandbox
        # image and environment snapshot id a reader needs to restore this
        # world and re-branch it by hand later (RFC §3.2; also on disk in the
        # parent run's stage_snapshots.json). Read — and retained/annotated —
        # BEFORE cleanup: cleanup destroys the committed image, and a report
        # serialized afterwards once published a snapshot ref that
        # ``docker image inspect`` could no longer resolve. cleanup always
        # runs, and never over a masked retention error (retain_stage_snapshot
        # records failures instead of raising).
        try:
            stage_registry = getattr(rollout, "_stage_snapshots", None)
            if stage_registry:
                from benchflow.branch_lineage import stage_snapshots_payload

                report.stage_snapshot = stage_snapshots_payload(stage_registry).get(
                    stage
                )
            await retain_stage_snapshot(
                report,
                sandbox=getattr(rollout, "env", None),
                keep_snapshots=request.keep_snapshots,
                out_dir=Path(request.out_dir),
            )
        finally:
            await rollout.cleanup()

    run_dir = getattr(rollout, "_rollout_dir", None)
    report.parent_run_dir = None if run_dir is None else str(run_dir)
    report.arms = outcomes_for_arms(
        request.arms,
        branch_children_of(rollout.tree),
        run_dir=run_dir,
        branch_error=branch_error,
        environment_stamps=environment_stamps,
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

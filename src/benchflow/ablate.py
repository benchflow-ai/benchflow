"""Stage-level ablation — run a task once, branch a stage, compare the arms.

The library half of ``bench eval ablate`` (rollout-branching RFC §5). The
branch machinery underneath is already complete: a rollout captures a stage
boundary (RFC §3.2), :meth:`~benchflow.rollout.Rollout.branch_at_stage` forks
that recorded world into one child per
:class:`~benchflow.branch_delta.BranchDelta` (RFC §3.3), and every child leaves
lineage artifacts (RFC §3.4). What was missing is the user-facing shape of the
experiment: *arms*.

An **arm** is one delta plus the name a reader recognizes it by
(``with-skill``, ``no-skill``, ``inject:<file>``). This module drives the
parent rollout to the requested boundary, forks it once into all arms, and
turns the per-arm rewards into an :class:`AblationReport` — a deterministic
``ablation.json`` plus a one-line, observation-only verdict per arm.

Attribution runs at two granularities, because a binary reward is a lossy
summary of what an arm did: the scalar comparison, and the per-test outcomes
mined from each arm's own verifier report (:func:`differing_tests`,
:func:`sub_test_attribution`). A measured skills ablation that scored 0.00/0.00
had *both* its sub-tests flip in opposite directions — attributing on the
scalar alone would have reported "no difference" about a large, reproducible
behavioral one.

Everything decidable from the request alone is decided *before* the parent
rollout runs, in :mod:`benchflow.ablate_arms` (:func:`parse_arms`,
:func:`validate_arms_for_stage` — re-exported here as the one ablation
surface): an ablation costs a full task run before the branch, so a request
the branch engine would reject at fork time must not cost that run first. The
engine keeps its own gates — the pre-flight is a mirror, never a replacement.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.ablate_arms import (  # noqa: F401  (pre-flight API re-exported here)
    ARM_KIND_CONFIG,
    ARM_KIND_ENV,
    ARM_KIND_INJECT,
    ARM_KIND_SKILL_MODE,
    CAPTURABLE_STAGES,
    CONFIG_PREFIX,
    ENV_PREFIX,
    INJECT_PREFIX,
    AblationArm,
    AblationError,
    AblationRunError,
    AblationSpecError,
    _split_arm_specs,
    parse_arm,
    parse_arms,
    resolve_ablation_environment_binding,
    resolve_ablation_task,
    validate_arms_for_stage,
)
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
from benchflow.branch_skill import SKILL_DELTA_LAYER
from benchflow.branch_stage import STAGE_ENV_READY, STAGE_POST_RESEARCH
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

logger = logging.getLogger(__name__)

#: The parent rollout's job name under ``--out-dir`` — fixed, not stamped, so
#: the run directory of an ablation is derivable from its output directory.
PARENT_JOB_NAME = "ablation"


# Request


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


async def export_stage_snapshot(
    sandbox: Any, *, sandbox_ref: str, out_dir: Path
) -> dict[str, Any]:
    """``docker save`` the branched stage's sandbox image into ``out_dir``.

    The durable half of ``--keep-snapshots`` (RFC §3.6): the tar lands at
    ``<out_dir>/snapshots/<ref>.tar`` and the returned record carries its
    path, content sha256 and image id — enough for a reader to verify,
    ``docker load`` and identity-check the world later (``bench eval
    import-snapshots``). Raises :class:`AblationError` when the sandbox
    backend cannot export (the caller records the failure; the report stays
    truthful). Thin ablation-facing wrapper over the shared retention
    machinery in :mod:`benchflow.branch_policy` — the same code path ``bench
    eval run --keep-snapshots`` exports through, so the two artifacts cannot
    drift.
    """
    from benchflow.branch_policy import SnapshotExportUnsupported
    from benchflow.branch_policy import export_stage_snapshot as _export_engine

    try:
        return await _export_engine(sandbox, sandbox_ref=sandbox_ref, out_dir=out_dir)
    except SnapshotExportUnsupported as exc:
        raise AblationError(str(exc)) from exc


async def retain_stage_snapshot(
    report: AblationReport,
    *,
    sandbox: Any,
    keep_snapshots: bool,
    out_dir: Path,
    run_dir: Path | str | None = None,
) -> None:
    """Make the report truthful about the stage snapshot's lifetime (RFC §3.6).

    Must run **before** ``rollout.cleanup()`` — cleanup's
    ``compose down --rmi all`` is what destroys the committed ``bf-snap-*``
    image, and a report serialized afterwards once published a handle
    ``docker image inspect`` could no longer resolve. With
    ``keep_snapshots`` the image is exported to a tar and the entry records
    ``ephemeral: false`` plus the tar's path, sha256 and image id; without it
    (or when the export fails — recorded as ``export_error``) the entry
    records ``ephemeral: true, exported: null`` so a reader knows the ref no
    longer resolves. Never raises: the arms' rewards must survive a failed
    export.

    ``run_dir`` (the parent rollout's run directory, when known) receives the
    same annotation into its ``stage_snapshots.json`` entry — the on-disk
    twin of ``report.stage_snapshot`` — so the parent artifact and the
    ablation report tell one story; ``Rollout.cleanup()``'s own lifetime
    finalization then preserves this entry instead of re-marking it.
    """
    from benchflow.branch_policy import annotate_stage_snapshot_lifetime

    snap = report.stage_snapshot
    if snap is None:
        return
    await annotate_stage_snapshot_lifetime(
        snap, sandbox=sandbox, keep=keep_snapshots, out_dir=Path(out_dir)
    )
    if run_dir is None:
        return
    try:
        from benchflow.branch_lineage import annotate_stage_snapshots_file

        annotate_stage_snapshots_file(
            run_dir=Path(run_dir), annotations={report.stage: snap}
        )
    except Exception:
        logger.warning(
            "stage-snapshot lifetime annotation under %s failed — the report "
            "is unaffected",
            run_dir,
            exc_info=True,
        )


# The run_ablation phases — resolve/prepare, parent leg, branch leg,
# retention, report — each with the failure-isolation rule it owns.


def _preflight_environment_arms(
    arms: Sequence[AblationArm], environment_manifest: Any
) -> dict[str, dict[str, Any]]:
    """Pre-flight the env arms' content gates; stamp the arms that pass.

    Runs once the parent's manifest is known, still *before* the parent run:
    an unresolvable ref, an image-changing manifest, or an entrypoint-owned
    lifecycle is decidable from the request alone, and the branch engine
    would reject it only after a full parent run. An arm that passes the gate
    is stamped with the environment it swaps in, so its report row names the
    world it ran against.
    """
    from benchflow.branch_skill import resolve_environment_ref_delta
    from benchflow.environment.manifest import load_manifest_binding

    environment_stamps: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm.kind != ARM_KIND_ENV or arm.delta.environment_ref is None:
            continue
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
    return environment_stamps


async def _branch_into_arms(
    rollout: Any, request: AblationRequest, report: AblationReport, stage: str
) -> str | None:
    """The branch leg: fork the recorded boundary once into all the arms.

    Returns the branch error instead of raising — a branch failure becomes
    reported arm errors, and the arms that did run keep their rewards. Two
    gates skip the fork outright, each naming what actually happened rather
    than what the engine would have raised:

    * a parent error classifying **request-global** (an unsupported reasoning
      effort / model — PR #1046 second review, P2-A): every child would
      restore the snapshot, reinstall the agent, and be rejected identically,
      so the error is a property of the request, never of a task or an arm;
    * a research-end trigger that never fired: branching would only raise
      ``BranchStageNotCaptured``, and the marker file — not the stage
      machinery — is what was missing.
    """
    from benchflow._utils.scoring import REQUEST_GLOBAL, classify_error
    from benchflow._utils.text import describe_exception

    if classify_error(report.parent_error) == REQUEST_GLOBAL:
        branch_error = (
            "the parent run failed with a request-global configuration "
            f"error, so the arms were not attempted: "
            f"{report.parent_error} — every branch child would re-run "
            "the same rejected agent/model configuration"
        )
        logger.error("ablation branch at %r skipped: %s", stage, branch_error)
        return branch_error
    if request.mark_research_end_on is not None and stage not in getattr(
        rollout, "_stage_snapshots", {}
    ):
        branch_error = (
            f"the research-end marker {request.mark_research_end_on!r} "
            "never appeared in the workspace during the parent run, so "
            f"{stage!r} was never captured and there is nothing to branch"
        )
        logger.error("ablation branch at %r failed: %s", stage, branch_error)
        return branch_error
    try:
        report.value = await rollout.branch_at_stage(
            stage,
            len(request.arms),
            deltas=[arm.delta for arm in request.arms],
        )
    except Exception as exc:
        branch_error = describe_exception(exc)
        logger.error("ablation branch at %r failed: %s", stage, branch_error)
        return branch_error
    return None


async def _retain_and_cleanup(
    rollout: Any, request: AblationRequest, report: AblationReport, stage: str
) -> None:
    """The retention leg, then — always — the parent sandbox's cleanup.

    The branched stage's recorded snapshot refs — the committed sandbox image
    and environment snapshot id a reader needs to restore this world and
    re-branch it by hand later (RFC §3.2; also on disk in the parent run's
    ``stage_snapshots.json``) — are read, retained and annotated **before**
    cleanup: cleanup destroys the committed image, and a report serialized
    afterwards once published a snapshot ref that ``docker image inspect``
    could no longer resolve. Cleanup always runs, and never over a masked
    retention error (:func:`retain_stage_snapshot` records failures instead
    of raising).
    """
    try:
        stage_registry = getattr(rollout, "_stage_snapshots", None)
        if stage_registry:
            from benchflow.branch_lineage import stage_snapshots_payload

            report.stage_snapshot = stage_snapshots_payload(stage_registry).get(stage)
        await retain_stage_snapshot(
            report,
            sandbox=getattr(rollout, "env", None),
            keep_snapshots=request.keep_snapshots,
            out_dir=Path(request.out_dir),
            run_dir=getattr(rollout, "_rollout_dir", None),
        )
    finally:
        await rollout.cleanup()


def _finalize_report(
    rollout: Any,
    request: AblationRequest,
    report: AblationReport,
    *,
    stage: str,
    branch_error: str | None,
    environment_stamps: dict[str, dict[str, Any]],
) -> AblationReport:
    """The report leg: per-arm outcomes off the tree, attribution, artifacts.

    A branch that failed before it forked anything (an uncaptured stage, a
    capability gap) has no arm to carry the error, so the report carries it.
    The parent's own result materialization is failure-isolated — the arms'
    rewards are already in the report and must survive a result-build error.
    """
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


async def run_ablation(request: AblationRequest) -> AblationReport:
    """Run the task once, fork the requested stage into the arms, score them.

    The whole command in one call: validate (again — the library is the
    contract, the CLI one caller), run the parent to the stage boundary, fork
    it once into ``len(arms)`` children carrying the arms' deltas, and read the
    per-arm rewards back off the tree the engine grew. The parent's sandbox is
    always cleaned up, and a branch failure becomes reported arm errors rather
    than an exception — the arms that did run keep their rewards.

    The phases, in order — each a named function carrying its own
    failure-isolation rule: pre-flight (:func:`validate_arms_for_stage`,
    :func:`resolve_ablation_environment_binding`,
    :func:`_preflight_environment_arms`, :func:`resolve_canonical_parent_config`
    — all before the parent run costs anything), the parent leg
    (:func:`_run_parent`), the branch leg (:func:`_branch_into_arms`),
    retention + cleanup (:func:`_retain_and_cleanup` — always runs), and the
    report (:func:`_finalize_report`).
    """
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
    environment_stamps = _preflight_environment_arms(request.arms, environment_manifest)
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
        branch_error = await _branch_into_arms(rollout, request, report, stage)
    finally:
        await _retain_and_cleanup(rollout, request, report, stage)
    return _finalize_report(
        rollout,
        request,
        report,
        stage=stage,
        branch_error=branch_error,
        environment_stamps=environment_stamps,
    )

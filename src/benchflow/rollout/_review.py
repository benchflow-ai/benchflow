"""Terminal review lifecycle hooks, separate from solver execution."""

from __future__ import annotations

import json
import logging
import posixpath
import shlex
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

from benchflow._root_python import root_python_sh
from benchflow._utils import kill_record
from benchflow._utils.text import describe_exception
from benchflow.agents.credentials import credential_evidence_overrides
from benchflow.models import RolloutResult
from benchflow.review.automatic import finish_review, prepare_review
from benchflow.review.evidence import EvidenceLimitError, capture_task_evidence
from benchflow.review.evidence_runtime import (
    ensure_capture_tools,
    ensure_evidence_python,
)
from benchflow.review.outcome import scoring_error, scoring_from_result
from benchflow.review.persistence import commit_scoring_result, scoring_lock
from benchflow.rollout._recovery_submission import capture_submission
from benchflow.rollout._separate_verifier import scan_outside_workspace_writes
from benchflow.sandbox._exec_output import exec_raw
from benchflow.task import keep_services
from benchflow.task.config import ArtifactConfig
from benchflow.task.verifier_sandbox import separate_verifier_requested

if TYPE_CHECKING:
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)


def prepare_terminal_review(rollout: Rollout) -> None:
    """Preflight only ordinary scored tasks; review children never recurse."""
    cfg = rollout._config
    if cfg.purpose == "reviewer" or cfg.skip_verify:
        return
    rollout._review_plan = prepare_review(cfg.task_path, cfg.reviewer)
    if rollout._review_plan is not None:
        digest = rollout._review_plan.task_digest
        if cfg.task_digest is not None and cfg.task_digest != digest:
            raise ValueError(
                "Task digest differs from the task selected for automatic review"
            )
        cfg.task_digest = digest


def _freeze_requested(rollout: Any) -> bool:
    """``--freeze-workspace``, or a separate verifier that needs the workspace."""
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    return bool(getattr(rollout._config, "freeze_workspace", False)) or (
        separate_verifier_requested(getattr(rollout._task, "config", None))
    )


async def prepare_capture_runtime(rollout: Rollout) -> None:
    """Ensure required capture dependencies exist before solver execution."""
    from benchflow.rollout._verifier_recovery import recovery_ineligible_reason
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    recovery_eligible = recovery_ineligible_reason(rollout) is None
    if (
        rollout._review_plan is not None
        or rollout._config.purpose == "reviewer"
        or recovery_eligible
    ):
        await ensure_evidence_python(
            rollout._env,
            timeout_sec=rollout._config.sandbox_setup_timeout,
            allow_install=not recovery_eligible,
        )
    elif separate_verifier_requested(getattr(rollout._task, "config", None)):
        # A separate verifier sees only the captured workspace. Capture runs
        # with python3 or, failing that, tar; nothing is installed into the
        # agent's image. Missing both fails here, before the agent runs.
        await ensure_capture_tools(
            rollout._env, timeout_sec=rollout._config.sandbox_setup_timeout
        )
    if separate_verifier_requested(getattr(rollout._task, "config", None)):
        from benchflow.rollout._separate_verifier import record_pristine_outputs

        # The clean control: what capture will read, measured before the
        # agent, so only a failure the solution caused is scored as its own.
        await record_pristine_outputs(rollout)


async def _quiesce_and_record(rollout: Rollout, sandbox_user: str) -> None:
    """Kill the agent's processes before the capture and keep the count (K1).

    This is the first kill before grading in this verify, so the hardening
    kill that follows usually finds nothing left. The count stays on
    ``rollout._kill_record``, which ``Rollout.verify`` hands to
    ``_verify_rollout`` so that kill adds to it instead of writing a zero. The
    record is written now as well: a separate verifier runs no kill of its
    own, so this is the only one such a trial has.
    """
    record: dict[str, Any] = {}
    rollout._kill_record = record
    try:
        with kill_record.recording_into(record):
            keep = keep_services.trial_keep(rollout._task, sandbox_user)
            if keep is None:
                await rollout._planes.quiesce_agent(rollout._env, sandbox_user)
            else:
                outcome = await rollout._planes.quiesce_agent(
                    rollout._env, sandbox_user, keep_ports=keep.ports
                )
                keep_services.mark_waited(rollout._task)
                keep_services.note_turn_end(rollout._task, outcome)
    finally:
        if record:
            record["quiesce_killed"] = record["killed"]
            try:
                kill_record.write_kill_record(rollout._require_rollout_dir(), record)
            except Exception:
                logger.debug("kill-before-grading record not written", exc_info=True)


def _separate(rollout: Rollout) -> bool:
    return separate_verifier_requested(getattr(rollout._task, "config", None))


async def mark_agent_phase_start(rollout: Rollout) -> None:
    """Read the sandbox clock as harness setup ends and the agent phase begins.

    Only for a separate verifier: files written after this instant outside
    everything that crosses are listed when the workspace is frozen.
    """
    if not _separate(rollout):
        return
    try:
        # date first, then root's protected Python (_CLOCK_COMMAND).
        result = await exec_raw(
            rollout._env.exec, _CLOCK_COMMAND, user="root", timeout_sec=30
        )
        rollout._agent_phase_started = _parse_clock(result.stdout or "")
    except Exception as exc:
        # One line, no traceback: a shell-only image is an ordinary task
        # (DF-p4-upgrader-9).
        logger.warning(
            "Could not read the sandbox clock (%s); files written outside the "
            "workspace will not be listed",
            exc,
        )
        logger.debug("Sandbox clock read failed", exc_info=True)


# The sandbox's clock with only `date` (coreutils prints nanoseconds; busybox
# prints a literal %N, so whole seconds follow), and Python last. A shell-only
# image (BenchFlow's own ubuntu:24.04 scaffold) has no python3
# (DF-p4-upgrader-9). The Python is root's protected one, never the image's
# python3 as root (C8); merge wave batch 30.
_CLOCK_COMMAND = shlex.join(
    [
        "sh",
        "-c",
        "date +%s.%N 2>/dev/null; date +%s 2>/dev/null; "
        + root_python_sh("import time; print(repr(time.time()))")
        + " 2>/dev/null; true",
    ]
)


def _parse_clock(stdout: str) -> float:
    """The first line of :data:`_CLOCK_COMMAND`'s output that is a time."""
    for line in stdout.splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            continue
        if value > 0:
            return value
    raise ValueError(f"no time in {stdout[-200:]!r}")


async def _list_outside_workspace_writes(rollout: Rollout) -> None:
    since = getattr(rollout, "_agent_phase_started", None)
    if since is None or not _separate(rollout):
        return
    cfg = rollout._config
    keep = [rollout._agent_cwd, *getattr(cfg, "separate_verifier_include", ())]
    if cfg.sandbox_user:
        keep.append(f"/home/{cfg.sandbox_user}")
    report = await scan_outside_workspace_writes(rollout._env, since=since, keep=keep)
    rollout._outside_workspace_writes = report
    if report and report.get("count"):
        logger.warning(
            "%d file(s) written outside the workspace %s during the agent phase "
            "will not reach the separate verifier (by directory: %s; e.g. %s). "
            "If the verifier reads them, pass --separate-verifier-include for "
            "their directory.",
            report["count"],
            rollout._agent_cwd,
            report.get("by_directory"),
            ", ".join(report.get("sample", [])[:5]),
        )


def _uncovered_includes(
    includes: tuple[str, ...] | list[str],
    artifacts: list[str | ArtifactConfig],
) -> list[str]:
    """Include paths a declared artifact does not already capture whole.

    An include equal to or inside a declared artifact's source would be
    captured twice and packed twice (PY8-MAIN-3); the artifact's capture
    carries it. Only an artifact without excludes covers it: one that excludes
    files (``*.bin``) could drop the very file the include names, so the
    include keeps its own capture and the payload packs the overlap once
    (PY8-MAIN-6).
    """
    sources = [
        posixpath.normpath(item if isinstance(item, str) else item.source)
        for item in artifacts
        if isinstance(item, str) or not item.exclude
    ]
    sources = [source for source in sources if source.startswith("/")]
    return [
        path
        for path in includes
        if not any(
            path == source or path.startswith(source.rstrip("/") + "/")
            for source in sources
        )
    ]


async def capture_terminal_workspace(rollout: Rollout) -> None:
    """Freeze solver or reviewer files before verifier hardening mutates them."""
    if rollout._branch_child_active:
        return
    # Collection can take longer than the solver's budget. Stop writers even
    # when no workspace evidence is requested, and do not swallow a failed
    # quiescence check as an optional capture error before admitting grading.
    await rollout.disconnect()
    if rollout._config.sandbox_user:
        await _quiesce_and_record(rollout, rollout._config.sandbox_user)
    from benchflow.rollout._verifier_recovery import recovery_ineligible_reason

    freeze = _freeze_requested(rollout)
    if (
        rollout._review_plan is None
        and rollout._config.purpose != "reviewer"
        and not freeze
        and recovery_ineligible_reason(rollout) is not None
    ):
        return
    includes: list[ArtifactConfig] = []
    if _separate(rollout):
        # Captured beside the workspace and carried into the verifier sandbox
        # at the same paths; the review sees them as outputs outside it.
        includes = [
            ArtifactConfig(source=path)
            for path in _uncovered_includes(
                getattr(rollout._config, "separate_verifier_include", ()),
                rollout._task.config.artifacts,
            )
        ]
    try:
        if rollout._task.config.verifier.submission_files:
            await capture_submission(rollout)
            if rollout._review_plan is None and not freeze:
                return
        await capture_task_evidence(
            rollout._env,
            rollout._agent_cwd,
            rollout._require_rollout_dir() / "evidence",
            artifacts=[*rollout._task.config.artifacts, *includes],
            excluded_paths=credential_evidence_overrides(
                rollout._agent_env,
                workspace=rollout._agent_cwd,
                cred_home=(
                    f"/home/{rollout._config.sandbox_user}"
                    if rollout._config.sandbox_user
                    else "/root"
                ),
            ),
        )
    except Exception as exc:
        # Tests may still provide useful diagnostics. A capture error prevents
        # final rubric scoring and must not become a solver capability error.
        rollout._export_error = f"Workspace evidence capture failed: {exc}"
        # A separate verifier can blame the solution for this one alone, and
        # only when the clean control measured every captured root: it does
        # not measure include paths (an image can ship them full, harness
        # installs can grow /root), so with includes it stays unscored
        # (PY8-MAIN-2).
        rollout._capture_over_limit = (
            isinstance(exc, EvidenceLimitError) and not includes
        )
        logger.exception("Workspace evidence capture failed")
        return
    await _list_outside_workspace_writes(rollout)


async def finish_terminal_review(
    rollout: Rollout, *, result: RolloutResult | None = None, lock_held: bool = False
) -> RolloutResult:
    """Review after cleanup finalized telemetry and released the solver VM."""
    assert rollout._review_plan is not None
    rollout._phase = "reviewing"
    # This phase record cannot be mistaken for a completed parent result.
    if result is None:
        result = rollout._build_result(result_filename="solver.json")
    try:
        with (
            nullcontext() if lock_held else scoring_lock(rollout._require_rollout_dir())
        ):
            scoring = await finish_review(
                rollout._review_plan, rollout._require_rollout_dir()
            )
            payload = commit_scoring_result(rollout._require_rollout_dir(), scoring)
    except Exception as exc:
        # A failed score write must not escape as a retryable solver crash.
        # solver.json remains available for review-only recovery even if the
        # filesystem cannot accept the final result at this moment.
        logger.exception("Could not commit automatic scoring")
        scoring = scoring_error("Scoring commit failed: " + describe_exception(exc))
        rollout._scoring = result.scoring = scoring
        rollout._rewards = result.rewards = None
        rollout._verifier_error = result.verifier_error = scoring.error
        rollout._phase = "cleaned"
        return result
    rollout._scoring = result.scoring = scoring
    rollout._rewards = result.rewards = payload.get("rewards")
    rollout._verifier_error = result.verifier_error = payload.get("verifier_error")
    result.verifier_error_category = payload.get("verifier_error_category")
    result.finished_at = datetime.fromisoformat(payload["finished_at"])
    rollout._timing = payload["timing"]
    rollout._phase = "cleaned"
    return result


def read_admitted_result(rollout: Rollout) -> RolloutResult | None:
    """Read an already-admitted terminal verdict without invoking any writers.

    Caller holds scoring_lock. Only known RolloutResult fields are hydrated;
    scoring and time fields retain their runtime types.
    """
    root = rollout._require_rollout_dir()
    path = root / "result.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    scoring = scoring_from_result(payload)
    admitted = scoring is not None and scoring.status == "complete"
    if not admitted and (root / "verification.json").is_file():
        from benchflow.rollout._verifier_recovery import verification_source

        verified = verification_source(root)
        admitted = (
            verified.get("rewards") is not None
            and not verified.get("verifier_error")
            and payload.get("rewards") == verified["rewards"]
            and not payload.get("verifier_error")
        )
    if not admitted:
        return None
    result = RolloutResult(
        task_name=payload.get("task_name", rollout._config.task_path.name)
    )
    values = {**payload, **(payload.get("agent_result") or {})}
    for name in vars(result):
        if name in values and name not in {
            "scoring",
            "started_at",
            "finished_at",
            "trajectory",
        }:
            setattr(result, name, values[name])
    result.scoring = scoring
    for name in ("started_at", "finished_at"):
        value = payload.get(name)
        if value is not None:
            setattr(result, name, datetime.fromisoformat(value))
    result.source_provenance = payload.get("source")
    trajectory = root / "trajectory" / "acp_trajectory.jsonl"
    if trajectory.is_file():
        result.trajectory = [
            json.loads(line)
            for line in trajectory.read_text().splitlines()
            if line.strip()
        ]
        # The stored copy is redacted: integrity re-audits it (never as live).
        result._trajectory_stored = True
    return result


def prepare_terminal_result(rollout: Rollout) -> RolloutResult:
    """Serialize solver/result publication against resumed scoring admission.

    Only review and verifier recovery read solver.json; a task with neither
    keeps the single result.json and its ordinary retries.

    A trial with a review plan publishes solver.json alone, as on main: its
    result.json is written once, by ``commit_scoring_result``, when the
    review's scoring commits. Written here, it would carry the unreviewed
    verifier reward, which ``bench train stream`` emits once and never
    rereads, and which a review cut off by a crash or a budget stop would
    leave as the trial's final score. Verifier recovery without a review
    keeps result.json beside solver.json.
    """
    from benchflow.rollout._verifier_recovery import recovery_ineligible_reason

    root = rollout._require_rollout_dir()
    with scoring_lock(root):
        admitted = read_admitted_result(rollout)
        if admitted is not None:
            return admitted
        solver_saved = (root / "solver.json").is_file()
        if rollout._review_plan is not None:
            # solver.json is immutable once written: it keeps the original
            # verifier verdict that recovery and resume start from.
            if solver_saved:
                return rollout._build_result(result_filename=None)
            return rollout._build_result(result_filename="solver.json")
        if recovery_ineligible_reason(rollout) is None and not solver_saved:
            rollout._build_result(result_filename="solver.json")
        return rollout._build_result()

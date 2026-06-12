"""Executable acceptance-live validation for task packages.

The static authoring gate proves that evidence is declared, pinned, and
well-formed. This module owns the next boundary: declared live verifier cases
that run through BenchFlow's sandbox and verifier contracts.

The model block (types, constants, dataclasses) lives in
``acceptance_live_model`` and the report plane (report writing, run summaries,
leaderboard suitability, and reward/flake expectation checks) lives in
``acceptance_live_report``. This module keeps the spec parsing and the
orchestration plane — the latter binds the patched rollout seams
(``Rollout``, ``default_rollout_planes``, ``_start_env_and_upload``,
``_resolve_agent_cwd``, ``_verify_rollout``) so they must resolve through this
module's namespace. Every name from the submodules is re-exported here so
``benchflow.task.acceptance_live`` stays the single import surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from benchflow._types import Scene
from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL,
    classify_verifier_error,
)
from benchflow.contracts import default_rollout_planes
from benchflow.rollout import (
    Rollout,
    RolloutConfig,
    _resolve_agent_cwd,
    _start_env_and_upload,
    _verify_rollout,
)

# --- Façade re-exports -------------------------------------------------------
# ``acceptance_live.py`` was split: the model block (types, constants,
# dataclasses) and the report plane (report writing, summaries, leaderboard
# suitability, reward/flake expectation checks) moved to private submodules.
# Re-export every moved symbol — including the de-facto public underscore
# helpers that the test suite imports and accesses — so every name that was
# importable from ``benchflow.task.acceptance_live`` still resolves unchanged.
# The redundant ``import x as x`` aliases mark these as intentional re-exports.
from benchflow.task.acceptance_live_model import _CASE_TYPES as _CASE_TYPES
from benchflow.task.acceptance_live_model import _DEFAULT_RERUNS as _DEFAULT_RERUNS
from benchflow.task.acceptance_live_model import (
    _DEP_INSTALL_FLAKE_HINT as _DEP_INSTALL_FLAKE_HINT,
)
from benchflow.task.acceptance_live_model import (
    _LEADERBOARD_CALIBRATION_TYPES as _LEADERBOARD_CALIBRATION_TYPES,
)
from benchflow.task.acceptance_live_model import _MAX_RERUNS as _MAX_RERUNS
from benchflow.task.acceptance_live_model import _STAGE_IGNORE as _STAGE_IGNORE
from benchflow.task.acceptance_live_model import (
    _WORKSPACE_SOURCE_CURRENT_WORKTREE as _WORKSPACE_SOURCE_CURRENT_WORKTREE,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceCase as LiveAcceptanceCase,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceCaseSource as LiveAcceptanceCaseSource,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceCaseType as LiveAcceptanceCaseType,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceExpectation as LiveAcceptanceExpectation,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceLeaderboard as LiveAcceptanceLeaderboard,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceRunResult as LiveAcceptanceRunResult,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceSpec as LiveAcceptanceSpec,
)
from benchflow.task.acceptance_live_model import (
    LiveAcceptanceWorkspace as LiveAcceptanceWorkspace,
)
from benchflow.task.acceptance_live_report import (
    _benchflow_version as _benchflow_version,
)
from benchflow.task.acceptance_live_report import _canonical_sha256 as _canonical_sha256
from benchflow.task.acceptance_live_report import (
    _case_failure_hint as _case_failure_hint,
)
from benchflow.task.acceptance_live_report import _case_summary as _case_summary
from benchflow.task.acceptance_live_report import (
    _check_case_flake_expectation as _check_case_flake_expectation,
)
from benchflow.task.acceptance_live_report import (
    _check_reward_expectation as _check_reward_expectation,
)
from benchflow.task.acceptance_live_report import _expectation_dict as _expectation_dict
from benchflow.task.acceptance_live_report import _file_sha256 as _file_sha256
from benchflow.task.acceptance_live_report import (
    _leaderboard_suitability as _leaderboard_suitability,
)
from benchflow.task.acceptance_live_report import (
    _live_report_output_path as _live_report_output_path,
)
from benchflow.task.acceptance_live_report import _report_summary as _report_summary
from benchflow.task.acceptance_live_report import _run_record as _run_record
from benchflow.task.acceptance_live_report import _spec_sha256 as _spec_sha256
from benchflow.task.acceptance_live_report import _tree_sha256 as _tree_sha256
from benchflow.task.acceptance_live_report import (
    _write_live_acceptance_report as _write_live_acceptance_report,
)
from benchflow.task.acceptance_live_validation import (
    _check_duplicate_case_names as _check_duplicate_case_names,
)
from benchflow.task.acceptance_live_validation import (
    _generated_case_from_calibration_report as _generated_case_from_calibration_report,
)
from benchflow.task.acceptance_live_validation import (
    _is_executable_file as _is_executable_file,
)
from benchflow.task.acceptance_live_validation import (
    _is_safe_sandbox_dir as _is_safe_sandbox_dir,
)
from benchflow.task.acceptance_live_validation import _number_value as _number_value
from benchflow.task.acceptance_live_validation import (
    _optional_reward as _optional_reward,
)
from benchflow.task.acceptance_live_validation import _parse_case as _parse_case
from benchflow.task.acceptance_live_validation import _parse_cases as _parse_cases
from benchflow.task.acceptance_live_validation import (
    _parse_expectation as _parse_expectation,
)
from benchflow.task.acceptance_live_validation import (
    _parse_generated_calibration_cases as _parse_generated_calibration_cases,
)
from benchflow.task.acceptance_live_validation import (
    _parse_leaderboard as _parse_leaderboard,
)
from benchflow.task.acceptance_live_validation import (
    _parse_report_output_path as _parse_report_output_path,
)
from benchflow.task.acceptance_live_validation import (
    _parse_report_path as _parse_report_path,
)
from benchflow.task.acceptance_live_validation import (
    _parse_workspace as _parse_workspace,
)
from benchflow.task.acceptance_live_validation import (
    _required_probability as _required_probability,
)
from benchflow.task.acceptance_live_validation import (
    _required_probability_range as _required_probability_range,
)
from benchflow.task.acceptance_live_validation import (
    _safe_relative_evidence_path as _safe_relative_evidence_path,
)
from benchflow.task.acceptance_live_validation import (
    _safe_relative_file_path as _safe_relative_file_path,
)
from benchflow.task.acceptance_live_validation import (
    parse_live_acceptance_spec as parse_live_acceptance_spec,
)
from benchflow.task.paths import RolloutPaths
from benchflow.task.task import Task


def run_live_acceptance_checks(
    task_dir: Path,
    *,
    sandbox_type: str,
    evidence: Mapping[str, object],
    report_output: Path | None = None,
    write_report: bool = True,
) -> list[str]:
    """Run declared acceptance-live cases and return validation issues.

    This sync wrapper keeps ``check_task`` sync while the real work remains
    async. If a caller is already inside an event loop, fail closed with a
    specific issue instead of trying to nest ``asyncio.run``.

    When ``write_report`` is ``False`` the declared report (and its ``.sha256``
    sidecar) is not written, so routine dogfood validates without dirtying the
    task package. Leaderboard suitability is still validated from the in-memory
    run, so the report contract stays enforced.
    """

    spec, issues = parse_live_acceptance_spec(
        task_dir,
        evidence=evidence,
        report_output=report_output,
    )
    if issues:
        return issues
    assert spec is not None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _run_live_acceptance_checks(
                task_dir,
                sandbox_type=sandbox_type,
                spec=spec,
                write_report=write_report,
            )
        )
    return [
        "acceptance-live validation cannot run inside an active event loop; "
        "call the async live acceptance runner instead"
    ]


async def _run_live_acceptance_checks(
    task_dir: Path,
    *,
    sandbox_type: str,
    spec: LiveAcceptanceSpec,
    write_report: bool = True,
) -> list[str]:
    stage_dir: tempfile.TemporaryDirectory[str] | None = None
    staged_worktree: Path | None = None
    if spec.workspace.source == _WORKSPACE_SOURCE_CURRENT_WORKTREE:
        stage_dir = tempfile.TemporaryDirectory(prefix="benchflow-acceptance-live-")
        staged_worktree = _stage_current_worktree(Path(stage_dir.name))

    try:
        issues: list[str] = []
        records: list[dict[str, Any]] = []
        for case in spec.cases:
            case_issues, case_records = await _run_live_acceptance_case(
                task_dir,
                sandbox_type=sandbox_type,
                workspace=spec.workspace,
                staged_worktree=staged_worktree,
                case=case,
            )
            issues.extend(case_issues)
            records.extend(case_records)
        leaderboard_suitability = _leaderboard_suitability(spec=spec, records=records)
        if spec.leaderboard.required:
            issues.extend(
                "acceptance-live leaderboard suitability: " + issue
                for issue in leaderboard_suitability["issues"]
            )
        if write_report and spec.report_path is not None:
            _write_live_acceptance_report(
                task_dir,
                sandbox_type=sandbox_type,
                spec=spec,
                records=records,
                staged_worktree=staged_worktree,
                leaderboard_suitability=leaderboard_suitability,
            )
        return issues
    finally:
        if stage_dir is not None:
            stage_dir.cleanup()


async def _run_live_acceptance_case(
    task_dir: Path,
    *,
    sandbox_type: str,
    workspace: LiveAcceptanceWorkspace,
    staged_worktree: Path | None,
    case: LiveAcceptanceCase,
) -> tuple[list[str], list[dict[str, Any]]]:
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    use_case_flake_threshold = case.expect.flake_rate_max is not None
    for run_index in range(1, case.reruns + 1):
        result = _coerce_run_result(
            await _run_single_case(
                task_dir,
                sandbox_type=sandbox_type,
                workspace=workspace,
                staged_worktree=staged_worktree,
                case=case,
                run_index=run_index,
            )
        )
        reward = result.reward
        error = result.error
        prefix = f"acceptance-live case {case.name!r} run {run_index}"
        expectation_issues: list[str] = []
        if error is not None:
            if not use_case_flake_threshold:
                issues.append(f"{prefix} failed: {error}")
        elif reward is None:
            if not use_case_flake_threshold:
                issues.append(f"{prefix} did not produce scalar reward")
        else:
            expectation_issues = _check_reward_expectation(prefix, reward, case.expect)
            if not use_case_flake_threshold:
                issues.extend(expectation_issues)
        records.append(
            _run_record(
                case=case,
                run_index=run_index,
                result=result,
                expectation_issues=expectation_issues,
            )
        )
    if use_case_flake_threshold:
        issues.extend(_check_case_flake_expectation(case=case, records=records))
    return issues, records


async def _run_single_case(
    task_dir: Path,
    *,
    sandbox_type: str,
    workspace: LiveAcceptanceWorkspace,
    staged_worktree: Path | None,
    case: LiveAcceptanceCase,
    run_index: int,
) -> LiveAcceptanceRunResult:
    if case.case_type == "oracle":
        return await _run_oracle_case(
            task_dir,
            sandbox_type=sandbox_type,
            workspace=workspace,
            staged_worktree=staged_worktree,
            case=case,
            run_index=run_index,
        )

    task = Task(task_dir)
    planes = default_rollout_planes()
    timing: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="benchflow-acceptance-live-run-") as tmp:
        rollout_paths = RolloutPaths(Path(tmp) / "rollout")
        rollout_paths.mkdir()
        rollout_name = (
            f"acceptance-live-{task_dir.name}-{case.name}-{run_index}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        env = planes.create_environment(
            sandbox_type,
            task,
            task_dir,
            rollout_name,
            rollout_paths,
            preserve_agent_network=False,
            environment_manifest=None,
        )
        try:
            await _start_env_and_upload(env, task_dir, timing)
            agent_cwd = await _resolve_agent_cwd(env, task)
            await _upload_workspace(
                env,
                workspace=workspace,
                staged_worktree=staged_worktree,
            )
            if case.command is not None:
                result = await env.exec(
                    f"cd {shlex.quote(workspace.target)} && {case.command}",
                    user="root",
                    timeout_sec=task.config.verifier.timeout_sec,
                )
                return_code = getattr(
                    result,
                    "return_code",
                    getattr(result, "exit_code", 0),
                )
                if isinstance(return_code, int) and return_code != 0:
                    return LiveAcceptanceRunResult(
                        reward=None,
                        error=f"setup command exited with rc={return_code}",
                        diagnostic_code="setup_command_failed",
                    )
            rewards, verifier_error, _diagnostic = await _verify_rollout(
                env,
                task,
                rollout_paths,
                timing,
                planes,
                sandbox_user=None,
                workspace=agent_cwd,
            )
            if verifier_error is not None:
                return _verifier_error_result(verifier_error)
            reward = _scalar_reward(rewards)
            return LiveAcceptanceRunResult(reward=reward, error=None)
        except Exception as exc:
            return LiveAcceptanceRunResult(reward=None, error=str(exc))
        finally:
            stop = getattr(env, "stop", None)
            if stop is not None:
                with contextlib.suppress(Exception):
                    await stop(delete=True)


async def _run_oracle_case(
    task_dir: Path,
    *,
    sandbox_type: str,
    workspace: LiveAcceptanceWorkspace,
    staged_worktree: Path | None,
    case: LiveAcceptanceCase,
    run_index: int,
) -> LiveAcceptanceRunResult:
    with tempfile.TemporaryDirectory(prefix="benchflow-acceptance-live-run-") as tmp:
        rollout_name = (
            f"acceptance-live-{task_dir.name}-{case.name}-{run_index}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        async def upload_live_workspace(env: Any) -> None:
            await _upload_workspace(
                env,
                workspace=workspace,
                staged_worktree=staged_worktree,
            )

        config = RolloutConfig(
            task_path=task_dir,
            environment=sandbox_type,
            agent="oracle",
            model=None,
            scenes=[Scene.single(agent="oracle", model=None, role_name="oracle")],
            jobs_dir=Path(tmp) / "jobs",
            rollout_name=rollout_name,
            pre_agent_hooks=[upload_live_workspace],
        )
        try:
            rollout = await Rollout.create(config)
            result = await rollout.run()
        except Exception as exc:
            return LiveAcceptanceRunResult(reward=None, error=str(exc))

    return_code = _oracle_return_code(result.trajectory)
    if return_code is None:
        return LiveAcceptanceRunResult(
            reward=None,
            error="oracle rerun did not record oracle trajectory event",
        )
    if return_code != 0:
        return LiveAcceptanceRunResult(
            reward=None,
            error=f"oracle exited with rc={return_code}",
        )
    if result.error is not None:
        return LiveAcceptanceRunResult(reward=None, error=result.error)
    if result.verifier_error is not None:
        return _verifier_error_result(result.verifier_error)
    return LiveAcceptanceRunResult(reward=_scalar_reward(result.rewards), error=None)


def _verifier_error_result(error: str) -> LiveAcceptanceRunResult:
    category = classify_verifier_error(error)
    artifact_hint = (
        "verifier/test-stdout.txt" if category == VERIFIER_DEP_INSTALL else None
    )
    return LiveAcceptanceRunResult(
        reward=None,
        error=error,
        verifier_error_category=category,
        diagnostic_code=category,
        artifact_hint=artifact_hint,
    )


def _coerce_run_result(value: object) -> LiveAcceptanceRunResult:
    if isinstance(value, LiveAcceptanceRunResult):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        reward, error = value
        return LiveAcceptanceRunResult(
            reward=cast(float | None, reward),
            error=cast(str | None, error),
        )
    raise TypeError("acceptance-live run result must be LiveAcceptanceRunResult")


async def _upload_workspace(
    env: Any,
    *,
    workspace: LiveAcceptanceWorkspace,
    staged_worktree: Path | None,
) -> None:
    if workspace.source != _WORKSPACE_SOURCE_CURRENT_WORKTREE:
        raise RuntimeError(
            f"unsupported acceptance-live workspace source: {workspace.source}"
        )
    if staged_worktree is None:
        raise RuntimeError("acceptance-live current-worktree was not staged")
    await env.upload_dir(staged_worktree, workspace.target)


def _stage_current_worktree(temp_root: Path) -> Path:
    source = Path.cwd().resolve()
    target = temp_root / "current-worktree"
    shutil.copytree(source, target, symlinks=False, ignore=_STAGE_IGNORE)
    return target


def _scalar_reward(rewards: Mapping[str, Any] | None) -> float | None:
    if not isinstance(rewards, Mapping):
        return None
    reward = rewards.get("reward")
    if not isinstance(reward, int | float) or isinstance(reward, bool):
        return None
    scalar = float(reward)
    if not 0.0 <= scalar <= 1.0:
        return None
    return scalar


def _oracle_return_code(trajectory: list[dict]) -> int | None:
    for event in trajectory:
        if event.get("type") != "oracle":
            continue
        return_code = event.get("return_code")
        if isinstance(return_code, int) and not isinstance(return_code, bool):
            return return_code
    return None

"""Executable acceptance-live validation for task packages.

The static authoring gate proves that evidence is declared, pinned, and
well-formed. This module owns the next boundary: declared live verifier cases
that run through BenchFlow's sandbox and verifier contracts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from benchflow._types import Scene
from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL,
    classify_verifier_error,
    contains_verifier_dep_install_marker,
)
from benchflow.contracts import default_rollout_planes
from benchflow.rollout import (
    Rollout,
    RolloutConfig,
    _resolve_agent_cwd,
    _start_env_and_upload,
    _verify_rollout,
)
from benchflow.task.paths import RolloutPaths
from benchflow.task.task import Task

LiveAcceptanceCaseType = Literal[
    "verifier",
    "oracle",
    "no-op",
    "known-bad",
    "partial",
    "reference",
]
LiveAcceptanceCaseSource = Literal["declared", "calibration-report"]
_CASE_TYPES: set[str] = {
    "verifier",
    "oracle",
    "no-op",
    "known-bad",
    "partial",
    "reference",
}
_WORKSPACE_SOURCE_CURRENT_WORKTREE = "current-worktree"
_DEFAULT_RERUNS = 1
_MAX_RERUNS = 20
_LEADERBOARD_CALIBRATION_TYPES = frozenset(
    {"no-op", "known-bad", "partial", "reference"}
)
_STAGE_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    "__pycache__",
    "htmlcov",
    "jobs",
    "node_modules",
    "*.pyc",
)
_DEP_INSTALL_FLAKE_HINT = (
    "first failed run indicates verifier dependency install failed "
    "(see verifier/test-stdout.txt in the run artifacts)"
)


@dataclass(frozen=True)
class LiveAcceptanceWorkspace:
    source: Literal["current-worktree"]
    target: str


@dataclass(frozen=True)
class LiveAcceptanceExpectation:
    reward_min: float | None = None
    reward_max: float | None = None
    reward_range: tuple[float, float] | None = None
    reward_equals: float | None = None
    flake_rate_max: float | None = None


@dataclass(frozen=True)
class LiveAcceptanceCase:
    name: str
    case_type: LiveAcceptanceCaseType
    command: str | None
    reruns: int
    expect: LiveAcceptanceExpectation
    source: LiveAcceptanceCaseSource = "declared"


@dataclass(frozen=True)
class LiveAcceptanceRunResult:
    reward: float | None
    error: str | None
    verifier_error_category: str | None = None
    diagnostic_code: str | None = None
    artifact_hint: str | None = None


@dataclass(frozen=True)
class LiveAcceptanceLeaderboard:
    required: bool = False
    max_flake_rate: float = 0.0


@dataclass(frozen=True)
class LiveAcceptanceSpec:
    workspace: LiveAcceptanceWorkspace
    cases: tuple[LiveAcceptanceCase, ...]
    report_path: Path | None
    leaderboard: LiveAcceptanceLeaderboard


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


def parse_live_acceptance_spec(
    task_dir: Path,
    *,
    evidence: Mapping[str, object],
    report_output: Path | None = None,
) -> tuple[LiveAcceptanceSpec | None, list[str]]:
    raw = evidence.get("acceptance_live")
    if not isinstance(raw, dict):
        return None, [
            "acceptance-live validation requires "
            "benchflow.evidence.acceptance_live mapping"
        ]
    mapping = cast(dict[str, object], raw)
    workspace, workspace_issues = _parse_workspace(task_dir, mapping.get("workspace"))
    cases, case_issues = _parse_cases(task_dir, mapping.get("cases"))
    generated_cases, generated_issues = _parse_generated_calibration_cases(
        task_dir,
        value=mapping.get("calibration"),
        evidence=evidence,
    )
    leaderboard, leaderboard_issues = _parse_leaderboard(mapping.get("leaderboard"))
    declared_report_path, report_issues = _parse_report_path(mapping.get("report"))
    output_report_path, output_issues = _parse_report_output_path(report_output)
    report_path = output_report_path or declared_report_path
    all_cases = [*cases, *generated_cases]
    if not all_cases:
        case_issues.append(
            "acceptance-live cases must be a non-empty list or generated "
            "calibration cases"
        )
    if leaderboard.required and report_path is None:
        report_issues.append("acceptance-live leaderboard.required requires report")
    case_issues.extend(_check_duplicate_case_names(all_cases))
    issues = [
        *workspace_issues,
        *case_issues,
        *generated_issues,
        *leaderboard_issues,
        *report_issues,
        *output_issues,
    ]
    if issues:
        return None, issues
    assert workspace is not None
    return (
        LiveAcceptanceSpec(
            workspace=workspace,
            cases=tuple(all_cases),
            report_path=report_path,
            leaderboard=leaderboard,
        ),
        [],
    )


def _parse_workspace(
    task_dir: Path,
    value: object,
) -> tuple[LiveAcceptanceWorkspace | None, list[str]]:
    source: Literal["current-worktree"] = _WORKSPACE_SOURCE_CURRENT_WORKTREE
    target: str | None = None
    if value is None:
        task = Task(task_dir)
        target = task.config.environment.workdir or "/app"
    elif isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        source_value = mapping.get("source", _WORKSPACE_SOURCE_CURRENT_WORKTREE)
        if source_value != _WORKSPACE_SOURCE_CURRENT_WORKTREE:
            return None, [
                "acceptance-live workspace.source currently supports only "
                "'current-worktree'"
            ]
        target_value = mapping.get("target")
        if target_value is not None and not isinstance(target_value, str):
            return None, ["acceptance-live workspace.target must be a string"]
        target = target_value or Task(task_dir).config.environment.workdir or "/app"
    else:
        return None, ["acceptance-live workspace must be a mapping when declared"]

    if not _is_safe_sandbox_dir(target):
        return None, [
            "acceptance-live workspace.target must be an absolute non-root "
            "sandbox path"
        ]
    return (
        LiveAcceptanceWorkspace(
            source=source,
            target=target,
        ),
        [],
    )


def _parse_cases(
    task_dir: Path,
    value: object,
) -> tuple[list[LiveAcceptanceCase], list[str]]:
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], ["acceptance-live cases must be a list when declared"]
    if not value:
        return [], ["acceptance-live cases must be non-empty when declared"]
    cases: list[LiveAcceptanceCase] = []
    issues: list[str] = []
    for index, item in enumerate(value):
        prefix = f"acceptance-live cases[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{prefix} must be a mapping")
            continue
        mapping = cast(dict[str, object], item)
        case = _parse_case(task_dir, mapping, prefix=prefix)
        if isinstance(case, LiveAcceptanceCase):
            cases.append(case)
        else:
            issues.extend(cast(list[str], case))
    return cases, issues


def _parse_generated_calibration_cases(
    task_dir: Path,
    *,
    value: object,
    evidence: Mapping[str, object],
) -> tuple[list[LiveAcceptanceCase], list[str]]:
    if value is None:
        return [], []
    source = "acceptance-live calibration"
    if not isinstance(value, dict):
        return [], [f"{source} must be a mapping when declared"]
    mapping = cast(dict[str, object], value)
    if mapping.get("from") != "calibration.report":
        return [], [f"{source}.from currently supports only calibration.report"]

    issues: list[str] = []
    reruns = mapping.get("reruns", _DEFAULT_RERUNS)
    if not isinstance(reruns, int) or isinstance(reruns, bool) or not (
        1 <= reruns <= _MAX_RERUNS
    ):
        issues.append(f"{source}.reruns must be an integer within 1..{_MAX_RERUNS}")
        reruns = _DEFAULT_RERUNS
    flake_rate_max = _optional_reward(
        mapping.get("flake_rate_max"),
        f"{source}.flake_rate_max",
        issues,
    )

    calibration = evidence.get("calibration")
    if not isinstance(calibration, dict):
        return [], [
            *issues,
            f"{source} requires benchflow.evidence.calibration mapping",
        ]
    calibration_mapping = cast(dict[str, object], calibration)
    report_rel = _safe_relative_evidence_path(
        calibration_mapping.get("report"),
        f"{source}.from",
        issues,
    )
    no_op_max = _required_probability(
        calibration_mapping.get("no_op_reward_max"),
        "acceptance calibration.no_op_reward_max",
        issues,
    )
    known_bad_max = _required_probability(
        calibration_mapping.get("known_bad_reward_max"),
        "acceptance calibration.known_bad_reward_max",
        issues,
    )
    partial_range = _required_probability_range(
        calibration_mapping.get("partial_solution_range"),
        "acceptance calibration.partial_solution_range",
        issues,
    )
    if report_rel is None:
        return [], issues

    report_path = task_dir / report_rel
    if not report_path.is_file():
        return [], [*issues, f"{source}.from report is missing: {report_rel.as_posix()}"]
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return [], [*issues, f"{source}.from report is not valid JSON: {exc}"]
    if not isinstance(report, dict):
        return [], [*issues, f"{source}.from report must be a JSON object"]
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        return [], [*issues, f"{source}.from report.cases must be a non-empty list"]

    generated: list[LiveAcceptanceCase] = []
    for index, raw_case in enumerate(raw_cases):
        case_source = f"{source}.from report.cases[{index}]"
        if not isinstance(raw_case, dict):
            issues.append(f"{case_source} must be a mapping")
            continue
        case_mapping = cast(dict[str, object], raw_case)
        generated_case = _generated_case_from_calibration_report(
            case_mapping,
            source=case_source,
            reruns=reruns,
            flake_rate_max=flake_rate_max,
            no_op_reward_max=no_op_max,
            known_bad_reward_max=known_bad_max,
            partial_range=partial_range,
        )
        if isinstance(generated_case, LiveAcceptanceCase):
            generated.append(generated_case)
        else:
            issues.extend(cast(list[str], generated_case))
    return generated, issues


def _generated_case_from_calibration_report(
    mapping: dict[str, object],
    *,
    source: str,
    reruns: int,
    flake_rate_max: float | None,
    no_op_reward_max: float | None,
    known_bad_reward_max: float | None,
    partial_range: tuple[float, float] | None,
) -> LiveAcceptanceCase | list[str]:
    issues: list[str] = []
    name = mapping.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append(f"{source}.name must be a non-empty string")
        name = "<unnamed>"
    case_type = mapping.get("type")
    if case_type not in {"no-op", "known-bad", "partial", "reference"}:
        issues.append(
            f"{source}.type must be no-op, known-bad, partial, or reference"
        )
        case_type = "reference"
    command = mapping.get("command")
    parsed_command: str | None = None
    if command is None:
        if case_type != "reference":
            issues.append(
                f"{source}.command must be declared for generated {case_type} "
                "live calibration cases"
            )
    elif not isinstance(command, str) or not command.strip():
        issues.append(f"{source}.command must be a non-empty string when declared")
    elif "\n" in command or "\r" in command:
        issues.append(f"{source}.command must be a single-line sandbox command")
    else:
        parsed_command = command.strip()

    reward = _number_value(mapping.get("reward"))
    if reward is None or not 0.0 <= reward <= 1.0:
        issues.append(f"{source}.reward must be numeric within 0..1")

    expect: LiveAcceptanceExpectation | None = None
    if case_type == "no-op":
        if no_op_reward_max is not None:
            expect = LiveAcceptanceExpectation(
                reward_max=no_op_reward_max,
                flake_rate_max=flake_rate_max,
            )
    elif case_type == "known-bad":
        if known_bad_reward_max is not None:
            expect = LiveAcceptanceExpectation(
                reward_max=known_bad_reward_max,
                flake_rate_max=flake_rate_max,
            )
    elif case_type == "partial":
        if partial_range is not None:
            expect = LiveAcceptanceExpectation(
                reward_range=partial_range,
                flake_rate_max=flake_rate_max,
            )
    elif reward is not None:
        expect = LiveAcceptanceExpectation(
            reward_equals=reward,
            flake_rate_max=flake_rate_max,
        )
    if expect is None:
        issues.append(f"{source} could not derive live reward expectations")

    if issues:
        return issues
    assert isinstance(name, str)
    assert isinstance(case_type, str)
    assert expect is not None
    return LiveAcceptanceCase(
        name=name.strip(),
        case_type=cast(LiveAcceptanceCaseType, case_type),
        command=parsed_command,
        reruns=reruns,
        expect=expect,
        source="calibration-report",
    )


def _parse_report_path(value: object) -> tuple[Path | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str) or not value.strip():
        return None, ["acceptance-live report must be a non-empty relative path"]
    path = _safe_relative_file_path(value)
    if path is None:
        return None, ["acceptance-live report must be a safe relative file path"]
    return path, []


def _parse_report_output_path(value: Path | None) -> tuple[Path | None, list[str]]:
    if value is None:
        return None, []
    try:
        path = value.expanduser()
    except RuntimeError as exc:
        return None, [f"acceptance-live report output cannot expand user: {exc}"]
    if not path.name:
        return None, ["acceptance-live report output must be a file path"]
    if path.exists() and path.is_dir():
        return None, [
            "acceptance-live report output must be a file path, not a directory"
        ]
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path, []


def _parse_leaderboard(value: object) -> tuple[LiveAcceptanceLeaderboard, list[str]]:
    if value is None:
        return LiveAcceptanceLeaderboard(), []
    source = "acceptance-live leaderboard"
    if not isinstance(value, dict):
        return LiveAcceptanceLeaderboard(), [f"{source} must be a mapping"]
    mapping = cast(dict[str, object], value)
    issues: list[str] = []
    required = mapping.get("required", False)
    if not isinstance(required, bool):
        issues.append(f"{source}.required must be boolean")
        required = False
    max_flake_rate = _optional_reward(
        mapping.get("max_flake_rate", 0.0),
        f"{source}.max_flake_rate",
        issues,
    )
    return (
        LiveAcceptanceLeaderboard(
            required=required,
            max_flake_rate=max_flake_rate if max_flake_rate is not None else 0.0,
        ),
        issues,
    )


def _check_duplicate_case_names(cases: list[LiveAcceptanceCase]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.name in seen:
            duplicates.add(case.name)
        seen.add(case.name)
    return [
        f"acceptance-live case names must be unique; duplicate case {name!r}"
        for name in sorted(duplicates)
    ]


def _parse_case(
    task_dir: Path,
    mapping: dict[str, object],
    *,
    prefix: str,
) -> LiveAcceptanceCase | list[str]:
    issues: list[str] = []
    name = mapping.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append(f"{prefix}.name must be a non-empty string")
        name = "<unnamed>"

    raw_type = mapping.get("type")
    if raw_type not in _CASE_TYPES:
        issues.append(
            f"{prefix}.type must be one of "
            "verifier, oracle, no-op, known-bad, partial, or reference"
        )
        raw_type = "verifier"

    command = mapping.get("command")
    if command is not None:
        if not isinstance(command, str) or not command.strip():
            issues.append(f"{prefix}.command must be a non-empty string when declared")
        elif "\n" in command or "\r" in command:
            issues.append(f"{prefix}.command must be a single-line sandbox command")

    if raw_type == "oracle":
        solve_path = Task(task_dir).paths.solve_path
        if command is not None:
            issues.append(
                f"{prefix}.command is not supported for oracle cases; "
                "acceptance-live oracle uses the selected oracle/solve.sh"
            )
        if not solve_path.is_file():
            issues.append(
                f"{prefix}.type=oracle requires executable oracle/solve.sh "
                f"or legacy solution/solve.sh: {solve_path}"
            )
        elif not _is_executable_file(solve_path):
            issues.append(f"{prefix}.type=oracle requires executable file: {solve_path}")

    reruns = mapping.get("reruns", _DEFAULT_RERUNS)
    if not isinstance(reruns, int) or isinstance(reruns, bool) or not (
        1 <= reruns <= _MAX_RERUNS
    ):
        issues.append(f"{prefix}.reruns must be an integer within 1..{_MAX_RERUNS}")
        reruns = _DEFAULT_RERUNS

    parsed_expect: LiveAcceptanceExpectation | None = None
    expect = _parse_expectation(mapping.get("expect"), prefix=f"{prefix}.expect")
    if isinstance(expect, LiveAcceptanceExpectation):
        parsed_expect = expect
    else:
        issues.extend(cast(list[str], expect))

    if issues:
        return issues
    assert parsed_expect is not None
    assert isinstance(name, str)
    assert isinstance(raw_type, str)
    return LiveAcceptanceCase(
        name=name.strip(),
        case_type=cast(LiveAcceptanceCaseType, raw_type),
        command=command.strip() if isinstance(command, str) else None,
        reruns=reruns,
        expect=parsed_expect,
    )


def _parse_expectation(
    value: object,
    *,
    prefix: str,
) -> LiveAcceptanceExpectation | list[str]:
    if not isinstance(value, dict):
        return [f"{prefix} must be a mapping"]
    mapping = cast(dict[str, object], value)
    issues: list[str] = []
    reward_min = _optional_reward(mapping.get("reward_min"), f"{prefix}.reward_min", issues)
    reward_max = _optional_reward(mapping.get("reward_max"), f"{prefix}.reward_max", issues)
    reward_equals = _optional_reward(
        mapping.get("reward_equals"),
        f"{prefix}.reward_equals",
        issues,
    )
    flake_rate_max = _optional_reward(
        mapping.get("flake_rate_max"),
        f"{prefix}.flake_rate_max",
        issues,
    )
    raw_range = mapping.get("reward_range")
    reward_range: tuple[float, float] | None = None
    if raw_range is not None:
        if isinstance(raw_range, list) and len(raw_range) == 2:
            low = _optional_reward(raw_range[0], f"{prefix}.reward_range[0]", issues)
            high = _optional_reward(raw_range[1], f"{prefix}.reward_range[1]", issues)
            if low is not None and high is not None:
                if low <= high:
                    reward_range = (low, high)
                else:
                    issues.append(f"{prefix}.reward_range min must be <= max")
        else:
            issues.append(f"{prefix}.reward_range must be [min, max]")
    if not any(
        value is not None
        for value in (reward_min, reward_max, reward_range, reward_equals)
    ):
        issues.append(
            f"{prefix} must declare reward_min, reward_max, reward_range, "
            "or reward_equals"
        )
    if issues:
        return issues
    return LiveAcceptanceExpectation(
        reward_min=reward_min,
        reward_max=reward_max,
        reward_range=reward_range,
        reward_equals=reward_equals,
        flake_rate_max=flake_rate_max,
    )


def _optional_reward(value: object, source: str, issues: list[str]) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        issues.append(f"{source} must be numeric within 0..1")
        return None
    reward = float(value)
    if not 0.0 <= reward <= 1.0:
        issues.append(f"{source} must be numeric within 0..1")
        return None
    return reward


def _required_probability(value: object, source: str, issues: list[str]) -> float | None:
    reward = _number_value(value)
    if reward is None or not 0.0 <= reward <= 1.0:
        issues.append(f"{source} must be numeric within 0..1")
        return None
    return reward


def _required_probability_range(
    value: object,
    source: str,
    issues: list[str],
) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        issues.append(f"{source} must be [min, max] within 0..1")
        return None
    low = _number_value(value[0])
    high = _number_value(value[1])
    if low is None or high is None or not 0.0 <= low <= high <= 1.0:
        issues.append(f"{source} must be [min, max] within 0..1")
        return None
    return low, high


def _number_value(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _safe_relative_evidence_path(
    value: object,
    source: str,
    issues: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{source} report must be declared")
        return None
    path = _safe_relative_file_path(value)
    if path is None:
        issues.append(f"{source} report must be a safe relative file path")
        return None
    return path


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


def _write_live_acceptance_report(
    task_dir: Path,
    *,
    sandbox_type: str,
    spec: LiveAcceptanceSpec,
    records: list[dict[str, Any]],
    staged_worktree: Path | None,
    leaderboard_suitability: dict[str, Any],
) -> None:
    if spec.report_path is None:
        return
    report_path = _live_report_output_path(task_dir, spec.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "acceptance-live-report",
        "schema_version": "1.0",
        "benchflow_version": _benchflow_version(),
        "generated_at": datetime.now(UTC).isoformat(),
        "sandbox": sandbox_type,
        "task": {
            "path": task_dir.name,
            "task_md_sha256": _file_sha256(task_dir / "task.md"),
            "oracle_sha256": _file_sha256(Task(task_dir).paths.solve_path),
            "verifier_sha256": _file_sha256(Task(task_dir).paths.test_path),
        },
        "workspace": {
            "source": spec.workspace.source,
            "target": spec.workspace.target,
            "staged_tree_sha256": (
                _tree_sha256(staged_worktree) if staged_worktree is not None else None
            ),
        },
        "spec_sha256": _spec_sha256(spec),
        "cases": [
            {
                "name": case.name,
                "type": case.case_type,
                "source": case.source,
                "command": case.command,
                "reruns": case.reruns,
                "expect": _expectation_dict(case.expect),
            }
            for case in spec.cases
        ],
        "case_summaries": [
            _case_summary(case=case, records=records) for case in spec.cases
        ],
        "leaderboard_suitability": leaderboard_suitability,
        "summary": _report_summary(records),
        "runs": records,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    digest = sha256(report_path.read_bytes()).hexdigest()
    sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {spec.report_path.as_posix()}\n")


def _live_report_output_path(task_dir: Path, report_path: Path) -> Path:
    if report_path.is_absolute():
        return report_path
    return task_dir / report_path


def _benchflow_version() -> str:
    with contextlib.suppress(Exception):
        from benchflow import __version__

        return __version__
    return "0+unknown"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    entries: list[str] = []
    for path in sorted(child for child in root.rglob("*") if child.is_file()):
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}\0{sha256(path.read_bytes()).hexdigest()}")
    return sha256("\n".join(entries).encode()).hexdigest()


def _spec_sha256(spec: LiveAcceptanceSpec) -> str:
    payload = {
        "workspace": {
            "source": spec.workspace.source,
            "target": spec.workspace.target,
        },
        "cases": [
            {
                "name": case.name,
                "type": case.case_type,
                "source": case.source,
                "command": case.command,
                "reruns": case.reruns,
                "expect": _expectation_dict(case.expect),
            }
            for case in spec.cases
        ],
        "leaderboard": {
            "required": spec.leaderboard.required,
            "max_flake_rate": spec.leaderboard.max_flake_rate,
        },
        "report": spec.report_path.as_posix() if spec.report_path else None,
    }
    return _canonical_sha256(payload)


def _run_record(
    *,
    case: LiveAcceptanceCase,
    run_index: int,
    result: LiveAcceptanceRunResult,
    expectation_issues: list[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case": case.name,
        "type": case.case_type,
        "source": case.source,
        "run_index": run_index,
        "reward": result.reward,
        "status": "failed"
        if result.error is not None or result.reward is None or expectation_issues
        else "passed",
        "error": result.error,
        "verifier_error_category": result.verifier_error_category,
        "diagnostic_code": result.diagnostic_code,
        "artifact_hint": result.artifact_hint,
        "expectation_issues": expectation_issues,
    }
    record["sha256"] = _canonical_sha256(record)
    return record


def _report_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    failed = sum(1 for record in records if record.get("status") != "passed")
    rewards = [
        float(record["reward"])
        for record in records
        if isinstance(record.get("reward"), int | float)
        and not isinstance(record.get("reward"), bool)
    ]
    return {
        "total_runs": total,
        "passed_runs": total - failed,
        "failed_runs": failed,
        "flake_rate": (failed / total) if total else 0.0,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
    }


def _leaderboard_suitability(
    *,
    spec: LiveAcceptanceSpec,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _report_summary(records)
    total_runs = int(summary["total_runs"])
    failed_runs = int(summary["failed_runs"])
    flake_rate = float(summary["flake_rate"])
    passed_records = [
        record for record in records if record.get("status") == "passed"
    ]
    generated_types = {
        str(record.get("type"))
        for record in passed_records
        if record.get("source") == "calibration-report"
        and record.get("type") in _LEADERBOARD_CALIBRATION_TYPES
    }
    missing_generated_types = sorted(_LEADERBOARD_CALIBRATION_TYPES - generated_types)
    has_oracle = any(record.get("type") == "oracle" for record in passed_records)
    has_reference = any(record.get("type") == "reference" for record in passed_records)
    checks = {
        "has_live_runs": total_runs > 0,
        "all_runs_passed": total_runs > 0 and failed_runs == 0,
        "flake_rate_within_limit": flake_rate <= spec.leaderboard.max_flake_rate + 1e-9,
        "has_oracle_proof": has_oracle,
        "has_reference_proof": has_reference,
        "has_generated_calibration_coverage": not missing_generated_types,
    }
    issues: list[str] = []
    if not checks["has_live_runs"]:
        issues.append("requires at least one live run")
    if not checks["all_runs_passed"]:
        issues.append("requires all live runs to pass")
    if not checks["flake_rate_within_limit"]:
        issues.append(
            f"flake_rate {flake_rate:.6g} exceeds max_flake_rate "
            f"{spec.leaderboard.max_flake_rate:.6g}"
        )
    if not has_oracle:
        issues.append("requires a passed oracle live case")
    if not has_reference:
        issues.append("requires a passed reference live case")
    if missing_generated_types:
        issues.append(
            "missing generated calibration live case types: "
            + ", ".join(missing_generated_types)
        )
    return {
        "status": "suitable" if not issues else "insufficient",
        "required": spec.leaderboard.required,
        "max_flake_rate": spec.leaderboard.max_flake_rate,
        "required_generated_calibration_types": sorted(
            _LEADERBOARD_CALIBRATION_TYPES
        ),
        "observed_generated_calibration_types": sorted(generated_types),
        "checks": checks,
        "issues": issues,
    }


def _expectation_dict(expect: LiveAcceptanceExpectation) -> dict[str, Any]:
    return {
        "reward_min": expect.reward_min,
        "reward_max": expect.reward_max,
        "reward_range": list(expect.reward_range) if expect.reward_range else None,
        "reward_equals": expect.reward_equals,
        "flake_rate_max": expect.flake_rate_max,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def _case_summary(
    *,
    case: LiveAcceptanceCase,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    case_records = [record for record in records if record.get("case") == case.name]
    total = len(case_records)
    failed = sum(1 for record in case_records if record.get("status") != "passed")
    rewards = [
        float(record["reward"])
        for record in case_records
        if isinstance(record.get("reward"), int | float)
        and not isinstance(record.get("reward"), bool)
    ]
    flake_rate = (failed / total) if total else 0.0
    threshold = case.expect.flake_rate_max
    status = (
        "passed"
        if (threshold is not None and flake_rate <= threshold)
        or (threshold is None and failed == 0)
        else "failed"
    )
    return {
        "case": case.name,
        "type": case.case_type,
        "source": case.source,
        "total_runs": total,
        "passed_runs": total - failed,
        "failed_runs": failed,
        "flake_rate": flake_rate,
        "flake_rate_max": threshold,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "status": status,
    }


def _check_case_flake_expectation(
    *,
    case: LiveAcceptanceCase,
    records: list[dict[str, Any]],
) -> list[str]:
    threshold = case.expect.flake_rate_max
    if threshold is None:
        return []
    summary = _case_summary(case=case, records=records)
    flake_rate = summary["flake_rate"]
    if not isinstance(flake_rate, int | float):
        return [f"acceptance-live case {case.name!r} did not produce flake rate"]
    if float(flake_rate) - 1e-9 > threshold:
        issue = (
            f"acceptance-live case {case.name!r} flake_rate "
            f"{float(flake_rate):.6g} exceeds flake_rate_max {threshold:.6g}"
        )
        hint = _case_failure_hint(records)
        if hint:
            issue += f"; {hint}"
        return [issue]
    return []


def _case_failure_hint(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record.get("status") == "passed":
            continue
        error = record.get("error")
        if isinstance(error, str) and contains_verifier_dep_install_marker(error):
            return _DEP_INSTALL_FLAKE_HINT
    return None


def _check_reward_expectation(
    prefix: str,
    reward: float,
    expect: LiveAcceptanceExpectation,
) -> list[str]:
    issues: list[str] = []
    if expect.reward_min is not None and reward + 1e-9 < expect.reward_min:
        issues.append(
            f"{prefix} reward {reward:.6g} is below reward_min "
            f"{expect.reward_min:.6g}"
        )
    if expect.reward_max is not None and reward - 1e-9 > expect.reward_max:
        issues.append(
            f"{prefix} reward {reward:.6g} is above reward_max "
            f"{expect.reward_max:.6g}"
        )
    if expect.reward_range is not None:
        low, high = expect.reward_range
        if reward + 1e-9 < low or reward - 1e-9 > high:
            issues.append(
                f"{prefix} reward {reward:.6g} is outside reward_range "
                f"[{low:.6g}, {high:.6g}]"
            )
    if expect.reward_equals is not None and abs(reward - expect.reward_equals) > 1e-9:
        issues.append(
            f"{prefix} reward {reward:.6g} does not equal reward_equals "
            f"{expect.reward_equals:.6g}"
        )
    return issues


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


def _is_safe_sandbox_dir(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = PurePosixPath(value)
    return path.is_absolute() and path != PurePosixPath("/")


def _safe_relative_file_path(value: str) -> Path | None:
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath("."):
        return None
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return Path(path.as_posix())


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0


def _oracle_return_code(trajectory: list[dict]) -> int | None:
    for event in trajectory:
        if event.get("type") != "oracle":
            continue
        return_code = event.get("return_code")
        if isinstance(return_code, int) and not isinstance(return_code, bool):
            return return_code
    return None

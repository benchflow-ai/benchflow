"""Spec parsing and validation for acceptance-live evidence.

This module owns the acceptance-live validation plane: turning the declared
``benchflow.evidence.acceptance_live`` mapping (and any generated calibration
cases) into a :class:`LiveAcceptanceSpec`, plus the reward/probability and path
safety helpers that gate parsing. These functions touch none of the patched
rollout seams, so they extract cleanly while the orchestration plane stays in
the ``benchflow.task.acceptance_live`` façade, which re-exports every name here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from benchflow.task.acceptance_live_model import (
    _CASE_TYPES,
    _DEFAULT_RERUNS,
    _MAX_RERUNS,
    _WORKSPACE_SOURCE_CURRENT_WORKTREE,
    LiveAcceptanceCase,
    LiveAcceptanceCaseType,
    LiveAcceptanceExpectation,
    LiveAcceptanceLeaderboard,
    LiveAcceptanceSpec,
    LiveAcceptanceWorkspace,
)
from benchflow.task.task import Task


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
            "acceptance-live workspace.target must be an absolute non-root sandbox path"
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
    if (
        not isinstance(reruns, int)
        or isinstance(reruns, bool)
        or not (1 <= reruns <= _MAX_RERUNS)
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
        return [], [
            *issues,
            f"{source}.from report is missing: {report_rel.as_posix()}",
        ]
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
        issues.append(f"{source}.type must be no-op, known-bad, partial, or reference")
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
            issues.append(
                f"{prefix}.type=oracle requires executable file: {solve_path}"
            )

    reruns = mapping.get("reruns", _DEFAULT_RERUNS)
    if (
        not isinstance(reruns, int)
        or isinstance(reruns, bool)
        or not (1 <= reruns <= _MAX_RERUNS)
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
    reward_min = _optional_reward(
        mapping.get("reward_min"), f"{prefix}.reward_min", issues
    )
    reward_max = _optional_reward(
        mapping.get("reward_max"), f"{prefix}.reward_max", issues
    )
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


def _required_probability(
    value: object, source: str, issues: list[str]
) -> float | None:
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

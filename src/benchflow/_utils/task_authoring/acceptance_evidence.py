"""Static acceptance/calibration evidence validator for native task packages.

A schema/DSL checker over ``benchflow.evidence``: oracle runs, verifier
stability reports, review attestations, and calibration reports, plus the
pinned-artifact accounting that backs them. ``_check_live_acceptance_execution``
hands declared live cases to the sandbox/verifier boundary.
"""

from pathlib import Path
from typing import cast

from benchflow.task.document import TaskDocument

from ._evidence_paths import (
    TASK_DOCUMENT_FILE,
    _check_declared_evidence_file,
    _check_primary_evidence_pins,
    _load_declared_evidence_json,
    _number_value,
)


def _check_acceptance_evidence(task_dir: Path) -> list[str]:
    """Static acceptance/calibration evidence gate for native task packages."""

    task_md = task_dir / TASK_DOCUMENT_FILE
    try:
        document = TaskDocument.from_path(task_md)
    except Exception as e:
        return [f"acceptance validation cannot parse task.md: {e}"]

    evidence = document.benchflow.get("evidence")
    if not isinstance(evidence, dict):
        return ["acceptance validation requires benchflow.evidence mapping"]
    evidence_mapping = cast(dict[str, object], evidence)

    issues: list[str] = []
    issues.extend(
        _check_oracle_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_verifier_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_review_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_calibration_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(_check_listed_evidence_artifacts(evidence_mapping, task_dir=task_dir))
    issues.extend(_check_primary_evidence_pins(evidence_mapping))
    return issues


def _check_live_acceptance_execution(
    task_dir: Path,
    *,
    sandbox_type: str | None,
    report_output: Path | None,
    write_report: bool = True,
) -> list[str]:
    """Run declared live acceptance cases through the sandbox/verifier boundary."""

    if sandbox_type is None:
        return [
            "acceptance-live validation requires --sandbox <backend> to execute "
            "oracle and calibration checks"
        ]

    task_md = task_dir / TASK_DOCUMENT_FILE
    try:
        document = TaskDocument.from_path(task_md)
    except Exception as e:
        return [f"acceptance-live validation cannot parse task.md: {e}"]
    evidence = document.benchflow.get("evidence")
    if not isinstance(evidence, dict):
        return ["acceptance-live validation requires benchflow.evidence mapping"]
    # Resolve through the package façade so test monkeypatches against
    # ``task_authoring.run_live_acceptance_checks`` take effect, matching the
    # pre-split module-global lookup.
    from benchflow._utils import task_authoring as _facade

    return _facade.run_live_acceptance_checks(
        task_dir,
        sandbox_type=sandbox_type,
        evidence=cast(dict[str, object], evidence),
        report_output=report_output,
        write_report=write_report,
    )


def _check_oracle_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    oracle_runs = evidence.get("oracle_runs")
    if not isinstance(oracle_runs, dict):
        return ["acceptance validation requires benchflow.evidence.oracle_runs"]
    oracle_mapping = cast(dict[str, object], oracle_runs)
    issues: list[str] = []
    reward = oracle_mapping.get("required_reward")
    reward_value = _number_value(reward)
    if reward_value is None or reward_value < 0.99:
        issues.append(
            "acceptance oracle_runs.required_reward must be numeric and >= 0.99"
        )
    if not any(
        isinstance(oracle_mapping.get(key), str)
        and str(oracle_mapping.get(key)).strip()
        for key in ("last_job", "artifact")
    ):
        issues.append(
            "acceptance oracle_runs must include last_job or artifact evidence"
        )
    artifact = oracle_mapping.get("artifact")
    if artifact is not None:
        issues.extend(
            _check_declared_evidence_file(
                artifact,
                task_dir=task_dir,
                source="acceptance oracle_runs.artifact",
            )
        )
        if reward_value is not None and reward_value >= 0.99:
            issues.extend(
                _check_oracle_run_artifact(
                    artifact,
                    task_dir=task_dir,
                    required_reward=reward_value,
                )
            )
    return issues


def _check_oracle_run_artifact(
    value: object,
    *,
    task_dir: Path,
    required_reward: float,
) -> list[str]:
    source = "acceptance oracle_runs.artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    issues = []
    reward = _number_value(artifact.get("reward"))
    if reward is None:
        reward = _number_value(artifact.get("expected_reward"))
    if reward is None or reward < required_reward:
        issues.append(f"{source}.reward must be >= oracle_runs.required_reward")
    status = artifact.get("status")
    if status is not None and status != "passed":
        issues.append(f"{source}.status must be passed when present")
    return issues


def _check_verifier_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    verifier = evidence.get("verifier")
    if not isinstance(verifier, dict):
        return ["acceptance validation requires benchflow.evidence.verifier"]
    verifier_mapping = cast(dict[str, object], verifier)
    issues: list[str] = []
    reruns = verifier_mapping.get("reruns")
    if not isinstance(reruns, int) or isinstance(reruns, bool) or reruns < 3:
        issues.append("acceptance verifier.reruns must be an integer >= 3")
    flake_rate = verifier_mapping.get("flake_rate")
    flake_rate_value = _number_value(flake_rate)
    if flake_rate_value is None or not 0.0 <= flake_rate_value <= 0.05:
        issues.append("acceptance verifier.flake_rate must be numeric and <= 0.05")
    report = verifier_mapping.get("report")
    issues.extend(
        _check_declared_evidence_file(
            report,
            task_dir=task_dir,
            source="acceptance verifier.report",
        )
    )
    if (
        isinstance(reruns, int)
        and not isinstance(reruns, bool)
        and reruns >= 3
        and flake_rate_value is not None
        and 0.0 <= flake_rate_value <= 0.05
    ):
        issues.extend(
            _check_verifier_stability_report(
                report,
                task_dir=task_dir,
                declared_reruns=reruns,
                declared_flake_rate=flake_rate_value,
            )
        )
    return issues


def _check_verifier_stability_report(
    value: object,
    *,
    task_dir: Path,
    declared_reruns: int,
    declared_flake_rate: float,
) -> list[str]:
    source = "acceptance verifier.report"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    report = cast(dict[str, object], data)

    issues = []
    if report.get("kind") != "verifier-stability-report":
        issues.append(f"{source}.kind must be verifier-stability-report")

    report_reruns = report.get("reruns")
    if (
        not isinstance(report_reruns, int)
        or isinstance(report_reruns, bool)
        or report_reruns < declared_reruns
    ):
        issues.append(f"{source}.reruns must be an integer >= declared verifier.reruns")

    min_reward = _number_value(report.get("min_reward"))
    if min_reward is None or not 0.0 <= min_reward <= 1.0:
        issues.append(f"{source}.min_reward must be numeric within 0..1")

    report_flake_rate = _number_value(report.get("flake_rate"))
    if report_flake_rate is None or not 0.0 <= report_flake_rate <= declared_flake_rate:
        issues.append(
            f"{source}.flake_rate must be numeric and <= declared verifier.flake_rate"
        )

    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        issues.append(f"{source}.runs must be a non-empty list")
        return issues
    if (
        report_reruns is not None
        and isinstance(report_reruns, int)
        and len(runs) != report_reruns
    ):
        issues.append(f"{source}.runs length must match report reruns")
    if len(runs) < declared_reruns:
        issues.append(f"{source}.runs must include at least declared verifier.reruns")

    failures = 0
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            issues.append(f"{source}.runs[{index}] must be a mapping")
            failures += 1
            continue
        run_mapping = cast(dict[str, object], run)
        status = run_mapping.get("status")
        if status not in {"passed", "failed"}:
            issues.append(f"{source}.runs[{index}].status must be passed or failed")
            failures += 1
        reward = _number_value(run_mapping.get("reward"))
        if reward is None or not 0.0 <= reward <= 1.0:
            issues.append(f"{source}.runs[{index}].reward must be numeric within 0..1")
            failures += 1
        elif min_reward is not None and status == "passed" and reward < min_reward:
            issues.append(f"{source}.runs[{index}].reward is below min_reward")
            failures += 1
        elif status == "failed":
            failures += 1

    computed_flake_rate = failures / len(runs)
    if computed_flake_rate > declared_flake_rate:
        issues.append(
            f"{source}.runs imply flake_rate {computed_flake_rate:.3f}, "
            "above declared verifier.flake_rate"
        )
    if report_flake_rate is not None and report_flake_rate + 1e-9 < computed_flake_rate:
        issues.append(f"{source}.flake_rate is lower than failures in runs")
    return issues


def _check_review_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    review = evidence.get("review")
    if not isinstance(review, dict):
        return ["acceptance validation requires benchflow.evidence.review"]
    review_mapping = cast(dict[str, object], review)
    issues: list[str] = []
    for key in ("anti_cheat", "instruction_alignment"):
        if review_mapping.get(key) != "passed":
            issues.append(f"acceptance review.{key} must be passed")
    artifact = review_mapping.get("artifact")
    if artifact is None:
        issues.append("acceptance review.artifact must be declared")
    else:
        issues.extend(
            _check_review_artifact(
                artifact,
                task_dir=task_dir,
                declared_review=review_mapping,
            )
        )
    return issues


def _check_review_artifact(
    value: object,
    *,
    task_dir: Path,
    declared_review: dict[str, object],
) -> list[str]:
    source = "acceptance review.artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    issues = []
    if artifact.get("kind") not in {None, "acceptance-review"}:
        issues.append(f"{source}.kind must be acceptance-review when present")
    for key in ("anti_cheat", "instruction_alignment"):
        if artifact.get(key) != declared_review.get(key):
            issues.append(f"{source}.{key} must match benchflow.evidence.review.{key}")
    return issues


def _check_calibration_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    calibration = evidence.get("calibration")
    if not isinstance(calibration, dict):
        return ["acceptance validation requires benchflow.evidence.calibration"]
    calibration_mapping = cast(dict[str, object], calibration)
    issues: list[str] = []
    no_op = calibration_mapping.get("no_op_reward_max")
    no_op_value = _number_value(no_op)
    if no_op_value is None or not 0.0 <= no_op_value <= 0.1:
        issues.append(
            "acceptance calibration.no_op_reward_max must be numeric and <= 0.1"
        )
    known_bad = calibration_mapping.get("known_bad_reward_max")
    known_bad_value = _number_value(known_bad)
    if known_bad_value is None or not 0.0 <= known_bad_value < 1.0:
        issues.append(
            "acceptance calibration.known_bad_reward_max must be numeric and < 1.0"
        )
    partial_range = calibration_mapping.get("partial_solution_range")
    partial_min: float | None = None
    partial_max: float | None = None
    if isinstance(partial_range, list) and len(partial_range) == 2:
        partial_min = _number_value(partial_range[0])
        partial_max = _number_value(partial_range[1])
    if (
        partial_min is None
        or partial_max is None
        or not (0.0 <= partial_min <= partial_max <= 1.0)
    ):
        issues.append(
            "acceptance calibration.partial_solution_range must be [min, max] "
            "within 0..1"
        )

    expected_rewards: list[float] = []
    examples = calibration_mapping.get("human_or_reference_examples")
    if not isinstance(examples, list) or not examples:
        issues.append(
            "acceptance calibration.human_or_reference_examples must be non-empty"
        )
    else:
        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                issues.append(
                    f"acceptance calibration.human_or_reference_examples[{index}] "
                    "must be a mapping"
                )
                continue
            example_mapping = cast(dict[str, object], example)
            expected = example_mapping.get("expected_reward")
            expected_value = _number_value(expected)
            if expected_value is None or not 0.0 <= expected_value <= 1.0:
                issues.append(
                    f"acceptance calibration.human_or_reference_examples[{index}] "
                    "expected_reward must be numeric within 0..1"
                )
            else:
                expected_rewards.append(expected_value)
            artifact = example_mapping.get("artifact")
            issues.extend(
                _check_declared_evidence_file(
                    artifact,
                    task_dir=task_dir,
                    source=(
                        "acceptance calibration.human_or_reference_examples"
                        f"[{index}].artifact"
                    ),
                )
            )
            if expected_value is not None and 0.0 <= expected_value <= 1.0:
                issues.extend(
                    _check_calibration_example_artifact(
                        artifact,
                        task_dir=task_dir,
                        expected_reward=expected_value,
                        index=index,
                    )
                )

    report = calibration_mapping.get("report")
    if report is None:
        issues.append("acceptance calibration.report must be declared")
    elif (
        no_op_value is not None
        and known_bad_value is not None
        and partial_min is not None
        and partial_max is not None
    ):
        issues.extend(
            _check_calibration_report(
                report,
                task_dir=task_dir,
                no_op_reward_max=no_op_value,
                known_bad_reward_max=known_bad_value,
                partial_min=partial_min,
                partial_max=partial_max,
                expected_rewards=expected_rewards,
            )
        )
    return issues


def _check_calibration_example_artifact(
    value: object,
    *,
    task_dir: Path,
    expected_reward: float,
    index: int,
) -> list[str]:
    source = f"acceptance calibration.human_or_reference_examples[{index}].artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    reward = _number_value(artifact.get("expected_reward"))
    if reward is None:
        reward = _number_value(artifact.get("reward"))
    if reward is None:
        return [f"{source}.expected_reward must be numeric"]
    if abs(reward - expected_reward) > 1e-9:
        return [f"{source}.expected_reward must match declared expected_reward"]
    return []


def _check_calibration_report(
    value: object,
    *,
    task_dir: Path,
    no_op_reward_max: float,
    known_bad_reward_max: float,
    partial_min: float,
    partial_max: float,
    expected_rewards: list[float],
) -> list[str]:
    source = "acceptance calibration.report"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    report = cast(dict[str, object], data)

    issues = []
    if report.get("kind") != "calibration-report":
        issues.append(f"{source}.kind must be calibration-report")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        issues.append(f"{source}.cases must be a non-empty list")
        return issues

    has_no_op = False
    has_known_bad = False
    has_partial = False
    has_reference = False
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            issues.append(f"{source}.cases[{index}] must be a mapping")
            continue
        case_mapping = cast(dict[str, object], case)
        case_type = case_mapping.get("type")
        if case_type not in {"no-op", "known-bad", "partial", "reference"}:
            issues.append(
                f"{source}.cases[{index}].type must be no-op, known-bad, "
                "partial, or reference"
            )
            continue
        reward = _number_value(case_mapping.get("reward"))
        if reward is None or not 0.0 <= reward <= 1.0:
            issues.append(f"{source}.cases[{index}].reward must be numeric within 0..1")
            continue
        if case_type == "no-op":
            has_no_op = True
            if reward > no_op_reward_max:
                issues.append(f"{source}.cases[{index}] exceeds no_op_reward_max")
        elif case_type == "known-bad":
            has_known_bad = True
            if reward > known_bad_reward_max:
                issues.append(f"{source}.cases[{index}] exceeds known_bad_reward_max")
        elif case_type == "partial":
            has_partial = True
            if not partial_min <= reward <= partial_max:
                issues.append(f"{source}.cases[{index}] is outside partial range")
        elif case_type == "reference":
            has_reference = True
            if expected_rewards and all(
                abs(reward - expected) > 1e-9 for expected in expected_rewards
            ):
                issues.append(
                    f"{source}.cases[{index}] does not match a declared "
                    "reference expected_reward"
                )

    required_cases = [
        ("no-op", has_no_op),
        ("known-bad", has_known_bad),
        ("partial", has_partial),
        ("reference", has_reference),
    ]
    for case_type, present in required_cases:
        if not present:
            issues.append(f"{source}.cases must include a {case_type} case")
    return issues


def _check_listed_evidence_artifacts(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    issues: list[str] = []
    for list_key in ("trajectories", "artifacts"):
        items = evidence.get(list_key)
        if items is None:
            continue
        if not isinstance(items, list):
            issues.append(f"acceptance evidence.{list_key} must be a list")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                issues.append(
                    f"acceptance evidence.{list_key}[{index}] must be a mapping"
                )
                continue
            item_mapping = cast(dict[str, object], item)
            expected_sha256 = item_mapping.get("sha256")
            if not isinstance(expected_sha256, str) or not expected_sha256.strip():
                issues.append(
                    f"acceptance evidence.{list_key}[{index}].sha256 must be declared"
                )
            issues.extend(
                _check_declared_evidence_file(
                    item_mapping.get("path"),
                    task_dir=task_dir,
                    source=f"acceptance evidence.{list_key}[{index}].path",
                    expected_sha256=expected_sha256,
                )
            )
    return issues

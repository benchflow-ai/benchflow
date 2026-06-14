"""Evidence builder for original-vs-BenchFlow environment-adapter parity.

The adoption loop for browser/computer-use benchmarks needs a concrete artifact
that the existing ``bench agent verify`` gate can score. This module turns a
single original-runner result plus the matching BenchFlow ``result.json`` and
trace artifact into the ``parity_experiment.json`` schema that verifier already
understands.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ADOPTION_LOOP_ROLES = (
    "scout",
    "builder",
    "original-runner",
    "benchflow-runner",
    "verifier",
    "auditor",
    "reviewer",
    "queue",
)


def build_environment_adapter_parity_experiment(
    *,
    benchmark: str,
    original: Mapping[str, Any],
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
    task_id: str | None = None,
    agent: str | None = None,
    sandbox: str | None = None,
    environment_adapter: str | None = None,
    benchmark_adapter: str | None = None,
    artifact_manifest: Sequence[Mapping[str, Any]] | None = None,
    unsupported: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a verifier-readable parity experiment for one adapter smoke slice."""

    resolved_task_id = task_id or str(
        original.get("task_id") or benchflow.get("task_name") or summary.get("task_id")
    )
    original_reward = _reward_from_original(original, summary)
    benchflow_reward = _reward_from_benchflow(benchflow, summary)
    criteria = [
        _criterion(
            "final-result",
            "Final result matches the original runner",
            original.get("final_result") is not None,
            original.get("final_result") == artifact.get("final_result"),
            _detail(original.get("final_result")),
            _detail(artifact.get("final_result")),
        ),
        _criterion(
            "trace-completeness",
            "Original and BenchFlow both recorded non-empty trajectories",
            _original_trace_complete(original, summary),
            _benchflow_trace_complete(benchflow, artifact, summary),
            _trace_detail(original.get("num_steps"), summary.get("original")),
            _trace_detail(
                _nested_int(summary, "benchflow", "trajectory_steps"),
                summary.get("benchflow"),
            ),
        ),
        _criterion(
            "artifact-shape",
            "BenchFlow artifact preserves the expected trace/screenshot shape",
            _original_artifact_shape_complete(original, summary),
            _benchflow_artifact_shape_complete(original, artifact, summary),
            _artifact_detail(original, summary.get("original")),
            _artifact_detail(artifact, summary.get("benchflow")),
        ),
        _criterion(
            "timing-recorded",
            "Both runners recorded timing metadata",
            _duration_present(original.get("duration_sec")),
            _duration_present(_benchflow_elapsed_sec(benchflow)),
            _detail(original.get("duration_sec")),
            _detail(_benchflow_elapsed_sec(benchflow)),
        ),
    ]
    benchflow_eval = summary.get("benchflow_eval")
    if isinstance(benchflow_eval, Mapping):
        criteria.append(
            _criterion(
                "eval-run-summary",
                "BenchFlow eval run report is complete and trace-auditable",
                True,
                _eval_run_summary_complete(benchflow_eval),
                "eval run report required",
                _eval_run_summary_detail(benchflow_eval),
            )
        )
    criteria.append(
        _criterion(
            "cleanup",
            "This run's provider resources were all cleaned up (no per-run leak)",
            True,
            _cleanup_complete(summary.get("cleanup")),
            "no resources leaked by this run",
            _cleanup_detail(summary.get("cleanup")),
        )
    )
    artifact_manifest_report = None
    if artifact_manifest is not None:
        artifact_manifest_report = _evaluate_artifact_manifest(
            artifact_manifest,
            original=original,
            benchflow=benchflow,
            artifact=artifact,
            summary=summary,
        )
        criteria.append(
            _criterion(
                "artifact-manifest",
                "BenchFlow produced every artifact required by the adapter",
                True,
                bool(artifact_manifest_report["ok"]),
                "manifest declared",
                _detail(artifact_manifest_report),
            )
        )
    if unsupported is not None:
        criteria.append(
            _criterion(
                "unsupported-reporting",
                "Unsupported tasks carry structured actionable reasons",
                True,
                _unsupported_report_complete(unsupported),
                "structured unsupported report",
                _detail(list(unsupported)),
            )
        )

    rewards_present = original_reward is not None and benchflow_reward is not None
    result: dict[str, Any] = {
        "experiment": "environment-adapter-side-by-side-parity",
        "benchmark": benchmark,
        "status": (
            "parity-confirmed"
            if all(bool(item["agreement"]) for item in criteria) and rewards_present
            else "parity-recorded"
        ),
        "task_id": resolved_task_id,
        "adapter_parity": {
            "agent": agent or benchflow.get("agent"),
            "sandbox": sandbox,
            "environment_adapter": environment_adapter,
            "benchmark_adapter": benchmark_adapter,
            "original_framework": original.get("framework"),
            "benchflow_agent": benchflow.get("agent"),
            "timing": {
                "original_duration_sec": original.get("duration_sec"),
                "benchflow_elapsed_sec": _benchflow_elapsed_sec(benchflow),
                "benchflow_timing": benchflow.get("timing"),
            },
            "summary": dict(summary),
        },
        "conversion_parity": {
            "tasks": [
                {
                    "task_id": resolved_task_id,
                    "criteria_results": criteria,
                }
            ]
        },
    }
    if artifact_manifest_report is not None:
        result["adapter_parity"]["artifact_manifest"] = artifact_manifest_report
    if original_reward is not None and benchflow_reward is not None:
        delta = abs(float(benchflow_reward) - float(original_reward))
        result["agent_parity"] = {
            "results": [
                {
                    "task_id": resolved_task_id,
                    "legacy_reward": float(original_reward),
                    "converted_reward": float(benchflow_reward),
                    "reward_delta": delta,
                    "original": {
                        "framework": original.get("framework"),
                        "score": float(original_reward),
                    },
                    "benchflow": {
                        "agent": benchflow.get("agent"),
                        "reward": float(benchflow_reward),
                    },
                }
            ]
        }
    return result


def write_environment_adapter_parity_experiment(
    path: Path,
    *,
    benchmark: str,
    original: Mapping[str, Any],
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
    task_id: str | None = None,
    agent: str | None = None,
    sandbox: str | None = None,
    environment_adapter: str | None = None,
    benchmark_adapter: str | None = None,
    artifact_manifest: Sequence[Mapping[str, Any]] | None = None,
    unsupported: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write verifier-readable adapter parity evidence to ``path``."""

    evidence = build_environment_adapter_parity_experiment(
        benchmark=benchmark,
        original=original,
        benchflow=benchflow,
        artifact=artifact,
        summary=summary,
        task_id=task_id,
        agent=agent,
        sandbox=sandbox,
        environment_adapter=environment_adapter,
        benchmark_adapter=benchmark_adapter,
        artifact_manifest=artifact_manifest,
        unsupported=unsupported,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def build_environment_adapter_adoption_report(
    *,
    parity_experiment: Mapping[str, Any],
    original: Mapping[str, Any],
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
    parity_experiment_path: str | Path | None = None,
    unsupported: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a scrubbed, reviewable report for one adapter-adoption loop.

    ``parity_experiment.json`` is the verifier contract. This report is the
    companion human/CI manifest: it records which architecture planes were used,
    whether parity criteria agreed, what artifacts existed, and whether cleanup
    was empty without copying screenshot payloads or raw trajectories into the
    durable evidence bundle.
    """

    adapter_parity = _mapping_value(parity_experiment.get("adapter_parity"))
    criteria = _criteria_results(parity_experiment)
    reward_sample = _first_reward_sample(parity_experiment)
    benchflow_eval = summary.get("benchflow_eval")
    artifact_manifest = _mapping_value(adapter_parity.get("artifact_manifest"))
    unsupported_reports = (
        [_safe_manifest_mapping(item) for item in unsupported]
        if unsupported is not None
        else None
    )

    report: dict[str, Any] = {
        "schema": "benchflow.environment-adapter-adoption-report.v1",
        "status": parity_experiment.get("status"),
        "benchmark": parity_experiment.get("benchmark"),
        "task_id": parity_experiment.get("task_id"),
        "planes": {
            "sandbox_provider": adapter_parity.get("sandbox"),
            "sandbox_provider_mode": _sandbox_provider_mode(
                adapter_parity.get("sandbox"), summary
            ),
            "environment_adapter": adapter_parity.get("environment_adapter"),
            "agent_adapter": adapter_parity.get("agent"),
            "benchmark_adapter": adapter_parity.get("benchmark_adapter"),
        },
        "parity": {
            "parity_experiment": (
                str(parity_experiment_path) if parity_experiment_path else None
            ),
            "criteria_compared": len(criteria),
            "criteria_agreed": sum(1 for item in criteria if item.get("agreement")),
            "reward_delta": (
                reward_sample.get("reward_delta") if reward_sample else None
            ),
        },
        "artifact_index": [
            _original_runner_artifact(original),
            _benchflow_result_artifact(benchflow),
            _benchflow_trace_artifact(artifact),
            _benchflow_eval_artifact(benchflow_eval),
        ],
        "cleanup": _safe_manifest_value(summary.get("cleanup")),
    }
    if artifact_manifest:
        report["artifact_requirements"] = artifact_manifest
    if unsupported_reports is not None:
        report["unsupported_reports"] = unsupported_reports
    return report


def write_environment_adapter_adoption_report(
    path: Path,
    *,
    parity_experiment: Mapping[str, Any],
    original: Mapping[str, Any],
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
    parity_experiment_path: str | Path | None = None,
    unsupported: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a scrubbed adapter-adoption report to ``path``."""

    report = build_environment_adapter_adoption_report(
        parity_experiment=parity_experiment,
        original=original,
        benchflow=benchflow,
        artifact=artifact,
        summary=summary,
        parity_experiment_path=parity_experiment_path,
        unsupported=unsupported,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_environment_adapter_loop_state(
    *,
    parity_experiment: Mapping[str, Any],
    adoption_report: Mapping[str, Any],
    commands: Sequence[str],
    artifacts: Mapping[str, str | Path] | None = None,
    source: Mapping[str, Any] | None = None,
    roles: Sequence[Mapping[str, Any]] | None = None,
    queue: Sequence[Mapping[str, Any] | str] | None = None,
    unsupported_summary: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build durable state for one adapter-adoption controller loop.

    ``parity_experiment.json`` is the scorer contract and
    ``adoption_report.json`` is the scrubbed evidence contract. This state file
    is the resumable flight recorder: it records the exact replay commands,
    artifact locations, role progress, cleanup summary, unsupported summary,
    and queued next chunks without copying raw traces or screenshots.
    """

    role_records = (
        [_safe_role(item) for item in roles]
        if roles is not None
        else _default_loop_roles(
            parity_experiment=parity_experiment,
            adoption_report=adoption_report,
            source=source,
            queue=queue,
        )
    )
    state: dict[str, Any] = {
        "schema": "benchflow.adapter-adoption-loop-state.v1",
        "status": _loop_status(role_records),
        "benchmark": adoption_report.get("benchmark")
        or parity_experiment.get("benchmark"),
        "task_id": adoption_report.get("task_id") or parity_experiment.get("task_id"),
        "planes": dict(_mapping_value(adoption_report.get("planes"))),
        "source": _safe_loop_value(source or {}),
        "commands": [_redact_command(str(command)) for command in commands],
        "artifacts": _loop_artifacts(artifacts, adoption_report),
        "roles": role_records,
        "checks": {
            "parity": _safe_loop_value(adoption_report.get("parity")),
            "cleanup": _safe_loop_value(
                _nested(parity_experiment, "adapter_parity", "summary", "cleanup")
                or adoption_report.get("cleanup")
            ),
            "artifact_requirements": _safe_loop_value(
                adoption_report.get("artifact_requirements")
            ),
        },
        "unsupported_summary": _safe_loop_value(
            unsupported_summary or _summarize_unsupported_reports(adoption_report)
        ),
        "queue": [_safe_loop_value(item) for item in (queue or [])],
    }
    if updated_at:
        state["updated_at"] = updated_at
    return state


def write_environment_adapter_loop_state(
    path: Path,
    *,
    parity_experiment: Mapping[str, Any],
    adoption_report: Mapping[str, Any],
    commands: Sequence[str],
    artifacts: Mapping[str, str | Path] | None = None,
    source: Mapping[str, Any] | None = None,
    roles: Sequence[Mapping[str, Any]] | None = None,
    queue: Sequence[Mapping[str, Any] | str] | None = None,
    unsupported_summary: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Write a resumable adapter-adoption loop state file."""

    state = build_environment_adapter_loop_state(
        parity_experiment=parity_experiment,
        adoption_report=adoption_report,
        commands=commands,
        artifacts=artifacts,
        source=source,
        roles=roles,
        queue=queue,
        unsupported_summary=unsupported_summary,
        updated_at=updated_at,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def validate_environment_adapter_loop_state(
    state: Mapping[str, Any],
    *,
    require_review: bool = False,
) -> list[str]:
    """Return issues that keep an adapter-adoption loop state from scaling."""

    issues: list[str] = []
    if state.get("schema") != "benchflow.adapter-adoption-loop-state.v1":
        issues.append("loop state has unexpected schema")
    if not state.get("benchmark"):
        issues.append("loop state missing benchmark")
    if not state.get("task_id"):
        issues.append("loop state missing task_id")

    planes = _mapping_value(state.get("planes"))
    for key in (
        "sandbox_provider",
        "environment_adapter",
        "agent_adapter",
        "benchmark_adapter",
    ):
        if not planes.get(key):
            issues.append(f"loop state planes missing {key}")

    commands = state.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        issues.append("loop state commands must be a list")
    elif not commands:
        issues.append("loop state commands missing replay commands")
    else:
        for index, command in enumerate(commands):
            if not isinstance(command, str) or not command.strip():
                issues.append(f"loop state command {index} is empty")
            elif _contains_secret_like_text(command):
                issues.append(f"loop state command {index} appears to contain a secret")

    artifacts = _mapping_value(state.get("artifacts"))
    for key in ("parity_experiment", "adoption_report"):
        if not artifacts.get(key):
            issues.append(f"loop state artifacts missing {key}")

    roles = _role_by_name(state.get("roles"))
    for role in ADOPTION_LOOP_ROLES:
        if role not in roles:
            issues.append(f"loop state missing {role} role")
    if (
        require_review
        and _mapping_value(roles.get("reviewer")).get("status") != "passed"
    ):
        issues.append("loop state reviewer role has not passed")
    if (
        state.get("status") == "scale-ready"
        and _mapping_value(roles.get("reviewer")).get("status") != "passed"
    ):
        issues.append("loop state cannot be scale-ready before reviewer passes")

    checks = _mapping_value(state.get("checks"))
    parity = _mapping_value(checks.get("parity"))
    compared = _coerce_float(parity.get("criteria_compared"))
    agreed = _coerce_float(parity.get("criteria_agreed"))
    if compared is None or compared <= 0:
        issues.append("loop state parity has no compared criteria")
    elif agreed != compared:
        issues.append("loop state parity criteria did not all agree")
    cleanup = checks.get("cleanup")
    if isinstance(cleanup, Mapping) and not _loop_cleanup_zero(cleanup):
        issues.append("loop state cleanup is not empty")
    if _contains_secret_like_text(json.dumps(state, default=str)):
        issues.append("loop state appears to contain a secret")
    return issues


def _criterion(
    criterion_id: str,
    title: str,
    original_ok: bool,
    adapted_ok: bool,
    original_detail: str,
    adapted_detail: str,
) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "criterion_title": title,
        "original_verdict": "pass" if original_ok else f"fail: {original_detail}",
        "adapted_verdict": "pass" if adapted_ok else f"fail: {adapted_detail}",
        "agreement": bool(original_ok and adapted_ok),
    }


def _reward_from_original(
    original: Mapping[str, Any], summary: Mapping[str, Any]
) -> float | None:
    direct = _coerce_float(original.get("score"))
    if direct is not None:
        return direct
    return _coerce_float(_nested(summary, "original", "score"))


def _reward_from_benchflow(
    benchflow: Mapping[str, Any], summary: Mapping[str, Any]
) -> float | None:
    rewards = benchflow.get("rewards")
    if isinstance(rewards, Mapping):
        reward = _coerce_float(rewards.get("reward"))
        if reward is not None:
            return reward
    return _coerce_float(_nested(summary, "benchflow", "reward"))


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _nested_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    value = _nested(mapping, *keys)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _detail(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _trace_detail(value: Any, summary: Any) -> str:
    return _detail({"steps": value, "summary": summary})


def _artifact_detail(payload: Mapping[str, Any], summary: Any) -> str:
    screenshots = payload.get("screenshots_b64")
    return _detail(
        {
            "steps": _sequence_len(payload.get("steps")),
            "screenshots_b64": _sequence_len(screenshots),
            "has_screenshot_key": "screenshots_b64" in payload,
            "summary": summary,
        }
    )


def _sequence_len(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _original_trace_complete(
    original: Mapping[str, Any], summary: Mapping[str, Any]
) -> bool:
    return _positive_int(original.get("num_steps")) or _positive_int(
        _nested(summary, "original", "num_steps")
    )


def _benchflow_trace_complete(
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    trajectory = benchflow.get("trajectory_summary")
    trajectory_steps = _nested_int(summary, "benchflow", "trajectory_steps")
    if trajectory_steps is None and isinstance(trajectory, Mapping):
        trajectory_steps = _as_int(trajectory.get("steps"))
    artifact_steps = _nested_int(summary, "benchflow", "artifact_steps")
    if artifact_steps is None:
        artifact_steps = _sequence_len(artifact.get("steps"))
    return _positive_int(trajectory_steps) and _positive_int(artifact_steps)


def _original_artifact_shape_complete(
    original: Mapping[str, Any], summary: Mapping[str, Any]
) -> bool:
    return (
        "screenshots_b64" in original
        or _nested_int(summary, "original", "screenshots_b64") is not None
    )


def _benchflow_artifact_shape_complete(
    original: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    original_screenshots = _sequence_len(original.get("screenshots_b64"))
    bench_screenshots = _sequence_len(artifact.get("screenshots_b64"))
    summary_bench_screenshots = _nested_int(summary, "benchflow", "screenshots_b64")
    if summary_bench_screenshots is not None:
        bench_screenshots = summary_bench_screenshots
    screenshot_ok = "screenshots_b64" in artifact
    if original_screenshots > 0:
        screenshot_ok = screenshot_ok and bench_screenshots > 0
    return screenshot_ok and _sequence_len(artifact.get("steps")) > 0


def _duration_present(value: Any) -> bool:
    duration = _coerce_float(value)
    return duration is not None and duration >= 0.0


def _benchflow_elapsed_sec(benchflow: Mapping[str, Any]) -> Any:
    direct = benchflow.get("elapsed_sec")
    if direct is not None:
        return direct
    timing = benchflow.get("timing")
    if isinstance(timing, Mapping):
        return timing.get("total")
    return None


def _cleanup_complete(value: Any) -> bool:
    """Cleanup is complete when THIS run leaked no tracked resources.

    The ``*containers``/``*networks`` counts in the cleanup summary are scoped
    to the resources the run itself created (present after the run but not
    before). They are therefore zero whenever the run cleaned up after itself,
    regardless of unrelated pre-existing or concurrently-running benchflow-owned
    resources. A non-zero count is a real per-run leak.
    """

    if not isinstance(value, Mapping):
        return False
    counts: list[int] = []
    for key, raw in value.items():
        if not (str(key).endswith("containers") or str(key).endswith("networks")):
            continue
        count = _as_int(raw)
        if count is None:
            return False
        counts.append(count)
    return bool(counts) and all(count == 0 for count in counts)


def _cleanup_detail(value: Any) -> str:
    return _detail(value if isinstance(value, Mapping) else {"cleanup": value})


def _eval_run_summary_complete(value: Mapping[str, Any]) -> bool:
    result = value.get("result")
    run_summary = value.get("summary")
    if not isinstance(result, Mapping) or not isinstance(run_summary, Mapping):
        return False
    result_total = _as_int(result.get("total"))
    summary_total = _as_int(run_summary.get("total"))
    result_errors = (_as_int(result.get("errored")) or 0) + (
        _as_int(result.get("verifier_errored")) or 0
    )
    summary_errors = (_as_int(run_summary.get("errored")) or 0) + (
        _as_int(run_summary.get("verifier_errored")) or 0
    )
    trajectory_steps = _as_int(run_summary.get("total_trajectory_steps"))
    return (
        value.get("status") == "completed"
        and value.get("ok") is True
        and bool(value.get("summary_path"))
        and result_total is not None
        and result_total > 0
        and summary_total == result_total
        and result_errors == 0
        and summary_errors == 0
        and _duration_present(result.get("elapsed_sec"))
        and _duration_present(run_summary.get("elapsed_sec"))
        and trajectory_steps is not None
        and trajectory_steps > 0
    )


def _eval_run_summary_detail(value: Mapping[str, Any]) -> str:
    result = value.get("result")
    run_summary = value.get("summary")
    return _detail(
        {
            "status": value.get("status"),
            "ok": value.get("ok"),
            "summary_path": value.get("summary_path"),
            "result": _safe_manifest_value(result),
            "summary": _safe_manifest_value(run_summary),
            "summary_total": (
                run_summary.get("total") if isinstance(run_summary, Mapping) else None
            ),
            "summary_errors": (
                {
                    "errored": run_summary.get("errored"),
                    "verifier_errored": run_summary.get("verifier_errored"),
                }
                if isinstance(run_summary, Mapping)
                else None
            ),
            "total_trajectory_steps": (
                run_summary.get("total_trajectory_steps")
                if isinstance(run_summary, Mapping)
                else None
            ),
            "elapsed_sec": (
                run_summary.get("elapsed_sec")
                if isinstance(run_summary, Mapping)
                else None
            ),
        }
    )


def _unsupported_report_complete(items: Sequence[Mapping[str, Any]]) -> bool:
    for item in items:
        has_task = bool(item.get("task_id") or item.get("task"))
        has_reason = bool(item.get("reason") or item.get("issue") or item.get("code"))
        if not (has_task and has_reason):
            return False
    return True


def _evaluate_artifact_manifest(
    requirements: Sequence[Mapping[str, Any]],
    *,
    original: Mapping[str, Any],
    benchflow: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    contexts: dict[str, Mapping[str, Any]] = {
        "original": original,
        "benchflow": benchflow,
        "artifact": artifact,
        "summary": summary,
    }
    evaluated = [
        _evaluate_artifact_requirement(requirement, contexts=contexts)
        for requirement in requirements
    ]
    return {
        "ok": bool(evaluated) and all(bool(item["ok"]) for item in evaluated),
        "requirements": evaluated,
    }


def _evaluate_artifact_requirement(
    requirement: Mapping[str, Any],
    *,
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = str(requirement.get("source") or "artifact")
    context = contexts.get(source, {})
    path = requirement.get("path") or requirement.get("id") or ""
    value, exists = _path_value(context, path)
    count = _sequence_len(value)
    checks: list[bool] = []

    if bool(requirement.get("required", True)):
        checks.append(exists and value not in (None, "", [], {}))

    if "exists" in requirement:
        checks.append(exists == bool(requirement.get("exists")))

    min_count = _as_int(requirement.get("min_count"))
    if min_count is not None:
        checks.append(count >= min_count)

    numeric_min = _coerce_float(requirement.get("numeric_min"))
    if numeric_min is not None:
        numeric_value = _coerce_float(value)
        checks.append(numeric_value is not None and numeric_value >= numeric_min)

    if "equals" in requirement:
        checks.append(value == requirement.get("equals"))
    if "contains" in requirement:
        needle = requirement.get("contains")
        checks.append(
            isinstance(value, str) and isinstance(needle, str) and needle in value
        )
    if "not_equals" in requirement:
        checks.append(value != requirement.get("not_equals"))

    return {
        "id": str(requirement.get("id") or _path_text(path) or "artifact"),
        "source": source,
        "path": _path_text(path),
        "kind": str(requirement.get("kind") or "field"),
        "exists": exists,
        "count": count,
        "ok": bool(checks) and all(checks),
        "value": _safe_manifest_value(value),
    }


def _path_value(root: Mapping[str, Any], path: Any) -> tuple[Any, bool]:
    parts = _path_parts(path)
    if not parts:
        return root, True
    value: Any = root
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return None, False
        value = value[part]
    return value, True


def _path_parts(path: Any) -> list[str]:
    if isinstance(path, str):
        return [part for part in path.split(".") if part]
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes)):
        return [str(part) for part in path]
    return []


def _path_text(path: Any) -> str:
    return ".".join(_path_parts(path))


def _safe_manifest_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > 120:
            return {"type": "str", "length": len(value)}
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {"type": "sequence", "count": len(value)}
    if isinstance(value, Mapping):
        return {"type": "mapping", "keys": sorted(str(key) for key in value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


def _mapping_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_manifest_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_manifest_value(item) for key, item in value.items()}


def _safe_loop_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_command(value) if _contains_secret_like_text(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_loop_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_safe_loop_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


def _safe_role(item: Mapping[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "").strip()
    status = str(item.get("status") or "pending").strip() or "pending"
    role: dict[str, Any] = {"name": name, "status": status}
    if item.get("artifact"):
        role["artifact"] = _safe_loop_value(item["artifact"])
    if item.get("notes"):
        role["notes"] = _safe_loop_value(item["notes"])
    return role


def _default_loop_roles(
    *,
    parity_experiment: Mapping[str, Any],
    adoption_report: Mapping[str, Any],
    source: Mapping[str, Any] | None,
    queue: Sequence[Mapping[str, Any] | str] | None,
) -> list[dict[str, Any]]:
    artifacts = _artifact_index_by_id(adoption_report.get("artifact_index"))
    parity = _mapping_value(adoption_report.get("parity"))
    requirements = _mapping_value(adoption_report.get("artifact_requirements"))
    cleanup = _nested(parity_experiment, "adapter_parity", "summary", "cleanup")
    parity_confirmed = parity_experiment.get("status") == "parity-confirmed"
    compared = _coerce_float(parity.get("criteria_compared")) or 0.0
    agreed = _coerce_float(parity.get("criteria_agreed"))
    artifact_ok = not requirements or requirements.get("ok") is True
    cleanup_ok = isinstance(cleanup, Mapping) and _loop_cleanup_zero(cleanup)

    return [
        {
            "name": "scout",
            "status": "passed" if source else "pending",
            "artifact": "source",
        },
        {
            "name": "builder",
            "status": (
                "passed" if _mapping_value(adoption_report.get("planes")) else "pending"
            ),
            "artifact": "adapter-diff",
        },
        {
            "name": "original-runner",
            "status": (
                "passed"
                if _original_runner_record_complete(artifacts.get("original-runner"))
                else "pending"
            ),
            "artifact": "original-runner",
        },
        {
            "name": "benchflow-runner",
            "status": (
                "passed"
                if _benchflow_runner_record_complete(artifacts.get("benchflow-result"))
                else "pending"
            ),
            "artifact": "benchflow-result",
        },
        {
            "name": "verifier",
            "status": (
                "passed"
                if parity_confirmed and compared > 0 and agreed == compared
                else "pending"
            ),
            "artifact": "parity_experiment",
        },
        {
            "name": "auditor",
            "status": "passed" if artifact_ok and cleanup_ok else "pending",
            "artifact": "adoption_report",
        },
        {
            "name": "reviewer",
            "status": "pending",
            "artifact": "review-report",
        },
        {
            "name": "queue",
            "status": "queued" if queue else "empty",
            "artifact": "next-chunks",
        },
    ]


def _loop_status(roles: Sequence[Mapping[str, Any]]) -> str:
    by_name = {str(item.get("name")): str(item.get("status")) for item in roles}
    parity_ready = all(
        by_name.get(role) == "passed"
        for role in (
            "builder",
            "original-runner",
            "benchflow-runner",
            "verifier",
            "auditor",
        )
    )
    if parity_ready and by_name.get("reviewer") == "passed":
        return "scale-ready"
    if parity_ready:
        return "review-ready"
    return "not-ready"


def _loop_artifacts(
    artifacts: Mapping[str, str | Path] | None,
    adoption_report: Mapping[str, Any],
) -> dict[str, str]:
    out = {str(key): str(value) for key, value in (artifacts or {}).items()}
    parity = _mapping_value(adoption_report.get("parity"))
    parity_path = parity.get("parity_experiment")
    if parity_path and "parity_experiment" not in out:
        out["parity_experiment"] = str(parity_path)
    return out


def _summarize_unsupported_reports(
    adoption_report: Mapping[str, Any],
) -> dict[str, Any]:
    reports = adoption_report.get("unsupported_reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        return {"count": 0, "issues": {}}
    issues: dict[str, int] = {}
    count = 0
    for item in reports:
        if not isinstance(item, Mapping):
            continue
        count += 1
        issue = str(
            item.get("reason") or item.get("issue") or item.get("code") or "unspecified"
        )
        issues[issue] = issues.get(issue, 0) + 1
    return {"count": count, "issues": issues}


def _role_by_name(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    roles: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and item.get("name"):
            roles[str(item["name"])] = item
    return roles


def _artifact_index_by_id(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and item.get("id"):
            out[str(item["id"])] = item
    return out


def _original_runner_record_complete(artifact: Mapping[str, Any] | None) -> bool:
    if not artifact:
        return False
    return (
        _coerce_float(artifact.get("score")) is not None
        and _positive_int(_as_int(artifact.get("trace_steps")))
        and artifact.get("error_present") is not True
    )


def _benchflow_runner_record_complete(artifact: Mapping[str, Any] | None) -> bool:
    if not artifact:
        return False
    return (
        _coerce_float(artifact.get("reward")) is not None
        and _positive_int(_as_int(artifact.get("trajectory_steps")))
        and _positive_int(_as_int(artifact.get("tool_calls")))
        and artifact.get("error_present") is not True
    )


def _loop_cleanup_zero(value: Mapping[str, Any]) -> bool:
    counts: list[int] = []
    for key, raw in value.items():
        key_text = str(key)
        if not (
            key_text.endswith("containers")
            or key_text.endswith("networks")
            or key_text.endswith("vms")
        ):
            continue
        count = _as_int(raw)
        if count is None:
            return False
        counts.append(count)
    return not counts or all(count == 0 for count in counts)


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<name>[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)="
    r"(?P<value>[^ \n]+)"
)
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|dtn_[A-Za-z0-9]{20,}|"
    r"sk_cua-api[0-9A-Za-z_-]*|AQ\.[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,})"
)


def _redact_command(command: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\g<name>=<redacted>", command)
    return _SECRET_VALUE_RE.sub("<redacted>", redacted)


def _contains_secret_like_text(value: str) -> bool:
    return bool(_SECRET_VALUE_RE.search(value))


def _criteria_results(evidence: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    conversion = _mapping_value(evidence.get("conversion_parity"))
    tasks = conversion.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        return []
    criteria: list[Mapping[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        items = task.get("criteria_results")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        criteria.extend(item for item in items if isinstance(item, Mapping))
    return criteria


def _first_reward_sample(evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    agent_parity = _mapping_value(evidence.get("agent_parity"))
    results = agent_parity.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        for item in results:
            if isinstance(item, Mapping):
                return item
    return None


def _sandbox_provider_mode(sandbox: Any, summary: Mapping[str, Any]) -> str | None:
    explicit = summary.get("sandbox_provider_mode")
    if isinstance(explicit, str) and explicit:
        return explicit
    if sandbox == "cua":
        cleanup = summary.get("cleanup")
        if isinstance(cleanup, Mapping) and "cloud_vms" in cleanup:
            return "cloud-probed"
        return "local"
    return None


def _original_runner_artifact(original: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": "original-runner",
        "kind": "runner-output",
        "framework": original.get("framework"),
        "score": _coerce_float(original.get("score")),
        "duration_sec": _coerce_float(original.get("duration_sec")),
        "trace_steps": _as_int(original.get("num_steps")),
        "screenshots_b64_count": _sequence_len(original.get("screenshots_b64")),
        "dimensions": _safe_manifest_value(original.get("dimensions")),
        "final_result_present": original.get("final_result") not in (None, ""),
        "error_present": original.get("error") not in (None, ""),
    }


def _benchflow_result_artifact(benchflow: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = _mapping_value(benchflow.get("trajectory_summary"))
    rewards = _mapping_value(benchflow.get("rewards"))
    return {
        "id": "benchflow-result",
        "kind": "result-json",
        "agent": benchflow.get("agent"),
        "reward": _coerce_float(rewards.get("reward")),
        "elapsed_sec": _coerce_float(_benchflow_elapsed_sec(benchflow)),
        "trajectory_steps": _as_int(trajectory.get("steps")),
        "tool_calls": _as_int(benchflow.get("n_tool_calls"))
        or _as_int(trajectory.get("tool_call_steps")),
        "timing_present": isinstance(benchflow.get("timing"), Mapping),
        "error_present": benchflow.get("error") not in (None, ""),
    }


def _benchflow_trace_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    environment = _mapping_value(artifact.get("environment"))
    readiness = _mapping_value(environment.get("readiness"))
    return {
        "id": "benchflow-agent-artifact",
        "kind": "agent-artifact",
        "schema": artifact.get("schema"),
        "framework": artifact.get("framework"),
        "trace_steps": _sequence_len(artifact.get("steps")),
        "screenshots_b64_count": _sequence_len(artifact.get("screenshots_b64")),
        "screenshot_method": artifact.get("screenshot_method"),
        "final_result_present": artifact.get("final_result") not in (None, ""),
        "environment_adapter": environment.get("adapter"),
        "environment_readiness": readiness.get("status"),
        "environment_content_sha256_present": bool(readiness.get("content_sha256")),
    }


def _benchflow_eval_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "id": "benchflow-eval-summary",
            "kind": "eval-summary",
            "present": False,
        }
    result = _mapping_value(value.get("result"))
    run_summary = _mapping_value(value.get("summary"))
    return {
        "id": "benchflow-eval-summary",
        "kind": "eval-summary",
        "present": True,
        "status": value.get("status"),
        "ok": value.get("ok"),
        "total": _as_int(result.get("total")),
        "errored": _as_int(result.get("errored")) or 0,
        "verifier_errored": _as_int(result.get("verifier_errored")) or 0,
        "summary_total": _as_int(run_summary.get("total")),
        "trajectory_steps": _as_int(run_summary.get("total_trajectory_steps")),
        "tool_calls": _as_int(run_summary.get("total_tool_calls")),
        "timing_recorded": _duration_present(result.get("elapsed_sec"))
        and _duration_present(run_summary.get("elapsed_sec")),
        "summary_path_present": bool(value.get("summary_path")),
    }


def _positive_int(value: Any) -> bool:
    number = _as_int(value)
    return number is not None and number > 0


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None

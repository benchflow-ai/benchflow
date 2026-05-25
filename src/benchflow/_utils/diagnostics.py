"""Shared structured diagnostic helpers for rollout result surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from benchflow._utils.scoring import (
    IDLE_TIMEOUT,
    INFRA_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    SANDBOX_SETUP,
    TIMED_OUT,
    VERIFIER_DEP_INSTALL,
    VERIFIER_TIMEOUT,
    classify_error,
    classify_verifier_error,
)

RESULT_DIAGNOSTIC_FIELDS = (
    "idle_timeout_info",
    "sandbox_startup_info",
    "transport_error_info",
    "verifier_timeout_info",
)

INFRA_ERROR_CATEGORIES = {
    INSTALL_FAILED,
    TIMED_OUT,
    IDLE_TIMEOUT,
    PIPE_CLOSED,
    SANDBOX_SETUP,
    INFRA_ERROR,
}

VERIFIER_DIAGNOSTIC_CATEGORIES = {
    VERIFIER_DEP_INSTALL,
    VERIFIER_TIMEOUT,
}


def serialize_flat_diagnostics(
    *,
    error: str | None,
    verifier_error: str | None,
    idle_timeout_info: dict | None = None,
    sandbox_startup_info: dict | None = None,
    transport_error_info: dict | None = None,
    verifier_timeout_info: dict | None = None,
) -> dict[str, Any]:
    """Return the legacy flat diagnostic fields for result.json."""
    return {
        "error_category": classify_error(error),
        "verifier_error_category": classify_verifier_error(verifier_error),
        "idle_timeout_info": idle_timeout_info,
        "sandbox_startup_info": sandbox_startup_info,
        "transport_error_info": transport_error_info,
        "verifier_timeout_info": verifier_timeout_info,
    }


def diagnostic_category_counts(
    results: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count agent and verifier diagnostic categories across result payloads."""
    error_category_counts: dict[str, int] = {}
    verifier_error_category_counts: dict[str, int] = {}
    for result in results:
        category = classify_error(result.get("error"))
        if category:
            error_category_counts[category] = (
                error_category_counts.get(category, 0) + 1
            )
        verifier_category = classify_verifier_error(result.get("verifier_error"))
        if verifier_category:
            verifier_error_category_counts[verifier_category] = (
                verifier_error_category_counts.get(verifier_category, 0) + 1
            )
    return error_category_counts, verifier_error_category_counts


@dataclass(frozen=True)
class SummaryWarningSpec:
    source: Literal["agent", "verifier"]
    category: str
    template: str


SUMMARY_WARNING_SPECS = (
    SummaryWarningSpec(
        source="agent",
        category=IDLE_TIMEOUT,
        template=(
            "{count} tasks ({pct:.0f}%) hit idle timeout — "
            "check idle_timeout_info in result.json for diagnostics"
        ),
    ),
    SummaryWarningSpec(
        source="agent",
        category=SANDBOX_SETUP,
        template=(
            "{count} tasks ({pct:.0f}%) failed during sandbox startup — "
            "check sandbox_startup_info in result.json for diagnostics"
        ),
    ),
    SummaryWarningSpec(
        source="agent",
        category=PIPE_CLOSED,
        template=(
            "{count} tasks ({pct:.0f}%) lost transport (pipe closed / rc=255) — "
            "check transport_error_info in result.json for diagnostics"
        ),
    ),
    SummaryWarningSpec(
        source="verifier",
        category=VERIFIER_DEP_INSTALL,
        template=(
            "{count} tasks ({pct:.0f}%) failed during verifier dependency install — "
            "check verifier_error_category in result.json and fix the task's index policy"
        ),
    ),
    SummaryWarningSpec(
        source="verifier",
        category=VERIFIER_TIMEOUT,
        template=(
            "{count} tasks ({pct:.0f}%) had verifier timeouts — "
            "check verifier_timeout_info in result.json for budget/elapsed details"
        ),
    ),
)


def summary_diagnostic_warnings(
    *,
    total: int,
    error_category_counts: Mapping[str, int],
    verifier_error_category_counts: Mapping[str, int],
) -> list[str]:
    """Render job-level warning messages from diagnostic category counts."""
    messages: list[str] = []
    for spec in SUMMARY_WARNING_SPECS:
        counts = (
            verifier_error_category_counts
            if spec.source == "verifier"
            else error_category_counts
        )
        count = counts.get(spec.category, 0)
        if count:
            pct = count / total * 100 if total else 0
            messages.append(spec.template.format(count=count, pct=pct))
    return messages


def agent_infra_category(result: Mapping[str, Any]) -> str | None:
    """Return the invalidating agent-side infra category for a result."""
    error = result.get("error")
    category = classify_error(str(error)) if error else None
    if category in INFRA_ERROR_CATEGORIES:
        return category
    return None


def verifier_diagnostic_category(result: Mapping[str, Any]) -> str | None:
    """Return the invalidating verifier-side diagnostic category for a result."""
    verifier_error = result.get("verifier_error")
    category = classify_verifier_error(verifier_error) if verifier_error else None
    if category in VERIFIER_DIAGNOSTIC_CATEGORIES:
        return category
    return None


def _task_name(result: Mapping[str, Any]) -> str:
    return str(result.get("task_name", "?"))


def _info(result: Mapping[str, Any], field: str) -> Mapping[str, Any] | None:
    value = result.get(field)
    return value if isinstance(value, Mapping) else None


def _idle_timeout_issue(task: str, result: Mapping[str, Any]) -> str:
    info = _info(result, "idle_timeout_info")
    if not info:
        return f"{task}: {result.get('error')}"
    return (
        f"{task}: idle timeout after {info.get('idle_duration_sec', '?')}s idle "
        f"({info.get('n_tool_calls', '?')} tool calls, "
        f"{info.get('wall_clock_elapsed_sec', '?')}s wall)"
    )


def _sandbox_startup_issue(task: str, result: Mapping[str, Any]) -> str:
    info = _info(result, "sandbox_startup_info")
    if not info:
        return f"{task}: {result.get('error')}"
    return (
        f"{task}: sandbox startup failed (sandbox_id={info.get('sandbox_id', '?')}, "
        f"state={info.get('sandbox_state', '?')}, attempts={info.get('attempts', '?')}, "
        f"build_timeout_sec={info.get('build_timeout_sec', '?')})"
    )


def _transport_issue(task: str, result: Mapping[str, Any]) -> str:
    info = _info(result, "transport_error_info")
    if not info:
        return f"{task}: {result.get('error')}"
    return (
        f"{task}: transport closed (rc={info.get('process_exit_code', '?')}, "
        f"diagnosis={info.get('transport_diagnosis', '?')}, "
        f"sandbox_reachable={info.get('sandbox_reachable', '?')})"
    )


_AGENT_ISSUE_RENDERERS = {
    IDLE_TIMEOUT: _idle_timeout_issue,
    SANDBOX_SETUP: _sandbox_startup_issue,
    PIPE_CLOSED: _transport_issue,
}

_AGENT_INVALIDATION_TEMPLATES = {
    IDLE_TIMEOUT: (
        "INVALIDATED: {count} task(s) hit idle timeout and should be rerun: {tasks}"
    ),
    SANDBOX_SETUP: (
        "INVALIDATED: {count} task(s) failed during sandbox startup and should "
        "be rerun: {tasks}"
    ),
    PIPE_CLOSED: (
        "INVALIDATED: {count} task(s) lost ACP transport (pipe closed / rc=255) "
        "and should be rerun: {tasks}"
    ),
}


def format_agent_infra_issue(result: Mapping[str, Any], category: str) -> str:
    """Render the checker issue line for an agent-side infra diagnostic."""
    renderer = _AGENT_ISSUE_RENDERERS.get(category)
    if renderer is None:
        return f"{_task_name(result)}: {result.get('error')}"
    return renderer(_task_name(result), result)


def format_agent_invalidation_messages(
    tasks_by_category: Mapping[str, list[str]],
) -> list[str]:
    return _format_invalidation_messages(tasks_by_category, _AGENT_INVALIDATION_TEMPLATES)


def _verifier_dep_install_issue(task: str, result: Mapping[str, Any]) -> str:
    return (
        f"{task}: verifier dependency install failed — "
        f"measurement invalid (verifier never reached tests)"
    )


def _verifier_timeout_issue(task: str, result: Mapping[str, Any]) -> str:
    info = _info(result, "verifier_timeout_info")
    budget = info.get("timeout_budget_sec", "?") if info else "?"
    elapsed = info.get("elapsed_sec", "?") if info else "?"
    return (
        f"{task}: verifier timed out (budget={budget}s, elapsed={elapsed}s) — "
        f"measurement invalid (verifier never produced reward)"
    )


_VERIFIER_ISSUE_RENDERERS = {
    VERIFIER_DEP_INSTALL: _verifier_dep_install_issue,
    VERIFIER_TIMEOUT: _verifier_timeout_issue,
}

_VERIFIER_INVALIDATION_TEMPLATES = {
    VERIFIER_DEP_INSTALL: (
        "INVALIDATED: {count} task(s) failed during verifier dependency install "
        "and should be rerun after fixing the index policy: {tasks}"
    ),
    VERIFIER_TIMEOUT: (
        "INVALIDATED: {count} task(s) had verifier timeouts — increase timeout_sec "
        "or reduce verifier cost: {tasks}"
    ),
}


def format_verifier_diagnostic_issue(result: Mapping[str, Any], category: str) -> str:
    renderer = _VERIFIER_ISSUE_RENDERERS[category]
    return renderer(_task_name(result), result)


def format_verifier_invalidation_messages(
    tasks_by_category: Mapping[str, list[str]],
) -> list[str]:
    return _format_invalidation_messages(
        tasks_by_category, _VERIFIER_INVALIDATION_TEMPLATES
    )


def _format_invalidation_messages(
    tasks_by_category: Mapping[str, list[str]],
    templates: Mapping[str, str],
) -> list[str]:
    messages: list[str] = []
    for category, template in templates.items():
        tasks = tasks_by_category.get(category)
        if tasks:
            messages.append(
                template.format(count=len(tasks), tasks=", ".join(tasks))
            )
    return messages

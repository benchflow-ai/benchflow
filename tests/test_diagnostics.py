from __future__ import annotations

from benchflow._utils.diagnostics import (
    agent_infra_category,
    diagnostic_category_counts,
    format_agent_infra_issue,
    format_agent_invalidation_messages,
    format_verifier_diagnostic_issue,
    format_verifier_invalidation_messages,
    serialize_flat_diagnostics,
    summary_diagnostic_warnings,
    verifier_diagnostic_category,
)


def test_flat_result_diagnostics_are_serialized_from_one_helper() -> None:
    """Guards the fix from PR TBD against #503: result.json diagnostic fields
    are serialized through the shared flat diagnostic contract."""
    transport_info = {"reason": "transport_closed"}
    verifier_timeout_info = {"timeout_budget_sec": 60, "elapsed_sec": 60.1}

    payload = serialize_flat_diagnostics(
        error="Process closed stdout (rc=255)",
        verifier_error="verifier timed out after 60s",
        transport_error_info=transport_info,
        verifier_timeout_info=verifier_timeout_info,
    )

    assert payload == {
        "error_category": "pipe_closed",
        "verifier_error_category": "verifier_timeout",
        "idle_timeout_info": None,
        "sandbox_startup_info": None,
        "transport_error_info": transport_info,
        "verifier_timeout_info": verifier_timeout_info,
    }


def test_diagnostic_registry_drives_summary_and_checker_rendering() -> None:
    """Guards the fix from PR TBD against #503: summary warnings and result
    checker diagnostics share one registry instead of parallel branches."""
    transport_result = {
        "task_name": "task-a",
        "error": "Process closed stdout (rc=255)",
        "verifier_error": None,
        "transport_error_info": {
            "process_exit_code": 255,
            "transport_diagnosis": "process_exited",
            "sandbox_reachable": False,
        },
    }
    verifier_timeout_result = {
        "task_name": "task-b",
        "error": None,
        "verifier_error": "verifier timed out after 60s",
        "verifier_timeout_info": {
            "timeout_budget_sec": 60,
            "elapsed_sec": 60.1,
        },
    }
    error_counts, verifier_counts = diagnostic_category_counts(
        [transport_result, verifier_timeout_result]
    )

    warnings = summary_diagnostic_warnings(
        total=2,
        error_category_counts=error_counts,
        verifier_error_category_counts=verifier_counts,
    )

    assert warnings == [
        "1 tasks (50%) lost transport (pipe closed / rc=255) — "
        "check transport_error_info in result.json for diagnostics",
        "1 tasks (50%) had verifier timeouts — "
        "check verifier_timeout_info in result.json for budget/elapsed details",
    ]
    agent_category = agent_infra_category(transport_result)
    verifier_category = verifier_diagnostic_category(verifier_timeout_result)
    assert agent_category == "pipe_closed"
    assert verifier_category == "verifier_timeout"
    assert (
        format_agent_infra_issue(transport_result, agent_category)
        == "task-a: transport closed (rc=255, diagnosis=process_exited, "
        "sandbox_reachable=False)"
    )
    assert (
        format_verifier_diagnostic_issue(verifier_timeout_result, verifier_category)
        == "task-b: verifier timed out (budget=60s, elapsed=60.1s) — "
        "measurement invalid (verifier never produced reward)"
    )
    assert format_agent_invalidation_messages({"pipe_closed": ["task-a"]}) == [
        "INVALIDATED: 1 task(s) lost ACP transport (pipe closed / rc=255) "
        "and should be rerun: task-a"
    ]
    assert format_verifier_invalidation_messages({"verifier_timeout": ["task-b"]}) == [
        "INVALIDATED: 1 task(s) had verifier timeouts — increase timeout_sec "
        "or reduce verifier cost: task-b"
    ]

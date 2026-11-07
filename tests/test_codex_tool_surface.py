"""Codex's fallback tool surface is a failure on the record, never a silent
degraded run.

A model Codex's catalog lacks runs on fallback metadata: no apply_patch, no
code mode (a published run's xhigh rows ran so on every trial). #61 stops an
ACP trial when codex-acp sends Codex's warning; these guard what it left: the
trial's cause (it read as the agent's own ACP error, retried twice), the
native harness (it recorded the warning and ran on), and a record of the
tools the agent's first request offered, so a fallback surface shows when no
warning was given.

The tool lists are the ones the pinned Codex (0.159.3) sent BenchFlow's
scripted model for a catalogued model (gpt-5.5), a catalogued code-mode-only
model (gpt-6-astra, whose tools ride an ``additional_tools`` input item) and a
model the catalog lacks (gpt-5.4).
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow._utils.evaluation_results import model_call_summary
from benchflow._utils.scoring import AGENT_INTEGRATION, classify_error
from benchflow.acp.client import ModelMetadataError
from benchflow.agents.codex_config import codex_fallback_tools
from benchflow.diagnostics import (
    IntegrationFailureDiagnostic,
    RolloutDiagnostics,
    UsageLimitDiagnostic,
)
from benchflow.evaluation import ApiErrorCircuitBreaker, Evaluation, RetryConfig
from benchflow.rollout import Rollout, _classify_model_catalog_fallback
from benchflow.rollout._results import _build_rollout_result
from benchflow.trajectories.model_calls import request_tools, write_model_calls

CATALOGUED = [
    {"type": "function", "name": "exec_command"},
    {"type": "function", "name": "write_stdin"},
    {"type": "function", "name": "request_user_input"},
    {"type": "custom", "name": "apply_patch"},
    {"type": "function", "name": "view_image"},
    {"type": "function", "name": "get_goal"},
    {"type": "function", "name": "create_goal"},
    {"type": "function", "name": "update_goal"},
    {"type": "tool_search"},
    {"type": "web_search"},
]
CODE_MODE = [
    {
        "type": "additional_tools",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {"type": "custom", "name": "exec"},
                    {"type": "function", "name": "wait"},
                ],
            },
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [{"type": "function", "name": "spawn_agent"}],
            },
        ],
    },
    {"type": "message", "role": "user", "content": "task"},
]
FALLBACK = [
    {"type": "function", "name": "exec_command"},
    {"type": "function", "name": "write_stdin"},
    {"type": "function", "name": "request_user_input"},
    {"type": "function", "name": "view_image"},
    {
        "type": "namespace",
        "name": "multi_agent_v1",
        "tools": [{"type": "function", "name": "spawn_agent"}],
    },
    {"type": "function", "name": "get_goal"},
    {"type": "function", "name": "create_goal"},
    {"type": "function", "name": "update_goal"},
    {"type": "web_search"},
]


def test_the_three_surfaces_the_pinned_codex_offers():
    catalogued = request_tools({"tools": CATALOGUED})
    code_mode = request_tools({"input": CODE_MODE})
    fallback = request_tools({"tools": FALLBACK})
    assert catalogued[3] == "apply_patch" and catalogued[-2:] == [
        "tool_search",
        "web_search",
    ]
    assert code_mode == [
        "functions.exec",
        "functions.wait",
        "collaboration.spawn_agent",
    ]
    assert "multi_agent_v1.spawn_agent" in fallback
    assert [codex_fallback_tools(n) for n in (catalogued, code_mode, fallback)] == [
        False,
        False,
        True,
    ]
    # No tools at all says nothing (a side call, a source without requests).
    assert codex_fallback_tools([]) is False


def _row(body, purpose="agent", status=200):
    return {
        "request": {"body": {"model": "gpt-5.4", **body}},
        "response": {"status_code": status, "body": {"status": "completed"}},
        "metadata": {"call_purpose": purpose},
    }


def _trial(tmp_path, rows, name="codex-task__1"):
    trial = tmp_path / name
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return trial


def test_model_calls_record_the_agents_first_tool_request(tmp_path):
    """A tool-less side call and a judge's call come first; the record is the
    agent's first request that offered tools, its names in order and their
    hash."""
    trial = _trial(
        tmp_path,
        [
            _row({"input": "title this"}),
            _row({"tools": CATALOGUED}, purpose="judge"),
            _row({"tools": FALLBACK}),
            _row({"tools": CATALOGUED}),
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    tools = summary["first_agent_tools"]
    assert tools["names"] == request_tools({"tools": FALLBACK})
    assert tools["count"] == len(tools["names"]) == 9
    assert len(tools["sha256"]) == 64
    again = json.loads((trial / "model-calls.json").read_text())
    assert again["first_agent_tools"] == tools


def _build(trial, agent, rewards=None):
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent=agent,
        agent_name=agent,
        model="gpt-5.4",
        n_tool_calls=2,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards or {"reward": 0.0},
        started_at=datetime.now(),
        timing={},
    )
    return json.loads((trial / "result.json").read_text())


@pytest.mark.parametrize(
    "agent,tools,fallback",
    [
        ("codex-acp", FALLBACK, True),
        ("codex-acp", CATALOGUED, False),
        ("claude-agent-acp", FALLBACK, None),
    ],
)
def test_result_json_names_the_tool_surface(tmp_path, agent, tools, fallback):
    trial = _trial(tmp_path, [_row({"tools": tools})])
    write_model_calls(trial)
    surface = _build(trial, agent)["tool_surface"]
    assert surface["names"] == request_tools({"tools": tools})
    assert surface["fallback"] is fallback


def _rollout(trial, *, agent="codex-acp", trajectory=(), rewards=None):
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(primary_agent=agent, primary_model="gpt-5.4")
    rollout._rollout_dir = trial
    rollout._diagnostics = RolloutDiagnostics()
    rollout._trajectory = list(trajectory)
    rollout._rewards = rewards if rewards is not None else {"reward": 1.0}
    rollout._error = None
    rollout._bare_timeout = False
    rollout._judge_timed_out_agent = False
    return rollout


def test_a_fallback_surface_fails_the_trial_without_a_warning(tmp_path):
    trial = _trial(tmp_path, [_row({"tools": FALLBACK})])
    write_model_calls(trial)
    rollout = _rollout(trial)

    assert _classify_model_catalog_fallback(rollout) is True

    assert rollout._rewards is None
    assert rollout._error.startswith(
        "agent integration failure [agent_model_catalog]: codex-acp ran gpt-5.4 "
        "on fallback model metadata: its first request offered 9 tools "
        "(exec_command, write_stdin"
    )
    info = rollout._diagnostics.get(IntegrationFailureDiagnostic.field)
    assert info.cause == "agent_model_catalog"
    assert info.reward_withheld == {"reward": 1.0}
    assert classify_error(rollout._error) == AGENT_INTEGRATION
    assert not RetryConfig().should_retry(rollout._error)
    result = MagicMock(error=rollout._error, error_category=None)
    assert (
        ApiErrorCircuitBreaker._fingerprint_of(result)
        == "integration:agent_model_catalog"
    )


def test_the_native_harness_warning_fails_the_trial(tmp_path):
    """codex exec reports the warning as an error item; the native harness
    records it as a notice and the CLI runs on, so the rollout judges it."""
    trial = _trial(tmp_path, [_row({"input": "no tools seen"})])
    write_model_calls(trial)
    notice = {
        "type": "notice",
        "severity": "warning",
        "title": "Model metadata for `gpt-5.4` not found. Defaulting to fallback "
        "metadata; this can degrade performance and cause issues.",
    }
    rollout = _rollout(trial, trajectory=[{"type": "user_message"}, notice])

    assert _classify_model_catalog_fallback(rollout) is True
    info = rollout._diagnostics.get(IntegrationFailureDiagnostic.field)
    assert (info.cause, info.evidence_source) == (
        "agent_model_catalog",
        "session notice",
    )
    assert "Model metadata for `gpt-5.4` not found" in rollout._error


@pytest.mark.parametrize("case", ["catalogued", "other agent", "usage limit", "judged"])
def test_what_is_not_a_fallback_or_has_a_cause_stays_as_it_was(tmp_path, case):
    tools = CATALOGUED if case == "catalogued" else FALLBACK
    trial = _trial(tmp_path, [_row({"tools": tools})])
    write_model_calls(trial)
    rollout = _rollout(
        trial, agent="claude-agent-acp" if case == "other agent" else "codex-acp"
    )
    if case == "usage limit":
        rollout._diagnostics.set(UsageLimitDiagnostic(login="x"))
    if case == "judged":
        rollout._diagnostics.set(IntegrationFailureDiagnostic(cause="no_activity"))

    assert _classify_model_catalog_fallback(rollout) is False
    assert rollout._rewards == {"reward": 1.0} and rollout._error is None


def test_codex_acp_warning_is_the_setups_failure_not_the_agents(tmp_path):
    """#61 raises ModelMetadataError on the warning; it read as an ACP error
    (the agent's, retried twice). It is the catalog cause now."""
    rollout = _rollout(tmp_path, rewards={})
    rollout._rewards = None
    error = ModelMetadataError(
        -32603,
        "Agent model catalog is missing the requested model; refusing fallback tool "
        "metadata. Use an engine/catalog that supports this model. Model metadata "
        "for `gpt-5.4` not found. Defaulting to fallback metadata",
    )

    message = Rollout._classify_acp_error(rollout, error)

    assert message == (
        "agent integration failure [agent_model_catalog]: codex-acp ran gpt-5.4 on "
        "fallback model metadata: Model metadata for `gpt-5.4` not found. "
        "Defaulting to fallback metadata"
    )
    assert classify_error(message) == AGENT_INTEGRATION
    assert not RetryConfig().should_retry(message)
    info = rollout._diagnostics.get(IntegrationFailureDiagnostic.field)
    assert (info.cause, info.evidence_source) == (
        "agent_model_catalog",
        "session notice",
    )


def test_the_job_summary_lists_each_surface_and_counts_fallbacks(tmp_path):
    results = {}
    for index, (agent, tools) in enumerate(
        [("codex-acp", FALLBACK), ("codex-acp", FALLBACK), ("codex-acp", CATALOGUED)]
    ):
        trial = _trial(tmp_path, [_row({"tools": tools})], name=f"t{index}__1")
        write_model_calls(trial)
        results[trial.name] = _build(trial, agent)
    results["no-proxy"] = {"tool_surface": None}

    block = model_call_summary(results)["model_calls"]

    assert block["fallback_tool_trials"] == 2
    assert [
        (s["trials"], s["fallback"], s["count"]) for s in block["tool_surfaces"]
    ] == [
        (2, True, 9),
        (1, False, 10),
    ]
    assert model_call_summary({"a": {"tool_surface": None}}) == {}


def test_a_fresh_trials_payload_gets_the_records_from_its_result(tmp_path):
    """summary.json totals fresh trials from payloads built in memory, which
    carry no model-call records: they are read from the trial's result.json."""
    job = tmp_path / "jobs" / "job"
    trial = job / "t__1"
    trial.mkdir(parents=True)
    persisted = {
        "timing": {},
        "tool_surface": {"sha256": "a", "count": 1, "names": ["x"], "fallback": True},
        "effort": {"matches": False},
        "agent_outcome": {"status": "completed"},
    }
    (trial / "result.json").write_text(json.dumps(persisted))
    payload: dict = {}
    host = SimpleNamespace(_jobs_dir=tmp_path / "jobs", _job_name="job")

    Evaluation._enrich_payload_with_persisted_timing(
        host, payload, SimpleNamespace(rollout_name="t__1")
    )

    for key in ("tool_surface", "effort", "agent_outcome"):
        assert payload[key] == persisted[key]

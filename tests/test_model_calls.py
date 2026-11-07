"""One record per model call, from the proxy or the agent's own session log.

The fixtures are content-free copies of real logs (settings, finish
reasons, usage and times kept; prompts and answers dropped): two Claude Code
trials on MiMo v2.6 flash through BenchFlow's proxy, with Claude Code's
session log kept beside the proxy's record (``hello``, and ``trunc``: an
essay under a 400-token output cap), and a Codex 0.166 session (with
``token_usage_record`` lines) and the same session as Codex 0.159 writes it
(``token_count`` events only).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from benchflow.trajectories import model_calls as mc

FIXTURES = Path(__file__).parent / "fixtures" / "model_calls"


def _trial(
    tmp_path: Path,
    *,
    proxy: str | None = None,
    claude: str | None = None,
    codex: str | None = None,
    otel: str | None = None,
) -> Path:
    trial = tmp_path / "trial"
    if proxy:
        (trial / "trajectory").mkdir(parents=True)
        shutil.copy(FIXTURES / proxy, trial / "trajectory" / "llm_trajectory.jsonl")
    if otel:
        (trial / "trajectory").mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / otel, trial / "trajectory" / "agent-otel.jsonl")
    if claude:
        folder = trial / "agent" / "claude-sessions" / "-app"
        folder.mkdir(parents=True)
        shutil.copy(FIXTURES / claude, folder / "session.jsonl")
    if codex:
        folder = trial / "agent" / "codex-sessions" / "2026" / "11" / "04"
        folder.mkdir(parents=True)
        shutil.copy(FIXTURES / codex, folder / "rollout.jsonl")
    return trial


def test_the_proxy_records_what_each_request_sent(tmp_path):
    calls = mc.model_calls(_trial(tmp_path, proxy="hello-proxy.jsonl"))
    assert [c["source"] for c in calls] == ["proxy", "proxy"]
    assert {(c["effort"], c["max_output_tokens"]) for c in calls} == {("high", 128000)}
    assert [c["finish_reason"] for c in calls] == ["tool_calls", "stop"]
    assert all(
        c["model_served"] == "mimo-v2.6-flash" and c["http_status"] == 200
        for c in calls
    )
    summary = mc.summarize(calls)
    assert summary["truncated_calls"] == 0 and summary["sources"]["effort"] == "proxy"
    assert summary["sources"]["sampling"] == "none sent"


def test_calls_cut_at_the_output_cap_count_as_truncated(tmp_path):
    summary = mc.summarize(mc.model_calls(_trial(tmp_path, proxy="trunc-proxy.jsonl")))
    assert summary["calls"] == 30
    assert summary["max_output_tokens"] == {"400": 30}
    assert summary["finish_reasons"] == {"length": 10, "tool_calls": 19, "stop": 1}
    assert (summary["truncated_calls"], summary["truncation_rate"]) == (10, 10 / 30)


def test_claude_codes_session_log_gives_the_proxys_figures(tmp_path):
    """The fallback for a run around the proxy, checked against the proxy on
    the same trial: the same calls, cuts, tokens and model time."""
    proxy = mc.summarize(mc._proxy_calls(FIXTURES / "trunc-proxy.jsonl"))
    session = mc.summarize(
        mc.model_calls(_trial(tmp_path, claude="trunc-claude-session.jsonl"))
    )
    assert session["source"] == "claude-code-session-log"
    assert session["calls"] == proxy["calls"] == 30
    assert session["finish_reasons"] == {
        "max_tokens": 10,
        "tool_use": 19,
        "end_turn": 1,
    }
    assert session["truncated_calls"] == proxy["truncated_calls"] == 10
    for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
        assert session["tokens"][key] == proxy["tokens"][key], key
    assert (
        abs(session["model_time_ms"] - proxy["model_time_ms"])
        < 0.01 * proxy["model_time_ms"]
    )
    # What a session log cannot see is said, not guessed.
    assert session["sources"]["effort"] == "not recorded by claude-code-session-log"
    assert session["sources"]["tokens"] == "claude-code-session-log"


def test_the_proxy_wins_where_both_recorded(tmp_path):
    trial = _trial(
        tmp_path, proxy="hello-proxy.jsonl", claude="hello-claude-session.jsonl"
    )
    assert {c["source"] for c in mc.model_calls(trial)} == {"proxy"}


def test_codex_sessions_count_each_response_once(tmp_path):
    new = mc.model_calls(_trial(tmp_path / "a", codex="codex-session.jsonl"))
    old = mc.model_calls(_trial(tmp_path / "b", codex="codex-session-old.jsonl"))
    assert len(new) == len(old) == 9
    assert {(c["model_served"], c["effort"]) for c in new} == {("gpt-6.1-sol", "xhigh")}
    total = lambda calls, key: sum(c[key] or 0 for c in calls)  # noqa: E731
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "reasoning_tokens",
    ):
        assert total(new, key) == total(old, key), key
    assert (
        mc.summarize(new)["sources"]["finish_reasons"]
        == "not recorded by codex-session-log"
    )


def test_a_codex_sub_agent_thread_is_no_main_call(tmp_path):
    """Guards the run checks merged at 47a52fdf7: without the proxy (Codex
    on a ChatGPT login) the calls come from Codex's session log, which has
    no tools count, so none was ever set apart. A sub-agent's thread is a
    rollout of its own, its session_meta source {"subagent": ...} (Codex
    0.159.3 names review, compact, thread_spawn and other), and it may run
    on another model and effort: compared as the agent's own, its calls
    failed the run on effort and made its last call the agent's."""
    from benchflow.run_checks import settings_sent

    trial = _trial(tmp_path, codex="codex-session.jsonl")
    folder = trial / "agent" / "codex-sessions" / "2026" / "11" / "04"
    rows = [
        {
            "timestamp": "2026-11-04T10:05:00Z",
            "type": "session_meta",
            "payload": {
                "id": "thread-2",
                "cli_version": "0.159.3",
                "source": {
                    "subagent": {
                        "thread_spawn": {"parent_thread_id": "thread-1", "depth": 1}
                    }
                },
            },
        },
        {
            "timestamp": "2026-11-04T10:05:01Z",
            "type": "turn_context",
            "payload": {"model": "gpt-6.1-luna", "effort": "low"},
        },
        {
            "timestamp": "2026-11-04T10:05:09Z",
            "type": "token_usage_record",
            "payload": {
                "response_id": "resp_sub",
                "usage": {"input_tokens": 900, "output_tokens": 40},
            },
        },
    ]
    (folder / "rollout-sub.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    calls = mc.model_calls(trial)
    assert sorted((c["model_served"], c["subagent"] or False) for c in calls) == [
        ("gpt-6.1-luna", True),
        *[("gpt-6.1-sol", False)] * 9,
    ]
    sent = settings_sent(calls, "openai/gpt-6.1-sol")
    assert (sent["main_calls"], sent["side_calls"]) == (9, 1)
    assert sent["effort"] == {"xhigh": 9}
    assert sent["side_models"] == {"gpt-6.1-luna": 1}
    assert mc.summarize(calls)["agent_effort"] == {"xhigh": 9}


def test_write_model_calls_writes_both_files(tmp_path):
    trial = _trial(tmp_path, proxy="trunc-proxy.jsonl")
    summary = mc.write_model_calls(trial)
    lines = (trial / "trajectory" / mc.CALLS_FILE).read_text().splitlines()
    assert len(lines) == 30 and json.loads(lines[0])["schema"] == mc.SCHEMA
    assert json.loads((trial / mc.SUMMARY_FILE).read_text()) == json.loads(
        json.dumps(summary)
    )
    assert summary["schema"] == mc.SUMMARY_SCHEMA


def test_a_trial_with_no_record_writes_nothing(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    assert mc.write_model_calls(trial) is None
    assert not (trial / mc.SUMMARY_FILE).exists()


def test_a_link_in_the_shared_agent_folder_is_not_read(tmp_path):
    trial = _trial(tmp_path, claude="hello-claude-session.jsonl")
    outside = tmp_path / "outside.jsonl"
    shutil.copy(FIXTURES / "trunc-claude-session.jsonl", outside)
    os.symlink(outside, trial / "agent" / "claude-sessions" / "-app" / "planted.jsonl")
    linked = trial / "agent" / "claude-sessions" / "linked"
    os.symlink(tmp_path, linked)
    calls = mc.model_calls(trial)
    assert len(calls) == len(
        mc._claude_code_calls(
            _trial(tmp_path / "c", claude="hello-claude-session.jsonl")
            / "agent"
            / "claude-sessions"
        )
    )


def test_the_committed_schemas_are_current_and_records_validate(tmp_path):
    import pytest

    jsonschema = pytest.importorskip("jsonschema")
    schemas = Path(__file__).resolve().parents[1] / "docs" / "reference" / "schemas"
    for name, schema in mc.SCHEMAS.items():
        committed = json.loads((schemas / name).read_text())
        jsonschema.Draft202012Validator.check_schema(committed)
        assert committed == json.loads(json.dumps(schema)), name
    record = mc.SCHEMAS["benchflow-model-call.v2.schema.json"]
    totals = mc.SCHEMAS["benchflow-model-calls.v1.schema.json"]
    # A record from before the per-call tools count still validates as v1.
    old = {**mc._record(mc.PROXY, index=0), "schema": mc.SCHEMA_V1}
    old.pop("tools")
    old.pop("seed")
    jsonschema.validate(old, mc.SCHEMAS["benchflow-model-call.v1.schema.json"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {**old, "tools": 3}, mc.SCHEMAS["benchflow-model-call.v1.schema.json"]
        )
    for fixture, kind in (
        ("trunc-proxy.jsonl", "proxy"),
        ("trunc-claude-session.jsonl", "claude"),
    ):
        trial = _trial(tmp_path / kind, **{kind: fixture})
        calls = mc.model_calls(trial)
        for call in calls:
            jsonschema.validate(call, record)
        jsonschema.validate(json.loads(json.dumps(mc.summarize(calls))), totals)


def test_claude_codes_opentelemetry_gives_the_proxys_calls(tmp_path):
    """Claude Code with CLAUDE_CODE_ENABLE_TELEMETRY=1 exported to BenchFlow's
    receiver (hello task, MiMo flash, through the proxy): its api_request
    events give the proxy's calls, tokens and effort, its own cost estimate,
    and a model time measured from the client."""
    proxy = mc.summarize(mc._proxy_calls(FIXTURES / "otel-proxy.jsonl"))
    otel = mc.summarize(mc.model_calls(_trial(tmp_path, otel="otel-claude-code.jsonl")))
    assert otel["source"] == "claude-code-otel"
    assert otel["calls"] == proxy["calls"] == 2
    for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
        assert otel["tokens"][key] == proxy["tokens"][key], key
    assert otel["effort"] == proxy["effort"] == {"high": 2}
    assert (
        abs(otel["model_time_ms"] - proxy["model_time_ms"])
        < 0.05 * proxy["model_time_ms"]
    )
    assert (
        otel["cost_usd"] is not None
        and otel["sources"]["cost_usd"] == "claude-code-otel"
    )
    assert otel["sources"]["finish_reasons"] == "not recorded by claude-code-otel"


def test_the_proxy_wins_over_the_agents_opentelemetry(tmp_path):
    trial = _trial(tmp_path, proxy="otel-proxy.jsonl", otel="otel-claude-code.jsonl")
    assert {c["source"] for c in mc.model_calls(trial)} == {"proxy"}


def _one_proxy_call(tmp_path: Path, answer: dict) -> dict:
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {"body": {"model": "synthetic"}},
                "response": {"status_code": 200, "body": answer},
            }
        )
        + "\n"
    )
    return mc.model_calls(trial)[0]


def test_a_refusal_that_ends_as_stop_or_completed_is_still_a_refusal(tmp_path):
    """The independent audit's probe (2026-11-04, M2): both answers read as
    refused=false and their Responses usage details as null."""
    chat = _one_proxy_call(
        tmp_path / "chat",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "Synthetic refusal",
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )
    responses = _one_proxy_call(
        tmp_path / "responses",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "Synthetic refusal"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 8},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        },
    )
    assert (chat["finish_reason"], chat["refused"]) == ("stop", True)
    assert (responses["finish_reason"], responses["refused"]) == ("completed", True)
    assert (responses["input_tokens"], responses["output_tokens"]) == (10, 2)
    assert (responses["cache_read_tokens"], responses["reasoning_tokens"]) == (8, 1)
    assert mc.summarize([chat, responses])["refused_calls"] == 2


def test_an_answer_with_an_empty_refusal_field_is_not_a_refusal(tmp_path):
    call = _one_proxy_call(
        tmp_path,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4", "refusal": None},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )
    assert call["refused"] is False and call["truncated"] is False
    assert (call["cache_read_tokens"], call["reasoning_tokens"]) == (0, 0)

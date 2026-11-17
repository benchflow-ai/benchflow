"""A turn that ends without its usage report still counts, and is priced.

frontier-design-benchmark PR #17 ran Claude Code on a subscription with
``--harness native`` on 0.10.0rc2. One Opus trial hit its 4-hour limit and
one Sonnet trial lost its sandbox: neither turn wrote Claude Code's final
``result`` event, the only place the native harness read usage from, so the
trials recorded no tokens and no cost. The stream BenchFlow keeps on the host
(``agent/claude-code.jsonl``) holds every message's usage; Claude Code's and
Codex's session logs hold every response's. Teardown now counts the trial
from them and prices it at list prices.

The streams here are Claude Code 2.1.316's own shapes (the fixtures under
``tests/fixtures/native_harness/claude-code-2.1.316``) or built from them;
the session logs are small synthetic ones in the shape each CLI writes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "native_harness" / "claude-code-2.1.316"
OPUS = "claude-opus-5-5"
# Opus 5.5 at the 2026-11-04 snapshot: input, output, cache read, cache write.
OPUS_RATES = (4e-6, 20e-6, 0.2e-6, 5e-6)


def _events(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def _stream_log(trial: Path, turns: list[list[dict]]) -> Path:
    """agent/claude-code.jsonl as the native client writes it: a marker line
    before each turn, then the CLI's events."""
    path = trial / "agent" / "claude-code.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, events in enumerate(turns, start=1):
        lines.append({"benchflow_native_turn": number, "argv": ["claude", "-p"]})
        lines.extend(events)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _start(message_id: str, usage: dict, model: str = OPUS) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "message_start",
            "message": {"id": message_id, "model": model, "usage": usage},
        },
        "parent_tool_use_id": None,
    }


def _delta(output: int) -> dict:
    return {
        "type": "stream_event",
        "event": {"type": "message_delta", "usage": {"output_tokens": output}},
        "parent_tool_use_id": None,
    }


def _assistant(message_id: str, usage: dict, model: str = OPUS) -> dict:
    return {
        "type": "assistant",
        "message": {"id": message_id, "model": model, "usage": usage, "content": []},
        "parent_tool_use_id": None,
    }


def _usage(fresh: int, read: int, write: int, output: int = 1) -> dict:
    return {
        "input_tokens": fresh,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "output_tokens": output,
    }


# A turn killed by its agent timeout after one completed model call: the
# first message streamed whole (its message_delta says 120 output tokens),
# then a tool ran, then the second call started and the stream stopped.
KILLED_MID_TURN = [
    {"type": "system", "subtype": "init", "model": OPUS},
    _start("msg_1", _usage(1500, 0, 3000)),
    _assistant("msg_1", _usage(1500, 0, 3000)),
    _delta(120),
    {"type": "stream_event", "event": {"type": "message_stop"}},
    {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
    _start("msg_2", _usage(20, 3000, 400)),
    {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta"}},
    },
]
KILLED_TOKENS = {
    "n_input_tokens": 1520,
    "n_output_tokens": 121,
    "n_cache_read_tokens": 3000,
    "n_cache_creation_tokens": 3400,
    "total_tokens": 1520 + 121 + 3000 + 3400,
}
KILLED_USD = (
    1520 * OPUS_RATES[0]
    + 121 * OPUS_RATES[1]
    + 3000 * OPUS_RATES[2]
    + 3400 * OPUS_RATES[3]
)


def _tokens(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "n_input_tokens",
            "n_output_tokens",
            "n_cache_read_tokens",
            "n_cache_creation_tokens",
            "total_tokens",
        )
    }


def test_a_stream_cut_before_result_counts_its_messages(tmp_path):
    """The 2.1.316 cancelled turn without its final result event: one
    message, 1,000 input tokens and 50 output tokens (its message_delta;
    message_start and the assistant lines say 1)."""
    from benchflow.rollout._salvage import claude_stream_usage

    events = [e for e in _events("cancelled.jsonl") if e.get("type") != "result"]
    counted = claude_stream_usage(_stream_log(tmp_path, [events]))
    assert counted is not None
    assert _tokens(counted) == {
        "n_input_tokens": 1000,
        "n_output_tokens": 50,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 1050,
    }
    assert counted["turns_without_report"] == 1
    assert counted["messages"] == 1


def test_a_reported_turn_counts_its_report_and_a_cut_turn_its_messages(tmp_path):
    """Turn 1 ended with its result (2,000 in, 100 out); turn 2 was killed
    mid-call: the trial counts the report plus the killed turn's messages."""
    from benchflow.rollout._salvage import claude_stream_usage

    counted = claude_stream_usage(
        _stream_log(tmp_path, [_events("turn.jsonl"), KILLED_MID_TURN])
    )
    assert counted is not None
    assert counted["n_input_tokens"] == 2000 + 1520
    assert counted["n_output_tokens"] == 100 + 121
    assert counted["turns"] == 2
    assert counted["turns_without_report"] == 1


def test_no_stream_no_count(tmp_path):
    from benchflow.rollout._salvage import claude_stream_usage

    assert claude_stream_usage(tmp_path / "missing.jsonl") is None
    empty = _stream_log(tmp_path, [[{"type": "system", "subtype": "init"}]])
    assert claude_stream_usage(empty) is None


def _rollout(tmp_path: Path, monkeypatch, *, agent: str, harness: str, home: Path):
    """A rollout after its agent's turn, its sandbox's home at *home*."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig, _session_log
    from benchflow.task import RolloutPaths, Task
    from tests.test_artifacts_collection import LocalTransport

    monkeypatch.setattr(_session_log, "_home", lambda cfg: str(home))
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Build the site.")

    class Box(LocalTransport):
        async def stop(self, *, delete: bool = True) -> None:
            pass

    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            agent=agent,
            model=None,
            task_digest=task_digest(task),
            harness=harness,
        )
    )
    trial = tmp_path / "jobs" / "job" / "task__0a1b2c3d"
    rollout._task = Task(task)
    rollout._rollout_dir = trial
    rollout._rollout_name = trial.name
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=trial)
    rollout._rollout_paths.mkdir()
    rollout._env = Box()
    rollout._agent_cwd = str(tmp_path)
    rollout._agent_env = {}
    return rollout, trial


@pytest.mark.asyncio
async def test_a_native_claude_trial_killed_by_its_timeout_is_counted_and_priced(
    tmp_path, monkeypatch
):
    """The forced-timeout shape: the agent timeout killed the turn after one
    completed model call, so Claude Code wrote no result event. The trial
    records the stream's tokens and their list price, from the dated
    snapshot (no session log in the sandbox to estimate from)."""
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True  # execute() saw the timeout

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == KILLED_TOKENS
    assert usage["usage_source"] == "stream_log"
    assert usage["usage_source_legacy"] == "agent_native_acp"
    assert usage["cost_usd"] == pytest.approx(KILLED_USD)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"
    count = usage["usage_details"]["token_count"]
    assert count["source"] == "claude-code-stream"
    assert count["turns_without_report"] == 1
    assert count["end_of_turn_reported"] is None


@pytest.mark.asyncio
async def test_a_lost_sandbox_keeps_the_host_stream_count(tmp_path, monkeypatch):
    """The sandbox is gone: no session log can be copied, but the stream log
    is on the host. The trial still has tokens and a list-price cost."""
    rollout, trial = _rollout(
        tmp_path,
        monkeypatch,
        agent="claude-agent-acp",
        harness="native",
        home=tmp_path / "gone",
    )
    _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True
    rollout._env = None  # the sandbox connection is lost

    await rollout._teardown_finalize_usage()
    await rollout._teardown_price()

    usage = rollout._usage_metrics
    assert _tokens(usage) == KILLED_TOKENS
    assert usage["usage_source"] == "stream_log"
    assert usage["cost_usd"] == pytest.approx(KILLED_USD)
    assert usage["cost_source"] == "list_price_estimate"


@pytest.mark.asyncio
async def test_a_turn_that_reported_is_not_recounted(tmp_path, monkeypatch):
    """Every turn sent its report: the end-of-turn count stands."""
    rollout, trial = _rollout(
        tmp_path,
        monkeypatch,
        agent="claude-agent-acp",
        harness="native",
        home=tmp_path / "gone",
    )
    _stream_log(trial, [KILLED_MID_TURN])
    reported = {
        "n_input_tokens": 7,
        "n_output_tokens": 7,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 14,
        "cost_usd": None,
        "usage_source": "agent_native_acp",
        "price_source": None,
        "usage_details": {},
    }
    rollout._native_usage_metrics = dict(reported)
    rollout._env = None

    await rollout._teardown_finalize_usage()

    assert _tokens(rollout._usage_metrics) == _tokens(reported)
    assert rollout._usage_metrics["usage_source"] == "agent_native_acp"


def _claude_session_log(home: Path, responses: list[tuple[str, dict]]) -> None:
    """Claude Code's session log (~/.claude/projects/<cwd>/<session>.jsonl):
    one assistant line per content block, the response's usage on each."""
    path = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    path.parent.mkdir(parents=True)
    lines = []
    for index, (model, usage) in enumerate(responses):
        for _block in range(2):
            lines.append(
                {
                    "type": "assistant",
                    "requestId": f"req_{index}",
                    "message": {"id": f"msg_{index}", "model": model, "usage": usage},
                }
            )
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


@pytest.mark.asyncio
async def test_a_cancelled_acp_turn_is_counted_from_the_session_log(
    tmp_path, monkeypatch
):
    """claude-agent-acp's prompt was cancelled at the timeout, so no
    PromptResponse.usage came back. The session log teardown copies holds
    every response: the trial counts it, and its cost is the estimate from
    the same log."""
    home = tmp_path / "sandbox-home"
    _claude_session_log(
        home,
        [
            (OPUS, _usage(1500, 0, 3000, 120)),
            (OPUS, _usage(20, 3000, 400, 300)),
        ],
    )
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == {
        "n_input_tokens": 1520,
        "n_output_tokens": 420,
        "n_cache_read_tokens": 3000,
        "n_cache_creation_tokens": 3400,
        "total_tokens": 1520 + 420 + 3000 + 3400,
    }
    assert usage["usage_source"] == "session_log"
    assert usage["cost_usd"] == pytest.approx(
        1520 * OPUS_RATES[0]
        + 420 * OPUS_RATES[1]
        + 3000 * OPUS_RATES[2]
        + 3400 * OPUS_RATES[3]
    )
    assert usage["price_source"] == "agent_session_log"
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"


@pytest.mark.asyncio
async def test_a_claude_response_logged_twice_counts_once(tmp_path, monkeypatch):
    """A response is one (message id, request id), however many lines and
    session files log it: here a second session file logs the
    same two responses again, the last one's first block (its output so
    far) after its full line. Each counts once, with its most output."""
    home = tmp_path / "sandbox-home"
    second = _usage(20, 3000, 400, 300)
    _claude_session_log(home, [(OPUS, _usage(1500, 0, 3000, 120)), (OPUS, second)])
    folder = home / ".claude" / "projects" / "-app"
    lines = (folder / "session-1.jsonl").read_text().splitlines()
    partial = json.loads(lines[-1])
    partial["message"]["usage"] = {**second, "output_tokens": 1}
    (folder / "session-2.jsonl").write_text(
        "".join(line + "\n" for line in [*lines, json.dumps(partial)])
    )
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == {
        "n_input_tokens": 1520,
        "n_output_tokens": 420,
        "n_cache_read_tokens": 3000,
        "n_cache_creation_tokens": 3400,
        "total_tokens": 1520 + 420 + 3000 + 3400,
    }
    assert usage["cost_usd"] == pytest.approx(
        1520 * OPUS_RATES[0]
        + 420 * OPUS_RATES[1]
        + 3000 * OPUS_RATES[2]
        + 3400 * OPUS_RATES[3]
    )


def _codex_session(home: Path, usages: list[dict], model: str = "gpt-6.1-sol") -> None:
    from tests.test_codex_acp_tokens import _codex_log

    _codex_log(home, usages, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_a_codex_login_trial_cut_off_is_counted_and_priced(
    tmp_path, monkeypatch, harness
):
    """A Codex trial on a ChatGPT login whose turn ended without its report
    (no turn.completed in the native harness, no PromptResponse in ACP):
    Codex's session log counts it, and gpt-6.1-sol's list price prices it."""
    from tests.test_codex_acp_tokens import THREE_CALLS

    home = tmp_path / "sandbox-home"
    _codex_session(home, THREE_CALLS)
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness=harness, home=home
    )
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    fresh = 12000 + 896 + 828
    cached = 11904 + 12672
    output = 300 + 150 + 84
    assert usage["n_input_tokens"] == fresh
    assert usage["n_cache_read_tokens"] == cached
    assert usage["n_output_tokens"] == output
    assert usage["usage_source"] == "session_log"
    assert usage["cost_usd"] == pytest.approx(
        fresh * 2e-6 + cached * 0.1e-6 + output * 10e-6
    )
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "lost"])
async def test_execute_marks_a_turn_that_ended_without_its_report(
    tmp_path, monkeypatch, failure
):
    """execute() records that the turn ended without its usage report, so
    teardown counts it from the agent's logs."""
    from benchflow.acp.session import ACPSession
    from benchflow.diagnostics import (
        AgentPromptTimeoutDiagnostic,
        AgentPromptTimeoutError,
    )

    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=tmp_path
    )
    rollout._session = ACPSession("s")
    rollout._acp_client = object()
    error: BaseException = (
        AgentPromptTimeoutError(
            "Agent prompt exceeded wall-clock budget 5s",
            trajectory=[],
            diagnostic=AgentPromptTimeoutDiagnostic(timeout_sec=5.0),
        )
        if failure == "timeout"
        else ConnectionError("sandbox connection lost")
    )
    monkeypatch.setattr(
        rollout._planes, "execute_prompts", AsyncMock(side_effect=error)
    )
    assert rollout._usage_report_missing is False
    with pytest.raises(type(error)):
        await rollout.execute(["Build the site."])
    assert rollout._usage_report_missing is True


# X2's live repro on rc3 (2026-11-13, ~/benchflow/run/hub/benchflow-e2e-20261113/
# x2.md): Codex ACP on a ChatGPT login, two model calls; the session log's
# running total is 25,887 tokens, and the trial recorded 12,958, its last call.
X2_CODEX_CALLS = [
    {
        "input_tokens": 12883,
        "cached_input_tokens": 11264,
        "cache_write_input_tokens": 0,
        "output_tokens": 46,
        "reasoning_output_tokens": 0,
        "total_tokens": 12929,
    },
    {
        "input_tokens": 12952,
        "cached_input_tokens": 12800,
        "cache_write_input_tokens": 0,
        "output_tokens": 6,
        "reasoning_output_tokens": 0,
        "total_tokens": 12958,
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_x2_codex_success_counts_every_call_on_both_harnesses(
    tmp_path, monkeypatch, harness
):
    """X2: the trial records 25,887 tokens, not its last call's 12,958, on
    either harness; the log's sum replaces the report, never adds to it."""
    from tests.test_codex_acp_tokens import _codex_acp_session, _codex_log

    home = tmp_path / "sandbox-home"
    _codex_log(home, X2_CODEX_CALLS)
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness=harness, home=home
    )
    rollout._session = _codex_acp_session(X2_CODEX_CALLS)
    rollout._collect_native_acp_usage()
    rollout._session = None
    assert rollout._native_usage_metrics["total_tokens"] == 12958

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["total_tokens"] == 25887
    assert usage["n_input_tokens"] == 25835 - 24064
    assert usage["n_cache_read_tokens"] == 24064
    assert usage["n_output_tokens"] == 52
    assert usage["n_cache_creation_tokens"] is None
    assert usage["usage_source"] == "session_log"
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["cost_usd"] == pytest.approx(
        (25835 - 24064) * 2e-6 + 24064 * 0.1e-6 + 52 * 10e-6
    )


@pytest.mark.asyncio
async def test_x2_claude_native_timeout_counts_the_stream_and_keeps_the_log_price(
    tmp_path, monkeypatch
):
    """X2: rc3's native Claude timeout recorded 0 tokens against 16,328 in
    the raw log, while its session-log price ($0.0168253) survived. Now the
    tokens come from the stream log and the cost stays Claude Code's own."""
    home = tmp_path / "sandbox-home"
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 8526,
        "cache_read_input_tokens": 7673,
        "output_tokens": 127,
    }
    _claude_session_log(home, [("claude-sonnet-5-5", usage)])
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    with log.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "cost-state",
                    "totalCostUSD": 0.0168253,
                    "modelUsage": {
                        "claude-sonnet-5-5": {
                            "inputTokens": 2,
                            "outputTokens": 127,
                            "cacheReadInputTokens": 7673,
                            "cacheCreationInputTokens": 8526,
                            "costUSD": 0.0168253,
                        }
                    },
                }
            )
            + "\n"
        )
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    _stream_log(
        trial,
        [
            [
                _start("msg_1", {**usage, "output_tokens": 1}, "claude-sonnet-5-5"),
                _delta(127),
            ]
        ],
    )
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    result = rollout._usage_metrics
    assert result["total_tokens"] == 16328
    assert result["n_output_tokens"] == 127
    assert result["usage_source"] == "stream_log"
    assert result["cost_usd"] == pytest.approx(0.0168253)
    assert result["price_source"] == "agent_session_log"
    assert result["cost_source"] == "list_price_estimate"


@pytest.mark.asyncio
async def test_a_failed_acp_turn_keeps_the_tokens_it_reported():
    """An ACP turn the agent ended on a failure (a typed sessionFailure on the
    prompt response) raised before its usage was recorded: the tokens it
    spent were lost. They are recorded first now."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession
    from benchflow.contracts import AgentProtocolError

    client = ACPClient(None)
    client._session = ACPSession("s")

    async def request(method, params):
        return {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 900, "outputTokens": 30, "totalTokens": 930},
            "_meta": {
                "jetbrains": {
                    "air": {
                        "sessionFailure": {
                            "severity": "error",
                            "category": "provider",
                            "title": "overloaded",
                        }
                    }
                }
            },
        }

    client._send_request = request  # type: ignore[method-assign]
    with pytest.raises(AgentProtocolError):
        await client.prompt("Do the task.")
    latest = client._session.latest_usage_totals()
    assert latest is not None
    assert latest["input_tokens"] == 900 and latest["output_tokens"] == 30


def test_an_empty_cost_state_is_no_cost(tmp_path):
    """a cost-state line with no response and $0 (Claude Code wrote
    its totals before any call answered) gave cost_usd 0.0 beside usage
    unavailable. Nothing was measured: the cost is None."""
    from benchflow._utils.session_cost import estimate_claude_code_cost

    folder = tmp_path / "claude-sessions" / "-app"
    folder.mkdir(parents=True)
    (folder / "s.jsonl").write_text(
        json.dumps({"type": "cost-state", "totalCostUSD": 0, "modelUsage": {}}) + "\n"
    )
    assert estimate_claude_code_cost(tmp_path / "claude-sessions") is None


@pytest.mark.asyncio
async def test_a_native_timeout_stays_a_timeout_with_its_timeout_info(
    tmp_path, monkeypatch
):
    """a native prompt timeout became agent_integration/no_activity
    with no timeout_info: its prompt could not be drained, so it carried no
    diagnostic, and without usage it looked like an agent that did nothing.
    The trial records error_category "timeout", the "wall-clock budget"
    error and agent_timeout_info, and its salvaged output shows activity."""
    rollout, trial = _rollout(
        tmp_path,
        monkeypatch,
        agent="claude-agent-acp",
        harness="native",
        home=tmp_path / "gone",
    )
    _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True
    rollout._executed_prompts = ["Build the site."]
    rollout._trajectory = [{"type": "user_message", "text": "Build the site."}]
    rollout._timeout = 15
    rollout._record_agent_timeout(
        TimeoutError("Agent prompt exceeded wall-clock budget 15s"), agent_phase=True
    )
    rollout._env = None

    await rollout._teardown_finalize_usage()
    await rollout._teardown_price()
    result = rollout._build_result()

    data = json.loads((trial / "result.json").read_text())
    assert result.error_category == "timeout"
    assert "wall-clock budget" in (result.error or "")
    assert data["agent_timeout_info"]["timeout_sec"] == 15
    assert data["agent_result"]["n_output_tokens"] == 121
    assert data.get("integration_failure_info") is None


@pytest.mark.asyncio
async def test_the_timeout_fallback_is_the_prompt_budget_s_only(tmp_path, monkeypatch):
    """The fallback records a prompt's wall-clock timeout that carried
    no diagnostic; a host-deadline stop is not that (it records its own, and
    an infrastructure wedge after it must stay infra_failure), and an unknown
    budget is None, never 0.0."""
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=tmp_path
    )
    rollout._timeout = None
    rollout._record_agent_timeout(
        TimeoutError(
            "Agent timed out: still running at the host hard deadline (900s); "
            "agent stopped so the verifier could run"
        ),
        agent_phase=True,
    )
    assert rollout._diagnostics.get("agent_timeout_info") is None
    rollout._record_agent_timeout(
        TimeoutError("Agent prompt exceeded wall-clock budget 15s"), agent_phase=True
    )
    info = rollout._diagnostics.get("agent_timeout_info")
    assert info is not None and info.timeout_sec is None

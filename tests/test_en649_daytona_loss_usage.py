"""a Daytona sandbox lost mid-turn keeps the usage its model proxy's
log held, and the trial says its count may miss calls.

X4's loss cells (rc 5d64b0d4): each trial's proxy logged its calls, the
live capture read them while the turn ran (they are in the trial's
``trajectory/llm_trajectory.jsonl``), and the sandbox was deleted. At
teardown the log could no longer be read, and result.json and summary.json
recorded no tokens and no cost. Each cell here replays its recorded calls'
usage (tokens and LiteLLM's cost, as X4 retained them; the messages are made
up) through the real ``SandboxLiteLLMProcess`` live capture against a stand-in
sandbox that is then lost, and through the rollout's real teardown and
``_build_result()``.
"""

from __future__ import annotations

import base64
import json
import re
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.runtime import ProviderRuntime

LOST = (
    "DaytonaBadRequestError: Failed to create session: bad request: failed to "
    "resolve container IP after 3 attempts: no IP address found. Is the Sandbox "
    "started? [status_code=400, error_code=SANDBOX_NOT_RUNNING]"
)
REASON = (
    "the in-sandbox model proxy's usage log may miss calls: the sandbox was lost "
    "before the log could be read at the end (RuntimeError: " + LOST + "): the "
    "{calls} call(s) read from it while the turn ran are counted, any after the "
    "last read are not"
)

# Per cell, each recorded call: path, prompt tokens (cache included),
# completion tokens, cached prompt tokens, LiteLLM's response_cost.
CELLS = {
    "pi-acp": [("/v1/chat/completions", 1396, 84, 1280, 4.3344e-05)],
    "claude-agent-acp": [("/v1/messages", 10961, 161, 8192, 0.0004556776)],
    "mimo": [
        ("/v1/chat/completions", 331, 63, 256, 2.88568e-05),
        ("/v1/chat/completions", 24178, 213, 24064, 0.0001429792),
    ],
    "deepagents": [("/v1/chat/completions", 2920, 79, 1024, 0.0002904272)],
    "openhands": [("/v1/chat/completions", 14002, 89, 6144, 0.0011422432)],
}
# X4's retained totals: 1,480 (Pi), 11,122 (Claude), 24,785 (MiMo), 2,999
# (Deepagents), 14,091 (OpenHands).
TOTALS = {
    "pi-acp": 1480,
    "claude-agent-acp": 11122,
    "mimo": 24785,
    "deepagents": 2999,
    "openhands": 14091,
}


def _record(path: str, prompt: int, completion: int, cached: int, cost: float) -> str:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    return json.dumps(
        {
            "event": "success",
            "request": {
                "method": "POST",
                "path": path,
                "body": {
                    "model": "mimo-v2.6-flash",
                    "messages": [{"role": "user", "content": "made up"}],
                },
            },
            "response": {
                "id": "chatcmpl-made-up",
                "model": "mimo-v2.6-flash",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": usage,
            },
            "usage": usage,
            "response_cost": cost,
            "start_time": "2026-11-14T10:00:00",
            "end_time": "2026-11-14T10:00:02",
            "duration_ms": 2000,
            "logged_at": "2026-11-14T10:00:02+00:00",
        },
        separators=(",", ":"),
    )


class _DaytonaSandbox:
    """The proxy's sandbox: its callback log answers the live capture's reads
    until the sandbox is lost; then every exec and download fails as X4's
    deleted Daytona sandbox did."""

    def __init__(self, log: bytes) -> None:
        self.log = log
        self.lost = False

    def _check(self) -> None:
        if self.lost:
            raise RuntimeError(LOST)

    async def exec(self, command: str, timeout_sec=None, **_kwargs):
        self._check()
        read = re.search(r"skip=(\d+) count=(\d+)", command)
        if read and "dd if=" in command:
            offset, count = int(read.group(1)), int(read.group(2))
            payload = base64.b64encode(self.log[offset : offset + count]).decode()
            return SimpleNamespace(
                return_code=0, stdout=f"{len(self.log)}\n{payload}", stderr=""
            )
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    exec_transient = exec

    async def download_file(self, source_path, target_path) -> None:
        self._check()
        raise AssertionError("a live sandbox is not what this test is about")


def _proxy(sandbox: _DaytonaSandbox, agent: str) -> runtime.SandboxLiteLLMProcess:
    return runtime.SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=None,  # type: ignore[arg-type]
        runtime_dir="/opt/runtime/proxy/0123456789abcdef",
        endpoint=runtime.LiteLLMEndpoint("http://127.0.0.1:1", "http://127.0.0.1:1"),
        log_path="/opt/runtime/proxy/0123456789abcdef/callback.jsonl",
        pid_path="/opt/runtime/proxy/0123456789abcdef/litellm.pid",
        stdout_path="/opt/runtime/proxy/0123456789abcdef/stdout.log",
        stderr_path="/opt/runtime/proxy/0123456789abcdef/stderr.log",
        session_id="lost",
        agent_name=agent,
    )


async def _lost_trial(tmp_path, monkeypatch, agent: str, calls):
    from tests.test_en374_salvage import _rollout

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    # The settle wait has nothing to settle on a lost sandbox.
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    home = tmp_path / "home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent=agent, harness="acp", home=home
    )
    log = "".join(_record(*call) + "\n" for call in calls).encode()
    sandbox = _DaytonaSandbox(log)
    proxy = _proxy(sandbox, agent)
    proxy.start_live_capture(trial / "trajectory" / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    sandbox.lost = True
    rollout._usage_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:1",
        backend_model="openai/mimo-v2.6-flash",
        server=proxy,
    )
    rollout._error = LOST
    await rollout._cleanup_before_teardown()
    rollout._build_result()
    return rollout, trial


async def _no_wait(*_args, **_kwargs) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", sorted(CELLS))
async def test_a_lost_daytona_sandbox_keeps_the_usage_its_proxy_logged(
    tmp_path, monkeypatch, agent
):
    """Fails before that change: total_tokens and cost_usd were None, and the
    summary counted the trial as one with no usage."""
    from benchflow._utils.evaluation_results import usage_summary

    calls = CELLS[agent]
    _rollout, trial = await _lost_trial(tmp_path, monkeypatch, agent, calls)
    data = json.loads((trial / "result.json").read_text())
    usage = data["agent_result"]

    assert usage["total_tokens"] == TOTALS[agent]
    assert usage["n_cache_read_tokens"] == sum(c[3] for c in calls)
    assert usage["n_input_tokens"] == sum(c[1] - c[3] for c in calls)
    assert usage["n_output_tokens"] == sum(c[2] for c in calls)
    assert usage["cost_usd"] == pytest.approx(sum(c[4] for c in calls))
    assert usage["usage_source"] == "proxy"
    # The calls that ran after the last read, or still ran, are not counted.
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == REASON.format(calls=len(calls))
    assert usage["cost_incomplete"] is True

    summary = usage_summary({trial.name: data})
    assert summary["total_tokens"] == TOTALS[agent]
    assert summary["trials_without_usage"] == 0
    assert summary["trials_without_cost"] == 0
    assert (summary["trials_usage_incomplete"], summary["trials_cost_incomplete"]) == (
        1,
        1,
    )
    # The published llm_trajectory.jsonl keeps the exchanges read, whole.
    mirror = (trial / "trajectory" / "llm_trajectory.jsonl").read_text().splitlines()
    assert len(mirror) == len(calls)
    assert json.loads(mirror[0])["request"]["body"]["messages"]


@pytest.mark.asyncio
async def test_a_sandbox_lost_before_any_call_was_read_is_unknown_not_zero(
    tmp_path, monkeypatch
):
    _rollout, trial = await _lost_trial(tmp_path, monkeypatch, "pi-acp", [])
    usage = json.loads((trial / "result.json").read_text())["agent_result"]
    assert usage["total_tokens"] is None and usage["cost_usd"] is None
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == REASON.format(calls=0)
    assert "cost_incomplete" not in usage


@pytest.mark.asyncio
async def test_a_live_capture_s_reader_imports_what_a_whole_read_would(tmp_path):
    """What the live capture read is the import of the log so far: the same
    exchanges, tokens and cost as reading those lines at the stop."""
    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    lines = [_record(*call) for call in CELLS["mimo"]]
    sandbox = _DaytonaSandbox("".join(line + "\n" for line in lines).encode())
    proxy = _proxy(sandbox, "mimo")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    await proxy._stop_live_capture()
    whole = trajectory_from_litellm_callback_log(
        "\n".join(lines), session_id="lost", agent_name="mimo"
    )
    live = proxy._live_reader.finish()
    assert extract_usage_from_trajectory(live) == extract_usage_from_trajectory(whole)
    assert live.metadata["cost_usd"] == whole.metadata["cost_usd"]
    assert len(live.exchanges) == len(whole.exchanges) == 2


@pytest.mark.asyncio
async def test_a_stop_that_lands_mid_chunk_keeps_every_line_already_read(
    tmp_path, monkeypatch
):
    """RUSAGE L1: three calls in one chunk, each parse slow (a large record's
    redaction); the sandbox is lost and stop() lands while the second is
    parsed. Every call read is counted, and the reason says three."""
    import asyncio
    import time

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    original = runtime._read_live_line

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_read_live_line", slow)
    calls = CELLS["mimo"] + CELLS["pi-acp"]
    sandbox = _DaytonaSandbox("".join(_record(*c) + "\n" for c in calls).encode())
    proxy = _proxy(sandbox, "mimo")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    for _ in range(100):
        if proxy._live_callback_offset:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.4)  # the first line parsed, the second in flight
    assert len(proxy._live_reader.trajectory.exchanges) < 3
    sandbox.lost = True
    await proxy.stop()
    assert len(proxy.trajectory.exchanges) == 3
    assert proxy.usage_log_shortfall is not None
    assert "the 3 call(s) read from it" in proxy.usage_log_shortfall


class _AnsweringSandbox(_DaytonaSandbox):
    """A sandbox that still runs commands, whose log alone cannot be read."""

    async def exec(self, command: str, timeout_sec=None, **_kwargs):
        if command == "true":
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        if self.lost and command.startswith("cat "):
            return SimpleNamespace(return_code=255, stdout="", stderr="")
        return await super().exec(command, timeout_sec=timeout_sec)

    async def download_file(self, source_path, target_path) -> None:
        raise RuntimeError("download failed")


@pytest.mark.asyncio
async def test_a_failed_read_on_a_sandbox_that_answers_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    sandbox = _AnsweringSandbox(_record(*CELLS["pi-acp"][0]).encode() + b"\n")
    proxy = _proxy(sandbox, "pi-acp")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    sandbox.lost = True  # only the log read fails now
    await proxy.stop()
    assert proxy.usage_log_shortfall == (
        "the log could not be read at the end, though the sandbox answers "
        "(RuntimeError: the callback log could not be read (exit 255)): the 1 "
        "call(s) read from it while the turn ran are counted, any after the last "
        "read are not"
    )


def test_another_step_s_reason_is_kept_beside_the_proxy_s():
    """When another count step marked the trial too (an interrupted stream),
    both reasons are kept, and marking again changes nothing."""
    from benchflow.rollout._cost import mark_incomplete_usage

    proxy = REASON.format(calls=1)
    trial = SimpleNamespace(
        _usage_metrics={
            "usage_source": "provider_response",
            "total_tokens": 9,
            "usage_incomplete": True,
            "usage_incomplete_reason": "a stream was cut before its usage",
        },
        _proxy_usage_incomplete_reason=proxy,
    )
    mark_incomplete_usage(trial)
    reason = trial._usage_metrics["usage_incomplete_reason"]
    assert reason == f"a stream was cut before its usage; {proxy}"
    mark_incomplete_usage(trial)
    assert trial._usage_metrics["usage_incomplete_reason"] == reason

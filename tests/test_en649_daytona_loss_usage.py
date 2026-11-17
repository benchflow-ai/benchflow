"""EN-649: a Daytona sandbox lost mid-turn keeps the usage its model proxy's
log held, and the trial says its count may miss calls.

X4's loss cells (rel-0.9.x 53cad31d): each trial's proxy logged its calls, the
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
# completion tokens, cached prompt tokens (None: not reported), LiteLLM's
# response_cost. X4's rel proxy priced MiMo (its rel cells carry
# price_source litellm); these are X4's figures.
CELLS = {
    "pi-acp": [("/v1/chat/completions", 1351, 77, 1280, 0.000035084)],
    "claude-agent-acp": [("/v1/messages", 16530, 97, 14336, 0.0003744608)],
    "mimo": [("/v1/chat/completions", 331, 90, 256, 0.0000364168)],
    "openhands": [("/v1/chat/completions", 14003, 168, None, 0.00200746)],
}
# X4's retained totals: 1,428 (Pi), 16,627 (Claude), 421 (MiMo), 14,171
# (OpenHands). Its Deepagents cell was lost before any call was read.
TOTALS = {"pi-acp": 1428, "claude-agent-acp": 16627, "mimo": 421, "openhands": 14171}


def _record(
    path: str,
    prompt: int,
    completion: int,
    cached: int | None,
    cost: float | None,
    provider_cost: float | None = None,
) -> str:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": None if cached is None else {"cached_tokens": cached},
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
            **({"provider_cost": provider_cost} if provider_cost is not None else {}),
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
    await _read_live(proxy, sandbox)
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


async def _read_live(proxy, sandbox: _DaytonaSandbox) -> None:
    """Until the live capture has read the whole log (it parses each read
    in the same step)."""
    import asyncio

    for _ in range(250):
        if getattr(proxy, "_live_callback_offset", 0) >= len(sandbox.log):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the live capture did not read the log")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", sorted(CELLS))
async def test_a_lost_daytona_sandbox_keeps_the_usage_its_proxy_logged(
    tmp_path, monkeypatch, agent
):
    """Fails before EN-649: total_tokens and cost_usd were None, and the
    summary counted the trial as one with no usage."""
    from benchflow._utils.evaluation_results import usage_summary

    calls = CELLS[agent]
    _rollout, trial = await _lost_trial(tmp_path, monkeypatch, agent, calls)
    data = json.loads((trial / "result.json").read_text())
    usage = data["agent_result"]

    assert usage["total_tokens"] == TOTALS[agent]
    assert usage["n_input_tokens"] == sum(c[1] - (c[3] or 0) for c in calls)
    assert usage["n_output_tokens"] == sum(c[2] for c in calls)
    assert usage["cost_usd"] == pytest.approx(sum(c[4] for c in calls))
    # The calls that ran after the last read, or still ran, are not counted.
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == REASON.format(calls=len(calls))
    assert usage["cost_incomplete"] is True

    summary = usage_summary({trial.name: data})
    assert summary["total_tokens"] == TOTALS[agent]
    assert summary["trials_without_usage"] == 0
    assert summary["trials_without_cost"] == 0
    # rel's summary has no incomplete-trial count (a reader rc has).
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
async def test_an_unpriced_model_keeps_an_unknown_cost_not_a_floor(
    tmp_path, monkeypatch
):
    """A proxy that did not price the model: rel prices no gateway count
    itself, so the tokens are kept and the cost stays unknown, never 0."""
    calls = [("/v1/chat/completions", 1351, 77, 1280, None)]
    _rollout, trial = await _lost_trial(tmp_path, monkeypatch, "pi-acp", calls)
    usage = json.loads((trial / "result.json").read_text())["agent_result"]
    assert usage["total_tokens"] == 1428
    assert usage["cost_usd"] is None
    assert usage["usage_incomplete"] is True
    assert "cost_incomplete" not in usage


async def _salvaged(tmp_path, lines: list[str]):
    sandbox = _DaytonaSandbox("".join(line + "\n" for line in lines).encode())
    proxy = _proxy(sandbox, "mimo")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await _read_live(proxy, sandbox)
    await proxy._stop_live_capture()
    sandbox.lost = True
    assert proxy._salvage_live_capture(RuntimeError(LOST))
    return proxy.trajectory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls",
    [
        # LiteLLM's price only.
        [
            ("/v1/chat/completions", 331, 63, 256, 2.88568e-05),
            ("/v1/chat/completions", 24178, 213, 24064, 0.0001429792),
        ],
        # The provider's bill on both calls (RUSAGE L2: the salvage gave
        # LiteLLM's $0.0018 where the provider billed $0.0023).
        [
            ("/v1/chat/completions", 1000, 50, 0, 0.0009, 0.0011),
            ("/v1/chat/completions", 1200, 60, 0, 0.0009, 0.0012),
        ],
        # A bill on one call only: the other at LiteLLM's price.
        [
            ("/v1/chat/completions", 1000, 50, 0, 0.0009, 0.0011),
            ("/v1/chat/completions", 1200, 60, 0, 0.0009),
        ],
    ],
    ids=["litellm", "billed", "mixed"],
)
async def test_the_salvage_imports_what_a_whole_read_of_the_lines_does(tmp_path, calls):
    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    lines = [_record(*call) for call in calls]
    salvaged = await _salvaged(tmp_path, lines)
    whole = trajectory_from_litellm_callback_log(
        "\n".join(lines), session_id="lost", agent_name="mimo"
    )
    assert extract_usage_from_trajectory(salvaged) == (
        extract_usage_from_trajectory(whole)
    )
    assert salvaged.metadata.get("provider_cost_usd") == whole.metadata.get(
        "provider_cost_usd"
    )
    if all(len(call) == 6 for call in calls):
        usage = extract_usage_from_trajectory(salvaged)
        assert usage["cost_usd"] == pytest.approx(0.0023)
        assert usage["cost_source"] == "provider"


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
    await _read_live(proxy, sandbox)
    sandbox.lost = True  # only the log read fails now
    await proxy.stop()
    assert proxy.usage_log_shortfall == (
        "the log could not be read at the end, though the sandbox answers "
        "(RuntimeError: the callback log could not be read (exit 255)): the 1 "
        "call(s) read from it while the turn ran are counted, any after the last "
        "read are not"
    )


def test_another_step_s_reason_is_kept_beside_the_proxy_s():
    """When another count step marked the trial too, both reasons are kept,
    and marking again changes nothing."""
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

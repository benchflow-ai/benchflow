from __future__ import annotations

import contextlib
import json
import logging
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_runtime import (
    HostLiteLLMProcess,
    LiteLLMEndpoint,
    SandboxLiteLLMProcess,
)
from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


def _trajectory(*, content: str = "ok") -> Trajectory:
    trajectory = Trajectory(session_id="run", agent_name="opencode")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(
                timestamp=datetime(2026, 7, 11),
                body={
                    "messages": [{"role": "user", "content": "hello"}],
                    "api_key": "sk-secret",
                },
            ),
            response=LLMResponse(
                timestamp=datetime(2026, 7, 11),
                body={"choices": [{"message": {"content": content}}]},
            ),
            duration_ms=12,
        )
    )
    return trajectory


def _callback_line(*, content: str = "ok", usage: dict | None = None) -> str:
    # Mirrors BenchFlowLiteLLMLogger.async_log_success_event's record shape:
    # ``usage`` rides both top-level and (when present) inside the response
    # body, exactly as the sandbox callback.jsonl contains it.
    record = {
        "event": "success",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "hello"}],
                "api_key": "sk-secret",
            },
        },
        "response": {"choices": [{"message": {"content": content}}]},
        "start_time": "2026-07-11T00:00:00Z",
        "end_time": "2026-07-11T00:00:01Z",
        "duration_ms": 1000,
    }
    if usage is not None:
        record["usage"] = usage
        record["response"]["usage"] = usage
    return json.dumps(record, separators=(",", ":")) + "\n"


def test_writer_redacts_and_atomically_replaces_snapshot(tmp_path):
    """Guards live redaction and atomic replacement from commit c86adfb."""
    path = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)

    assert writer.write(_trajectory()) is True

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert "sk-secret" not in path.read_text()
    assert rows[0]["request"]["body"]["api_key"] == "***REDACTED***"
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_writer_persists_exchange_metadata_for_training_exports(tmp_path):
    """Guards PR #925: live llm_trajectory rows retain call-purpose metadata."""
    path = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)
    trajectory = _trajectory()
    trajectory.exchanges[0].metadata = {
        "call_purpose": "agent",
        "request_model": "benchflow-model",
    }

    assert writer.write(trajectory) is True

    row = json.loads(path.read_text())
    assert row["metadata"] == {
        "call_purpose": "agent",
        "request_model": "benchflow-model",
    }


def test_writer_deduplicates_unchanged_snapshot_and_reconciles(tmp_path):
    """Guards snapshot deduplication and reconciliation from commit c86adfb."""
    path = tmp_path / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)
    trajectory = _trajectory()

    assert writer.write(trajectory) is True
    assert writer.write(trajectory) is False

    trajectory.exchanges.extend(_trajectory(content="second").exchanges)
    assert writer.reconcile(trajectory) is True
    assert len(path.read_text().splitlines()) == 2


def test_writer_does_not_create_empty_live_artifact(tmp_path):
    """Guards empty-artifact suppression from commit c86adfb."""
    path = tmp_path / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)

    assert writer.write(Trajectory(session_id="run")) is False
    assert not path.exists()


@pytest.mark.asyncio
async def test_host_proxy_mirrors_callback_before_stop(tmp_path, monkeypatch):
    """Guards host-side live callback mirroring from commit c86adfb."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    log_path = tmp_path / "callback.jsonl"
    output_path = tmp_path / "rollout" / "trajectory" / "llm_trajectory.jsonl"
    process = HostLiteLLMProcess(
        route=SimpleNamespace(),
        process=SimpleNamespace(poll=lambda: None),
        runtime_dir=tmp_path,
        endpoint=LiteLLMEndpoint("http://agent", "http://local"),
        log_path=log_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        session_id="run",
        agent_name="opencode",
    )

    process.start_live_capture(output_path)
    log_path.write_text(_callback_line())
    for _ in range(50):
        if output_path.exists():
            break
        await runtime_mod.asyncio.sleep(0.01)
    await process._stop_live_capture()

    assert output_path.exists()
    assert len(output_path.read_text().splitlines()) == 1
    assert "sk-secret" not in output_path.read_text()


class _SandboxWithCallbackLog:
    """A sandbox whose exec channel serves byte ranges of a canned log.

    Mirrors the wire shape the real reader parses: ``<size>\\n<base64 range>``,
    where ``size`` is the whole log (what ``stat -c %s`` reports) and the
    payload is only the requested window.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[tuple[int, int]] = []

    async def exec(self, command: str, timeout_sec: int):
        del timeout_sec
        skip = int(re.search(r"skip=(\d+)", command).group(1))
        count = int(re.search(r"count=(\d+)", command).group(1))
        self.reads.append((skip, count))
        import base64

        encoded = base64.b64encode(self.data[skip : skip + count]).decode()
        return SimpleNamespace(
            return_code=0, stdout=f"{len(self.data)}\n{encoded}", stderr=""
        )


class _TransientSandboxWithCallbackLog(_SandboxWithCallbackLog):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.transient_calls = 0

    async def exec(self, command: str, timeout_sec: int):
        raise AssertionError("Daytona callback polling must use transient exec")

    async def exec_transient(self, command: str, timeout_sec: int):
        self.transient_calls += 1
        return await super().exec(command, timeout_sec)


@pytest.mark.asyncio
async def test_daytona_proxy_incrementally_mirrors_callback(tmp_path, monkeypatch):
    """Guards incremental Daytona callback mirroring from commit c86adfb."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    sandbox = _SandboxWithCallbackLog(_callback_line(content="first").encode())
    output_path = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    process = SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=SimpleNamespace(),
        runtime_dir="/tmp/runtime",
        endpoint=LiteLLMEndpoint("http://agent", "http://local"),
        log_path="/tmp/runtime/callback.jsonl",
        pid_path="/tmp/runtime/pid",
        stdout_path="/tmp/runtime/stdout",
        stderr_path="/tmp/runtime/stderr",
        session_id="run",
        agent_name="opencode",
    )

    process.start_live_capture(output_path)
    for _ in range(50):
        if output_path.exists():
            break
        await runtime_mod.asyncio.sleep(0.01)
    sandbox.data += _callback_line(content="second").encode()
    for _ in range(50):
        if output_path.exists() and len(output_path.read_text().splitlines()) == 2:
            break
        await runtime_mod.asyncio.sleep(0.01)
    await process._stop_live_capture()

    assert len(output_path.read_text().splitlines()) == 2


def _sandbox_process(sandbox) -> SandboxLiteLLMProcess:
    return SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=SimpleNamespace(),
        runtime_dir="/tmp/runtime",
        endpoint=LiteLLMEndpoint("http://agent", "http://local"),
        log_path="/tmp/runtime/callback.jsonl",
        pid_path="/tmp/runtime/pid",
        stdout_path="/tmp/runtime/stdout",
        stderr_path="/tmp/runtime/stderr",
        session_id="run",
        agent_name="prime-agent",
    )


async def _wait_for(predicate, *, ticks: int = 100) -> bool:
    for _ in range(ticks):
        if predicate():
            return True
        await runtime_mod.asyncio.sleep(0.01)
    return predicate()


@pytest.mark.asyncio
async def test_live_capture_accumulates_usage_tokens_mid_run(tmp_path, monkeypatch):
    """Guards the mid-prompt live-token counter for the eval dashboard.

    A single-prompt rollout has no ACP usage snapshot until the prompt
    completes, so the dashboard's only mid-run token signal is the gateway's
    live callback capture: ``live_usage_tokens()`` must be None before any
    usage-bearing record, then advance per mirrored LLM request using the
    same canonical parser/accounting scoring uses, and freeze (no leaked
    poller) after ``_stop_live_capture``.
    """
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    sandbox = _SandboxWithCallbackLog(
        _callback_line(
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ).encode()
    )
    process = _sandbox_process(sandbox)
    assert process.live_usage_tokens() is None  # before capture starts

    process.start_live_capture(tmp_path / "trajectory" / "llm_trajectory.jsonl")
    assert await _wait_for(lambda: process.live_usage_tokens() == 120)

    sandbox.data += _callback_line(
        content="second",
        usage={"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80},
    ).encode()
    assert await _wait_for(lambda: process.live_usage_tokens() == 200)

    await process._stop_live_capture()
    sandbox.data += _callback_line(
        content="late", usage={"prompt_tokens": 1, "completion_tokens": 1}
    ).encode()
    await runtime_mod.asyncio.sleep(0.05)
    assert process.live_usage_tokens() == 200  # cancelled: no further advance


@pytest.mark.asyncio
async def test_live_usage_stays_none_without_usage_records(tmp_path, monkeypatch):
    """Usage-less successes, garbage lines, and failure records: the counter
    must stay None (the dashboard's "no signal yet"), never a fake zero, while
    the exchange mirror itself keeps working."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    failure = (
        json.dumps(
            {
                "event": "failure",
                "request": {"method": "POST", "path": "/v1/chat/completions"},
                "error": {"type": "Timeout", "message": "boom"},
                "start_time": "2026-07-11T00:00:00Z",
                "end_time": "2026-07-11T00:00:01Z",
            }
        )
        + "\n"
    )
    sandbox = _SandboxWithCallbackLog(
        _callback_line().encode() + b"not json at all\n" + failure.encode()
    )
    process = _sandbox_process(sandbox)
    output_path = tmp_path / "trajectory" / "llm_trajectory.jsonl"

    process.start_live_capture(output_path)
    assert await _wait_for(output_path.exists)
    await process._stop_live_capture()

    assert process.live_usage_tokens() is None


@pytest.mark.asyncio
async def test_live_usage_survives_exec_failure_silently(monkeypatch, tmp_path):
    """A raising exec channel (sandbox teardown race, transient Daytona error)
    must degrade to "no signal" — never propagate into the capture loop's
    caller or the dashboard read."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)

    class _BrokenSandbox:
        async def exec(self, command: str, timeout_sec: int):
            raise RuntimeError("exec channel down")

    process = _sandbox_process(_BrokenSandbox())
    process.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await runtime_mod.asyncio.sleep(0.05)
    await process._stop_live_capture()  # must not raise

    assert process.live_usage_tokens() is None


@pytest.mark.asyncio
async def test_live_usage_resets_when_capture_restarts_on_new_path(
    tmp_path, monkeypatch
):
    """A fresh capture target (new rollout attempt) must not inherit the
    previous attempt's token total."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    sandbox = _SandboxWithCallbackLog(
        _callback_line(
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        ).encode()
    )
    process = _sandbox_process(sandbox)
    process.start_live_capture(tmp_path / "a" / "llm_trajectory.jsonl")
    assert await _wait_for(lambda: process.live_usage_tokens() == 10)
    await process._stop_live_capture()

    # Grow the log so attempt b's true total (25) differs from attempt a's
    # stale value (10) — without this, a missing reset would still converge
    # to the expected number and the test could not see it.
    sandbox.data += _callback_line(
        content="second",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    ).encode()
    process.start_live_capture(tmp_path / "b" / "llm_trajectory.jsonl")
    # The reset's synchronous observable: BEFORE the new capture task has run,
    # the stale total from attempt a must already read as "no signal" — not 10.
    assert process.live_usage_tokens() is None
    try:
        # The counter restarts from the log's beginning (offset reset), so it
        # re-converges to the log's true total (10 + 15), not 10 + 25.
        assert await _wait_for(lambda: process.live_usage_tokens() == 25)
    finally:
        await process._stop_live_capture()


@pytest.mark.asyncio
async def test_gateway_live_tokens_reach_rollout_activity_snapshot(
    tmp_path, monkeypatch
):
    """Cross-boundary seam guard: a REAL SandboxLiteLLMProcess fed a canned
    callback log, wrapped in a REAL ProviderRuntime, must surface its live
    total through Rollout.activity_snapshot — pinning the
    ``server.live_usage_tokens`` accessor name across the provider/rollout
    boundary, where the unit tests on either side use fakes (a rename on
    either side fails HERE, not silently in production)."""
    from benchflow.providers.runtime import ProviderRuntime
    from benchflow.rollout import Rollout

    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_INTERVAL_SEC", 0.01)
    sandbox = _SandboxWithCallbackLog(
        _callback_line(
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ).encode()
    )
    process = _sandbox_process(sandbox)
    process.start_live_capture(tmp_path / "trajectory" / "llm_trajectory.jsonl")
    try:
        assert await _wait_for(lambda: process.live_usage_tokens() == 120)

        class _MidPromptSession:
            # A single-prompt ACP session mid-prompt: tool calls exist, the
            # ACP usage snapshot does not (it lands only at prompt end).
            distinct_tool_titles = 1

            def progress_snapshot(self):
                return 3, "IPython cell"

            def latest_usage_totals(self):
                return None

        rollout = SimpleNamespace(
            _acp_client=SimpleNamespace(session=_MidPromptSession()),
            _phase="connected",
            _usage_runtime=ProviderRuntime(
                kind="litellm",
                agent_base_url="http://127.0.0.1:4000",
                server=process,
            ),
        )
        snap = Rollout.activity_snapshot(rollout)
    finally:
        await process._stop_live_capture()

    assert snap.counters is not None
    assert snap.counters.total_tokens == 120


@pytest.mark.asyncio
async def test_daytona_proxy_uses_transient_exec_for_callback_poll(tmp_path):
    """Guards the Daytona live-capture session fix in PR #921."""
    sandbox = _TransientSandboxWithCallbackLog(_callback_line().encode())
    process = SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=SimpleNamespace(),
        runtime_dir="/tmp/runtime",
        endpoint=LiteLLMEndpoint("http://agent", "http://local"),
        log_path="/tmp/runtime/callback.jsonl",
        pid_path="/tmp/runtime/pid",
        stdout_path="/tmp/runtime/stdout",
        stderr_path="/tmp/runtime/stderr",
        session_id="run",
        agent_name="openhands",
    )

    chunk = await process._read_callback_chunk(0, 24 * 1024)

    assert chunk.data
    assert sandbox.transient_calls == 1


# --------------------------------------------------------------------------- #
# Capture throughput: the tail must not fall permanently behind a fast log.
# --------------------------------------------------------------------------- #


class _RecordingSandbox(_SandboxWithCallbackLog):
    """Records the raw command so the read *shape* can be asserted."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.commands: list[str] = []

    async def exec_transient(self, command: str, timeout_sec: int):
        self.commands.append(command)
        return await super().exec(command, timeout_sec)


async def _arm_capture(process, path) -> None:
    """Arm the live capture but suppress its poller, so a test steps ticks.

    ``start_live_capture`` creates the poll task and returns without awaiting,
    so cancelling it here guarantees it never runs a tick — each subsequent
    ``_capture_live_records()`` is then exactly one poll, with no timing race.
    """
    process.start_live_capture(path)
    task = process._live_capture_task
    task.cancel()
    with contextlib.suppress(runtime_mod.asyncio.CancelledError):
        await task
    process._live_capture_task = None


@pytest.mark.asyncio
async def test_callback_read_is_byte_ranged_with_a_block_sized_transfer(tmp_path):
    """The read must select its range by *seeking*, not by copying byte by byte.

    ``dd bs=1 skip=<offset> count=<limit>`` — the shipped form — issues two
    syscalls per byte transferred, and each read costs a whole sandbox exec
    round trip on top. That capped the tail's throughput at one small window
    per round trip, which a full-message provider log outgrows. Pin the shape:
    a byte-ranged transfer with a block size in the KB range, reporting the
    log's size alongside the requested window.
    """
    payload = bytes(range(256)) * 64  # 16KB of every byte value
    sandbox = _RecordingSandbox(payload)
    process = _sandbox_process(sandbox)

    chunk = await process._read_callback_chunk(4096, 2048)

    assert chunk.data == payload[4096:6144]
    assert chunk.size == len(payload)
    command = sandbox.commands[0]
    assert "bs=1 " not in command  # the per-byte form must be gone
    assert "iflag=skip_bytes,count_bytes" in command
    assert f"bs={runtime_mod._LIVE_CAPTURE_READ_BLOCK_BYTES}" in command
    assert runtime_mod._LIVE_CAPTURE_READ_BLOCK_BYTES >= 4096


@pytest.mark.asyncio
async def test_live_counter_keeps_advancing_when_log_outgrows_a_tick(
    tmp_path, monkeypatch
):
    """The #965 flatline regression, unit-shaped.

    The dogfood that motivated this fix watched the footer climb to 56.8k
    tokens and then hold that exact value for 20 minutes of a live run whose
    final total was 1.70M — 3.3% of the truth, rendered as if it were current.
    The mechanism: the gateway log grew faster than one poll's read budget
    drained it. Drive that state directly — a log growing at ~2x the per-tick
    budget — and require the counter to keep advancing every tick and to
    converge on the true total once the log stops growing.
    """
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_CHUNK_BYTES", 256)
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_MAX_READS_PER_TICK", 4)
    tick_budget = 256 * 4

    record = _callback_line(
        content="x" * 300,
        usage={"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000},
    ).encode()
    assert len(record) * 3 > tick_budget  # the log really does outrun a tick

    sandbox = _RecordingSandbox(record)
    process = _sandbox_process(sandbox)
    await _arm_capture(process, tmp_path / "trajectory" / "llm_trajectory.jsonl")

    growth_ticks, total_records = 8, 1
    seen: list[int | None] = []
    for tick in range(48):
        await process._capture_live_records()
        seen.append(process.live_usage_tokens())
        if tick < growth_ticks:
            sandbox.data += record * 3
            total_records += 3

    advancing = [t for t in seen[:growth_ticks] if t is not None]
    assert advancing == sorted(advancing)
    # The counter must not sit on one value while the log grows — that is the
    # frozen-but-plausible figure users read to decide whether to kill a run.
    assert len(set(advancing)) > 1
    assert advancing[-1] > advancing[0]
    # ...and once growth stops it converges on the log's true total.
    assert seen[-1] == total_records * 1000
    # Each round trip carried a full window: draining N bytes must not cost
    # more round trips than N/window (+1 for the poll that finds EOF per tick).
    max_reads = -(-len(sandbox.data) // 256) + len(seen)
    assert len(sandbox.reads) <= max_reads
    assert all(count == 256 for _, count in sandbox.reads)


class _TimingOutSandbox(_SandboxWithCallbackLog):
    """Reads fail the way `timeout <n> ...` kills one: non-zero, no payload."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.failing = False

    async def exec_transient(self, command: str, timeout_sec: int):
        if self.failing:
            return SimpleNamespace(return_code=124, stdout="", stderr="")
        return await super().exec(command, timeout_sec)


class _RaisingSandbox(_TimingOutSandbox):
    """The other half of the same failure: the exec poll's own deadline."""

    async def exec_transient(self, command: str, timeout_sec: int):
        if self.failing:
            raise RuntimeError("Command timed out after 20 seconds")
        return await _SandboxWithCallbackLog.exec(self, command, timeout_sec)


@pytest.mark.parametrize("sandbox_cls", [_TimingOutSandbox, _RaisingSandbox])
@pytest.mark.asyncio
async def test_stalled_reader_is_reported_as_lag_not_as_a_drained_log(
    sandbox_cls, tmp_path, monkeypatch, caplog
):
    """A read that fails must register as *behind*, loudly — never as caught up.

    The shipped reader returned bare bytes, so a failed read (the sandbox
    ``timeout`` killing the command, a truncated transport) was indistinguishable
    from a drained log: the poll ended, the offset stayed put, and the token
    figure kept rendering its last value with nothing in the logs above debug.
    A read that *raised* was worse still — the loop's blanket handler swallowed
    the whole tick before any lag was recorded. Now both shapes are a known lag,
    and a lag that stops advancing entirely escalates once — and only once — to
    a warning.
    """
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_STALL_WARN_TICKS", 3)

    sandbox = sandbox_cls(
        _callback_line(
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ).encode()
    )
    process = _sandbox_process(sandbox)
    await _arm_capture(process, tmp_path / "llm_trajectory.jsonl")

    await process._capture_live_records()
    assert process.live_usage_tokens() == 120
    assert process._live_capture_lag_bytes == 0  # drained

    # The log grows; every read now fails. The counter holds its stale value —
    # that is unavoidable — but the state is no longer silent.
    sandbox.data += _callback_line(
        content="second",
        usage={"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80},
    ).encode()
    sandbox.failing = True
    with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
        for _ in range(6):
            await process._capture_live_records()

    assert process.live_usage_tokens() == 120  # frozen, as observed in #965
    assert process._live_capture_lag_bytes is None  # behind by an unknown amount
    assert process._live_capture_stall_ticks == 6
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # latched: a stall must not flood the log
    assert "stalled" in warnings[0].getMessage()

    # Recovery clears the signal and the counter catches up.
    sandbox.failing = False
    await process._capture_live_records()
    assert process.live_usage_tokens() == 200
    assert process._live_capture_lag_bytes == 0
    assert process._live_capture_stall_ticks == 0


@pytest.mark.asyncio
async def test_partial_read_of_a_growing_log_is_recorded_as_measured_lag(
    tmp_path, monkeypatch
):
    """Behind-but-advancing must be quantified, and must not latch a warning."""
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_CHUNK_BYTES", 128)
    monkeypatch.setattr(runtime_mod, "_LIVE_CAPTURE_MAX_READS_PER_TICK", 1)

    record = _callback_line(
        content="y" * 400,
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    ).encode()
    sandbox = _RecordingSandbox(record)
    process = _sandbox_process(sandbox)
    await _arm_capture(process, tmp_path / "llm_trajectory.jsonl")

    await process._capture_live_records()

    assert process._live_capture_lag_bytes == len(record) - 128
    assert process._live_capture_stall_ticks == 0  # advancing, so not a stall
    assert process._live_capture_stall_warned is False

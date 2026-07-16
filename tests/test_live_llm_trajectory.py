from __future__ import annotations

import json
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


def _callback_line(*, content: str = "ok") -> str:
    return (
        json.dumps(
            {
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
            },
            separators=(",", ":"),
        )
        + "\n"
    )


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
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def exec(self, command: str, timeout_sec: int):
        del timeout_sec
        skip = int(re.search(r"skip=(\d+)", command).group(1))
        count = int(re.search(r"count=(\d+)", command).group(1))
        import base64

        encoded = base64.b64encode(self.data[skip : skip + count]).decode()
        return SimpleNamespace(return_code=0, stdout=encoded)


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

    assert chunk
    assert sandbox.transient_calls == 1

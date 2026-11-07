"""Guards a report that a proxied run's LLM log filled the
runner: on 0.9 a 52 MB Codex callback log grew the runner by
about 470 MB.

The live mirror held every parsed exchange plus a raw and a redacted copy of
each record and rebuilt the whole file as one string on every write; the stop
read the log as one string and parsed a second full trajectory; the rollout's
second writer serialized everything again. Each record is now serialized and
redacted once and appended, the stop reads the log line by line and only
repairs what the mirror missed, a stopped proxy keeps lean exchanges, and the
second writer finds the file already published.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
from benchflow.providers.litellm_runtime import (
    HostLiteLLMProcess,
    LiteLLMEndpoint,
    extract_usage,
)
from benchflow.rollout import Rollout
from benchflow.trajectories import _llm_capture
from benchflow.trajectories._capture import _parse_provider_tool_evidence
from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from benchflow.trajectories.types import redact_trajectory_obj

_MEMORY_BUDGET_MB = 48


def _record(n: int, body: dict, *, usage: dict | None = None) -> dict:
    usage = usage or {"prompt_tokens": 100 * n, "completion_tokens": 10}
    return {
        "event": "success",
        "request_model": body["model"],
        "call_type": "acompletion",
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": body},
        "response": {"id": f"r{n}", "choices": [], "usage": usage},
        "usage": usage,
        "response_cost": 0.01 + n / 1000,
        "start_time": f"2026-10-03T01:{n:02d}:00+00:00",
        "end_time": f"2026-10-03T01:{n:02d}:07+00:00",
        "duration_ms": 7000.0,
    }


def _chat_log(path: Path, records: int = 5) -> list[str]:
    """A chat-completions log whose requests carry tool calls and results."""
    messages: list[dict] = [{"role": "user", "content": "solve it"}]
    lines = []
    for n in range(records):
        messages = [
            *messages,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": f"ls {n}"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{n}", "content": f"out {n}"},
        ]
        body = {"model": "m", "messages": messages, "api_key": "sk-synthetic-secret"}
        lines.append(json.dumps(_record(n, body), separators=(",", ":")) + "\n")
    path.write_text("".join(lines))
    return lines


def _process(tmp_path: Path, log: Path) -> HostLiteLLMProcess:
    return HostLiteLLMProcess(
        route=types.SimpleNamespace(),
        process=types.SimpleNamespace(poll=lambda: 0),
        runtime_dir=tmp_path / "runtime",
        endpoint=LiteLLMEndpoint("http://agent", "http://local"),
        log_path=log,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        session_id="run",
        agent_name="opencode",
    )


async def _mirror(process: HostLiteLLMProcess, published: Path) -> None:
    """Start the live mirror without its polling task, then drain the log."""
    process.start_live_capture(published)
    process._live_capture_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await process._live_capture_task
    while process._live_callback_offset < process.log_path.stat().st_size:
        await process._capture_live_records()


def _expected(text: str, agent_name: str = "opencode") -> str:
    """The file the whole-log parse and its redaction give."""
    trajectory = trajectory_from_litellm_callback_log(
        text, session_id="run", agent_name=agent_name
    )
    return "\n".join(
        json.dumps(redact_trajectory_obj(e.model_dump(mode="json")))
        for e in trajectory.exchanges
    )


async def test_the_published_file_is_the_whole_logs_with_missed_lines_added(tmp_path):
    log = tmp_path / "callback.jsonl"
    lines = _chat_log(log)
    log.write_text("".join(lines[:3]))
    published = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    process = _process(tmp_path, log)
    await _mirror(process, published)
    assert len(published.read_text().splitlines()) == 3
    # The mirror misses the last lines (it stops before they are logged).
    process._live_capture_task = None
    log.write_text("".join(lines))

    process._load_callback_log()
    process._reconcile_live_capture()

    assert published.read_text() == _expected("".join(lines))
    assert "sk-synthetic-secret" not in published.read_text()


async def test_a_stopped_proxy_keeps_only_status_usage_and_model(tmp_path):
    log = tmp_path / "callback.jsonl"
    text = "".join(_chat_log(log))
    process = _process(tmp_path, log)
    await _mirror(process, tmp_path / "trajectory" / "llm_trajectory.jsonl")

    process._load_callback_log()

    whole = trajectory_from_litellm_callback_log(
        text, session_id="run", agent_name="opencode"
    )
    kept = process.trajectory
    assert [e.request.body for e in kept.exchanges] == [{"model": "m"}] * 5
    assert [e.response.status_code for e in kept.exchanges] == [200] * 5
    assert kept.metadata == whole.metadata
    runtime = types.SimpleNamespace(kind="litellm", server=process)
    assert extract_usage(runtime) == extract_usage(
        types.SimpleNamespace(
            kind="litellm", server=types.SimpleNamespace(trajectory=whole)
        )
    )


async def test_the_tool_evidence_outlives_the_request_bodies(tmp_path):
    log = tmp_path / "callback.jsonl"
    text = "".join(_chat_log(log))
    process = _process(tmp_path, log)
    await _mirror(process, tmp_path / "trajectory" / "llm_trajectory.jsonl")
    process._load_callback_log()
    lossy = [
        {"type": "tool_call", "tool_call_id": f"call_{n}", "status": "completed"}
        for n in range(5)
    ]
    rollout = types.SimpleNamespace(_rollout_dir=tmp_path, _trajectory=lossy)

    Rollout._reconcile_acp_tool_evidence(
        rollout, types.SimpleNamespace(kind="litellm", server=process)
    )

    whole = trajectory_from_litellm_callback_log(
        text, session_id="run", agent_name="opencode"
    )
    evidence = _parse_provider_tool_evidence(whole.exchanges)
    assert [e.get("title") for e in rollout._trajectory] == [
        e.get("title") for e in evidence
    ]
    assert rollout._trajectory[4]["content"] == evidence[4]["content"]


async def test_the_rollouts_second_writer_leaves_the_published_file(tmp_path):
    log = tmp_path / "callback.jsonl"
    _chat_log(log)
    published = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    process = _process(tmp_path, log)
    await _mirror(process, published)
    process._load_callback_log()
    process._reconcile_live_capture()
    before = (published.stat().st_ino, published.read_bytes())

    rollout = types.SimpleNamespace(_rollout_dir=tmp_path)
    Rollout._write_llm_trajectory(
        rollout, types.SimpleNamespace(kind="litellm", server=process)
    )

    assert (published.stat().st_ino, published.read_bytes()) == before


async def test_a_published_file_removed_mid_run_is_written_whole_at_stop(tmp_path):
    log = tmp_path / "callback.jsonl"
    lines = _chat_log(log)
    published = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    process = _process(tmp_path, log)
    await _mirror(process, published)
    published.unlink()  # a cleanup, or another tool, took it

    process._load_callback_log()

    assert published.read_text() == _expected("".join(lines))


async def test_a_failed_replace_at_stop_is_published_by_the_reconcile(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    lines = _chat_log(log)
    log.write_text("".join(lines[:3]))
    published = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    process = _process(tmp_path, log)
    await _mirror(process, published)
    # The final import rewrites from a record that differs from its live line.
    process._live_writer._digests[1] = b"not the published line"
    log.write_text("".join(lines))
    real_replace = os.replace
    failures = []

    def fail_once(src, dst):
        if not failures:
            failures.append(dst)
            raise OSError(5, "dummy: input/output error")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_once)
    process._load_callback_log()
    assert failures and not process._live_reconciled

    process._reconcile_live_capture()

    assert process._live_reconciled
    assert published.read_text() == _expected("".join(lines))


def test_a_failed_append_ends_the_file_at_its_last_whole_line(tmp_path, monkeypatch):
    path = tmp_path / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)
    whole = trajectory_from_litellm_callback_log(
        "".join(_chat_log(tmp_path / "log.jsonl")), session_id="run", agent_name="a"
    )
    writer.append(whole.exchanges[:2])
    good = path.read_bytes()

    def half(fd, data):
        os.write(fd, data[: len(data) // 2])
        raise OSError(28, "dummy: no space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(_llm_capture, "_write_all", half)
        with pytest.raises(OSError):
            writer.append(whole.exchanges[2:3])
    assert path.read_bytes() == good
    writer.append(whole.exchanges[3:])
    writer.reconcile(whole)
    assert path.read_text() == _expected((tmp_path / "log.jsonl").read_text(), "a")


@pytest.mark.skipif(
    not Path("/proc/self/clear_refs").exists(), reason="reads the peak RSS in /proc"
)
def test_a_long_proxied_run_never_holds_its_log_in_memory(tmp_path):
    """The live mirror, the stop, the second writer, the usage and the tool
    evidence over a ~27 MB Codex log (each request repeats the conversation,
    in messages and in input) stay within a budget the whole log exceeds
    several times; 0.9 grew by about 240 MB here."""
    source = tmp_path / "source.jsonl"
    conversation: list[dict] = [
        {"type": "message", "role": "developer", "content": "r" * 4000}
    ]
    with source.open("w") as out:
        for n in range(30):
            conversation += [
                {
                    "type": "reasoning",
                    "id": f"rs_{n}",
                    "encrypted_content": "gA" * 7000,
                },
                {"type": "custom_tool_call", "call_id": f"c{n}", "input": "x" * 400},
                {
                    "type": "custom_tool_call_output",
                    "call_id": f"c{n}",
                    "output": f"step {n} \xb5 " + "y" * 14000,
                },
            ]
            body = {"model": "m", "messages": conversation, "input": conversation}
            usage = {"prompt_tokens": 1000 * n, "completion_tokens": 10}
            out.write(json.dumps(_record(n, body, usage=usage)) + "\n")
    assert 20e6 < source.stat().st_size < 35e6
    script = textwrap.dedent(
        """
        import asyncio, contextlib, gc, json, sys, types
        from pathlib import Path
        from benchflow.providers.litellm_runtime import HostLiteLLMProcess, extract_usage
        from benchflow.rollout import Rollout

        def kb(key):
            with open("/proc/self/status") as status:
                for line in status:
                    if line.startswith(key + ":"):
                        return int(line.split()[1])

        source, root = Path(sys.argv[1]), Path(sys.argv[2])
        published = root / "rollout" / "trajectory" / "llm_trajectory.jsonl"
        log = root / "callback.jsonl"
        log.write_bytes(b"")
        process = object.__new__(HostLiteLLMProcess)
        process.runtime_dir, process.trajectory, process.log_path = root, None, log
        process.session_id, process.agent_name = "s", "codex-acp"
        gc.collect()
        with open("/proc/self/clear_refs", "w") as refs:
            refs.write("5")
        base = kb("VmRSS")

        async def run():
            process.start_live_capture(published)
            process._live_capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await process._live_capture_task
            with source.open("rb") as src, log.open("ab") as dst:
                for line in src:
                    dst.write(line)
                    dst.flush()
                    while process._live_callback_offset < log.stat().st_size:
                        await process._capture_live_records()
            await process._stop_live_capture()
            process._load_callback_log()
            process._reconcile_live_capture()

        asyncio.run(run())
        runtime = types.SimpleNamespace(kind="litellm", server=process, backend_model="m")
        usage = extract_usage(runtime)
        rollout = types.SimpleNamespace(_rollout_dir=root / "rollout", _trajectory=[])
        Rollout._write_llm_trajectory(rollout, runtime)
        Rollout._reconcile_acp_tool_evidence(rollout, runtime)
        print(json.dumps({"growth_mb": (kb("VmHWM") - base) / 1024,
                          "total_tokens": usage["total_tokens"],
                          "lines": len(published.read_bytes().splitlines())}))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(source), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    measured = json.loads(done.stdout.strip().splitlines()[-1])
    assert measured["lines"] == 30
    assert measured["total_tokens"] == sum(1000 * n + 10 for n in range(30))
    assert measured["growth_mb"] < _MEMORY_BUDGET_MB, measured

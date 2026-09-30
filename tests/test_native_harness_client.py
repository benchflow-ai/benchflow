"""NativeCLIClient end to end on this machine, with a stand-in CLI.

The client's real launch script, kill script and parsers run; only the CLI is
replaced by ``fixtures/native_harness/replay_cli.py``, which replays a sample
the pinned CLI recorded, and the sandbox by a double that runs commands
locally (as the Docker transport runs them in a container: ``bash -c`` behind
the ACP line filter). Linux only: the kill script reads ``/proc``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import classify_error
from benchflow.acp.runtime import execute_prompts
from benchflow.acp.types import StopReason
from benchflow.diagnostics import AgentPromptTimeoutError, TransportClosedError
from benchflow.native_harness import client as client_module
from benchflow.native_harness.client import (
    NativeCLIClient,
    alive_script,
    encode_prompt,
    launch_script,
)
from benchflow.native_harness.harnesses import CLAUDE_CODE, CODEX
from benchflow.native_harness.session import NativeSession
from benchflow.native_harness.spec import NativeHarnessError
from benchflow.sandbox.process import SubprocessLiveProcess
from benchflow.sandbox.process._acp_lines import filter_command

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/environ").exists(), reason="needs Linux /proc"
)

FIXTURES = Path(__file__).parent / "fixtures" / "native_harness"
CLAUDE_SAMPLES = FIXTURES / "claude-code-2.1.280"
CODEX_SAMPLES = FIXTURES / "codex-0.156.1"


class _LocalProcess(SubprocessLiveProcess):
    async def start(self, command, env=None, cwd=None) -> None:
        self._set_process(
            await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                filter_command(command),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(env or {})},
                cwd=cwd,
            )
        )


class _LocalSandbox:
    """Live processes and root ``exec`` on this machine."""

    def __init__(self) -> None:
        self.exec_calls: list[str] = []
        self.fail_exec = False

    async def live_process(self, *, agent=None):
        return _LocalProcess()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.exec_calls.append(command)
        if self.fail_exec:
            raise ConnectionError("sandbox is gone")
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout_sec or 60)
        return SimpleNamespace(
            return_code=proc.returncode, stdout=out.decode(), stderr=err.decode()
        )


def _harness(base, tmp_path: Path):
    """The shipped harness, with the replay script standing in for the CLI."""
    cli = tmp_path / "cli"
    cli.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} "
        f"{shlex.quote(str(FIXTURES / 'replay_cli.py'))} \"$@\"\n"
    )
    cli.chmod(0o755)
    return dataclasses.replace(base, executable=str(cli))


def _client(tmp_path, sandbox, harness, sample: Path, **replay) -> NativeCLIClient:
    env = {
        "REPLAY_SAMPLE": str(sample),
        "REPLAY_PROMPT_OUT": str(tmp_path / "prompt.txt"),
        "REPLAY_ARGV_OUT": str(tmp_path / "argv.jsonl"),
        **{f"REPLAY_{k.upper()}": str(v) for k, v in replay.items()},
    }
    return NativeCLIClient(
        env=sandbox,
        harness=harness,
        agent=harness.agent,
        launch_env=env,
        sandbox_user=None,
        cwd=str(tmp_path),
        rollout_dir=tmp_path / "trial",
    )


def _argvs(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "argv.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


async def _marked_alive(sandbox: _LocalSandbox, run_id: str) -> bool:
    result = await sandbox.exec(alive_script(run_id))
    return result.stdout.strip().endswith("alive")


@pytest.mark.asyncio
async def test_a_turn_runs_the_cli_with_the_prompt_on_stdin(tmp_path):
    sandbox = _LocalSandbox()
    client = _client(tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "turn.jsonl")
    session_id = client.cli_session_id
    prompt = "Write it.\nLine two with 'quotes', \"double\", $HOME and ünïcode [[fake-llm:x]]"
    client.session.record_user_prompt(prompt)
    result = await client.prompt(prompt)
    client.session.mark_prompt_end()

    assert result.stop_reason == "end_turn"
    assert (tmp_path / "prompt.txt").read_text() == prompt
    (argv,) = _argvs(tmp_path)
    assert argv[argv.index("--session-id") + 1] == session_id
    assert "--resume" not in argv
    kinds = [e["type"] for e in client.session.events]
    assert kinds == ["user_message", "agent_message", "tool_call", "agent_message"]
    assert client.session.tool_calls[0].tool_call_id == "toolu_fake_hello_0"
    assert client.session.latest_usage_totals()["input_tokens"] == 2000
    # Evidence: the raw stream (with a turn header) and a per-turn record.
    stream = (tmp_path / "trial" / "agent" / "claude-code.jsonl").read_text().splitlines()
    assert json.loads(stream[0])["benchflow_native_turn"] == 1
    (turn,) = json.loads((tmp_path / "trial" / "agent" / "native-turns.json").read_text())
    assert turn["stop_reason"] == "end_turn" and turn["exit_code"] == 0
    assert turn["cli_cost_usd"] == pytest.approx(0.0025)
    # The prompt never reached a command line or an environment variable.
    assert prompt not in json.dumps(turn)


@pytest.mark.asyncio
async def test_the_next_turn_resumes_the_cli_session(tmp_path):
    sandbox = _LocalSandbox()
    client = _client(tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "turn.jsonl")
    first_id = client.cli_session_id
    await client.prompt("one")
    await client.prompt("two")
    first, second = _argvs(tmp_path)
    assert first[first.index("--session-id") + 1] == first_id
    # The CLI reported its session (the sample's placeholder id): resume it.
    assert second[second.index("--resume") + 1] == "00000000-0000-4000-8000-000000000000"
    assert client.session.session_id == "00000000-0000-4000-8000-000000000000"
    # Usage adds up across turns (the session keeps cumulative snapshots).
    assert client.session.latest_usage_totals()["input_tokens"] == 4000


@pytest.mark.asyncio
async def test_codex_usage_is_the_difference_between_thread_totals(tmp_path):
    sandbox = _LocalSandbox()
    harness = _harness(CODEX, tmp_path)
    client = _client(tmp_path, sandbox, harness, CODEX_SAMPLES / "turn.jsonl")
    await client.prompt("one")
    client._launch_env["REPLAY_SAMPLE"] = str(CODEX_SAMPLES / "resumed.jsonl")
    await client.prompt("two")
    first, second = _argvs(tmp_path)
    assert first[:2] == ["exec", "--json"]
    assert second[:3] == ["exec", "resume", "00000000-0000-4000-8000-000000000000"]
    # Totals 2000 then 4000: two turns of 2000 each, not 6000.
    assert client.session.latest_usage_totals()["input_tokens"] == 4000
    ids = [c.tool_call_id for c in client.session.tool_calls]
    assert ids == ["turn1-item_1", "turn2-item_1"]


@pytest.mark.asyncio
async def test_cancel_kills_the_cli_group_and_its_detached_children(tmp_path):
    sandbox = _LocalSandbox()
    # The turn up to its tool call, then a hang with a detached child.
    client = _client(
        tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "cancelled.jsonl",
        cut=5, hang=1, child=1,
    )
    task = asyncio.create_task(client.prompt("sleep"))
    for _ in range(200):
        if client.session.tool_calls:
            break
        await asyncio.sleep(0.05)
    run_id = client._run_id
    assert run_id and await _marked_alive(sandbox, run_id)
    await client.cancel()
    result = await asyncio.wait_for(task, 10)
    assert result.stop_reason == "cancelled"
    assert not await _marked_alive(sandbox, run_id)
    # The trajectory stops at the cancel: the call stays pending.
    assert client.session.pending_tool_call_ids() == ["toolu_fake_sleep_0"]


@pytest.mark.asyncio
async def test_a_cli_that_dies_mid_turn_is_an_agent_error(tmp_path):
    sandbox = _LocalSandbox()
    client = _client(
        tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "turn.jsonl",
        cut=4, exit=137,
    )
    with pytest.raises(NativeHarnessError) as caught:
        await client.prompt("go")
    assert caught.value.exit_code == 137
    assert "exit code 137" in str(caught.value)
    # Classified with ACP errors, as the ACP path files a dead Claude Code CLI.
    assert classify_error(str(caught.value)) == "acp_error"
    (turn,) = json.loads((tmp_path / "trial" / "agent" / "native-turns.json").read_text())
    assert "exit code 137" in turn["failure"]


@pytest.mark.asyncio
async def test_a_lost_sandbox_is_a_transport_failure_not_an_agent_error(tmp_path):
    sandbox = _LocalSandbox()
    client = _client(
        tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "turn.jsonl",
        cut=4, exit=1,
    )
    sandbox.fail_exec = True
    with pytest.raises(TransportClosedError):
        await client.prompt("go")


@pytest.mark.asyncio
async def test_a_turn_the_cli_reports_as_failed_raises(tmp_path):
    sandbox = _LocalSandbox()
    sample = tmp_path / "failed.jsonl"
    sample.write_text(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": "API Error: 401 invalid bearer token",
                "session_id": "s",
                "usage": {},
            }
        )
        + "\n"
    )
    client = _client(tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), sample)
    with pytest.raises(NativeHarnessError) as caught:
        await client.prompt("go")
    assert classify_error(str(caught.value)) == "provider_auth"


@pytest.mark.asyncio
async def test_execute_prompts_times_out_a_native_turn_like_an_acp_turn(tmp_path):
    """The kernel's own loop (wall clock, bounded cancel) drives the native client."""
    sandbox = _LocalSandbox()
    client = _client(
        tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "cancelled.jsonl",
        cut=5, hang=1,
    )
    with pytest.raises(AgentPromptTimeoutError) as caught:
        await execute_prompts(client, client.session, ["sleep"], timeout=3, idle_timeout=None)
    assert caught.value.diagnostic.pending_tool_call_ids == ["toolu_fake_sleep_0"]
    assert [e["type"] for e in caught.value.trajectory][-1] == "agent_timeout"
    run_ids = [c for c in sandbox.exec_calls if "BENCHFLOW_NATIVE_RUN=" in c]
    assert run_ids, "cancel ran the kill script"


@pytest.mark.asyncio
async def test_a_cli_lingering_after_its_result_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(client_module, "EXIT_GRACE_SEC", 1)
    sandbox = _LocalSandbox()
    client = _client(
        tmp_path, sandbox, _harness(CLAUDE_CODE, tmp_path), CLAUDE_SAMPLES / "turn.jsonl",
        hang=1,
    )
    result = await asyncio.wait_for(client.prompt("go"), 20)
    assert result.stop_reason == "end_turn"


def test_ask_user_handlers_are_refused_not_silently_dropped(tmp_path):
    client = _client(tmp_path, _LocalSandbox(), CLAUDE_CODE, CLAUDE_SAMPLES / "turn.jsonl")
    session = NativeSession(client)
    session.on_ask_user(None)  # clearing is fine

    async def handler(request):
        return "allow"

    with pytest.raises(NativeHarnessError, match="no permission channel"):
        session.on_ask_user(handler)
    assert session.capabilities.ask_user is False
    assert session.capabilities.nudges is True


def test_launch_script_reads_the_prompt_line_and_starts_its_own_group():
    script = launch_script("/opt/cli", ("-p", "it's"), "abc123")
    assert "IFS= read -r p" in script and "base64 -d" in script
    assert "set -m; /opt/cli -p 'it'\"'\"'s' <&3 3<&- & set +m; wait $!" in script
    assert "benchflow_native_exit" in script
    assert encode_prompt("héllo\n") == "aMOpbGxvCg=="

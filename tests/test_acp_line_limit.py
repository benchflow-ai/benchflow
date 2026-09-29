"""A multi-megabyte ACP line is rewritten to fit, never dropped (#1138).

Guards #1138. A Claude Code ``Read`` of a PDF returns its pages as inline
base64 images: one ``tool_call_update`` line of several megabytes. Daytona
closed the PTY websocket on it (1008), deterministically, so the rollout
died on every attempt; Docker's reader dropped any line over its 10 MB
buffer, and with it the tool call's final update, leaving the call pending.
Now a filter in the sandbox rewrites lines over 1 MiB (images and base64 to
notes, then the longest strings cut) before the transport sees them, and the
host applies the same rewrite to any line that still arrives whole.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import subprocess

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.session import ACPSession
from benchflow.sandbox.process import LiveProcess
from benchflow.sandbox.process._acp_lines import LIMIT, filter_command, shrink_line
from benchflow.sandbox.process._base import read_whole_line

PNG = base64.b64encode(os.urandom(3 * 1024 * 1024)).decode()  # ~4 MB of base64


def _pdf_read_update(status: str = "completed") -> dict:
    """Claude Code's update for a Read of PDF pages: ACP image blocks, and
    the Anthropic-shaped rawOutput carrying the same base64 again."""
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "toolu_01BpHP5HFrYzKoUXoy659Ecd",
                "status": status,
                "content": [
                    {
                        "type": "content",
                        "content": {
                            "type": "image",
                            "data": PNG,
                            "mimeType": "image/png",
                        },
                    }
                ],
                "rawOutput": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG,
                        },
                    }
                ],
            },
        },
    }


def _line(message: dict) -> bytes:
    return json.dumps(message).encode() + b"\n"


def test_a_line_under_the_limit_passes_byte_for_byte():
    line = b'{"jsonrpc": "2.0", "id": 7, "result": {"stopReason": "end_turn"}}\n'
    assert shrink_line(line) is line


def test_images_become_notes_and_the_update_keeps_its_tool_call():
    line = _line(_pdf_read_update())
    assert len(line) > 8 * 1024 * 1024
    shrunk = shrink_line(line)
    assert len(shrunk) <= LIMIT + 1 and shrunk.endswith(b"\n")
    message = json.loads(shrunk)
    update = message["params"]["update"]
    assert message["method"] == "session/update"
    assert message["params"]["sessionId"] == "s1"
    assert update["toolCallId"] == "toolu_01BpHP5HFrYzKoUXoy659Ecd"
    assert update["status"] == "completed"
    note = update["content"][0]["content"]
    assert note["type"] == "text"
    assert "image data (image/png)" in note["text"] and str(len(PNG)) in note["text"]
    assert "base64 data" in update["rawOutput"][0]["source"]["data"]
    assert PNG[:1000] not in shrunk.decode()


def test_long_text_is_cut_to_fit_with_a_note():
    text = "line of tool output\n" * 200_000  # ~4 MB, not base64
    message = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call-2",
                "status": "failed",
                "rawOutput": {"stdout": text, "stderr": "é" * 300_000},
            },
        },
    }
    shrunk = shrink_line(_line(message))
    assert len(shrunk) <= LIMIT + 1
    update = json.loads(shrunk)["params"]["update"]
    assert (update["toolCallId"], update["status"]) == ("call-2", "failed")
    assert update["rawOutput"]["stdout"].startswith("line of tool output\n")
    assert "the rest of this text" in update["rawOutput"]["stdout"]


def test_a_non_json_line_keeps_its_start():
    line = b"x" * (3 * LIMIT) + b"\n"
    shrunk = shrink_line(line)
    assert shrunk.startswith(b"x" * 1000) and len(shrunk) < 70 * 1024
    assert b"non-JSON line" in shrunk


# ---------------------------------------------------------------------------
# The sandbox-side filter, run for real through bash and python3
# ---------------------------------------------------------------------------

needs_bash_python = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason="needs bash and python3",
)


def _run(command: str, stdin: bytes = b"", env: dict | None = None):
    return subprocess.run(
        ["bash", "-c", command],
        input=stdin,
        capture_output=True,
        timeout=60,
        env=env,
    )


@needs_bash_python
def test_the_filter_rewrites_only_oversized_lines_and_keeps_order(tmp_path):
    big = tmp_path / "big.jsonl"
    big.write_bytes(_line(_pdf_read_update()))
    agent = (
        f'printf \'%s\\n\' \'{{"jsonrpc":"2.0","id":1,"result":{{}}}}\'; '
        f"cat {shlex.quote(str(big))}; printf 'done\\n'"
    )
    done = _run(filter_command(agent))
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines(keepends=True)
    assert lines[0] == b'{"jsonrpc":"2.0","id":1,"result":{}}\n'
    assert len(lines[1]) <= LIMIT + 1
    assert json.loads(lines[1])["params"]["update"]["status"] == "completed"
    assert lines[2] == b"done\n"


@needs_bash_python
def test_the_filter_passes_stdin_to_the_agent_and_its_exit_status_back():
    done = _run(filter_command("cat; exit 3"), stdin=b'{"jsonrpc":"2.0"}\n')
    assert done.stdout == b'{"jsonrpc":"2.0"}\n'
    assert done.returncode == 3


@needs_bash_python
def test_without_a_usable_python3_the_agent_runs_unfiltered(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_text("#!/bin/sh\nexit 1\n")
    (fake / "python3").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    big = tmp_path / "big.jsonl"
    big.write_bytes(_line(_pdf_read_update()))
    done = _run(filter_command(f"cat {shlex.quote(str(big))}"), env=env)
    assert done.returncode == 0
    assert done.stdout == big.read_bytes()


def test_the_filter_is_off_when_the_limit_is(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "0")
    assert filter_command("claude-agent-acp") == "claude-agent-acp"
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "2048")
    assert "python3 -u -c" in filter_command("claude-agent-acp")


async def test_docker_starts_the_agent_through_the_filter(monkeypatch):
    from unittest.mock import MagicMock

    from benchflow.sandbox.process.docker import DockerProcess

    argvs: list[list[str]] = []

    async def fake_exec(*argv, **kwargs):
        argvs.append(list(argv))
        return MagicMock(pid=1, returncode=None, stderr=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    process = DockerProcess("p", "/d", [], client_env={})
    await process.start("claude-agent-acp --acp")
    command = argvs[-1][-1]
    assert argvs[-1][-3:-1] == ["bash", "-c"]
    assert command.startswith("if python3 -c 'import json, re, shlex, sys'")
    assert "{ claude-agent-acp --acp\n} | python3 -u -c" in command
    assert command.endswith("else claude-agent-acp --acp\nfi")


async def test_daytona_pty_starts_the_agent_through_the_filter():
    from tests.test_daytona_pty_liveness import FakePtyHandle, _sandbox

    handles: list[FakePtyHandle] = []
    sandbox = _sandbox(handles)
    from benchflow.sandbox.process.daytona import DaytonaPtyProcess

    process = DaytonaPtyProcess(sandbox, "", "docker compose -p t")
    await process.start(command="claude-agent-acp --acp")
    script = sandbox.process.exec.await_args_list[-1].args[0]
    assert "python3 -u -c" in script and "claude-agent-acp --acp" in script
    await process.close()


# ---------------------------------------------------------------------------
# The host side: read a long line whole, then shrink it
# ---------------------------------------------------------------------------


async def test_a_line_over_the_reader_limit_is_read_whole_not_dropped():
    reader = asyncio.StreamReader(limit=64)
    long_line = b"y" * 1000 + b"\n"
    reader.feed_data(long_line + b"next\n")
    reader.feed_eof()
    assert await read_whole_line(reader) == long_line
    assert await read_whole_line(reader) == b"next\n"
    assert await read_whole_line(reader) == b""


async def test_a_line_over_the_cap_is_skipped_and_the_next_one_read():
    reader = asyncio.StreamReader(limit=64)
    reader.feed_data(b"z" * 1000 + b"\nnext\n")
    reader.feed_eof()
    assert await read_whole_line(reader, cap=500) is None
    assert await read_whole_line(reader, cap=500) == b"next\n"


class _Lines(LiveProcess):
    """An agent's output, one line per read (what the Docker reader returns)."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)

    async def start(self, command, env=None, cwd=None) -> None:
        return None

    async def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        await asyncio.Future()
        return b""

    async def writeline(self, data: str) -> None:
        return None

    async def close(self) -> None:
        return None

    @property
    def is_running(self) -> bool:
        return True


async def test_an_oversized_final_update_still_completes_its_tool_call():
    session = ACPSession("s1")
    start = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "toolu_01BpHP5HFrYzKoUXoy659Ecd",
                "title": "Read paper.pdf",
                "kind": "read",
                "status": "pending",
            },
        },
    }
    done = {"jsonrpc": "2.0", "id": 100001, "result": {"stopReason": "end_turn"}}
    process = _Lines([_line(start), _line(_pdf_read_update()), _line(done)])
    client = ACPClient(ContainerTransport(process, command="never-started"))
    client._session = session
    result = await asyncio.wait_for(client.prompt("read it"), timeout=30)
    assert result.stop_reason == "end_turn"
    (call,) = session.tool_calls
    assert call.status == "completed"
    assert session.pending_tool_call_state() == ()

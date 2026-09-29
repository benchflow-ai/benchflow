"""Gated live guard for the pinned claude-agent-acp config-option contract.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + ``node`` +
network for the install):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

It installs the exact adapter and Claude Code CLI pins selected by
``benchflow.agents.registry``, starts the adapter over ACP stdio with the CLI
handed over through ``CLAUDE_CODE_EXECUTABLE`` as the sandbox launcher does,
and proves the complete Fable model + effort path works: ``initialize``,
``session/new``, and both ``session/set_config_option`` calls, then that
``claude-opus-5-5`` (#1137) resolves in the model option. On Linux it also
checks that the Claude Code process the adapter started is the pinned CLI, not
the one its SDK bundles. Claude Code 2.1.280 validates a full model id with a
one-token ``POST /v1/messages`` before switching to it, so the CLI talks to a
local stub endpoint with a fake key: no credentials, and no model traffic
leaves the host. Re-run when bumping either pin.
"""

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchflow.agents.registry import _CLAUDE_AGENT_ACP_PACKAGE, _CLAUDE_CODE_PACKAGE

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="gated live ACP guard; set RUN_ACP_DEP_GUARD=1 (needs npm + node + network)",
)

EXPECTED_OPTION_IDS = {"model", "effort"}
FABLE_MODEL = "claude-fable-5-1"
FABLE_EFFORT = "xhigh"
OPUS_MODEL = "claude-opus-5-5"


def _tool_or_skip(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not available")
    return path


@contextlib.contextmanager
def _stub_anthropic() -> Iterator[tuple[str, list[tuple[str, str, str | None]]]]:
    """A local Anthropic API stand-in; yields its URL and the requests it saw."""
    seen: list[tuple[str, str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_HEAD(self) -> None:
            seen.append(("HEAD", self.path, None))
            self._answer({})

        def do_GET(self) -> None:
            seen.append(("GET", self.path, None))
            self._answer({})

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            model = None
            with contextlib.suppress(ValueError, AttributeError):
                model = json.loads(raw).get("model")
            seen.append(("POST", self.path, model))
            self._answer(
                {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def _executables_under(prefix: Path) -> set[str]:
    """Resolved executables of this user's processes installed under ``prefix``."""
    root = os.path.realpath(prefix)
    found = set()
    for proc in Path("/proc").glob("[0-9]*"):
        with contextlib.suppress(OSError):
            exe = os.readlink(proc / "exe")
            if exe.startswith(root + os.sep):
                found.add(exe)
    return found


def _current(client) -> dict[str, str]:
    return {
        o["id"]: o["currentValue"]
        for o in client.session.config_options or []
        if isinstance(o, dict)
        and o.get("id") in EXPECTED_OPTION_IDS
        and isinstance(o.get("currentValue"), str)
    }


async def _exercise_config_options(
    entry: Path, cli: Path, endpoint: str
) -> tuple[set[str], dict[str, str], str, set[str]]:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.transport import StdioTransport

    client = ACPClient(
        StdioTransport(
            "node",
            [str(entry)],
            env={
                "CLAUDE_CODE_EXECUTABLE": str(cli),
                "ANTHROPIC_BASE_URL": endpoint,
                "ANTHROPIC_API_KEY": "fake-key",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
            },
            cwd="/tmp",
        )
    )
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/tmp"), timeout=90)
        # session/new starts the CLI (the model list comes from it).
        running = _executables_under(cli.parent.parent)
        opts = client.session.config_options or []
        ids = {
            o["id"]
            for o in opts
            if isinstance(o, dict) and isinstance(o.get("id"), str)
        }
        current: dict[str, str] = {}
        opus = ""
        if ids >= EXPECTED_OPTION_IDS:
            await asyncio.wait_for(
                client.set_config_option("model", FABLE_MODEL), timeout=60
            )
            await asyncio.wait_for(
                client.set_config_option("effort", FABLE_EFFORT), timeout=60
            )
            current = _current(client)
            await asyncio.wait_for(
                client.set_config_option("model", OPUS_MODEL), timeout=60
            )
            opus = _current(client).get("model", "")
        return ids, current, opus, running
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def test_pinned_claude_acp_supports_fable_model_and_effort(tmp_path):
    """Guards PR #1086's Fable-compatible adapter and ACP config contract."""
    npm = _tool_or_skip("npm")
    _tool_or_skip("node")
    prefix = tmp_path / "claude"
    prefix.mkdir()
    for spec in (_CLAUDE_AGENT_ACP_PACKAGE, _CLAUDE_CODE_PACKAGE):
        subprocess.run(
            [npm, "install", "--prefix", str(prefix), spec],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    cli = prefix / "node_modules" / ".bin" / "claude"
    version = _CLAUDE_CODE_PACKAGE.rpartition("@")[2]
    reported = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, timeout=60
    ).stdout.strip()
    assert reported == f"{version} (Claude Code)", reported
    entry = (
        prefix
        / "node_modules"
        / "@agentclientprotocol"
        / "claude-agent-acp"
        / "dist"
        / "index.js"
    )
    assert entry.is_file(), f"pinned agent entry not found: {entry}"

    with _stub_anthropic() as (endpoint, seen):
        ids, current, opus, running = asyncio.run(
            _exercise_config_options(entry, cli, endpoint)
        )
    if Path("/proc/self/exe").exists():
        # The pinned CLI, not the binary the adapter's SDK bundles.
        assert running == {os.path.realpath(cli)}, running
    missing = EXPECTED_OPTION_IDS - ids
    assert not missing, (
        f"pinned {_CLAUDE_AGENT_ACP_PACKAGE} no longer advertises config option(s) "
        f"{sorted(missing)!r} (advertised: {sorted(ids)!r}); the registry "
        f"model/effort wiring is stale — re-verify acp_model_config_id / "
        f"acp_effort_config_id"
    )
    assert current.get("model", "").split("[", 1)[0] == FABLE_MODEL, current
    assert current.get("effort") == FABLE_EFFORT, current
    # The CLI validated the Fable id through the (stub) Messages API.
    assert ("POST", "/v1/messages?beta=true", FABLE_MODEL) in seen, seen
    assert opus.startswith("opus"), opus

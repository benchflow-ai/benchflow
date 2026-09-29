"""Gated live guard: codex-acp's title thread stays on the run's gateway.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + ``node``
+ network for the install):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_codex_title_thread_guard.py -q

It installs the registry's codex-acp pin, starts it through BenchFlow's own
launcher with the env the LiteLLM route builds, and sends one prompt carrying a
canary. The gateway is a local mock that serves only the run's model and
refuses others with 400, as BenchFlow's gateway does; ``HTTPS_PROXY`` points at
a recorder that refuses and logs every attempt to reach anything else. With
the home config (the fix), the title thread's ``gpt-5.6-luna`` request, task
prompt included, reaches only the gateway; without it (the control run),
codex-acp 1.13.1 tries api.openai.com, which proves the recorder sees the
leak. Re-run when bumping the codex-acp pin.
"""

import asyncio
import contextlib
import json
import os
import shutil
import socketserver
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchflow.agents.codex_config import (
    CODEX_HOME_CONFIG_ENV,
    apply_codex_launch_config,
    apply_codex_provider_config,
)
from benchflow.agents.registry import _CODEX_ACP_PACKAGE, CODEX_ACP_BUILTIN_LAUNCH

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="gated live ACP guard; set RUN_ACP_DEP_GUARD=1 (needs npm + node + network)",
)

MODEL = "gpt-6-astra"
TITLE_MODEL = "gpt-5.6-luna"
CANARY = "canary-7f3a"


def _completed_stream(model: str) -> bytes:
    message = {
        "id": "msg_mock",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
    }

    def response(status: str, output: list) -> dict:
        return {
            "id": "resp_mock",
            "object": "response",
            "created_at": 0,
            "status": status,
            "model": model,
            "output": output,
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }

    events = [
        ("response.created", {"response": response("in_progress", [])}),
        (
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
        ),
        (
            "response.output_text.delta",
            {
                "item_id": "msg_mock",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok",
            },
        ),
        ("response.output_item.done", {"output_index": 0, "item": message}),
        ("response.completed", {"response": response("completed", [message])}),
    ]
    return b"".join(
        f"event: {name}\ndata: {json.dumps({'type': name, 'sequence_number': i, **data})}\n\n".encode()
        for i, (name, data) in enumerate(events)
    )


@contextlib.contextmanager
def _gateway_and_egress() -> Iterator[tuple[int, int, list[dict]]]:
    seen: list[dict] = []

    class Gateway(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            seen.append({"at": "gateway", "method": "GET", "url": self.path})
            data = {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
            self._reply(200, json.dumps(data).encode(), "application/json")

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            model = None
            with contextlib.suppress(ValueError, AttributeError):
                model = json.loads(raw).get("model")
            seen.append(
                {
                    "at": "gateway",
                    "method": "POST",
                    "url": self.path,
                    "model": model,
                    "canary": CANARY.encode() in raw,
                }
            )
            if model != MODEL:
                error = {"error": {"message": f"model_not_served: {model}"}}
                self._reply(400, json.dumps(error).encode(), "application/json")
            else:
                self._reply(200, _completed_stream(MODEL), "text/event-stream")

        def log_message(self, *args: object) -> None:
            pass

    class Egress(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline().decode("latin-1").split()
            seen.append({"at": "egress", "request": " ".join(line[:2])})
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")

    gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    egress = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Egress)
    egress.daemon_threads = True
    for server in (gateway, egress):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield gateway.server_address[1], egress.server_address[1], seen
    finally:
        for server in (gateway, egress):
            server.shutdown()
            server.server_close()


def _launch_env(home: Path, gateway: int, egress: int, *, fixed: bool) -> dict:
    base = f"http://127.0.0.1:{gateway}/v1"
    env = {"OPENAI_BASE_URL": base, "OPENAI_API_KEY": "sk-master"}
    apply_codex_provider_config(
        env, base_url=base, model=MODEL, provider_name="litellm", strict=True
    )
    env, _ = apply_codex_launch_config(
        "codex-acp", env, model=MODEL, reasoning_effort=None, sandboxed=True
    )
    if not fixed:
        env.pop(CODEX_HOME_CONFIG_ENV, None)
    proxy = f"http://127.0.0.1:{egress}"
    # Explicit homes: the transport inherits this process's environment, and
    # the launcher writes config.toml under CODEX_HOME.
    return {
        **env,
        "HOME": str(home),
        "BENCHFLOW_AGENT_HOME": str(home),
        "CODEX_HOME": str(home / ".codex"),
        **{
            key: proxy
            for key in (
                "HTTPS_PROXY",
                "https_proxy",
                "HTTP_PROXY",
                "http_proxy",
                "ALL_PROXY",
            )
        },
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }


async def _one_turn(launcher: str, env: dict, cwd: Path) -> None:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.transport import StdioTransport

    client = ACPClient(StdioTransport("sh", ["-c", launcher], env=env, cwd=str(cwd)))
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd=str(cwd)), timeout=180)
        await asyncio.wait_for(
            client.prompt(f"{CANARY}: reply with the word ok."), timeout=240
        )
        # The title thread starts after the turn completes (0.1 s on the VM).
        await asyncio.sleep(20)
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def _run(tmp_path: Path, adapter: Path, *, fixed: bool) -> list[dict]:
    home = tmp_path / ("fixed" if fixed else "control")
    work = home / "work"
    work.mkdir(parents=True)
    launcher = CODEX_ACP_BUILTIN_LAUNCH.replace(
        "/opt/benchflow/bin/codex-acp", str(adapter)
    )
    with _gateway_and_egress() as (gateway, egress, seen):
        asyncio.run(
            _one_turn(launcher, _launch_env(home, gateway, egress, fixed=fixed), work)
        )
        return list(seen)


def test_title_thread_reaches_only_the_runs_gateway(tmp_path):
    npm = shutil.which("npm") or pytest.skip("npm not available")
    shutil.which("node") or pytest.skip("node not available")
    prefix = tmp_path / "codex"
    prefix.mkdir()
    subprocess.run(
        [npm, "install", "--prefix", str(prefix), _CODEX_ACP_PACKAGE],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    adapter = prefix / "node_modules" / ".bin" / "codex-acp"

    control = _run(tmp_path, adapter, fixed=False)
    fixed = _run(tmp_path, adapter, fixed=True)

    def openai_attempts(seen: list[dict]) -> list[dict]:
        return [
            s for s in seen if s["at"] == "egress" and "api.openai.com" in s["request"]
        ]

    # The control run proves the recorder sees the leak this guards against.
    assert openai_attempts(control), control
    assert not openai_attempts(fixed), fixed
    posts = [s for s in fixed if s["at"] == "gateway" and s["method"] == "POST"]
    assert {MODEL, TITLE_MODEL} <= {s["model"] for s in posts}, posts
    # The title thread's task prompt reached the run's gateway, which refuses
    # a model the run does not serve (tests/test_litellm_model_gate.py).
    assert any(s["canary"] for s in posts if s["model"] == TITLE_MODEL), posts

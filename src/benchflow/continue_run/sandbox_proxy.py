"""Sandbox-local replay proxy for ``benchflow continue``.

Remote sandboxes such as Daytona cannot reach a host-local replay proxy at
``127.0.0.1``. For those environments the continuation stack runs entirely
inside the sandbox:

    OpenHands -> sandbox replay proxy -> sandbox LiteLLM proxy -> provider

The host uploads the recorded exchanges and a small stdlib-only proxy script,
then downloads the live suffix after the rollout finishes so the normal stitched
``llm_trajectory.jsonl`` artifact remains identical in shape to host replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.trajectories.types import LLMExchange

SANDBOX_REPLAY_ROOT = "/tmp/benchflow-replay"
DEFAULT_SANDBOX_REPLAY_PORT = 61357


def sandbox_replay_base_url(port: int = DEFAULT_SANDBOX_REPLAY_PORT) -> str:
    """Return the in-sandbox OpenAI-compatible replay endpoint."""
    return f"http://127.0.0.1:{port}/v1"


def _sandbox_proxy_source() -> str:
    return r"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _n_messages(body):
    messages = body.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _completion_to_sse(body):
    base_id = body.get("id") or f"chatcmpl-replay-{int(time.time() * 1000)}"
    created = body.get("created") or int(time.time())
    model = body.get("model") or "replay"
    choices = body.get("choices") or [{}]
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "stop"

    def chunk(delta, finish=None):
        return json.dumps(
            {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    payloads = [chunk({"role": "assistant"})]
    content = message.get("content")
    if content:
        delta = {"content": content}
        for key in ("reasoning_content", "thinking"):
            if message.get(key):
                delta[key] = message[key]
        payloads.append(chunk(delta))

    for i, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        payloads.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": tool_call.get("index", i),
                            "id": tool_call.get("id"),
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    ]
                }
            )
        )

    final = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if isinstance(body.get("usage"), dict):
        final["usage"] = body["usage"]
    payloads.append(json.dumps(final))
    return payloads


@dataclass
class ReplayState:
    recorded: list
    upstream_url: str
    upstream_api_key: str
    upstream_model: str
    live_log_path: str
    strict_divergence: bool = False
    cursor: int = 0
    divergences: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _check_divergence(self, incoming, recorded_request):
        want = _n_messages(recorded_request)
        got = _n_messages(incoming)
        if want and got and want != got:
            self.divergences += 1
            message = (
                f"replay divergence at turn {self.cursor}: agent sent {got} "
                f"messages, recorded turn had {want}"
            )
            if self.strict_divergence:
                raise RuntimeError(message)
            print(message, file=sys.stderr, flush=True)

    def next_response(self, request_body):
        with self.lock:
            if self.cursor < len(self.recorded):
                exchange = self.recorded[self.cursor]
                self._check_divergence(
                    request_body,
                    ((exchange.get("request") or {}).get("body") or {}),
                )
                self.cursor += 1
                response = exchange.get("response") or {}
                return "replay", int(response.get("status_code") or 200), dict(response.get("body") or {})
            self.cursor += 1

        status, body = self._forward_live(request_body)
        self._append_live_exchange(request_body, status, body)
        return "live", status, body

    def _forward_live(self, request_body):
        forwarded = dict(request_body)
        forwarded["model"] = self.upstream_model
        forwarded["stream"] = False
        data = json.dumps(forwarded).encode("utf-8")
        request = urllib.request.Request(
            self.upstream_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.upstream_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode("utf-8")
                return int(response.status), json.loads(raw or "{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw or "{}")
            except json.JSONDecodeError:
                body = {"error": {"message": raw or str(exc)}}
            return int(exc.code), body
        except Exception as exc:
            traceback.print_exc()
            return 500, {"error": {"message": str(exc)}}

    def _append_live_exchange(self, request_body, status, body):
        row = {
            "request": {"body": request_body},
            "response": {"status_code": status, "body": body},
        }
        with open(self.live_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()


class ReplayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("replay-proxy " + fmt % args, file=sys.stderr, flush=True)

    @property
    def state(self):
        return self.server.state

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, payloads):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for payload in payloads:
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/health/liveliness", "/v1/health"):
            self._send_json(200, {"status": "ok"})
            return
        if path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [{"id": "replay", "object": "model"}]})
            return
        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": f"not found: {path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"bad request: {exc}"}})
            return

        want_stream = bool(body.get("stream"))
        try:
            _, status, response = self.state.next_response(body)
        except RuntimeError as exc:
            self._send_json(409, {"error": {"message": str(exc), "type": "divergence"}})
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        if status >= 400 or not response.get("choices"):
            self._send_json(status, response)
            return
        if want_stream:
            self._send_sse(_completion_to_sse(response))
        else:
            self._send_json(status, response)


class ReplayServer(ThreadingHTTPServer):
    def __init__(self, address, handler, state):
        super().__init__(address, handler)
        self.state = state


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    state = ReplayState(
        recorded=cfg["recorded"],
        upstream_url=cfg["upstream_url"],
        upstream_api_key=cfg["upstream_api_key"],
        upstream_model=cfg["upstream_model"],
        live_log_path=cfg["live_log_path"],
        strict_divergence=bool(cfg.get("strict_divergence")),
    )
    server = ReplayServer(("127.0.0.1", int(cfg["port"])), ReplayHandler, state)
    with open(cfg["state_path"], "w", encoding="utf-8") as handle:
        json.dump({"port": int(cfg["port"])}, handle)
    server.serve_forever()


if __name__ == "__main__":
    main()
"""


async def _upload_text(sandbox: Any, text: str, target_path: str, suffix: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        await sandbox.upload_file(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _read_remote_text(sandbox: Any, path: str, *, timeout_sec: int = 15) -> str:
    download_file = getattr(sandbox, "download_file", None)
    if download_file is not None:
        with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            await download_file(path, tmp_path)
            return tmp_path.read_text()
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)
    result = await sandbox.exec(
        f"cat {shlex.quote(path)} 2>/dev/null || true",
        timeout_sec=timeout_sec,
    )
    return result.stdout or ""


@dataclass
class SandboxReplayProxy:
    """A replay proxy process running on sandbox loopback."""

    sandbox: Any
    runtime_dir: str
    port: int
    pid_path: str
    live_log_path: str
    state_path: str
    stdout_path: str
    stderr_path: str
    live_exchanges: list[LLMExchange] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return sandbox_replay_base_url(self.port)

    @classmethod
    async def start(
        cls,
        *,
        sandbox: Any,
        recorded: list[LLMExchange],
        upstream_url: str,
        upstream_api_key: str,
        upstream_model: str,
        strict_divergence: bool = False,
        port: int = DEFAULT_SANDBOX_REPLAY_PORT,
    ) -> SandboxReplayProxy:
        token = uuid4().hex[:16]
        runtime_dir = f"{SANDBOX_REPLAY_ROOT}/{token}"
        paths = {
            "script": f"{runtime_dir}/replay_proxy.py",
            "config": f"{runtime_dir}/config.json",
            "state": f"{runtime_dir}/state.json",
            "pid": f"{runtime_dir}/replay.pid",
            "stdout": f"{runtime_dir}/stdout.log",
            "stderr": f"{runtime_dir}/stderr.log",
            "live_log": f"{runtime_dir}/live_exchanges.jsonl",
        }
        result = await sandbox.exec(
            f"mkdir -p {shlex.quote(runtime_dir)} && "
            f"chmod 700 {shlex.quote(runtime_dir)}",
            timeout_sec=20,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "prepare sandbox replay runtime failed: "
                f"{result.stderr or result.stdout}"
            )

        recorded_rows = [
            exchange.model_dump(mode="json", exclude_none=True) for exchange in recorded
        ]
        config = {
            "recorded": recorded_rows,
            "upstream_url": upstream_url,
            "upstream_api_key": upstream_api_key,
            "upstream_model": upstream_model,
            "strict_divergence": strict_divergence,
            "port": port,
            "state_path": paths["state"],
            "live_log_path": paths["live_log"],
        }
        await _upload_text(sandbox, _sandbox_proxy_source(), paths["script"], ".py")
        await _upload_text(sandbox, json.dumps(config), paths["config"], ".json")

        command = (
            f"rm -f {shlex.quote(paths['state'])} {shlex.quote(paths['pid'])} "
            f"{shlex.quote(paths['live_log'])}; "
            f"(python3 {shlex.quote(paths['script'])} {shlex.quote(paths['config'])} "
            f"> {shlex.quote(paths['stdout'])} 2> {shlex.quote(paths['stderr'])} & "
            f"echo $! > {shlex.quote(paths['pid'])})"
        )
        result = await sandbox.exec(command, timeout_sec=20)
        if result.return_code != 0:
            raise RuntimeError(
                f"start sandbox replay proxy failed: {result.stderr or result.stdout}"
            )
        proxy = cls(
            sandbox=sandbox,
            runtime_dir=runtime_dir,
            port=port,
            pid_path=paths["pid"],
            live_log_path=paths["live_log"],
            state_path=paths["state"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
        )
        try:
            await proxy._wait_until_ready()
        except BaseException:
            await proxy.stop()
            raise
        return proxy

    async def _wait_until_ready(self) -> None:
        probe = (
            "python3 - <<'PY'\n"
            "import urllib.request\n"
            f"urllib.request.urlopen('http://127.0.0.1:{self.port}/health', timeout=2).read()\n"
            "PY"
        )
        last = ""
        for _ in range(120):
            result = await self.sandbox.exec(probe, timeout_sec=5)
            if result.return_code == 0:
                return
            last = (result.stderr or result.stdout or "").strip()
            await asyncio.sleep(0.25)
        stderr = await _read_remote_text(self.sandbox, self.stderr_path, timeout_sec=5)
        raise RuntimeError(
            f"sandbox replay proxy did not become healthy: {last or stderr.strip()}"
        )

    async def stop(self) -> None:
        self.live_exchanges = await self._load_live_exchanges()
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"if [ -s {shlex.quote(self.pid_path)} ]; then "
                f"kill -TERM $(cat {shlex.quote(self.pid_path)}) 2>/dev/null || true; "
                "fi",
                timeout_sec=10,
            )
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self.runtime_dir)}",
                timeout_sec=10,
            )

    async def _load_live_exchanges(self) -> list[LLMExchange]:
        text = await _read_remote_text(self.sandbox, self.live_log_path)
        exchanges: list[LLMExchange] = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            try:
                exchanges.append(LLMExchange.model_validate_json(raw))
            except Exception:
                continue
        return exchanges

"""Tests for sandbox-local provider usage telemetry."""

from __future__ import annotations

import base64
import contextlib
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from benchflow.providers import usage_proxy_runtime as usage_runtime_mod
from benchflow.trajectories.types import Trajectory


def test_agent_kill_pattern_excludes_usage_proxy_agent_name_argument():
    """Guards PR #587: agent cleanup must not kill the usage proxy."""
    from benchflow.rollout import _agent_process_kill_pattern

    pattern = _agent_process_kill_pattern("/opt/benchflow/bin/codex-acp")

    assert pattern is not None
    assert re.search(pattern, "/opt/benchflow/bin/codex-acp")
    assert re.search(pattern, "node /opt/benchflow/js-agents/bin/codex-acp --flag")
    assert not re.search(pattern, "node /tmp/benchflow-usage-proxy/proxy.js")
    assert not re.search(pattern, "proxy.js --agent-name=codex-acp")


@pytest.mark.asyncio
async def test_daytona_uses_sandbox_local_proxy_not_host_proxy(monkeypatch):
    """Guards PR #587: Daytona agents must not use host-local proxy URLs."""
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    class FakeSandboxUsageProxy:
        target = "https://api.anthropic.com"
        base_url = "http://127.0.0.1:49000"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.started = False
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    monkeypatch.setattr(
        usage_runtime_mod,
        "TrajectoryProxy",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("host proxy must not start")
        ),
    )
    monkeypatch.setattr(
        usage_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    env = {
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_API_KEY": "sk-real-key",
    }
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="claude-haiku-4-5-20251001",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert runtime is not None
    assert runtime.server.started is True
    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49000"


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_imports_raw_captures():
    """Guards PR #587: sandbox captures reuse the canonical usage parser."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    capture = {
        "duration_ms": 12,
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps({"model": "claude-haiku-4-5-20251001"}).encode()
            ).decode(),
        },
        "response": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps(
                    {
                        "model": "claude-haiku-4-5-20251001",
                        "usage": {"input_tokens": 13, "output_tokens": 5},
                    }
                ).encode()
            ).decode(),
        },
    }

    class FakeSandbox:
        def __init__(self):
            self.uploads = []
            self.commands = []
            self.state_reads = 0

        async def upload_file(self, source_path, target_path):
            assert any(command.startswith("mkdir -p ") for command in self.commands)
            self.uploads.append((source_path, target_path))

        async def exec(self, command, timeout_sec=None):
            self.commands.append(command)
            if command.startswith("mkdir -p "):
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            if "command -v node" in command:
                return SimpleNamespace(
                    return_code=0, stdout="/usr/bin/node\n", stderr=""
                )
            if "node -e" in command or "node' -e" in command:
                assert "nohup" not in command
                assert "--agent-name" not in command
                return SimpleNamespace(return_code=0, stdout="123\n", stderr="")
            if "state.json" in command and command.strip().startswith("cat "):
                self.state_reads += 1
                if self.state_reads == 1:
                    return SimpleNamespace(return_code=0, stdout="{", stderr="")
                return SimpleNamespace(
                    return_code=0,
                    stdout='{"port":49000,"pid":123}\n',
                    stderr="",
                )
            if "captures.jsonl" in command and command.strip().startswith("cat "):
                return SimpleNamespace(
                    return_code=0,
                    stdout=json.dumps(capture) + "\n",
                    stderr="",
                )
            if "kill -TERM" in command:
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            if command.startswith("rm -rf "):
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            return SimpleNamespace(return_code=1, stdout="", stderr=command)

    sandbox = FakeSandbox()
    proxy = SandboxUsageProxy(
        sandbox=sandbox,
        target="https://api.anthropic.com",
        session_id="rollout-1",
        agent_name="claude-agent-acp",
    )
    await proxy.start()
    await proxy.stop()

    runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url=proxy.base_url,
        backend_model="claude-haiku-4-5-20251001",
        server=proxy,
    )
    usage = extract_usage(runtime)

    assert proxy.base_url == "http://127.0.0.1:49000"
    assert sandbox.state_reads == 2
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 13
    assert usage["n_output_tokens"] == 5


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_downloads_capture_log(tmp_path):
    """Guards PR #587: large sandbox capture logs avoid exec stdout limits."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    capture = {
        "duration_ms": 12,
        "request": {
            "method": "POST",
            "path": "/responses",
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps({"model": "gpt-5.5"}).encode()
            ).decode(),
        },
        "response": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "usage": {
                            "input_tokens": 21,
                            "output_tokens": 8,
                            "total_tokens": 29,
                        },
                    }
                ).encode()
            ).decode(),
        },
    }

    class FakeSandbox:
        def __init__(self):
            self.exec_commands = []
            self.downloads = []

        async def download_file(self, source_path, target_path):
            self.downloads.append((source_path, target_path))
            Path(target_path).write_text(json.dumps(capture) + "\n")

        async def exec(self, command, timeout_sec=None):
            self.exec_commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = FakeSandbox()
    proxy = SandboxUsageProxy(
        sandbox=sandbox,
        target="https://api.openai.com/v1",
        session_id="rollout-1",
        agent_name="codex-acp",
    )
    await proxy._load_captures()

    runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://127.0.0.1:49000",
        backend_model="gpt-5.5",
        server=proxy,
    )
    usage = extract_usage(runtime)

    assert [source for source, _target in sandbox.downloads] == [proxy._log_path]
    assert not any("captures.jsonl" in command for command in sandbox.exec_commands)
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 21
    assert usage["n_output_tokens"] == 8


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_liveness_reports_pid_status():
    """Guards PR #587: stale sandbox proxies are detected by PID liveness."""
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    class FakeSandbox:
        async def exec(self, command, timeout_sec=None):
            assert "kill -0" in command
            return SimpleNamespace(return_code=0, stdout="yes\n", stderr="")

    proxy = SandboxUsageProxy(
        sandbox=FakeSandbox(),
        target="https://api.anthropic.com",
        session_id="rollout-1",
        agent_name="claude-agent-acp",
    )

    assert await proxy.is_running() is True


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_stop_kills_and_cleans_when_capture_read_fails():
    """Guards PR #587: capture import failures still terminate the proxy."""
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    commands = []

    class FakeSandbox:
        async def exec(self, command, timeout_sec=None):
            commands.append(command)
            if "captures.jsonl" in command:
                raise TimeoutError("capture read timed out")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    proxy = SandboxUsageProxy(
        sandbox=FakeSandbox(),
        target="https://api.anthropic.com",
        session_id="rollout-1",
        agent_name="claude-agent-acp",
    )

    await proxy.stop()

    assert any("kill -TERM" in command for command in commands)
    assert any(command.startswith("rm -rf ") for command in commands)


@pytest.mark.asyncio
async def test_daytona_auto_usage_proxy_start_failure_leaves_env_untouched(monkeypatch):
    """Guards PR #587: auto mode degrades instead of failing Daytona runs."""
    from benchflow.providers.runtime import ensure_usage_proxy_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    stopped = []

    class BrokenSandboxUsageProxy:
        target = "https://api.anthropic.com"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        async def start(self):
            raise RuntimeError("launcher failed")

        async def stop(self):
            stopped.append(True)

    monkeypatch.setattr(
        usage_runtime_mod, "SandboxUsageProxy", BrokenSandboxUsageProxy
    )

    env = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="claude-haiku-4-5-20251001",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        usage_tracking=UsageTrackingConfig(mode="auto"),
        sandbox=object(),
    )

    assert updated == env
    assert runtime is None
    assert stopped == [True]


@pytest.mark.asyncio
async def test_daytona_required_usage_proxy_start_failure_raises(monkeypatch):
    """Guards PR #587: required mode still fails fast on proxy startup errors."""
    from benchflow.providers.runtime import ensure_usage_proxy_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    class BrokenSandboxUsageProxy:
        target = "https://api.anthropic.com"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        async def start(self):
            raise RuntimeError("launcher failed")

        async def stop(self):
            return None

    monkeypatch.setattr(
        usage_runtime_mod, "SandboxUsageProxy", BrokenSandboxUsageProxy
    )

    with pytest.raises(RuntimeError, match=r"required.*failed to start"):
        await ensure_usage_proxy_runtime(
            agent="claude-agent-acp",
            agent_env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
            model="claude-haiku-4-5-20251001",
            runtime=None,
            environment="daytona",
            session_id="rollout-1",
            usage_tracking=UsageTrackingConfig(mode="required"),
            sandbox=object(),
        )


@pytest.mark.asyncio
async def test_usage_runtime_recreated_when_sandbox_proxy_is_dead(monkeypatch):
    """Guards PR #587: dead sandbox proxies are not reused across reconnects."""
    from benchflow.providers.runtime import ProviderRuntime, ensure_usage_proxy_runtime

    stopped = []
    started = []

    class DeadServer:
        target = "https://api.anthropic.com"

        async def is_running(self):
            return False

        async def stop(self):
            stopped.append("dead")

    class FakeSandboxUsageProxy:
        target = "https://api.anthropic.com"
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            started.append(self.target)

        async def stop(self):
            stopped.append("new")

    monkeypatch.setattr(
        usage_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    stale_runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://127.0.0.1:49000",
        backend_model="claude-haiku-4-5-20251001",
        server=DeadServer(),
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        model="claude-haiku-4-5-20251001",
        runtime=stale_runtime,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert stopped == ["dead"]
    assert started == ["https://api.anthropic.com"]
    assert runtime is not None
    assert runtime is not stale_runtime
    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49001"


def test_raw_capture_json_error_beats_stream_request_hint():
    """Guards PR #587: JSON error responses are not parsed as SSE."""
    from benchflow.trajectories.proxy import exchange_from_raw_capture

    exchange = exchange_from_raw_capture(
        {
            "request": {
                "method": "POST",
                "path": "/v1/messages",
                "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(
                    json.dumps({"stream": True}).encode()
                ).decode(),
            },
            "response": {
                "status_code": 400,
                "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(
                    json.dumps(
                        {"error": {"message": "Budget has been exceeded"}}
                    ).encode()
                ).decode(),
            },
        }
    )

    assert exchange.response.body["error"]["message"] == "Budget has been exceeded"


def test_node_proxy_forwards_and_imports_redacted_raw_captures(tmp_path):
    """Guards PR #587: Node proxy forwards, redacts, and imports captures."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from benchflow.providers.sandbox_usage_proxy import _NODE_PROXY_SOURCE
    from benchflow.trajectories.proxy import exchange_from_raw_capture

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for sandbox usage proxy integration smoke")

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            if self.path == "/v1/error":
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.send_header("set-cookie", "secret-cookie")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"error": {"message": "Budget has been exceeded"}}
                    ).encode()
                )
                return
            if self.path == "/v1/stream":
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(
                    b'data: {"model":"gpt-4.1-mini","choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n'
                )
                return
            if self.path == "/v1/gzip":
                payload = gzip.compress(
                    json.dumps(
                        {
                            "model": "gpt-4.1-mini",
                            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                        }
                    ).encode()
                )
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-encoding", "gzip")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "received_gzip": self.headers.get("content-encoding") == "gzip",
                        "body_len": len(body),
                        "model": "claude-haiku-4-5-20251001",
                        "usage": {"input_tokens": 13, "output_tokens": 5},
                    }
                ).encode()
            )

        def log_message(self, *_args):
            return None

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    runtime_dir = tmp_path / "proxy"
    runtime_dir.mkdir()
    script = runtime_dir / "proxy.js"
    state = runtime_dir / "state.json"
    log_path = runtime_dir / "captures.jsonl"
    pid_path = runtime_dir / "proxy.pid"
    script.write_text(_NODE_PROXY_SOURCE)
    env = {
        **os.environ,
        "BENCHFLOW_USAGE_PROXY_TARGET": (
            f"http://127.0.0.1:{upstream.server_address[1]}"
        ),
        "BENCHFLOW_USAGE_PROXY_STATE_PATH": str(state),
        "BENCHFLOW_USAGE_PROXY_LOG_PATH": str(log_path),
        "BENCHFLOW_USAGE_PROXY_PID_PATH": str(pid_path),
        "BENCHFLOW_USAGE_PROXY_SESSION_ID": "rollout-1",
        "BENCHFLOW_USAGE_PROXY_AGENT_NAME": "codex-acp",
    }
    proc = subprocess.Popen([node, str(script)], env=env)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not state.exists():
            time.sleep(0.05)
        assert state.exists()
        proxy_port = json.loads(state.read_text())["port"]

        def post(path, payload, headers=None):
            body = (
                payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            )
            request = Request(
                f"http://127.0.0.1:{proxy_port}{path}",
                data=body,
                headers={"content-type": "application/json", **(headers or {})},
                method="POST",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    return response.status, response.read()
            except HTTPError as exc:
                return exc.code, exc.read()

        gzipped_request = gzip.compress(json.dumps({"stream": False}).encode())
        assert (
            post(
                "/v1/messages?key=secret-query&safe=1",
                gzipped_request,
                {
                    "authorization": "Bearer secret",
                    "x-api-key": "secret",
                    "content-encoding": "gzip",
                },
            )[0]
            == 200
        )
        assert post("/v1/error", {"stream": True})[0] == 400
        assert post("/v1/stream", {"stream": True})[0] == 200
        assert post("/v1/gzip", {"stream": False})[0] == 200

        capture_lines = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            capture_lines = log_path.read_text().splitlines()
            if len(capture_lines) >= 4:
                break
            time.sleep(0.05)
        captures = [json.loads(line) for line in capture_lines]
        exchanges = [exchange_from_raw_capture(record) for record in captures]

        assert len(exchanges) == 4
        first = exchanges[0]
        assert first.request.path == "/v1/messages?key=__BENCHFLOW_REDACTED__&safe=1"
        assert first.request.headers["authorization"] == "__BENCHFLOW_REDACTED__"
        assert first.request.headers["x-api-key"] == "__BENCHFLOW_REDACTED__"
        assert first.request.body["stream"] is False
        assert first.response.body["usage"]["input_tokens"] == 13

        error_exchange = exchanges[1]
        assert error_exchange.response.status_code == 400
        assert (
            error_exchange.response.body["error"]["message"]
            == "Budget has been exceeded"
        )
        assert error_exchange.response.headers["set-cookie"] == "__BENCHFLOW_REDACTED__"

        stream_exchange = exchanges[2]
        assert stream_exchange.response.body["choices"][0]["message"]["content"] == "hi"
        assert stream_exchange.response.body["usage"]["total_tokens"] == 5

        gzip_exchange = exchanges[3]
        assert gzip_exchange.response.body["usage"]["prompt_tokens"] == 7
    finally:
        with contextlib.suppress(ProcessLookupError, FileNotFoundError):
            os.kill(int(pid_path.read_text()), signal.SIGTERM)
        proc.wait(timeout=5)
        upstream.shutdown()

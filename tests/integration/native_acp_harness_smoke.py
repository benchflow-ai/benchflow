"""Credential-free Docker smoke for native OpenScience and DSH ACP profiles.

This checker builds the registry install command in a clean Ubuntu image, starts
the BenchFlow launcher, and exercises ACP initialize + session/new without
sending a model prompt.  It deliberately uses a dead local provider endpoint
and a fake key; no provider credential is required or logged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from uuid import uuid4

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from benchflow.acp.client import ACPClient
from benchflow.acp.runtime import _prompt_with_wall_clock_budget
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import ImageContent, McpServerSpec
from benchflow.agents.registry import _DEEPSEEK_HARNESS_INTEGRITY, AGENTS
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.trajectories._capture import _capture_session_trajectory

SUPPORTED = ("openscience", "deepseek-harness")
SKILL_NAME = "benchflow-smoke-skill"
SKILL_DESCRIPTION_MARKER = "BENCHFLOW_SKILL_DESCRIPTION_4f51c1"
SKILL_BODY_MARKER = "BENCHFLOW_SKILL_BODY_8d7f20"
MCP_RESULT_MARKER = "MCP_ECHO_OK_78c29b"
PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _latest_user_text(request: dict[str, object]) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if text.startswith("Current runtime context."):
            continue
        return text
    return ""


def _mcp_echo_tool_name(request: dict[str, object]) -> str | None:
    tools = request.get("tools")
    if not isinstance(tools, list):
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        function = tool.get("function")
        if not isinstance(name, str) and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and "echo" in name.lower():
            return name
    return None


def _tool_names(request: dict[str, object]) -> list[str]:
    tools = request.get("tools")
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        function = tool.get("function")
        if not isinstance(name, str) and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


class _MockMCP:
    def __init__(self) -> None:
        self.called = Event()
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            self.port = probe.getsockname()[1]
        app = FastMCP(
            "benchflow-smoke-mcp",
            host="0.0.0.0",
            port=self.port,
            json_response=True,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
        )

        @app.tool()
        def benchflow_mcp_echo(value: str = "") -> str:
            """Return deterministic BenchFlow MCP smoke evidence."""
            self.called.set()
            return f"{MCP_RESULT_MARKER}:{value}"

        config = uvicorn.Config(
            app.streamable_http_app(),
            host="0.0.0.0",
            port=self.port,
            log_level="critical",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.thread = Thread(target=self.server.run, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://host.docker.internal:{self.port}/mcp"

    @property
    def loopback_endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def __enter__(self) -> _MockMCP:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.server.started:
            raise RuntimeError("mock MCP server did not start")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


class _MockProvider:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.requests: list[dict[str, object]] = []
        self.cancel_started = Event()
        self.cancel_release = Event()
        self.timeout_started = Event()
        self.timeout_release = Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def handle(self) -> None:
                with suppress(BrokenPipeError, ConnectionResetError):
                    super().handle()

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                owner.paths.append(self.path)
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "deepseek-v4-flash",
                                "object": "model",
                                "created": 1,
                                "owned_by": "benchflow-smoke",
                            },
                            {
                                "id": "deepseek-v4-pro",
                                "object": "model",
                                "created": 1,
                                "owned_by": "benchflow-smoke",
                            },
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                owner.paths.append(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                raw_request = self.rfile.read(length) or b"{}"
                request = json.loads(raw_request)
                owner.requests.append(request)
                active_prompt = _latest_user_text(request)
                is_messages = self.path.endswith("/messages")
                if "PROVIDER_ERROR_SMOKE" in active_prompt:
                    body = json.dumps(
                        {
                            "error": {
                                "message": "credential-free provider error smoke",
                                "type": "benchflow_smoke_error",
                            }
                        }
                    ).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if "CANCEL_SMOKE" in active_prompt:
                    owner.cancel_started.set()
                    owner.cancel_release.wait(timeout=30)
                if "TIMEOUT_SMOKE" in active_prompt:
                    owner.timeout_started.set()
                    owner.timeout_release.wait(timeout=30)
                if is_messages:
                    tool_smoke = "TOOL_SMOKE" in active_prompt
                    tool_result = (
                        b'"type":"tool_result"' in raw_request
                        or b'"type": "tool_result"' in raw_request
                    )
                    skill_smoke = "SKILL_FIDELITY_SMOKE" in active_prompt
                    skill_result = SKILL_BODY_MARKER.encode() in raw_request
                    reasoning_smoke = "REASONING_SMOKE" in active_prompt
                    artifact_smoke = "ARTIFACT_SMOKE" in active_prompt
                    artifact_result = tool_result
                    mcp_smoke = "MCP_SMOKE" in active_prompt
                    mcp_result = MCP_RESULT_MARKER.encode() in raw_request
                    mcp_tool_name = _mcp_echo_tool_name(request)
                    if mcp_smoke and not mcp_tool_name:
                        mcp_smoke = False
                    start = {
                        "type": "message_start",
                        "message": {
                            "id": "msg_benchflow_smoke",
                            "model": request.get("model", "deepseek-v4-flash"),
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                        },
                    }
                    if reasoning_smoke:
                        blocks = [
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "",
                                },
                            },
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": "REASONING_OK",
                                },
                            },
                            {"type": "content_block_stop", "index": 0},
                            {
                                "type": "content_block_start",
                                "index": 1,
                                "content_block": {"type": "text", "text": ""},
                            },
                            {
                                "type": "content_block_delta",
                                "index": 1,
                                "delta": {"type": "text_delta", "text": "SMOKE_OK"},
                            },
                            {"type": "content_block_stop", "index": 1},
                        ]
                        stop_reason = "end_turn"
                    elif (
                        (tool_smoke and not tool_result)
                        or (skill_smoke and not skill_result)
                        or (artifact_smoke and not artifact_result)
                        or (mcp_smoke and not mcp_result)
                    ):
                        tool_name = (
                            mcp_tool_name
                            if mcp_smoke
                            else "skill"
                            if skill_smoke
                            else "bash"
                        )
                        tool_arguments = (
                            {"value": "MCP_ECHO_INPUT"}
                            if mcp_smoke
                            else {"name": SKILL_NAME}
                            if skill_smoke
                            else {
                                "command": "printf 'Hello from benchflow!\\n' > hello.txt",
                                "description": "Create the demo task artifact",
                                "workdir": "/app",
                            }
                            if artifact_smoke
                            else {
                                "command": 'printf \'UID=%s CWD=%s\' "$(id -u)" "$PWD"; printf WORKSPACE_OK > benchflow_workspace_probe.txt',
                                "description": "Print runtime identity and workspace",
                                "workdir": "/workspace",
                            }
                        )
                        blocks = [
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": "call_benchflow_smoke",
                                    "name": tool_name,
                                    "input": {},
                                },
                            },
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(tool_arguments),
                                },
                            },
                            {"type": "content_block_stop", "index": 0},
                        ]
                        stop_reason = "tool_use"
                    else:
                        content = (
                            "SKILL_FIDELITY_DONE"
                            if skill_smoke
                            else "MCP_SMOKE_DONE"
                            if mcp_smoke
                            else "ARTIFACT_SMOKE_DONE"
                            if artifact_smoke
                            else "TOOL_SMOKE_DONE"
                            if tool_smoke
                            else "SMOKE_OK"
                        )
                        blocks = [
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            },
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": content},
                            },
                            {"type": "content_block_stop", "index": 0},
                        ]
                        stop_reason = "end_turn"
                    events = [
                        start,
                        *blocks,
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason},
                            "usage": {"output_tokens": 1},
                        },
                        {"type": "message_stop"},
                    ]
                    body = "".join(
                        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                        for event in events
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    with suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(body)
                    return
                if request.get("stream"):
                    tool_smoke = "TOOL_SMOKE" in active_prompt
                    tool_result = (
                        b'"role":"tool"' in raw_request
                        or b'"role": "tool"' in raw_request
                    )
                    skill_smoke = "SKILL_FIDELITY_SMOKE" in active_prompt
                    skill_result = SKILL_BODY_MARKER.encode() in raw_request
                    reasoning_smoke = "REASONING_SMOKE" in active_prompt
                    artifact_smoke = "ARTIFACT_SMOKE" in active_prompt
                    artifact_result = tool_result
                    mcp_smoke = "MCP_SMOKE" in active_prompt
                    mcp_result = MCP_RESULT_MARKER.encode() in raw_request
                    mcp_tool_name = _mcp_echo_tool_name(request)
                    if mcp_smoke and not mcp_tool_name:
                        mcp_smoke = False
                    if reasoning_smoke:
                        chunks = [
                            {
                                "id": "chatcmpl-benchflow-smoke-reasoning",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": request.get("model", "deepseek-v4-flash"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "reasoning_content": "REASONING_OK",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            {
                                "id": "chatcmpl-benchflow-smoke-reasoning",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": request.get("model", "deepseek-v4-flash"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "SMOKE_OK"},
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            {
                                "id": "chatcmpl-benchflow-smoke-reasoning",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": request.get("model", "deepseek-v4-flash"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 3,
                                    "completion_tokens": 2,
                                    "total_tokens": 5,
                                    "completion_tokens_details": {
                                        "reasoning_tokens": 1
                                    },
                                },
                            },
                        ]
                    elif (
                        (tool_smoke and not tool_result)
                        or (skill_smoke and not skill_result)
                        or (artifact_smoke and not artifact_result)
                        or (mcp_smoke and not mcp_result)
                    ):
                        tool_name = (
                            mcp_tool_name
                            if mcp_smoke
                            else "skill"
                            if skill_smoke
                            else "bash"
                        )
                        tool_arguments = (
                            {"value": "MCP_ECHO_INPUT"}
                            if mcp_smoke
                            else {"name": SKILL_NAME}
                            if skill_smoke
                            else {
                                "command": "printf 'Hello from benchflow!\\n' > hello.txt",
                                "description": "Create the demo task artifact",
                                "workdir": "/app",
                            }
                            if artifact_smoke
                            else {
                                "command": 'printf \'UID=%s CWD=%s\' "$(id -u)" "$PWD"; printf WORKSPACE_OK > benchflow_workspace_probe.txt',
                                "description": "Print runtime identity and workspace",
                                "workdir": "/workspace",
                            }
                        )
                        chunks = [
                            {
                                "id": "chatcmpl-benchflow-smoke-tool",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": "deepseek-v4-flash",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": "call_benchflow_smoke",
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_name,
                                                        "arguments": json.dumps(
                                                            tool_arguments
                                                        ),
                                                    },
                                                }
                                            ],
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            {
                                "id": "chatcmpl-benchflow-smoke-tool",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": "deepseek-v4-flash",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 3,
                                    "completion_tokens": 1,
                                    "total_tokens": 4,
                                },
                            },
                        ]
                    else:
                        content = (
                            "SKILL_FIDELITY_DONE"
                            if skill_smoke
                            else "MCP_SMOKE_DONE"
                            if mcp_smoke
                            else "ARTIFACT_SMOKE_DONE"
                            if artifact_smoke
                            else "TOOL_SMOKE_DONE"
                            if tool_smoke
                            else "SMOKE_OK"
                        )
                        chunks = [
                            {
                                "id": "chatcmpl-benchflow-smoke",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": "deepseek-v4-flash",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            {
                                "id": "chatcmpl-benchflow-smoke",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": "deepseek-v4-flash",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 3,
                                    "completion_tokens": 1,
                                    "total_tokens": 4,
                                },
                            },
                        ]
                    body = (
                        b"".join(
                            f"data: {json.dumps(chunk)}\n\n".encode()
                            for chunk in chunks
                        )
                        + b"data: [DONE]\n\n"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    with suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(body)
                    return
                body = json.dumps(
                    {
                        "id": "chatcmpl-benchflow-smoke",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "SMOKE_OK"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://host.docker.internal:{self.server.server_port}/v1"

    def __enter__(self) -> _MockProvider:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _build_image(agent: str, image: str) -> None:
    # BuildKit runs in a separate network namespace in some CI/dev hosts and
    # may not share the working registry/GitHub route used by rollout
    # containers.  Install through a normal disposable container, then commit
    # exactly that filesystem for the protocol smoke.
    container = f"bf-native-acp-smoke-{agent}-{uuid4().hex[:10]}"
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--name",
                container,
                "ubuntu:22.04",
                "sh",
                "-c",
                AGENTS[agent].install_cmd,
            ],
            check=True,
            timeout=AGENTS[agent].install_timeout + 180,
        )
        subprocess.run(["docker", "commit", container, image], check=True, timeout=120)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _assert_native_failure_paths(agent: str, image: str) -> dict[str, object]:
    cfg = AGENTS[agent]
    if agent == "openscience":
        failure_cmd = (
            "mkdir -p /tmp/benchflow-fail-bin && "
            "printf '%s\\n' '#!/bin/sh' 'exit 22' > /tmp/benchflow-fail-bin/curl && "
            "chmod +x /tmp/benchflow-fail-bin/curl && "
            "rm -f /opt/benchflow/bin/openscience && "
            'export PATH="/tmp/benchflow-fail-bin:$PATH" && ' + cfg.install_cmd
        )
        failed = subprocess.run(
            ["docker", "run", "--rm", image, "sh", "-c", failure_cmd],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if failed.returncode == 0:
            raise RuntimeError("OpenScience install ignored a forced download failure")
        return {"install_failure_rc": failed.returncode}

    bad_integrity_cmd = cfg.install_cmd.replace(
        _DEEPSEEK_HARNESS_INTEGRITY,
        "sha512-intentionally-invalid-benchflow-smoke",
    )
    failed = subprocess.run(
        ["docker", "run", "--rm", image, "sh", "-c", bad_integrity_cmd],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    if failed.returncode == 0:
        raise RuntimeError("DSH install ignored an npm integrity mismatch")

    boot = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image,
            "/opt/benchflow/bin/dsh",
            "--profile",
            "acp",
            "--patch",
            "/definitely/missing/benchflow.patch.yml",
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if boot.returncode == 0:
        raise RuntimeError("DSH ACP profile accepted a missing patch")
    return {
        "install_failure_rc": failed.returncode,
        "profile_boot_failure_rc": boot.returncode,
    }


def _agent_env(
    agent: str,
    provider: _MockProvider,
    model: str = "deepseek-v4-flash",
    protocol: str = "openai-completions",
) -> list[str]:
    env = [
        "-e",
        "HOME=/tmp/benchflow-agent",
        "-e",
        "BENCHFLOW_AGENT_HOME=/tmp/benchflow-agent",
        "-e",
        "BENCHFLOW_WORKSPACE=/workspace",
        "-e",
        f"BENCHFLOW_PROVIDER_MODEL={model}",
        "-e",
        f"BENCHFLOW_PROVIDER_BASE_URL={provider.endpoint}",
        "-e",
        "BENCHFLOW_PROVIDER_API_KEY=credential-free-smoke",
        "-e",
        f"BENCHFLOW_PROVIDER_PROTOCOL={protocol}",
    ]
    if agent == "openscience":
        env += [
            "-e",
            f"OPENSCIENCE_BENCHFLOW_MODEL={model}",
            "-e",
            "OPENSCIENCE_BENCHFLOW_API_KEY=credential-free-smoke",
        ]
    else:
        env += [
            "-e",
            f"DSH_BENCHFLOW_MODEL={model}",
            "-e",
            "DEEPSEEK_API_KEY=credential-free-smoke",
            "-e",
            f"DEEPSEEK_BASE_URL={provider.endpoint}",
        ]
    return env


def _transport(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
    model: str = "deepseek-v4-flash",
    protocol: str = "openai-completions",
    host_network: bool = False,
) -> StdioTransport:
    cfg = AGENTS[agent]
    workspace_dir = agent_home / "task-workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return StdioTransport(
        "docker",
        [
            "run",
            "--rm",
            "-i",
            *(["--network", "host"] if host_network else []),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-v",
            f"{agent_home}:/tmp/benchflow-agent",
            "-v",
            f"{workspace_dir}:/workspace",
            "-w",
            "/workspace",
            *_agent_env(agent, provider, model, protocol),
            image,
            "sh",
            "-c",
            cfg.launch_cmd,
        ],
    )


async def _smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
    protocol: str = "openai-completions",
) -> dict[str, object]:
    transport = _transport(agent, image, provider, agent_home, protocol=protocol)
    client = ACPClient(transport)
    try:
        await asyncio.wait_for(client.connect(), timeout=30)
        initialized = await asyncio.wait_for(client.initialize(), timeout=60)
        session = await asyncio.wait_for(
            client.session_new(cwd="/workspace"), timeout=90
        )
        capabilities = initialized.agent_capabilities
        prompt_capabilities = capabilities.prompt_capabilities
        image_advertised = bool(prompt_capabilities and prompt_capabilities.image)
        image_prompt_stop_reason: str | None = None
        if image_advertised:
            image_prompt = await asyncio.wait_for(
                client.prompt(
                    "IMAGE_SMOKE: reply with SMOKE_OK.",
                    content=[
                        ImageContent(
                            type="image",
                            data=PNG_1X1_BASE64,
                            mime_type="image/png",
                            uri="benchflow-smoke.png",
                        )
                    ],
                ),
                timeout=120,
            )
            image_prompt_stop_reason = str(image_prompt.stop_reason)
        prompt = await asyncio.wait_for(
            client.prompt("Reply with SMOKE_OK and do not use tools."), timeout=120
        )
        reasoning_prompt = await asyncio.wait_for(
            client.prompt("REASONING_SMOKE: reason briefly, then reply SMOKE_OK."),
            timeout=120,
        )
        tool_prompt = await asyncio.wait_for(
            client.prompt("TOOL_SMOKE: run printf TOOL_OK with bash, then finish."),
            timeout=120,
        )
        cancel_task = asyncio.create_task(client.prompt("CANCEL_SMOKE"))
        cancel_seen = await asyncio.wait_for(
            asyncio.to_thread(provider.cancel_started.wait, 30), timeout=35
        )
        if not cancel_seen:
            raise RuntimeError("mock provider did not receive cancellation prompt")
        await client.cancel()
        await asyncio.sleep(0.25)
        provider.cancel_release.set()
        cancelled = await asyncio.wait_for(cancel_task, timeout=30)
        session.record_user_prompt("TIMEOUT_SMOKE")
        timeout_error: AgentPromptTimeoutError | None = None
        try:
            await _prompt_with_wall_clock_budget(
                client,
                session,
                "TIMEOUT_SMOKE",
                timeout=1,
            )
        except AgentPromptTimeoutError as exc:
            timeout_error = exc
        finally:
            provider.timeout_release.set()
        if not provider.timeout_started.is_set():
            raise RuntimeError("mock provider did not receive timeout prompt")
        if timeout_error is None:
            raise RuntimeError("BenchFlow wall-clock timeout was not raised")
        options = session.config_options or []
        tool_statuses = [str(call.status) for call in session.tool_calls]
        if not tool_statuses:
            raise RuntimeError("native ACP smoke did not expose a tool call")
        if any(status != "completed" for status in tool_statuses):
            raise RuntimeError(
                f"native ACP tool call did not complete: {tool_statuses}"
            )
        tool_evidence = json.dumps(
            [
                {
                    "content": call.content,
                    "raw_output": call.raw_output,
                }
                for call in session.tool_calls
            ],
            default=str,
            sort_keys=True,
        )
        expected_runtime = f"UID={os.getuid()} CWD=/workspace"
        if expected_runtime not in tool_evidence:
            raise RuntimeError(
                "native tool escaped expected uid/workspace boundary: " + tool_evidence
            )
        workspace_probe = agent_home / "task-workspace/benchflow_workspace_probe.txt"
        if workspace_probe.read_text(encoding="utf-8") != "WORKSPACE_OK":
            raise RuntimeError("native tool output did not persist in task workspace")
        if "REASONING_OK" not in session.full_thought:
            raise RuntimeError("native ACP smoke did not expose provider reasoning")
        trajectory = _capture_session_trajectory(session)
        event_types = sorted(
            {
                event_type
                for event in trajectory
                if isinstance(event, dict)
                and isinstance((event_type := event.get("type")), str)
            }
        )
        required_event_types = {"agent_thought", "tool_call", "agent_timeout"}
        if not required_event_types.issubset(event_types):
            raise RuntimeError(
                f"native ACP trajectory missing required events: {event_types}"
            )
        return {
            "acp_event_types": event_types,
            "agent": agent,
            "protocol_version": getattr(initialized, "protocol_version", None),
            "session_id": session.session_id,
            "stop_reason": str(prompt.stop_reason),
            "reasoning_stop_reason": str(reasoning_prompt.stop_reason),
            "reasoning_visible": True,
            "tool_stop_reason": str(tool_prompt.stop_reason),
            "tool_calls": len(session.tool_calls),
            "tool_statuses": tool_statuses,
            "tool_runtime": expected_runtime,
            "cancel_stop_reason": str(cancelled.stop_reason),
            "timeout_error": type(timeout_error).__name__,
            "provider_paths": provider.paths,
            "config_option_ids": sorted(
                option["id"]
                for option in options
                if isinstance(option, dict) and isinstance(option.get("id"), str)
            ),
            "image_advertised": image_advertised,
            "image_prompt_stop_reason": image_prompt_stop_reason,
            "mcp_http_advertised": bool(
                capabilities.mcp_capabilities and capabilities.mcp_capabilities.http
            ),
            "session_load_advertised": bool(capabilities.load_session),
            "session_resume_advertised": bool(
                capabilities.session_capabilities
                and capabilities.session_capabilities.resume is not None
            ),
        }
    finally:
        provider.cancel_release.set()
        provider.timeout_release.set()
        with suppress(Exception):
            await client.close()


async def _mcp_smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    mcp: _MockMCP,
    agent_home: Path,
) -> dict[str, object]:
    before = len(provider.requests)
    mcp_servers: list[McpServerSpec] = [
        McpServerSpec(name="benchflow-smoke", type="http", url=mcp.endpoint)
    ]
    if agent == "openscience":
        task_mcp_path = agent_home / ".openscience-benchflow/config/task-mcp.json"
        task_mcp_path.parent.mkdir(parents=True, exist_ok=True)
        task_mcp_path.write_text(
            json.dumps(
                {
                    "mcp": {
                        "benchflow-smoke": {
                            "type": "remote",
                            "url": mcp.loopback_endpoint,
                            "headers": {},
                            "oauth": False,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        mcp_servers = []
    transport = _transport(
        agent,
        image,
        provider,
        agent_home,
        host_network=agent == "openscience",
    )
    client = ACPClient(transport)
    agent_stderr = ""
    try:
        await asyncio.wait_for(client.connect(), timeout=30)
        initialized = await asyncio.wait_for(client.initialize(), timeout=60)
        capabilities = initialized.agent_capabilities.mcp_capabilities
        if not capabilities or not capabilities.http:
            raise RuntimeError(f"{agent} did not advertise HTTP MCP")
        session = await asyncio.wait_for(
            client.session_new(
                cwd="/workspace",
                mcp_servers=mcp_servers,
            ),
            timeout=90,
        )
        await asyncio.sleep(2)
        prompt = await asyncio.wait_for(
            client.prompt(
                "MCP_SMOKE: call the benchflow MCP echo tool with any value, then finish."
            ),
            timeout=120,
        )
    finally:
        with suppress(Exception):
            await client.close()
        process = transport._process
        if process and process.stderr:
            with suppress(Exception):
                agent_stderr = (await process.stderr.read()).decode(errors="replace")

    if not mcp.called.is_set():
        tool_names = sorted(
            {
                name
                for request in provider.requests[before:]
                for name in _tool_names(request)
            }
        )
        raise RuntimeError(
            f"{agent} did not invoke the task-declared MCP tool; "
            f"provider tools={tool_names}; "
            f"user texts={[_latest_user_text(r) for r in provider.requests[before:]]}; "
            f"stderr={agent_stderr[-4000:]}"
        )
    payload = json.dumps(provider.requests[before:], sort_keys=True)
    if MCP_RESULT_MARKER not in payload:
        raise RuntimeError(f"{agent} did not return MCP tool evidence to the model")
    statuses = [str(call.status) for call in session.tool_calls]
    if not statuses or any(status != "completed" for status in statuses):
        raise RuntimeError(f"{agent} MCP ACP tool lifecycle incomplete: {statuses}")
    return {
        "mcp_called": True,
        "mcp_result_visible": True,
        "mcp_stop_reason": str(prompt.stop_reason),
        "mcp_tool_statuses": statuses,
    }


async def _session_recovery_smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
) -> dict[str, object]:
    first = ACPClient(_transport(agent, image, provider, agent_home))
    try:
        await asyncio.wait_for(first.connect(), timeout=30)
        await asyncio.wait_for(first.initialize(), timeout=60)
        session = await asyncio.wait_for(
            first.session_new(cwd="/workspace"), timeout=90
        )
        await asyncio.wait_for(
            first.prompt("RECOVERY_SEED_SMOKE: reply with SMOKE_OK."), timeout=120
        )
        session_id = session.session_id
    finally:
        with suppress(Exception):
            await first.close()

    recovered = ACPClient(_transport(agent, image, provider, agent_home))
    try:
        await asyncio.wait_for(recovered.connect(), timeout=30)
        await asyncio.wait_for(recovered.initialize(), timeout=60)
        restored = await asyncio.wait_for(
            recovered.session_recover(session_id, cwd="/workspace"), timeout=90
        )
        follow_up = await asyncio.wait_for(
            recovered.prompt("RECOVERY_FOLLOWUP_SMOKE: reply with SMOKE_OK."),
            timeout=120,
        )
    finally:
        with suppress(Exception):
            await recovered.close()

    return {
        "recovered_session_id": restored.session_id,
        "recovery_stop_reason": str(follow_up.stop_reason),
    }


def _write_smoke_skill(agent: str, agent_home: Path) -> None:
    root = ".claude/skills" if agent == "openscience" else ".agents/skills"
    skill_dir = agent_home / root / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {SKILL_NAME}",
                f"description: {SKILL_DESCRIPTION_MARKER}",
                "---",
                "",
                SKILL_BODY_MARKER,
                "",
            ]
        ),
        encoding="utf-8",
    )


async def _skill_fidelity_smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
    protocol: str = "openai-completions",
) -> dict[str, object]:
    before = len(provider.requests)
    _write_smoke_skill(agent, agent_home)
    transport = _transport(agent, image, provider, agent_home, protocol=protocol)
    client = ACPClient(transport)
    try:
        await asyncio.wait_for(client.connect(), timeout=30)
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/workspace"), timeout=90)
        skill_prompt = (
            f"/{SKILL_NAME}\nSKILL_FIDELITY_SMOKE"
            if agent == "deepseek-harness"
            else f"SKILL_FIDELITY_SMOKE: load {SKILL_NAME} with the skill tool."
        )
        prompt = await asyncio.wait_for(client.prompt(skill_prompt), timeout=120)
    finally:
        with suppress(Exception):
            await client.close()

    no_skill_payload = json.dumps(provider.requests[:before], sort_keys=True)
    with_skill_payload = json.dumps(provider.requests[before:], sort_keys=True)
    diagnostics = {
        "models": [request.get("model") for request in provider.requests[before:]],
        "skill_name_visible": SKILL_NAME in with_skill_payload,
        "skill_description_visible": SKILL_DESCRIPTION_MARKER in with_skill_payload,
        "skill_body_visible": SKILL_BODY_MARKER in with_skill_payload,
        "tool_names": sorted(
            {
                function.get("name")
                for request in provider.requests[before:]
                for tool in request.get("tools", [])
                if isinstance(tool, dict)
                and isinstance((function := tool.get("function")), dict)
                and isinstance(function.get("name"), str)
            }
        ),
    }
    if (
        SKILL_DESCRIPTION_MARKER in no_skill_payload
        or SKILL_BODY_MARKER in no_skill_payload
    ):
        raise RuntimeError("no-skill native session unexpectedly exposed smoke skill")
    if SKILL_NAME not in with_skill_payload:
        raise RuntimeError(
            "with-skill native session did not advertise selected skill: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    if SKILL_BODY_MARKER not in with_skill_payload:
        raise RuntimeError(
            "with-skill native session did not inject selected skill body: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    return {
        "skill_name": SKILL_NAME,
        "skill_catalog_visible": True,
        "skill_description_visible": SKILL_DESCRIPTION_MARKER in with_skill_payload,
        "skill_body_visible": True,
        "skill_stop_reason": str(prompt.stop_reason),
    }


async def _model_selection_smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
    protocol: str = "openai-completions",
) -> dict[str, object]:
    before = len(provider.requests)
    transport = _transport(
        agent,
        image,
        provider,
        agent_home,
        model="deepseek-v4-pro",
        protocol=protocol,
    )
    client = ACPClient(transport)
    try:
        await asyncio.wait_for(client.connect(), timeout=30)
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/workspace"), timeout=90)
        prompt = await asyncio.wait_for(
            client.prompt("PRO_MODEL_SMOKE: reply with SMOKE_OK."), timeout=120
        )
    finally:
        with suppress(Exception):
            await client.close()

    models = [request.get("model") for request in provider.requests[before:]]
    if not models or any(model != "deepseek-v4-pro" for model in models):
        raise RuntimeError(f"native ACP Pro model routing mismatch: {models}")
    return {
        "pro_model": "deepseek-v4-pro",
        "pro_stop_reason": str(prompt.stop_reason),
    }


async def _provider_error_smoke(
    agent: str,
    image: str,
    provider: _MockProvider,
    agent_home: Path,
    protocol: str = "openai-completions",
) -> dict[str, object]:
    before = len(provider.requests)
    transport = _transport(agent, image, provider, agent_home, protocol=protocol)
    client = ACPClient(transport)
    error_type: str | None = None
    error_stop_reason: str | None = None
    try:
        await asyncio.wait_for(client.connect(), timeout=30)
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/workspace"), timeout=90)
        try:
            failed = await asyncio.wait_for(
                client.prompt("PROVIDER_ERROR_SMOKE"), timeout=120
            )
            error_stop_reason = str(failed.stop_reason)
        except Exception as exc:
            error_type = type(exc).__name__
    finally:
        with suppress(Exception):
            await client.close()

    recovery_transport = _transport(
        agent, image, provider, agent_home, protocol=protocol
    )
    recovery_client = ACPClient(recovery_transport)
    try:
        await asyncio.wait_for(recovery_client.connect(), timeout=30)
        await asyncio.wait_for(recovery_client.initialize(), timeout=60)
        await asyncio.wait_for(
            recovery_client.session_new(cwd="/workspace"), timeout=90
        )
        recovery = await asyncio.wait_for(
            recovery_client.prompt("PROVIDER_RECOVERY_SMOKE: reply with SMOKE_OK."),
            timeout=120,
        )
    finally:
        with suppress(Exception):
            await recovery_client.close()

    requests = provider.requests[before:]
    if not any("PROVIDER_ERROR_SMOKE" in json.dumps(request) for request in requests):
        raise RuntimeError("mock provider did not receive provider-error prompt")
    return {
        "provider_error_exception": error_type,
        "provider_error_stop_reason": error_stop_reason,
        "provider_recovery_stop_reason": str(recovery.stop_reason),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=SUPPORTED)
    parser.add_argument("--keep-image", action="store_true")
    parser.add_argument("--reuse-image", action="store_true")
    args = parser.parse_args()
    image = f"benchflow-native-acp-smoke:{args.agent}"
    try:
        if not args.reuse_image:
            _build_image(args.agent, image)
        failure_evidence = _assert_native_failure_paths(args.agent, image)
        with (
            TemporaryDirectory(prefix=f"bf-{args.agent}-home-") as home,
            _MockProvider() as provider,
        ):
            result = asyncio.run(_smoke(args.agent, image, provider, Path(home)))
            result.update(failure_evidence)
            with _MockMCP() as mcp:
                result.update(
                    asyncio.run(
                        _mcp_smoke(args.agent, image, provider, mcp, Path(home))
                    )
                )
            result.update(
                asyncio.run(
                    _skill_fidelity_smoke(args.agent, image, provider, Path(home))
                )
            )
            result.update(
                asyncio.run(
                    _model_selection_smoke(args.agent, image, provider, Path(home))
                )
            )
            result.update(
                asyncio.run(
                    _provider_error_smoke(args.agent, image, provider, Path(home))
                )
            )
            result.update(
                asyncio.run(
                    _session_recovery_smoke(args.agent, image, provider, Path(home))
                )
            )
        if args.agent in SUPPORTED:
            with (
                TemporaryDirectory(
                    prefix=f"bf-{args.agent}-anthropic-home-"
                ) as anthropic_home,
                _MockProvider() as anthropic_provider,
            ):
                anthropic_result = asyncio.run(
                    _smoke(
                        args.agent,
                        image,
                        anthropic_provider,
                        Path(anthropic_home),
                        protocol="anthropic-messages",
                    )
                )
                anthropic_result.update(
                    asyncio.run(
                        _skill_fidelity_smoke(
                            args.agent,
                            image,
                            anthropic_provider,
                            Path(anthropic_home),
                            protocol="anthropic-messages",
                        )
                    )
                )
                anthropic_result.update(
                    asyncio.run(
                        _model_selection_smoke(
                            args.agent,
                            image,
                            anthropic_provider,
                            Path(anthropic_home),
                            protocol="anthropic-messages",
                        )
                    )
                )
                anthropic_result.update(
                    asyncio.run(
                        _provider_error_smoke(
                            args.agent,
                            image,
                            anthropic_provider,
                            Path(anthropic_home),
                            protocol="anthropic-messages",
                        )
                    )
                )
                if set(anthropic_result["provider_paths"]) != {"/v1/messages"}:
                    raise RuntimeError(
                        f"{args.agent} Anthropic test API used unexpected paths: "
                        f"{anthropic_result['provider_paths']}"
                    )
                result["anthropic_messages"] = anthropic_result
        print(json.dumps(result, sort_keys=True))
    finally:
        if not args.keep_image:
            subprocess.run(
                ["docker", "image", "rm", "-f", image],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ACP shim for the computer-use smoke fixture.

This is a small desktop-environment adapter dogfood, not a full CUA model loop.
It speaks ACP, performs a file roundtrip in the task workspace, captures a
desktop screenshot when the sandbox exposes one, and writes a computer-use
shaped artifact for the parity harness.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchflow.environment.desktop_runtime import desktop_runtime_session

_DIAG_TRUNCATE = 2000
_DEFAULT_EXPECTED = "computer-use-smoke: ready"
_FALLBACK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def recv() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if line:
            return json.loads(line)


def _prompt_text(params: dict[str, Any]) -> str:
    parts = params.get("prompt", [])
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
    return "\n".join(text_parts)


def _expected_from_prompt(text: str) -> str:
    match = re.search(r"exactly:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"').strip("`")
    return _DEFAULT_EXPECTED


def _emit_text(session_id: str, text: str) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text[:_DIAG_TRUNCATE]},
                },
            },
        }
    )


def _emit_tool_call(
    session_id: str,
    tool_call_id: str,
    name: str,
    *,
    arguments: str = "",
) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tool_call_id,
                    "title": name,
                    "kind": "computer",
                    "status": "in_progress",
                    "input": arguments[:_DIAG_TRUNCATE],
                },
            },
        }
    )


def _emit_tool_result(
    session_id: str,
    tool_call_id: str,
    *,
    result: str,
    status: str = "completed",
) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_call_id,
                    "status": status,
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "text",
                                "text": result[:_DIAG_TRUNCATE],
                            },
                        }
                    ],
                },
            },
        }
    )


def _artifact_dir() -> Path:
    return Path(
        os.environ.get("BENCHFLOW_COMPUTER_USE_ARTIFACT_DIR", "/logs/artifacts")
    )


def _screenshot_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    home_xauthority = Path.home() / ".Xauthority"
    if home_xauthority.is_file():
        env.setdefault("XAUTHORITY", str(home_xauthority))
    return env


def _capture_screenshot(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = artifact_dir / "computer-use-smoke.png"
    command = ["gnome-screenshot", "-f", str(screenshot_path)]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_screenshot_env(),
        timeout=20,
        check=False,
    )
    if result.returncode == 0 and screenshot_path.is_file():
        data = screenshot_path.read_bytes()
        return {
            "method": "gnome-screenshot",
            "path": str(screenshot_path),
            "bytes": len(data),
            "b64": base64.b64encode(data).decode(),
            "error": None,
        }

    data = base64.b64decode(_FALLBACK_PNG_B64)
    screenshot_path.write_bytes(data)
    return {
        "method": "fallback-png",
        "path": str(screenshot_path),
        "bytes": len(data),
        "b64": _FALLBACK_PNG_B64,
        "error": (
            result.stderr or result.stdout or "screenshot command failed"
        ).strip(),
    }


def _run_computer_use_smoke(cwd: Path, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    expected = _expected_from_prompt(prompt)
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result_path = cwd / "computer_use_result.txt"
    roundtrip_path = cwd / "computer_use_roundtrip.txt"
    result_path.write_text(expected + "\n")
    roundtrip_path.write_text(result_path.read_text())
    observed = roundtrip_path.read_text().strip()
    if observed != expected:
        raise RuntimeError(
            f"roundtrip result mismatch: expected={expected!r} observed={observed!r}"
        )

    screenshot = _capture_screenshot(artifact_dir)
    steps = [
        {"action": "write_file", "path": str(result_path)},
        {"action": "read_file", "path": str(roundtrip_path), "value": observed},
        {
            "action": "screenshot",
            "method": screenshot["method"],
            "bytes": screenshot["bytes"],
        },
    ]
    session = desktop_runtime_session(
        sandbox_provider="cua",
        sandbox_provider_mode=os.environ.get("BENCHFLOW_CUA_PROVIDER_MODE"),
    )
    artifact = session.write_trace_artifact(
        artifact_dir / "computer-use-smoke-trace.json",
        framework="benchflow-computer-use-smoke-agent",
        steps=steps,
        screenshots_b64=[screenshot["b64"]],
        screenshot_method=screenshot["method"],
        screenshot_error=screenshot["error"],
        final_result=observed,
        duration_sec=round(time.perf_counter() - started, 6),
        extra={
            "files": {
                "result": str(result_path),
                "roundtrip": str(roundtrip_path),
                "screenshot": screenshot["path"],
            }
        },
    )
    return artifact


def main() -> int:
    session_id = "computer-use-smoke"
    cwd = Path("/app")

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"computer-use-smoke ACP decode error: {exc}", file=sys.stderr)
            return 1

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        "agentInfo": {
                            "name": "computer-use-smoke",
                            "version": "0.1",
                        },
                    },
                }
            )
        elif method == "session/new":
            session_id = f"computer-use-smoke-{uuid.uuid4().hex[:8]}"
            cwd = Path(str(params.get("cwd", "/app")))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )
        elif method == "session/set_model":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            prompt = _prompt_text(params)
            calls = [
                ("computer_use.write_file", {"cwd": str(cwd)}),
                ("computer_use.read_file", {"cwd": str(cwd)}),
                ("computer_use.screenshot", {"artifact_dir": str(_artifact_dir())}),
            ]
            tool_ids = []
            for name, arguments in calls:
                tool_call_id = f"computer-use-smoke-{uuid.uuid4().hex[:8]}"
                tool_ids.append(tool_call_id)
                _emit_tool_call(
                    session_id,
                    tool_call_id,
                    name,
                    arguments=json.dumps(arguments, sort_keys=True),
                )
            try:
                artifact = _run_computer_use_smoke(cwd, prompt)
            except Exception as exc:
                for tool_call_id in tool_ids:
                    _emit_tool_result(
                        session_id,
                        tool_call_id,
                        result=f"error: {type(exc).__name__}: {exc}",
                        status="failed",
                    )
                _emit_text(session_id, f"Computer Use smoke failed: {exc}")
            else:
                for index, tool_call_id in enumerate(tool_ids):
                    _emit_tool_result(
                        session_id,
                        tool_call_id,
                        result=json.dumps(artifact["steps"][index], sort_keys=True),
                    )
                _emit_text(
                    session_id,
                    f"Computer Use smoke final result: {artifact['final_result']}",
                )
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )
        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())

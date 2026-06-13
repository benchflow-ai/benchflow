#!/usr/bin/env python3
"""ACP shim for the Browser Use smoke fixture.

This is intentionally a fixture adapter, not a full Browser Use integration.
It lets BenchFlow dogfood the agent-adapter lane on a Browser Use-shaped task:
the shim speaks ACP, performs the local browser-fixture check, emits trajectory
events, and writes the artifact shape the parity harness compares against the
fixture's original runner.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchflow.environment.browser_runtime import (
    browser_runtime_session,
    expected_from_prompt,
)

_DIAG_TRUNCATE = 2000
_DEFAULT_EXPECTED = "browser-use-smoke: ready"


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
    return expected_from_prompt(text) or _DEFAULT_EXPECTED


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
    kind: str = "browser",
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
                    "kind": kind,
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
    return Path(os.environ.get("BENCHFLOW_BROWSER_USE_ARTIFACT_DIR", "/logs/artifacts"))


def _run_browser_use_smoke(cwd: Path, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    expected = _expected_from_prompt(prompt)
    fixture = cwd / "browser_fixture"
    if not fixture.is_dir():
        raise FileNotFoundError(f"missing browser fixture: {fixture}")

    with browser_runtime_session(cwd, require_ready=True) as browser_session:
        html = (fixture / "index.html").read_text()
        if expected not in html:
            raise RuntimeError(f"expected status not found in fixture: {expected!r}")

    (cwd / "final_result.txt").write_text(expected + "\n")
    artifact_dir = _artifact_dir()
    artifact = browser_session.write_trace_artifact(
        artifact_dir / "browser-use-smoke-trace.json",
        framework="benchflow-browser-use-smoke-agent",
        steps=[
            "serve local browser fixture",
            "check browser environment readiness",
            "extract final result",
            "write final result",
        ],
        final_result=expected,
        duration_sec=round(time.perf_counter() - started, 6),
    )
    return artifact


def main() -> int:
    session_id = "browser-use-smoke"
    cwd = Path("/app")
    model = ""

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"browser-use-smoke ACP decode error: {exc}", file=sys.stderr)
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
                            "name": "browser-use-smoke",
                            "version": "0.1",
                        },
                    },
                }
            )
        elif method == "session/new":
            session_id = f"browser-use-smoke-{uuid.uuid4().hex[:8]}"
            cwd = Path(str(params.get("cwd", "/app")))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )
        elif method == "session/set_model":
            model = str(params.get("modelId", ""))
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            prompt = _prompt_text(params)
            tool_call_id = f"browser-use-smoke-{uuid.uuid4().hex[:8]}"
            _emit_tool_call(
                session_id,
                tool_call_id,
                "browser_use_smoke.open_fixture",
                arguments=json.dumps(
                    {
                        "cwd": str(cwd),
                        "fixture": "browser_fixture",
                        "model": model,
                    },
                    sort_keys=True,
                ),
            )
            try:
                artifact = _run_browser_use_smoke(cwd, prompt)
            except Exception as exc:
                _emit_tool_result(
                    session_id,
                    tool_call_id,
                    result=f"error: {type(exc).__name__}: {exc}",
                    status="failed",
                )
                _emit_text(session_id, f"Browser Use smoke failed: {exc}")
            else:
                _emit_tool_result(
                    session_id,
                    tool_call_id,
                    result=json.dumps(
                        {
                            "final_result": artifact["final_result"],
                            "steps": len(artifact["steps"]),
                        },
                        sort_keys=True,
                    ),
                )
                _emit_text(
                    session_id,
                    f"Browser Use smoke final result: {artifact['final_result']}",
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

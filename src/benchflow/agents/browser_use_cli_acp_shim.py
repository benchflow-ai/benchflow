#!/usr/bin/env python3
"""ACP shim for Browser Use CLI direct-control smoke runs.

This wraps the real ``browser-use`` CLI browser harness. It is still a small
smoke adapter, not the full LLM-driven Browser Use Agent loop: on a prompt, it
opens the task's local browser fixture, reads page HTML, captures a screenshot,
writes the final result, and emits ACP tool-call updates for the commands.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
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
_TOOL_RESULT_TRUNCATE = 4000
_DEFAULT_EXPECTED = "browser-use-smoke: ready"
_DEFAULT_BROWSER_USE_BIN = "/opt/benchflow/browser-use-cli-venv/bin/browser-use"


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
    arguments: str,
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
                    "kind": "browser",
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
                                "text": result[:_TOOL_RESULT_TRUNCATE],
                            },
                        }
                    ],
                },
            },
        }
    )


def _artifact_dir() -> Path:
    return Path(os.environ.get("BENCHFLOW_BROWSER_USE_ARTIFACT_DIR", "/logs/artifacts"))


def _browser_use_home(session_name: str) -> Path:
    root = Path(
        os.environ.get(
            "BENCHFLOW_BROWSER_USE_HOME_ROOT",
            "/tmp/benchflow-browser-use-cli",
        )
    )
    return root / session_name


def _command_env(session_name: str) -> dict[str, str]:
    browser_home = _browser_use_home(session_name)
    browser_home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["BROWSER_USE_HOME"] = str(browser_home)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/benchflow/ms-playwright")
    return env


def _run_browser_use(
    session_id: str,
    session_name: str,
    *args: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    browser_use_bin = os.environ.get("BROWSER_USE_BIN", _DEFAULT_BROWSER_USE_BIN)
    tool_call_id = f"browser-use-cli-{uuid.uuid4().hex[:8]}"
    command = [browser_use_bin, "--session", session_name, *args]
    _emit_tool_call(
        session_id,
        tool_call_id,
        "browser-use " + " ".join(args[:2] or args),
        json.dumps(command, sort_keys=True),
    )
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_command_env(session_name),
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        _emit_tool_result(
            session_id,
            tool_call_id,
            result=f"{type(exc).__name__}: {exc}",
            status="failed",
        )
        raise
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        _emit_tool_result(session_id, tool_call_id, result=output, status="failed")
        raise RuntimeError(
            f"browser-use {' '.join(args)} failed with rc {result.returncode}: {output}"
        )
    _emit_tool_result(session_id, tool_call_id, result=output)
    return result


def _run_browser_use_cli(cwd: Path, prompt: str, session_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    expected = _expected_from_prompt(prompt)

    session_name = f"benchflow-{uuid.uuid4().hex[:8]}"
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = artifact_dir / "browser-use-cli-screenshot.png"
    steps: list[dict[str, Any]] = []

    try:
        with browser_runtime_session(cwd, require_ready=True) as browser_session:
            browser_env = browser_session.handle
            if browser_env.url is None:
                raise FileNotFoundError(
                    f"missing browser fixture: {cwd / 'browser_fixture'}"
                )

            open_result = _run_browser_use(
                session_id,
                session_name,
                "open",
                browser_env.url,
                timeout=90,
            )
            steps.append(
                {
                    "action": "open",
                    "environment": browser_session.environment,
                    "output": (open_result.stdout or "").strip(),
                }
            )

            html_result = _run_browser_use(
                session_id,
                session_name,
                "get",
                "html",
                timeout=60,
            )
            html = html_result.stdout or ""
            steps.append({"action": "get_html", "contains_expected": expected in html})
            if expected not in html:
                raise RuntimeError(
                    "expected status not found in Browser Use HTML output"
                )

            screenshot_result = _run_browser_use(
                session_id,
                session_name,
                "screenshot",
                str(screenshot_path),
                timeout=60,
            )
            steps.append(
                {
                    "action": "screenshot",
                    "output": (screenshot_result.stdout or "").strip(),
                }
            )
    finally:
        try:
            close_result = _run_browser_use(
                session_id,
                session_name,
                "close",
                timeout=30,
            )
            steps.append(
                {"action": "close", "output": (close_result.stdout or "").strip()}
            )
        except Exception as exc:
            steps.append({"action": "close", "error": str(exc)})

    screenshot_b64 = ""
    if screenshot_path.is_file():
        screenshot_b64 = base64.b64encode(screenshot_path.read_bytes()).decode()

    (cwd / "final_result.txt").write_text(expected + "\n")
    artifact = browser_session.write_trace_artifact(
        artifact_dir / "browser-use-smoke-trace.json",
        framework="benchflow-browser-use-cli-agent",
        steps=steps,
        screenshots_b64=[screenshot_b64] if screenshot_b64 else [],
        final_result=expected,
        duration_sec=round(time.perf_counter() - started, 6),
    )
    return artifact


def main() -> int:
    session_id = "browser-use-cli"
    cwd = Path("/app")
    model = ""

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"browser-use-cli ACP decode error: {exc}", file=sys.stderr)
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
                        "agentInfo": {"name": "browser-use-cli", "version": "0.1"},
                    },
                }
            )
        elif method == "session/new":
            session_id = f"browser-use-cli-{uuid.uuid4().hex[:8]}"
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
            try:
                artifact = _run_browser_use_cli(cwd, prompt, session_id)
            except Exception as exc:
                _emit_text(session_id, f"Browser Use CLI smoke failed: {exc}")
            else:
                _emit_text(
                    session_id,
                    "Browser Use CLI final result: "
                    f"{artifact['final_result']} ({len(artifact['steps'])} steps, "
                    f"model={model or 'unset'})",
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

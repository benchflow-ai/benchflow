#!/usr/bin/env python3
"""ACP shim for the LLM-driven Browser Use Agent loop."""

from __future__ import annotations

import asyncio
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
_TOOL_RESULT_TRUNCATE = 4000
_DEFAULT_MODEL = "gemini-2.5-flash"


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


def _normalize_model(model: str) -> str:
    if not model:
        return _DEFAULT_MODEL
    for prefix in ("google/", "gemini/"):
        if model.startswith(prefix):
            return model.split("/", 1)[1]
    return model


def _build_llm(model: str):
    from browser_use.llm.google import ChatGoogle

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    return ChatGoogle(model=_normalize_model(model), api_key=api_key, temperature=0)


def _history_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _method_or_default(obj: Any, name: str, default: Any) -> Any:
    method = getattr(obj, name, None)
    if not callable(method):
        return default
    try:
        return method()
    except Exception:
        return default


async def _run_browser_use_agent_async(
    *,
    cwd: Path,
    prompt: str,
    session_id: str,
    model: str,
) -> dict[str, Any]:
    from browser_use import Agent, BrowserProfile

    expected = expected_from_prompt(prompt)
    started = time.perf_counter()
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with browser_runtime_session(cwd) as browser_session:
        browser_env = browser_session.handle
        if browser_env.url is not None and not browser_session.readiness.ok:
            raise RuntimeError(
                f"browser environment not ready: {browser_session.readiness.to_dict()}"
            )
        task = browser_session.task_instruction(
            prompt=prompt,
            expected=expected,
        )
        tool_call_id = f"browser-use-agent-{uuid.uuid4().hex[:8]}"
        _emit_tool_call(
            session_id,
            tool_call_id,
            "browser_use.Agent.run",
            json.dumps(
                {
                    "model": _normalize_model(model),
                    "environment": browser_session.environment,
                    "max_steps": 6,
                },
                sort_keys=True,
            ),
        )
        try:
            profile = BrowserProfile(
                headless=True,
                chromium_sandbox=False,
                user_data_dir=str(
                    Path("/tmp") / f"browser-use-agent-{uuid.uuid4().hex[:8]}"
                ),
                allowed_domains=browser_env.allowed_domains,
            )
            agent = Agent(
                task=task,
                llm=_build_llm(model),
                browser_profile=profile,
                use_vision=False,
                use_judge=False,
                max_failures=2,
                step_timeout=90,
                llm_timeout=90,
                enable_signal_handler=False,
            )
            history = await agent.run(max_steps=6)
        except Exception as exc:
            _emit_tool_result(
                session_id,
                tool_call_id,
                result=f"{type(exc).__name__}: {exc}",
                status="failed",
            )
            raise

    final_result = str(_method_or_default(history, "final_result", "") or "")
    actions = _history_list(_method_or_default(history, "action_names", []))
    screenshots = [
        screenshot
        for screenshot in _history_list(_method_or_default(history, "screenshots", []))
        if screenshot
    ]
    errors = [
        error
        for error in _history_list(_method_or_default(history, "errors", []))
        if error
    ]
    steps_count = int(_method_or_default(history, "number_of_steps", len(actions)) or 0)
    successful = _method_or_default(history, "is_successful", None)

    if expected is not None and expected not in final_result:
        raise RuntimeError(
            f"Browser Use Agent final result did not contain expected value: {final_result!r}"
        )

    final_result_for_artifact = expected or final_result
    (cwd / "final_result.txt").write_text(final_result_for_artifact + "\n")
    artifact = browser_session.write_trace_artifact(
        artifact_dir / "browser-use-smoke-trace.json",
        framework="benchflow-browser-use-agent",
        steps=[{"action": action} for action in actions],
        screenshots_b64=screenshots,
        final_result=final_result_for_artifact,
        duration_sec=round(time.perf_counter() - started, 6),
        extra={
            "history_final_result": final_result,
            "history_steps": steps_count,
            "successful": successful,
            "errors": errors,
        },
    )
    _emit_tool_result(
        session_id,
        tool_call_id,
        result=json.dumps(
            {
                "final_result": final_result,
                "steps": steps_count,
                "actions": actions,
                "screenshots": len(screenshots),
            },
            sort_keys=True,
        ),
    )
    return artifact


def _run_browser_use_agent(
    *,
    cwd: Path,
    prompt: str,
    session_id: str,
    model: str,
) -> dict[str, Any]:
    return asyncio.run(
        _run_browser_use_agent_async(
            cwd=cwd,
            prompt=prompt,
            session_id=session_id,
            model=model,
        )
    )


def main() -> int:
    session_id = "browser-use-agent"
    cwd = Path("/app")
    model = _DEFAULT_MODEL

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"browser-use-agent ACP decode error: {exc}", file=sys.stderr)
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
                        "agentInfo": {"name": "browser-use-agent", "version": "0.1"},
                    },
                }
            )
        elif method == "session/new":
            session_id = f"browser-use-agent-{uuid.uuid4().hex[:8]}"
            cwd = Path(str(params.get("cwd", "/app")))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )
        elif method == "session/set_model":
            model = str(params.get("modelId", "")) or _DEFAULT_MODEL
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            prompt = _prompt_text(params)
            try:
                artifact = _run_browser_use_agent(
                    cwd=cwd,
                    prompt=prompt,
                    session_id=session_id,
                    model=model,
                )
            except Exception as exc:
                _emit_text(session_id, f"Browser Use Agent failed: {exc}")
            else:
                _emit_text(
                    session_id,
                    "Browser Use Agent final result: "
                    f"{artifact['final_result']} ({artifact['history_steps']} steps)",
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

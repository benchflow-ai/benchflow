#!/usr/bin/env python3
"""ACP shim for a real CUA (computer-using agent) model loop.

Unlike ``computer_use_smoke_acp_shim`` (a scripted fixture), this is a genuine
perception->action loop: it screenshots the desktop, asks a vision model
(Gemini 3.5 Flash by default) for the next action, executes that action inside
the sandbox with ``xdotool``, and repeats until the model reports ``done`` or
the step budget is exhausted.

The sandbox provider is intentionally untouched: perception (gnome-screenshot)
and control (xdotool) both run in-sandbox, exactly where this shim already runs.
The agent is the decision plane; the sandbox is only "where the world runs".
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from benchflow.environment.desktop_runtime import desktop_runtime_session

_DIAG_TRUNCATE = 2000
_TOOL_RESULT_TRUNCATE = 4000
_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_MAX_STEPS = 12
_FALLBACK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

_SYSTEM_PROMPT = (
    "You are a computer-using agent controlling a Linux desktop. Each turn you "
    "receive a screenshot. Decide the single next action and reply with ONE "
    "JSON object and nothing else. Allowed actions:\n"
    '  {"action":"click","x":INT,"y":INT}\n'
    '  {"action":"double_click","x":INT,"y":INT}\n'
    '  {"action":"right_click","x":INT,"y":INT}\n'
    '  {"action":"type","text":STR}\n'
    '  {"action":"key","keys":STR}            (xdotool key spec, e.g. "ctrl+s")\n'
    '  {"action":"scroll","dy":INT}           (negative = up, positive = down)\n'
    '  {"action":"wait","ms":INT}\n'
    '  {"action":"done","result":STR}         (STR = your final answer/result)\n'
    "Coordinates are pixels from the top-left of the screen. When the task is "
    "complete, emit a done action whose result states the outcome."
)


# --------------------------------------------------------------------------- #
# ACP plumbing (mirrors the other BenchFlow ACP shims).
# --------------------------------------------------------------------------- #
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
                                "text": result[:_TOOL_RESULT_TRUNCATE],
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


def _max_steps() -> int:
    raw = os.environ.get("BENCHFLOW_CUA_AGENT_MAX_STEPS")
    if not raw:
        return _DEFAULT_MAX_STEPS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_STEPS
    return max(1, value)


def _normalize_model(model: str) -> str:
    if not model:
        return _DEFAULT_MODEL
    for prefix in ("google/", "gemini/"):
        if model.startswith(prefix):
            return model.split("/", 1)[1]
    return model


def _expected_from_prompt(text: str) -> str | None:
    import re

    match = re.search(r"exactly:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"').strip("`")
    return None


# --------------------------------------------------------------------------- #
# Perception: screenshot the desktop (shared shape with the smoke shim).
# --------------------------------------------------------------------------- #
def _desktop_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    home_xauthority = Path.home() / ".Xauthority"
    if home_xauthority.is_file():
        env.setdefault("XAUTHORITY", str(home_xauthority))
    return env


def _capture_screenshot(artifact_dir: Path, index: int) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = artifact_dir / f"computer-use-agent-{index:02d}.png"
    result = subprocess.run(
        ["gnome-screenshot", "-f", str(screenshot_path)],
        text=True,
        capture_output=True,
        env=_desktop_env(),
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


# --------------------------------------------------------------------------- #
# Action: execute one model action inside the sandbox with xdotool.
# --------------------------------------------------------------------------- #
def _xdotool(args: list[str]) -> str:
    result = subprocess.run(
        ["xdotool", *args],
        text=True,
        capture_output=True,
        env=_desktop_env(),
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xdotool {shlex.join(args)} failed (rc={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _coords(action: dict[str, Any]) -> list[str]:
    x = int(action["x"])
    y = int(action["y"])
    return ["mousemove", "--sync", str(x), str(y)]


def _execute_action(action: dict[str, Any]) -> str:
    kind = str(action.get("action", "")).strip().lower()
    if kind == "click":
        _xdotool([*_coords(action), "click", "1"])
        return f"click {action['x']},{action['y']}"
    if kind == "double_click":
        _xdotool([*_coords(action), "click", "--repeat", "2", "1"])
        return f"double_click {action['x']},{action['y']}"
    if kind == "right_click":
        _xdotool([*_coords(action), "click", "3"])
        return f"right_click {action['x']},{action['y']}"
    if kind == "type":
        text = str(action.get("text", ""))
        _xdotool(["type", "--clearmodifiers", "--", text])
        return f"type {text!r}"
    if kind == "key":
        keys = str(action.get("keys", "")).strip()
        if not keys:
            raise ValueError("key action requires non-empty 'keys'")
        _xdotool(["key", "--clearmodifiers", keys])
        return f"key {keys}"
    if kind == "scroll":
        dy = int(action.get("dy", 0))
        button = "4" if dy < 0 else "5"
        for _ in range(max(1, abs(dy))):
            _xdotool(["click", button])
        return f"scroll {dy}"
    if kind == "wait":
        ms = max(0, int(action.get("ms", 0)))
        time.sleep(min(ms, 5000) / 1000.0)
        return f"wait {ms}ms"
    raise ValueError(f"unsupported action: {kind!r}")


# --------------------------------------------------------------------------- #
# Policy: the model that chooses the next action from a screenshot.
# --------------------------------------------------------------------------- #
class Policy(Protocol):
    def decide(
        self, *, prompt: str, screenshot_b64: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


def _parse_action(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"model response was not a JSON action: {raw!r}")
    action = json.loads(text[start : end + 1])
    if not isinstance(action, dict) or "action" not in action:
        raise ValueError(f"model action missing 'action' key: {action!r}")
    return action


class _ScriptedPolicy:
    """Test policy that replays a fixed list of actions from a JSON file.

    Enabled by ``BENCHFLOW_CUA_AGENT_FAKE_ACTIONS=<path>`` so the loop can be
    exercised hermetically (no API key, no network). Not used in production.
    """

    def __init__(self, path: Path) -> None:
        self._actions = json.loads(path.read_text())
        if not isinstance(self._actions, list):
            raise ValueError("fake-actions file must contain a JSON list")
        self._index = 0

    def decide(
        self, *, prompt: str, screenshot_b64: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self._index >= len(self._actions):
            return {"action": "done", "result": "scripted actions exhausted"}
        action = self._actions[self._index]
        self._index += 1
        return action


class _GeminiPolicy:
    """Real policy: Gemini vision model picks the next action per screenshot."""

    def __init__(self, model: str) -> None:
        self._model = _normalize_model(model)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        from google import genai  # lazy: only needed for live runs

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def decide(
        self, *, prompt: str, screenshot_b64: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        from google.genai import types

        history_note = ""
        if history:
            recent = [step.get("result", step.get("action")) for step in history[-6:]]
            history_note = "\nActions so far: " + json.dumps(recent)
        content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"{_SYSTEM_PROMPT}\n\nTask: {prompt}{history_note}"
                ),
                types.Part.from_bytes(
                    data=base64.b64decode(screenshot_b64),
                    mime_type="image/png",
                ),
            ],
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=content,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return _parse_action(response.text or "")


def _load_policy(model: str) -> Policy:
    fake = os.environ.get("BENCHFLOW_CUA_AGENT_FAKE_ACTIONS")
    if fake:
        return _ScriptedPolicy(Path(fake))
    return _GeminiPolicy(model)


# --------------------------------------------------------------------------- #
# The loop.
# --------------------------------------------------------------------------- #
def _run_computer_use_agent(
    *,
    cwd: Path,
    prompt: str,
    session_id: str,
    model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected = _expected_from_prompt(prompt)
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    policy = _load_policy(model)
    max_steps = _max_steps()

    steps: list[dict[str, Any]] = []
    screenshots_b64: list[str] = []
    history: list[dict[str, Any]] = []
    final_result = ""
    screenshot_method = "gnome-screenshot"
    screenshot_error: str | None = None
    done = False

    for index in range(max_steps):
        shot = _capture_screenshot(artifact_dir, index)
        screenshots_b64.append(shot["b64"])
        screenshot_method = shot["method"]
        if shot["error"]:
            screenshot_error = shot["error"]

        tool_call_id = f"computer-use-agent-{uuid.uuid4().hex[:8]}"
        _emit_tool_call(
            session_id,
            tool_call_id,
            "computer_use.decide",
            json.dumps(
                {"model": _normalize_model(model), "step": index}, sort_keys=True
            ),
        )
        try:
            action = policy.decide(
                prompt=prompt, screenshot_b64=shot["b64"], history=history
            )
        except Exception as exc:
            _emit_tool_result(
                session_id,
                tool_call_id,
                result=f"decide failed: {type(exc).__name__}: {exc}",
                status="failed",
            )
            raise

        kind = str(action.get("action", "")).strip().lower()
        if kind == "done":
            final_result = str(action.get("result", ""))
            step = {
                "action": "done",
                "result": final_result,
                "screenshot": shot["path"],
            }
            steps.append(step)
            history.append(step)
            _emit_tool_result(
                session_id, tool_call_id, result=json.dumps(step, sort_keys=True)
            )
            done = True
            break

        try:
            outcome = _execute_action(action)
        except Exception as exc:
            _emit_tool_result(
                session_id,
                tool_call_id,
                result=f"action failed: {type(exc).__name__}: {exc}",
                status="failed",
            )
            raise
        step = {"action": kind, "result": outcome, "screenshot": shot["path"]}
        steps.append(step)
        history.append(step)
        _emit_tool_result(
            session_id, tool_call_id, result=json.dumps(step, sort_keys=True)
        )

    if not done:
        raise RuntimeError(
            f"computer-use-agent did not finish within {max_steps} steps "
            f"({len(steps)} actions taken)"
        )
    if expected is not None and expected not in final_result:
        raise RuntimeError(
            f"computer-use-agent final result did not contain expected value: "
            f"expected={expected!r} final_result={final_result!r}"
        )

    (cwd / "computer_use_result.txt").write_text((expected or final_result) + "\n")
    session = desktop_runtime_session(
        sandbox_provider="cua",
        sandbox_provider_mode=os.environ.get("BENCHFLOW_CUA_PROVIDER_MODE"),
    )
    artifact = session.write_trace_artifact(
        artifact_dir / "computer-use-agent-trace.json",
        framework="benchflow-computer-use-agent",
        steps=steps,
        screenshots_b64=screenshots_b64,
        screenshot_method=screenshot_method,
        screenshot_error=screenshot_error,
        final_result=expected or final_result,
        duration_sec=round(time.perf_counter() - started, 6),
        extra={
            "model": _normalize_model(model),
            "history_final_result": final_result,
            "history_steps": len(steps),
        },
    )
    return artifact


def main() -> int:
    session_id = "computer-use-agent"
    cwd = Path("/app")
    model = _DEFAULT_MODEL

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"computer-use-agent ACP decode error: {exc}", file=sys.stderr)
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
                        "agentInfo": {"name": "computer-use-agent", "version": "0.1"},
                    },
                }
            )
        elif method == "session/new":
            session_id = f"computer-use-agent-{uuid.uuid4().hex[:8]}"
            cwd = Path(str(params.get("cwd", "/app")))
            send({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id}})
        elif method == "session/set_model":
            model = str(params.get("modelId", "")) or _DEFAULT_MODEL
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            prompt = _prompt_text(params)
            try:
                artifact = _run_computer_use_agent(
                    cwd=cwd, prompt=prompt, session_id=session_id, model=model
                )
            except Exception as exc:
                _emit_text(session_id, f"Computer Use Agent failed: {exc}")
            else:
                _emit_text(
                    session_id,
                    "Computer Use Agent final result: "
                    f"{artifact['final_result']} ({artifact['history_steps']} steps)",
                )
            send({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})
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

#!/usr/bin/env python3
"""ACP-native Claude agent — speaks ACP on stdio, uses Anthropic API for reasoning.

This agent:
1. Accepts ACP messages from benchflow on stdin (initialize, session/new, session/prompt)
2. Calls Claude API with tool_use for reasoning
3. Proxies tool calls back through ACP (fs/read, fs/write, terminal/create)
4. benchflow routes those to the Docker environment

Usage:
    benchflow run -a "python -m benchflow.agents.acp_claude" --agent-transport stdio

Or standalone:
    python -m benchflow.agents.acp_claude
"""

import json
import os
import sys
import urllib.request
from typing import Any

# --- Config ---
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
MODEL = os.environ.get("BENCHFLOW_MODEL", "claude-sonnet-4-20250514")
MAX_TURNS = int(os.environ.get("BENCHFLOW_MAX_TURNS", "30"))
MAX_TOKENS = int(os.environ.get("BENCHFLOW_MAX_TOKENS", "16384"))

# --- JSON-RPC helpers ---
_request_id = 0


def _next_id() -> int:
    global _request_id
    _request_id += 1
    return _request_id


def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def send_response(req_id: Any, result: dict) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def send_error(req_id: Any, code: int, message: str) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def send_notification(method: str, params: dict) -> None:
    send({"jsonrpc": "2.0", "method": method, "params": params})


def send_update(session_id: str, update: dict) -> None:
    send_notification("session/update", {"sessionId": session_id, "update": update})


def read_message() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def send_request_and_wait(method: str, params: dict) -> dict:
    """Send a JSON-RPC request to the client (benchflow) and wait for response."""
    req_id = _next_id()
    send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
    # Read messages until we get our response
    while True:
        msg = read_message()
        if msg is None:
            raise ConnectionError("Client disconnected")
        if msg.get("id") == req_id:
            if "error" in msg:
                raise RuntimeError(f"Client error: {msg['error']}")
            return msg.get("result", {})
        # Ignore notifications/other messages


# --- ACP tool implementations (proxy to client/benchflow) ---


def acp_bash(session_id: str, command: str, cwd: str | None = None) -> str:
    """Execute a command via ACP terminal/create."""
    result = send_request_and_wait(
        "terminal/create",
        {
            "sessionId": session_id,
            "command": "bash",
            "args": ["-c", command],
            "cwd": cwd,
        },
    )
    output = result.get("output", "")
    exit_code = result.get("exitStatus", {}).get("exitCode", -1)
    return (
        f"{output}\n[exit code: {exit_code}]" if output else f"[exit code: {exit_code}]"
    )


def acp_read_file(session_id: str, path: str) -> str:
    """Read a file via ACP fs/read_text_file."""
    result = send_request_and_wait(
        "fs/read_text_file",
        {
            "sessionId": session_id,
            "path": path,
        },
    )
    return result.get("contents", "")


def acp_write_file(session_id: str, path: str, contents: str) -> str:
    """Write a file via ACP fs/write_text_file."""
    send_request_and_wait(
        "fs/write_text_file",
        {
            "sessionId": session_id,
            "path": path,
            "contents": contents,
        },
    )
    return f"Wrote {len(contents)} chars to {path}"


# --- Claude API ---

TOOLS = [
    {
        "name": "bash",
        "description": "Execute a bash command in the environment. Returns stdout/stderr and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the environment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the environment. Creates directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
]


def call_claude(messages: list[dict]) -> dict:
    data = json.dumps(
        {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": "You are a skilled software engineer solving a coding task inside a Linux environment.\nYour working directory is /app. All file paths MUST be absolute paths under /app/ (e.g. /app/hello.txt, not /hello.txt).\nUse the bash, read_file, and write_file tools to explore and solve the task. Verify your solution before finishing.",
            "tools": TOOLS,
            "messages": messages,
        }
    ).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())


# --- Main ACP server loop ---


def handle_prompt(session_id: str, prompt_text: str, req_id: Any) -> None:
    """Handle a session/prompt request — run the multi-turn agent loop."""
    messages = [{"role": "user", "content": prompt_text}]

    for _turn in range(MAX_TURNS):
        response = call_claude(messages)
        content = response.get("content", [])
        stop_reason = response.get("stop_reason", "")

        # Send text chunks as session updates
        for block in content:
            if block.get("type") == "text" and block.get("text"):
                send_update(
                    session_id,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": block["text"]},
                    },
                )

        messages.append({"role": "assistant", "content": content})

        if stop_reason == "end_turn":
            send_response(req_id, {"stopReason": "end_turn"})
            return

        if stop_reason == "tool_use":
            tool_results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name = block["name"]
                tool_input = block.get("input", {})
                tool_id = block["id"]

                # Send tool_call update
                send_update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": tool_id,
                        "title": f"{tool_name}: {json.dumps(tool_input)[:80]}",
                        "kind": "bash"
                        if tool_name == "bash"
                        else "write"
                        if tool_name == "write_file"
                        else "read",
                        "status": "pending",
                    },
                )

                # Execute via ACP
                try:
                    if tool_name == "bash":
                        result = acp_bash(session_id, tool_input.get("command", ""))
                    elif tool_name == "read_file":
                        result = acp_read_file(session_id, tool_input.get("path", ""))
                    elif tool_name == "write_file":
                        result = acp_write_file(
                            session_id,
                            tool_input.get("path", ""),
                            tool_input.get("content", ""),
                        )
                    else:
                        result = f"Unknown tool: {tool_name}"
                except Exception as e:
                    result = f"Error: {e}"

                # Send tool_call_update
                send_update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_id,
                        "status": "completed",
                        "content": [
                            {
                                "type": "content",
                                "content": {"type": "text", "text": result[:3000]},
                            }
                        ],
                    },
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result[:5000],
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            send_response(req_id, {"stopReason": stop_reason or "end_turn"})
            return

    # Hit max turns
    send_response(req_id, {"stopReason": "max_turn_requests"})


def main():
    session_id = ""

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send_response(
                req_id,
                {
                    "protocolVersion": 0,
                    "agentInfo": {"name": "claude-acp", "version": "1.0.0"},
                    "agentCapabilities": {
                        "promptCapabilities": {"image": False, "audio": False},
                    },
                },
            )

        elif method == "session/new":
            session_id = f"session-{_next_id()}"
            send_response(req_id, {"sessionId": session_id})

        elif method == "session/prompt":
            session_id = params.get("sessionId", session_id)
            prompt_blocks = params.get("prompt", [])
            prompt_text = " ".join(
                b.get("text", "") for b in prompt_blocks if b.get("type") == "text"
            )
            handle_prompt(session_id, prompt_text, req_id)

        elif method == "session/cancel":
            pass  # Best effort — can't interrupt Claude API call

        else:
            send_error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()

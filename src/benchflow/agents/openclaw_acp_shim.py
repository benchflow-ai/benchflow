#!/usr/bin/env python3
"""ACP shim for OpenClaw — wraps `openclaw agent --local` as an ACP server.

openclaw's native ACP bridge requires a gateway with chat-thread sessions.
This shim speaks ACP on stdio and internally calls `openclaw agent --local`
for each prompt, then parses openclaw's session JSONL to emit proper ACP
tool_call and text updates.

Architecture:
  benchflow ACP client ←stdio→ this shim ←subprocess→ openclaw agent --local
                                          ←file read→  ~/.openclaw/agents/main/sessions/*.jsonl

Key details:
  - Workspace: symlinks ~/.openclaw/workspace → task cwd (openclaw ignores subprocess cwd)
  - Skills: if task env has ~/.claude/skills/, also copies to ~/.openclaw/workspace/.claude/skills/
  - Trajectory: parses session JSONL for tool calls, thinking, text → emits ACP session/update
  - Model: set via openclaw config on session/set_model
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def recv():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if not line:
            continue
        return json.loads(line)


def setup_workspace(cwd: str):
    """Point openclaw's workspace at the task directory."""
    home = os.environ.get("HOME", os.path.expanduser("~"))
    oc_workspace = Path(home) / ".openclaw" / "workspace"

    if oc_workspace.is_symlink() or oc_workspace.exists():
        if oc_workspace.is_symlink():
            oc_workspace.unlink()
        elif oc_workspace.is_dir():
            shutil.rmtree(oc_workspace)

    oc_workspace.parent.mkdir(parents=True, exist_ok=True)
    oc_workspace.symlink_to(cwd)


def find_session_jsonl() -> Path | None:
    """Find the most recent openclaw session JSONL file."""
    home = os.environ.get("HOME", os.path.expanduser("~"))
    sessions_dir = Path(home) / ".openclaw" / "agents" / "main" / "sessions"
    if not sessions_dir.exists():
        return None

    jsonl_files = sorted(
        sessions_dir.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    # Skip .lock files
    for f in jsonl_files:
        if not f.name.endswith(".lock"):
            return f
    return None


def parse_session_jsonl(path: Path, session_id: str) -> list[dict]:
    """Parse openclaw session JSONL and convert to ACP session/update events."""
    updates = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = entry.get("role", "")
                content = entry.get("content", "")

                if role == "assistant":
                    # Parse assistant content blocks
                    if isinstance(content, list):
                        for block in content:
                            block_type = block.get("type", "")

                            if block_type == "text":
                                updates.append({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "text_update",
                                            "text": block.get("text", ""),
                                        },
                                    },
                                })

                            elif block_type == "tool_use":
                                updates.append({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "type": "tool_call",
                                            "tool_call_id": block.get("id", ""),
                                            "kind": "other",
                                            "title": block.get("name", "tool"),
                                            "status": "completed",
                                            "content": [
                                                {
                                                    "type": "content",
                                                    "content": {
                                                        "type": "text",
                                                        "text": json.dumps(
                                                            block.get("input", {})
                                                        )[:500],
                                                    },
                                                }
                                            ],
                                        },
                                    },
                                })

                            elif block_type == "thinking":
                                updates.append({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "type": "agent_thought",
                                            "text": block.get("thinking", ""),
                                        },
                                    },
                                })

                    elif isinstance(content, str) and content:
                        updates.append({
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "text_update",
                                    "text": content,
                                },
                            },
                        })

                elif role == "tool":
                    # Tool result
                    tool_id = entry.get("tool_use_id", "")
                    result_content = content
                    if isinstance(content, list):
                        result_content = " ".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    updates.append({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {
                                "type": "tool_result",
                                "tool_call_id": tool_id,
                                "content": str(result_content)[:1000],
                            },
                        },
                    })

    except Exception:
        pass

    return updates


def main():
    session_id = "openclaw-shim"
    cwd = "/app"

    while True:
        try:
            msg = recv()
        except EOFError:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": False,
                        "promptCapabilities": {"image": False, "audio": False},
                    },
                    "agentInfo": {"name": "openclaw", "version": "1.0"},
                },
            })

        elif method == "session/new":
            cwd = params.get("cwd", "/app")
            setup_workspace(cwd)
            session_id = "openclaw-shim"
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"sessionId": session_id},
            })

        elif method == "session/set_model":
            model = params.get("modelId", "")
            if model:
                subprocess.run(
                    ["openclaw", "config", "set", "agents.defaults.model",
                     f"anthropic/{model}"],
                    capture_output=True, timeout=10,
                )
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            prompt_parts = params.get("prompt", [])
            text = ""
            for part in prompt_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")

            try:
                result = subprocess.run(
                    [
                        "openclaw", "agent", "--local", "--agent", "main",
                        "--json", "-m", text, "--timeout", "900",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=920,
                    env={**os.environ},
                )

                # Parse openclaw's session JSONL for full trajectory
                session_jsonl = find_session_jsonl()
                if session_jsonl:
                    updates = parse_session_jsonl(session_jsonl, session_id)
                    for update in updates:
                        send(update)

                # If no JSONL trajectory, fall back to text response
                if not session_jsonl:
                    try:
                        response = json.loads(result.stdout)
                        agent_text = response.get("payloads", [{}])[0].get("text", "")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        agent_text = result.stdout[:2000] if result.stdout else ""

                    if agent_text:
                        send({
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "text_update",
                                    "text": agent_text,
                                },
                            },
                        })

                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                })

            except subprocess.TimeoutExpired:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                })
            except Exception as e:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                })

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0].get("optionId", "default") if options else "default"
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"outcome": {"outcome": "selected", "optionId": option_id}},
            })

        else:
            if req_id:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()

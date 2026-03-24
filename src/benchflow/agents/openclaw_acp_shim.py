#!/usr/bin/env python3
"""ACP shim for OpenClaw — wraps `openclaw agent --local` as an ACP server.

openclaw's native ACP bridge requires a gateway with chat-thread sessions.
This shim instead speaks ACP on stdio and internally calls `openclaw agent --local`
for each prompt, making openclaw work as a standalone ACP agent.
"""

import json
import subprocess
import sys
import os


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


def main():
    session_id = None
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
            session_id = "openclaw-shim"
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"sessionId": session_id},
            })

        elif method == "session/set_model":
            # Model is set via ANTHROPIC_MODEL env var which openclaw reads
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            prompt_parts = params.get("prompt", [])
            text = ""
            for part in prompt_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")

            # Call openclaw agent --local
            try:
                result = subprocess.run(
                    [
                        "openclaw", "agent", "--local", "--agent", "main",
                        "--json", "-m", text, "--timeout", "900",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=920,
                    cwd=cwd,
                    env={**os.environ},
                )

                # Parse response for tool calls
                try:
                    response = json.loads(result.stdout)
                    agent_text = response.get("payloads", [{}])[0].get("text", "")
                except (json.JSONDecodeError, IndexError, KeyError):
                    agent_text = result.stdout[:1000] if result.stdout else "No response"

                # Emit text update
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

                # Send prompt response
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
            # Auto-approve
            options = params.get("options", [])
            option_id = options[0].get("optionId", "default") if options else "default"
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"outcome": {"outcome": "selected", "optionId": option_id}},
            })

        else:
            # Unknown method
            if req_id:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()

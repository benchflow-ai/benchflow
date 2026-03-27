#!/usr/bin/env python3
"""Minimal ACP agent over stdio — responds to initialize, session/new, session/prompt."""

import json
import sys


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
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

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 0,
                        "agentInfo": {"name": "mock-agent", "version": "1.0.0"},
                        "agentCapabilities": {},
                    },
                }
            )

        elif method == "session/new":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": "mock-session-1"},
                }
            )

        elif method == "session/prompt":
            session_id = msg.get("params", {}).get("sessionId", "")
            prompt = msg.get("params", {}).get("prompt", [])
            prompt_text = prompt[0].get("text", "") if prompt else ""

            # Send a tool call notification
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc_1",
                            "title": "echo hello",
                            "kind": "bash",
                            "status": "pending",
                        },
                    },
                }
            )
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "tc_1",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "content",
                                    "content": {"type": "text", "text": "hello"},
                                }
                            ],
                        },
                    },
                }
            )

            # Send agent message
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {
                                "type": "text",
                                "text": f"I received: {prompt_text}",
                            },
                        },
                    },
                }
            )

            # Send prompt response
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )

        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }
            )


if __name__ == "__main__":
    main()


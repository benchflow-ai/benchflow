#!/usr/bin/env python3
"""ACP agent that sends interleaved notifications + agent request before response."""

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
                        "agentInfo": {"name": "interleaved-agent", "version": "1.0.0"},
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

            # 1. Send a notification (tool_call)
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc_1",
                            "title": "ls /",
                            "kind": "bash",
                            "status": "pending",
                        },
                    },
                }
            )

            # 2. Send an agent request (request_permission) — has id + method
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 9000,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": session_id,
                        "options": [
                            {"optionId": "allow_once", "kind": "allow_once"},
                        ],
                    },
                }
            )

            # Agent reads the permission response (we just consume it)
            # The client will send back a response to id=9000
            sys.stdin.readline()
            # (ignore the response content — we just needed the round trip)

            # 3. Send tool_call_update notification
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
                        },
                    },
                }
            )

            # 4. Send agent message notification
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "done"},
                        },
                    },
                }
            )

            # 5. Finally, send the actual response for the prompt
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

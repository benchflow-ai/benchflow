#!/usr/bin/env python3
"""ACP agent that handles multiple prompts with thought→tool→message per prompt.

Each prompt gets:
  1. agent_thought_chunk (thinking about the prompt)
  2. tool_call + tool_call_update (running a command)
  3. agent_message_chunk (response text)

The response text includes a turn counter so tests can verify
per-prompt boundaries and non-cumulative text.
"""

import json
import sys

turn_counter = 0


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    global turn_counter

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
            requested = msg.get("params", {}).get("protocolVersion", 1)
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": min(requested, 1),
                        "agentInfo": {
                            "name": "multi-turn-agent",
                            "version": "1.0.0",
                        },
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
            turn_counter += 1
            session_id = msg.get("params", {}).get("sessionId", "")

            # 1. Thought
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {
                                "type": "text",
                                "text": f"thinking-turn-{turn_counter}",
                            },
                        },
                    },
                }
            )

            # 2. Tool call
            tc_id = f"tc_{turn_counter}"
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": tc_id,
                            "title": f"cmd-{turn_counter}",
                            "kind": "bash",
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
                            "toolCallId": tc_id,
                            "status": "completed",
                        },
                    },
                }
            )

            # 3. Agent message
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
                                "text": f"response-turn-{turn_counter}",
                            },
                        },
                    },
                }
            )

            # Response
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )

        elif method == "session/set_model":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}",
                    },
                }
            )


if __name__ == "__main__":
    main()

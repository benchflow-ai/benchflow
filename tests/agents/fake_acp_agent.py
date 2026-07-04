#!/usr/bin/env python3
"""Scripted fake ACP agent: minimal JSON-RPC over stdio for run/resume tests.

Env knobs:
  FAKE_ACP_LOG      — append every received request {method, params} as a JSON line
  FAKE_LOADSESSION  — "0" advertises loadSession:false (default true)
  FAKE_SLEEP        — seconds to stall before answering session/prompt
"""

import json
import os
import sys
import time


def log(entry):
    path = os.environ.get("FAKE_ACP_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params", {})
    if method:
        log({"method": method, "params": params})
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": params.get("protocolVersion", 1),
                    "agentCapabilities": {
                        "loadSession": os.environ.get("FAKE_LOADSESSION", "1") != "0",
                    },
                    "agentInfo": {"name": "fake-agent", "version": "1.0.0"},
                },
            }
        )
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "fake-sess-1"}})
    elif method == "session/load":
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "session/prompt":
        time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
        sid = params.get("sessionId", "fake-sess-1")
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "hello from fake"},
                    },
                },
            }
        )
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})

#!/usr/bin/env python3
"""Deterministic fake model provider for BenchFlow's deterministic integration tier.

It speaks just enough of the Anthropic Messages API (``POST /v1/messages``,
streaming SSE and plain JSON, ``/v1/messages/count_tokens``) for Claude Code
behind the LiteLLM proxy to run a scripted conversation: tool calls, then a
final message. It needs only the Python standard library, so it runs inside a
task image (``python3``) as well as on the host.

The server is stateless. Every reply is a function of the request alone:

- The script is chosen by a ``[[fake-llm:NAME]]`` marker in the most recent
  user message that carries one (the task instruction, a ``--prompt``, a branch
  child's prompt or a retry prompt). Scripts live in ``scripts.json`` next to
  this file.
- The step within that script is the number of assistant messages that follow
  the marker message, so a restarted server, a branch child resumed from a
  snapshot, or a retried request gets the same reply.
- Requests without tools (Claude Code's side calls such as title generation)
  get a fixed ``ok`` with ``SIDE_USAGE`` (0 in, 1 out).

Every main-loop reply reports fixed usage (``USAGE``) so token and cost fields
of a rollout are exact. Each request is appended to ``--log`` as JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

MARKER = re.compile(r"\[\[fake-llm:([A-Za-z0-9_.-]+)\]\]")
USAGE = {
    "input_tokens": 1000,
    "output_tokens": 50,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
# Side calls (no tools) report one output token, as LiteLLM would count "ok".
SIDE_USAGE = dict(dict.fromkeys(USAGE, 0), output_tokens=1)
DEFAULT_SCRIPTS = Path(__file__).with_name("scripts.json")


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    out: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
    return out


def locate(messages: list[dict[str, Any]]) -> tuple[str | None, int]:
    """Return (script name, step index) for a conversation.

    The script is named by the last marker in the latest user message holding
    one; the step is how many assistant messages follow that message.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        names = [m for t in _texts(message.get("content")) for m in MARKER.findall(t)]
        if names:
            step = sum(1 for m in messages[index + 1 :] if m.get("role") == "assistant")
            return names[-1], step
    return None, 0


def _resolve_tool(requested: str, tools: list[dict[str, Any]]) -> str | None:
    names = [str(t.get("name", "")) for t in tools if isinstance(t, dict)]
    for name in names:
        if name == requested:
            return name
    for name in names:
        if name.endswith("__" + requested):
            return name
    return None


def plan_reply(
    body: dict[str, Any], scripts: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Build the (non-streaming) Messages API reply for one request."""
    model = str(body.get("model") or "fake-model")
    tools = body.get("tools") or []
    if not tools:
        return _message(
            "msg_fake_side",
            model,
            [{"type": "text", "text": "ok"}],
            "end_turn",
            SIDE_USAGE,
        )

    name, step = locate(body.get("messages") or [])
    if name is None:
        content = [
            {
                "type": "text",
                "text": "fake-llm: no [[fake-llm:NAME]] marker in the prompt",
            }
        ]
        return _message("msg_fake_nomarker", model, content, "end_turn", USAGE)
    script = scripts.get(name)
    if script is None:
        content = [{"type": "text", "text": f"fake-llm: unknown script {name!r}"}]
        return _message(f"msg_fake_{name}_unknown", model, content, "end_turn", USAGE)
    if step >= len(script):
        content = [{"type": "text", "text": "Done."}]
        return _message(f"msg_fake_{name}_{step}", model, content, "end_turn", USAGE)

    entry = script[step]
    content: list[dict[str, Any]] = []
    if entry.get("text"):
        content.append({"type": "text", "text": entry["text"]})
    stop_reason = "end_turn"
    if entry.get("tool"):
        tool_name = _resolve_tool(entry["tool"], tools)
        if tool_name is None:
            offered = sorted(
                str(t.get("name", "")) for t in tools if isinstance(t, dict)
            )
            content.append(
                {
                    "type": "text",
                    "text": f"fake-llm: tool {entry['tool']!r} not offered; offered: {offered}",
                }
            )
        else:
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_fake_{name}_{step}",
                    "name": tool_name,
                    "input": entry.get("input") or {},
                }
            )
            stop_reason = "tool_use"
    return _message(f"msg_fake_{name}_{step}", model, content, stop_reason, USAGE)


def _message(
    msg_id: str,
    model: str,
    content: list[dict[str, Any]],
    stop: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": dict(usage),
    }


def sse_events(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Render a reply as the Anthropic streaming event sequence."""
    usage = message["usage"]
    start_usage = dict(usage, output_tokens=min(1, usage["output_tokens"]))
    head = dict(message, content=[], stop_reason=None, usage=start_usage)
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": head})
    ]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            )
        else:
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": dict(block, input={}),
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block["input"]),
                        },
                    },
                )
            )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": index})
        )
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return events


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeLLM/1"
    protocol_version = "HTTP/1.1"
    scripts: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    log_path: Path | None = None
    lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _log(self, entry: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        with self.lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json(200, {"ok": True, "scripts": sorted(self.scripts)})
        else:
            self._json(
                404,
                {
                    "type": "error",
                    "error": {"type": "not_found_error", "message": path},
                },
            )

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self._json(
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad json"},
                },
            )
            return
        if path.endswith("/messages/count_tokens"):
            self._log({"path": path, "kind": "count_tokens"})
            self._json(200, {"input_tokens": USAGE["input_tokens"]})
            return
        if not path.endswith("/messages"):
            self._log({"path": path, "kind": "unsupported"})
            self._json(
                404,
                {
                    "type": "error",
                    "error": {"type": "not_found_error", "message": path},
                },
            )
            return
        reply = plan_reply(body, self.scripts)
        self._log(
            {
                "path": path,
                "kind": "messages",
                "stream": bool(body.get("stream")),
                "n_messages": len(body.get("messages") or []),
                "n_tools": len(body.get("tools") or []),
                "reply_id": reply["id"],
                "stop_reason": reply["stop_reason"],
            }
        )
        if not body.get("stream"):
            self._json(200, reply)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event, data in sse_events(reply):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    _Handler.scripts = json.loads(args.scripts.read_text())
    _Handler.log_path = args.log
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

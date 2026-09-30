#!/usr/bin/env python3
"""Deterministic fake model provider for BenchFlow's deterministic integration tier.

It speaks just enough of the Anthropic Messages API (``POST /v1/messages``,
streaming SSE and plain JSON, ``/v1/messages/count_tokens``) for Claude Code
behind the LiteLLM proxy to run a scripted conversation: tool calls, then a
final message. ``POST /v1/chat/completions`` serves the same scripts as a
self-hosted policy server with token ids and logprobs, shaped like vLLM
(``/v1/...``) or SGLang (``/sglang/v1/...``); see the section below. It needs only the Python standard library, so it runs inside a
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
- A step with ``delay_sec`` answers after that many seconds (a slow model,
  for timeouts that hit a model call rather than a tool).

Every main-loop reply reports fixed usage (``USAGE``) so token and cost fields
of a rollout are exact. Each request is appended to ``--log`` as JSONL.

``POST /v1/responses`` serves the same scripts as the OpenAI Responses API
that Codex speaks (streaming SSE and plain JSON): a step's ``Bash`` tool
becomes a call of the shell tool Codex offers (``exec_command``,
``shell_command`` or ``shell``), and its call id is
``call_fake_<script>_<step>``. With ``--wire-log`` every request body and its
non-secret headers are appended there too, which the native-harness wire
parity check diffs.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
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


def script_delay(
    messages: list[dict[str, Any]], scripts: dict[str, list[dict[str, Any]]]
) -> float:
    """Seconds the step's ``delay_sec`` holds the reply back (a slow model)."""
    name, step = locate(messages)
    script = scripts.get(name or "") or []
    if step < len(script):
        return float(script[step].get("delay_sec") or 0)
    return 0.0


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


# ---------------------------------------------------------------------------
# OpenAI chat completions with token ids and logprobs (vLLM / SGLang shapes)
# ---------------------------------------------------------------------------
#
# A self-hosted policy server behind the ``vllm/`` or ``sglang/`` route. The
# scripted reply is the same as on the Messages API; what is added is a
# deterministic chat template and tokenizer so token ids behave like a real
# server's: the prompt is ``render(messages) + "<|assistant|>"``, the sampled
# tokens are the assistant turn as the template renders it (text, then each
# tool call as ``<tool_call>{json}</tool_call>``, then ``<|end|>``), and one
# character is one token (id = code point). An agent that resends history
# unchanged therefore produces prompts that extend the previous prompt and
# sampled tokens exactly (token-in/token-out), as with a real server.

END = "<|end|>"


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def _canonical_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return arguments
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _render_tool_calls(tool_calls: Any) -> str:
    out = ""
    for call in tool_calls or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            payload = {
                "name": function.get("name"),
                "arguments": json.loads(_canonical_args(function.get("arguments"))),
            }
            out += f"<tool_call>{json.dumps(payload, sort_keys=True, separators=(',', ':'))}</tool_call>"
    return out


def render_chat(body: dict[str, Any]) -> str:
    """The deterministic chat template: tools, then every message, then the turn opener."""
    names = sorted(
        _tool_name(t) for t in body.get("tools") or [] if isinstance(t, dict)
    )
    text = f"<|tools|>{','.join(names)}{END}" if names else ""
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = "".join(_texts(message.get("content")))
        text += (
            f"<|{role}|>{content}{_render_tool_calls(message.get('tool_calls'))}{END}"
        )
    return text + "<|assistant|>"


def tokenize(text: str) -> list[int]:
    return [ord(char) for char in text]


def chat_logprob(index: int) -> float:
    return -0.125 * ((index % 8) + 1)


def plan_chat_reply(
    body: dict[str, Any], scripts: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """``{"content", "tool_calls", "finish_reason", "sampled"}`` for one chat request."""
    tools = body.get("tools") or []
    anthropic_tools = [{"name": _tool_name(t)} for t in tools if isinstance(t, dict)]
    plan = plan_reply(
        {
            "model": body.get("model"),
            "tools": anthropic_tools,
            "messages": body.get("messages") or [],
        },
        scripts,
    )
    text = "".join(b["text"] for b in plan["content"] if b["type"] == "text")
    tool_calls = [
        {
            "id": b["id"].replace("toolu_", "call_"),
            "type": "function",
            "function": {"name": b["name"], "arguments": _canonical_args(b["input"])},
        }
        for b in plan["content"]
        if b["type"] == "tool_use"
    ]
    sampled = text + _render_tool_calls(tool_calls) + END
    return {
        "content": text or None,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "sampled": sampled,
    }


def _chat_logprobs(sampled: str, start: int = 0) -> dict[str, Any]:
    return {
        "content": [
            {
                "token": char,
                "logprob": chat_logprob(start + i),
                "bytes": list(char.encode()),
                "top_logprobs": [],
            }
            for i, char in enumerate(sampled)
        ]
    }


def chat_response(
    body: dict[str, Any], reply: dict[str, Any], flavor: str
) -> dict[str, Any]:
    prompt_ids = tokenize(render_chat(body))
    sampled_ids = tokenize(reply["sampled"])
    message: dict[str, Any] = {"role": "assistant", "content": reply["content"]}
    if reply["tool_calls"]:
        message["tool_calls"] = reply["tool_calls"]
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": reply["finish_reason"],
        "logprobs": _chat_logprobs(reply["sampled"]) if body.get("logprobs") else None,
    }
    response: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": str(body.get("model") or "fake-model"),
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(sampled_ids),
            "total_tokens": len(prompt_ids) + len(sampled_ids),
        },
    }
    if flavor == "vllm" and body.get("return_token_ids"):
        response["prompt_token_ids"] = prompt_ids
        choice["token_ids"] = sampled_ids
    ext = _sglext(body, prompt_ids, sampled_ids) if flavor == "sglang" else None
    if ext:
        response["sglext"] = ext
    return response


def _sglext(
    body: dict[str, Any], prompt_ids: list[int], sampled_ids: list[int]
) -> dict[str, Any] | None:
    ext: dict[str, Any] = {}
    if body.get("return_input_ids_in_sglext"):
        ext["input_ids"] = prompt_ids
    if body.get("return_output_ids_in_sglext"):
        ext["output_ids"] = [sampled_ids]
    return ext or None


def chat_chunks(
    body: dict[str, Any], reply: dict[str, Any], flavor: str
) -> list[dict[str, Any]]:
    """The streamed form: role, text, tool calls, end marker, finish (+ usage), sglext."""
    full = chat_response(body, reply, flavor)
    base = {k: full[k] for k in ("id", "created", "model")} | {
        "object": "chat.completion.chunk"
    }
    vllm_ids = flavor == "vllm" and bool(body.get("return_token_ids"))
    want_logprobs = bool(body.get("logprobs"))
    pieces: list[tuple[dict[str, Any], str]] = []
    if reply["content"]:
        pieces.append(({"content": reply["content"]}, reply["content"]))
    for index, call in enumerate(reply["tool_calls"]):
        pieces.append(
            (
                {"tool_calls": [dict(call, index=index)]},
                _render_tool_calls([call]),
            )
        )
    pieces.append(({}, END))
    first: dict[str, Any] = {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    if vllm_ids:
        first["prompt_token_ids"] = full["prompt_token_ids"]
    chunks = [first]
    position = 0
    for delta, text in pieces:
        choice: dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "logprobs": None,
            "finish_reason": None,
        }
        if want_logprobs:
            choice["logprobs"] = _chat_logprobs(text, position)
        if vllm_ids:
            choice["token_ids"] = tokenize(text)
        position += len(text)
        chunks.append({**base, "choices": [choice]})
    chunks.append(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": reply["finish_reason"],
                }
            ],
            "usage": full["usage"],
        }
    )
    if "sglext" in full:
        chunks.append({**base, "choices": [], "sglext": full["sglext"]})
    return chunks


# ---------------------------------------------------------------------------
# OpenAI Responses API (Codex)
# ---------------------------------------------------------------------------

# Codex's shell tools, most preferred first, and how a script's Bash input
# becomes their arguments.
_RESPONSES_SHELL_TOOLS = ("exec_command", "shell_command", "shell")


def _responses_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Responses ``input`` items as the role/content messages :func:`locate` reads."""
    items = body.get("input")
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    messages: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = item.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [
                str(c.get("text", ""))
                for c in content or []
                if isinstance(c, dict) and "text" in c
            ]
        )
        messages.append(
            {"role": role, "content": [{"type": "text", "text": t} for t in texts]}
        )
    return messages


def _responses_tool_names(body: dict[str, Any]) -> list[str]:
    names = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            names.append(str(tool.get("name") or tool.get("type") or ""))
    return names


def _shell_arguments(tool: str, entry_input: dict[str, Any]) -> dict[str, Any]:
    command = str(entry_input.get("command", ""))
    timeout_ms = entry_input.get("timeout")
    if tool == "exec_command":
        args: dict[str, Any] = {"cmd": command}
        if timeout_ms:
            # Wait for the command, as Claude Code's Bash tool does.
            args["yield_time_ms"] = int(timeout_ms)
        return args
    if tool == "shell_command":
        args = {"command": command}
    else:
        args = {"command": ["bash", "-lc", command]}
    if timeout_ms:
        args["timeout_ms"] = int(timeout_ms)
    return args


def plan_responses_reply(
    body: dict[str, Any], scripts: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """The Responses API ``response`` object for one request."""
    model = str(body.get("model") or "fake-model")
    names = _responses_tool_names(body)
    shell = next((t for t in _RESPONSES_SHELL_TOOLS if t in names), None)
    # A script's Bash step runs through Codex's shell tool.
    offered = [{"name": n} for n in names] + ([{"name": "Bash"}] if shell else [])
    anthropic_like = {
        "model": model,
        "tools": offered,
        "messages": _responses_messages(body),
    }
    plan = plan_reply(anthropic_like, scripts)
    name, step = locate(anthropic_like["messages"])
    usage = plan["usage"]
    output: list[dict[str, Any]] = []
    text = "".join(b["text"] for b in plan["content"] if b["type"] == "text")
    if text:
        output.append(
            {
                "type": "message",
                "id": plan["id"].replace("msg_", "msg_out_"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    entry = (scripts.get(name or "") or [])[step : step + 1]
    wanted = entry[0].get("tool") if entry else None
    if wanted and names:
        tool = (
            shell
            if wanted == "Bash" and shell
            else _resolve_tool(wanted, [{"name": n} for n in names])
        )
        if tool is not None:
            arguments = (
                _shell_arguments(tool, entry[0].get("input") or {})
                if tool in _RESPONSES_SHELL_TOOLS
                else entry[0].get("input") or {}
            )
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_fake_{name}_{step}",
                    "call_id": f"call_fake_{name}_{step}",
                    "name": tool,
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                }
            )
    return {
        "id": plan["id"].replace("msg_", "resp_"),
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage["output_tokens"],
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        },
    }


def responses_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The Responses API streaming events for one response."""
    head = dict(response, status="in_progress", output=[], usage=None)
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": head},
        {"type": "response.in_progress", "response": head},
    ]
    for index, item in enumerate(response["output"]):
        opened = dict(item, status="in_progress")
        if item["type"] == "message":
            opened["content"] = []
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": opened,
            }
        )
        if item["type"] == "message":
            text = item["content"][0]["text"]
            part = {"type": "output_text", "text": "", "annotations": []}
            events += [
                {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": part,
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                },
                {
                    "type": "response.output_text.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                },
                {
                    "type": "response.content_part.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": dict(part, text=text),
                },
            ]
        else:
            events += [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "arguments": item["arguments"],
                },
            ]
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    events.append({"type": "response.completed", "response": response})
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


# Headers the wire log keeps: what identifies the client and the protocol,
# never credentials.
_WIRE_HEADERS = frozenset(
    {
        "user-agent",
        "anthropic-beta",
        "anthropic-version",
        "x-app",
        "content-type",
        "openai-beta",
        "originator",
        "version",
    }
)


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeLLM/1"
    protocol_version = "HTTP/1.1"
    scripts: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    log_path: Path | None = None
    wire_log_path: Path | None = None
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
        if self.wire_log_path is not None:
            headers = {
                k.lower(): v
                for k, v in self.headers.items()
                if k.lower() in _WIRE_HEADERS
            }
            with self.lock, self.wire_log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"path": path, "headers": headers, "body": body}) + "\n"
                )
        if path.endswith("/chat/completions"):
            self._chat(path, body)
            return
        if path.endswith("/responses"):
            self._responses(path, body)
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
        if body.get("tools"):
            time.sleep(script_delay(body.get("messages") or [], self.scripts))
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

    def _responses(self, path: str, body: dict[str, Any]) -> None:
        if body.get("tools"):
            time.sleep(script_delay(_responses_messages(body), self.scripts))
        response = plan_responses_reply(body, self.scripts)
        self._log(
            {
                "path": path,
                "kind": "responses",
                "stream": bool(body.get("stream")),
                "n_input": len(body.get("input") or []),
                "n_tools": len(body.get("tools") or []),
                "reply_id": response["id"],
                "calls": [
                    item["call_id"]
                    for item in response["output"]
                    if item["type"] == "function_call"
                ],
            }
        )
        if not body.get("stream"):
            self._json(200, response)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in responses_events(response):
            self.wfile.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
            self.wfile.flush()
        self.close_connection = True

    def _chat(self, path: str, body: dict[str, Any]) -> None:
        # ``/sglang/v1/...`` answers like SGLang, anything else like vLLM.
        flavor = "sglang" if "/sglang/" in path else "vllm"
        if flavor == "sglang" and body.get("stream") and body.get("return_token_ids"):
            self._log({"path": path, "kind": "chat", "refused": "return_token_ids"})
            self._json(
                400,
                {
                    "object": "error",
                    "message": "return_token_ids is not supported with stream=true",
                    "type": "BadRequestError",
                    "code": 400,
                },
            )
            return
        reply = plan_chat_reply(body, self.scripts)
        self._log(
            {
                "path": path,
                "kind": "chat",
                "flavor": flavor,
                "stream": bool(body.get("stream")),
                "n_messages": len(body.get("messages") or []),
                "n_tools": len(body.get("tools") or []),
                "logprobs": bool(body.get("logprobs")),
                "return_token_ids": bool(body.get("return_token_ids")),
                "sglext_ids": bool(body.get("return_output_ids_in_sglext")),
                "finish_reason": reply["finish_reason"],
            }
        )
        if not body.get("stream"):
            self._json(200, chat_response(body, reply, flavor))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chat_chunks(body, reply, flavor):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--wire-log", type=Path, default=None)
    args = parser.parse_args()
    _Handler.scripts = json.loads(args.scripts.read_text())
    _Handler.log_path = args.log
    _Handler.wire_log_path = args.wire_log
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

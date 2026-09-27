#!/usr/bin/env python3
"""Local OpenAI-compatible server that returns token ids and logprobs.

It imitates the fields a vLLM OpenAI-compatible server returns so BenchFlow's
LiteLLM gateway can be tested end to end without a GPU or network:

- ``POST /v1/chat/completions``: with ``logprobs: true`` each choice carries
  ``logprobs.content`` (``token``, ``logprob``, ``bytes``, ``top_logprobs``);
  with ``return_token_ids: true`` the response carries top-level
  ``prompt_token_ids`` and per-choice ``token_ids``. Streaming (SSE) puts
  ``prompt_token_ids`` on the first chunk and per-chunk ``token_ids`` and
  ``logprobs`` on each delta, like vLLM.
- SGLang mode (``start_server(flavor="sglang")``): token ids follow SGLang's
  response-level ``sglext`` extension instead. ``return_input_ids_in_sglext``
  / ``return_output_ids_in_sglext`` add ``sglext.input_ids`` and
  ``sglext.output_ids`` (one list per choice) to the body, or, when
  streaming, to one final ``choices: []`` chunk before ``[DONE]``. SGLang's
  older ``return_token_ids`` is refused on streamed chat (HTTP 400), as SGLang
  does, and vLLM's field names are never returned.
- ``POST /v1/responses``: with ``include: ["message.output_text.logprobs"]``
  the ``output_text`` part carries ``logprobs`` (OpenAI Responses shape). The
  Responses API has no token-id field.
- ``POST /v1/messages``: an Anthropic Messages response (no logprobs exist in
  that API), for checking the explicit ``unavailable`` record.

Tokenization is deterministic: each character of the reply is one token whose
id is ``1000 + ord(char)``; prompt ids are ``ord`` of each character of the
concatenated message text. Every request body is kept in ``requests`` so tests
can assert what the gateway sent upstream.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

REPLY = "ok!"
MODEL = "mock-policy"


def token_id(char: str) -> int:
    return 1000 + ord(char)


def logprob(index: int) -> float:
    return -0.125 * (index + 1)


def prompt_token_ids(body: dict[str, Any]) -> list[int]:
    text = ""
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            text += "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
    return [ord(char) for char in text]


def _logprob_entry(index: int, char: str, top_n: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "token": char,
        "logprob": logprob(index),
        "bytes": list(char.encode()),
        "top_logprobs": [],
    }
    if top_n:
        entry["top_logprobs"] = [
            {"token": char, "logprob": logprob(index), "bytes": list(char.encode())},
            {"token": "?", "logprob": -9.0, "bytes": [63]},
        ][:top_n]
    return entry


def _sglext(body: dict[str, Any]) -> dict[str, Any] | None:
    ext: dict[str, Any] = {}
    if body.get("return_input_ids_in_sglext"):
        ext["input_ids"] = prompt_token_ids(body)
    if body.get("return_output_ids_in_sglext"):
        ext["output_ids"] = [[token_id(char) for char in REPLY]]
    return ext or None


def chat_completion(body: dict[str, Any], flavor: str = "vllm") -> dict[str, Any]:
    top_n = int(body.get("top_logprobs") or 0)
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": REPLY},
        "finish_reason": "stop",
        "logprobs": None,
    }
    if body.get("logprobs"):
        choice["logprobs"] = {
            "content": [_logprob_entry(i, char, top_n) for i, char in enumerate(REPLY)]
        }
    response: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_token_ids(body)),
            "completion_tokens": len(REPLY),
            "total_tokens": len(prompt_token_ids(body)) + len(REPLY),
        },
    }
    if flavor == "sglang":
        ext = _sglext(body)
        if ext:
            response["sglext"] = ext
        if body.get("return_token_ids"):
            choice["prompt_token_ids"] = prompt_token_ids(body)
            choice["token_ids"] = [token_id(char) for char in REPLY]
    elif body.get("return_token_ids"):
        response["prompt_token_ids"] = prompt_token_ids(body)
        choice["token_ids"] = [token_id(char) for char in REPLY]
    return response


def chat_completion_chunks(
    body: dict[str, Any], flavor: str = "vllm"
) -> list[dict[str, Any]]:
    top_n = int(body.get("top_logprobs") or 0)
    base = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
    }
    chunks: list[dict[str, Any]] = []
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
    vllm_ids = flavor == "vllm" and bool(body.get("return_token_ids"))
    if vllm_ids:
        first["prompt_token_ids"] = prompt_token_ids(body)
    chunks.append(first)
    for i, char in enumerate(REPLY):
        choice: dict[str, Any] = {
            "index": 0,
            "delta": {"content": char},
            "logprobs": None,
            "finish_reason": None,
        }
        if body.get("logprobs"):
            choice["logprobs"] = {"content": [_logprob_entry(i, char, top_n)]}
        if vllm_ids:
            choice["token_ids"] = [token_id(char)]
        chunks.append({**base, "choices": [choice]})
    chunks.append(
        {
            **base,
            "choices": [
                {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids(body)),
                "completion_tokens": len(REPLY),
                "total_tokens": len(prompt_token_ids(body)) + len(REPLY),
            },
        }
    )
    ext = _sglext(body) if flavor == "sglang" else None
    if ext:
        chunks.append({**base, "choices": [], "sglext": ext})
    return chunks


def responses_response(body: dict[str, Any]) -> dict[str, Any]:
    include = body.get("include") or []
    top_n = int(body.get("top_logprobs") or 0)
    part: dict[str, Any] = {"type": "output_text", "text": REPLY, "annotations": []}
    if "message.output_text.logprobs" in include:
        part["logprobs"] = [
            _logprob_entry(i, char, top_n) for i, char in enumerate(REPLY)
        ]
    return {
        "id": "resp_mock",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": MODEL,
        "output": [
            {
                "type": "message",
                "id": "msg_mock",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 3,
            "output_tokens": len(REPLY),
            "total_tokens": 3 + len(REPLY),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def anthropic_message(body: dict[str, Any]) -> dict[str, Any]:
    del body
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": REPLY}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": len(REPLY)},
    }


class MockTokenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], flavor: str = "vllm") -> None:
        super().__init__(address, _Handler)
        if flavor not in {"vllm", "sglang"}:
            raise ValueError(f"unknown flavor {flavor!r}")
        self.flavor = flavor
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    def record(self, path: str, body: dict[str, Any]) -> None:
        with self._lock:
            self.requests.append({"path": path, "body": body})


class _Handler(BaseHTTPRequestHandler):
    server: MockTokenServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, chunks: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": MODEL}]})
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        path = self.path.split("?", 1)[0].rstrip("/")
        self.server.record(path, body)
        flavor = self.server.flavor
        if path.endswith("/chat/completions"):
            if (
                flavor == "sglang"
                and body.get("stream")
                and body.get("return_token_ids")
            ):
                self._send_json(
                    {
                        "object": "error",
                        "message": "return_token_ids is not supported with "
                        "stream=true for chat completions",
                        "type": "BadRequestError",
                        "code": 400,
                    },
                    status=400,
                )
            elif body.get("stream"):
                self._send_sse(chat_completion_chunks(body, flavor))
            else:
                self._send_json(chat_completion(body, flavor))
        elif path.endswith("/responses"):
            self._send_json(responses_response(body))
        elif path.endswith("/messages"):
            self._send_json(anthropic_message(body))
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)


def start_server(
    host: str = "127.0.0.1", port: int = 0, flavor: str = "vllm"
) -> MockTokenServer:
    """Start the server on a background thread and return it."""
    server = MockTokenServer((host, port), flavor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--flavor", choices=("vllm", "sglang"), default="vllm")
    args = parser.parse_args()
    server = MockTokenServer((args.host, args.port), args.flavor)
    print(f"mock token server on {server.base_url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

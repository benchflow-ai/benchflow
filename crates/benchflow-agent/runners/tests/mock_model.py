"""A scripted OpenAI-compatible model for testing BenchFlow's runner scripts.

Written for BenchFlow. Serves `POST /v1/chat/completions` (plain and
streamed) on 127.0.0.1:<port> with the Python standard library only:

- a request whose bearer key is `bad-key` gets OpenAI's 401 for a wrong key;
- a request that offers tools and has no tool result yet gets one call to a
  shell-like tool, running `echo benchflow > bf_mock_out.txt && cat bf_mock_out.txt`;
- any other request gets the answer `BF-MOCK-DONE`.

Every response reports usage (100 prompt tokens, 10 completion tokens), so a
runner's totals can be checked. Usage: python3 mock_model.py <port>
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMAND = "echo benchflow > bf_mock_out.txt && cat bf_mock_out.txt"
ANSWER = "BF-MOCK-DONE: bf_mock_out.txt says benchflow."
SHELL_NAMES = (
    "terminal",
    "bash",
    "shell",
    "sandbox_shell",
    "execute_bash",
    "run_shell_command",
    "execute",
    "run_command",
)


def pick_tool(tools: list[dict]) -> dict | None:
    functions = [t.get("function", t) for t in tools if isinstance(t, dict)]
    for name in SHELL_NAMES:
        for f in functions:
            if f.get("name") == name:
                return f
    for f in functions:
        if "command" in (f.get("parameters") or {}).get("properties", {}):
            return f
    return None


def arguments_for(function: dict) -> dict:
    params = function.get("parameters") or {}
    props = params.get("properties", {})
    args: dict = {}
    for name in set(params.get("required", [])) | ({"command"} & set(props)):
        schema = props.get(name, {})
        if name in ("command", "cmd"):
            args[name] = COMMAND
        elif "enum" in schema:
            args[name] = "LOW" if "LOW" in schema["enum"] else schema["enum"][0]
        elif schema.get("type") == "boolean":
            args[name] = False
        elif schema.get("type") in ("integer", "number"):
            args[name] = 1
        elif schema.get("type") == "array":
            args[name] = []
        else:
            args[name] = "run the mock command"
    return args


def usage() -> dict:
    return {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write("mock_model: " + (format % args) + "\n")

    def do_GET(self) -> None:
        self.send_json(200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
        if self.headers.get("authorization", "") == "Bearer bad-key":
            self.send_json(
                401,
                {
                    "error": {
                        "message": "Incorrect API key provided: bad-key. You can find your API key at https://platform.openai.com/account/api-keys.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
            return
        messages = body.get("messages", [])
        answered = any(m.get("role") == "tool" for m in messages)
        tool = None if answered else pick_tool(body.get("tools") or [])
        model = body.get("model", "mock-model")
        if tool is not None:
            call = {
                "id": f"call_{uuid.uuid4().hex[:10]}",
                "type": "function",
                "function": {"name": tool["name"], "arguments": json.dumps(arguments_for(tool))},
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": ANSWER}
            finish = "stop"
        if body.get("stream"):
            self.stream(model, message, finish, body)
        else:
            self.send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage(),
                },
            )

    def stream(self, model: str, message: dict, finish: str, body: dict) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {"id": f"chatcmpl-{uuid.uuid4().hex[:10]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        delta: dict = {"role": "assistant"}
        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            delta["tool_calls"] = [{"index": 0, **call}]
        else:
            delta["content"] = message["content"]
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        ]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks.append({**base, "choices": [], "usage": usage()})
        else:
            chunks[-1]["usage"] = usage()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

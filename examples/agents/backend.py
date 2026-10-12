"""A scripted service for `bench run -a external`: it decides every step of the hello task.

    python3 agents/backend.py port.txt &          # serves on a free port, written to port.txt
    bench run tasks/hello -a external --backend http://127.0.0.1:$(cat port.txt) -m scripted

BenchFlow asks the service for each next message (GET /api/internal/get_message, 404 while there is
none) and runs its tool calls in the sandbox with the builtin loop's tools (bash, view, replace, write,
submit). Everything BenchFlow reports (the task, each tool's result, the reward) arrives as events on
POST /api/internal/append_transcript. A real service would read those events and ask a model; this one
plays three fixed messages per run. Standard library only.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def tool_call(call_id, name, arguments):
    """A `message_added` event whose assistant message makes one tool call."""
    return {
        "type": "message_added",
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}],
        },
    }


SCRIPT = [
    tool_call("c1", "write", {"file_path": "/app/hello.txt", "content": "Hello, world!"}),
    tool_call("c2", "bash", {"command": "cat /app/hello.txt"}),
    tool_call("c3", "submit", {"summary": "wrote /app/hello.txt"}),
]

lock = threading.Lock()
runs = {}  # run id -> {"events": {seq: event}, "next": index into SCRIPT, "states": [...]}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def route(self):
        url = urlparse(self.path)
        run_id = parse_qs(url.query).get("run_id", [""])[0]
        return url.path.removeprefix("/api/internal/"), run_id

    def do_GET(self):
        name, run_id = self.route()
        with lock:
            run = runs.get(run_id)
            if name == "transcript_length":
                return self.reply(200, len(run["events"])) if run else self.reply(404, {"detail": "no transcript"})
            if name == "get_message":
                if run is None or run["next"] >= len(SCRIPT):
                    return self.reply(404, {"detail": "no message"})
                return self.reply(200, SCRIPT[run["next"]])
        self.reply(404, {"detail": "unknown route"})

    def do_POST(self):
        name, run_id = self.route()
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        with lock:
            run = runs.setdefault(run_id, {"events": {}, "next": 0, "states": []})
            if name == "append_transcript":
                run["events"][body["seq"]] = body["event"]
                if body["event"].get("type") == "scoring":
                    print(f"{run_id}: reward {body['event']['scoring'].get('score')}", flush=True)
            elif name == "update_run_state":
                run["states"].append(body)
            elif name == "delete_message":
                run["next"] += 1
        self.reply(200, None)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    with open(sys.argv[1] if len(sys.argv) > 1 else "port.txt", "w") as f:
        f.write(str(server.server_address[1]))
    server.serve_forever()

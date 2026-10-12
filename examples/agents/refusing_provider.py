"""A model provider that refuses every request with HTTP 400, for a trial the provider fails.

    python3 agents/refusing_provider.py port.txt &          # serves on a free port, written to port.txt
    BENCHFLOW_NATIVE_BASE_URL=http://127.0.0.1:$(cat port.txt)/v1 MIMO_API_KEY=placeholder \
        bench run tasks/hello -a builtin -m mimo/mimo-v2.6-flash

The builtin loop calls the model from this machine, and `BENCHFLOW_NATIVE_BASE_URL` (read only from
the shell, as https or http to this machine) sends those calls here instead of to the provider, so
the run needs no key. Every answer is a 400 with an OpenAI-style error, as a provider gives for a
request it will not serve. The trial is errored (R214, owned by the model provider), never a scored
0. Each request is printed as `<method> <path> -> 400`. Standard library only.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ERROR = {"error": {"message": "this endpoint refuses every request", "type": "invalid_request_error",
                   "code": "request_refused"}}


class Handler(BaseHTTPRequestHandler):
    def refuse(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        data = json.dumps(ERROR).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        print(f"{self.command} {self.path} -> 400", flush=True)

    do_GET = do_POST = do_PUT = do_DELETE = refuse

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    with open(sys.argv[1] if len(sys.argv) > 1 else "port.txt", "w") as f:
        f.write(str(server.server_address[1]))
    server.serve_forever()

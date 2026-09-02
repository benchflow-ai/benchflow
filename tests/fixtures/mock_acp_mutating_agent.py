#!/usr/bin/env python3
"""ACP fixture whose TERM-resistant child continuously mutates a file."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _append_event(event: str) -> None:
    with Path(os.environ["EVENT_PATH"]).open("a") as stream:
        stream.write(event + "\n")
        stream.flush()


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _start_mutating_child() -> None:
    program = (
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['MUTATION_PATH'])\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    with path.open('a') as stream:\n"
        "        stream.write('x')\n"
        "        stream.flush()\n"
        "    time.sleep(0.01)\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def main() -> None:
    _start_mutating_child()
    try:
        for line in sys.stdin:
            message = json.loads(line)
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": 1,
                            "agentInfo": {
                                "name": "mutating-agent",
                                "version": "1.0.0",
                            },
                            "agentCapabilities": {},
                            "authMethods": [],
                        },
                    }
                )
            elif method == "session/new":
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"sessionId": "mutating-session"},
                    }
                )
            elif method == "session/cancel":
                _append_event("cancel")
            elif request_id is not None:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "unsupported"},
                    }
                )
    finally:
        _append_event("session_closed")


if __name__ == "__main__":
    main()

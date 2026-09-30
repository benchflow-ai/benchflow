"""A stand-in for bridge.py: the same protocol, scripted, with no BenchFlow and no network.

The tests point ``benchflow_python`` at a shim that runs this file instead of the
real bridge. ``$FAKE_BRIDGE_SCENARIO`` names a JSON file:

    {
      "log": "<path>",                 # every event, one JSON line each
      "tasks": [{...}],                # rows printed in `tasks` mode
      "start": {...},                  # the reply to `start`
      "bash": [{...}, ...],            # replies to `bash`, in order (the last repeats)
      "write": {...}, "verify": {...},
      "die_on": "bash",                # exit abruptly when this op arrives
      "hang_on": "verify"              # never answer this op
    }

Events: ``spawned`` (with the names of the environment variables it received),
``request``, ``sandbox_closed`` (reason ``close``, ``eof`` or ``sigterm``), ``dying``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> int:
    scenario = json.loads(Path(os.environ["FAKE_BRIDGE_SCENARIO"]).read_text())
    log = Path(scenario["log"])

    def record(event: str, **fields) -> None:
        with log.open("a") as handle:
            handle.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n")

    if sys.argv[1] == "tasks":
        for row in scenario.get("tasks", []):
            print(json.dumps(row))
        return 0

    closed = {"done": False}

    def close(reason: str) -> None:
        if not closed["done"]:
            closed["done"] = True
            record("sandbox_closed", reason=reason)

    def on_term(signum, frame):
        close("sigterm")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    record("spawned", argv=sys.argv[1:], env=sorted(os.environ))
    bash_replies = list(scenario.get("bash") or [{"ok": True, "return_code": 0, "stdout": "", "stderr": ""}])
    for line in sys.stdin:
        request = json.loads(line)
        op = request.get("op")
        record("request", op=op, request=request)
        if op == scenario.get("die_on"):
            record("dying", op=op)
            os._exit(3)
        if op == scenario.get("hang_on"):
            time.sleep(3600)
        if op == "close":
            close("close")
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            return 0
        if op == "bash":
            reply = bash_replies[0] if len(bash_replies) == 1 else bash_replies.pop(0)
        else:
            reply = scenario.get(op) or {"ok": True}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    close("eof")
    return 0


if __name__ == "__main__":
    sys.exit(main())

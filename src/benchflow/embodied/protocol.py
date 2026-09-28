"""The episode wire protocol: JSON requests over a Unix socket, one request per connection.

Version 2 adds action groups (`act` with `groups`), typed skills (`skill` with list or dict args), roles, and the
`embodiment` block in `info`. Every version-1 request still works: flat `act`, `move_to`, `grip`. See docs/embodied.md.
Standard library only.
"""

from __future__ import annotations

import json
import socket
from typing import Any

PROTOCOL_VERSION = 2

READ_OPS = ("info", "status", "observe")
STEP_OPS = ("act", "skill", "move_to", "grip")
END_OPS = ("done", "give_up")
ALL_OPS = READ_OPS + STEP_OPS + END_OPS

# version-1 ops that are skills now: op -> (skill role, how to build args)
LEGACY_SKILL_OPS = {"move_to": "move_to", "grip": "grip"}


def legacy_skill_request(req: dict) -> tuple[str, dict]:
    """Translate a version-1 `move_to` / `grip` request into (skill alias, keyword args)."""
    op = req.get("op")
    if op == "move_to":
        pos = req.get("pos")
        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError("pos must be [x, y, z]")
        kw: dict[str, Any] = {"x": pos[0], "y": pos[1], "z": pos[2]}
        if req.get("grip") is not None:
            kw["grip"] = req["grip"]
        if req.get("max_steps") is not None:
            kw["max_steps"] = req["max_steps"]
        if req.get("tol") is not None:
            kw["tol"] = req["tol"]
        return "move_to", kw
    if op == "grip":
        kw = {"value": req.get("value", 1.0)}
        if req.get("steps") is not None:
            kw["steps"] = req["steps"]
        return "grip", kw
    raise ValueError(f"not a legacy skill op: {op!r}")


def send(path: str, req: dict, timeout: float = 120.0) -> dict:
    """Send one request to the episode server at `path` and return its response (errors become ok=false)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode() or '{"ok": false, "error": "empty response"}')
    except OSError as e:
        return {"ok": False, "error": f"episode server unreachable: {e}"}
    finally:
        s.close()

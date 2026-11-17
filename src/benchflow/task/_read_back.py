"""What root reads back from the sandbox during or after grading.

A receipt, the verifier's output folder: BenchFlow reads them as root, and a
process the agent left running is alive through grading in integrity off and
audit. :func:`read_back` admits such a read only when nothing on the path is
the agent's to change: every folder on it is owned by root (or the verifier's
own user) and not writable by group or others, and the entry itself is not a
link and is owned the same way. A path that does not exist passes (the read
finds nothing). It looks with bfx, component by component, following no link.
"""

from __future__ import annotations

import base64
import stat
from typing import Any


class ReadBackRefused(RuntimeError):
    """A path root would read back is not BenchFlow's own to trust."""


async def read_back(
    env: Any,
    path: str,
    *,
    owners: tuple[int, ...] = (0,),
    service: str = "main",
    timeout_sec: int = 30,
) -> None:
    """Admit root's read of *path*, or raise :class:`ReadBackRefused`."""
    if not path.startswith("/"):
        raise ReadBackRefused(f"{path!r} is not an absolute path")
    from benchflow.sandbox._verifier_links import walker_unavailable

    if service != "main" or walker_unavailable(env) is not None:
        # No bfx to look with (no route, or none delivered): a model agent's
        # sandbox without it is refused before its turn in off and audit
        # (Q4), so only the oracle, strict, and a verifier in another
        # service (a sidecar) read unchecked, a named gap.
        return
    raw = path.encode("utf-8", "surrogateescape")
    try:
        request: dict[str, Any] = {"paths": [raw.decode("utf-8")]}
    except UnicodeDecodeError:
        request = {"paths_b64": [base64.b64encode(raw).decode("ascii")]}
    request.update(acl=False, mounts=False)
    try:
        answer = await env.bfx_call("scan", request, timeout_sec=timeout_sec)
    except Exception as exc:
        raise ReadBackRefused(f"{path}: could not be looked at ({exc})") from exc
    entries = [e for e in answer.get("entries") or () if isinstance(e, dict)]
    for e in entries:
        if e.get("exists") is False:
            continue
        name = e.get("path") or e.get("path_b64")
        kind = e.get("type")
        uid = e.get("uid")
        mode = e.get("mode")
        if not isinstance(uid, int) or not isinstance(mode, int):
            raise ReadBackRefused(f"{path}: {name} gave no owner or mode")
        if uid not in owners:
            raise ReadBackRefused(f"{path}: {name} is owned by uid {uid}")
        if kind == "link":
            raise ReadBackRefused(f"{path}: {name} is a link")
        last = e is entries[-1]
        sticky_root = uid == 0 and mode & stat.S_ISVTX
        if (
            kind == "dir"
            and mode & (stat.S_IWGRP | stat.S_IWOTH)
            and (last or not sticky_root)
        ):
            # A sticky folder of root's (/run/runtime, /tmp) on the way is no
            # one else's to rename or remove a root-owned entry in.
            raise ReadBackRefused(
                f"{path}: {name} is writable by group or others ({mode & 0o7777:o})"
            )

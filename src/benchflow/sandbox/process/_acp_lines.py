"""Keep every ACP line an agent writes under the size its transport carries (#1138).

A Claude Code ``Read`` of a PDF returns the pages as inline base64 images:
one ACP ``tool_call_update`` line of several megabytes. Daytona closes the
PTY websocket on such a message (close code 1008), so the rollout dies, every
attempt; Docker's reader dropped any line over its 10 MB buffer, losing the
tool call's final update with it. A line over ``LIMIT`` bytes is rewritten
instead:

1. image and audio content blocks become a text block that says what was
   removed, and any other long base64 string becomes such a note;
2. if the line is still too long, the longest remaining strings are cut, with
   a note, until it fits.

Keys that identify the message (``jsonrpc``, ``id``, ``method``, ``sessionId``,
``sessionUpdate``, ``toolCallId``, ``status``, ...) are never touched, so the
update still closes its tool call. A line that is not JSON keeps its first
64 KiB. Smaller lines pass through byte for byte.

This file runs in two places. Inside the sandbox, when ``python3`` is there,
``filter_command`` pipes the agent's stdout through ``python3 -c <this
file>`` (``main``), before the transport sees it. On the host, the transport
applies ``shrink_line`` to any line still over the limit (an image without
``python3``), so both paths deliver the same rewritten line. It therefore
uses the standard library only and Python 3.6 syntax: no annotations or
walrus.
"""

import contextlib
import json
import os
import re
import shlex
import sys

# Bytes per line, newline excluded. Over this a line is rewritten.
LIMIT = 1024 * 1024
LIMIT_ENV = "BENCHFLOW_ACP_LINE_LIMIT"
# Strings shorter than this are never replaced or cut.
_LONG = 4096
_NON_JSON_KEEP = 64 * 1024
_KEEP_KEYS = frozenset(
    [
        "jsonrpc",
        "id",
        "method",
        "sessionId",
        "sessionUpdate",
        "toolCallId",
        "status",
        "kind",
        "title",
        "stopReason",
        "type",
        "mimeType",
    ]
)
_BASE64 = re.compile(r"[A-Za-z0-9+/=\r\n]+")


def line_limit():
    """The limit in bytes; None when ``BENCHFLOW_ACP_LINE_LIMIT`` is 0/off."""
    raw = os.environ.get(LIMIT_ENV, "").strip().lower()
    if not raw:
        return LIMIT
    if raw in ("0", "off", "none", "false"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return LIMIT
    return value if value > 0 else None


def _note(what, size, limit):
    return (
        f"[BenchFlow removed {what} ({size} bytes): "
        f"the ACP line was over the {limit}-byte limit]"
    )


def _strip_media(node, limit):
    if isinstance(node, dict):
        kind = node.get("type")
        data = node.get("data")
        if kind in ("image", "audio") and isinstance(data, str) and len(data) >= _LONG:
            what = f"{kind} data ({node.get('mimeType') or 'unknown type'})"
            return {"type": "text", "text": _note(what, len(data), limit)}
        out = {}
        for key, value in node.items():
            out[key] = value if key in _KEEP_KEYS else _strip_media(value, limit)
        return out
    if isinstance(node, list):
        return [_strip_media(value, limit) for value in node]
    if isinstance(node, str) and len(node) >= _LONG and _BASE64.fullmatch(node):
        return _note("base64 data", len(node), limit)
    return node


def _long_strings(node, path, found):
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in _KEEP_KEYS:
                _long_strings(value, (*path, key), found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _long_strings(value, (*path, index), found)
    elif isinstance(node, str) and len(node) >= _LONG:
        found.append((len(node), path))


def _replace(root, path, value):
    for step in path[:-1]:
        root = root[step]
    root[path[-1]] = value


def _dumps(node):
    return json.dumps(node, ensure_ascii=False, separators=(",", ":"))


def _fit(node, limit):
    text = _dumps(node)
    for _ in range(64):
        excess = len(text.encode("utf-8")) - limit
        if excess <= 0:
            break
        found = []
        _long_strings(node, (), found)
        if not found:
            break
        size, path = max(found, key=lambda item: item[0])
        value = node
        for step in path:
            value = value[step]
        keep = max(_LONG // 2, size - excess - 256)
        _replace(
            node,
            path,
            value[:keep] + " " + _note("the rest of this text", size - keep, limit),
        )
        text = _dumps(node)
    return text


def shrink_line(line, limit=LIMIT):
    """``line`` (bytes, newline included or not), rewritten to fit ``limit``."""
    body = line.rstrip(b"\r\n")
    if len(body) <= limit:
        return line
    try:
        node = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        node = None
    if not isinstance(node, (dict, list)):
        note = _note("the rest of a non-JSON line", len(body), limit)
        return body[: min(limit, _NON_JSON_KEEP)] + b" " + note.encode() + b"\n"
    return _fit(_strip_media(node, limit), limit).encode("utf-8") + b"\n"


def main():
    """Copy stdin to stdout line by line, rewriting lines over the limit."""
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else LIMIT
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    while True:
        line = source.readline()
        if not line:
            return
        if len(line) > limit:
            # A line kept whole beats a stopped agent.
            with contextlib.suppress(Exception):
                line = shrink_line(line, limit)
        sink.write(line)
        sink.flush()


def filter_command(command, limit=None):
    """``command`` with its stdout piped through this filter when the sandbox
    has a ``python3`` that can run it; ``command`` itself otherwise, or when
    the limit is off. For a ``bash -c`` / ``bash -lc`` command string."""
    if limit is None:
        limit = line_limit()
    if limit is None:
        return command
    with open(__file__, encoding="utf-8") as handle:
        source = handle.read()
    runner = f"python3 -u -c {shlex.quote(source)} {int(limit)}"
    probe = "python3 -c 'import json, re, shlex, sys' >/dev/null 2>&1"
    return (
        f"if {probe}; then set -o pipefail; {{ {command}\n}} | {runner}; "
        f"else {command}\nfi"
    )


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError, KeyboardInterrupt):
        main()

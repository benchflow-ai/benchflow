"""Where agent CLIs keep their logins on the host when their folder can move.

Claude Code reads its login from ``$CLAUDE_CONFIG_DIR/.credentials.json`` and
Codex from ``$CODEX_HOME/auth.json`` when those variables are set, else from
``~/.claude`` and ``~/.codex``. BenchFlow looked only in the home folders, so
a login kept elsewhere read as "no login" (Rust round backlog dx-100, Rust
edd7d060). Every host-side lookup of these files goes through here; the
paths inside the sandbox stay the defaults, where the agent looks there.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# The home folder each variable replaces.
LOGIN_DIR_VARS: dict[str, str] = {
    ".claude": "CLAUDE_CONFIG_DIR",
    ".codex": "CODEX_HOME",
}


def _split(path: str) -> tuple[str, str] | None:
    """``~/.claude/x`` as ``("CLAUDE_CONFIG_DIR", "x")``; None for other paths."""
    if not path.startswith("~/"):
        return None
    folder, _, rest = path[2:].partition("/")
    var = LOGIN_DIR_VARS.get(folder)
    if var is None or not rest:
        return None
    return var, rest


def _folder(var: str, environ: Mapping[str, str] | None) -> str | None:
    value = (os.environ if environ is None else environ).get(var, "")
    return value if value.strip() else None


def login_path(
    path: str,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """The host file a ``~/...`` login path names, honouring the agent's variable.

    ``~/.claude/.credentials.json`` is ``$CLAUDE_CONFIG_DIR/.credentials.json``
    when that variable is set (``~/.codex/...`` likewise with ``CODEX_HOME``);
    any other path is only expanded. ``environ`` and ``home`` default to this
    process's.
    """
    split = _split(path)
    folder = _folder(split[0], environ) if split else None
    if split and folder:
        base = Path(folder)
        if home is not None and folder.startswith("~/"):
            base = home / folder[2:]
        return base.expanduser() / split[1]
    if home is not None and path.startswith("~/"):
        return home / path[2:]
    return Path(path).expanduser()


def login_display(path: str, *, environ: Mapping[str, str] | None = None) -> str:
    """How messages name a login path: ``$CLAUDE_CONFIG_DIR/.credentials.json``
    when the variable decides where it is, else the path as written."""
    split = _split(path)
    if split and _folder(split[0], environ):
        return f"${split[0]}/{split[1]}"
    return path

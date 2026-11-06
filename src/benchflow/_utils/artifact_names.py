"""Preserve supported agent log paths and encode only refused spellings."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def agent_log_name(agent: str) -> str:
    """Keep the pre-hardening log name whenever it stays inside its folder.

    Relative paths, dots, empty agent names and punctuation keep their exact
    old names. ReachablePath still owns no-follow traversal and writes; this
    mapper only prevents a command spelling from failing its relative-path
    admission. Names the filesystem could not store also use the fallback.

    The fallback's leading hyphen is a reserved namespace by construction:
    every hyphen in a legacy name was replaced by an underscore. No legacy
    log can therefore have this name, even if the agent spells the hash.
    ACP and native logs must call this same function, including on reconnect.
    """
    legacy = f"{agent.replace('-', '_')}.txt"
    path = Path(legacy)
    valid = not path.is_absolute() and ".." not in path.parts
    try:
        if os.name == "nt":
            from benchflow.rewards._safe_evidence import _windows_component

            for part in path.parts:
                _windows_component(part)
                valid = (
                    valid
                    and len(part.encode("utf-16-le", errors="surrogatepass")) <= 510
                )
        else:
            valid = valid and all(
                "\0" not in part and len(os.fsencode(part)) <= 255
                for part in path.parts
            )
    except (UnicodeError, ValueError):
        valid = False
    if valid:
        return legacy
    digest = hashlib.sha256(agent.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"-raw-abs-{digest}.txt"

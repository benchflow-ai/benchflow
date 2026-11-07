"""Literal path arguments for commands printed in diagnostics and hints."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def command_arg(value: object, placeholder: str) -> str:
    """Quote a path as one shell word, or show a placeholder if it cannot display.

    Relative paths beginning with a dash get ./ so they cannot become an
    option. Line breaks and controls cannot survive a single-line hint; never
    flatten them into a different path (D2b-C1, FF-CLI-C2, D2b-CONF7-2).
    This is plain text: render it literally, without markup or emoji parsing.
    """
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _secrets

    def render(fields):
        text = fields[0]
        if not text or not text.isprintable():
            return placeholder
        if text.startswith("-"):
            text = f"./{text}"
        return shlex.quote(text)

    return render_diagnostic(
        (str(value) if value is not None else "",), _secrets(), render
    )


def local_path_arg(value: object, placeholder: str) -> str:
    """Quote an explicitly local path for a command that also accepts URIs.

    Anchor every relative path with ./, without resolving it or interpreting
    a scheme. Shell quoting alone leaves hf: names in the viewer's remote
    namespace (D2b-CONF8-1). Keep this independent of supported URI schemes.
    """
    text = str(value) if value is not None else ""
    if text and not Path(text).is_absolute():
        text = f"./{text}"
    return command_arg(text, placeholder)


def looked_in(value: object) -> str:
    """``jobs/oracle (looked in /home/me/w2/jobs/oracle)``: a path and where it is.

    A relative path names a different place in another folder; the absolute
    one shows a user in the wrong folder that they are (DF-p1-first-run-9).
    An absolute path is shown once.
    """
    text = str(value) if value is not None else ""
    path = Path(text)
    if not text or path.is_absolute():
        return text
    try:
        where = os.path.normpath(Path.cwd() / path)
    except OSError:
        return text
    return f"{text} (looked in {where})"

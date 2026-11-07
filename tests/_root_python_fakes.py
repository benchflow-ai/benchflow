"""Fake sandboxes recognise root's Python helpers as BenchFlow sends them.

Since round 10 every root Python helper goes through
:mod:`benchflow._root_python`: ``root_python_command`` (argv text,
``python3 -I -S -B -c <prelude + script> args``, or the kit's interpreter)
and ``root_python_sh`` (a subshell choosing the interpreter in the
sandbox). A fake that answered "python3 -c ..." asks :func:`is_root_python`
and reads the helper's script and arguments with :func:`root_python_parts`.
"""

from __future__ import annotations

import re
import shlex

from benchflow._root_python import PRELUDE

_SH_MARK = 'exec "$bf_py" -I -S -B '
_ARGV = re.compile(r"(?:^|[\s;&|(])(\S*python3) -I -S -B (?=-c )")


def is_root_python(command: object) -> bool:
    """Whether *command* runs a root Python helper (either form), possibly
    after other shell text."""
    return isinstance(command, str) and (
        _SH_MARK in command or _ARGV.search(command) is not None
    )


def _words(command: str) -> list[str] | None:
    """The helper's words from ``-c`` on, up to the end of its subshell
    (sh form) or of the command."""
    if _SH_MARK in command:
        rest = command.split(_SH_MARK, 1)[1]
    else:
        found = _ARGV.search(command)
        if found is None:
            return None
        rest = command[found.end() :]
    lexer = shlex.shlex(rest, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    words = []
    try:
        for word in lexer:
            if word in (")", ";", "&&", "||", "|", "&"):
                break
            words.append(word)
    except ValueError:
        return None
    return words if words and words[0] == "-c" else None


def root_python_parts(command: str) -> tuple[str, list[str]] | None:
    """(script without the prelude, arguments) of the first helper in
    *command*, or None."""
    words = _words(command)
    if words is None:
        return None
    return words[1].replace(PRELUDE, "", 1), words[2:]


def python3_argv(command: str, python: str) -> list[str] | None:
    """The helper as argv for *python* (the test's interpreter), its flags
    and prelude kept, so a fake can run it as the sandbox would."""
    words = _words(command)
    return None if words is None else [python, "-I", "-S", "-B", *words]

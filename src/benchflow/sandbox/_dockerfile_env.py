"""The ``ENV`` values a Dockerfile's final stage sets, read on the host.

Modal sets some variables in every sandbox over the image's own (#1245), so
the Modal backend puts the image's values back, and reads them here from the
task's Dockerfile. A value a registry base image sets (``FROM <image>``) is not
seen: reading it would need the image.
"""

from __future__ import annotations

import re
import shlex

#: A heredoc opener in an instruction (``RUN <<EOF``, ``COPY <<-'EOF' /f``).
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z0-9_.-]+)\1")
#: ``$NAME`` or ``${NAME}`` in a value; ``${NAME:-word}`` and the other
#: modifier forms are not resolved.
_REFERENCE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_]\w*)\}|(?P<bare>[A-Za-z_]\w*))")


def _instructions(text: str) -> list[tuple[str, str]]:
    """``(KEYWORD, arguments)`` for each instruction of *text*, with line
    continuations joined and comments and heredoc bodies skipped."""
    out: list[tuple[str, str]] = []
    current = ""
    lines = iter(text.splitlines())
    for raw in lines:
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        whole = (current + line).strip()
        current = ""
        for match in _HEREDOC.finditer(whole):
            end = match.group(2)
            for body in lines:
                if body.strip() == end:
                    break
        keyword, _, arguments = whole.partition(" ")
        out.append((keyword.upper(), arguments.strip()))
    if current.strip():
        keyword, _, arguments = current.strip().partition(" ")
        out.append((keyword.upper(), arguments.strip()))
    return out


def _words(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _resolve(value: str, seen: dict[str, str | None]) -> str | None:
    """*value* with each ``$NAME`` and ``${NAME}`` replaced by the value set so
    far (an earlier ``ENV``, or an ``ARG``'s default: BenchFlow passes no
    build arguments), else empty. None when it holds another ``${...}`` form
    or names a value that is itself unknown."""
    names = [m["braced"] or m["bare"] for m in _REFERENCE.finditer(value)]
    if any(name in seen and seen[name] is None for name in names):
        return None
    resolved = _REFERENCE.sub(lambda m: seen.get(m["braced"] or m["bare"]) or "", value)
    return None if "${" in resolved else resolved


def _env_pairs(arguments: str) -> list[tuple[str, str]]:
    """One ``ENV``'s ``(name, raw value)`` pairs: ``K=V ...``, or ``K V``."""
    words = _words(arguments)
    if words and "=" in words[0]:
        return [(k, v) for k, _, v in (w.partition("=") for w in words if "=" in w)]
    name, _, rest = arguments.partition(" ")
    return [(name, " ".join(_words(rest)))] if name else []


def final_stage_env(text: str) -> dict[str, str | None]:
    """The ``ENV`` values the final stage of the Dockerfile *text* sets,
    through the earlier stages it is built ``FROM``.

    References are resolved as Docker resolves them (:func:`_resolve`: an
    ``ENV`` value over a stage's ``ARG`` of the same name); a value that
    cannot be is None.
    """
    stages: list[tuple[str, str | None, list[tuple[str, str]]]] = []
    global_args: dict[str, str] = {}
    for keyword, arguments in _instructions(text):
        if keyword == "FROM":
            words = [w for w in arguments.split() if not w.startswith("--")]
            base = words[0].lower() if words else ""
            alias = (
                words[2].lower()
                if len(words) >= 3 and words[1].upper() == "AS"
                else None
            )
            stages.append((base, alias, []))
        elif not stages:
            if keyword == "ARG":
                name, _, default = arguments.partition("=")
                global_args[name.strip()] = " ".join(_words(default))
        elif keyword in ("ENV", "ARG"):
            stages[-1][2].append((keyword, arguments))
    if not stages:
        return {}
    # The final stage, then the earlier stage each is built FROM, if any.
    chain = [len(stages) - 1]
    while True:
        base = stages[chain[-1]][0]
        earlier = [i for i in range(chain[-1]) if stages[i][1] == base]
        if not earlier:
            break
        chain.append(earlier[-1])
    values: dict[str, str | None] = {}
    for index in reversed(chain):
        args: dict[str, str | None] = {}
        for keyword, arguments in stages[index][2]:
            seen = {**args, **values}
            if keyword == "ARG":
                name, eq, default = arguments.partition("=")
                name = name.strip()
                args[name] = (
                    _resolve(" ".join(_words(default)), seen)
                    if eq
                    else global_args.get(name, "")
                )
            else:
                values.update({k: _resolve(v, seen) for k, v in _env_pairs(arguments)})
    return values

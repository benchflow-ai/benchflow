"""The ``ENV`` values a Dockerfile's final stage sets, read on the host.

Modal sets some variables in every sandbox over the image's own (#1245), so
the Modal backend puts the image's values back, and reads them here from the
task's Dockerfile. A value a registry base image sets (``FROM <image>``) is not
seen: reading it would need the image.

Where this reader cannot tell what Docker would make of a value, the value is
None (and a Dockerfile it cannot read at all gives None), so the backend
keeps Modal's value for that name and never gives a wrong one.
"""

from __future__ import annotations

import re

#: A heredoc opener in an instruction (``RUN <<EOF``, ``COPY <<-'EOF' /f``).
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z0-9_.-]+)\1")
#: A parser directive at the top of a Dockerfile (``# escape=`` ...).
_DIRECTIVE = re.compile(r"#\s*([A-Za-z]+)\s*=\s*(.*)")
#: A variable name, as ``$NAME`` and ``${NAME}`` take it.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _Unreadable(ValueError):
    """An instruction this reader cannot follow as Docker would."""


def _escape_character(text: str) -> str:
    """The escape character the parser directives name (Docker's default ``\\``)."""
    for raw in text.splitlines():
        match = _DIRECTIVE.fullmatch(raw.strip())
        if match is None:
            break  # the directives end at the first other line
        if match.group(1).lower() == "escape":
            return match.group(2).strip()
    return "\\"


def _instructions(text: str) -> list[tuple[str, str]]:
    """``(KEYWORD, arguments)`` for each instruction of *text*: continuation
    lines joined, and comment and empty lines skipped, inside a continuation
    too (as Docker does), heredoc bodies skipped."""
    out: list[tuple[str, str]] = []
    current = ""
    lines = iter(text.splitlines())
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
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
        out.append(_keyword(whole))
    if current.strip():
        out.append(_keyword(current.strip()))
    return out


def _keyword(instruction: str) -> tuple[str, str]:
    """An instruction's keyword and arguments, split at any whitespace."""
    parts = instruction.split(None, 1)
    return parts[0].upper(), parts[1].strip() if len(parts) > 1 else ""


def _words(text: str) -> list[str]:
    """*text* split at whitespace outside quotes and escapes; each word keeps
    its quotes and escapes for :func:`_expand`."""
    words: list[str] = []
    word: list[str] = []
    started = False
    quote = None
    i = 0
    while i < len(text):
        c = text[i]
        if quote is None and c.isspace():
            if started:
                words.append("".join(word))
                word, started = [], False
            i += 1
            continue
        if c == "\\" and quote != "'" and i + 1 < len(text):
            word.append(text[i : i + 2])
            started = True
            i += 2
            continue
        if c in "'\"" and quote in (None, c):
            quote = None if quote == c else c
        word.append(c)
        started = True
        i += 1
    if quote is not None:
        raise _Unreadable("an unterminated quote")
    if started:
        words.append("".join(word))
    return words


def _expand(word: str, seen: dict[str, str | None]) -> str | None:
    """*word* as Docker gives an ``ENV`` value or ``ARG`` default: quotes
    removed, ``\\`` escapes resolved (``\\$`` is a literal ``$``), nothing
    expanded inside single quotes, and each ``$NAME`` and ``${NAME}`` replaced
    by the value set so far (an earlier ``ENV``, or an ``ARG``'s: BenchFlow
    passes no build arguments), else empty. None for what this reader does
    not follow: ``${NAME:-word}`` and the other modifier forms, ``$1``-style
    names, or a name whose own value is unknown."""
    out: list[str] = []
    quote = None
    i = 0
    while i < len(word):
        c = word[i]
        if c == "\\" and quote != "'" and i + 1 < len(word):
            following = word[i + 1]
            if quote == '"' and following not in '"$\\':
                out.append(c)  # kept inside double quotes, as Docker keeps it
                i += 1
                continue
            out.append(following)
            i += 2
            continue
        if c in "'\"" and quote in (None, c):
            quote = None if quote == c else c
            i += 1
            continue
        if c == "$" and quote != "'":
            if word.startswith("${", i):
                end = word.find("}", i)
                name = word[i + 2 : end] if end > 0 else ""
                if not _NAME.fullmatch(name):
                    return None
                i = end + 1
            else:
                match = _NAME.match(word, i + 1)
                if match is None:
                    if i + 1 < len(word) and not word[i + 1].isspace():
                        return None
                    out.append(c)
                    i += 1
                    continue
                name = match.group(0)
                i = match.end()
            if name in seen and seen[name] is None:
                return None
            out.append(seen.get(name) or "")
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _pairs(arguments: str) -> list[tuple[str, str | None]]:
    """``(name, raw value)`` for each ``NAME=value`` word; a bare ``NAME``
    (an ``ARG`` without a default) has None."""
    out: list[tuple[str, str | None]] = []
    for word in _words(arguments):
        name, eq, value = word.partition("=")
        if not _NAME.fullmatch(name):
            raise _Unreadable(f"the name {name!r}")
        out.append((name, value if eq else None))
    return out


def _env_pairs(arguments: str) -> list[tuple[str, str]]:
    """One ``ENV``'s ``(name, raw value)`` pairs: ``K=V ...``, or the legacy
    ``K V``, whose value is the rest of the line."""
    words = _words(arguments)
    if words and "=" in words[0]:
        pairs = _pairs(arguments)
        if any(value is None for _name, value in pairs):
            raise _Unreadable("an ENV mixing its two forms")
        return [(name, value or "") for name, value in pairs]
    parts = arguments.split(None, 1)
    if not parts:
        return []
    if not _NAME.fullmatch(parts[0]):
        raise _Unreadable(f"the name {parts[0]!r}")
    return [(parts[0], parts[1].strip() if len(parts) > 1 else "")]


def final_stage_env(text: str) -> dict[str, str | None] | None:
    """The ``ENV`` values the final stage of the Dockerfile *text* sets,
    through the earlier stages it is built ``FROM``, or None when this reader
    cannot follow the file (another escape character, an instruction it
    cannot parse).

    References are resolved as Docker resolves them (:func:`_expand`): an
    ``ENV`` value over an ``ARG`` of the same name, and a stage's ``ARG``
    values carried into a stage built ``FROM`` it; an ``ARG`` re-declared
    without a default keeps the value it carries, else the global one (an
    ``ARG`` before the first ``FROM``), else empty. A value it cannot resolve
    is None.
    """
    if _escape_character(text) != "\\":
        return None
    try:
        return _final_stage_env(text)
    except _Unreadable:
        return None


def _final_stage_env(text: str) -> dict[str, str | None]:
    stages: list[tuple[str, str | None, list[tuple[str, str]]]] = []
    global_args: dict[str, str | None] = {}
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
                for name, raw in _pairs(arguments):
                    global_args[name] = (
                        "" if raw is None else _expand(raw, dict(global_args))
                    )
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
    # A stage's ARG values carry into the stages built FROM it.
    args: dict[str, str | None] = {}
    for index in reversed(chain):
        for keyword, arguments in stages[index][2]:
            seen = {**args, **values}
            if keyword == "ARG":
                for name, raw in _pairs(arguments):
                    if raw is not None:
                        args[name] = _expand(raw, seen)
                    elif name not in args:
                        args[name] = global_args.get(name, "")
            else:
                values.update({k: _expand(v, seen) for k, v in _env_pairs(arguments)})
    return values

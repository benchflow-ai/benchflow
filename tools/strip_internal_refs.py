"""Strip internal tracker keys and review tags from what BenchFlow ships.

Usage: python tools/strip_internal_refs.py [--write] [PATH ...]

Default paths: src/benchflow, CHANGELOG.md, README.md. In Python files only
comments and docstrings change; code, and strings that are not docstrings, are
reported for a person to fix. In other text files every line is cleaned.

A reference is a private issue key (EN-123, BEN3-45, BD-6, ENG-7) or an
internal review tag (PY8-D123-..., *-CODEX-*, *-OPUS-*, *-SOL-*, *-MUSE-*),
the forms tools/release_leak_gate.py refuses. In a parenthetical such as
"(#1136, PY8-D273)" only the internal items go; a group left empty goes with
the space before it. A reference anywhere else is reported, not rewritten.
Exits 1 when anything is left to fix by hand.
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

REF = re.compile(
    r"\b(?:EN|BEN3|BD|ENG)-\d+\b"
    r"|\bPY\d+-D\d+(?:-[A-Za-z0-9-]+)?"
    r"|\b(?:[A-Z0-9]+-)*(?:CODEX|OPUS|SOL|MUSE)-[A-Z0-9-]+"
)
# One item of a parenthetical: a reference, optionally with the finding
# numbers or words that qualify it ("PY8-D908-PT-integ 1", "V2-CODEX-1 r2").
_QUALIFIER = r"(?:r?\d+[a-z]?|[A-Za-z]+-?\d+|\(\d+\)|and\s+\d+|\d+\.\.\d+)"
_ITEM_REF = re.compile(
    rf"^\s*(?:guards\s+|see\s+)?(?:{REF.pattern})(?:\.\.\d+|/D\d+)*"
    rf"(?:\s+{_QUALIFIER})*\s*$",
    re.IGNORECASE,
)
# A parenthetical, with one level of "(2)" inside it.
_GROUP = re.compile(r"(\s?)\(((?:[^()\n]|\([^()\n]*\))*)\)")
_LABEL = re.compile(r"(?m)^([ \t]*(?:#+[ \t]*)?)\(((?:[^()\n]|\([^()\n]*\))*)\):[ \t]*")
# "PY8-D21: ..." or "Guards ENG-160: ..." opening a comment or docstring line.
_LEADING = re.compile(
    rf"(?m)^([ \t]*(?:#+[ \t]*|\"\"\"|)?)(?:Guards[ \t]+)?(?:{REF.pattern})"
    rf"(?:[ \t]*\(\d+\))?:[ \t]*"
)


def clean(text: str) -> str:
    """*text* with every parenthetical internal reference removed."""

    def group(match: re.Match[str]) -> str:
        space, inner = match.group(1), match.group(2)
        if not REF.search(inner):
            return match.group(0)
        items = re.split(r"\s*[,;]\s*", inner)
        kept = [item for item in items if not _ITEM_REF.match(item)]
        if not kept:
            return ""
        if len(kept) == len(items):
            return match.group(0)
        return f"{space}({', '.join(kept)})"

    # "(PY8-D896-V3): ..." opening a line labels it: the label and its colon go.
    text = _LABEL.sub(
        lambda m: m.group(1) if _all_refs(m.group(2)) else m.group(0), text
    )
    text = _GROUP.sub(group, text)
    return _LEADING.sub(lambda m: m.group(1), text)


def _all_refs(inner: str) -> bool:
    return all(_ITEM_REF.match(item) for item in re.split(r"\s*[,;]\s*", inner))


def _spans(source: str) -> list[tuple[int, int]]:
    """Offsets of the comments and docstrings of a Python source."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(row: int, col: int) -> int:
        return starts[row - 1] + col

    spans = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    previous = None
    for index, token in enumerate(tokens):
        if token.type == tokenize.COMMENT:
            spans.append((offset(*token.start), offset(*token.end)))
            continue
        if token.type == tokenize.STRING:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            statement = previous is None or (
                previous.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
                or (previous.type == tokenize.OP and previous.string == ":")
            )
            if (
                statement
                and following is not None
                and following.type
                in (
                    tokenize.NEWLINE,
                    tokenize.NL,
                    tokenize.COMMENT,
                    tokenize.ENDMARKER,
                )
            ):
                spans.append((offset(*token.start), offset(*token.end)))
        if token.type not in (tokenize.NL, tokenize.COMMENT):
            previous = token
    return spans


def clean_python(source: str) -> tuple[str, list[tuple[int, str]]]:
    """The source with its comments and docstrings cleaned, and every
    reference left anywhere, by line."""
    out, last = [], 0
    for start, end in _spans(source):
        out.append(source[last:start])
        out.append(clean(source[start:end]))
        last = end
    out.append(source[last:])
    cleaned = "".join(out)
    left = [
        (number, line.strip())
        for number, line in enumerate(cleaned.splitlines(), 1)
        if REF.search(line)
    ]
    return cleaned, left


def clean_text(text: str) -> tuple[str, list[tuple[int, str]]]:
    cleaned = clean(text)
    left = [
        (number, line.strip())
        for number, line in enumerate(cleaned.splitlines(), 1)
        if REF.search(line)
    ]
    return cleaned, left


def files(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found += sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix
                in {
                    ".py",
                    ".md",
                    ".txt",
                    ".toml",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".sh",
                    ".j2",
                    ".html",
                    ".css",
                    ".js",
                }
            )
    return found


def main(argv: list[str]) -> int:
    write = "--write" in argv
    paths = [Path(a) for a in argv if a != "--write"] or [
        Path("src/benchflow"),
        Path("CHANGELOG.md"),
        Path("README.md"),
    ]
    changed = left_total = 0
    for path in files(paths):
        text = path.read_text()
        cleaned, left = (clean_python if path.suffix == ".py" else clean_text)(text)
        if cleaned != text:
            changed += 1
            if write:
                path.write_text(cleaned)
        for number, line in left:
            left_total += 1
            print(f"{path}:{number}: {line[:160]}")
    print(
        f"{'cleaned' if write else 'would clean'} {changed} file(s); "
        f"{left_total} reference(s) left to fix by hand",
        file=sys.stderr,
    )
    return 1 if left_total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Split provider-shaped string literals in Python source, so the value is built at runtime.

    python tools/keyshape_fix.py FILE.py...

Each string literal whose source text matches one of ``tools/ghpp_patterns.py``'s patterns is split inside the
match, two characters in and then every 8 (``"sk" + "_live_..."``), so no piece is 10 characters of a shape. The
pieces keep the literal's prefix and quotes, and the literal's group of implicitly joined strings is wrapped in
parentheses, so operator precedence can't change. Refused, and listed as MANUAL: f-strings, docstrings, comments
and any split whose pieces don't evaluate back to the literal. Each changed module must fold back to the AST it
had (``tools/foldproof.py``): the runtime values are the same. Prints one row per split literal and per MANUAL
place; run ``ruff format --force-exclude`` on the files afterwards (an explicit path ignores ruff's excludes).
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.foldproof import folded
from tools.ghpp_patterns import RX

_SKIP = {tokenize.NL, tokenize.COMMENT}
_FS = getattr(tokenize, "FSTRING_START", -1)
_FE = getattr(tokenize, "FSTRING_END", -1)


def _shaped(text: str, needles: tuple[str, ...]) -> bool:
    return any(rx.search(text) for rx in RX.values()) or any(
        nd in text for nd in needles
    )


def docstring_rows(tree: ast.AST) -> set[int]:
    rows: set[int] = set()
    for n in ast.walk(tree):
        if (
            isinstance(
                n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and n.body
        ):
            first = n.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                rows.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return rows


def split_literal(text: str, needles: tuple[str, ...] = ()) -> str | None:
    """The literal's source text split inside every match, or ``""`` when nothing matches, or None when it can't
    be done safely (an escape would be cut, or a piece still matches)."""
    i = 0
    while i < len(text) and text[i] in "rRbBuU":
        i += 1
    prefix, rest = text[:i], text[i:]
    q = rest[:3] if rest[:3] in ('"""', "'''") else rest[:1]
    body = rest[len(q) : len(rest) - len(q)]
    spans = [(m.start(), m.end()) for rx in RX.values() for m in rx.finditer(body)]
    for nd in needles:
        k = body.find(nd)
        while k >= 0:
            spans.append((k, k + len(nd)))
            k = body.find(nd, k + 1)
    cuts = sorted({c for a, e in spans for c in range(a + 2, e, 8)})
    if not cuts:
        return ""
    pieces, last = [], 0
    for c in cuts:
        k, n = c - 1, 0
        while k >= 0 and body[k] == "\\":
            n, k = n + 1, k - 1
        if n % 2:
            return None
        pieces.append(body[last:c])
        last = c
    pieces.append(body[last:])
    parts = [f"{prefix}{q}{p}{q}" for p in pieces]
    try:
        values = [ast.literal_eval(p) for p in parts]
        if values[0][:0].join(values) != ast.literal_eval(text):
            return None
    except (ValueError, SyntaxError):
        return None
    if any(_shaped(p, needles) for p in pieces):
        return None
    return " + ".join(parts)


def _group_bounds(toks: list[tokenize.TokenInfo], i: int) -> tuple[int, int]:
    """The first and last token of the implicitly joined string group around token ``i``."""
    a = i
    while a > 0:
        j = a - 1
        while j >= 0 and toks[j].type in _SKIP:
            j -= 1
        if j >= 0 and toks[j].type == tokenize.STRING:
            a = j
        elif j >= 0 and toks[j].type == _FE:
            depth = 0
            while j >= 0:
                depth += toks[j].type == _FE
                depth -= toks[j].type == _FS
                if depth == 0:
                    break
                j -= 1
            a = j
        else:
            break
    b = i
    while True:
        j = b + 1
        while j < len(toks) and toks[j].type in _SKIP:
            j += 1
        if j < len(toks) and toks[j].type == tokenize.STRING:
            b = j
        elif j < len(toks) and toks[j].type == _FS:
            depth = 0
            while j < len(toks):
                depth += toks[j].type == _FS
                depth -= toks[j].type == _FE
                if depth == 0:
                    break
                j += 1
            b = j
        else:
            break
    return a, b


def fix_source(
    src: str, rel: str = "<source>", needles: tuple[str, ...] = ()
) -> tuple[str, list[str], list[str]]:
    """``(new source, SPLIT rows, MANUAL rows)``. The new source folds to the old one's AST."""
    tree = ast.parse(src)
    docs = docstring_rows(tree)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    offs = [0]
    for line in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))

    def pos(rc: tuple[int, int]) -> int:
        return offs[rc[0] - 1] + rc[1]

    done: list[str] = []
    manual: list[str] = []
    edits: list[tuple[int, int, str]] = []
    depth = 0
    for i, t in enumerate(toks):
        if t.type == _FS:
            depth += 1
        elif t.type == _FE:
            depth -= 1
        elif depth and _shaped(t.string, needles):
            manual.append(f"MANUAL\t{rel}\t{t.start[0]}\tf-string")
        if t.type == tokenize.COMMENT and _shaped(t.string, needles):
            manual.append(f"MANUAL\t{rel}\t{t.start[0]}\tcomment")
        if t.type != tokenize.STRING or not _shaped(t.string, needles):
            continue
        if t.start[0] in docs:
            manual.append(f"MANUAL\t{rel}\t{t.start[0]}\tdocstring")
            continue
        new = split_literal(t.string, needles)
        if not new:
            manual.append(f"MANUAL\t{rel}\t{t.start[0]}\tsplit refused")
            continue
        a, b = _group_bounds(toks, i)
        edits.append((pos(t.start), pos(t.end), new))
        edits.append((pos(toks[a].start), pos(toks[a].start), "("))
        edits.append((pos(toks[b].end), pos(toks[b].end), ")"))
        done.append(f"SPLIT\t{rel}\t{t.start[0]}")
    if not edits:
        return src, done, manual
    unique, seen = [], set()
    for e in edits:
        if e[2] in "()" and (e[0], e[2]) in seen:
            continue
        seen.add((e[0], e[2]))
        unique.append(e)
    out = src
    # Right to left; at one position the literal's replacement goes before the "(" inserted there.
    for s, e, text in sorted(unique, key=lambda x: (x[0], x[1] > x[0]), reverse=True):
        out = out[:s] + text + out[e:]
    compile(out, rel, "exec")
    if folded(out) != folded(src):
        raise ValueError(f"the split would change a value: {rel}")
    return out, done, manual


def split_source(src: str) -> str:
    """``src`` with every provider-shaped literal split (see :func:`fix_source`)."""
    return fix_source(src)[0]


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print((__doc__ or "").split("\n\n")[1].strip(), file=sys.stderr)
        return 2
    total = refused = 0
    for name in paths:
        path = Path(name)
        try:
            new, done, manual = fix_source(path.read_text(), name)
        except (SyntaxError, ValueError) as exc:
            done, manual, new = (
                [],
                [f"MANUAL\t{name}\t0\trefused: {type(exc).__name__}"],
                None,
            )
        if new is not None and done:
            path.write_text(new)
        total += len(done)
        refused += len(manual)
        for row in done + manual:
            print(row)
    print(f"keyshape_fix: split {total}, manual {refused}", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

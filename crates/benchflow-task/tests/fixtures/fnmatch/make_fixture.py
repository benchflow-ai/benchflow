"""Write python-fnmatchcase.jsonl: glob matches as BenchFlow 0.8 decides them.

0.8 selects tasks with --include/--exclude through
benchflow.evaluation.task_name_selected, which calls fnmatch.fnmatchcase
(case-sensitive on every platform). Each line is a task name, a pattern and
what fnmatch.fnmatchcase(name, pattern) returned.

The cases are a hand-picked list, then seeded random names and patterns
over a small alphabet full of glob syntax (`*?[]!^-\\`, `/`, a few letters,
and now and then a newline, `.`, `&~|` or a non-ASCII letter): patterns of
random characters, wildcards and bracket expressions (negated with `!` or
`^`, ranges, backwards ranges, unclosed), and patterns made from a name by
swapping some of its characters for wildcards or classes, so about a third
of the cases match. Duplicates are dropped; the seed makes a rerun write
the same file.

Run it with 0.8's Python (from any directory):

    uv run --project /path/to/benchflow-0.8.0 python -I make_fixture.py

It rewrites python-fnmatchcase.jsonl next to this script; `--check`
compares instead and exits 1 on a difference.
"""

from __future__ import annotations

import fnmatch
import json
import random
import sys
from pathlib import Path

SEED = 20261011
COUNT = 20_000
OUT = Path(__file__).resolve().parent / "python-fnmatchcase.jsonl"

PICKED = [
    ("fix-a", "fix-{a,b}"), ("^x", "[^a]x"), ("bx", "[^a]x"),
    ("task[1", "task[1"), ("a\\b", "a\\b"), ("foo[1]", "foo[1]"),
    ("y", "[z-a]"), ("y", "[!z-a]"), ("bx", "[!a]x"), ("a/b", "a*b"),
    ("A", "a"), ("a", "A"), ("x", "**"), ("ab", "a**"), ("a\nb", "a*b"),
    ("a\n", "a?"), ("a\n", "a"), ("é", "?"), ("é", "[é]"), ("é", "[!a]"),
    ("]", "[]]"), ("]", "[!]]"), ("-", "[a-]"), ("-", "[-a]"),
    ("!", "[!]"), ("!", "[!!]"), ("[", "[[]"), ("a]", "[!]a]"),
    ("&", "[&&]"), ("~", "[~~]"), ("|", "[||]"), ("b", "[a&&b]"),
    ("\\", "[\\]"), ("\\", "\\"), ("a\\", "a\\"), ("x.y", "*.*"),
    ("", ""), ("", "*"), ("", "?"), ("", "[]"),
]
NAME_CHARS = "ab z!-/[\\]^".replace(" ", "")
RARE = ["\n", ".", "&", "~", "|", "é", "ß", "A"]


def name_char(rng: random.Random) -> str:
    return rng.choice(RARE) if rng.random() < 0.04 else rng.choice(NAME_CHARS)


def random_name(rng: random.Random) -> str:
    return "".join(name_char(rng) for _ in range(rng.randrange(7)))


def bracket(rng: random.Random) -> str:
    body = rng.choice(["", "", "!", "^", "]", "!]", "^]"])
    for _ in range(rng.randrange(4)):
        r = rng.random()
        if r < 0.3:
            a, b = name_char(rng), name_char(rng)
            body += f"{a}-{b}"
        elif r < 0.4:
            body += rng.choice(["&&", "~~", "||", "--", "-"])
        else:
            body += name_char(rng)
    return "[" + body + ("" if rng.random() < 0.1 else "]")


def random_pattern(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randrange(7)):
        r = rng.random()
        if r < 0.15:
            parts.append("*")
        elif r < 0.3:
            parts.append("?")
        elif r < 0.5:
            parts.append(bracket(rng))
        else:
            parts.append(name_char(rng))
    return "".join(parts)


def pattern_from(name: str, rng: random.Random) -> str:
    out = []
    for c in name:
        r = rng.random()
        if r < 0.15:
            out.append("?")
        elif r < 0.25:
            out.append("*")
        elif r < 0.35:
            out.append("[" + rng.choice(["", "!", "^"]) + c + name_char(rng) + "]")
        else:
            out.append(c)
    if rng.random() < 0.2:
        out.insert(rng.randrange(len(out) + 1), "*")
    return "".join(out)


def main() -> int:
    rng = random.Random(SEED)
    seen: set[tuple[str, str]] = set()
    cases = []

    def add(name: str, pattern: str) -> None:
        if (name, pattern) not in seen:
            seen.add((name, pattern))
            cases.append((name, pattern))

    for name, pattern in PICKED:
        add(name, pattern)
    while len(cases) < COUNT:
        name = random_name(rng)
        if rng.random() < 0.5:
            add(name, random_pattern(rng))
        else:
            add(name, pattern_from(name, rng))
    lines = [
        json.dumps({"name": n, "pattern": p, "match": fnmatch.fnmatchcase(n, p)})
        for n, p in cases
    ]
    text = "\n".join(lines) + "\n"
    matched = sum(fnmatch.fnmatchcase(n, p) for n, p in cases)
    if "--check" in sys.argv[1:]:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'same' if same else 'DIFFERENT'} ({len(cases)} cases)")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases ({matched} match) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

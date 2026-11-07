"""The first-run journeys' pages, and how the docs tests read them.

A new user copies commands from the README and five journeys: install and a
first score, a model run, a new task, several tasks on Daytona, and an export
for training. ``test_docs_journeys.py`` checks these pages without Docker;
``test_docs_run_journeys.py`` runs their commands in CI.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# README first, then the pages each journey sends the reader to.
JOURNEY_PAGES = (
    "README.md",
    "docs/start/quick-start.md",
    "docs/start/install.md",
    "docs/evaluate/agents.md",
    "docs/evaluate/read-results.md",
    "docs/tasks/write-a-task.md",
    "docs/audit/check-a-benchmark.md",
    "docs/scale/daytona.md",
    "docs/train/export.md",
)

SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh"}
_FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n`]*)$")


@dataclass(frozen=True)
class Block:
    """One fenced code block: its language, text, first line and heading."""

    lang: str
    text: str
    line: int
    heading: str


def read(page: str) -> str:
    return (REPO / page).read_text(encoding="utf-8")


def blocks(text: str) -> list[Block]:
    """Every fenced code block, with the nearest heading above it."""
    out: list[Block] = []
    heading = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            if re.match(r"#{1,6} ", lines[i]):
                heading = lines[i].lstrip("#").strip()
            i += 1
            continue
        fence, lang = m.group("fence"), m.group("info").strip().split(" ")[0].lower()
        start, body = i + 1, []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            body.append(lines[i])
            i += 1
        out.append(Block(lang, "\n".join(body), start, heading))
        i += 1
    return out


def prose(text: str, *, inline_code: bool = False) -> str:
    """The page without fenced code blocks, and without inline code spans
    unless ``inline_code`` (headings keep their code text in their anchor)."""
    kept: list[str] = []
    fence = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group("fence")
            continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            continue
        kept.append(line)
    joined = "\n".join(kept)
    return joined if inline_code else re.sub(r"`[^`\n]*`", "", joined)


def inline_commands(text: str) -> list[str]:
    """Inline code spans in the prose that hold a ``bench`` command."""
    spans = re.findall(r"`([^`\n]+)`", prose(text, inline_code=True))
    return [s for s in spans if re.match(r"(bench|benchflow) ", s)]


def shell_lines(block: Block) -> list[str]:
    """A shell block's commands: continuations joined, comments and output dropped."""
    joined = re.sub(r"\\\n\s*", " ", block.text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def bench_argvs(line: str) -> list[list[str]]:
    """The ``bench``/``benchflow`` invocations in one shell line, as argv lists
    (without the program name). Environment assignments before the program,
    pipes, ``&&``, ``;`` and trailing comments are handled; anything else is
    kept as a literal token, as the shell would pass it."""
    lex = shlex.shlex(line, posix=True, punctuation_chars=";&|")
    lex.whitespace_split = True
    lex.commenters = "#"
    try:
        tokens = list(lex)
    except ValueError:
        return []
    argvs, current = [], []
    skip_target = False
    for tok in [*tokens, ";"]:
        if skip_target:
            skip_target = False
            continue
        # `<job>` is a placeholder the reader fills in, not a redirection.
        redirect = (
            None
            if re.fullmatch(r"<[^<>\s]+>", tok)
            else re.match(r"^\d*(>>?|<)(&\d+)?(.*)$", tok)
        )
        if redirect and tok not in {";", "&&", "||", "|", "&"}:
            # `> file`, `2>&1`, `>file`: a redirection, not an argument.
            skip_target = not redirect.group(2) and not redirect.group(3)
            continue
        if tok in {";", "&&", "||", "|", "&"}:
            while current and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", current[0]):
                current = current[1:]
            if current and current[0] in {"bench", "benchflow"}:
                argvs.append(current[1:])
            current = []
        else:
            current.append(tok)
    return argvs

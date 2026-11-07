"""The first-run journeys' commands, code and links are real.

The README and the journeys' pages (``tests/docs_pages.py``) are what a new
user copies from. These checks need no Docker and no network:

- every ``bench`` command in their shell blocks and inline code names a real
  command, real options with the values they need, and the arguments it
  requires;
- no journey page uses a command the CLI hides or marks deprecated;
- every ``python`` block compiles and names real ``benchflow`` attributes;
- every relative link and ``#anchor`` in ``README.md`` and ``docs/`` resolves;
- the README names the version this package is;
- the journey pages under ``docs/`` are safe for the website's MDX build.

``test_docs_run_journeys.py`` runs the commands themselves in the docs CI job.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import click
import pytest
import typer.main

import benchflow as bf
from benchflow.cli.main import app
from tests.docs_pages import (
    JOURNEY_PAGES,
    REPO,
    SHELL_LANGS,
    bench_argvs,
    blocks,
    inline_commands,
    prose,
    read,
    shell_lines,
)

ROOT = typer.main.get_command(app)


def _commands(page: str) -> list[tuple[int, list[str]]]:
    """Every bench command on the page: in shell blocks, and in inline code
    in the prose (line 0), where an example is as easy to copy."""
    out = [
        (0, argv)
        for span in inline_commands(read(page))
        for argv in bench_argvs(span)
        if argv and re.fullmatch(r"[a-z][a-z-]*|-{1,2}[a-z][a-z-]*", argv[0])
    ]
    for block in blocks(read(page)):
        if block.lang not in SHELL_LANGS:
            continue
        for line in shell_lines(block):
            if block.lang == "console":
                if not line.startswith("$ "):
                    continue
                line = line[2:]
            for argv in bench_argvs(line):
                out.append((block.line, argv))
    return out


def _options(command: click.Command) -> dict[str, click.Option | None]:
    found: dict[str, click.Option | None] = {"--help": None}
    for param in command.params:
        if isinstance(param, click.Option):
            for name in (*param.opts, *param.secondary_opts):
                found[name] = param
    return found


def _takes_value(option: click.Option | None) -> bool:
    return option is not None and not option.is_flag and not option.count


def _resolve(
    argv: list[str],
) -> tuple[click.Command | None, list[str], list[str], list[click.Command]]:
    """Walk ``argv`` down the command tree, skipping group-level flags."""
    command: click.Command = ROOT
    path: list[str] = []
    chain = [ROOT]
    rest = list(argv)
    while isinstance(command, click.Group) and rest:
        if rest[0].startswith("-"):
            option = _options(command).get(rest[0])
            if rest[0] in _options(command) and not _takes_value(option):
                rest.pop(0)
                continue
            break
        ctx = click.Context(command, info_name="bench")
        sub = command.get_command(ctx, rest[0])
        if sub is None:
            return None, [*path, rest[0]], rest, chain
        command = sub
        path.append(rest.pop(0))
        chain.append(command)
    return command, path, rest, chain


def _problems(argv: list[str], *, inline: bool = False) -> list[str]:
    """What is wrong with ``bench <argv>``. An inline mention in the prose
    (``inline``) often leaves out arguments and values, so for it only an
    unknown command or option counts."""
    command, path, rest, _ = _resolve(argv)
    name = "bench " + " ".join(path)
    if command is None:
        return [f"there is no command `{name}`"]
    options = _options(command)
    positionals: list[str] = []
    problems: list[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--":
            positionals += rest[i + 1 :]
            break
        if tok.startswith("--"):
            flag, eq, _ = tok.partition("=")
            if flag not in options:
                problems.append(f"`{flag}` is not an option of `{name}`")
            elif _takes_value(options[flag]) and not eq:
                if i + 1 >= len(rest) and not inline:
                    problems.append(f"`{flag}` of `{name}` needs a value")
                i += 1
            i += 1
            continue
        if tok.startswith("-") and len(tok) > 1 and not re.match(r"^-\d", tok):
            if tok in options:
                if _takes_value(options[tok]):
                    if i + 1 >= len(rest) and not inline:
                        problems.append(f"`{tok}` of `{name}` needs a value")
                    i += 1
            elif not _takes_value(options.get(tok[:2])):
                problems.append(f"`{tok}` is not an option of `{name}`")
            i += 1
            continue
        positionals.append(tok)
        i += 1
    if inline or "--help" in rest or isinstance(command, click.Group):
        return problems
    arguments = [p for p in command.params if isinstance(p, click.Argument)]
    required = sum(1 for a in arguments if a.required)
    if len(positionals) < required:
        problems.append(
            f"`{name}` needs {required} argument(s), got {len(positionals)}"
        )
    if not any(a.nargs == -1 for a in arguments):
        most = sum(max(a.nargs, 1) for a in arguments)
        if len(positionals) > most:
            problems.append(
                f"`{name}` takes at most {most} argument(s), got {positionals!r}"
            )
    return problems


@pytest.mark.parametrize("page", JOURNEY_PAGES)
def test_every_bench_command_parses_against_the_cli(page: str) -> None:
    commands = _commands(page)
    assert commands or page in {"docs/start/install.md"}, f"{page} has no bench command"
    bad = [
        f"{page}:{line}: bench {' '.join(argv)}: {p}"
        for line, argv in commands
        for p in _problems(argv, inline=line == 0)
    ]
    assert not bad, "\n".join(bad)


@lru_cache
def _retired() -> dict[tuple[str, ...], str]:
    """Command paths the CLI hides or marks deprecated."""
    found: dict[tuple[str, ...], str] = {}

    def walk(command: click.Command, path: tuple[str, ...]) -> None:
        if path and (command.hidden or getattr(command, "deprecated", False)):
            found[path] = "hidden" if command.hidden else "deprecated"
        if isinstance(command, click.Group):
            ctx = click.Context(command, info_name="bench")
            for name in command.list_commands(ctx):
                sub = command.get_command(ctx, name)
                if sub is not None:
                    walk(sub, (*path, name))

    walk(ROOT, ())
    return found


@pytest.mark.parametrize("page", JOURNEY_PAGES)
def test_journey_pages_use_no_retired_command(page: str) -> None:
    bad = []
    for line, argv in _commands(page):
        _, path, _, _ = _resolve(argv)
        for n in range(1, len(path) + 1):
            kind = _retired().get(tuple(path[:n]))
            if kind:
                bad.append(f"{page}:{line}: `bench {' '.join(path[:n])}` is {kind}")
    assert not bad, "\n".join(bad)


def _python_blocks(page: str):
    return [b for b in blocks(read(page)) if b.lang in {"python", "py"}]


@pytest.mark.parametrize("page", JOURNEY_PAGES)
def test_python_blocks_compile_and_name_real_apis(page: str) -> None:
    bad = []
    for block in _python_blocks(page):
        tree = compile(
            block.text,
            f"{page}:{block.line}",
            "exec",
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "bf"
                and not hasattr(bf, node.attr)
            ):
                bad.append(f"{page}:{block.line}: bf.{node.attr} does not exist")
            if isinstance(node, ast.ImportFrom) and node.module == "benchflow":
                for alias in node.names:
                    if not hasattr(bf, alias.name):
                        bad.append(
                            f"{page}:{block.line}: benchflow has no {alias.name}"
                        )
    assert not bad, "\n".join(bad)


# --- links ------------------------------------------------------------------


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@lru_cache
def _anchors(path: Path) -> frozenset[str]:
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    found = set(re.findall(r'<a\s+(?:name|id)="([^"]+)"', text))
    for line in prose(text, inline_code=True).splitlines():
        m = re.match(r"#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not m:
            continue
        slug = _slug(m.group(1))
        n = seen.get(slug, 0)
        found.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return frozenset(found)


def _markdown_files() -> list[Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]


def _links(text: str) -> list[str]:
    body = prose(text)
    found = re.findall(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", body)
    found += re.findall(r"^\s*\[[^\]]+\]:\s*(\S+)", body, re.M)
    return found


def test_every_relative_link_and_anchor_resolves() -> None:
    bad = []
    for page in _markdown_files():
        for link in _links(page.read_text(encoding="utf-8")):
            if re.match(r"^[a-z][a-z0-9+.-]*:", link, re.I):
                continue
            target, _, fragment = link.partition("#")
            path = (page.parent / target).resolve() if target else page
            if not path.exists():
                bad.append(f"{page.relative_to(REPO)}: {link}: no such file")
                continue
            if (
                fragment
                and path.suffix == ".md"
                and fragment.lower() not in _anchors(path)
            ):
                bad.append(f"{page.relative_to(REPO)}: {link}: no such heading")
    assert not bad, "\n".join(bad)


# --- version and the site build ---------------------------------------------


def test_readme_names_this_version() -> None:
    m = re.search(r"These docs describe BenchFlow (\d+\.\d+)\.", read("README.md"))
    assert m, "the README says which BenchFlow version its docs describe"
    assert version("benchflow").startswith(m.group(1) + "."), (
        m.group(1),
        version("benchflow"),
    )


@pytest.mark.parametrize("page", [p for p in JOURNEY_PAGES if p.startswith("docs/")])
def test_journey_pages_are_safe_for_the_site_build(page: str) -> None:
    """www.benchflow.ai renders these pages as MDX, which rejects HTML comments
    and reads a bare `<tag` or `{` outside code as JSX."""
    text = prose(read(page))
    bad = [
        f"{page}: {line.strip()[:80]}"
        for line in text.splitlines()
        if "<!--" in line or re.search(r"<[A-Za-z/!]", line) or re.search(r"[{}]", line)
    ]
    assert not bad, "\n".join(bad)

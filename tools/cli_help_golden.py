"""Golden of every ``bench`` command's ``--help`` text.

Every command and group, hidden and deprecated ones included, is rendered
with ``--help`` in a fresh interpreter with a fixed environment (terminal
width 100, no colour, a throwaway HOME), and written to one file, root help
first. ``tests/test_cli_help_golden.py`` compares the live output with it, so
a branch that changes a flag, a verb or any help text shows the change as a
diff of this file.

Regenerate it on a build VM (never by hand) with the test command's
environment (``uv sync --locked --extra dev --extra sandbox-daytona
--extra sandbox-modal``), then read ``git diff`` before committing::

    uv run python tools/cli_help_golden.py --write

``--check`` exits 1 and prints a unified diff when the help differs.
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "fixtures" / "cli_help_golden.txt"
REGENERATE = "uv run python tools/cli_help_golden.py --write"
WIDTH = "100"
_CHILD = "_BENCHFLOW_HELP_GOLDEN_CHILD"
_HEADER = "==> bench {} <=="


def _child_env(home: str) -> dict[str, str]:
    """A fixed environment: nothing from the caller's shell reaches the help."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "COLUMNS": WIDTH,
        "TERMINAL_WIDTH": WIDTH,
        "_TYPER_FORCE_DISABLE_TERMINAL": "1",
        "PYTHONHASHSEED": "0",
        _CHILD: "1",
    }


def _command_paths(command, path: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """Every command path under ``command``, depth first, in listing order."""
    import click

    yield path
    if isinstance(command, click.Group):
        ctx = click.Context(command, info_name=path[-1] if path else "bench")
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None:
                yield from _command_paths(sub, (*path, name))


def _render() -> str:
    """Render every command's ``--help`` in this interpreter (the child)."""
    import typer.main
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    runner = CliRunner()

    def help_text(path: tuple[str, ...]) -> str:
        result = runner.invoke(app, [*path, "--help"], prog_name="bench")
        if result.exit_code != 0:
            raise SystemExit(
                f"bench {' '.join(path)} --help exited {result.exit_code}:\n"
                f"{result.output}"
            )
        return result.stdout

    # Root help first, before the walk below asks every group for its
    # commands: the root listing is what `bench --help` prints in a fresh
    # process, where no command group has been imported yet.
    parts = [_HEADER.format("--help"), help_text(())]
    for path in _command_paths(typer.main.get_command(app), ()):
        if not path:
            continue
        parts.append(_HEADER.format(" ".join((*path, "--help"))))
        parts.append(help_text(path))
    return "\n".join(parts)


def render() -> str:
    """Every command's ``--help``, rendered in a fresh interpreter."""
    with tempfile.TemporaryDirectory(prefix="bench-help-golden-") as home:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=_child_env(home),
            cwd=home,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"rendering --help failed ({proc.returncode}):\n{proc.stderr}"
            )
        if home in proc.stdout:
            raise SystemExit(
                "a --help text shows the throwaway HOME; the golden would not "
                "be the same on another machine"
            )
        return proc.stdout


def diff(expected: str, actual: str) -> str:
    """A unified diff of the golden against the live help ('' when equal)."""
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(GOLDEN.relative_to(REPO)),
            tofile="live --help",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(_CHILD) == "1":
        sys.stdout.write(_render())
        return 0
    parser = argparse.ArgumentParser(
        description="Write or check the golden of every bench command's --help."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite the golden")
    mode.add_argument("--check", action="store_true", help="diff against it")
    args = parser.parse_args(argv)
    actual = render()
    if args.write:
        GOLDEN.write_text(actual, encoding="utf-8")
        print(f"wrote {GOLDEN.relative_to(REPO)}")
        return 0
    expected = GOLDEN.read_text(encoding="utf-8") if GOLDEN.exists() else ""
    if expected == actual:
        return 0
    sys.stdout.write(diff(expected, actual))
    print(f"\n--help differs from the golden; regenerate with: {REGENERATE}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

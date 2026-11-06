"""Every `bench eval run` / `bench eval branch` flag has a documented Python side.

The tables live in ``benchflow.cli_parity``; this fails when a flag is added,
renamed or removed without a row, when a row names a flag the CLI no longer
has, when a Python target stops existing, or when the generated docs page is
stale (regenerate with ``python -m benchflow.cli_parity >
docs/reference/cli-python-parity.md``).
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
import typer.main

from benchflow import cli_parity
from benchflow.cli.main import app

DOC = Path(__file__).resolve().parents[1] / "docs/reference/cli-python-parity.md"


def _flags(command: str) -> set[str]:
    root = typer.main.get_command(app)
    _, group_name, name = command.split()
    group = root.get_command(click.Context(root), group_name)
    cmd = group.get_command(click.Context(group), name)
    return {
        next(o for o in p.opts if o.startswith("--"))
        for p in cmd.params
        if isinstance(p, click.Option) and not p.hidden
    }


def test_train_convert_is_covered() -> None:
    """``bench train convert`` gained --reward-vector, --group-advantage and
    --group-by; its flags are tabled like
    the eval commands'."""
    assert "bench train convert" in cli_parity.TABLES
    assert {"--reward-vector", "--group-advantage", "--group-by"} <= set(
        cli_parity.TABLES["bench train convert"]
    )


@pytest.mark.parametrize("command", list(cli_parity.TABLES))
def test_every_flag_has_a_row_and_no_row_is_stale(command: str) -> None:
    flags = _flags(command)
    rows = set(cli_parity.TABLES[command])
    assert sorted(flags - rows) == [], "flags without a Python row"
    assert sorted(rows - flags) == [], "rows for flags the CLI no longer has"


@pytest.mark.parametrize(
    "command,flag",
    [(c, f) for c, table in cli_parity.TABLES.items() for f in table],
)
def test_every_python_target_exists(command: str, flag: str) -> None:
    eq = cli_parity.TABLES[command][flag]
    if eq.kind == "cli":
        assert eq.note, f"{flag}: a CLI-only flag needs a reason"
        return
    cli_parity.resolve(eq.target)
    if eq.kind == "sdk":
        import benchflow as bf

        head = eq.target.split(".")[1].split("(")[0]
        assert head in bf.__all__, f"{flag}: {head} is not public"


def test_the_docs_page_is_current() -> None:
    assert DOC.read_text() == cli_parity.markdown()

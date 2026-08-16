"""Unit tests for the shared ``bench traj`` terminal design language.

The kit is presentation-only: these tests pin the compatibility contract —
styled labels unstyle back to their plain copy, and every interactive
affordance degrades to the plain numbered/prompt flow off-TTY — not pixels.
"""

from __future__ import annotations

import io
from pathlib import Path

import click
import typer
from rich.console import Console

import benchflow.cli._traj_tui as tui


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=100, force_terminal=False), buffer


def test_styled_labels_unstyle_to_plain_copy() -> None:
    """Prompts keep their greppable copy under click.unstyle (agent contract)."""
    assert click.unstyle(tui.field_label("Email")) == "◇ Email"
    assert click.unstyle(tui.question_label("Upload this trajectory?")) == (
        "◆ Upload this trajectory?"
    )


def test_banner_names_the_command() -> None:
    console, buffer = _console()
    tui.banner(console, "traj upload", "subtitle copy")
    output = buffer.getvalue()
    assert "benchflow" in output
    assert "traj upload" in output
    assert "subtitle copy" in output


def test_tty_picker_arrow_burst_navigates_instead_of_cancelling(
    monkeypatch,
) -> None:
    """Guards the PR #1026 picker against buffered-stdin arrow decoding: an
    arrow key arrives as ``ESC [ B`` in one burst, and reading the buffered
    text stream swallowed ``[ B`` after the ESC, so every arrow press
    cancelled the picker and dumped the user at the manual path prompt."""
    import os
    import pty
    import termios
    import threading

    master, slave = pty.openpty()
    try:
        # Nothing reads the master side in this test, so slave echo would make
        # setcbreak's TCSADRAIN wait forever on undrained output.
        attributes = termios.tcgetattr(slave)
        attributes[3] &= ~termios.ECHO
        termios.tcsetattr(slave, termios.TCSANOW, attributes)
        # setcbreak flushes queued input (TCSAFLUSH), so the burst must land
        # after the picker is reading, exactly like a real keypress.
        writer = threading.Timer(
            0.3, os.write, (master, b"\x1b[B\x1b[B\r")
        )  # down, down, enter — one burst
        writer.start()
        with open(slave, closefd=False) as stream:
            monkeypatch.setattr(tui.sys, "stdin", stream)
            console, _buffer = _console()
            choice = tui._select_tty(
                console,
                title="Pick a session to upload",
                options=[
                    tui.SelectOption(label="first"),
                    tui.SelectOption(label="second"),
                    tui.SelectOption(label="third"),
                ],
                initial=0,
            )
        writer.join()
    finally:
        os.close(master)
        os.close(slave)

    assert choice == 2


def test_select_uses_numbered_fallback_without_tty(monkeypatch) -> None:
    """Off-TTY, the picker is a plain numbered prompt that retries bad input."""
    monkeypatch.setattr(tui, "interactive_terminal", lambda: False)
    answers = iter(["oops", "9", "2"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(answers))
    console, buffer = _console()

    choice = tui.select(
        console,
        title="Recent sessions",
        options=[
            tui.SelectOption(label="one"),
            tui.SelectOption(label="two", hint="a hint"),
        ],
    )

    assert choice == 1
    output = buffer.getvalue()
    assert "Recent sessions" in output
    assert "one" in output and "two" in output and "a hint" in output
    assert "Need a number between 1 and 2" in output
    # The selection is echoed so the transcript records the decision.
    assert f"{tui.GLYPH_POINTER} two" in output


def test_select_fallback_quits_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(tui, "interactive_terminal", lambda: False)
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: "q")
    console, buffer = _console()

    choice = tui.select(console, title="", options=[tui.SelectOption(label="only")])

    assert choice is None
    assert tui.GLYPH_POINTER not in buffer.getvalue()


def test_select_with_no_options_is_none() -> None:
    console, _buffer = _console()
    assert tui.select(console, title="x", options=[]) is None


def test_compact_path_prefers_home_relative_and_preserves_tail() -> None:
    home = Path.home()
    inside = home / ".claude" / "projects" / "demo" / "session.jsonl"
    assert tui.compact_path(inside) == "~/.claude/projects/demo/session.jsonl"

    deep = Path("/very/long/path") / ("x" * 80) / "project" / "session.jsonl"
    label = tui.compact_path(deep)
    assert label.startswith("…/")
    assert label.endswith("project/session.jsonl")


def test_escaped_neutralizes_rich_markup() -> None:
    """User-controlled labels (paths, snippets) cannot inject Rich markup."""
    console, buffer = _console()
    console.print(tui._escaped("[red]not markup[/red]"))
    assert "[red]not markup[/red]" in buffer.getvalue()

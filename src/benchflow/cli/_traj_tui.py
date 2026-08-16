"""Shared terminal design language for the ``bench traj`` command family.

``bench traj setup``, ``bench traj upload``, and ``bench traj status`` render
through this one small kit — a banner, a glyph vocabulary, styled prompts, and
an arrow-key list picker — so the family reads as one coherent tool instead of
three ad-hoc scripts. The look borrows from opencode's TUI: a single accent
color, dim structural text, `◆`/`◇` step glyphs, and a `❯` selection pointer.

Two hard rules keep this compatible with agents and the upload skill:

1. Machine-read lines stay plain. Copy such as ``Masked for you: …``,
   ``Digest: sha256:…``, and the ``Repo:`` line are printed by the command
   modules with ``print`` — never boxed or wrapped here.
2. Every interactive affordance has a dumb-terminal fallback. The picker
   degrades to a numbered prompt whenever stdin/stdout is not a TTY (CI,
   agents piping input, Windows without termios), preserving the exact
   prompt-driven contract the tests pin.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

ACCENT = "cyan"
OK = "green"
WARN = "yellow"
ERR = "red"

GLYPH_STEP = "◆"
GLYPH_FIELD = "◇"
# The heavy angle quotation is the intended selection pointer (same glyph
# opencode and modern shell prompts use); it is not a mistyped ``>``.
GLYPH_POINTER = "❯"  # noqa: RUF001
GLYPH_OK = "✓"
GLYPH_FAIL = "✗"

# Preview/table styling for trajectory step kinds, matching the color language
# of the browser viewer (PR #1020) so the terminal report and the viewer agree.
KIND_STYLES = {
    "Human": "bold green",
    "Assistant": "cyan",
    "Thinking": "magenta",
    "Tool call": "yellow",
}

_PICKER_HELP = "↑/↓ move · enter select · 1-9 jump · esc cancel"


def banner(console: Console, command: str, subtitle: str = "") -> None:
    """One-line product banner: ``◆ benchflow · <command>`` plus a dim subtitle."""
    console.print(
        f"[bold {ACCENT}]{GLYPH_STEP} benchflow[/] [dim]·[/] [bold]{command}[/]"
    )
    if subtitle:
        console.print(f"  [dim]{subtitle}[/]")
    console.print()


def section(console: Console, title: str) -> None:
    """A dim rule that separates flow phases (report / upload / verify)."""
    console.print(Rule(Text(title, style=f"bold {ACCENT}"), style="dim", align="left"))


def field_label(label: str) -> str:
    """Style an input prompt label; the trailing ``:`` stays typer's."""
    return typer.style(f"{GLYPH_FIELD} ", fg="cyan") + typer.style(label, bold=True)


def question_label(label: str) -> str:
    """Style a yes/no confirmation label."""
    return typer.style(f"{GLYPH_STEP} ", fg="cyan") + typer.style(label, bold=True)


@dataclass(frozen=True)
class SelectOption:
    """One row of the picker: a primary label and an optional dim hint."""

    label: str
    hint: str = ""


def interactive_terminal() -> bool:
    """True when arrow-key interaction is possible on this terminal."""
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    import os

    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError:  # Windows: fall back to the numbered prompt
        return False
    return True


def select(
    console: Console,
    *,
    title: str,
    options: Sequence[SelectOption],
    initial: int = 0,
) -> int | None:
    """Pick one option; returns its index, or ``None`` when cancelled.

    Arrow-key navigation on a real terminal, a numbered prompt everywhere
    else. The chosen label is echoed with a ``❯`` so the transcript records
    the decision after the transient picker disappears.
    """
    if not options:
        return None
    initial = max(0, min(initial, len(options) - 1))
    if interactive_terminal():
        choice = _select_tty(console, title=title, options=options, initial=initial)
    else:
        choice = _select_numbered(console, title=title, options=options)
    if choice is not None:
        console.print(f"[{ACCENT}]{GLYPH_POINTER}[/] {_escaped(options[choice].label)}")
    return choice


def _escaped(value: str) -> str:
    from rich.markup import escape

    return escape(value)


def _select_tty(
    console: Console,
    *,
    title: str,
    options: Sequence[SelectOption],
    initial: int,
) -> int | None:
    import termios
    import tty

    from rich.live import Live

    index = initial
    # Hints share the row; keep every row one physical line so the Live
    # region never jitters while navigating.
    width = console.width or 100

    def row(position: int, option: SelectOption) -> Text:
        selected = position == index
        pointer = f" {GLYPH_POINTER} " if selected else "   "
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(pointer, style=ACCENT if selected else "")
        text.append(option.label, style=f"bold {ACCENT}" if selected else "")
        if option.hint:
            text.append("  " + option.hint, style="dim")
        text.truncate(max(20, width - 2), overflow="ellipsis")
        return text

    def view() -> Group:
        lines: list[Text | Rule] = []
        if title:
            lines.append(Text(f"{GLYPH_STEP} {title}", style=f"bold {ACCENT}"))
        lines.extend(row(position, option) for position, option in enumerate(options))
        lines.append(Text(_PICKER_HELP, style="dim"))
        return Group(*lines)

    stream = sys.stdin
    descriptor = stream.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        with Live(view(), console=console, auto_refresh=False, transient=True) as live:
            while True:
                key = _read_key(stream)
                if key in {"up", "k"}:
                    index = (index - 1) % len(options)
                elif key in {"down", "j"}:
                    index = (index + 1) % len(options)
                elif key in {"enter"}:
                    return index
                elif key in {"esc", "q", "interrupt"}:
                    return None
                elif key.isdigit() and 1 <= int(key) <= len(options):
                    index = int(key) - 1
                    return index
                live.update(view(), refresh=True)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def _read_key(stream) -> str:
    """Decode one keypress, folding escape sequences into named keys.

    Reads the raw descriptor, never the buffered text stream: an arrow key
    arrives as ``ESC [ A`` in one burst, and a buffered ``read(1)`` would
    swallow ``[ A`` into Python's buffer, making the descriptor look idle and
    the ESC read as a bare cancel.
    """
    import os
    import select as selectors

    descriptor = stream.fileno()
    char = os.read(descriptor, 1).decode("utf-8", "replace")
    if char in {"\r", "\n"}:
        return "enter"
    if char == "\x03":
        return "interrupt"
    if char != "\x1b":
        return char
    # Distinguish a bare Escape from a CSI arrow sequence.
    if not selectors.select([descriptor], [], [], 0.05)[0]:
        return "esc"
    second = os.read(descriptor, 1).decode("utf-8", "replace")
    if second != "[":
        return "esc"
    final = os.read(descriptor, 1).decode("utf-8", "replace")
    return {"A": "up", "B": "down"}.get(final, "esc")


def _select_numbered(
    console: Console,
    *,
    title: str,
    options: Sequence[SelectOption],
) -> int | None:
    if title:
        console.print(f"[bold {ACCENT}]{GLYPH_STEP} {title}[/]")
    for position, option in enumerate(options, start=1):
        console.print(f"[bold {ACCENT}]{position:>2}[/] {_escaped(option.label)}")
        if option.hint:
            console.print(f"   [dim]{_escaped(option.hint)}[/]")
    while True:
        raw = typer.prompt("Which number?", default="1").strip()
        if raw.lower() in {"q", "quit", "esc"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print(f"[{ERR}]Need a number between 1 and {len(options)}.[/]")
            continue
        if 1 <= value <= len(options):
            return value - 1
        console.print(f"[{ERR}]Need a number between 1 and {len(options)}.[/]")


def compact_path(path: Path, *, limit: int = 64) -> str:
    """Home-relative, tail-preserving path label for one-line picker rows."""
    try:
        text = "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        text = str(path)
    if len(text) <= limit:
        return text
    tail = "/".join(Path(text).parts[-2:])
    return f"…/{tail}" if tail else text

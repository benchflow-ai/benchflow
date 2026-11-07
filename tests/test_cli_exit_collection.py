"""The ``bench`` console script leaves its objects to the process exit.

At shutdown the interpreter's collections walk every object the command
made: about 25 ms after ``bench eval run`` has printed its result, with the
engine's modules loaded. The console script freezes them on the way out
(``gc.freeze``); a caller that runs the app in its own process (``argv``
given, as tests do) is left alone.
"""

from __future__ import annotations

import gc
import sys

import pytest

from benchflow.cli._errors import run_cli
from benchflow.cli.main import app


@pytest.fixture
def freezes(monkeypatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(gc, "freeze", lambda: calls.append(1))
    return calls


def test_the_console_script_freezes_on_its_way_out(monkeypatch, freezes):
    monkeypatch.setattr(sys, "argv", ["bench", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        run_cli(app)
    assert exit_info.value.code in (0, None)
    assert freezes == [1]


def test_a_failing_command_freezes_too(monkeypatch, freezes):
    monkeypatch.setattr(sys, "argv", ["bench", "no-such-command"])
    with pytest.raises(SystemExit) as exit_info:
        run_cli(app)
    assert exit_info.value.code == 2
    assert freezes == [1]


def test_an_in_process_caller_is_left_alone(freezes):
    with pytest.raises(SystemExit):
        run_cli(app, ["--version"])
    assert freezes == []

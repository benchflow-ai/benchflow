"""The lazily registered commands list and behave as eager registration did.

``bench --help`` and ``bench eval --help`` list most commands from the facts in
``cli/main.py``'s ``_ROOT_LAYOUT`` and ``_EVAL_LAYOUT`` without importing their
modules. These tests keep those facts equal to the real commands, keep the
order and the set of commands, and check that reading a command imports it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import click
import click.testing
import pytest
import typer
from typer.testing import CliRunner

from benchflow.cli._lazy import LazyCommand, LazyGroup, listing_help
from benchflow.cli.main import _EVAL_LAYOUT, _ROOT_LAYOUT, app

ROOT_ORDER = [
    # The verb set, in Rust's groups (cli/verbs.py), then "More", then hidden.
    "new",
    "check",
    "import",
    "run",
    "branch",
    "results",
    "view",
    "regrade",
    "export",
    "doctor",
    "agents",
    "explain",
    "cache",
    "eval",
    "tasks",
    "agent",
    "review",
    "assess",
    "preferences",
    "bundle",
    "skills",
    "traj",
    "train",
    "hub",
    "daemon",
    "sandbox",
    "embodied",
    "continue",
    "continue-batch",
    "adopt",
    "environment",
    "monitor",
]
EVAL_ORDER = [
    "smoke",
    "run",
    "create",
    "adopt",
    "compare-lift",
    "score",
    "regrade",
    "list",
    "metrics",
    "view",
    "continue",
    "continue-batch",
    "branch",
    "inspect",
    "compare",
    "resume",
    "branches",
]


def _root() -> LazyGroup:
    root = typer.main.get_command(app)
    assert isinstance(root, LazyGroup)
    return root


def _groups() -> list[tuple[str, LazyGroup]]:
    root = _root()
    eval_group = root.commands["eval"]
    assert isinstance(eval_group, LazyGroup)
    return [("bench", root), ("bench eval", eval_group)]


def _lazy_cases() -> list[tuple[str, LazyCommand]]:
    return [
        *(("bench", e) for e in _ROOT_LAYOUT if isinstance(e, LazyCommand)),
        *(("bench eval", e) for e in _EVAL_LAYOUT if isinstance(e, LazyCommand)),
    ]


@pytest.mark.parametrize(
    ("parent", "spec"),
    _lazy_cases(),
    ids=[f"{p} {s.name}" for p, s in _lazy_cases()],
)
def test_listed_facts_match_the_real_command(parent, spec):
    group = dict(_groups())[parent]
    real = group.commands[spec.name]
    assert not isinstance(real, LazyCommand)
    assert real.name == spec.name
    assert listing_help(real) == spec.help, (
        f"`{parent} --help` lists {spec.name!r} with a stale line: copy the "
        f"command's first help paragraph into cli/main.py's layout:\n"
        f"{listing_help(real)!r}"
    )
    assert real.hidden == spec.hidden
    assert bool(real.deprecated) == spec.deprecated
    assert (getattr(real, "rich_help_panel", None) or None) == spec.rich_help_panel


def test_groups_keep_their_command_order():
    groups = dict(_groups())
    assert list(groups["bench"].commands) == ROOT_ORDER
    assert list(groups["bench eval"].commands) == EVAL_ORDER


def test_reading_commands_gives_real_commands():
    """Code walking ``group.commands`` gets callable commands, never stand-ins."""
    for _, group in _groups():
        for name, command in group.commands.items():
            assert command.callback is not None or isinstance(command, click.Group), (
                name
            )
        assert all(
            group.commands.get(name) is group.commands[name] for name in group.commands
        )


def test_unknown_command_still_suggests_a_close_name():
    result = CliRunner().invoke(app, ["agnt"], prog_name="bench")
    assert result.exit_code == 2
    # `bench agents` (the verb set) is as close: both are suggested.
    assert "Did you mean 'agent', 'agents'?" in result.output


def test_lazy_commands_load_on_typer_before_0_20(monkeypatch):
    """pyproject allows typer>=0.9; ``suggest_commands`` first exists in 0.20.

    Before 0.20 a group has no ``suggest_commands`` attribute and ``typer.Typer``
    takes no such keyword, so loading a lazy command must ask for neither.
    """
    spec = next(
        e for e in _ROOT_LAYOUT if isinstance(e, LazyCommand) and e.name == "doctor"
    )
    real_typer = typer.Typer

    def typer_before_0_20(**options):
        if "suggest_commands" in options:
            raise TypeError("unexpected keyword argument 'suggest_commands'")
        return real_typer(**options)

    class Group(LazyGroup):
        layout = (spec,)

    group = Group(name="bench")
    del group.suggest_commands
    monkeypatch.setattr(typer, "Typer", typer_before_0_20)

    result = click.testing.CliRunner().invoke(group, ["doctor", "--help"])
    assert result.exit_code == 0, repr(result.exception)
    assert group.commands["doctor"].callback is not None


def _modules_after(args: list[str]) -> set[str]:
    code = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from benchflow.cli.main import app\n"
        f"r = CliRunner().invoke(app, {args!r}, prog_name='bench')\n"
        "assert r.exit_code == 0, r.output\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return set(proc.stdout.split())


def test_help_imports_no_command_module():
    lazy_modules = {spec.module for _, spec in _lazy_cases()}
    assert _modules_after(["--help"]) & lazy_modules == set()
    assert _modules_after(["eval", "--help"]) & lazy_modules == set()


def test_a_command_imports_only_its_own_module():
    lazy_modules = {spec.module for _, spec in _lazy_cases()}
    loaded = _modules_after(["agent", "--help"]) & lazy_modules
    assert loaded == {"benchflow.cli.agent"}


def test_a_failed_load_fails_again_instead_of_running_the_stand_in(monkeypatch):
    """A lazy command whose import fails keeps failing on the next lookup.

    The loader used to forget the spec before importing, so a second
    ``doctor`` in the same command tree found the callback-less stand-in and
    exited 0 having done nothing.
    """
    spec = next(
        e for e in _ROOT_LAYOUT if isinstance(e, LazyCommand) and e.name == "doctor"
    )

    class Group(LazyGroup):
        layout = (spec,)

    group = Group(name="bench")
    real_import = importlib.import_module
    broken = {"on": True}

    def import_module(name, *args, **kwargs):
        if broken["on"] and name == spec.module:
            raise ImportError(f"injected: {name} cannot be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    runner = click.testing.CliRunner()
    for _ in range(2):
        result = runner.invoke(group, ["doctor", "--help"])
        assert isinstance(result.exception, ImportError), result.output
        assert result.exit_code != 0
    broken["on"] = False
    result = runner.invoke(group, ["doctor", "--help"])
    assert result.exit_code == 0, repr(result.exception)
    assert group.commands["doctor"].callback is not None


@pytest.mark.parametrize("path", [["doctor"], ["eval", "smoke"]])
def test_a_command_registered_over_a_lazy_one_replaces_it(path):
    """An application's own ``app.command("doctor")`` runs, not the built-in.

    Eager registration ran ``register_doctor(app)`` first, so a command the
    application registered later under the same name won. The lazy entry
    stayed pending and its first lookup loaded the built-in over the
    replacement.
    """
    code = (
        "import sys, typer\n"
        "from typer.testing import CliRunner\n"
        "from benchflow.cli.main import app, eval_app\n"
        f"target = eval_app if {path!r} == ['eval', 'smoke'] else app\n"
        f"@target.command({path[-1]!r})\n"
        "def replacement() -> None:\n"
        "    print('replacement ran')\n"
        f"r = CliRunner().invoke(app, {path!r}, prog_name='bench')\n"
        "assert r.exit_code == 0, (r.output, r.exception)\n"
        "print(r.output)\n"
        "print('loaded' if 'benchflow.cli.doctor' in sys.modules else 'not loaded')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "replacement ran" in proc.stdout
    assert proc.stdout.split()[-2:] == ["not", "loaded"]

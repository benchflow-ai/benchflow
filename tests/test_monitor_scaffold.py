"""Monitor mode API surface — scaffold guarantees for #386.

These tests pin the *contract* of the Monitor scaffold (importable names,
types on the public package, CLI subcommands present, fail-closed semantics).
They are intentionally not behavioural — when the runtime lands, the
behavioural tests will live in a separate module and these can be retained
as a "the public API has not silently disappeared" guard.
"""

from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.monitor import (
    Monitor,
    MonitorConfig,
    MonitorNotImplementedError,
    MonitorResult,
    not_implemented_message,
)


def test_monitor_types_are_exported_from_package() -> None:
    """Architecture names Monitor as a first-class mode — public package
    must expose it alongside Eval/Train types."""
    assert benchflow.Monitor is Monitor
    assert benchflow.MonitorConfig is MonitorConfig
    assert benchflow.MonitorResult is MonitorResult
    assert benchflow.MonitorNotImplementedError is MonitorNotImplementedError
    for name in (
        "Monitor",
        "MonitorConfig",
        "MonitorResult",
        "MonitorNotImplementedError",
    ):
        assert name in benchflow.__all__, f"{name} missing from benchflow.__all__"


def test_monitor_not_implemented_error_is_subclass_of_notimplementederror() -> None:
    """Callers may catch ``NotImplementedError`` generically; subclass
    relationship keeps that contract."""
    assert issubclass(MonitorNotImplementedError, NotImplementedError)


def test_monitor_config_minimal_construction() -> None:
    """Config must accept just a source — the only field whose absence
    would make the call ambiguous."""
    cfg = MonitorConfig(source="jobs/prod-trace/abc123")
    assert str(cfg.source) == "jobs/prod-trace/abc123"
    assert cfg.jobs_dir == "jobs/monitor"  # separable from eval evidence
    assert cfg.rubric_path is None
    assert cfg.metadata == {}


def test_monitor_result_has_alert_distinct_from_passed() -> None:
    """`alert` must exist as a distinct field — issue called out that
    monitoring failures must not look like eval regressions on dashboards."""
    result = MonitorResult(run_name="r1", source="x")
    assert hasattr(result, "alert")
    assert result.alert is False
    # error and verifier_error remain the eval/train failure taxonomy
    assert hasattr(result, "error")
    assert hasattr(result, "verifier_error")


def test_monitor_run_raises_not_implemented_with_issue_pointer() -> None:
    """Every entry point must fail closed and point users to #386."""
    cfg = MonitorConfig(source="anywhere")
    monitor = Monitor(cfg)

    with pytest.raises(MonitorNotImplementedError) as exc:
        asyncio.run(monitor.run())
    assert "386" in str(exc.value)


def test_monitor_replay_and_watch_also_fail_closed() -> None:
    """Replay and watch are part of the documented surface — same
    fail-closed semantics so callers cannot accidentally believe a run
    succeeded."""
    cfg = MonitorConfig(source="anywhere")
    monitor = Monitor(cfg)

    with pytest.raises(MonitorNotImplementedError):
        asyncio.run(monitor.replay("jobs/eval/old/task__abc"))
    with pytest.raises(MonitorNotImplementedError):
        asyncio.run(monitor.watch())


def test_not_implemented_message_is_actionable() -> None:
    """The message must point to an issue, name the mode, and be a single
    string (used directly by the CLI)."""
    msg = not_implemented_message()
    assert isinstance(msg, str)
    assert "Monitor" in msg
    assert "386" in msg


def test_cli_monitor_subcommand_exists() -> None:
    """`bench monitor --help` must list run/replay/watch — the issue
    asked for a monitor CLI even before the runtime exists."""
    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "run" in out
    assert "replay" in out
    assert "watch" in out


def test_cli_monitor_run_fails_closed_with_exit_code_2() -> None:
    """Exit code must be 2 (not 1) so dashboards can distinguish 'feature
    absent' from 'feature ran and failed'."""
    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "run", "anywhere"])
    assert result.exit_code == 2
    assert "not implemented" in result.output.lower()
    assert "386" in result.output


def test_cli_monitor_replay_and_watch_also_exit_with_code_2() -> None:
    runner = CliRunner()
    for argv in (
        ["monitor", "replay", "some/path"],
        ["monitor", "watch", "queue://events"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, (argv, result.output)
        assert "386" in result.output

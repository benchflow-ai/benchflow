"""Console progress heartbeat during agent execution.

Fresh-user dogfood (2026-08-09): between "Prompt 1/1: ..." and the verifier
there was NO console output for 18 minutes on a passing rollout — the single
biggest first-run confidence gap. The ACP session now emits a throttled
"  … Nmin, K tool calls" line from the same update stream that feeds the
trajectory writer; multi-concurrency jobs auto-gate it off, and explicit
BENCHFLOW_PROGRESS / --quiet always win.
"""

from __future__ import annotations

import logging

import pytest

from benchflow.acp import session as session_mod
from benchflow.acp.session import ACPSession, _console_progress_enabled


@pytest.mark.parametrize(
    ("progress", "auto", "expected"),
    [
        ("", "", True),  # bare default: on
        ("", "0", False),  # auto-gate (multi-concurrency) off
        ("on", "0", True),  # explicit on beats the auto-gate
        ("off", "1", False),  # explicit off beats everything
        ("false", "", False),
        ("1", "0", True),
    ],
)
def test_progress_enablement_contract(monkeypatch, progress, auto, expected):
    if progress:
        monkeypatch.setenv("BENCHFLOW_PROGRESS", progress)
    else:
        monkeypatch.delenv("BENCHFLOW_PROGRESS", raising=False)
    if auto:
        monkeypatch.setenv("BENCHFLOW_PROGRESS_AUTO", auto)
    else:
        monkeypatch.delenv("BENCHFLOW_PROGRESS_AUTO", raising=False)
    assert _console_progress_enabled() is expected


def _drive_tool_call(session: ACPSession, call_id: str) -> None:
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": "IPython cell",
            "kind": "execute",
        }
    )


def test_heartbeat_emits_throttled_line_during_prompt(monkeypatch, caplog):
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "on")
    monkeypatch.setattr(session_mod, "_PROGRESS_INTERVAL_SEC", 0.0)
    s = ACPSession("sid")
    s.record_user_prompt("do the task")
    with caplog.at_level(logging.INFO, logger=session_mod.__name__):
        _drive_tool_call(s, "tc1")
    lines = [r.message for r in caplog.records if "tool calls" in r.message]
    assert lines, "no heartbeat line emitted"
    assert "1 tool calls" in lines[-1]
    assert "IPython cell" in lines[-1]


def test_heartbeat_silent_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "off")
    monkeypatch.setattr(session_mod, "_PROGRESS_INTERVAL_SEC", 0.0)
    s = ACPSession("sid")
    s.record_user_prompt("do the task")
    with caplog.at_level(logging.INFO, logger=session_mod.__name__):
        _drive_tool_call(s, "tc1")
    assert not [r for r in caplog.records if "tool calls" in r.message]


def test_progress_snapshot_exposes_heartbeat_counters(monkeypatch):
    """The dashboard's activity cell polls the same counters the line logs."""
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "off")
    s = ACPSession("sid")
    assert s.progress_snapshot() == (0, "")
    _drive_tool_call(s, "tc1")
    assert s.progress_snapshot() == (1, "IPython cell")


def test_heartbeat_silent_outside_prompt(monkeypatch, caplog):
    """No heartbeat before the first prompt or after mark_prompt_end."""
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "on")
    monkeypatch.setattr(session_mod, "_PROGRESS_INTERVAL_SEC", 0.0)
    s = ACPSession("sid")
    with caplog.at_level(logging.INFO, logger=session_mod.__name__):
        _drive_tool_call(s, "tc0")  # pre-prompt: no active prompt yet
    assert not [r for r in caplog.records if "tool calls" in r.message]

    s.record_user_prompt("go")
    s.mark_prompt_end()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=session_mod.__name__):
        _drive_tool_call(s, "tc1")  # post-prompt: heartbeat window closed
    assert not [r for r in caplog.records if "tool calls" in r.message]

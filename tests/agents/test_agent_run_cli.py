"""CLI behavior for `bench agent run` (headless run + resume)."""

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app

FAKE_AGENT = Path(__file__).parent / "fake_acp_agent.py"


def test_agent_run_prints_json_with_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_AGENT_SESSIONS_DIR", str(tmp_path / "store"))
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.chdir(work)

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "run",
            "fake-agent",
            "-p",
            "say hi",
            "--launch-cmd",
            f"{sys.executable} {FAKE_AGENT}",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["agent"] == "fake-agent"
    assert "hello from fake" in payload["result"]
    assert payload["session_id"]
    assert payload["stop_reason"] == "end_turn"


def test_agent_run_resume_continues_the_same_session(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_AGENT_SESSIONS_DIR", str(tmp_path / "store"))
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.chdir(work)
    base = [
        "agent",
        "run",
        "fake-agent",
        "--launch-cmd",
        f"{sys.executable} {FAKE_AGENT}",
        "--output-format",
        "json",
    ]

    first = CliRunner().invoke(app, [*base, "-p", "say hi"])
    assert first.exit_code == 0, first.output
    sid = json.loads(first.stdout)["session_id"]

    second = CliRunner().invoke(app, [*base, "-p", "again", "--resume", sid])
    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["session_id"] == sid

    # -c / --continue resumes the latest session in this cwd
    third = CliRunner().invoke(app, [*base, "-p", "more", "-c"])
    assert third.exit_code == 0, third.output
    assert json.loads(third.stdout)["session_id"] == sid


def test_agent_run_times_out_with_clean_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_AGENT_SESSIONS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("FAKE_SLEEP", "10")
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.chdir(work)

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "run",
            "fake-agent",
            "-p",
            "say hi",
            "--launch-cmd",
            f"{sys.executable} {FAKE_AGENT}",
            "--timeout",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "timed out" in result.output

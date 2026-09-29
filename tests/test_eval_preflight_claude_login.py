"""`bench eval run` warns up front about an expired Claude login file.

Regression test: with only an expired
`~/.claude/.credentials.json` (the usual state on macOS, where the Claude CLI
keeps live logins in the Keychain), `bench eval run --agent claude` built the
image and installed the agent before failing with
"OAuth session expired and could not be refreshed" and no fix hint. The
preflight reuses doctor's `claude_auth` / `check_agent_auth` and prints
doctor's fix before any job exists. It warns rather than refuses: on Linux the
same file is the live login and Claude refreshes an expired access token.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow import doctor as doctor_mod
from benchflow.cli import main as cli_main
from benchflow.cli.main import app
from benchflow.doctor import Check

_CLAUDE_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in _CLAUDE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(
        doctor_mod,
        "check_docker",
        lambda probes, *, required: [
            Check("docker", "sandbox", "docker", "pass", "Docker ready")
        ],
    )
    return home


@pytest.fixture
def batch_calls(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli_main, "run_batch_eval", lambda *args, **kwargs: calls.append(args)
    )
    return calls


def _login_file(home: Path, *, expires_in_days: float) -> None:
    creds = home / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True)
    expires_ms = int((time.time() + expires_in_days * 86400) * 1000)
    creds.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-" + "x" * 40,
                    "refreshToken": "sk-ant-ort01-" + "y" * 40,
                    "expiresAt": expires_ms,
                }
            }
        )
    )


def _run(tmp_path: Path, *extra: str):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    return CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--sandbox",
            "docker",
            *extra,
        ],
    )


def test_expired_login_file_warns_with_doctors_fix_before_the_run(
    tmp_path, home, batch_calls
):
    _login_file(home, expires_in_days=-30)

    result = _run(tmp_path, "--agent", "claude", "--model", "claude-haiku-4-5")

    assert result.exit_code == 0, result.output
    assert len(batch_calls) == 1
    warning = result.stderr
    assert "~/.claude/.credentials.json" in warning
    assert "expired" in warning
    assert "claude setup-token" in warning
    assert "CLAUDE_CODE_OAUTH_TOKEN" in warning
    assert "Keychain" in warning
    assert "x" * 40 not in result.output
    assert "y" * 40 not in result.output


def test_token_passed_with_agent_env_silences_the_warning(tmp_path, home, batch_calls):
    _login_file(home, expires_in_days=-30)

    result = _run(
        tmp_path,
        "--agent",
        "claude",
        "--model",
        "claude-haiku-4-5",
        "--agent-env",
        "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-fake",
    )

    assert result.exit_code == 0, result.output
    assert "credentials.json" not in result.stderr


def test_valid_login_file_is_quiet(tmp_path, home, batch_calls):
    _login_file(home, expires_in_days=3)

    result = _run(tmp_path, "--agent", "claude", "--model", "claude-haiku-4-5")

    assert result.exit_code == 0, result.output
    assert "credentials.json" not in result.stderr


@pytest.mark.parametrize(
    "agent_args",
    [
        ("--agent", "oracle"),
        ("--agent", "claude", "--model", "deepseek/deepseek-chat"),
    ],
)
def test_runs_that_do_not_use_the_claude_login_are_quiet(
    tmp_path, home, batch_calls, agent_args
):
    _login_file(home, expires_in_days=-30)

    result = _run(tmp_path, *agent_args)

    assert result.exit_code == 0, result.output
    assert "credentials.json" not in result.stderr

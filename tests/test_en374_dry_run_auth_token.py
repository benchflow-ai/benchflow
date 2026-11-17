"""A dry run refuses the ANTHROPIC_AUTH_TOKEN its run would refuse (review of
PR #122, L2).

An ANTHROPIC_AUTH_TOKEN that is neither an Anthropic API key nor a Claude
login token, with no ANTHROPIC_BASE_URL, is refused before launch, naming
the variable. ``bench run --dry-run`` exited 0 instead, with a run card whose
``capture.auth`` said ``subscription``. It now refuses the same way. The
values here are made up; none may appear in any output.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app

TASK = Path(__file__).parent / "examples" / "hello-world-task"
UNKNOWN = "bfprobe-UNKNOWN-7f3a9c2e1d5b8a6f4e2c0d9b7a5f3e1c"
API_KEY = (
    "sk"
    + "-ant-api"
    + "03-bfpro"
    + "beFAKEKE"
    + "Y-9d8c7b"
    + "6a5f4e3d"
    + "2c1b0a9f"
    + "8e7d6c5b"
    + "4a"
)


def _dry_run(tmp_path, monkeypatch, *args, env=None):
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return CliRunner().invoke(
        app,
        [
            "run",
            str(TASK),
            "-a",
            "claude-agent-acp",
            "-m",
            "anthropic/claude-opus-5-5",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--dry-run",
            *args,
        ],
    )


@pytest.mark.parametrize("where", ["--agent-env", "inherited"])
def test_an_unknown_token_is_refused_by_the_dry_run(tmp_path, monkeypatch, where):
    if where == "--agent-env":
        result = _dry_run(
            tmp_path, monkeypatch, "--agent-env", f"ANTHROPIC_AUTH_TOKEN={UNKNOWN}"
        )
    else:
        result = _dry_run(tmp_path, monkeypatch, env={"ANTHROPIC_AUTH_TOKEN": UNKNOWN})
    assert result.exit_code == 1, result.output
    assert "ANTHROPIC_AUTH_TOKEN" in result.output
    assert UNKNOWN not in result.output
    assert '"subscription"' not in result.output
    assert not (tmp_path / "jobs").exists()


def test_an_api_key_in_the_token_slot_plans_the_proxy(tmp_path, monkeypatch):
    result = _dry_run(
        tmp_path, monkeypatch, "--agent-env", f"ANTHROPIC_AUTH_TOKEN={API_KEY}"
    )
    assert result.exit_code == 0, result.output
    assert API_KEY not in result.output
    assert '"auth": "api_key"' in result.output


@pytest.mark.parametrize("where", ["the agent's env", "inherited"])
def test_a_job_matrix_dry_run_refuses_it_too(tmp_path, monkeypatch, where):
    """Review of PR #122 (F2b-R122B, L-e): a ``--config`` job of several
    agents planned its dry run cell by cell and exited 0 with the unknown
    token, where the run refuses it."""
    import json

    from tests.test_job_matrix import _task

    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "ds-a", "alpha")
    claude = {"name": "claude-agent-acp", "model_name": "anthropic/claude-opus-5-5"}
    if where == "the agent's env":
        claude["env"] = {"ANTHROPIC_AUTH_TOKEN": UNKNOWN}
        env = None
    else:
        env = {"ANTHROPIC_AUTH_TOKEN": UNKNOWN}
    config = tmp_path / "job.json"
    config.write_text(
        json.dumps(
            {
                "agents": [claude, {"name": "oracle"}],
                "datasets": [{"path": "ds-a"}],
                "jobs_dir": str(tmp_path / "jobs"),
            }
        )
    )
    result = _config_dry_run(tmp_path, monkeypatch, config, env=env)
    assert result.exit_code == 1, result.output
    assert "ANTHROPIC_AUTH_TOKEN" in result.output
    assert UNKNOWN not in result.output
    assert '"job_matrix"' not in result.output
    assert not (tmp_path / "jobs").exists()

    # Without the token the same plan is printed.
    claude.pop("env", None)
    config.write_text(
        json.dumps(
            {
                "agents": [claude, {"name": "oracle"}],
                "datasets": [{"path": "ds-a"}],
                "jobs_dir": str(tmp_path / "jobs"),
            }
        )
    )
    # The run needs a Claude login for the claude-agent-acp cell (the
    # dry run resolves the agent's env as each trial does), so give it a
    # made-up one.
    result = _config_dry_run(
        tmp_path,
        monkeypatch,
        config,
        env={"CLAUDE_CODE_OAUTH_TOKEN": "fake-not-a-token"},
    )
    assert result.exit_code == 0, result.output
    assert '"job_matrix"' in result.output


def _config_dry_run(tmp_path, monkeypatch, config, env=None):
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return CliRunner().invoke(app, ["run", "--config", str(config), "--dry-run"])

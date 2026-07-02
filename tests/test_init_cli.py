"""`bench init` / `bench doctor` CLI behavior (thin glue over onboarding).

Non-interactive mode is the contract CI relies on: every prompt has a flag,
so a fully-flagged invocation must never block on stdin.
"""

from __future__ import annotations

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def _init_args(home, extra=()):
    return [
        "init",
        "--model",
        "deepseek/deepseek-v4-flash",
        "--agent",
        "pi-acp",
        "--dataset",
        "skillsbench",
        "--sandbox",
        "docker",
        "--api-key",
        "sk-test-123",
        "--skip-smoke",
        *extra,
    ]


def test_non_interactive_writes_files_and_prints_command(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(app, _init_args(tmp_path))
    assert result.exit_code == 0, result.output
    # secrets → private env file; prefs → config; final command printed
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {
        "DEEPSEEK_API_KEY": "sk-test-123"
    }
    assert ((tmp_path / ".env").stat().st_mode & 0o777) == 0o600
    prefs = onboarding.load_prefs(tmp_path / "config.toml")
    assert prefs["agent"] == "pi-acp"
    assert (
        "bench eval run --agent pi-acp --model deepseek/deepseek-v4-flash"
        in result.output
    )


def test_incompatible_agent_for_model_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--agent",
            "codex-acp",  # openai-responses wire: the run path would reject it
            "--dataset",
            "skillsbench",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code != 0
    assert "codex-acp" in result.output and "deepseek" in result.output


def test_doctor_without_setup_hints_at_init(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "bench init" in result.output


def test_doctor_reports_rows_and_fails_on_broken_setup(tmp_path, monkeypatch):
    """Saved setup but no key and no docker: every row shows, exit code 1."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    from benchflow import onboarding

    onboarding.save_prefs(
        tmp_path / "config.toml",
        {
            "agent": "pi-acp",
            "model": "deepseek/deepseek-v4-flash",
            "dataset": "skillsbench",
            "sandbox": "docker",
        },
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "docker" in result.output
    assert "DEEPSEEK_API_KEY" in result.output
    assert "❌" in result.output


def test_interactive_wizard_prompts_and_completes(tmp_path, monkeypatch):
    """No flags: the wizard prompts for model → agent → dataset → sandbox →
    hidden key, then persists and prints the final command."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    answers = "\n".join(
        [
            "deepseek/deepseek-v4-flash",  # model
            "pi-acp",  # agent
            "skillsbench",  # task set
            "docker",  # sandbox
            "sk-wizard-key",  # hidden api key
        ]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {
        "DEEPSEEK_API_KEY": "sk-wizard-key"
    }
    assert "bench eval run --agent pi-acp" in result.output


def test_startup_autoloads_saved_env_file(tmp_path, monkeypatch):
    """A key stored by a previous init is visible to later invocations
    without exporting anything (the CLI auto-loads ~/.benchflow/.env)."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from benchflow import onboarding

    onboarding.write_env_file(tmp_path / ".env", {"DEEPSEEK_API_KEY": "sk-saved"})
    result = runner.invoke(app, [*_init_args(tmp_path)[:-3], "--skip-smoke"])
    # (same init args minus --api-key: the saved key must be found)
    assert result.exit_code == 0, result.output
    assert "already set" in result.output

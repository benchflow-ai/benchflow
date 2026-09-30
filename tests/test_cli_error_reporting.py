"""How ``bench`` ends on an error: one message when expected, a log when not.

Guards the dx/errors fix for errors that escaped as Rich tracebacks (an
integration review found task-format loading errors doing so in ``bench eval
branch``; on 2026-09-30 a task.md naming a missing environment manifest
ended that command in a 110-line traceback box), and for a missing login,
which ran a job whose every trial logged a traceback.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from benchflow.cli._errors import RECENT_LOG, run_cli
from benchflow.errors import MissingCredentialError
from benchflow.task._document_normalize import TaskDocumentParseError

SECRET = "sk-test-0123456789abcdefghijklmnop"


def _app(exc: BaseException) -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def go(agent_env: list[str] = typer.Option(None, "--agent-env")) -> None:  # noqa: B008
        logging.getLogger("benchflow.test").info("working on it")
        raise exc

    @app.command()
    def other() -> None:
        pass

    return app


def _run(app: typer.Typer, argv: list[str]) -> int:
    with pytest.raises(SystemExit) as caught:
        run_cli(app, argv)
    return int(caught.value.code or 0)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            TaskDocumentParseError("task.md: frontmatter is not valid YAML (line 3)"),
            "task.md: frontmatter is not valid YAML (line 3)",
        ),
        (
            FileNotFoundError(2, "No such file or directory", "/tasks/x/missing.toml"),
            "No such file or directory: /tasks/x/missing.toml",
        ),
        (
            MissingCredentialError(
                "ANTHROPIC_API_KEY required for model 'm' but not set.",
                hint="run `claude setup-token`",
            ),
            "ANTHROPIC_API_KEY required for model 'm' but not set.\n  next: run `claude setup-token`",
        ),
    ],
)
def test_an_expected_error_is_one_message(exc, expected, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    assert _run(_app(exc), ["go"]) == 1
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err
    assert not (tmp_path / "logs").exists()


def test_an_unexpected_error_prints_a_traceback_and_writes_a_log(
    capsys, caplog, tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SOME_API_KEY", SECRET)
    caplog.set_level(logging.INFO)
    code = _run(
        _app(RuntimeError(f"boom while using {SECRET}")),
        ["go", "--agent-env", "OPENAI_API_KEY=sk-other-secret-value-1234"],
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "RuntimeError: boom while using ***" in err
    assert "unexpected error (RuntimeError)" in err
    assert "This is a bug in BenchFlow" in err
    logs = list((tmp_path / "logs").glob("bench-*.log"))
    assert len(logs) == 1
    assert str(logs[0]) in err
    text = logs[0].read_text()
    assert "command: bench go --agent-env OPENAI_API_KEY=***" in text
    assert "RuntimeError: boom while using ***" in text
    assert "working on it" in text  # the run's last log lines
    for secret in (SECRET, "sk-other-secret-value-1234"):
        assert secret not in text
        assert secret not in err


def test_the_recent_log_survives_the_live_dashboard():
    from benchflow.cli._errors import install_recent_log
    from benchflow.cli._live_progress import quiet_root_logging

    install_recent_log()
    before = len(RECENT_LOG.lines)
    with quiet_root_logging():
        logging.getLogger("benchflow.evaluation").warning("hidden by the dashboard")
    assert len(RECENT_LOG.lines) == before + 1
    assert RECENT_LOG.lines[-1].endswith("hidden by the dashboard")


def test_click_usage_errors_and_exits_are_left_to_click(capsys):
    app = _app(RuntimeError("never"))
    assert _run(app, ["no-such-command"]) == 2
    assert _run(app, ["other"]) == 0


def test_the_console_script_is_the_wrapper():
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'bench = "benchflow.cli.main:main"' in text
    assert 'benchflow = "benchflow.cli.main:main"' in text


def test_a_missing_manifest_in_eval_branch_is_one_message(tmp_path):
    """The before-state: a Rich traceback through branch_run and pathlib."""
    task = tmp_path / "tasks" / "bad-manifest"
    src = Path(__file__).parents[1] / "src" / "benchflow" / "demo_task"
    import shutil

    shutil.copytree(src, task)
    doc = task / "task.md"
    head, sep, rest = doc.read_text().partition("\n---\n")
    doc.write_text(
        head
        + "\nbenchflow:\n  environment:\n    manifest: environment/missing.toml"
        + sep
        + rest
    )
    env = {**os.environ, "BENCHFLOW_SKIP_PREFLIGHT": "1", "NO_COLOR": "1"}
    env["BENCHFLOW_LOG_DIR"] = str(tmp_path / "logs")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "eval",
            "branch",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "╭" not in proc.stderr  # no Rich traceback box
    assert "missing.toml" in proc.stderr


_CLAUDE_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
)
HELLO = Path(__file__).parent / "examples" / "hello-world-task"


@pytest.fixture
def no_login(tmp_path, monkeypatch):
    """No Claude credential anywhere: no variables, an empty HOME, no .env."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    for name in _CLAUDE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    return home


def test_a_missing_login_stops_bench_eval_run_before_a_job(no_login, tmp_path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    jobs = tmp_path / "jobs"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(HELLO),
            "--agent",
            "claude-agent-acp",
            "--model",
            "claude-haiku-4-5-20251001",
            "--sandbox",
            "daytona",
            "--jobs-dir",
            str(jobs),
        ],
    )
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY required for model" in result.stderr
    assert "claude setup-token" in result.stderr
    assert "No job was created." in result.stderr
    assert not jobs.exists()


def test_the_sdk_raises_a_missing_login_before_the_run(no_login, monkeypatch):
    import benchflow as bf
    from benchflow.rollout import Rollout, RolloutConfig

    created: list[object] = []

    async def fake_create(config):
        created.append(config)
        raise AssertionError("the rollout must not start")

    monkeypatch.setattr(Rollout, "create", staticmethod(fake_create))
    with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY required"):
        bf.run_sync(
            RolloutConfig(
                task_path=HELLO,
                agent="claude-agent-acp",
                model="claude-haiku-4-5-20251001",
                environment="daytona",
            )
        )
    assert created == []


def test_the_credential_check_looks_only_at_what_the_rollout_uses(
    no_login, monkeypatch
):
    """dx/errors review: check_credentials checked the legacy agent/model even
    when scenes were set, ignored role.env, and resolved a default model the
    rollout would not use, so a Gemini-only run was refused with
    "ANTHROPIC_API_KEY required"."""
    from benchflow._types import Role, Scene
    from benchflow.rollout import RolloutConfig
    from benchflow.runtime import check_credentials

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    key = "AIza" + "g" * 35
    gemini_role = Role(
        name="solver",
        agent="gemini",
        model="gemini-2.5-flash",
        env={"GEMINI_API_KEY": key},
    )
    # A Gemini scene on a machine with no Claude login: the legacy
    # claude-agent-acp default is not what runs.
    check_credentials(
        [RolloutConfig(task_path=HELLO, scenes=[Scene(name="s", roles=[gemini_role])])]
    )
    # The role's own env is what counts; without it the role is refused.
    bare = Role(name="solver", agent="gemini", model="gemini-2.5-flash")
    with pytest.raises(MissingCredentialError, match="GEMINI_API_KEY"):
        check_credentials(
            [RolloutConfig(task_path=HELLO, scenes=[Scene(name="s", roles=[bare])])]
        )
    # No model: the rollout resolves without one, and so does the check.
    check_credentials([RolloutConfig(task_path=HELLO, agent="claude-agent-acp")])

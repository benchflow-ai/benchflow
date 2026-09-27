"""Evaluation used from Python makes the same pre-run checks as bf.run and the CLI.

``bench eval run`` checks Docker and an expired Claude login before creating a
job, and ``bf.run``/``bf.run_batch`` do, but
``bf.Evaluation(...).run()`` called from Python did neither, nor did it refuse
a misspelt agent or sandbox: every task then failed inside its own sandbox.
``Evaluation.run()`` now makes the checks before the job directory exists;
``Evaluation(..., preflight=False)`` or ``BENCHFLOW_SKIP_PREFLIGHT=1`` skips
them, and the CLI passes ``preflight=False`` because it already checked.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow as bf
from benchflow import doctor as doctor_mod
from benchflow.doctor import Check
from benchflow.models import RolloutResult


@pytest.fixture(autouse=True)
def _daytona_key_ok(monkeypatch):
    """These tests are about other checks; the live Daytona key check passes."""
    monkeypatch.setattr(
        doctor_mod,
        "check_daytona",
        lambda probes, *, required, offline: Check(
            "daytona", "sandbox", "daytona", "ok", "stubbed"
        ),
    )


@pytest.fixture
def host(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    return home


def _docker(monkeypatch, status: str) -> None:
    monkeypatch.setattr(
        doctor_mod,
        "check_docker",
        lambda probes, *, required: [
            Check(
                "docker",
                "sandbox",
                "docker daemon",
                status,
                "daemon unreachable",
                fix="colima start",
            )
        ],
    )


def _expired_login(home: Path) -> None:
    creds = home / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True)
    creds.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-" + "x" * 40,
                    "refreshToken": "sk-ant-ort01-" + "y" * 40,
                    "expiresAt": int((time.time() - 76 * 86400) * 1000),
                }
            }
        )
    )


def _job(tmp_path: Path, **cfg) -> bf.Evaluation:
    tasks = tmp_path / "tasks" / "t0"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
    )
    preflight = cfg.pop("preflight", True)
    job = bf.Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(retry=bf.RetryConfig(max_retries=0), **cfg),
        preflight=preflight,
    )
    job._run_single_task = AsyncMock(
        side_effect=lambda p, c: RolloutResult(p.name, rewards={"reward": 1.0})
    )
    return job


def test_docker_not_ready_refuses_before_the_job_exists(host, tmp_path, monkeypatch):
    _docker(monkeypatch, "fail")
    job = _job(tmp_path, agent="oracle", environment="docker")
    with pytest.raises(RuntimeError, match="colima start"):
        job.run_sync()
    assert job._run_single_task.await_count == 0
    assert not (tmp_path / "jobs").exists()


def test_expired_claude_login_warns_and_runs(host, tmp_path, monkeypatch):
    _expired_login(host)
    job = _job(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        environment="daytona",
    )
    with pytest.warns(UserWarning, match="expired"):
        assert job.run_sync().total == 1


def test_misspelt_agent_and_sandbox_are_refused(host, tmp_path):
    with pytest.raises(ValueError, match="did you mean 'claude-agent-acp'"):
        _job(tmp_path, agent="claud-agent-acp", environment="daytona").run_sync()
    with pytest.raises(ValueError, match="daytona"):
        _job(tmp_path / "b", agent="oracle", environment="daytonaa").run_sync()


def test_preflight_false_and_env_opt_out_skip_the_checks(host, tmp_path, monkeypatch):
    _docker(monkeypatch, "fail")
    _expired_login(host)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        job = _job(
            tmp_path / "a", agent="oracle", environment="docker", preflight=False
        )
        assert job.run_sync().total == 1
        monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
        job = _job(tmp_path / "b", agent="oracle", environment="docker")
        assert job.run_sync().total == 1


def test_the_cli_does_not_check_twice() -> None:
    """The CLI runs eval_preflight itself; every Evaluation it builds opts out."""
    import inspect

    from benchflow import eval_worker
    from benchflow.cli import main as cli_main

    for module in (cli_main, eval_worker):
        source = inspect.getsource(module)
        constructions = source.count("Evaluation(\n") + source.count(
            "Evaluation(tasks_dir"
        )
        assert constructions > 0
        assert source.count("preflight=False") >= constructions, module.__name__

"""The SDK makes the CLI's pre-run checks.

With no Claude credential in the environment and
only an expired ``~/.claude/.credentials.json`` (the usual state on macOS),
``bf.run_sync(... agent="claude-agent-acp" ...)`` started a Daytona sandbox
and failed after 60 s with "OAuth session expired and could not be
refreshed". ``bench eval run`` has warned about exactly this, and refuses to
start without a Docker daemon; the SDK entry points now make
the same two checks, reusing doctor's: a UserWarning with doctor's fix for
the login file, a RuntimeError for Docker. ``BENCHFLOW_SKIP_PREFLIGHT=1``
skips both, as for the CLI.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pytest

import benchflow as bf
from benchflow import doctor as doctor_mod
from benchflow.doctor import Check
from benchflow.rollout import Rollout, RolloutConfig


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


TASK = Path(__file__).parent / "examples" / "hello-world-task"
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
    return home


@pytest.fixture
def ran(monkeypatch) -> list[RolloutConfig]:
    seen: list[RolloutConfig] = []

    async def fake_create(config: RolloutConfig):
        seen.append(config)

        class _R:
            async def run(self):
                return bf.RolloutResult("hello-world-task")

        return _R()

    monkeypatch.setattr(Rollout, "create", staticmethod(fake_create))
    return seen


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


def _claude(sandbox: str = "daytona") -> RolloutConfig:
    return RolloutConfig(
        task_path=TASK,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        environment=sandbox,
    )


def test_expired_login_file_warns_before_the_run(home, ran) -> None:
    _login_file(home, expires_in_days=-76)
    with pytest.warns(UserWarning, match="expired") as caught:
        bf.run_sync(_claude())
    assert "claude setup-token" in str(caught[0].message) or "login" in str(
        caught[0].message
    )
    assert len(ran) == 1  # a warning, not a refusal


def test_a_live_credential_does_not_warn(home, ran, monkeypatch) -> None:
    _login_file(home, expires_in_days=-76)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-" + "z" * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        bf.run_sync(_claude())


def test_docker_not_ready_raises_before_the_run(home, ran, monkeypatch) -> None:
    _docker(monkeypatch, "fail")
    with pytest.raises(RuntimeError, match="colima start"):
        bf.run_sync(RolloutConfig(task_path=TASK, agent="oracle", environment="docker"))
    assert ran == []


def test_batch_checks_once_before_anything_runs(home, ran, monkeypatch) -> None:
    _docker(monkeypatch, "fail")
    cfg = RolloutConfig(task_path=TASK, agent="oracle", environment="docker")
    with pytest.raises(RuntimeError, match="Docker"):
        bf.run_batch([cfg, cfg])
    assert ran == []


def test_opt_out_skips_both(home, ran, monkeypatch) -> None:
    _docker(monkeypatch, "fail")
    _login_file(home, expires_in_days=-76)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        bf.run_sync(RolloutConfig(task_path=TASK, agent="oracle", environment="docker"))
        bf.run_sync(_claude())
    assert len(ran) == 2

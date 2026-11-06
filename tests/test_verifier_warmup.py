"""Separate verifier on Docker: its sandbox starts as the agent's turn begins.

SV3 started it at verify() entry. The warm-up makes the same start (task
files only) when the trial's sandbox is ready, so its seconds overlap the
agent's turn; verify() takes it, and a trial that never verifies deletes it
in cleanup(), exactly once. The trials are real ``Rollout`` lifecycles with
fake sandbox steps; the verifier sandbox's start and deletion are recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _separate_verifier as sv
from benchflow.rollout._setup import _init_rollout

TOML = (
    'version = "1.0"\n[verifier]\ntimeout_sec = 60\nenvironment_mode = "separate"\n'
    '[agent]\ntimeout_sec = 60\n[environment]\ndocker_image = "example.invalid/v:1"\n'
)


def _task(root: Path) -> Path:
    task = root / "task"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(TOML)
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    return task


@pytest.fixture
def events(monkeypatch) -> list[tuple[str, Any]]:
    log: list[tuple[str, Any]] = []

    async def setup(self, *, verifier_only: bool = False) -> None:
        cfg = self._config
        (
            self._task,
            self._rollout_dir,
            self._rollout_paths,
            self._started_at,
            self._job_name,
            self._rollout_name,
        ) = _init_rollout(cfg.task_path, cfg.job_name, cfg.rollout_name, cfg.jobs_dir)
        self._env = object()
        self._phase = "setup"

    async def start(self) -> None:
        self._phase = "started"

    async def install_agent(self) -> None:
        log.append(("install", self))
        self._phase = "installed"

    async def verify(self) -> dict | None:
        log.append(("verify", self))
        self._rewards = {"reward": 1.0}
        return self._rewards

    async def before_teardown(self) -> None:
        log.append(("cleanup", self))

    async def teardown(self) -> None:
        self._phase = "cleaned"

    def begin_start(self, **_: object) -> None:
        log.append(("verifier_start", self))

    async def discard(self, *, cancelled: bool) -> None:
        log.append(("verifier_discard", self))

    monkeypatch.setattr(Rollout, "setup", setup)
    monkeypatch.setattr(Rollout, "start", start)
    monkeypatch.setattr(Rollout, "install_agent", install_agent)
    monkeypatch.setattr(Rollout, "verify", verify)
    monkeypatch.setattr(Rollout, "_cleanup_before_teardown", before_teardown)
    monkeypatch.setattr(Rollout, "_teardown_sandbox", teardown)
    monkeypatch.setattr(sv.VerifierSandboxRun, "begin_start", begin_start)
    monkeypatch.setattr(sv.VerifierSandboxRun, "discard", discard)
    monkeypatch.delenv(sv.VERIFIER_PREPARE_ENV, raising=False)
    return log


def _rollout(tmp_path: Path, environment: str = "docker") -> Rollout:
    return Rollout(
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path / "tasks"),
            agent="nop",
            environment=environment,
            jobs_dir=str(tmp_path / "jobs"),
            job_name="warm",
        )
    )


def _steps(log: list[tuple[str, Any]]) -> list[str]:
    return [step for step, _ in log]


@pytest.mark.asyncio
async def test_verifier_sandbox_starts_when_the_agents_turn_begins(tmp_path, events):
    await _rollout(tmp_path).run()
    # Started after the trial's own sandbox was ready, before verify().
    assert _steps(events)[:3] == ["install", "verifier_start", "verify"]
    starts = [obj for step, obj in events if step == "verifier_start"]
    assert len(starts) == 1


@pytest.mark.asyncio
async def test_a_warmed_verifier_no_verify_took_is_deleted_once_at_cleanup(
    tmp_path, events
):
    # verify() here does not take it (the fake): cleanup must delete it.
    await _rollout(tmp_path).run()
    starts = [obj for step, obj in events if step == "verifier_start"]
    discards = [obj for step, obj in events if step == "verifier_discard"]
    assert len(starts) == 1
    assert discards == starts
    assert _steps(events).index("verifier_discard") > _steps(events).index("cleanup")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["serial", "entry"])
async def test_prepare_switch_keeps_the_later_start(
    tmp_path, events, monkeypatch, value
):
    monkeypatch.setenv(sv.VERIFIER_PREPARE_ENV, value)
    await _rollout(tmp_path).run()
    assert "verifier_start" not in _steps(events)


@pytest.mark.asyncio
async def test_no_warmup_off_docker(tmp_path, events):
    await _rollout(tmp_path, environment="daytona").run()
    assert "verifier_start" not in _steps(events)

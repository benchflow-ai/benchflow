"""The solution is uploaded only for an oracle run (GAP-ORACLE-READABLE).

The lockdown keeps ``/oracle`` and ``/solution`` from the ``agent`` user, not
from an agent run as root, so no other run gets a copy in the sandbox. The
real-Docker check is tests/test_deterministic_solution_hidden.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout._setup import _start_env_and_upload
from tests.test_rollout_upload import FakeUploadEnv
from tests.test_trial_install_agent_timeout import _make_trial


def _task(root: Path, dirname: str) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / dirname).mkdir()
    (task / "instruction.md").write_text("solve\n")
    (task / dirname / "solve.sh").write_text("echo ok\n")
    return task


@pytest.mark.parametrize("dirname", ["oracle", "solution"])
async def test_start_uploads_no_solution_by_default(tmp_path, dirname):
    env = FakeUploadEnv()
    await _start_env_and_upload(env, _task(tmp_path, dirname), {})
    assert env.uploaded_dirs == []
    assert not [c for c in env.exec_calls if "/oracle" in c[0] or "/solution" in c[0]]


@pytest.mark.parametrize(
    ("dirname", "target"), [("oracle", "/oracle"), ("solution", "/solution")]
)
async def test_start_uploads_the_solution_when_asked(tmp_path, dirname, target):
    env = FakeUploadEnv()
    task = _task(tmp_path, dirname)
    await _start_env_and_upload(env, task, {}, upload_solution=True)
    assert env.uploaded_dirs == [(task / dirname, target)]


@pytest.mark.parametrize(
    ("agent", "uploads"),
    [("oracle", True), ("nop", False), ("claude-agent-acp", False)],
)
async def test_only_an_oracle_rollout_asks_for_the_solution(
    tmp_path, monkeypatch, agent, uploads
):
    import benchflow.rollout as rollout_mod

    asked: list[bool] = []

    async def fake_start(env, task_path, timing, **kwargs):
        asked.append(kwargs.get("upload_solution", False))

    monkeypatch.setattr(rollout_mod, "_start_env_and_upload", fake_start)
    monkeypatch.setattr(rollout_mod, "_run_environment_setup_commands", AsyncMock())
    monkeypatch.setattr(rollout_mod, "_run_environment_healthcheck", AsyncMock())
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=1)
    await trial.start()
    assert asked == [uploads]

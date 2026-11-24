"""What removes names from a trial's sandbox-reachable folders runs only once the sandbox is gone (#1247, F37).

put_back's removal of names test.sh didn't write, and the drop of links that
leave agent/ or artifacts/, both act by name in folders the sandbox reaches.
While a process in the sandbox still runs it can swap a folder for a link
between a look and an unlink, so both run in the pass after the sandbox's
stop (``_after_sandbox_stop``): inline, or later in a job, where Docker's stop
is queued and ``_teardown_sandbox`` returns before the sandbox is gone. A kept
or caller-owned sandbox never gets that pass.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import benchflow.rollout as rollout_mod
from benchflow.rollout import Rollout
from benchflow.task import verifier_outputs as vo
from benchflow.task.paths import RolloutPaths


def _trial(tmp_path: Path) -> tuple[Path, MagicMock]:
    """A trial whose verifier outputs were taken, a name planted in
    verifier/ after that, and an escaping link in agent/."""
    trial = tmp_path / "trial"
    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    (trial / "verifier" / "reward.txt").write_text("1\n")
    (trial / "integrity").mkdir(exist_ok=True)
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"status": "checked", "flags": [], "sections": {}})
    )
    holder = MagicMock()
    holder._benchflow_verifier_outputs_at_exit = vo.take(trial)
    holder._benchflow_verifier_written_after_exit = {}
    (trial / "verifier" / "extra.txt").write_text("x")
    (trial / "agent").mkdir(exist_ok=True)
    (trial / "agent" / "out").symlink_to("/etc/hostname")
    return trial, holder


def test_the_pass_after_the_stop_removes_plants_and_escaping_links(tmp_path):
    trial, holder = _trial(tmp_path)

    rollout_mod._after_sandbox_stop(trial, holder)

    assert not (trial / "verifier" / "extra.txt").exists()
    assert not (trial / "agent" / "out").is_symlink()
    assert (trial / "verifier" / "reward.txt").read_text() == "1\n"


def _rollout(trial: Path, env: object) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._environment = None
    rollout._env = env
    rollout._env_externally_owned = False
    rollout._agent_sandbox_stopped = False
    rollout._task_tmp = None
    rollout._recovery_lease_owned = False
    rollout._rollout_dir = None
    rollout._rollout_paths = SimpleNamespace(rollout_dir=trial)
    rollout._config = SimpleNamespace(task_path=trial, keep_sandbox="never")
    rollout._teardown_done = asyncio.Event()
    return rollout


@pytest.mark.asyncio
async def test_the_teardown_removes_only_after_the_stop_returns(tmp_path, monkeypatch):
    trial, holder = _trial(tmp_path)
    order: list[str] = []

    async def stop(delete: bool = False) -> None:
        # Still up while it stops: nothing may be removed yet.
        assert (trial / "verifier" / "extra.txt").exists()
        assert (trial / "agent" / "out").is_symlink()
        order.append("stop")

    holder.stop = stop
    plain = rollout_mod._after_sandbox_stop

    def after(rollout_dir, env):
        order.append("after")
        plain(rollout_dir, env)

    monkeypatch.setattr(rollout_mod, "_after_sandbox_stop", after)
    await _rollout(trial, holder)._teardown_sandbox()

    assert order == ["stop", "after"]
    assert not (trial / "verifier" / "extra.txt").exists()
    assert not (trial / "agent" / "out").is_symlink()


@pytest.mark.asyncio
async def test_a_failed_stop_or_a_kept_sandbox_removes_nothing(tmp_path, monkeypatch):
    trial, holder = _trial(tmp_path)

    async def failing_stop(delete: bool = False) -> None:
        raise RuntimeError("the daemon did not answer")

    holder.stop = failing_stop
    await _rollout(trial, holder)._teardown_sandbox()
    assert (trial / "verifier" / "extra.txt").exists()
    assert (trial / "agent" / "out").is_symlink()

    async def kept(self, env, task_tmp):
        return True

    async def never(delete: bool = False) -> None:
        raise AssertionError("a kept sandbox is not stopped")

    holder.stop = never
    monkeypatch.setattr(Rollout, "_keep_sandbox", kept)
    await _rollout(trial, holder)._teardown_sandbox()
    assert (trial / "verifier" / "extra.txt").exists()
    assert (trial / "agent" / "out").is_symlink()

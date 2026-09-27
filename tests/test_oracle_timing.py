"""An oracle rollout records how long solve.sh ran.

Oracle trials recorded ``timing`` with only ``environment_setup``,
``verifier`` and ``total``: the time solve.sh ran appeared nowhere, so a slow
reference solution could not be told apart from slow sandbox setup. Agent rollouts
record ``agent_execution``; the oracle now records it too.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig


def _task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Write hello.txt.\n")
    return task_dir


@pytest.mark.asyncio
async def test_oracle_rollout_records_agent_execution(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.rollout as rollout_module

    rollout = Rollout(RolloutConfig(task_path=_task(tmp_path / "t"), agent="oracle"))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__oracle"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()
    rollout._env = AsyncMock()
    rollout._agent_cwd = "/app"

    async def slow_oracle(*_a, **_k):
        await asyncio.sleep(0.3)
        return [{"type": "oracle", "return_code": 0}], "oracle"

    monkeypatch.setattr(rollout_module, "_run_oracle", slow_oracle)

    async def verify():
        rollout._rewards = {"reward": 1.0}
        rollout._phase = "verified"
        return rollout._rewards

    async def cleanup():
        rollout._phase = "cleaned"

    rollout.verify = AsyncMock(side_effect=verify)
    rollout.cleanup = AsyncMock(side_effect=cleanup)

    await rollout.run()

    timing = json.loads((trial / "result.json").read_text())["timing"]
    assert timing.get("agent_execution", 0) >= 0.3

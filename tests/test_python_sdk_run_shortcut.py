"""``bf.run("agent", task_path=...)`` honours its RuntimeConfig.

#378 wired ``RuntimeConfig.rollout_name`` and ``timeout`` into the
``Agent + Environment`` path; the string shortcut kept dropping both, so the
same config named and timed a rollout differently depending on the calling
form. ``RuntimeConfig.max_rounds``, ``snapshot_policy`` and ``reward_stream``
were never read by anything; setting them now warns. Passing an
``Environment`` object with an agent name used to be ignored silently (the run
built a second, Docker sandbox).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

import benchflow as bf
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.runtime import Environment, RuntimeConfig

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"


@pytest.fixture
def captured(monkeypatch) -> list[RolloutConfig]:
    seen: list[RolloutConfig] = []

    class _Stub:
        async def run(self) -> RolloutResult:
            return RolloutResult(task_name="hello-world-task")

    async def fake_create(config: RolloutConfig) -> Any:
        seen.append(config)
        return _Stub()

    monkeypatch.setattr(Rollout, "create", staticmethod(fake_create))
    return seen


@pytest.mark.asyncio
async def test_string_shortcut_forwards_rollout_name_and_timeout(captured) -> None:
    await bf.run(
        "oracle",
        task_path=TASK_PATH,
        config=RuntimeConfig(rollout_name="my-run", timeout=42),
    )
    (config,) = captured
    assert config.rollout_name == "my-run"
    assert config.timeout == 42


@pytest.mark.asyncio
async def test_string_shortcut_keeps_the_task_timeout_by_default(captured) -> None:
    await bf.run("oracle", task_path=TASK_PATH, config=RuntimeConfig(rollout_name="x"))
    assert captured[0].timeout is None


@pytest.mark.asyncio
async def test_string_shortcut_keeps_the_sandbox_string(captured) -> None:
    await bf.run("oracle", env="daytona", task_path=TASK_PATH)
    assert captured[0].environment == "daytona"


@pytest.mark.asyncio
async def test_string_shortcut_rejects_an_environment_object(captured) -> None:
    env = Environment(inner=object(), task_path=TASK_PATH, sandbox="daytona")
    with pytest.raises(TypeError, match="Agent"):
        await bf.run("oracle", env, task_path=TASK_PATH)
    assert captured == []


@pytest.mark.parametrize(
    "field,value",
    [("max_rounds", 3), ("snapshot_policy", "every-turn"), ("reward_stream", False)],
)
def test_unused_runtime_config_fields_warn(field: str, value: Any) -> None:
    with pytest.warns(DeprecationWarning, match=field):
        RuntimeConfig(**{field: value})


def test_default_runtime_config_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RuntimeConfig(timeout=10)

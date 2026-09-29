"""Every bf.run form uses the task's own agent timeout by default.

``RuntimeConfig.timeout`` defaulted to 900 s, which the Agent + Environment
form always applied, overriding the task's ``[agent] timeout_sec``; the
agent-name shortcut kept the task's own. Now ``timeout`` defaults to ``None``
(the task's value) in every form, an explicit value overrides, and the Agent +
Environment form warns (FutureWarning) when a task's timeout differs from the
old 900 s default, since such a caller may have relied on it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

import benchflow as bf
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"  # timeout_sec 300


class _FakeInner:
    async def start(self, *a: Any, **kw: Any) -> None: ...

    async def stop(self, *a: Any, **kw: Any) -> None: ...


@pytest.fixture
def captured(monkeypatch) -> list[RolloutConfig]:
    seen: list[RolloutConfig] = []
    real_create = Rollout.create

    async def fake_create(config: RolloutConfig) -> Rollout:
        seen.append(config)
        return await real_create(config)

    async def fake_run(self: Rollout) -> RolloutResult:
        return RolloutResult(task_name="hello-world-task", rollout_name="r")

    monkeypatch.setattr(Rollout, "create", staticmethod(fake_create))
    monkeypatch.setattr(Rollout, "run", fake_run)
    return seen


def test_runtime_config_timeout_defaults_to_the_task() -> None:
    assert RuntimeConfig().timeout is None


@pytest.mark.asyncio
async def test_agent_form_uses_task_timeout_and_warns_about_the_old_default(
    captured,
) -> None:
    env = Environment(inner=_FakeInner(), task_path=TASK_PATH, sandbox="docker")
    with pytest.warns(FutureWarning, match="300"):
        await Runtime(env, Agent("claude-agent-acp", "m")).execute()
    assert captured[0].timeout is None


@pytest.mark.asyncio
async def test_agent_form_explicit_timeout_overrides_without_warning(captured) -> None:
    env = Environment(inner=_FakeInner(), task_path=TASK_PATH, sandbox="docker")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        await Runtime(
            env, Agent("claude-agent-acp", "m"), RuntimeConfig(timeout=900)
        ).execute()
    assert captured[0].timeout == 900


@pytest.mark.asyncio
async def test_string_form_forwards_an_explicit_900(captured) -> None:
    await bf.run("oracle", task_path=TASK_PATH, config=RuntimeConfig(timeout=900))
    assert captured[0].timeout == 900


@pytest.mark.asyncio
async def test_string_form_default_is_the_task(captured) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        await bf.run("oracle", task_path=TASK_PATH)
    assert captured[0].timeout is None

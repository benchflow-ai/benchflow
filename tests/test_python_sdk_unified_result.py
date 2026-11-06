"""Every bf.run form returns RolloutResult.

``bf.run(bf.Agent(...), bf.Environment...)`` / ``Runtime.execute()`` used to
return a separate ``RuntimeResult`` with a different attribute set. It now
returns the same ``RolloutResult`` as every other form; the attributes only
``RuntimeResult`` had (``verified``, ``messages``, ``snapshots``) keep working
with a ``DeprecationWarning``, and constructing ``RuntimeResult`` warns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchflow.models import RolloutResult
from benchflow.rollout import Rollout
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig, RuntimeResult

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"


class _FakeInner:
    async def start(self, *a: Any, **kw: Any) -> None: ...

    async def stop(self, *a: Any, **kw: Any) -> None: ...


@pytest.mark.asyncio
async def test_agent_environment_form_returns_rollout_result(monkeypatch) -> None:
    produced = RolloutResult(
        task_name="hello-world-task",
        rollout_name="r1",
        rewards={"reward": 1.0},
        n_input_tokens=5,
    )

    async def fake_run(self: Rollout) -> RolloutResult:
        self._rollout_dir = Path("/tmp/unified-result-dir")
        return produced

    monkeypatch.setattr(Rollout, "run", fake_run)
    env = Environment(inner=_FakeInner(), task_path=TASK_PATH, sandbox="docker")
    result = await Runtime(env, Agent("claude-agent-acp", "claude-haiku-4-5")).execute()

    assert type(result) is RolloutResult
    assert result is produced
    # Fields RuntimeResult dropped are now there too.
    assert result.n_input_tokens == 5
    assert result.rollout_dir == Path("/tmp/unified-result-dir")
    assert result.reward == 1.0 and result.passed


@pytest.mark.parametrize(
    "rewards,error,expected",
    [
        ({"reward": 1.0}, None, True),
        ({"reward": 0.0}, None, True),
        (None, "boom", False),
    ],
)
def test_verified_is_kept_with_a_warning(rewards, error, expected) -> None:
    result = RolloutResult(task_name="t", rewards=rewards, error=error)
    with pytest.warns(DeprecationWarning, match="score_outcome"):
        assert result.verified is expected


@pytest.mark.parametrize("name", ["messages", "snapshots"])
def test_never_populated_runtime_fields_warn_and_are_empty(name: str) -> None:
    with pytest.warns(DeprecationWarning, match=name):
        assert getattr(RolloutResult(task_name="t"), name) == []


def test_constructing_runtime_result_warns() -> None:
    with pytest.warns(DeprecationWarning, match="RolloutResult"):
        r = RuntimeResult(
            task_name="t",
            rollout_name="r",
            reward=1.0,
            rewards={"reward": 1.0},
            n_tool_calls=0,
            error=None,
            verifier_error=None,
            trajectory=[],
        )
    assert r.passed


def test_runtime_execute_is_annotated_with_rollout_result() -> None:
    import typing

    hints = typing.get_type_hints(Runtime.execute)
    assert hints["return"] is RolloutResult
    assert RuntimeConfig  # imported for the module's public surface

"""Caller mistakes fail before a sandbox starts.

A misspelt agent (``claud-agent-acp``) started a Daytona
sandbox and failed 60 s later as "ACP initialize timed out" (category
``pipe_closed``); a missing task directory came back as an errored result with
no category; ``bf.run_sync("<task dir>")`` reported the path as an unknown
agent; and ``bf.run_batch(["<path>"])`` raised ``AttributeError``. The SDK entry
points now check these before anything runs. Raw agent commands (with a space
or a ``/``) are still accepted, as the registry allows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import benchflow as bf
from benchflow.rollout import Rollout, RolloutConfig

TASK = Path(__file__).parent / "examples" / "hello-world-task"


@pytest.fixture
def no_rollouts(monkeypatch) -> list[RolloutConfig]:
    started: list[RolloutConfig] = []

    async def fake_create(config: RolloutConfig) -> Rollout:
        started.append(config)
        raise AssertionError("a rollout must not start")

    monkeypatch.setattr(Rollout, "create", staticmethod(fake_create))
    return started


def test_misspelt_agent_is_refused_with_a_suggestion(no_rollouts) -> None:
    with pytest.raises(ValueError, match="did you mean 'claude-agent-acp'"):
        bf.run_sync(RolloutConfig(task_path=TASK, agent="claud-agent-acp"))
    assert no_rollouts == []


def test_misspelt_agent_in_the_string_form(no_rollouts) -> None:
    with pytest.raises(ValueError, match="did you mean"):
        bf.run_sync("codx-acp", task_path=TASK)


def test_missing_task_dir_raises(no_rollouts, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope"):
        bf.run_sync(RolloutConfig(task_path=tmp_path / "nope", agent="oracle"))


def test_unknown_sandbox_is_refused(no_rollouts) -> None:
    with pytest.raises(ValueError, match="daytona"):
        bf.run_sync(
            RolloutConfig(task_path=TASK, agent="oracle", environment="daytonaa")
        )


def test_a_task_path_given_as_the_agent_is_explained(no_rollouts) -> None:
    with pytest.raises(TypeError, match="first argument is the agent"):
        bf.run_sync(str(TASK))


def test_batch_items_must_be_configs(no_rollouts) -> None:
    with pytest.raises(TypeError, match=r"item 0 .* RolloutConfig"):
        bf.run_batch([str(TASK)])


def test_batch_is_checked_before_anything_starts(no_rollouts) -> None:
    with pytest.raises(ValueError, match="did you mean"):
        bf.run_batch(
            [
                RolloutConfig(task_path=TASK, agent="oracle"),
                RolloutConfig(task_path=TASK, agent="claud-agent-acp"),
            ]
        )
    assert no_rollouts == []


@pytest.mark.parametrize("agent", ["my-agent --serve", "/opt/agents/run.sh", "dummy"])
def test_raw_commands_and_unrelated_names_pass(agent: str) -> None:
    from benchflow.runtime import check_rollout_config

    check_rollout_config(RolloutConfig(task_path=TASK, agent=agent))

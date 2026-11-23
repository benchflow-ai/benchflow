"""Branching a rollout whose model calls go through BenchFlow's LiteLLM proxy.

Guards 0.10.0rc6's fix: ``branch()`` refused any rollout with an active provider
runtime ("needs a runtime fork contract"), so an inline ``bench branch
--checkpoint-after-prompt 1`` on an API-key model failed after the parent's
paid prompt. The fork now hands the parent's proxy off the way a Scene role
switch does: stopped after the agent disconnects and before the checkpoint,
kept with the retired runtimes, and the agent env put back to the one the
first connect started from.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.paths import RolloutPaths

SOURCE_ENV = {"DEEPSEEK_API_KEY": "sk-upstream-test", "PATH": "/usr/bin"}
PROXY_ENV = {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "LITELLM_MASTER_KEY": "sk-runtime-test",
}


def _litellm_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        kind="litellm",
        source_env=dict(SOURCE_ENV),
        master_key="sk-runtime-test",
        stopped=False,
        server=SimpleNamespace(trajectory=None),
    )


def _rollout(tmp_path, calls: list[str]) -> Rollout:
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._rollout_dir = tmp_path / "parent"
    rollout._rollout_paths = RolloutPaths(rollout._rollout_dir)
    rollout._rollout_paths.mkdir()

    async def snapshot(*_args, **_kwargs):
        calls.append("checkpoint")
        return StateSnapshot(id="parent", path="/tmp/snapshot")

    rollout._environment = type(
        "Environment", (), {"snapshot": snapshot, "restore": AsyncMock()}
    )()

    async def disconnect():
        calls.append("disconnect")

    rollout.disconnect = disconnect
    return rollout


def _planes(calls: list[str], *, fails: bool = False) -> SimpleNamespace:
    async def stop(runtime, **_kwargs):
        calls.append("stop")
        if fails:
            raise OSError("proxy stop failed")
        runtime.stopped = True

    return SimpleNamespace(stop_provider_runtime=stop)


async def test_fork_hands_the_parent_proxy_off_before_the_checkpoint(tmp_path):
    """B1, B2: stopped once after the agent disconnected and before the
    checkpoint, retained, and the agent env is the first connect's again."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, calls)
    runtime = _litellm_runtime()
    rollout._planes = _planes(calls)
    rollout._usage_runtime = runtime
    rollout._agent_env = dict(PROXY_ENV)
    during: list[tuple] = []

    async def child(node):
        calls.append("child")
        during.append((rollout._usage_runtime, list(rollout._retired_usage_runtimes)))
        return 1

    assert await rollout.branch(2, child) == 1

    assert calls[:3] == ["disconnect", "stop", "checkpoint"]
    assert calls.count("stop") == 1
    assert runtime.stopped is True
    assert rollout._usage_runtime is None
    assert rollout._retired_usage_runtimes == [runtime]
    assert rollout._agent_env == SOURCE_ENV
    # Each in-place child starts its own proxy with none of the parent's
    # before it, as a --from-checkpoint child does.
    assert during == [(None, []), (None, [])]


async def test_a_failed_stop_fails_the_fork_before_any_child(tmp_path):
    """B3: the runtime stays in the rollout for cleanup to import."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, calls)
    runtime = _litellm_runtime()
    rollout._planes = _planes(calls, fails=True)
    rollout._usage_runtime = runtime
    rollout._agent_env = dict(PROXY_ENV)
    child = AsyncMock(return_value=1)

    with pytest.raises(OSError, match="proxy stop failed"):
        await rollout.branch(2, child)

    child.assert_not_called()
    assert "checkpoint" not in calls
    assert rollout._usage_runtime is runtime
    assert rollout._retired_usage_runtimes == []
    assert rollout._agent_env == PROXY_ENV


async def test_a_runtime_without_its_source_env_is_refused_before_disconnect(
    tmp_path,
):
    """B4: nothing to start the children's proxies from."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, calls)
    runtime = SimpleNamespace(kind="bedrock")
    rollout._planes = _planes(calls)
    rollout._usage_runtime = runtime

    with pytest.raises(RuntimeError, match=r"model runtime \(bedrock\) keeps no"):
        await rollout.branch(2, AsyncMock(return_value=1))

    assert calls == []
    assert rollout._usage_runtime is runtime


async def test_a_child_s_own_retired_proxy_never_joins_the_parent(tmp_path):
    """Review of #382: a proxy an in-place child retired while it ran (its
    connect replaced it) is the child's; the parent's ledger keeps only its own."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, calls)
    runtime = _litellm_runtime()
    rollout._planes = _planes(calls)
    rollout._usage_runtime = runtime
    rollout._agent_env = dict(PROXY_ENV)

    async def child(node):
        # What Rollout.connect does when ensure_litellm_runtime replaces the
        # child's proxy: the old one is stopped and retired.
        rollout._retired_usage_runtimes.append(_litellm_runtime())
        return 1

    assert await rollout.branch(2, child) == 1
    assert rollout._retired_usage_runtimes == [runtime]

"""Regression coverage for final rollout lifecycle ordering and teardown."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import benchflow.rollout as rollout_module
from benchflow.rollout import Rollout, RolloutConfig


def _rollout_for_verify(tmp_path: Path) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=tmp_path / "task")
    rollout._trajectory = [{"type": "message"}]
    rollout._trajectory_source = "acp"
    rollout._env = object()
    rollout._task = object()
    rollout._rollout_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    rollout._timing = {}
    rollout._planes = object()
    rollout._diagnostics = Mock()
    rollout._agent_cwd = "/app"
    rollout.disconnect = AsyncMock()
    return rollout


@pytest.mark.asyncio
async def test_hard_verify_disconnects_before_publishing_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1109: hard verify must quiesce the agent first."""
    rollout = _rollout_for_verify(tmp_path)
    calls: list[str] = []
    rollout.disconnect.side_effect = lambda **kwargs: calls.append("disconnect")

    async def publish(*args, **kwargs):
        calls.append("publish")

    async def verify(*args, **kwargs):
        calls.append("verify")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(rollout_module, "_publish_trajectory_for_verifier", publish)
    monkeypatch.setattr(rollout_module, "_verify_rollout", verify)

    assert await Rollout.verify(rollout) == {"reward": 1.0}
    assert calls == ["disconnect", "publish", "verify"]
    rollout.disconnect.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_soft_verify_does_not_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1109: intermediate verification stays connected."""
    rollout = _rollout_for_verify(tmp_path)
    rollout._rollout_paths.verifier_dir = tmp_path / "verifier"
    rollout._agent_cwd = "/app"
    rollout._planes = SimpleNamespace(
        clear_verifier_output_dir=AsyncMock(),
        ensure_legacy_app_dir=AsyncMock(),
        cleanup_verifier_python_hooks=AsyncMock(),
        verifier=Mock(
            return_value=SimpleNamespace(
                verify=AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0}))
            )
        ),
    )
    rollout._task = SimpleNamespace(
        task_dir=tmp_path / "task",
        config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=5)),
    )
    rollout._env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(stdout=""))
    )
    monkeypatch.setattr(
        rollout_module, "_ensure_canonical_rewards", lambda rewards, task: rewards
    )

    rewards, _, error = await Rollout.soft_verify(rollout)

    assert rewards == {"reward": 1.0}
    assert error is None
    rollout.disconnect.assert_not_awaited()


def _rollout_for_cleanup(tmp_path: Path, disconnect_error: BaseException) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=tmp_path / "task")
    rollout._trajectory = []
    rollout._capture_partial_acp_trajectory = Mock()
    rollout.disconnect = AsyncMock(side_effect=disconnect_error)
    rollout._agent_launch = ""
    rollout._acp_client = None
    rollout._usage_runtime = object()
    rollout._usage_metrics = {"usage_source": "unavailable"}
    rollout._native_usage_metrics = None
    rollout._provider_failure_cached = None
    rollout._provider_auth_status_cached = None
    rollout._api_failure_summary_cached = None
    rollout._write_llm_trajectory = Mock()
    rollout._reconcile_acp_tool_evidence = Mock()
    rollout._finalize_usage_metrics = Mock()
    rollout._enforce_required_usage_tracking = Mock()
    rollout._planes = SimpleNamespace(
        stop_provider_runtime=AsyncMock(),
        extract_usage=Mock(return_value={"usage_source": "unavailable"}),
    )
    rollout._environment = SimpleNamespace(teardown=AsyncMock())
    rollout._env = SimpleNamespace(stop=AsyncMock())
    rollout._env_externally_owned = False
    rollout._task_tmp = tmp_path / "task-copy"
    rollout._task_tmp.mkdir()
    return rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("disconnect failed"), asyncio.CancelledError("cancelled")]
)
async def test_cleanup_finishes_teardown_and_preserves_disconnect_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """Guards PR #1109: teardown continues without losing disconnect failure."""
    rollout = _rollout_for_cleanup(tmp_path, failure)
    monkeypatch.setattr(
        rollout_module, "_provider_failure_from_runtime", lambda runtime: None
    )
    monkeypatch.setattr(
        rollout_module,
        "_provider_api_failure_summary_from_runtime",
        lambda runtime: None,
    )

    with pytest.raises(type(failure)) as caught:
        await Rollout.cleanup(rollout)

    assert caught.value is failure
    rollout._planes.stop_provider_runtime.assert_awaited_once()
    assert rollout._environment is None
    rollout._env.stop.assert_awaited_once_with(delete=True)
    assert not rollout._task_tmp.exists()
    assert rollout._phase == "cleaned"


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_active_primary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1109: cleanup errors stay secondary to an active failure."""
    rollout = _rollout_for_cleanup(tmp_path, RuntimeError("disconnect failed"))
    monkeypatch.setattr(
        rollout_module, "_provider_failure_from_runtime", lambda runtime: None
    )
    monkeypatch.setattr(
        rollout_module,
        "_provider_api_failure_summary_from_runtime",
        lambda runtime: None,
    )

    async def fail_then_cleanup() -> None:
        try:
            raise ValueError("primary failure")
        finally:
            await Rollout.cleanup(rollout)

    with pytest.raises(ValueError, match="primary failure"):
        await fail_then_cleanup()

    rollout._planes.stop_provider_runtime.assert_awaited_once()
    rollout._env.stop.assert_awaited_once_with(delete=True)
    assert not rollout._task_tmp.exists()

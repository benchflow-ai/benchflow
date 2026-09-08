"""Regression coverage for final rollout lifecycle ordering and teardown."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import benchflow.rollout as rollout_module
from benchflow.rollout import Rollout, RolloutConfig


class _LocalShellEnv:
    async def exec(self, command: str, timeout_sec: int):
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_sec
        )
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )


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
    """Guards rollout-finalization PR: hard verify must quiesce agent first."""
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
    rollout.disconnect.assert_awaited_once_with(require_terminated=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["close", "kill", "still-running"])
async def test_hard_verify_aborts_when_agent_termination_is_unproven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Guards rollout-finalization PR: verifier cannot race a live agent."""
    rollout = _rollout_for_verify(tmp_path)
    rollout.disconnect = Rollout.disconnect.__get__(rollout, Rollout)
    rollout._is_session_factory = False
    rollout._capture_partial_acp_trajectory = Mock()
    rollout._collect_native_acp_usage = Mock()
    rollout._session = object()
    rollout._session_adapter = object()
    rollout._agent_launch = "/opt/benchflow/bin/claude-agent-acp"
    rollout._active_role = None
    rollout._session_tool_count = 0
    rollout._session_traj_count = 0
    close_error = RuntimeError("close failed") if failure == "close" else None
    rollout._acp_client = SimpleNamespace(close=AsyncMock(side_effect=close_error))
    if failure == "kill":
        exec_mock = AsyncMock(side_effect=RuntimeError("exec failed"))
    else:
        return_code = 1 if failure == "still-running" else 0
        exec_mock = AsyncMock(return_value=SimpleNamespace(return_code=return_code))
    rollout._env = SimpleNamespace(exec=exec_mock)
    publish = AsyncMock()
    verify = AsyncMock(return_value=({"reward": 1.0}, None, None))
    monkeypatch.setattr(rollout_module, "_publish_trajectory_for_verifier", publish)
    monkeypatch.setattr(rollout_module, "_verify_rollout", verify)

    with pytest.raises(RuntimeError):
        await Rollout.verify(rollout)

    exec_mock.assert_awaited_once()
    publish.assert_not_awaited()
    verify.assert_not_awaited()


def _rollout_for_strict_disconnect(
    agent_name: str, *, skip_agent_install: bool = False, agent_seen: bool = True
) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=Path("task"), skip_agent_install=skip_agent_install
    )
    rollout._is_session_factory = False
    rollout._capture_partial_acp_trajectory = Mock()
    rollout._collect_native_acp_usage = Mock()
    rollout._acp_client = None
    rollout._session = None
    rollout._session_adapter = None
    rollout._agent_name = agent_name if agent_seen else ""
    rollout._agent_launch = f"/opt/benchflow/bin/{agent_name}" if agent_name else ""
    rollout._active_role = None
    rollout._session_tool_count = 0
    rollout._session_traj_count = 0
    rollout._phase = "executed"
    rollout._env = _LocalShellEnv()
    return rollout


@pytest.mark.asyncio
async def test_strict_disconnect_real_shell_succeeds_when_agent_absent() -> None:
    """Guards rollout-finalization PR: attestation command must not match itself."""
    rollout = _rollout_for_strict_disconnect("benchflow-agent-definitely-absent")

    await asyncio.wait_for(
        Rollout.disconnect(rollout, require_terminated=True), timeout=3
    )


@pytest.mark.asyncio
async def test_strict_disconnect_real_shell_terminates_matching_agent() -> None:
    """Guards rollout-finalization PR with a live process and real pkill/pgrep."""
    agent_name = "benchflow-finalization-dummy-agent"
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", f"exec -a {agent_name} sleep 30"
    )
    rollout = _rollout_for_strict_disconnect(agent_name)
    try:
        await asyncio.wait_for(
            Rollout.disconnect(rollout, require_terminated=True), timeout=3
        )
        assert await asyncio.wait_for(process.wait(), timeout=1) < 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_strict_disconnect_allows_verifier_only_runtime() -> None:
    """Guards rollout-finalization PR: no managed agent is vacuously terminated."""
    rollout = _rollout_for_strict_disconnect(
        "", skip_agent_install=True, agent_seen=False
    )
    rollout._env = SimpleNamespace(exec=AsyncMock())

    await Rollout.disconnect(rollout, require_terminated=True)

    rollout._env.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_disconnect_rejects_missing_pattern_after_agent_seen() -> None:
    """Guards rollout-finalization PR: started agent still fails closed."""
    rollout = _rollout_for_strict_disconnect(
        "", skip_agent_install=True, agent_seen=False
    )
    rollout._agent_name = "managed-agent"

    with pytest.raises(RuntimeError, match="process pattern unavailable"):
        await Rollout.disconnect(rollout, require_terminated=True)


@pytest.mark.asyncio
async def test_soft_verify_does_not_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards rollout-finalization PR: intermediate verification stays connected."""
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
async def test_cleanup_finishes_teardown_after_disconnect_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards rollout-finalization PR: first failure cannot leak later resources."""
    failure = RuntimeError("disconnect failed")
    rollout = _rollout_for_cleanup(tmp_path, failure)
    monkeypatch.setattr(
        rollout_module, "_provider_failure_from_runtime", lambda runtime: None
    )
    monkeypatch.setattr(
        rollout_module,
        "_provider_api_failure_summary_from_runtime",
        lambda runtime: None,
    )

    with pytest.raises(RuntimeError, match="disconnect failed") as caught:
        await Rollout.cleanup(rollout)

    assert caught.value is failure
    rollout._planes.stop_provider_runtime.assert_awaited_once()
    assert rollout._environment is None
    rollout._env.stop.assert_awaited_once_with(delete=True)
    assert not rollout._task_tmp.exists()
    assert rollout._phase == "cleaned"


@pytest.mark.asyncio
async def test_cleanup_finishes_teardown_and_preserves_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards rollout-finalization PR: cancellation cannot interrupt teardown ledger."""
    cancellation = asyncio.CancelledError("cancelled")
    rollout = _rollout_for_cleanup(tmp_path, cancellation)
    monkeypatch.setattr(
        rollout_module, "_provider_failure_from_runtime", lambda runtime: None
    )
    monkeypatch.setattr(
        rollout_module,
        "_provider_api_failure_summary_from_runtime",
        lambda runtime: None,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await Rollout.cleanup(rollout)

    assert caught.value is cancellation
    rollout._planes.stop_provider_runtime.assert_awaited_once()
    assert rollout._environment is None
    rollout._env.stop.assert_awaited_once_with(delete=True)
    assert not rollout._task_tmp.exists()
    assert rollout._phase == "cleaned"


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_active_primary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards rollout-finalization PR: cleanup errors stay secondary."""
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

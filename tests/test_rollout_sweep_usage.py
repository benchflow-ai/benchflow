"""Guards FF-sw-rollout cumulative provider accounting against 363d815b."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.acp.session import ACPSession
from benchflow.providers.runtime import ProviderRuntime, extract_usage
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from tests.test_usage_litellm import _trajectory


def _runtime(tokens, cost, model):
    trajectory = _trajectory(
        {"model": model, "usage": {"input_tokens": tokens, "output_tokens": 1}},
        model=model,
    )
    trajectory.metadata["cost_usd"] = cost
    return ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:4000",
        backend_model=model,
        server=SimpleNamespace(trajectory=trajectory, stop=AsyncMock()),
    )


@pytest.mark.parametrize(
    "sequence", [("a", "b"), ("a", "a", "b", "b"), ("a", "b", "a"), ("a", None)]
)
async def test_model_changes_keep_every_runtime_exactly_once(tmp_path, sequence):
    """Retain usage across the C26 close-before-reconnect barrier."""
    roles = [
        Role(name=f"role{i}", agent="claude-agent-acp", model=model)
        for i, model in enumerate(sequence)
    ]
    trial = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene(roles=roles)])
    )
    trial._env = {}
    trial._rollout_dir = tmp_path
    trial._agent_cfg = MagicMock(credential_files=[])
    trial._phase = "idle"
    planes = MagicMock()
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.resolve_agent_env.side_effect = lambda _agent, _model, env: env or {}
    retired = []
    created = []

    async def ensure(**kwargs):
        old, model = kwargs["runtime"], kwargs["model"]
        if old is not None and old.backend_model == model:
            return kwargs["agent_env"], old
        if old is not None:
            await old.server.stop()
            retired.append(old)
        if model is None:
            return kwargs["agent_env"], None
        runtime = _runtime(100 if model == "a" else 10, 9 if model == "a" else 1, model)
        created.append(runtime)
        return kwargs["agent_env"], runtime

    planes.ensure_litellm_runtime = AsyncMock(side_effect=ensure)
    for name in [
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "apply_web_tool_policy",
    ]:
        setattr(planes, name, AsyncMock())

    def connected_session(**kwargs):
        # A reconnect now closes and snapshots the previous session first.
        # Usage/trajectory snapshots are synchronous methods on ACPSession,
        # not coroutines fabricated by a blanket AsyncMock.
        session = ACPSession("dummy-session")
        return AsyncMock(session=session), session, MagicMock(), "agent"

    planes.connect_acp = AsyncMock(side_effect=connected_session)
    planes.stop_provider_runtime = AsyncMock(side_effect=lambda runtime: None)
    # Per-connection credential lifecycle: staging, web policy, and the
    # quiescence of a role transition (C23, C26, C30).
    planes.write_credential_files = AsyncMock(return_value=[])
    planes.upload_subscription_auth = AsyncMock(return_value=[])
    planes.apply_web_tool_policy = AsyncMock()
    planes.quiesce_agent = AsyncMock()
    planes.extract_usage = extract_usage
    trial._planes = planes
    for role in roles:
        await trial.connect_as(role)
    trial._acp_client = None
    trial.disconnect = AsyncMock()
    trial._env = None
    await trial.cleanup()
    expected = sum(
        runtime.server.trajectory.total_provider_tokens for runtime in created
    )
    assert trial._usage_metrics["total_tokens"] == expected
    assert trial._usage_metrics["cost_usd"] == sum(
        runtime.server.trajectory.total_cost_usd for runtime in created
    )
    published = [
        json.loads(line)
        for line in (tmp_path / "trajectory/llm_trajectory.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(published) == len(created)
    assert [line["request"]["body"]["model"] for line in published] == [
        r.backend_model for r in created
    ]
    await trial.cleanup()
    assert trial._usage_metrics["total_tokens"] == expected


@pytest.mark.parametrize("environment", ["docker", "remote-docker", "daytona"])
async def test_new_proxy_live_snapshots_include_stopped_proxy(
    tmp_path, monkeypatch, environment
):
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.runtime import ensure_litellm_runtime
    from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
    from tests.test_litellm_runtime import FakeLiteLLMServer

    old = _runtime(100, 9, "old")
    imported = old.server.trajectory
    old.server.trajectory = _trajectory()

    async def stop():
        old.server.trajectory = imported  # sandbox capture is imported at stop

    old.server.stop = AsyncMock(side_effect=stop)

    class Server(FakeLiteLLMServer):
        def start_live_capture(self, path, **kwargs):
            if hasattr(self, "writer"):
                return
            self.writer = LiveLLMTrajectoryWriter(path, **kwargs)

    async def start(**kwargs):
        return Server("http://127.0.0.1:32123", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", start)
    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", start)
    path = tmp_path / "trajectory.jsonl"
    arguments = dict(
        agent="opencode",
        agent_env={"OPENAI_API_KEY": "dummy-key"},
        model="openai/gpt-4.1-mini",
        runtime=old,
        environment=environment,
        sandbox=SimpleNamespace(),
        live_trajectory_path=path,
        prior_runtimes=(old,),
    )
    _, current = await ensure_litellm_runtime(**arguments)
    current.server.trajectory = _trajectory(
        {"usage": {"input_tokens": 10, "output_tokens": 1}}, model="new"
    )
    for _ in range(3):
        current.server.writer.reconcile(current.server.trajectory)
        assert len(path.read_text().splitlines()) == 2
    # Reuse receives itself in the ledger: it must keep its original prefix.
    arguments.update(runtime=current, prior_runtimes=(old, current))
    await ensure_litellm_runtime(**arguments)
    current.server.writer.reconcile(current.server.trajectory)
    assert len(path.read_text().splitlines()) == 2


async def test_failed_final_import_never_claims_partial_usage_is_complete(tmp_path):
    """Guards FF-sw-rollout while preserving 363d815b's fail-closed stop behavior."""
    trial = Rollout(RolloutConfig(task_path=tmp_path / "task"))
    trial._rollout_dir = tmp_path
    trial._retired_usage_runtimes = [_runtime(100, 9, "old")]
    trial._usage_runtime = _runtime(10, 1, "new")
    trial._planes = SimpleNamespace(
        extract_usage=extract_usage,
        stop_provider_runtime=AsyncMock(
            side_effect=RuntimeError("dummy failed capture import")
        ),
    )
    await trial.cleanup()
    assert trial._usage_metrics["usage_source"] == "unavailable"
    assert trial._usage_metrics["cost_usd"] is None
    assert (
        len((tmp_path / "trajectory/llm_trajectory.jsonl").read_text().splitlines())
        == 2
    )


@pytest.mark.parametrize("change_path", [False, True])
async def test_restarted_live_capture_keeps_preceding_roles(tmp_path, change_path):
    """Guards 24b9615a when a reused proxy restarts its polling task."""
    from benchflow.providers.litellm_runtime import LiteLLMProcess

    class Process(LiteLLMProcess):
        session_id = "dummy-session"
        agent_name = "opencode"

        async def _live_capture_loop(self):
            pass  # a completed poll task must be restartable

    process = Process()
    previous = _runtime(100, 9, "old").server.trajectory
    current = _runtime(10, 1, "new").server.trajectory
    path = tmp_path / "trajectory.jsonl"
    process.start_live_capture(path, preceding_trajectories=(previous,))
    await process._live_capture_task
    process._live_writer.write(current)
    assert len(path.read_text().splitlines()) == 2
    if change_path:
        path = tmp_path / "continued.jsonl"
    process.start_live_capture(path)  # the runtime reuse route
    await process._live_capture_task
    process._live_writer.write(current)
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("environment", ["docker", "remote-docker", "daytona"])
async def test_failed_model_switch_preserves_retired_usage_at_cleanup(
    tmp_path, monkeypatch, environment
):
    """Guards the failed-switch sibling of the usage loss present at 363d815b."""
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.runtime import (
        ensure_litellm_runtime,
        stop_provider_runtime,
    )

    old = _runtime(100, 9, "old")

    async def stop():
        # Real host/sandbox stop imports the callback log, then removes it.
        # Loading that removed log a second time replaces the snapshot.
        if old.server.stop.await_count > 1:
            old.server.trajectory = _trajectory()

    old.server.stop = AsyncMock(side_effect=stop)
    start = AsyncMock(side_effect=RuntimeError("dummy failed proxy start"))
    monkeypatch.setattr(runtime_mod, "_start_host_litellm", start)
    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", start)
    with pytest.raises(RuntimeError, match="dummy failed proxy start"):
        await ensure_litellm_runtime(
            agent="opencode",
            agent_env={"OPENAI_API_KEY": "dummy-key"},
            model="openai/gpt-4.1-mini",
            runtime=old,
            environment=environment,
            sandbox=SimpleNamespace(),
        )
    # Assignment of the replacement failed, so rollout cleanup still owns old.
    await stop_provider_runtime(old)
    assert extract_usage(old)["total_tokens"] == 101
    assert extract_usage(old)["cost_usd"] == 9
    old.server.stop.assert_awaited_once()


async def test_failed_runtime_stop_can_retry_before_becoming_final():
    """Guards the b27 failed-switch fix without suppressing failed imports."""
    from benchflow.providers.runtime import stop_provider_runtime

    runtime = _runtime(100, 9, "old")
    runtime.server.stop = AsyncMock(
        side_effect=[RuntimeError("dummy failed import"), None]
    )
    with pytest.raises(RuntimeError, match="dummy failed import"):
        await stop_provider_runtime(runtime)
    await stop_provider_runtime(runtime)
    await stop_provider_runtime(runtime)
    assert runtime.server.stop.await_count == 2
    assert extract_usage(runtime)["total_tokens"] == 101

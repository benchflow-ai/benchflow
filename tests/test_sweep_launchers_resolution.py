"""The launcher resolver remains the metadata admission/dispatch boundary."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents import registry
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from benchflow.sandbox.cloud_metadata import CloudMetadataPlan
from benchflow.sandbox.lockdown import CloudMetadataPowerError


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "new-role"])
@pytest.mark.parametrize("protected", [True, False])
async def test_resolved_factory_controls_metadata_refusal_and_dispatch(
    tmp_path, monkeypatch, entry, protected
):
    """Guards the launcher/S2 merge c480d19f: honor the rollout resolver once.

    The registry's default is ACP, but rollout resolves a factory. Metadata
    protection must refuse it; an explicit opt-out must dispatch that exact
    entrypoint even if a subsequent resolution would return ACP.
    """
    agent = "claude-agent-acp"
    factory = registry.AgentConfig(
        name=agent,
        install_cmd="true",
        launch_cmd="true",
        protocol="session-factory",
        session_factory="fixture.module:build_agent",
    )
    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[Scene(roles=[Role(name="agent", agent=agent, model="m")])],
            skip_agent_install=True,
        )
    )
    trial._env = MagicMock()
    trial._rollout_dir = tmp_path
    # Unprotected means the run opted out: only an explicit opt-out lets the
    # agent start without the block.
    trial._cloud_metadata = CloudMetadataPlan(
        protected, "fixture opt-out", opt_out="" if protected else "fixture"
    )
    monkeypatch.setattr(
        trial._planes,
        "ensure_litellm_runtime",
        AsyncMock(side_effect=lambda **kw: (kw["agent_env"], None)),
    )
    for method in ("write_credential_files", "apply_web_tool_policy"):
        monkeypatch.setattr(trial._planes, method, AsyncMock())
    monkeypatch.setattr(
        "benchflow.rollout.enforce_codex_apps_policy",
        AsyncMock(side_effect=lambda env, **kw: kw["agent_env"]),
    )
    audit = AsyncMock(side_effect=AssertionError("factory must never reach audit"))
    monkeypatch.setattr(trial._planes, "block_cloud_metadata", audit)
    dispatch = AsyncMock(return_value=(None, None, None, agent))
    monkeypatch.setattr(trial._planes, "connect_session_factory", dispatch)
    live = AsyncMock(return_value=(None, None, None, agent))
    monkeypatch.setattr(trial, "_connect_live_agent", live)
    resolver = MagicMock(side_effect=[factory, registry.AGENTS[agent]])
    monkeypatch.setattr("benchflow.rollout.resolve_agent", resolver)

    async def connect():
        if entry == "connect":
            await trial.connect()
        else:
            await trial.connect_as(Role(name="next", agent=agent, model="m"))

    if protected:
        with pytest.raises(
            CloudMetadataPowerError, match="Choose an ACP or native agent"
        ):
            await connect()
        dispatch.assert_not_awaited()
    else:
        await connect()
        dispatch.assert_awaited_once()
        assert dispatch.await_args.kwargs["session_factory"] == factory.session_factory
    audit.assert_not_awaited()
    live.assert_not_awaited()
    resolver.assert_called_once_with(agent)

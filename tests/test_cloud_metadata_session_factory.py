"""Metadata admission must match the agent's actual execution protocol."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow import rollout as rollout_module
from benchflow.agents import registry
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from benchflow.sandbox.cloud_metadata import CloudMetadataPlan, cloud_metadata_record
from benchflow.sandbox.lockdown import CloudMetadataPowerError

BACKENDS = (
    "docker",
    "remote-docker",
    "daytona",
    "daytona-dind",
    "apple-container",
    "agentcore",
    "modal",
)


@pytest.fixture(autouse=True)
def firewall_audit(monkeypatch):
    # These transport tests supply an explicitly safe image. The real S3
    # audit and its refusal paths are exercised by test_firewall_power.py.
    audit = AsyncMock(return_value=({"state": "clear"}, None))
    monkeypatch.setattr(rollout_module, "check_firewall_power", audit)
    return audit


@pytest.fixture
def session_factory_agents(monkeypatch):
    config = registry.AgentConfig(
        name="omnigent-s2-fixture",
        install_cmd="true",
        launch_cmd="true",
        protocol="session-factory",
        session_factory="tests.test_session_factory_runtime:build_fake_agent",
    )
    monkeypatch.setitem(registry.AGENTS, config.name, config)
    monkeypatch.setitem(registry.AGENT_ALIASES, "s2-sf-alias", config.name)
    for name in (
        config.name,
        "s2-sf-alias",
        "omnigent:s2-fixture",
        "acp/omnigent-s2-fixture",
        "acp/s2-sf-alias",
        "acp/omnigent:s2-fixture",
    ):
        monkeypatch.setitem(registry.AGENT_LAUNCH, name, "true")


def _trial(tmp_path, monkeypatch, primary, plan, harness="acp", environment="docker"):
    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[Scene(roles=[Role(name="agent", agent=primary, model="m")])],
            skip_agent_install=True,
            harness=harness,
            environment=environment,
        )
    )
    trial._env = MagicMock()
    trial._rollout_dir = tmp_path
    trial._cloud_metadata = plan
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
    monkeypatch.setattr(
        trial._planes,
        "block_cloud_metadata",
        AsyncMock(return_value=cloud_metadata_record("on", "dummy proof")),
    )
    result = (None, None, None, primary)
    monkeypatch.setattr(
        trial._planes, "connect_session_factory", AsyncMock(return_value=result)
    )
    monkeypatch.setattr(trial, "_connect_live_agent", AsyncMock(return_value=result))
    return trial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [
        "omnigent-s2-fixture",
        "s2-sf-alias",
        "omnigent:s2-fixture",
        "acp/omnigent-s2-fixture",
        "acp/s2-sf-alias",
        "acp/omnigent:s2-fixture",
    ],
)
@pytest.mark.parametrize("entry", ["connect", "reconnect", "same-role", "new-role"])
@pytest.mark.parametrize("environment", BACKENDS)
async def test_metadata_refuses_session_factory_before_any_audit_or_adapter(
    tmp_path,
    monkeypatch,
    session_factory_agents,
    firewall_audit,
    agent,
    entry,
    environment,
):
    """Guards S2-B18-CONF-1 at 7123fea5 across aliases and both role routes.

    Refusal precedes the audit, so no inventory result, missing utility,
    backend exec API or adapter's choice of user can admit this protocol.
    """
    primary = "claude-agent-acp" if entry == "new-role" else agent
    trial = _trial(
        tmp_path, monkeypatch, primary, CloudMetadataPlan(True), environment=environment
    )
    if entry == "reconnect":
        trial._phase = "installed"
        (tmp_path / "cloud-metadata.json").write_text(
            json.dumps(cloud_metadata_record("on", "previous connection"))
        )
    with pytest.raises(CloudMetadataPowerError, match="Choose an ACP or native agent"):
        if entry in {"connect", "reconnect"}:
            await trial.connect()
        else:
            await trial.connect_as(Role(name="agent", agent=agent, model="m"))
    trial._planes.block_cloud_metadata.assert_not_awaited()
    # Guards the S3 merge at 90b55b7e: refusal must precede S3's audit too.
    firewall_audit.assert_not_awaited()
    trial._planes.connect_session_factory.assert_not_awaited()
    trial._connect_live_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "new-role"])
@pytest.mark.parametrize(
    "plan",
    [
        None,
        CloudMetadataPlan(False, "the run opts out (k = true)", opt_out="k = true"),
    ],
)
async def test_session_factory_keeps_explicit_unprotected_routes(
    tmp_path, monkeypatch, session_factory_agents, entry, plan
):
    """Guards S2-B18-CONF-1 at 7123fea5 without silently changing task opt-outs."""
    agent = "omnigent-s2-fixture"
    primary = "claude-agent-acp" if entry == "new-role" else agent
    trial = _trial(tmp_path, monkeypatch, primary, plan)
    if entry == "connect":
        await trial.connect()
    else:
        await trial.connect_as(Role(name="agent", agent=agent, model="m"))
    trial._planes.block_cloud_metadata.assert_not_awaited()
    trial._planes.connect_session_factory.assert_awaited_once()


@pytest.mark.asyncio
async def test_role_switch_from_session_factory_to_kernel_launcher_is_admitted(
    tmp_path, monkeypatch, session_factory_agents
):
    """Guards S2-B18-CONF-1 at 7123fea5: admission uses the current role."""
    trial = _trial(
        tmp_path, monkeypatch, "omnigent-s2-fixture", CloudMetadataPlan(True)
    )
    await trial.connect_as(Role(name="agent", agent="claude-agent-acp", model="m"))
    trial._planes.block_cloud_metadata.assert_awaited_once()
    trial._connect_live_agent.assert_awaited_once()
    trial._planes.connect_session_factory.assert_not_awaited()


class _LaunchObserved(Exception):
    """Stop at the live-process boundary, before an external agent can run."""


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "new-role"])
@pytest.mark.parametrize("first", ["acp", "key-error", "value-error"])
async def test_metadata_admission_and_dispatch_share_one_transport_resolution(
    tmp_path, monkeypatch, session_factory_agents, firewall_audit, entry, first
):
    """Guards b661a08f: a second lookup must not change an admitted ACP route.

    Registry resolution may fail or change while the metadata audit awaits;
    admission and dispatch must consume one value even if a later lookup
    would return a session factory. No factory operation may then run.
    """
    trial = _trial(tmp_path, monkeypatch, "claude-agent-acp", CloudMetadataPlan(True))
    initial = {
        "acp": registry.AGENTS["claude-agent-acp"],
        "key-error": KeyError("transient resolution failure"),
        "value-error": ValueError("transient resolution failure"),
    }[first]
    resolver = MagicMock(side_effect=[initial, registry.AGENTS["omnigent-s2-fixture"]])
    monkeypatch.setattr("benchflow.rollout.resolve_agent", resolver)
    if entry == "connect":
        await trial.connect()
    else:
        await trial.connect_as(Role(name="next", agent="claude-agent-acp", model="m"))
    trial._planes.block_cloud_metadata.assert_awaited_once()
    trial._planes.connect_session_factory.assert_not_awaited()
    trial._connect_live_agent.assert_awaited_once()
    resolver.assert_called_once_with("claude-agent-acp")
    firewall_audit.assert_awaited_once()
    assert firewall_audit.await_args.kwargs["can_drop"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", BACKENDS)
@pytest.mark.parametrize("entry", ["connect", "new-role"])
@pytest.mark.parametrize(
    "resolution", ["registered", "acpx", "key-error", "value-error"]
)
async def test_acp_dispatch_and_resolution_fallback_keep_the_barrier_on_retry(
    tmp_path, monkeypatch, environment, entry, resolution
):
    """Guards dc89d6b3: the ACP fallback must retain 70400f55's launch barrier.

    Use the real rollout dispatch, ACP runtime and ContainerTransport. Only
    the wire client and backend process are doubles; observe every start,
    including a retry, rather than mocking the protected launch itself.
    """
    from benchflow.acp import runtime

    agent = "claude-agent-acp"
    if resolution == "acpx":
        config = registry.resolve_agent("acpx/claude-agent-acp")
        agent = config.name
        monkeypatch.setitem(registry.AGENTS, agent, config)
        monkeypatch.setitem(registry.AGENT_LAUNCH, agent, config.launch_cmd)
    trial = _trial(
        tmp_path,
        monkeypatch,
        agent,
        CloudMetadataPlan(True),
        environment=environment,
    )
    monkeypatch.setattr(
        trial, "_connect_live_agent", Rollout._connect_live_agent.__get__(trial)
    )
    if resolution in {"key-error", "value-error"}:
        error = KeyError if resolution == "key-error" else ValueError
        monkeypatch.setattr(
            "benchflow.rollout.resolve_agent", MagicMock(side_effect=error("fixture"))
        )
    trial._agent_launch = "dummy-acp --stdio"
    monkeypatch.setitem(registry.AGENT_LAUNCH, agent, trial._agent_launch)
    process = AsyncMock()
    process.start.side_effect = [ConnectionError("retry fixture"), _LaunchObserved()]
    trial._env.live_process = AsyncMock(return_value=process)
    trial._env.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stdout=""))

    def client(transport, **kwargs):
        fake = MagicMock()
        fake.connect = AsyncMock(side_effect=transport.start)
        fake.close = AsyncMock()
        return fake

    monkeypatch.setattr(runtime, "ACPClient", client)
    monkeypatch.setattr(runtime, "_ACP_CONNECT_BASE_DELAY", 0)
    with pytest.raises(_LaunchObserved):
        if entry == "connect":
            await trial.connect()
        else:
            await trial.connect_as(Role(name="next", agent=agent, model="m"))
    trial._planes.block_cloud_metadata.assert_awaited_once()
    trial._planes.connect_session_factory.assert_not_awaited()
    assert process.start.await_count == 2
    for call in process.start.await_args_list:
        command = call.kwargs["command"]
        assert "dummy-acp --stdio" in command
        for option in (
            "--no-new-privs",
            "--bounding-set=-net_admin,-net_raw",
            "--inh-caps=-all",
            "--ambient-caps=-all",
        ):
            assert option in command


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "codex-acp"])
@pytest.mark.parametrize("user", ["agent", "bench-user", "65534"])
async def test_native_turns_keep_the_same_privilege_barrier(
    tmp_path, monkeypatch, agent, user
):
    """Guards dc89d6b3: both native harnesses keep 70400f55's barrier per turn."""
    from benchflow.native_harness.client import NativeCLIClient
    from benchflow.native_harness.harnesses import native_harness_for

    process = AsyncMock()
    process.start.side_effect = _LaunchObserved()
    process.stderr_tail = ""
    env = MagicMock()
    env.live_process = AsyncMock(return_value=process)
    client = NativeCLIClient(
        env=env,
        harness=native_harness_for(agent),
        agent=agent,
        launch_env={},
        sandbox_user=user,
        cwd="/tmp/workspace",
        rollout_dir=tmp_path,
    )
    monkeypatch.setattr(client, "_kill", AsyncMock())
    for _ in range(2):
        with pytest.raises(_LaunchObserved):
            await client.prompt("dummy prompt")
    assert process.start.await_count == 2
    for call in process.start.await_args_list:
        command = call.kwargs["command"]
        assert f"--reuid={user}" in command
        for option in (
            "--no-new-privs",
            "--bounding-set=-net_admin,-net_raw",
            "--inh-caps=-all",
            "--ambient-caps=-all",
        ):
            assert option in command


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_kernel_launchers_still_start_under_metadata_protection(
    tmp_path, monkeypatch, harness
):
    """Guards S2-B18-CONF-1 at 7123fea5: both protected launchers still start."""
    trial = _trial(
        tmp_path, monkeypatch, "claude-agent-acp", CloudMetadataPlan(True), harness
    )
    await trial.connect()
    trial._planes.block_cloud_metadata.assert_awaited_once()
    trial._connect_live_agent.assert_awaited_once()


HOSTED_BACKENDS = ("daytona", "apple-container", "agentcore", "modal")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "new-role"])
@pytest.mark.parametrize("environment", HOSTED_BACKENDS)
async def test_a_root_session_factory_agent_on_a_provider_starts_provider_managed(
    tmp_path, monkeypatch, session_factory_agents, entry, environment
):
    """A session-factory agent that runs as root on a hosted backend is not
    refused for its transport: the block gives it provider-managed,
    the result says so, and the agent starts. On 55c66cf4 it
    was refused with "Choose an ACP or native agent" before the block ran."""
    agent = "omnigent-s2-fixture"
    primary = "claude-agent-acp" if entry == "new-role" else agent
    trial = _trial(
        tmp_path,
        monkeypatch,
        primary,
        CloudMetadataPlan(True, root_agent=True),
        environment=environment,
    )
    trial._config.sandbox_user = None
    trial._planes.block_cloud_metadata = AsyncMock(
        return_value=cloud_metadata_record("provider-managed", "the provider's host")
    )
    if entry == "connect":
        await trial.connect()
    else:
        await trial.connect_as(Role(name="agent", agent=agent, model="m"))
    trial._planes.block_cloud_metadata.assert_awaited_once()
    trial._planes.connect_session_factory.assert_awaited_once()
    assert trial._cloud_metadata_result == {"state": "provider-managed"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "starts"), [("on", True), ("not_installed", False), ("unproven", False)]
)
async def test_a_root_session_factory_agent_on_docker_is_held_by_the_block(
    tmp_path, monkeypatch, session_factory_agents, state, starts
):
    """On Docker a root session-factory agent gets the block's own checks:
    it starts when the block is on, and the block's stop refuses it
    otherwise (not the transport rule)."""
    from benchflow.sandbox.cloud_metadata import CloudMetadataUnprotectedError

    trial = _trial(
        tmp_path,
        monkeypatch,
        "omnigent-s2-fixture",
        CloudMetadataPlan(True, root_agent=True),
    )
    trial._config.sandbox_user = None
    trial._planes.block_cloud_metadata = AsyncMock(
        return_value=cloud_metadata_record(state, "the block's own result")
    )
    if starts:
        await trial.connect()
        trial._planes.connect_session_factory.assert_awaited_once()
    else:
        with pytest.raises(CloudMetadataUnprotectedError):
            await trial.connect()
        trial._planes.connect_session_factory.assert_not_awaited()
    trial._planes.block_cloud_metadata.assert_awaited_once()

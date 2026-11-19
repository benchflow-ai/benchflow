"""an ACP agent's uid firewall (no-web, denylist, the allowlist
proxy) goes in before its process starts, and a firewall that cannot go in
starts no process.

On bf844d299 ``connect_acp`` started the agent (its transport, connect,
initialize and session/new) and installed the firewall only after: until
then a no-web or proxied agent had the network. The agent's transport
standing in for its process, as in tests/test_firewall_power.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_firewall_power import LaunchEnv

NO_WEB = {
    "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
    "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:9/v1",
}
PROXY = "http://127.0.0.1:18080"
DENYLIST = {
    "BENCHFLOW_EGRESS_DENYLIST": "1",
    "HTTPS_PROXY": PROXY,
    "HTTP_PROXY": PROXY,
    "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:9/v1",
}
ALLOWLIST = {
    **DENYLIST,
    "BENCHFLOW_EGRESS_ALLOWLIST": "1",
    "BENCHFLOW_EGRESS_ALLOW_NETWORKS": "203.0.113.0/24",
}


class _Started(Exception):
    """The agent's process would start here."""


async def _connect(tmp_path, agent_env, sandbox_user, firewall, events):
    from benchflow.acp import runtime

    def transport(*, command, **_):
        events.append("agent process")
        raise _Started

    with (
        patch.object(runtime, "enforce_agent_egress_firewall", firewall),
        patch.object(runtime, "ContainerTransport", side_effect=transport),
    ):
        await runtime.connect_acp(
            env=LaunchEnv(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp --stdio",
            agent_env=dict(agent_env),
            sandbox_user=sandbox_user,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_env", "sandbox_user"),
    [
        (NO_WEB, None),
        (NO_WEB, "agent"),
        (DENYLIST, "agent"),
        (ALLOWLIST, "agent"),
    ],
    ids=["no-web-root", "no-web-agent", "denylist-agent", "allowlist-agent"],
)
async def test_the_firewall_goes_in_before_the_agent_process_starts(
    tmp_path, agent_env, sandbox_user
):
    events: list[str] = []

    async def firewall(env, user, env_vars):
        assert user == sandbox_user
        events.append("firewall")

    with pytest.raises(_Started):
        await _connect(tmp_path, agent_env, sandbox_user, firewall, events)
    assert events == ["firewall", "agent process"]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_env", [NO_WEB, DENYLIST], ids=["no-web", "denylist"])
async def test_a_firewall_that_cannot_go_in_starts_no_agent_process(
    tmp_path, agent_env
):
    """The helper's "status none" under load: the trial stops with the
    reason, and the agent never starts."""
    events: list[str] = []
    reason = (
        "Failed to enforce sandbox-user egress firewall: BenchFlow's helper "
        "could not add the rules to main's namespace (status none)"
    )

    async def firewall(env, user, env_vars):
        events.append("firewall")
        raise RuntimeError(reason)

    with pytest.raises(RuntimeError, match="Failed to enforce sandbox-user egress"):
        await _connect(tmp_path, agent_env, "agent", firewall, events)
    assert events == ["firewall"]


def test_a_firewall_that_cannot_go_in_is_the_sandboxs_setup():
    """The trial ends unscored, as an infrastructure problem (sandbox_setup),
    not as an agent's failure: the agent was never started."""
    from benchflow._utils.scoring import SANDBOX_SETUP, classify_error

    assert (
        classify_error(
            "RuntimeError: Failed to enforce sandbox-user egress firewall: "
            "BenchFlow's helper could not add the rules to main's namespace "
            "(status none)"
        )
        == SANDBOX_SETUP
    )


@pytest.mark.asyncio
async def test_a_root_agent_under_the_egress_proxy_is_refused_before_it_starts(
    tmp_path,
):
    """The egress proxy's modes refuse a root agent (the real firewall code,
    not a stand-in): since that change before the agent's process starts. On
    bf844d299 the root agent started first and was refused after
    session/new."""
    from benchflow.acp import runtime

    events: list[str] = []

    def transport(*, command, **_):
        events.append("agent process")
        raise _Started

    with (
        patch.object(runtime, "ContainerTransport", side_effect=transport),
        pytest.raises(RuntimeError, match="requires a sandbox_user"),
    ):
        await runtime.connect_acp(
            env=LaunchEnv(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp --stdio",
            agent_env=dict(DENYLIST),
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["property", "attribute", "method"])
async def test_status_none_names_the_helpers_own_reason(shape):
    """The helper could not run the uid firewall's script ("status none"):
    the error carries the helper's own reason, given as DockerSandbox gives
    it (a property), as a plain attribute or as a method."""
    from benchflow.sandbox.verifier_network import install_from_helper

    reason = "the metadata firewall helper's image could not be built"

    class Env:
        async def metadata_rules_install(self, command):
            return None

    if shape == "property":
        Env.metadata_rules_unavailable = property(lambda self: reason)
    elif shape == "method":
        Env.metadata_rules_unavailable = lambda self: reason
    env = Env()
    if shape == "attribute":
        env.metadata_rules_unavailable = reason
    with pytest.raises(RuntimeError, match="status none") as raised:
        await install_from_helper(env, "true", phase="agent-uid")
    assert reason in str(raised.value)

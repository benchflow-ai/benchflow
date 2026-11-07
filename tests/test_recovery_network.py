"""GH1136: verifier recovery reinstalls policy without connecting a solver."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import recovery_network as network


def config(mode="denylist"):
    return SimpleNamespace(
        network_mode=mode,
        blocked_hosts=["blocked.test"],
        blocked_urls=["https://example.test/private"],
    )


@pytest.mark.asyncio
async def test_recovery_policy_exact_lists_and_no_model_gateway(monkeypatch):
    events = []
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="1001\n"),
                SimpleNamespace(return_code=0),
            ]
        )
    )

    async def start(runtime, user, policy, **kwargs):
        events.append("proxy")
        assert runtime is env and user == "agent"
        assert policy.blocked_hosts == ("blocked.test",)
        assert policy.blocked_urls == ("https://example.test/private",)
        assert kwargs == {"model_gateway_url": None}

    async def firewall(runtime, user, values):
        events.append("firewall")
        assert runtime is env and user == "agent"
        assert values["HTTPS_PROXY"] == network.PROXY_URL
        assert "LLM_BASE_URL" not in values

    monkeypatch.setattr(network, "start_egress_denylist", start)
    monkeypatch.setattr(network, "enforce_agent_egress_firewall", firewall)
    await network.prepare_recovery_network(env, config(), "agent")
    assert events == ["proxy", "firewall"]
    assert env.exec.call_args.kwargs["user"] == "agent"
    assert "/healthz" in env.exec.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["0", "unknown", ""])
async def test_invalid_uid_never_installs_policy(monkeypatch, identity):
    start = AsyncMock()
    monkeypatch.setattr(network, "start_egress_denylist", start)
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=identity))
    )
    with pytest.raises(RuntimeError, match="non-root"):
        await network.prepare_recovery_network(env, config(), "agent")
    start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["proxy", "firewall", "probe"])
async def test_policy_failures_abort_recovery(monkeypatch, failure):
    start = AsyncMock(side_effect=RuntimeError("proxy") if failure == "proxy" else None)
    firewall = AsyncMock(
        side_effect=RuntimeError("firewall") if failure == "firewall" else None
    )
    monkeypatch.setattr(network, "start_egress_denylist", start)
    monkeypatch.setattr(network, "enforce_agent_egress_firewall", firewall)
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="1001"),
                SimpleNamespace(return_code=1),
            ]
        )
    )
    with pytest.raises(RuntimeError):
        await network.prepare_recovery_network(env, config(), "agent")
    if failure == "proxy":
        firewall.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_modes_do_not_install_denylist():
    env = SimpleNamespace(exec=AsyncMock())
    await network.prepare_recovery_network(env, config("none"), None)
    env.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_web_recovery_restores_uid_firewall_without_gateway(monkeypatch):
    """GH1136: model bootstrap's network access must not remove the UID policy."""
    cfg = config("none")
    cfg.allow_internet = False
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="12345"))
    )
    firewall = AsyncMock()
    proxy = AsyncMock()
    monkeypatch.setattr(network, "enforce_sandbox_uid_egress", firewall)
    monkeypatch.setattr(network, "start_egress_denylist", proxy)
    await network.prepare_recovery_network(env, cfg, "agent")
    firewall.assert_awaited_once_with(env, "agent")
    proxy.assert_not_awaited()
    # A failed policy installation must stop recovery.
    firewall.side_effect = RuntimeError("firewall unavailable")
    with pytest.raises(RuntimeError, match="firewall unavailable"):
        await network.prepare_recovery_network(env, cfg, "agent")

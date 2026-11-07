"""A host model proxy starts while the sandbox guards run (PERF).

The guards (the firewall-power check, then the cloud metadata block) keep
their order and still end before the agent starts; only the host proxy's
start, which runs nothing in the sandbox, moves beside them instead of
before them. Its health wait (about 4 s a trial on a build VM) no longer
adds to the guards' seconds.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow import rollout as rollout_module
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from benchflow.sandbox.cloud_metadata import CloudMetadataPlan, cloud_metadata_record
from benchflow.sandbox.lockdown import EGRESS_DENYLIST_ENV, CloudMetadataPowerError

AGENT = "claude-agent-acp"
FIREWALL = "firewall power: the cloud metadata block"


class _Steps:
    """Records a connection's steps; the proxy start ends only once the
    block has run (``held``), so a proxy started first never ends."""

    def __init__(
        self,
        trial: Rollout,
        monkeypatch: pytest.MonkeyPatch,
        *,
        held: bool = True,
        proxy_env: dict[str, str] | None = None,
        proxy_error: Exception | None = None,
        block_error: Exception | None = None,
        block_hangs: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.release = asyncio.Event()
        if not held:
            self.release.set()
        self.runtime = SimpleNamespace(kind="litellm")

        async def proxy(**kw):
            self.events.append("proxy started")
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.events.append("proxy cancelled")
                raise
            self.events.append("proxy ready")
            if proxy_error is not None:
                raise proxy_error
            return {**kw["agent_env"], **(proxy_env or {})}, self.runtime

        async def firewall(env, user, *, holds, can_drop=True):
            self.events.append("firewall power: " + " + ".join(holds))
            return {"state": "clear"}, None

        async def block(env, user):
            self.events.append("metadata block")
            if block_hangs:
                await asyncio.Event().wait()
            self.release.set()
            if block_error is not None:
                raise block_error
            return cloud_metadata_record("on", "dummy proof")

        async def connect(**kw):
            self.events.append("agent started")
            return (None, None, None, AGENT)

        monkeypatch.setattr(trial._planes, "ensure_litellm_runtime", proxy)
        monkeypatch.setattr(rollout_module, "check_firewall_power", firewall)
        monkeypatch.setattr(trial._planes, "block_cloud_metadata", block)
        monkeypatch.setattr(trial, "_connect_live_agent", connect)


def _trial(tmp_path, monkeypatch, *, environment: str = "docker") -> Rollout:
    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[Scene(roles=[Role(name="agent", agent=AGENT, model="m")])],
            skip_agent_install=True,
            environment=environment,
        )
    )
    trial._env = MagicMock()
    trial._rollout_dir = tmp_path
    trial._cloud_metadata = CloudMetadataPlan(True)
    for method in ("write_credential_files", "apply_web_tool_policy"):
        monkeypatch.setattr(trial._planes, method, AsyncMock())
    monkeypatch.setattr(
        "benchflow.rollout.enforce_codex_apps_policy",
        AsyncMock(side_effect=lambda env, **kw: kw["agent_env"]),
    )
    return trial


def _open(trial: Rollout, entry: str):
    if entry == "connect":
        return trial.connect()
    return trial.connect_as(Role(name="agent", agent=AGENT, model="m"))


ENTRIES = ["connect", "connect_as"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_host_proxy_starts_while_the_guards_run(tmp_path, monkeypatch, entry):
    """The proxy ends only after the block: started first and awaited, as
    before, the connection would never get to the guards."""
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(trial, monkeypatch)
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == [
        "proxy started",
        FIREWALL,
        "metadata block",
        "proxy ready",
        "agent started",
    ]
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "on"
    assert trial._usage_runtime is steps.runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_guard_refusal_waits_for_the_proxy_and_keeps_it_for_clean_up(
    tmp_path, monkeypatch, entry
):
    """As with the proxy first: the refusal raises, the agent never starts,
    and the started proxy is on the rollout, where clean-up stops it."""
    trial = _trial(tmp_path, monkeypatch)
    refusal = CloudMetadataPowerError("sandbox startup refused: test")
    steps = _Steps(trial, monkeypatch, block_error=refusal)
    with pytest.raises(CloudMetadataPowerError, match="startup refused"):
        await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == ["proxy started", FIREWALL, "metadata block", "proxy ready"]
    assert trial._usage_runtime is steps.runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_proxy_failure_raises_once_the_guards_are_done(
    tmp_path, monkeypatch, entry
):
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(
        trial, monkeypatch, proxy_error=RuntimeError("LiteLLM proxy failed to start")
    )
    with pytest.raises(RuntimeError, match="failed to start"):
        await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == ["proxy started", FIREWALL, "metadata block", "proxy ready"]
    assert trial._usage_runtime is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_cancelled_connection_cancels_the_proxy_start(
    tmp_path, monkeypatch, entry
):
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(trial, monkeypatch, block_hangs=True)
    task = asyncio.ensure_future(_open(trial, entry))
    for _ in range(100):
        if "metadata block" in steps.events:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert steps.events == [
        "proxy started",
        FIREWALL,
        "metadata block",
        "proxy cancelled",
    ]
    assert trial._usage_runtime is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_wiring_that_changes_the_rules_holding_the_agent_checks_again(
    tmp_path, monkeypatch, entry
):
    """The guards checked the env before the proxy's wiring; an env after it
    that other rules hold gets its own firewall-power check, before the
    agent starts."""
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(trial, monkeypatch, proxy_env={EGRESS_DENYLIST_ENV: "1"})
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == [
        "proxy started",
        FIREWALL,
        "metadata block",
        "proxy ready",
        "firewall power: the denylist's uid firewall + the cloud metadata block",
        "agent started",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_proxy_in_the_sandbox_still_starts_first(tmp_path, monkeypatch, entry):
    """Its install and start run in the sandbox, so the guards come after."""
    from benchflow.providers.litellm_runtime import proxy_runs_on_host

    assert not proxy_runs_on_host("daytona")
    trial = _trial(tmp_path, monkeypatch, environment="daytona")
    steps = _Steps(trial, monkeypatch, held=False)
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == [
        "proxy started",
        "proxy ready",
        FIREWALL,
        "metadata block",
        "agent started",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_connection_with_an_earlier_proxy_keeps_the_order(
    tmp_path, monkeypatch, entry
):
    """Stopping or reusing an earlier proxy may run in the sandbox."""
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(trial, monkeypatch, held=False)
    trial._usage_runtime = steps.runtime  # reused by this connection
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == [
        "proxy started",
        "proxy ready",
        FIREWALL,
        "metadata block",
        "agent started",
    ]


def test_proxy_placement_is_the_one_ensure_litellm_runtime_uses():
    from benchflow.providers.litellm_runtime import proxy_runs_on_host

    assert proxy_runs_on_host("docker")
    assert not proxy_runs_on_host("docker", force_sandbox_local=True)
    assert not proxy_runs_on_host("daytona")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_the_guards_take_no_time_of_their_own_beside_a_host_proxy_start(
    tmp_path, monkeypatch, entry
):
    """Timed through the connection itself, with nothing but the module's
    public seams (it also runs on the commit before 8560bffc): a host proxy
    that takes 1 s to come up and guards that take 0.8 s together end in
    about 1 s, not the 1.8 s of one after the other."""
    trial = _trial(tmp_path, monkeypatch)

    async def proxy(**kw):
        await asyncio.sleep(1.0)
        return dict(kw["agent_env"]), SimpleNamespace(kind="litellm")

    async def firewall(*args, **kwargs):
        await asyncio.sleep(0.4)
        return {"state": "clear"}, None

    async def block(*args, **kwargs):
        await asyncio.sleep(0.4)
        return cloud_metadata_record("on", "dummy proof")

    async def connect(**kw):
        return (None, None, None, AGENT)

    monkeypatch.setattr(trial._planes, "ensure_litellm_runtime", proxy)
    monkeypatch.setattr(rollout_module, "check_firewall_power", firewall)
    monkeypatch.setattr(trial._planes, "block_cloud_metadata", block)
    monkeypatch.setattr(trial, "_connect_live_agent", connect)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert loop.time() - started < 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_ahead_zero_restores_the_order_from_before(tmp_path, monkeypatch, entry):
    """``BENCHFLOW_HOST_PROXY_AHEAD=0`` gives back the connection's old order,
    the guards after the proxy's start, for bisecting."""
    monkeypatch.setenv("BENCHFLOW_HOST_PROXY_AHEAD", "0")
    trial = _trial(tmp_path, monkeypatch)
    steps = _Steps(trial, monkeypatch, held=False)

    async def slow_proxy(**kw):
        # Beside the guards, they would run while it comes up.
        steps.events.append("proxy started")
        await asyncio.sleep(0.2)
        steps.events.append("proxy ready")
        return dict(kw["agent_env"]), steps.runtime

    monkeypatch.setattr(trial._planes, "ensure_litellm_runtime", slow_proxy)
    await asyncio.wait_for(_open(trial, entry), timeout=10)
    assert steps.events == [
        "proxy started",
        "proxy ready",
        FIREWALL,
        "metadata block",
        "agent started",
    ]

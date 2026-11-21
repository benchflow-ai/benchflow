"""A Modal denylist held at the egress gateway (benchflow.sandbox.egress_gateway).

Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``), against main at
d9c0e5f7, where every Modal trial with a restricted network was refused. The
Modal SDK is a fake (as in tests/test_modal_gpu.py) and the sandbox's
commands are scripted, but the gateway is a real one on loopback, running
BenchFlow's own proxy: the create's allowlist, the start's proofs, each step
that moves the rollout's phase, the logs and the trial's record.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import egress_gateway, modal_impl
from benchflow.sandbox import egress_gateway_server as server
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.egress_denylist import (
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.modal_impl import ModalSandbox
from benchflow.sandbox.setup import _create_benchflow_modal_environment_class
from benchflow.task.config import TaskConfig
from benchflow.task.paths import RolloutPaths
from tests.receipt_fakes import receipt_answer

DENYLIST = (
    '[environment]\nnetwork_mode = "denylist"\n'
    'blocked_hosts = ["blocked.test"]\nblocked_urls = ["paper.test/abs/1"]\n'
)
CONFINED = {"direct": "TimeoutError", "direct6": "OSError", "relay": 200}
REFUSED = {"url": "https://paper.test/abs/1", "status": 403, "blocked": "1"}


def _config(toml: str = DENYLIST) -> TaskConfig:
    return TaskConfig.model_validate_toml('version = "1.0"\n' + toml)


@pytest.fixture
def gateway(monkeypatch):
    folder = Path(tempfile.mkdtemp(prefix="bfeg-"))
    gateway = server.Gateway(
        folder / "state",
        ("127.0.0.1", 0),
        "127.0.0.1:1",
        openssl=shutil.which("openssl") or "/usr/bin/false",
    )
    gateway.open()
    gateway.public = f"127.0.0.1:{gateway.address[1]}"
    monkeypatch.setenv(egress_gateway.ENV, str(gateway.control_path))
    monkeypatch.setattr(egress_gateway, "_LOOPBACK_PUBLIC_OK", True)
    yield gateway
    gateway.close()
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def provider(monkeypatch):
    module = ModuleType("modal")
    module.App = SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock()))
    module.Image = MagicMock()
    module.Secret = MagicMock()
    module.Volume = MagicMock()
    remote = SimpleNamespace(
        object_id="sb-fake",
        filesystem=SimpleNamespace(make_directory=SimpleNamespace(aio=AsyncMock())),
        terminate=SimpleNamespace(aio=AsyncMock()),
        wait=SimpleNamespace(aio=AsyncMock()),
        # Modal's poll: None while the sandbox runs (#1248's platform ends).
        poll=SimpleNamespace(aio=AsyncMock(return_value=None)),
    )
    module.Sandbox = SimpleNamespace(
        create=SimpleNamespace(aio=AsyncMock(return_value=remote))
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setattr(
        "benchflow.sandbox.modal_impl.bfx_delivery.provider_enabled", lambda _: False
    )
    return module


class Script:
    """The sandbox's commands: each recorded, the probes answered as set."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []
        self.probe: dict = dict(CONFINED)
        self.deny: dict = dict(REFUSED)
        #: The sandbox; each command's ``_rlimits`` is kept in :attr:`held`.
        self.sandbox: ModalSandbox | None = None
        self.held: list[object] = []
        #: The su check's answer under ``BENCHFLOW_MODAL_RLIMITS``.
        self.su_check = ""
        #: ``(words, exec)``: a command holding *words* runs on *exec*.
        self.real: tuple[str, object] | None = None

    def answer(self, command: str) -> ExecResult:
        if "/proc/self/limits" in command:
            out = self.su_check
        elif "BF egress-gateway-probe" in command:
            out = "BF egress-gateway-probe " + json.dumps(self.probe)
        elif "BF egress-gateway-deny" in command:
            out = "BF egress-gateway-deny " + json.dumps(self.deny)
        else:
            out = ""
        return ExecResult(stdout=out, stderr="", return_code=0)

    async def __call__(self, command, **kwargs):
        self.commands.append((str(command), kwargs.get("user")))
        self.held.append(getattr(self.sandbox, "_rlimits", None))
        if self.real is not None and self.real[0] in str(command):
            return await self.real[1](command, **kwargs)
        return receipt_answer(str(command), self.answer)

    def ran(self, words: str) -> list[str]:
        return [c for c, _ in self.commands if words in c]


@pytest.fixture
def make_sandbox(tmp_path, provider):
    def make(
        toml: str = DENYLIST,
        session: str = "task__trial-1",
        *,
        factory: bool = False,
        dockerfile: str = "FROM python:3.12-slim\n",
        **kwargs: object,
    ) -> ModalSandbox:
        """*factory*: the launcher ``bench`` runs (BenchFlowModalSandbox,
        whose own ``start`` holds the gateway too)."""
        (tmp_path / "Dockerfile").write_text(dockerfile)
        paths = RolloutPaths(rollout_dir=tmp_path / session)
        paths.mkdir()
        cls = _create_benchflow_modal_environment_class() if factory else ModalSandbox
        sandbox = cls(
            environment_dir=tmp_path,
            environment_name="task",
            session_id=session,
            rollout_paths=paths,
            task_env_config=_config(toml).sandbox,
            **kwargs,
        )
        sandbox._refuse_startup_names = AsyncMock()
        sandbox.exec = Script()
        sandbox.exec.sandbox = sandbox
        sandbox.upload_dir = AsyncMock()
        return sandbox

    return make


def _status(gateway: server.Gateway) -> dict:
    return server.request(gateway.control_path, {"command": "status"})


def _phase(gateway: server.Gateway) -> str:
    (row,) = _status(gateway)["rollouts"]
    return row["phase"]


def _blocked_request(gateway: server.Gateway, hold: egress_gateway.Hold) -> bytes:
    """A request through the gateway, as the sandbox's relay sends it."""
    with socket.create_connection(gateway.address, timeout=10) as sock:
        line = b""
        while not line.endswith(b"\n"):
            line += sock.recv(1)
        nonce = line.decode().split()[1]
        mac = server.answer(hold.token, nonce, hold.rollout)
        sock.sendall(
            f"{hold.rollout} {mac}\n".encode()
            + b"GET http://blocked.test/ HTTP/1.1\r\nHost: blocked.test\r\n\r\n"
        )
        data = b""
        while chunk := sock.recv(4096):
            data += chunk
        return data


@pytest.mark.asyncio
async def test_a_denylist_sandbox_is_created_with_the_gateway_alone_and_proved(
    gateway, make_sandbox, provider
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    sandbox = make_sandbox()
    await sandbox.start(force_build=False)

    kwargs = provider.Sandbox.create.aio.await_args.kwargs
    assert kwargs["outbound_cidr_allowlist"] == ["127.0.0.1/32"]
    assert kwargs["block_network"] is False
    hold = egress_gateway.held(sandbox)
    assert hold is not None and hold.public == gateway.public
    assert hold.sandbox_id == "sb-fake"
    script = sandbox.exec
    order = [
        script.ran("chmod 700 /opt/runtime-egress"),
        script.ran("BF egress-ca ok"),
        script.ran(f"mkdir -p {egress_gateway.RELAY_DIR}"),
        script.ran("relay.json </dev/null"),
        script.ran("BF egress-gateway-probe"),
        script.ran("BF egress-gateway-deny"),
    ]
    assert all(order), order
    flat = [c for c, _ in script.commands]
    assert [flat.index(step[0]) for step in order] == sorted(
        flat.index(step[0]) for step in order
    )
    first = flat.index(order[0][0])
    assert all(user == "root" for _, user in script.commands[first:])
    assert sandbox._persistent_env["HTTPS_PROXY"] == "http://127.0.0.1:18628"
    assert "BENCHFLOW_EGRESS_DENYLIST" not in sandbox._persistent_env
    assert [e["event"] for e in hold.events] == ["registered", "deny", "open"]
    assert _phase(gateway) == "open"
    # The relay's config (token included) went up in one upload, never in a command.
    assert not any(hold.token in c for c in flat)

    await sandbox.stop(True)
    assert egress_gateway.held(sandbox) is None
    assert _status(gateway)["rollouts"] == []
    record = json.loads(
        (sandbox.rollout_paths.rollout_dir / egress_gateway.RECORD_NAME).read_text()
    )
    assert record["outbound_cidr_allowlist"] == ["127.0.0.1/32"]
    assert record["rollout"] == "task__trial-1"
    assert [p["at"] for p in record["proofs"]] == ["start", "start, deny"]
    assert record["events"][-1]["event"] == "deregistered"
    assert hold.token not in json.dumps(record)


@pytest.mark.parametrize(
    "toml",
    ["", "[environment]\nallow_internet = false\n", DENYLIST],
    ids=["public", "no-network", "denylist-without-the-variable"],
)
@pytest.mark.asyncio
async def test_any_other_sandbox_is_created_as_before(
    make_sandbox, provider, monkeypatch, toml
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    monkeypatch.delenv(egress_gateway.ENV, raising=False)
    if toml != DENYLIST:
        monkeypatch.setenv(egress_gateway.ENV, "/nonexistent/control.sock")
    sandbox = make_sandbox(toml)
    await sandbox.start(force_build=False)
    assert (
        "outbound_cidr_allowlist" not in provider.Sandbox.create.aio.await_args.kwargs
    )
    assert egress_gateway.held(sandbox) is None
    assert sandbox.exec.ran("egress-gateway") == []


@pytest.mark.parametrize(
    ("probe", "deny"),
    [
        ({**CONFINED, "direct": "connected"}, REFUSED),
        ({**CONFINED, "direct6": "connected"}, REFUSED),
        ({**CONFINED, "relay": "URLError"}, REFUSED),
        (CONFINED, {"url": "https://paper.test/abs/1", "status": 200}),
    ],
    ids=["direct-v4", "direct-v6", "no-relay", "not-refused"],
)
@pytest.mark.asyncio
async def test_a_failed_start_proof_stops_the_sandbox_and_deregisters(
    gateway, make_sandbox, provider, monkeypatch, probe, deny
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    monkeypatch.setattr(egress_gateway, "HEALTH_TIMEOUT_SEC", 0)
    sandbox = make_sandbox()
    sandbox.exec.probe, sandbox.exec.deny = probe, deny
    remote = provider.Sandbox.create.aio.return_value
    with pytest.raises(egress_gateway.EgressGatewayError):
        await sandbox.start(force_build=False)
    # The stop asked Modal, which said the sandbox runs: a refusal at the
    # gateway is not an end Modal made (#1248's platform_end).
    remote.poll.aio.assert_awaited()
    assert sandbox.platform_end is None
    remote.terminate.aio.assert_awaited()
    assert sandbox._sandbox is None
    assert _status(gateway)["rollouts"] == []


@pytest.mark.asyncio
async def test_a_start_cancelled_while_registering_deregisters(
    gateway, make_sandbox, provider, monkeypatch
):
    """A start cancelled while the gateway registers its rollout (the
    request runs on in its thread) waits for the gateway's answer, so the
    stop deregisters the rollout and nothing of it runs on at the gateway.

    Guards the third review round of PR #1253 (adds
    ``BENCHFLOW_EGRESS_GATEWAY``): the hold was set only once ``register``
    returned, so a cancelled attach stopped the sandbox with no hold and the
    rollout's open proxy ran on to its time to live."""
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    register = gateway._cmd_register

    def slow_register(request: dict) -> dict:
        entered.set()
        release.wait(30)
        try:
            return register(request)
        finally:
            done.set()

    monkeypatch.setattr(gateway, "_cmd_register", slow_register)
    sandbox = make_sandbox()
    start = asyncio.ensure_future(sandbox.start(force_build=False))
    assert await asyncio.to_thread(entered.wait, 30)
    start.cancel()
    await asyncio.sleep(0.2)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await start
    assert await asyncio.to_thread(done.wait, 30)
    provider.Sandbox.create.aio.return_value.terminate.aio.assert_awaited()
    assert _status(gateway)["rollouts"] == []


@pytest.mark.asyncio
async def test_a_gateway_that_does_not_answer_fails_the_create(
    make_sandbox, provider, monkeypatch
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    monkeypatch.setenv(egress_gateway.ENV, "/nonexistent/control.sock")
    sandbox = make_sandbox()
    with pytest.raises(egress_gateway.EgressGatewayError):
        await sandbox.start(force_build=False)
    provider.Sandbox.create.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_hello_takes_a_global_ipv4_address_only(gateway, monkeypatch):
    """The allowlist is the gateway's address on every port: a private,
    shared, loopback, link-local, multicast or reserved one (a mistyped
    ``--public``) is refused before any sandbox.

    Guards the review round of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    monkeypatch.setattr(egress_gateway, "_LOOPBACK_PUBLIC_OK", False)
    control = str(gateway.control_path)
    for public in (
        "10.0.0.5:443",
        "172.16.0.1:443",
        "192.168.1.1:443",
        "100.64.0.1:443",
        "127.0.0.1:443",
        "169.254.169.254:80",
        "0.0.0.0:443",
        "224.0.0.1:443",
    ):
        gateway.public = public
        with pytest.raises(egress_gateway.EgressGatewayError, match="global IPv4"):
            await egress_gateway.hello(control)
    gateway.public = "8.8.8.8:443"
    assert (await egress_gateway.hello(control))["public"] == "8.8.8.8:443"


@pytest.mark.asyncio
@pytest.mark.parametrize("python", ["this python", "cpython 3.12.3"])
async def test_hello_refuses_192_0_0_0_24_on_every_python(gateway, monkeypatch, python):
    """192.0.0.0/24 (IETF protocol assignments) is refused, but for the two
    addresses IANA lists as globally reachable (192.0.0.9 and 192.0.0.10),
    whatever ``ipaddress`` says: cpython before 3.12.4 classed most of it global.

    Guards the second review round of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    if python == "cpython 3.12.3":
        is_global = ipaddress.IPv4Address.is_global.fget
        ietf = ipaddress.IPv4Network("192.0.0.0/24")
        private = ("192.0.0.0/29", "192.0.0.170/31")  # its only private parts then

        def old_is_global(address: ipaddress.IPv4Address) -> bool:
            if address in ietf:
                return not any(address in ipaddress.IPv4Network(n) for n in private)
            return is_global(address)

        monkeypatch.setattr(ipaddress.IPv4Address, "is_global", property(old_is_global))
    monkeypatch.setattr(egress_gateway, "_LOOPBACK_PUBLIC_OK", False)
    control = str(gateway.control_path)
    for public in ("192.0.0.8:443", "192.0.0.100:443", "192.0.0.255:443"):
        gateway.public = public
        with pytest.raises(egress_gateway.EgressGatewayError, match="global IPv4"):
            await egress_gateway.hello(control)
    for public in ("192.0.0.9:443", "192.0.0.10:443"):
        gateway.public = public
        assert (await egress_gateway.hello(control))["public"] == public


@pytest.fixture
async def started(gateway, make_sandbox):
    sandbox = make_sandbox()
    await sandbox.start(force_build=False)
    yield sandbox
    await sandbox.stop(True)


@pytest.mark.asyncio
async def test_the_agent_connection_moves_the_phase_to_deny_and_proves_the_agent(
    gateway, started
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.egress_denylist import denylist_agent_env
    from benchflow.sandbox.lockdown import (
        arm_agent_uid_before_its_proxy,
        enforce_agent_egress_firewall,
    )

    policy = egress_denylist_for(started.task_env_config)
    assert not await arm_agent_uid_before_its_proxy(
        started, "agent", no_web=False, egress_denylist=policy, create_user=True
    )
    agent_env = denylist_agent_env(
        {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000"}, policy
    )
    # Before the egress start the phase is open: the agent is not held.
    with pytest.raises(RuntimeError, match="not held in the deny phase"):
        await enforce_agent_egress_firewall(started, "agent", agent_env)
    await start_egress_denylist(
        started, "agent", policy, model_gateway_url="http://127.0.0.1:4000"
    )
    assert _phase(gateway) == "deny"
    await enforce_agent_egress_firewall(started, "agent", agent_env)
    script = started.exec
    assert script.ran("iptables") == [] and script.ran("proxy.py") == []
    hold = egress_gateway.held(started)
    assert hold.proofs[-1]["user"] == "agent" and hold.proofs[-1]["phase"] == "deny"
    # A no-web launch holds the agent's uid alone: refused here.
    with pytest.raises(RuntimeError, match="no-web"):
        await enforce_agent_egress_firewall(
            started, "agent", {**agent_env, "BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}
        )


@pytest.mark.asyncio
async def test_an_agent_launch_without_the_denylist_marker_is_refused(started):
    """On a held sandbox an agent launch that carries neither the no-web nor
    the denylist marker is refused before the agent starts; on any other
    sandbox such a launch needs nothing, as on main.

    Guards the review round of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``):
    ``enforce_agent_egress_firewall`` returned on the missing markers before
    its held dispatch, so ``enforce_agent``'s refusal could never fire."""
    from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

    agent_env = {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000"}
    with pytest.raises(RuntimeError, match="network_mode='denylist' only"):
        await enforce_agent_egress_firewall(started, "agent", agent_env)
    assert "agent launch" not in [p["at"] for p in egress_gateway.held(started).proofs]
    other = MagicMock()
    await enforce_agent_egress_firewall(other, "agent", agent_env)
    other.exec.assert_not_called()


@pytest.mark.asyncio
async def test_the_model_gateway_url_is_checked_as_before(started):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    policy = egress_denylist_for(started.task_env_config)
    with pytest.raises(ValueError, match="loopback"):
        await start_egress_denylist(
            started, "agent", policy, model_gateway_url="http://10.0.0.1:4000"
        )


@pytest.mark.asyncio
async def test_each_stop_takes_the_lines_since_the_last(gateway, started, tmp_path):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    policy = egress_denylist_for(started.task_env_config)
    hold = egress_gateway.held(started)
    await start_egress_denylist(started, "agent", policy)
    assert _blocked_request(gateway, hold).startswith(b"HTTP/1.1 403")
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        RolloutPaths(rollout_dir=folder).mkdir()
    await stop_egress_denylist(started, first)
    assert _phase(gateway) == "deny"
    await start_egress_denylist(started, "agent", policy)
    _blocked_request(gateway, hold)
    _blocked_request(gateway, hold)
    await stop_egress_denylist(started, second)
    lines = [
        (folder / "trajectory" / "egress_denylist.jsonl").read_text().splitlines()
        for folder in (first, second)
    ]
    assert [len(x) for x in lines] == [1, 2]
    assert all(json.loads(x)["action"] == "blocked" for x in lines[0] + lines[1])
    allowed = (second / "trajectory" / "egress_allowed.jsonl").read_text()
    assert allowed == ""


@pytest.mark.asyncio
async def test_a_log_longer_than_one_read_is_read_to_its_end(
    gateway, started, tmp_path, monkeypatch
):
    """Each ``log`` answer holds a bounded part of each log: the trajectory
    gets every line, and the verifier counts from the log's real end.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.lockdown import _arm_verifier_network
    from benchflow.sandbox.verifier_network import refused_since_arm

    monkeypatch.setattr(server, "MAX_LOG", 200)
    hold = egress_gateway.held(started)
    await start_egress_denylist(
        started, "agent", egress_denylist_for(started.task_env_config)
    )
    for _ in range(5):
        _blocked_request(gateway, hold)
    RolloutPaths(rollout_dir=tmp_path / "turn").mkdir()
    await stop_egress_denylist(started, tmp_path / "turn")
    log = tmp_path / "turn" / "trajectory" / "egress_denylist.jsonl"
    assert len(log.read_text().splitlines()) == 5
    for _ in range(4):
        _blocked_request(gateway, hold)  # after the turn, before the verifier
    await _arm_verifier_network(started, SimpleNamespace(config=_config()), {})
    assert (await refused_since_arm(started))[1] == 0
    for _ in range(3):
        _blocked_request(gateway, hold)
    assert (await refused_since_arm(started))[1] == 3


@pytest.mark.asyncio
async def test_the_oracle_runs_under_deny_and_its_refusals_are_kept(
    gateway, started, tmp_path
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.oracle_egress import arm_oracle_egress, disarm_oracle_egress

    policy = egress_denylist_for(started.task_env_config)
    armed = await arm_oracle_egress(started, policy, run_as=None)
    assert armed.match == egress_gateway.ORACLE_MATCH and armed.gid is None
    assert armed.env["HTTPS_PROXY"] == "http://127.0.0.1:18628"
    assert _phase(gateway) == "deny"
    _blocked_request(gateway, egress_gateway.held(started))
    RolloutPaths(rollout_dir=tmp_path / "oracle").mkdir()
    await disarm_oracle_egress(started, armed, tmp_path / "oracle")
    log = tmp_path / "oracle" / "trajectory" / "egress_denylist.jsonl"
    assert len(log.read_text().splitlines()) == 1
    # Held through the handover to the verifier.
    assert _phase(gateway) == "deny"
    assert started.exec.ran("iptables") == []


@pytest.mark.asyncio
async def test_a_public_verifier_runs_open(gateway, started):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.lockdown import _arm_verifier_network
    from benchflow.sandbox.verifier_network import refused_since_arm

    await start_egress_denylist(
        started, "agent", egress_denylist_for(started.task_env_config)
    )
    task = SimpleNamespace(
        config=_config(DENYLIST + '[verifier]\nnetwork_mode = "public"\n')
    )
    await _arm_verifier_network(started, task, {})
    assert _phase(gateway) == "open"
    armed, refused = await refused_since_arm(started)
    assert armed.mode == "public" and armed.state == "open" and refused == 0


@pytest.mark.asyncio
async def test_a_denylist_verifier_runs_under_deny_its_refusals_counted_from_the_arm(
    gateway, started
):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.lockdown import _arm_verifier_network
    from benchflow.sandbox.verifier_network import refused_since_arm

    hold = egress_gateway.held(started)
    await start_egress_denylist(
        started, "agent", egress_denylist_for(started.task_env_config)
    )
    _blocked_request(gateway, hold)  # the agent's, before the arm
    await egress_gateway.open_phase(started, "a test")
    task = SimpleNamespace(config=_config())
    await _arm_verifier_network(started, task, {})
    assert _phase(gateway) == "deny"
    armed = started._benchflow_verifier_network
    assert armed.proxy == egress_gateway.PROXY and armed.launch is False
    assert armed.state == "held" and armed.match is None
    assert task.config.verifier.env["HTTPS_PROXY"] == "http://127.0.0.1:18628"
    assert hold.proofs[-1] == {
        **CONFINED,
        "user": "root",
        "at": "verifier",
        "phase": "deny",
    }
    assert (await refused_since_arm(started))[1] == 0
    _blocked_request(gateway, hold)
    armed, refused = await refused_since_arm(started)
    assert refused == 1 and armed.proxy_refused == 1
    assert armed.proxy_refusals[0]["rule"] == "host:blocked.test"


@pytest.mark.asyncio
async def test_a_verifier_the_gateway_cannot_prove_is_not_held(gateway, started):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.lockdown import _arm_verifier_network
    from benchflow.task.verifier_errors import VerifierNetworkUnheldError

    started.exec.probe = {**CONFINED, "direct": "connected"}
    with pytest.raises(VerifierNetworkUnheldError, match="egress gateway"):
        await _arm_verifier_network(started, SimpleNamespace(config=_config()), {})


@pytest.mark.asyncio
async def test_no_uid_firewall_is_named_for_the_firewall_power_check(started):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    from benchflow.sandbox.egress_denylist import denylist_agent_env
    from benchflow.sandbox.firewall_power import firewall_rules_hold_agent

    agent_env = denylist_agent_env({}, egress_denylist_for(started.task_env_config))
    assert (
        firewall_rules_hold_agent(started, agent_env, cloud_metadata_armed=False) == ()
    )
    plain = SimpleNamespace(task_env_config=None)
    assert firewall_rules_hold_agent(plain, agent_env, cloud_metadata_armed=False) == (
        "the denylist's uid firewall",
    )


def test_a_mock_never_looks_held():
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    assert egress_gateway.held(MagicMock()) is None
    assert egress_gateway.held(SimpleNamespace()) is None


@pytest.mark.parametrize(
    ("session", "expected"),
    [("task__abc", "task__abc"), ("..", None), ("a b", None), ("x" * 200, None)],
)
def test_rollout_ids_are_safe_folder_names(session, expected):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    rid = egress_gateway.rollout_id(session)
    assert server.ROLLOUT_ID.fullmatch(rid)
    if expected is not None:
        assert rid == expected


def test_modals_sdk_takes_the_allowlist():
    """The create's keyword, on the pinned SDK (a renamed one would fail every
    held create; the start proof would also find the direct connect open).

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    modal = pytest.importorskip("modal")
    assert (
        "outbound_cidr_allowlist" in inspect.signature(modal.Sandbox.create).parameters
    )


# With #1248 (Modal's Docker parity) on main: its settings and the gateway.

#: ``cat /proc/self/limits``'s rows for Docker's limits.
_DOCKER_LIMITS = (
    "Max open files            1024                 524288               files\n"
    "Max stack size            8388608              unlimited            bytes\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
async def test_the_create_takes_the_image_env_and_the_allowlist_together(
    gateway, make_sandbox, provider, factory
):
    """Both launchers create a held sandbox with #1248's image env beside the
    gateway's allowlist, and the rollout's time to live follows the sandbox's
    lifetime (``BENCHFLOW_MODAL_SANDBOX_TIMEOUT``, given to the launcher)."""
    sandbox = make_sandbox(
        factory=factory,
        dockerfile="FROM python:3.12-slim\nENV PYTHONHASHSEED=0 OMP_NUM_THREADS=2\n",
        sandbox_timeout_secs=600,
    )
    await sandbox.start(force_build=False)
    kwargs = provider.Sandbox.create.aio.await_args.kwargs
    assert kwargs["outbound_cidr_allowlist"] == ["127.0.0.1/32"]
    assert kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
    hold = egress_gateway.held(sandbox)
    assert hold is not None
    assert hold.events[0]["event"] == "registered"
    assert hold.events[0]["ttl_sec"] == 600 + egress_gateway.TTL_MARGIN_SEC
    await sandbox.stop(True)
    assert _status(gateway)["rollouts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
async def test_the_relay_and_its_proofs_run_behind_the_held_rlimits(
    gateway, make_sandbox, monkeypatch, factory
):
    """With ``BENCHFLOW_MODAL_RLIMITS``, both launchers hold the limits before
    the gateway's attach: the relay starts, and every proof runs, behind the
    held prefix (prlimit, timeout, setpriv), as each exec does.

    Checks order and wrapping only. That the relay outlives its launch under
    gVisor is shown by the live Modal run at dee2c46d (relay 200 at start and
    at the verifier, 66 held execs).

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``) with PR #1248's
    ``BENCHFLOW_MODAL_RLIMITS``."""
    monkeypatch.setenv("BENCHFLOW_MODAL_RLIMITS", "1")
    sandbox = make_sandbox(factory=factory)
    script = sandbox.exec
    script.su_check = f"== su root rc=0\n{_DOCKER_LIMITS}"
    await sandbox.start(force_build=False)
    flat = [c for c, _ in script.commands]
    conf = flat.index(script.ran("benchflow-rlimits.conf")[0])
    attach = flat.index(script.ran("chmod 700 /opt/runtime-egress")[0])
    relay = flat.index(script.ran("relay.json </dev/null")[0])
    assert conf < attach < relay
    assert all(
        limits == modal_impl.DOCKER_RLIMITS for limits in script.held[attach:]
    ), script.held[attach:]
    assert sandbox.hold_argv(["/bin/bash", "-c", flat[relay]])[0] == (
        modal_impl.PRLIMIT
    )
    assert _phase(gateway) == "open"
    await sandbox.stop(True)
    assert _status(gateway)["rollouts"] == []


@pytest.mark.asyncio
async def test_a_sandbox_refused_for_its_rlimits_never_reaches_the_gateway(
    gateway, make_sandbox, provider, monkeypatch
):
    """The limits are held, or refused, before the attach: a sandbox that
    cannot hold them is ended with no rollout at the gateway.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``) with PR #1248's
    ``BENCHFLOW_MODAL_RLIMITS``."""
    monkeypatch.setenv("BENCHFLOW_MODAL_RLIMITS", "1")
    sandbox = make_sandbox()
    sandbox.exec.su_check = "== su root rc=1\n"
    with pytest.raises(modal_impl.ModalRlimitsRefused):
        await sandbox.start(force_build=False)
    provider.Sandbox.create.aio.return_value.terminate.aio.assert_awaited()
    assert sandbox.exec.ran(egress_gateway.RELAY_DIR) == []
    assert egress_gateway.held(sandbox) is None
    assert _status(gateway)["rollouts"] == []


@pytest.mark.asyncio
async def test_a_sandbox_modal_ends_during_the_proofs_is_deregistered(
    gateway, make_sandbox, provider
):
    """Modal ends the sandbox during the start proof (its lifetime, a memory
    kill): the probe's exec fails as #1248 fails it, an infrastructure
    error, and the stop that follows deregisters the rollout without a
    terminate."""
    remote = provider.Sandbox.create.aio.return_value
    remote.poll.aio.return_value = 137
    remote.exec = SimpleNamespace(
        aio=AsyncMock(
            side_effect=RuntimeError(
                "Modal Sandbox with container ID ta-1 not found. This means "
                "this Sandbox has already shut down."
            )
        )
    )
    sandbox = make_sandbox()

    async def real_exec(command, **kwargs):
        return await ModalSandbox.exec(sandbox, command, **kwargs)

    sandbox.exec.real = ("BF egress-gateway-probe", real_exec)
    with pytest.raises(modal_impl.ModalSandboxEndedError, match="returncode 137"):
        await sandbox.start(force_build=False)
    assert sandbox.platform_end is not None
    assert sandbox.platform_end["seen_at"] == "command"
    remote.exec.aio.assert_awaited()
    remote.terminate.aio.assert_not_awaited()
    assert _status(gateway)["rollouts"] == []
    record = json.loads(
        (sandbox.rollout_paths.rollout_dir / egress_gateway.RECORD_NAME).read_text()
    )
    assert record["events"][-1]["event"] == "deregistered"


@pytest.mark.asyncio
async def test_an_end_modal_made_at_the_agents_launch_proof_is_infrastructure(
    gateway, started, provider
):
    """Modal ends the sandbox while the agent's launch proof runs: the proof
    fails, and its error, which keeps the exec's, is classed as #1248 classes
    an end Modal made (an infrastructure error), not as a sandbox let out."""
    from benchflow._utils.scoring import INFRA_ERROR, classify_error
    from benchflow.sandbox.egress_denylist import denylist_agent_env
    from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

    policy = egress_denylist_for(started.task_env_config)
    await start_egress_denylist(
        started, "agent", policy, model_gateway_url="http://127.0.0.1:4000"
    )
    remote = provider.Sandbox.create.aio.return_value
    remote.poll.aio.return_value = 137
    remote.exec = SimpleNamespace(
        aio=AsyncMock(side_effect=RuntimeError("Modal Sandbox is shutting down."))
    )

    async def real_exec(command, **kwargs):
        return await ModalSandbox.exec(started, command, **kwargs)

    started.exec.real = ("BF egress-gateway-probe", real_exec)
    agent_env = denylist_agent_env(
        {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000"}, policy
    )
    with pytest.raises(RuntimeError, match="not held in the deny phase") as raised:
        await enforce_agent_egress_firewall(started, "agent", agent_env)
    assert started.platform_end is not None
    assert classify_error(str(raised.value)) == INFRA_ERROR

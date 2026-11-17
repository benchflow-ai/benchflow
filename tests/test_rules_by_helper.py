"""Main never holds NET_ADMIN or NET_RAW while BenchFlow's helper serves.

RCI on 86fd158d6: under allowlist, denylist and the offline shape main kept
NET_ADMIN for the uid firewall, so a root oracle (and every image binary
BenchFlow's own root steps ran after it) deleted and flushed the rules and
reached the metadata server and the internet while the record said "held".
Now the helper adds every rule from outside main, and main starts without
either capability under every policy.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _compose
from benchflow.sandbox import rule_ledger as rl
from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command
from benchflow.sandbox.lockdown import _agent_egress_firewall_cmd
from benchflow.task.config import NetworkMode, SandboxConfig
from tests.test_helper_report_lines import MAIN, STUB, ShellDocker, _helper


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    folder = tmp_path / "bin"
    folder.mkdir()
    for name in ("iptables", "ip6tables", "ip"):
        (folder / name).write_text(STUB)
        (folder / name).chmod(0o755)
    return folder


@pytest.mark.parametrize(
    "mode,allow",
    [
        (NetworkMode.ALLOWLIST, True),
        (NetworkMode.DENYLIST, True),
        (NetworkMode.NO_NETWORK, True),  # the offline shape kept open
        (NetworkMode.PUBLIC, True),
    ],
)
def test_main_gets_neither_capability_when_the_helper_adds_the_rules(
    tmp_path, mode, allow
):
    kw = {"network_mode": mode, "allow_internet": allow}
    if mode == NetworkMode.ALLOWLIST:
        kw["allowed_hosts"] = ["example.com"]
    if mode == NetworkMode.DENYLIST:
        kw["blocked_hosts"] = ["example.org"]
    config = SandboxConfig(**kw)
    with_helper = _compose.compose_file_paths(tmp_path, config, rules_by_helper=True)
    assert _compose.COMPOSE_NET_ADMIN_PATH not in with_helper
    assert _compose.COMPOSE_NO_NET_RAW_PATH in with_helper
    without = _compose.compose_file_paths(tmp_path, config)
    if mode in (NetworkMode.ALLOWLIST, NetworkMode.DENYLIST):
        assert _compose.COMPOSE_NET_ADMIN_PATH in without  # the old in-main installs


def test_the_uid_firewall_for_the_helper_takes_the_uid_and_installs_nothing():
    script = _agent_egress_firewall_cmd(
        "agent", uid=1003, allow_networks=("10.0.0.0/8",)
    )
    assert "agent_uid=1003;" in script
    assert "id -u" not in script and "apt-get" not in script


def test_captured_rules_are_the_specs_iptables_c_takes():
    out = "\n".join(
        [
            "BF rule-added iptables filter -A OUTPUT -o lo -m owner --uid-owner 1003 -j ACCEPT",
            "BF rule-added iptables filter -A OUTPUT -m owner --uid-owner 1003 -j REJECT --reject-with icmp-port-unreachable",
            "BF rule-added ip6tables filter -A OUTPUT -m owner --uid-owner 1003 -j REJECT",
            "BF rule-added iptables nat -A OUTPUT -p udp -m udp --dport 53 -m owner --uid-owner 1003 -j REDIRECT --to-ports 18629",
            "BF rule-added iptables filter -P OUTPUT ACCEPT",
            "BF rules-installed status=0",
        ]
    )
    v4, v6, nat = rl.added_rules(out)
    assert v4 == [
        "-o lo -m owner --uid-owner 1003 -j ACCEPT",
        "-m owner --uid-owner 1003 -j REJECT --reject-with icmp-port-unreachable",
    ]
    assert v6 == ["-m owner --uid-owner 1003 -j REJECT"]
    assert nat == [
        "-p udp -m udp --dport 53 -m owner --uid-owner 1003 -j REDIRECT --to-ports 18629"
    ]


@pytest.mark.asyncio
async def test_the_uid_firewall_goes_in_from_the_helper_and_into_the_ledger(stub_bin):
    """main holds no NET_ADMIN: the script runs in the helper (real script,
    stub iptables), the agent's uid read from main first, its rules recorded."""
    from benchflow.sandbox import lockdown

    helper = _helper(ShellDocker(stub_bin))

    class Env:
        rules_by_helper = True

        async def metadata_rules_install(self, command):
            return await helper.installed(MAIN, command)

        async def exec(self, command, *args, **kwargs):
            # The uid read is a checked receipt (lockdown's reads, c40).
            framed = unwrap_receipt_command(command)
            assert framed is not None and "id -u agent" in framed[0], command
            out = encode_receipt(b"1003\n", 0, framed[1])
            return SimpleNamespace(stdout=out, stderr="", return_code=0)

    env = Env()
    await lockdown.enforce_sandbox_uid_egress(env, "agent")
    (entry,) = rl.entries(env)
    assert (entry.where, entry.phase) == ("main", "agent-uid")


def test_compose_files_exist():
    assert Path(_compose.COMPOSE_NO_NET_RAW_PATH).is_file()


def _config(mode: str) -> SandboxConfig:
    if mode == "allow":
        return SandboxConfig(
            network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"]
        )
    if mode == "deny":
        return SandboxConfig(
            network_mode=NetworkMode.DENYLIST, blocked_hosts=["example.org"]
        )
    if mode == "offline":  # the offline shape: no-network kept open for the verifier
        config = SandboxConfig(network_mode=NetworkMode.NO_NETWORK)
        config.allow_internet = True
        return config
    if mode == "nonet":
        return SandboxConfig(network_mode=NetworkMode.NO_NETWORK, allow_internet=False)
    return SandboxConfig()


#: The restricted policy the sandbox factory sets per mode (setup.py,
#: service_cover.restricted_policy): "" when every phase is open.
_POLICY = {
    "allow": "allowlist",
    "deny": "denylist",
    "offline": "no-network",
    "nonet": "no-network",
    "verifier-nonet": "no-network, the verifier's",
    "open": "",
}


def _stand_in(mode: str, **kw):
    import types

    from benchflow.sandbox.docker import DockerSandbox

    fields = {
        "_wants_rules_by_helper": True,
        "task_env_config": _config("open" if mode == "verifier-nonet" else mode),
        "_network_policy": _POLICY[mode],
        "_recovery_baseline": None,
        "_main_metadata_rule": True,
        "_helper_docker": None,
        "_helper_daemon": lambda: "local",
        **kw,
    }
    env = SimpleNamespace(**fields)
    env._refuse_main_net_admin = types.MethodType(
        DockerSandbox._refuse_main_net_admin, env
    )
    return env, DockerSandbox._decide_rules_by_helper


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["allow", "deny", "offline", "nonet", "verifier-nonet"]
)
@pytest.mark.parametrize(
    "case,words",
    [
        ("helper-off", "BENCHFLOW_METADATA_HELPER=0"),
        ("main-net-admin", "BENCHFLOW_METADATA_MAIN_NET_ADMIN=1"),
        ("recovery", "a recovery start"),
        ("build-failed", "helper image could not be built"),
    ],
)
async def test_every_old_start_under_a_restrictive_policy_is_refused(
    monkeypatch, mode, case, words
):
    """F2b (lane B scoping): each override that keeps the old start would
    leave main holding NET_ADMIN under these policies; it refuses, naming it.
    Under plain no-network and a verifier's own restricted network it would
    leave NET_RAW (RCI, a80425fb): refused the same way."""
    from benchflow.sandbox import _metadata_helper as helper_mod

    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)
    monkeypatch.delenv(helper_mod.MAIN_NET_ADMIN_ENV, raising=False)

    async def built(*args, **kwargs):
        if case == "build-failed":
            raise helper_mod.HelperUnavailable("forced build failure")
        return "sha256:" + "ab" * 32

    monkeypatch.setattr(helper_mod, "ensure_image", built)
    if case == "helper-off":
        monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    if case == "main-net-admin":
        monkeypatch.setenv(helper_mod.MAIN_NET_ADMIN_ENV, "1")
    env, decide = _stand_in(
        mode, **({"_recovery_baseline": object()} if case == "recovery" else {})
    )
    if case == "build-failed":
        env._main_metadata_rule = False  # an opt-out task: still refused here
    with pytest.raises(RuntimeError, match=words):
        await decide(env)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["open"])
async def test_the_old_start_stays_where_main_holds_no_net_admin_anyway(
    monkeypatch, mode
):
    from benchflow.sandbox import _metadata_helper as helper_mod

    monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    env, decide = _stand_in(mode)
    await decide(env)
    assert env._rules_by_helper is False


@pytest.mark.asyncio
async def test_a_sandbox_built_directly_keeps_the_old_start(monkeypatch):
    from benchflow.sandbox import _metadata_helper as helper_mod

    monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    env, decide = _stand_in(
        "allow",
    )
    env._wants_rules_by_helper = False
    await decide(env)
    assert env._rules_by_helper is False


@pytest.mark.parametrize("mode", ["allow", "deny", "offline", "nonet", "open"])
@pytest.mark.parametrize("override", ["", "helper-off", "main-net-admin"])
def test_the_gate_resolves_the_files_the_start_reads(
    tmp_path, monkeypatch, mode, override
):
    """NETSCAN's trap: the launch gate resolved the old start's NET_ADMIN
    overlay whatever the start would read, so a check of the gate's model
    would have refused every denylist task. The gate's -f list is the start's
    now, from one decision (_metadata_helper.old_start_reason)."""
    from benchflow.sandbox import _metadata_helper as helper_mod
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        COMPOSE_NO_NET_RAW_PATH,
        compose_needs_net_admin,
    )
    from benchflow.task import compose_env

    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)
    monkeypatch.delenv(helper_mod.MAIN_NET_ADMIN_ENV, raising=False)
    if override == "helper-off":
        monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    if override == "main-net-admin":
        monkeypatch.setenv(helper_mod.MAIN_NET_ADMIN_ENV, "1")
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    (environment / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    seen: dict = {}

    def freeze(**kwargs):
        seen["files"] = [p.resolve() for p in kwargs["files"]]
        return compose_env.ComposeCheck()

    monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    monkeypatch.setattr(compose_env, "gate_connection", lambda *a: ([], {}))
    monkeypatch.setattr(compose_env, "daemon_filesystem_local", lambda *a: True)
    config = _config(mode)
    compose_env._check_compose(
        environment, sandbox_type="docker", sandbox_config=config
    )
    by_helper = override == ""
    assert (COMPOSE_NET_ADMIN_PATH.resolve() in seen["files"]) is (
        not by_helper and compose_needs_net_admin(config)
    )
    assert (COMPOSE_NO_NET_RAW_PATH.resolve() in seen["files"]) is by_helper

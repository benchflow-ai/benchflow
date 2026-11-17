"""Every compose service of a trial is covered from BenchFlow's helper (EN-609).

A task's other services (a database, an app the agent drives) have their own
network namespaces, and Docker NATs them out through the host: before, agent
code that reached one reached the VM's metadata server and any host. Now each
service's namespace gets the metadata rule (every user) and the agent's policy,
proven by a refused, counted connect from uid 0 there, and service-network.json
lists every service with its proof.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import service_cover as sc


def _target(**kw):
    base = {
        "service": "db",
        "id": "c" * 64,
        "network_mode": "bf_default",
        "cap_add": [],
        "privileged": False,
        "subnets": ["172.20.0.0/16"],
        "gateways": ["172.20.0.1"],
    }
    return {**base, **kw}


class Env:
    """A sandbox whose helper answers each service's install, connect and count."""

    def __init__(self, targets, *, connect="refused", counts=((0, 0), (1, 1))):
        self.targets = targets
        self.connect = connect
        self.counts = list(counts)
        self.installed: list[tuple[str, str]] = []
        self.closed = 0

    async def metadata_service_targets(self):
        return self.targets

    async def metadata_service_install(self, container, command):
        self.installed.append((container, command))
        m, e = self.counts[0]
        return SimpleNamespace(
            stdout=f"BF service-rules status=0\nBF service-counters metadata={m} egress={e}\n"
        ), ""

    async def metadata_service_run(self, container, command, *, timeout_sec, user=None):
        if "BF service-connect" in command:
            assert user == "0:0"
            return SimpleNamespace(stdout=f"BF service-connect {self.connect}\n"), ""
        m, e = self.counts[-1]
        return SimpleNamespace(
            stdout=f"BF service-counters metadata={m} egress={e}\n"
        ), ""

    def metadata_service_close(self):
        self.closed += 1


def _config(**metadata):
    return SimpleNamespace(metadata=metadata)


def test_a_service_s_rules_refuse_metadata_to_root_and_close_its_egress():
    v4, v6 = sc.service_rules(
        metadata=True,
        egress="closed",
        subnets=("172.20.0.0/16",),
        gateways=("172.20.0.1",),
    )
    assert (
        v4[0]
        == "-d 169.254.169.254 -p tcp ! --dport 53 -j REJECT --reject-with tcp-reset"
    )
    assert not any("--uid-owner" in r for r in v4 + v6)  # root too
    assert v4.index("-d 172.20.0.1 -j REJECT") < v4.index("-d 172.20.0.0/16 -j ACCEPT")
    assert "-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT" in v4
    assert v4[-1] == "-j REJECT" and v6[-1] == "-j REJECT"
    allow, _ = sc.service_rules(
        metadata=False,
        egress="allowlist",
        subnets=(),
        gateways=(),
        allow_networks=("10.9.0.0/24",),
    )
    assert "-d 10.9.0.0/24 -j ACCEPT" in allow
    open_, _ = sc.service_rules(metadata=True, egress="open", subnets=(), gateways=())
    assert all("169.254" in r or "100.100" in r or "168.63" in r for r in open_)


@pytest.mark.parametrize(
    ("mode", "service", "opened", "kind"),
    [
        ("no-network", "db", (), "closed"),
        ("denylist", "db", (), "closed"),  # the lead's ruling: fail closed
        ("denylist", "web", ("web",), "open"),  # unless the task names it
        ("allowlist", "db", (), "allowlist"),
        ("public", "db", (), "open"),
    ],
)
def test_which_egress_a_service_gets(mode, service, opened, kind):
    assert sc.egress_kind(mode, service, opened) == kind


def test_the_denylist_switch_is_one_line(monkeypatch):
    monkeypatch.setattr(sc, "DENYLIST_SIDECARS_FAIL_CLOSED", False)
    assert sc.egress_kind("denylist", "db", ()) == "open"


@pytest.mark.asyncio
async def test_every_service_is_covered_and_proven():
    env = Env([_target(), _target(service="cache", id="d" * 64)])
    records = await sc.cover_task_services(
        env, _config(), mode="no-network", metadata=True, opt_out=False
    )
    assert [r["service"] for r in records] == ["db", "cache"]
    for record in records:
        assert record["state"] == "covered" and record["proven"] is True
        assert record["proof"] == {
            "metadata_connect": "refused",
            "egress_connect": "refused",
            "metadata_counted": 1,
            "egress_counted": 1,
        }
    assert len(env.installed) == 2 and env.closed == 1


@pytest.mark.asyncio
async def test_a_service_the_task_opens_is_recorded_as_open():
    env = Env([_target(service="web")], counts=((0, 0), (1, 0)))
    (record,) = await sc.cover_task_services(
        env,
        _config(**{"x-service-egress": ["web"]}),
        mode="denylist",
        metadata=True,
        opt_out=False,
    )
    assert record["egress"] == "open"
    assert record["egress_opened_by"] == "metadata.x-service-egress"
    assert record["proof"]["metadata_connect"] == "refused"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "words"),
    [
        (_target(cap_add=["NET_ADMIN"]), "holds NET_ADMIN"),
        (_target(cap_add=["net_raw"]), "holds NET_RAW"),
        (_target(privileged=True), "holds privileged"),
    ],
)
async def test_a_service_that_could_remove_its_rules_refuses_the_trial(target, words):
    env = Env([target])
    with pytest.raises(sc.ServiceCoverError, match=words):
        await sc.cover_task_services(
            env, _config(), mode="public", metadata=True, opt_out=False
        )
    assert env.installed == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("compose", "words"),
    [
        ("services:\n  main:\n    cap_add: [NET_ADMIN]\n", "gives main NET_ADMIN"),
        (
            "services:\n  main:\n    cap_add: [net_raw, SYS_TIME]\n",
            "gives main NET_RAW",
        ),
        ("services:\n  main:\n    cap_add: [ALL]\n", "gives main ALL"),
    ],
)
async def test_a_main_the_task_gives_net_admin_refuses_the_trial(
    tmp_path, compose, words
):
    """TasksMiner's trapped-ions task (EN-626): main's own cap_add NET_ADMIN."""
    (tmp_path / "docker-compose.yaml").write_text(compose)
    env = Env([])
    with pytest.raises(sc.ServiceCoverError, match=words):
        await sc.cover_task_services(
            env,
            _config(),
            mode="public",
            metadata=True,
            opt_out=False,
            environment_dir=tmp_path,
        )
    (tmp_path / "docker-compose.yaml").write_text(
        "services:\n  main:\n    cap_add: [SYS_TIME]\n"
    )
    assert (
        await sc.cover_task_services(
            env,
            _config(),
            mode="public",
            metadata=True,
            opt_out=False,
            environment_dir=tmp_path,
        )
        == []
    )


@pytest.mark.asyncio
async def test_the_task_opt_out_admits_a_capable_service_and_records_it():
    env = Env([_target(cap_add=["NET_ADMIN"])], counts=((0, 0), (0, 0)))
    (record,) = await sc.cover_task_services(
        env, _config(), mode="public", metadata=False, opt_out=True
    )
    assert record["capabilities"] == ["NET_ADMIN"] and record["state"] == "no rule"


@pytest.mark.asyncio
async def test_services_with_no_route_or_a_borrowed_namespace_need_no_rule():
    env = Env(
        [
            _target(network_mode="none"),
            _target(service="side", network_mode="container:abc"),
        ]
    )
    records = await sc.cover_task_services(
        env, _config(), mode="no-network", metadata=True, opt_out=False
    )
    assert [r["state"] for r in records] == ["no route", "shares a namespace"]
    assert env.installed == []


@pytest.mark.asyncio
async def test_an_unproven_service_refuses_the_trial():
    env = Env([_target()], connect="reached", counts=((0, 0), (0, 0)))
    with pytest.raises(sc.ServiceCoverError, match="not proven"):
        await sc.cover_task_services(
            env, _config(), mode="no-network", metadata=True, opt_out=False
        )


@pytest.mark.asyncio
async def test_a_sandbox_without_the_helper_is_not_touched():
    assert (
        await sc.cover_task_services(
            MagicMock(spec=[]), _config(), mode="public", metadata=True, opt_out=False
        )
        is None
    )


def test_the_helper_scripts_are_valid_sh():
    import subprocess

    v4, v6 = sc.service_rules(
        metadata=True, egress="closed", subnets=("172.20.0.0/16",), gateways=()
    )
    for script in (
        sc.install_cmd(v4, v6),
        sc.counters_cmd(),
        sc.connect_cmd("192.0.2.1"),
    ):
        assert subprocess.run(["sh", "-n", "-c", script]).returncode == 0


class MainEnv(Env):
    """The sandbox lists main too when asked (its metadata rule, EN-609)."""

    async def metadata_service_targets(self, *, include_main=False):
        main = _target(service="main", id="a" * 64, cap_add=["NET_ADMIN"])
        return ([main] if include_main else []) + self.targets


@pytest.mark.asyncio
async def test_main_gets_the_metadata_rule_for_every_user_from_the_start():
    """The lead's ruling (22:40Z): root in main is refused the metadata
    server during the agent's turn too; main's NET_ADMIN (BenchFlow's own,
    for the agent firewall) never refuses the trial."""
    env = MainEnv([_target()])
    records = await sc.cover_task_services(
        env, _config(), mode="public", metadata=True, opt_out=False, main_metadata=True
    )
    assert [r["service"] for r in records] == ["main", "db"]
    main = records[0]
    assert main["state"] == "covered" and main["every_user"] is True
    assert main["proof"] == {"metadata_connect": "refused", "metadata_counted": 1}
    container, command = env.installed[0]
    assert container == "a" * 64
    assert "169.254.169.254" in command and "--uid-owner" not in command
    assert "BF service-socket" not in command or "/proc/net/" in command


@pytest.mark.asyncio
async def test_only_the_task_opts_main_out():
    env = MainEnv([])
    (main,) = await sc.cover_task_services(
        env, _config(), mode="public", metadata=False, opt_out=True, main_metadata=False
    )
    assert main == {
        "service": "main",
        "network_mode": "bf_default",
        "state": "no rule",
        "why": "the task opts out",
        "proven": True,
    }
    assert env.installed == []


def test_sockets_seen_before_coverage_are_decoded_with_their_destination():
    out = "\n".join(
        [
            # 172.20.0.3:41234 -> 169.254.169.254:80, established
            "BF service-socket tcp FEA9FEA9:0050 01",
            # -> 1.1.1.1:443 in time-wait
            "BF service-socket tcp 01010101:01BB 06",
            # another service of the project, and loopback, and a listener
            "BF service-socket tcp 030014AC:1538 01",
            "BF service-socket tcp 0100007F:0035 01",
            "BF service-socket tcp 00000000:0000 0A",
            # IPv6 2001:db8::1 port 443
            "BF service-socket tcp6 B80D0120000000000000000001000000:01BB 02",
            "BF service-conntrack ipv4     2 udp      17 25 src=172.20.0.3 dst=8.8.8.8 sport=5353 dport=53",
        ]
    )
    seen = sc.seen_before_coverage(out, ("172.20.0.0/16",))
    assert seen == [
        {
            "proto": "tcp",
            "remote": "169.254.169.254:80",
            "state": "established",
            "outside_project": True,
        },
        {
            "proto": "tcp",
            "remote": "1.1.1.1:443",
            "state": "time-wait",
            "outside_project": True,
        },
        {"proto": "tcp", "remote": "172.20.0.3:5432", "state": "established"},
        {
            "proto": "tcp6",
            "remote": "2001:db8::1:443",
            "state": "syn-sent",
            "outside_project": True,
        },
        {
            "proto": "conntrack-udp",
            "remote": "8.8.8.8:53",
            "state": "tracked",
            "outside_project": True,
        },
    ]


@pytest.mark.asyncio
async def test_a_service_record_shows_what_its_sockets_held_before_coverage():
    class Seen(Env):
        async def metadata_service_install(self, container, command):
            result, failed = await super().metadata_service_install(container, command)
            result.stdout = "BF service-socket tcp 01010101:01BB 01\n" + result.stdout
            return result, failed

    env = Seen([_target(started_at="2026-11-14T22:00:00.1Z")])
    (record,) = await sc.cover_task_services(
        env, _config(), mode="no-network", metadata=True, opt_out=False
    )
    assert record["seen_before_coverage"] == [
        {
            "proto": "tcp",
            "remote": "1.1.1.1:443",
            "state": "established",
            "outside_project": True,
        }
    ]
    assert record["started_at"] == "2026-11-14T22:00:00.1Z" and record["covered_at"]


@pytest.mark.asyncio
async def test_main_is_covered_before_release_while_it_waits_in_the_bootstrap():
    """RCI on 1e05060ab: a root service of the task's environment was live in
    main 2.6 s before main's rule. Now the rule and its proof go in while main
    still waits in the inert bootstrap, and the rollout keeps that record."""
    from benchflow.sandbox.docker import DockerSandbox

    env = MainEnv([])
    env._main_metadata_rule = True
    env._main_container = "a" * 64
    env._bootstrap_services = {"main"}
    await DockerSandbox._cover_main_before_release(env)
    record = env._main_cover_record
    assert record["before_release"] is True and record["every_user"] is True
    assert record["proof"] == {"metadata_connect": "refused", "metadata_counted": 1}
    assert env.closed == 1
    installs = len(env.installed)
    (main,) = await sc.cover_task_services(
        env, _config(), mode="public", metadata=True, opt_out=False, main_metadata=True
    )
    assert main is record and len(env.installed) == installs


@pytest.mark.asyncio
async def test_main_not_waiting_in_the_bootstrap_refuses_the_start():
    from benchflow.sandbox.docker import DockerSandbox

    env = Env([])  # lists no main
    env._main_metadata_rule = True
    env._main_container = "a" * 64

    async def targets(*, include_main=False):
        return []

    env.metadata_service_targets = targets
    with pytest.raises(sc.ServiceCoverError, match="inert bootstrap"):
        await DockerSandbox._cover_main_before_release(env)


@pytest.mark.asyncio
async def test_the_task_opt_out_leaves_main_to_its_own_start():
    from benchflow.sandbox.docker import DockerSandbox

    env = MainEnv([])
    env._main_metadata_rule = False
    await DockerSandbox._cover_main_before_release(env)
    assert env._main_cover_record is None and env.installed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rule", [True, False])
async def test_the_bootstrap_holds_main_for_its_rule_whatever_its_binds(
    monkeypatch, rule
):
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    async def protected(_env, _project, _prototypes):
        return set()  # no host bind anywhere (a remote daemon, say)

    monkeypatch.setattr(bootstrap, "protected_services", protected)
    config = {
        "Labels": {"com.docker.compose.service": "main"},
        "Env": [],
        "Entrypoint": ["/entry.sh"],
        "Cmd": ["serve"],
    }

    async def inspect(cid):
        return {"State": {"Running": False}, "Config": config}

    env = SimpleNamespace(_main_metadata_rule=rule, _inspect_container=inspect)
    document = await bootstrap.overlay(env, {"services": {"main": {}}}, ["c" * 64])
    assert env._bootstrap_services == ({"main"} if rule else set())
    assert ("main" in document["services"]) is rule


@pytest.mark.asyncio
async def test_every_trial_end_records_main_metadata_refusals(tmp_path, monkeypatch):
    """F2b (gemini-cli): a trial that ended before its verifier still shows
    whether anything tried the metadata server: the counter read at cleanup."""
    import json as _json

    from benchflow.rollout import Rollout
    from benchflow.sandbox import verifier_network as vn

    async def counted(env):
        return 2

    monkeypatch.setattr(vn, "main_metadata_refused", counted)
    (tmp_path / sc.SERVICE_NETWORK_RECORD).write_text(
        _json.dumps({"phase": "agent", "services": []})
    )
    stand_in = SimpleNamespace(_env=object(), _rollout_dir=tmp_path)
    await Rollout._record_main_metadata_at_end(stand_in)
    data = _json.loads((tmp_path / sc.SERVICE_NETWORK_RECORD).read_text())
    assert data == {"phase": "agent", "services": [], "main_metadata_refused_at_end": 2}

    async def broken(env):
        raise RuntimeError("helper gone")

    monkeypatch.setattr(vn, "main_metadata_refused", broken)
    await Rollout._record_main_metadata_at_end(stand_in)  # never raises


@pytest.mark.asyncio
async def test_a_service_on_an_internal_network_is_held_by_having_no_route():
    """RCI on 0650e09: an allowlist task's service on an internal network
    (EN-240 asks for one) was refused, "not proven (egress_connect:
    unreachable)". The kernel's unreachable is the proof there: nothing
    leaves, so no rule sees it."""
    env = Env([_target(gateways=[])], connect="unreachable", counts=((0, 0), (0, 0)))
    (record,) = await sc.cover_task_services(
        env, _config(), mode="allowlist", metadata=True, opt_out=False
    )
    assert record["state"] == "covered" and record["proven"] is True
    assert (
        record["proof"]["metadata_connect"]
        == record["proof"]["egress_connect"]
        == "unreachable"
    )


@pytest.mark.asyncio
async def test_a_connect_that_reached_is_never_held():
    env = Env([_target()], connect="reached", counts=((0, 0), (0, 0)))
    with pytest.raises(sc.ServiceCoverError, match="not proven"):
        await sc.cover_task_services(
            env, _config(), mode="allowlist", metadata=True, opt_out=False
        )

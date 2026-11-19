"""A network policy holds every service of the trial: no compose service may
hold a way around its rules (NET_ADMIN, NET_RAW, ALL, ``privileged``, the
host's network), whatever the cloud metadata opt-out or the operator's
host-access flags say (the lead's 03:28Z and 03:59Z rulings). Only an
open network, for the agent and the verifier, keeps them; and no rule of the
trial's ever goes into the host's own network namespace."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox.service_cover import (
    _cover_one,
    compose_network_powers,
    policy_power_refusal,
)
from benchflow.task.config import TaskConfig

POLICIES = {
    "allowlist": '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    "denylist": '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.com"]\n',
    "no-network": "[environment]\nallow_internet = false\n",
    "offline": (
        "[environment]\nallow_internet = false\n"
        '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n'
    ),
    "verifier allowlist": '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
}
POWERS = {
    "main NET_ADMIN": ("main", "    cap_add: [NET_ADMIN]\n", "NET_ADMIN"),
    "sidecar NET_RAW": ("db", "    cap_add: [NET_RAW]\n", "NET_RAW"),
    "sidecar ALL": ("db", "    cap_add: [ALL]\n", "ALL"),
    "main privileged": ("main", "    privileged: true\n", "privileged"),
    "main host network": ("main", "    network_mode: host\n", "the host's network"),
}


def _config(toml: str, opt_out: bool) -> TaskConfig:
    meta = "[metadata]\nx-allow-cloud-metadata = true\n" if opt_out else ""
    return TaskConfig.model_validate_toml('version = "1.0"\n' + meta + toml)


def _compose(tmp_path, service: str, lines: str):
    main = lines if service == "main" else ""
    db = lines if service == "db" else ""
    (tmp_path / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: x\n" + main + "  db:\n    image: y\n" + db
    )
    return tmp_path


@pytest.mark.parametrize("opt_out", [False, True], ids=["no-opt-out", "opt-out"])
@pytest.mark.parametrize("power", list(POWERS))
@pytest.mark.parametrize("policy", list(POLICIES))
def test_a_policy_refuses_every_way_around_its_rules_before_start(
    tmp_path, policy, power, opt_out, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    monkeypatch.setenv("BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD", "1")
    service, lines, named = POWERS[power]
    why = policy_power_refusal(
        _config(POLICIES[policy], opt_out), _compose(tmp_path, service, lines)
    )
    assert why is not None and why.startswith(
        "sandbox startup refused: the task's network policy ("
    ), why
    assert ") holds every service, and " in why, why
    assert f"{service} {named}" in why
    # The opt-out and both operator flags are named as unable to lift it.
    assert "opt-out lifts only the metadata rule" in why
    assert "BENCHFLOW_ALLOW_TASK_HOST_ACCESS" in why
    assert "BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD" in why


@pytest.mark.parametrize("power", list(POWERS))
def test_an_open_network_keeps_them(tmp_path, power):
    service, lines, _ = POWERS[power]
    assert (
        policy_power_refusal(_config("", True), _compose(tmp_path, service, lines))
        is None
    )


def test_a_policy_with_no_way_around_starts(tmp_path):
    compose = _compose(tmp_path, "db", "    cap_add: [SYS_PTRACE]\n")
    assert compose_network_powers(compose) == {}
    for policy in POLICIES.values():
        assert policy_power_refusal(_config(policy, False), compose) is None


@pytest.mark.asyncio
async def test_a_host_network_service_under_an_open_network_gets_no_rule():
    """Only an open network gets here (a policy refused it before start): the
    record says so, and nothing is installed in the host's namespace."""
    env = SimpleNamespace()
    target = {
        "service": "db",
        "id": "c" * 64,
        "network_mode": "host",
        "cap_add": [],
        "privileged": False,
    }
    record = await _cover_one(env, target, "public", True, False, (), ())
    assert record["state"] == "host network"
    assert record["why"] == "no rules; service 'db' is the host's network"
    assert record["proven"] is False


class _Docker:
    """``docker inspect`` and the netns probes, answered."""

    def __init__(self, mode: str, inode_main: str, inode_host: str) -> None:
        self.mode, self.inode_main, self.inode_host = mode, inode_main, inode_host
        self.calls: list[list[str]] = []

    async def __call__(self, args, *, timeout, stdin=None):
        self.calls.append(list(args))
        out = ""
        if args[:1] == ["inspect"]:
            out = self.mode
        elif args[:1] == ["run"] and "stat" in args:
            out = (
                self.inode_host
                if args[args.index("--network") + 1] == "host"
                else self.inode_main
            )
        return SimpleNamespace(stdout=out + "\n", stderr="", return_code=0)


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(helper_mod, "_HOST_NETNS", {})
    monkeypatch.setattr(helper_mod, "_NETNS_IS_HOST", {})

    async def image(docker, daemon):
        return "sha256:" + "e" * 64

    monkeypatch.setattr(helper_mod, "ensure_image", image)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "inode_main", "inode_host", "where"),
    [
        ("host", "", "", "host"),
        ("bf_default", "4026532900", "4026531992", ""),
        # Named as its own network, but the same namespace as the host's.
        ("bf_default", "4026531992", "4026531992", "host"),
        ("container:" + "d" * 64, "4026531992", "4026531992", "host"),
        # Unread inodes: a network the daemon names is its own; else unread.
        ("bf_default", "", "", ""),
        ("", "", "", "unread"),
    ],
)
async def test_the_hosts_namespace_is_told_apart(
    fresh, mode, inode_main, inode_host, where
):
    docker = _Docker(mode, inode_main, inode_host)
    assert await helper_mod.host_netns(docker, "d1", "c" * 64) == where


@pytest.mark.asyncio
async def test_no_helper_ever_joins_the_hosts_namespace(fresh):
    """The guard RCI asked for: on a main whose namespace is the host's, the
    helper never runs a container joined to it, so no rule is installed."""
    docker = _Docker("host", "", "")
    helper = helper_mod.MetadataHelper(
        docker, daemon="d1", name="t-bf-metadata", labels={}
    )
    assert await helper.installed("c" * 64, "echo rules") is None
    assert "host's" in helper.unavailable
    joined = [
        c
        for c in docker.calls
        if c[:1] == ["run"] and "container:" + "c" * 64 in c and "stat" not in c
    ]
    assert joined == []


def test_a_policy_refusal_is_never_retried(tmp_path):
    """RCI on 39fb4e057: a start refusal decided by the task's files and the
    run's settings ends once, with its reason; another attempt meets it again.
    A helper build or netns probe that failed stays retryable."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error

    why = policy_power_refusal(
        _config(POLICIES["allowlist"], True),
        _compose(tmp_path, "main", "    cap_add: [NET_ADMIN]\n"),
    )
    assert is_unrecoverable_startup_error(f"ServiceCoverError: {why}")
    for refused in (
        "sandbox startup refused: the task's compose file gives main NET_ADMIN, so ",
        "sandbox startup refused: the task's compose service 'db' holds privileged, so ",
        "sandbox startup refused: main joins the network namespace of service 'api' "
        "(network_mode: service:api), which is the host's own network",
        "RuntimeError: sandbox startup refused: the task's network policy (allowlist) "
        "needs BenchFlow's helper container to add main's rules from outside it; "
        "BENCHFLOW_METADATA_HELPER=0 would leave main holding NET_ADMIN",
        # Plain no-network and a verifier's own network (RCI, a80425fb).
        "RuntimeError: sandbox startup refused: the task's network policy "
        "(no-network, the verifier's) needs BenchFlow's helper container to add "
        "main's rules from outside it; BENCHFLOW_METADATA_MAIN_NET_ADMIN=1 would "
        "leave main holding NET_RAW, whose packet sockets go around rules in its "
        "namespace",
    ):
        assert is_unrecoverable_startup_error(refused), refused
    for transient in (
        "sandbox startup refused: the task's network policy (allowlist) needs "
        "BenchFlow's helper container to add main's rules from outside it; a helper "
        "image that failed to build would leave main holding NET_ADMIN",
        "sandbox startup refused: main's network namespace could not be told from "
        "the host's, so its cloud metadata rule could not go in",
    ):
        assert not is_unrecoverable_startup_error(transient), transient

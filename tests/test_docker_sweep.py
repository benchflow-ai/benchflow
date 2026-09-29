"""The Docker leftover sweep never removes a live sandbox's container or network.

Guards the fix for concurrent Evaluations on Docker deleting each other's
containers. ``Evaluation._prune_docker`` ran ``docker container prune`` and
``docker network prune`` filtered only by ``benchflow.owned=true`` when a job
started, before a retry and when it ended. That removed every stopped
BenchFlow container and unused BenchFlow network on the daemon, including
another rollout's container between Compose's create and start ("container
is marked for removal and cannot be started", the hill-climb Docker scenario
before b714ca2c), its network before the container joined ("network
<project>_default not found", retried as a daemon race since 782aee45), and
the ``main`` container branch restore had just stopped ("removal ... already
in progress", test_branch_two_children_in_place on a shared daemon).

The fake daemon below applies Docker's own rules: ``ps``/``network ls``
filters (label AND, status OR), ``rm`` without ``--force`` refuses a running
container, ``network rm`` refuses a network a running container uses.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.evaluation import BENCHFLOW_OWNED_LABEL, Evaluation, EvaluationConfig
from benchflow.sandbox import _docker_sweep as sweep
from benchflow.sandbox._docker_sweep import (
    ProcessId,
    claim_project,
    current_process,
    owner_state,
    process_token,
    release_project,
)

ALIVE_PID = 4242
GONE_PID = 4343


@dataclass
class FakeResource:
    id: str
    labels: dict[str, str]
    status: str = "running"  # containers: created/running/exited/dead
    network: str | None = None  # the network a container is attached to


@dataclass
class FakeDaemon:
    containers: list[FakeResource] = field(default_factory=list)
    networks: list[FakeResource] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def ids(self, kind: str) -> set[str]:
        return {r.id for r in getattr(self, kind)}

    def run(self, argv, **kwargs):
        assert argv[0] == "docker"
        args = list(argv[1:])
        self.calls.append(args)
        if args[0] == "ps":
            return self._listing(self.containers, args)
        if args[:2] == ["network", "ls"]:
            return self._listing(self.networks, args)
        if args[0] == "rm":
            assert "--force" not in args and "-f" not in args
            return self._rm_containers([a for a in args[1:] if not a.startswith("-")])
        if args[:2] == ["network", "rm"]:
            return self._rm_networks(args[2:])
        raise AssertionError(f"unexpected docker call: {args}")

    @staticmethod
    def _filters(args: list[str]) -> tuple[dict[str, str], set[str]]:
        labels: dict[str, str] = {}
        statuses: set[str] = set()
        for i, arg in enumerate(args):
            if arg != "--filter":
                continue
            key, _, value = args[i + 1].partition("=")
            if key == "label":
                name, _, wanted = value.partition("=")
                labels[name] = wanted
            elif key == "status":
                statuses.add(value)
        return labels, statuses

    def _listing(self, resources, args):
        labels, statuses = self._filters(args)
        rows = []
        for r in resources:
            if any(r.labels.get(k) != v for k, v in labels.items()):
                continue
            if statuses and r.status not in statuses:
                continue
            rows.append(
                "\t".join(
                    [
                        r.id,
                        r.labels.get("benchflow.process", ""),
                        r.labels.get("com.docker.compose.project", ""),
                    ]
                )
            )
        return subprocess.CompletedProcess(args, 0, "\n".join(rows) + "\n", "")

    def _rm_containers(self, ids):
        errors = []
        for cid in ids:
            (c,) = [c for c in self.containers if c.id == cid]
            if c.status == "running":
                errors.append(f"cannot remove container {cid}: container is running")
                continue
            self.containers.remove(c)
        return subprocess.CompletedProcess(
            ids, 1 if errors else 0, "", "\n".join(errors)
        )

    def _rm_networks(self, ids):
        errors = []
        for nid in ids:
            (n,) = [n for n in self.networks if n.id == nid]
            if any(c.network == nid and c.status == "running" for c in self.containers):
                errors.append(f"network {nid} has active endpoints")
                continue
            self.networks.remove(n)
        return subprocess.CompletedProcess(
            ids, 1 if errors else 0, "", "\n".join(errors)
        )


def _labels(project: str, token: str | None) -> dict[str, str]:
    labels = {"benchflow.owned": "process", "com.docker.compose.project": project}
    if token is not None:
        labels["benchflow.process"] = token
    return labels


def _token(pid: int, *, host: str | None = None, boot: str | None = None) -> str:
    me = current_process()
    return ProcessId(
        host if host is not None else me.host,
        pid,
        me.boot if boot is None else boot,
        "",
    ).token()


def _add_project(daemon, project, token, *, status="created"):
    """A compose project's default network and ``main`` container."""
    net = FakeResource(f"net-{project}", _labels(project, token))
    main = FakeResource(f"c-{project}", _labels(project, token), status, net.id)
    daemon.networks.append(net)
    daemon.containers.append(main)
    return main.id, net.id


@pytest.fixture
def daemon(monkeypatch):
    fake = FakeDaemon()
    monkeypatch.setattr(sweep.subprocess, "run", fake.run)
    monkeypatch.setattr(sweep, "_pid_exists", lambda pid: pid != GONE_PID)
    yield fake
    for project in list(sweep.live_projects()):
        if project.startswith("t-"):
            release_project(project)


def _evaluation(tmp_path: Path, name: str) -> Evaluation:
    tasks = tmp_path / name / "tasks"
    tasks.mkdir(parents=True)
    return Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / name / "jobs",
        job_name=name,
        config=EvaluationConfig(environment="docker"),
    )


def test_two_evaluations_at_once_keep_each_others_containers(daemon, tmp_path):
    """Job A's sandbox is between Compose's create and start (container
    ``created``, network with no running container) while job B, in the same
    process, starts and ends: B's sweeps leave A's container and network, and
    remove A's leftovers only once A's sandbox is torn down."""
    job_a, job_b = _evaluation(tmp_path, "a"), _evaluation(tmp_path, "b")
    claim_project("t-job-a")
    a_container, a_network = _add_project(daemon, "t-job-a", process_token())

    async def both():
        # B's start and end sweeps, A's retry sweep, all concurrently.
        await asyncio.gather(
            job_b._sweep_docker(), job_a._sweep_docker(), job_b._sweep_docker()
        )

    asyncio.run(both())
    assert a_container in daemon.ids("containers")
    assert a_network in daemon.ids("networks")

    release_project("t-job-a")
    job_b._prune_docker()
    assert a_container not in daemon.ids("containers")
    assert a_network not in daemon.ids("networks")


@pytest.mark.parametrize(
    ("token", "kept"),
    [
        pytest.param(lambda: _token(ALIVE_PID), True, id="another-live-process"),
        pytest.param(lambda: _token(GONE_PID), False, id="exited-process"),
        pytest.param(
            lambda: _token(ALIVE_PID, host="other-host"), True, id="another-host"
        ),
        pytest.param(lambda: None, True, id="no-label-older-benchflow"),
        pytest.param(lambda: "garbage", True, id="unparseable-label"),
    ],
)
def test_other_processes_resources_are_removed_only_when_that_process_is_gone(
    daemon, tmp_path, token, kept
):
    container, network = _add_project(daemon, "t-other", token())
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert (container in daemon.ids("containers")) is kept
    assert (network in daemon.ids("networks")) is kept


def test_a_previous_boot_is_gone_even_if_its_pid_is_reused(daemon, tmp_path):
    if not current_process().boot:
        pytest.skip("boot ids are read from /proc (Linux)")
    container, _ = _add_project(daemon, "t-boot", _token(ALIVE_PID, boot="0" * 36))
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container not in daemon.ids("containers")


def test_a_reused_pid_is_gone(monkeypatch):
    """Same host and boot, the pid exists, but it started at another time."""
    monkeypatch.setattr(sweep, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(sweep, "_start_ticks", lambda pid: "200")
    me = current_process()
    assert owner_state(ProcessId(me.host, ALIVE_PID, me.boot, "100").token()) == "gone"
    assert owner_state(ProcessId(me.host, ALIVE_PID, me.boot, "200").token()) == "alive"


def test_running_containers_and_their_networks_are_never_removed(daemon, tmp_path):
    """Even a gone process's running container (``docker rm`` without
    ``--force`` refuses it) and the network it uses stay."""
    container, network = _add_project(
        daemon, "t-running", _token(GONE_PID), status="running"
    )
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container in daemon.ids("containers")
    assert network in daemon.ids("networks")


def test_this_process_without_a_project_label_is_kept(daemon, tmp_path):
    container, _ = _add_project(daemon, "", process_token())
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container in daemon.ids("containers")


def test_every_listing_is_scoped_to_benchflow_resources_and_nothing_is_pruned(
    daemon, tmp_path
):
    """#418: never a daemon-wide prune, and only BenchFlow-labelled ids removed.
    An older BenchFlow's container (``benchflow.owned=true``) is not listed:
    the sweep cannot tell whether its process lives."""
    foreign = FakeResource("foreign", {"com.docker.compose.project": "x"}, "exited")
    older = FakeResource(
        "older",
        {"benchflow.owned": "true", "benchflow.process": _token(GONE_PID)},
        "exited",
    )
    daemon.containers += [foreign, older]
    _add_project(daemon, "t-gone", _token(GONE_PID))
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert foreign in daemon.containers and older in daemon.containers
    listings = [c for c in daemon.calls if c[0] == "ps" or c[:2] == ["network", "ls"]]
    assert len(listings) == 2
    for call in listings:
        assert f"label={BENCHFLOW_OWNED_LABEL}" in call
    assert not any("prune" in call for call in daemon.calls)


def test_the_sweep_skips_other_sandboxes_and_swallows_docker_failures(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: calls.append(a))
    job = Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        job_name="j",
        config=EvaluationConfig(environment="daytona"),
    )
    job._prune_docker()
    asyncio.run(job._sweep_docker())
    assert calls == []

    def boom(*args, **kwargs):
        raise OSError("docker not found")

    monkeypatch.setattr(sweep.subprocess, "run", boom)
    _evaluation(tmp_path, "sweeper")._prune_docker()  # does not raise


def test_process_token_round_trips_and_names_this_process():
    parsed = ProcessId.parse(process_token())
    assert parsed == current_process()
    assert owner_state(process_token()) == "this"
    assert owner_state(None) == "unknown"
    assert ProcessId.parse("host:notapid::") is None
    assert ProcessId.parse("host:0::") is None


def test_a_forked_child_gets_its_own_token(monkeypatch):
    me = current_process()
    with patch.object(sweep.os, "getpid", return_value=me.pid + 1):
        child = current_process()
    assert child.pid == me.pid + 1
    assert current_process().pid == me.pid

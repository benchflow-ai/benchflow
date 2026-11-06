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
container, ``network rm`` refuses a network a running container uses, and both
print the ids they removed and exit 1 when any was refused.

Since DF-p3-eval-runner-9 (PY8 DX-G) a running container whose process is
provably gone in this pid namespace is force-removed: after a ``kill -9`` the
dead run's sandboxes kept running through the resume.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
from benchflow.sandbox._trial_volumes import LABEL as TRIAL_VOLUME_LABEL

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
    forced: list[str] = field(default_factory=list)
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)

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
            assert "-f" not in args
            ids = [a for a in args[1:] if not a.startswith("-")]
            if "--force" in args:
                self.forced += ids
            return self._rm_containers(ids, force="--force" in args)
        if args[:2] == ["network", "rm"]:
            return self._rm_networks(args[2:])
        if args[:2] == ["volume", "ls"]:
            assert args[2:4] == ["--filter", f"label={TRIAL_VOLUME_LABEL}"]
            rows = [
                f"{name}\t{labels[TRIAL_VOLUME_LABEL]}"
                for name, labels in self.volumes.items()
                if TRIAL_VOLUME_LABEL in labels
            ]
            return subprocess.CompletedProcess(args, 0, "\n".join(rows), "")
        if args[:2] == ["volume", "rm"]:
            assert "-f" not in args and "--force" not in args
            for name in args[2:]:
                self.volumes.pop(name)
            return subprocess.CompletedProcess(args, 0, "\n".join(args[2:]), "")
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
        fmt = args[args.index("--format") + 1]
        rows = []
        for r in resources:
            if any(r.labels.get(k) != v for k, v in labels.items()):
                continue
            if statuses and r.status not in statuses:
                continue
            columns = [
                r.id,
                r.labels.get("benchflow.process", ""),
                r.labels.get("com.docker.compose.project", ""),
            ]
            if "{{.State}}" in fmt:
                columns += [r.status, f"name-{r.id}"]
            rows.append("\t".join(columns))
        return subprocess.CompletedProcess(args, 0, "\n".join(rows) + "\n", "")

    def _rm_containers(self, ids, *, force=False):
        errors, removed = [], []
        for cid in ids:
            (c,) = [c for c in self.containers if c.id == cid]
            if c.status == "running" and not force:
                errors.append(f"cannot remove container {cid}: container is running")
                continue
            self.containers.remove(c)
            removed.append(cid)
        return subprocess.CompletedProcess(
            ids,
            1 if errors else 0,
            "".join(f"{i}\n" for i in removed),
            "\n".join(errors),
        )

    def _rm_networks(self, ids):
        errors, removed = [], []
        for nid in ids:
            (n,) = [n for n in self.networks if n.id == nid]
            if any(c.network == nid and c.status == "running" for c in self.containers):
                errors.append(f"network {nid} has active endpoints")
                continue
            self.networks.remove(n)
            removed.append(nid)
        return subprocess.CompletedProcess(
            ids,
            1 if errors else 0,
            "".join(f"{i}\n" for i in removed),
            "\n".join(errors),
        )


def _labels(project: str, token: str | None) -> dict[str, str]:
    labels = {"benchflow.owned": "process", "com.docker.compose.project": project}
    if token is not None:
        labels["benchflow.process"] = token
    return labels


def _this_pid_namespace() -> str:
    """This process's pid namespace inode, read here so the labels below are
    the 0.8.3 form whatever the code under test computes."""
    try:
        return "".join(c for c in os.readlink("/proc/self/ns/pid") if c.isdigit())
    except OSError:
        return ""


def _token(
    pid: int,
    *,
    host: str | None = None,
    boot: str | None = None,
    ns: str | None = None,
) -> str:
    """A ``benchflow.process`` label: ``<host>:<pid>:<boot>:<start>[:<ns>]``."""
    me = current_process()
    ns = _this_pid_namespace() if ns is None else ns
    label = f"{me.host if host is None else host}:{pid}:"
    label += f"{me.boot if boot is None else boot}:"
    return f"{label}:{ns}" if ns else label


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


def test_a_stopped_container_from_before_0_8_3_and_a_previous_boot_is_removed(
    daemon, tmp_path
):
    """Labels without the namespace part keep the stopped-only clean-up."""
    if not current_process().boot:
        pytest.skip("boot ids are read from /proc (Linux)")
    old = ProcessId(current_process().host, ALIVE_PID, "0" * 36, "1")
    container, network = _add_project(daemon, "t-old-boot", old.token())
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container not in daemon.ids("containers")
    assert network not in daemon.ids("networks")
    assert daemon.forced == []


def test_a_reused_pid_is_gone(monkeypatch):
    """Same host and boot, the pid exists, but it started at another time."""
    monkeypatch.setattr(sweep, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(sweep, "_start_ticks", lambda pid: "200")
    me = current_process()
    assert owner_state(ProcessId(me.host, ALIVE_PID, me.boot, "100").token()) == "gone"
    assert owner_state(ProcessId(me.host, ALIVE_PID, me.boot, "200").token()) == "alive"


def test_a_killed_runs_running_container_and_its_network_are_removed(daemon, tmp_path):
    """DF-p3-eval-runner-9: after a kill -9 the resume finished the job and
    left the dead run's containers ``Up`` with their networks. A running
    container whose process is provably gone in this pid namespace is now
    force-removed, and its network with it."""
    if not _this_pid_namespace():
        pytest.skip("pid namespaces are Linux-only; elsewhere nothing is forced")
    container, network = _add_project(
        daemon, "t-killed", _token(GONE_PID), status="running"
    )
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container not in daemon.ids("containers")
    assert network not in daemon.ids("networks")
    assert daemon.forced == [container]


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(lambda: _token(ALIVE_PID), id="live-process"),
        pytest.param(lambda: process_token(), id="this-process"),
        pytest.param(lambda: _token(GONE_PID, host="other-host"), id="another-host"),
        # A pid checked in our namespace says nothing about another's.
        pytest.param(lambda: _token(GONE_PID, ns="1"), id="another-pid-namespace"),
        # Labels before 0.8.3 name no namespace: never forced.
        pytest.param(
            lambda: ProcessId(
                current_process().host, GONE_PID, current_process().boot, ""
            ).token(),
            id="label-without-namespace",
        ),
        # DX-G-ASTRA-1: nor from another boot; its pid may still run.
        pytest.param(
            lambda: ProcessId(current_process().host, 99999999, "0" * 36, "1").token(),
            id="label-without-namespace-another-boot",
        ),
        pytest.param(lambda: "garbage", id="unparseable-label"),
    ],
)
def test_other_running_containers_and_their_networks_stay(daemon, tmp_path, token):
    if _this_pid_namespace() == "1":
        pytest.skip("this process's pid namespace is the test's 'other' one")
    container, network = _add_project(daemon, "t-running", token(), status="running")
    claim_project("t-running")
    _evaluation(tmp_path, "sweeper")._prune_docker()
    assert container in daemon.ids("containers")
    assert network in daemon.ids("networks")
    assert daemon.forced == []


def test_a_dry_run_lists_and_removes_nothing(daemon):
    """``bench sandbox cleanup --dry-run`` lists what the sweep would remove."""
    if not _this_pid_namespace():
        pytest.skip("pid namespaces are Linux-only; elsewhere nothing is forced")
    running, running_net = _add_project(
        daemon, "t-killed", _token(GONE_PID), status="running"
    )
    stopped, _ = _add_project(daemon, "t-stopped", _token(GONE_PID))
    result = sweep.sweep_leftovers(dry_run=True)
    assert sorted(result.containers) == sorted([running, stopped])
    assert result.running == [running]
    assert running_net in result.networks
    assert result.projects == ["t-killed", "t-stopped"]
    assert not [c for c in daemon.calls if c[0] == "rm" or c[:2] == ["network", "rm"]]


def test_a_dry_run_leaves_out_a_network_a_kept_container_uses(daemon):
    """A gone process's running container with a label before 0.8.3 is
    kept, so its network cannot be removed either: the dry run must not
    list it (a shared VM's other run showed up as \"would remove\")."""
    old = ProcessId(current_process().host, GONE_PID, current_process().boot, "")
    _, network = _add_project(daemon, "t-old", old.token(), status="running")
    result = sweep.sweep_leftovers(dry_run=True)
    assert network not in result.networks
    assert result.projects == []


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
    # Stopped containers, running containers, networks.
    assert len(listings) == 3
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
    # A label written before 0.8.3 (no namespace part) still parses.
    assert ProcessId.parse("host:7:boot:99") == ProcessId("host", 7, "boot", "99")
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


# --- what the sweep reports (DF-p1-first-run-5) ------------------------------


@pytest.fixture
def fresh_reports(monkeypatch):
    """Each test starts with no running orphan reported yet."""
    monkeypatch.setattr(sweep, "_reported_orphans", set(), raising=False)


def _sweep_messages(caplog) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="benchflow"):
        sweep.sweep_leftovers()
    return [r.getMessage() for r in caplog.records]


# The running orphans below carry a label from before 0.8.3 (no pid namespace):
# the sweep may not force-remove those (is_dead_running), so they stay and are
# reported. A 0.8.3 label of a gone process is force-removed (DX-G, below).
_OLD_GONE = {"ns": ""}


def test_the_log_counts_what_docker_removed_not_what_was_asked(
    daemon, caplog, fresh_reports
):
    """A gone process left one stopped project and one running one. Docker
    removes the stopped one; it refuses the network the running container
    still uses, so that network is not counted as removed."""
    _add_project(daemon, "t-stopped", _token(GONE_PID), status="exited")
    _, held = _add_project(
        daemon, "t-orphan", _token(GONE_PID, **_OLD_GONE), status="running"
    )

    messages = _sweep_messages(caplog)

    assert held in daemon.ids("networks")
    assert (
        "Removed leftover BenchFlow Docker resources: 1 container(s), 1 network(s)"
        in messages
    )


def test_nothing_removed_prints_no_removal_line(daemon, caplog, fresh_reports):
    """The dogfood host: 4 running orphans hold 4 networks; nothing goes, and
    the log said "Removed ... 0 container(s), 4 network(s)" on every run."""
    for n in range(4):
        _add_project(
            daemon, f"t-orphan-{n}", _token(GONE_PID, **_OLD_GONE), status="running"
        )

    messages = _sweep_messages(caplog)

    assert not any(m.startswith("Removed leftover") for m in messages)
    assert len(daemon.ids("networks")) == 4


def test_running_orphans_are_named_once_with_the_command_that_clears_them(
    daemon, caplog, fresh_reports
):
    cid, _ = _add_project(
        daemon, "t-orphan", _token(GONE_PID, **_OLD_GONE), status="running"
    )
    name = f"name-{cid}"

    first = _sweep_messages(caplog)
    second = _sweep_messages(caplog)

    (warning,) = [m for m in first if "still running" in m]
    assert warning.startswith("1 BenchFlow container(s) are still running")
    assert "the BenchFlow process that started them has exited" in warning
    assert name in warning
    assert "they hold 1 BenchFlow network(s)" in warning
    assert f"`docker rm -f {name}` removes them" in warning
    assert cid in daemon.ids("containers")  # reported, never stopped
    assert not [m for m in second if "still running" in m]


def test_a_live_process_running_container_is_not_reported(
    daemon, caplog, fresh_reports
):
    _add_project(daemon, "t-live", _token(ALIVE_PID), status="running")

    messages = _sweep_messages(caplog)

    assert not [m for m in messages if "still running" in m]


def test_a_forced_removal_docker_refused_is_reported_not_counted(
    daemon, caplog, fresh_reports, monkeypatch
):
    """Merge wave batch 34 (integ's accounting x DX-G's forced removal): a
    0.8.3-labelled running container of a gone process is force-removed, but
    when Docker does not confirm it, it is named as still running and not
    counted as removed."""
    cid, _ = _add_project(daemon, "t-dead", _token(GONE_PID), status="running")
    real = sweep._docker_remove
    monkeypatch.setattr(
        sweep,
        "_docker_remove",
        lambda args, ids: [] if "--force" in args else real(args, ids),
    )

    messages = _sweep_messages(caplog)

    (warning,) = [m for m in messages if "still running" in m]
    assert f"name-{cid}" in warning
    assert cid in daemon.ids("containers")
    assert not any(m.startswith("Removed leftover") for m in messages)


def test_a_swept_project_takes_only_the_volumes_labelled_for_it(daemon):
    """DOCKER-ANON-VOLUME-LEAK r3: no ``rm --volumes`` (which would also take
    the operator's volume a container inherited through ``volumes_from``),
    and no inspect of what a container mounted: the swept project's volumes
    are the ones BenchFlow labelled for it at creation. Another project's,
    the operator's and bfx's stay."""
    cid, _ = _add_project(daemon, "t-gone", _token(GONE_PID), status="exited")
    _add_project(daemon, "t-live", _token(os.getpid()), status="running")
    claim_project("t-live")
    daemon.volumes = {
        "t-gone_bfvol1_main_0": {TRIAL_VOLUME_LABEL: "t-gone"},
        "t-live_bfvol1_main_0": {TRIAL_VOLUME_LABEL: "t-live"},
        "f" * 64: {"com.docker.volume.anonymous": ""},
        "operator-data": {},
        "benchflow-bfx-4e2b672d1163": {"benchflow.bfx.volume": "x"},
    }

    result = sweep.sweep_leftovers()

    assert cid not in daemon.ids("containers")
    assert not any("--volumes" in c or c[0] == "inspect" for c in daemon.calls)
    assert result.removed_volumes == ["t-gone_bfvol1_main_0"]
    assert set(daemon.volumes) == {
        "t-live_bfvol1_main_0",
        "f" * 64,
        "operator-data",
        "benchflow-bfx-4e2b672d1163",
    }

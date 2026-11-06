"""Kept services are held by a frozen cgroup, not by SIGSTOP (K1-CONFIRM-1).

At 38eb884a the kills held a kept listener only with SIGSTOP, and the kernel
sends SIGCONT with no sender running (a POSIX timer the listener armed with
SIGCONT, a parent-death or child-exit signal, an orphaned process group), so
the listener could run on through the capture and the hardening. The kills
now move each kept process into the frozen leaf /sys/fs/cgroup/benchflow-kept
as soon as they find it, print their receipt only once the leaf is frozen
and holds it, and only the release thaws the leaf, after its checks. Where
root cannot hold the leaf, a kill that would keep something fails.

These run the real scripts against K1's stand-in /proc with the cgroup files
in a folder: ``freezer_write`` and ``freezer_members`` are stubbed with the
kernel's meaning (a write to cgroup.freeze sets cgroup.events), ``stat``
reports the folder as cgroup2fs and root's. The kernel side (a timer's
SIGCONT cannot thaw the leaf) is shown in a container on the build VM, in
the K1 round report.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.sandbox.lockdown import (
    KeptServiceFreezerUnavailable,
    _kill_sandbox_user_procs_keeping,
    probe_kept_service_freezer,
    release_kept_services,
)
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task, keep_services
from benchflow.task.plan_validation import validate_task_plan
from tests.test_keep_services_kill import _Async, _logs, _NetProc
from tests.test_verifier_process_quiescence import _LIVE


def _events(proc):
    path = proc.cgroups / "events"
    return path.read_text().splitlines() if path.exists() else []


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["setpriv", "su"])
@pytest.mark.parametrize("v6", [False, True])
@pytest.mark.parametrize("wait", [False, True])
async def test_sigcont_cannot_open_the_hold(tmp_path, helper, v6, wait):
    """Both kills hold the listener in the frozen leaf; a SIGCONT does not
    thaw it; only the release does, as its last step."""
    proc = _NetProc(tmp_path)
    proc.freezer_overrides = (
        f"{'su' if helper == 'setpriv' else 'setpriv'}() {{ return 1; }}\n"
    )
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000, v6=v6))
    proc.place(pid)
    env = _Async(proc)
    for initial_wait in (wait, False):
        result = await _kill_sandbox_user_procs_keeping(
            env, "agent", (5000,), wait=initial_wait
        )
        assert [p.pid for p in result.kept] == [pid]
        events = _events(proc)
        assert "cgroup.freeze 1" in events
        assert f"cgroup.procs {pid}" in events
        assert "cgroup.freeze 0" not in events
        # A timer's SIGCONT ends the job-control stop and nothing else.
        proc.run(f"kill -s CONT {pid}")
        assert (proc.freezer / "cgroup.freeze").read_text().strip() == "1"
    out = await release_kept_services(
        env, "agent", (pid,), logs_dir=str(_logs(tmp_path))
    )
    assert out.released == 1
    assert _events(proc)[-1] == "cgroup.freeze 0"


@pytest.mark.asyncio
async def test_kept_listener_is_frozen_before_the_others_die(tmp_path):
    """The listener enters the frozen leaf before the kill signals anything
    else, so it forks nothing while the rest are killed."""
    proc = _NetProc(tmp_path)
    signals = proc.signals
    proc.freezer_overrides = (
        "freezer_write() {\n"
        '    [ "$1" = cgroup.procs ] && echo "MOVE $2" >> ' + f"'{signals}'\n"
        '    case "$1" in\n'
        '        cgroup.procs) echo "$2" >> "$C/$1" ;;\n'
        '        cgroup.freeze) echo "$2" > "$C/$1"; echo "frozen $2" > "$C/cgroup.events" ;;\n'
        "    esac\n"
        "}\n"
    )
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    other = proc.add(_LIVE)
    await _kill_sandbox_user_procs_keeping(_Async(proc), "agent", (5000,), wait=False)
    lines = signals.read_text().splitlines()
    moved = lines.index(f"MOVE {pid}")
    ended = [
        i for i, s in enumerate(lines) if s.endswith(f" {other}") and "STOP" not in s
    ]
    assert ended, lines
    assert moved < ended[0], lines


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["unsupported", "write-freeze", "write-procs", "not-frozen", "not-moved"],
)
async def test_no_keeping_receipt_without_an_enforced_boundary(tmp_path, failure):
    """No cgroup v2, a failed freeze or move, or a leaf that never reports
    frozen: the kill fails rather than keep the listener unheld."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    proc.freezer_overrides = {
        "unsupported": "stat() { return 1; }",
        "write-freeze": 'freezer_write() { [ "$1" != cgroup.freeze ]; }',
        "write-procs": 'freezer_write() { [ "$1" != cgroup.procs ]; }',
        "not-frozen": 'freezer_write() { echo "frozen 0" > "$C/cgroup.events"; }',
        "not-moved": "freezer_members() { :; }",
    }[failure] + "\n"
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill_sandbox_user_procs_keeping(
            _Async(proc), "agent", (5000,), wait=False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fstype", ["cgroup2", "tmpfs", "unmounted"])
async def test_the_cgroup_mount_type_comes_from_mountinfo(tmp_path, fstype):
    """K1-R8-1: BusyBox's ``stat -f -c %T`` prints UNKNOWN for cgroup v2,
    so the type is read from /proc/self/mountinfo: a cgroup2 mount keeps the
    listener whatever stat says; cgroup v1 (tmpfs) or no mount refuses."""
    proc = _NetProc(tmp_path)
    if fstype == "unmounted":
        (proc.root / "self/mountinfo").write_text(
            "498 400 0:60 / / rw,relatime - overlay overlay rw\n"
        )
    else:
        proc.mount_cgroup(fstype)
    proc.freezer_overrides = (
        'stat() { [ "$1" = -f ] && { echo UNKNOWN; return 0; }; cgroup_stat "$@"; }\n'
    )
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    env = _Async(proc)
    if fstype == "cgroup2":
        result = await _kill_sandbox_user_procs_keeping(
            env, "agent", (5000,), wait=False
        )
        assert [p.pid for p in result.kept] == [pid]
        assert f"cgroup.procs {pid}" in _events(proc)
    else:
        with pytest.raises(RuntimeError, match="not cgroup v2"):
            await _kill_sandbox_user_procs_keeping(env, "agent", (5000,), wait=False)
        assert _events(proc) == []


@pytest.mark.asyncio
async def test_no_keeping_outside_a_cgroup_namespace_of_its_own(tmp_path):
    """With the host's cgroup namespace the leaf would be the host's."""
    proc = _NetProc(tmp_path)
    (proc.root / "self/cgroup").write_text("0::/system.slice/docker-1.scope\n")
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill_sandbox_user_procs_keeping(
            _Async(proc), "agent", (5000,), wait=False
        )
    assert f"cgroup.procs {pid}" not in _events(proc)


@pytest.mark.asyncio
async def test_a_kill_that_keeps_nothing_needs_no_freezer(tmp_path):
    """A task whose agent started no listener is killed as before, on a
    backend with no writable cgroup."""
    proc = _NetProc(tmp_path)
    proc.freezer_overrides = "stat() { return 1; }\n"
    other = proc.add(_LIVE)
    result = await _kill_sandbox_user_procs_keeping(
        _Async(proc), "agent", (5000,), wait=False
    )
    assert result.kept == ()
    assert result.killed == 1
    assert other not in proc.live_pids()
    assert _events(proc) == []


@pytest.mark.asyncio
async def test_a_kept_service_that_exits_is_lost_not_a_failure(tmp_path):
    """A kept process gone before it could be moved is no failure of the
    kill (the release lists it as lost)."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    proc.freezer_overrides = (
        "freezer_write() {\n"
        '    case "$1" in\n'
        '        cgroup.procs) command rm -rf "$P/$2"; return 1 ;;\n'
        '        cgroup.freeze) echo "$2" > "$C/$1"; echo "frozen $2" > "$C/cgroup.events" ;;\n'
        "    esac\n"
        "}\n"
    )
    result = await _kill_sandbox_user_procs_keeping(
        _Async(proc), "agent", (5000,), wait=False
    )
    assert [p.pid for p in result.kept] == [pid]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node",
    [
        ".",
        "benchflow-kept",
        "cgroup.procs",
        "benchflow-kept/cgroup.procs",
        "benchflow-kept/cgroup.freeze",
        "benchflow-kept/cgroup.events",
    ],
)
@pytest.mark.parametrize("mode", [0o775, 0o777])
async def test_agent_writable_freezer_controls_are_refused(tmp_path, node, mode):
    """The agent can neither thaw the leaf nor move a process out of it
    through the common parent."""
    proc = _NetProc(tmp_path)
    (proc.cgroups / node).chmod(mode)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill_sandbox_user_procs_keeping(
            _Async(proc), "agent", (5000,), wait=False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["unreadable", "writable", "missing-member", "extra-member"]
)
async def test_release_rechecks_the_boundary_and_exact_population(tmp_path, failure):
    """A leaf whose state cannot be read, that others may write, that lacks
    a kept process or holds another fails the release, and the leaf is never
    thawed."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.place(pid)
    if failure == "unreadable":
        (proc.freezer / "cgroup.freeze").write_text("")
    elif failure == "writable":
        (proc.freezer / "cgroup.freeze").chmod(0o666)
    elif failure == "missing-member":
        (proc.freezer / "cgroup.procs").write_text("")
    else:
        proc.add(_LIVE)
    with pytest.raises(RuntimeError, match="releasing"):
        await release_kept_services(
            _Async(proc), "agent", (pid,), logs_dir=str(_logs(tmp_path))
        )
    assert "cgroup.freeze 0" not in _events(proc)


@pytest.mark.asyncio
async def test_a_retried_release_finds_the_leaf_thawed(tmp_path):
    """A verifier retried in place releases again after the first release
    thawed the leaf: the services already run, so it goes on without
    freezing or thawing anything."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.place(pid)
    env = _Async(proc)
    logs = str(_logs(tmp_path))
    assert (
        await release_kept_services(env, "agent", (pid,), logs_dir=logs)
    ).released == 1
    assert _events(proc) == ["cgroup.freeze 0"]
    # The service forked a worker after the release: it joins the leaf.
    worker = proc.add(_LIVE)
    proc.place(worker)
    out = await release_kept_services(env, "agent", (pid,), logs_dir=logs)
    assert out.released == 1
    assert _events(proc) == ["cgroup.freeze 0"]


def _docker_task(root: Path, keep: bool, *, compose: bool = False) -> Path:
    task_dir = root / ("keeps" if keep else "plain")
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    metadata = (
        '[metadata]\n"x-keep-services" = [{ port = 5000, name = "api" }]\n'
        if keep
        else ""
    )
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        f"{metadata}"
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose:
        # A task compose file selects the Daytona DinD strategy.
        (task_dir / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main: {}\n"
        )
    return task_dir


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("keep", [True, False])
def test_docker_gives_a_keeping_task_a_cgroup_root_may_write(
    tmp_path, monkeypatch, backend, keep
):
    """The factory asks for nothing (the start decides); once
    the start has decided for the freezer, main gets the override, which asks
    for a private cgroup namespace with writable-cgroups."""
    if backend == "remote-docker":
        monkeypatch.setenv("DOCKER_HOST", "ssh://k1@example.invalid")
    task_dir = _docker_task(tmp_path, keep)
    sandbox = _create_sandbox_environment(
        backend,
        Task(task_dir),
        task_dir,
        "rollout",
        RolloutPaths(rollout_dir=tmp_path / "rollout"),
    )
    override = sandbox._DOCKER_COMPOSE_KEPT_FREEZER_PATH
    assert override.name == "docker-compose-kept-freezer.yaml"
    assert sandbox.kept_service_freezer is False
    assert override not in sandbox._docker_compose_paths
    sandbox.kept_service_freezer = keep  # what the start decided
    assert (override in sandbox._docker_compose_paths) is keep
    main = yaml.safe_load(override.read_text())["services"]["main"]
    assert main["cgroup"] == "private"
    assert main["security_opt"] == ["writable-cgroups=true"]


@pytest.mark.parametrize("keep", [True, False])
def test_daytona_compose_tasks_upload_and_stack_the_override(tmp_path, keep):
    """K1-R8-2: Daytona runs a Compose task in Docker inside the sandbox
    (DinD); its ``main`` gets the same override as on Docker, uploaded with
    the other BenchFlow compose files, so a kept service can be frozen."""
    task_dir = _docker_task(tmp_path, keep, compose=True)
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
    ):
        sandbox = _create_sandbox_environment(
            "daytona",
            Task(task_dir),
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / "rollout"),
        )
    assert isinstance(sandbox._strategy, _DaytonaDinD)
    name = "docker-compose-kept-freezer.yaml"
    uploaded = {path.name for path in _DaytonaDinD._BENCHFLOW_COMPOSE_FILES}
    assert name in uploaded
    assert sandbox.kept_service_freezer is False
    sandbox.kept_service_freezer = keep  # what the start decided
    flags = sandbox._strategy._compose_file_flags()
    assert (f"{_DaytonaDinD._COMPOSE_DIR}/{name}" in flags) is keep


@pytest.mark.parametrize(
    "sandbox, compose, kept, refused",
    [
        ("docker", False, True, False),
        ("remote-docker", False, True, False),
        ("daytona", True, True, False),
        ("daytona", False, True, True),
        ("modal", False, True, True),
        ("modal", False, False, False),
    ],
)
def test_a_sandbox_that_cannot_hold_kept_services_is_refused_first(
    tmp_path, sandbox, compose, kept, refused
):
    """K1-R8-2: a provider that gives root no writable cgroup of its own is
    refused a trial that keeps services, with what to run instead; a trial
    that keeps nothing by its measured identities (the agent runs as root,
    the verifier is not root) is not. Setup asks once the identities are
    measured, before any agent code."""
    task_dir = _docker_task(tmp_path, True, compose=compose)
    task = Task(task_dir)
    refusal = keep_services.provider_refusal(
        task,
        task_dir,
        sandbox=sandbox,
        keep=keep_services.for_task(task) if kept else None,
        agent="claude-agent-acp",
    )
    assert (refusal is not None) is refused
    if refused:
        assert "port 5000" in refusal
        assert "--sandbox docker" in refusal


@pytest.mark.parametrize("verifier_user", [None, "root", "tester", "1000"])
def test_the_plan_gate_leaves_it_to_the_measured_identities(tmp_path, verifier_user):
    """``bench tasks check`` and the run's plan gate (validate_task_plan)
    measure no account, so they cannot tell a root verifier, which keeps the
    services, from a non-root one, which keeps nothing and is never refused
    (DT2-R12-1): neither refuses the sandbox."""
    task_dir = _docker_task(tmp_path, True)
    if verifier_user is not None:
        toml = task_dir / "task.toml"
        toml.write_text(
            toml.read_text().replace(
                "[verifier]\n", f'[verifier]\nuser = "{verifier_user}"\n'
            )
        )
        assert f'user = "{verifier_user}"' in toml.read_text()
    validate_task_plan(
        task_dir, sandbox="daytona", agent="claude-agent-acp", sandbox_user="agent"
    )


# K1-R9-1: an admitted sandbox that cannot hold kept services (a cgroup v1
# host, an Engine without writable-cgroups) found out only in the kill after
# the agent's turn. The trial now tries the freezer in main before any agent
# code installs or runs.


def _probe_proc(tmp_path):
    proc = _NetProc(tmp_path)
    # The probe's throwaway child is a real process, not a stand-in entry.
    proc.freezer_overrides = 'freezer_members() { cat "$C/cgroup.procs"; }\n'
    return proc


@pytest.mark.asyncio
async def test_the_probe_freezes_a_process_and_leaves_the_leaf_thawed(tmp_path):
    proc = _probe_proc(tmp_path)
    await probe_kept_service_freezer(_Async(proc))
    events = _events(proc)
    assert events[0].startswith("cgroup.procs ")
    assert events[1:] == ["cgroup.freeze 1", "cgroup.freeze 0"]
    assert (proc.freezer / "cgroup.freeze").read_text().strip() == "0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, why",
    [
        ("cgroup-v1", "not cgroup v2"),
        ("host-namespace", "cgroup namespace"),
        ("writable", "writable by others"),
        ("no-move", "cannot move"),
        ("not-frozen", "did not report frozen"),
        ("not-listed", "not listed"),
    ],
)
async def test_the_probe_refuses_a_sandbox_that_cannot_hold(tmp_path, failure, why):
    """Each failing check stops the trial with the reason and what to do."""
    proc = _probe_proc(tmp_path)
    if failure == "cgroup-v1":
        proc.mount_cgroup("tmpfs")
    elif failure == "host-namespace":
        (proc.root / "self/cgroup").write_text("0::/system.slice/docker-1.scope\n")
    elif failure == "writable":
        (proc.freezer / "cgroup.freeze").chmod(0o666)
    else:
        proc.freezer_overrides = {
            "no-move": 'freezer_write() { [ "$1" != cgroup.procs ]; }\n'
            'freezer_members() { cat "$C/cgroup.procs"; }\n',
            "not-frozen": 'freezer_write() { echo "frozen 0" > "$C/cgroup.events"; }\n'
            'freezer_members() { cat "$C/cgroup.procs"; }\n',
            "not-listed": "freezer_members() { :; }\n",
        }[failure]
    with pytest.raises(KeptServiceFreezerUnavailable) as caught:
        await probe_kept_service_freezer(_Async(proc))
    message = str(caught.value)
    assert why in message
    assert "No agent code ran" in message
    assert "Cgroup Version: 2" in message

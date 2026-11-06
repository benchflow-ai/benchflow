"""The kills before grading keep the services a task names (K2, x-keep-services).

A task names TCP ports in ``x-keep-services``; the agent user's processes
listening on them (IPv4 or IPv6) outlive the kill at the end of the agent's
turn and the kill before grading, every other process of the agent's dies as
before (K1), and root's are never signalled. Kept processes stay frozen while
root prepares the verifier and go on just before the test command.

These run the real kill through ``harden_before_verify`` and the turn-end
capture against K1's file-backed stand-in ``/proc`` (``id``, ``sleep``,
``kill``, ``setpriv``, ``su`` and here ``readlink`` are shell functions or act
on the stand-in only), extended with ``net/tcp``, ``net/tcp6`` and ``fd``
links, so no real process is ever signalled. On ``integ/py-0.8`` (no key) the
kills kill the listeners, so the keeping tests fail there.
"""

from __future__ import annotations

import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.rollout import _review
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox._identities import Identity
from benchflow.sandbox.lockdown import _close_workspace_cmd, harden_before_verify
from tests._receipt_fakes import inner_command, receipt_aware
from tests.test_sandbox_hardening import _blank_manifest, _make_env, _manifest_env
from tests.test_verifier_process_quiescence import (
    _LIVE,
    _ROOT,
    _ZOMBIE,
    _ZOMBIE_LEADER_LIVE,
    UID,
    _FakeProc,
)

# A process whose real uid is the agent's and whose effective uid is root's
# (a setuid program the agent started).
_SETUID_ROOT = f"State:\tS (sleeping)\nUid:\t{UID}\t0\t0\t0\nThreads:\t1\n"

_TCP_HEADER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
    "retrnsmt   uid  timeout inode\n"
)


class _NetProc(_FakeProc):
    """K1's stand-in /proc with sockets: ``listen`` adds a socket line to
    net/tcp or net/tcp6, ``hold`` gives a process an fd link to a socket.

    ``readlink`` of an fd under a process marked ``ptrace_only`` fails for
    root and works for the agent user (inside ``setpriv``), as in a Docker
    container where root has no CAP_SYS_PTRACE. Under one marked
    ``ptrace_empty`` root's read fails the same way on Linux 7.0, except
    for the ``ns/`` links, which read as an empty name with status 0.
    """

    def __init__(self, tmp_path, **kw):
        super().__init__(tmp_path, **kw)
        (self.root / "net").mkdir()
        (self.root / "net/tcp").write_text(_TCP_HEADER)
        (self.root / "net/tcp6").write_text(_TCP_HEADER)
        self._inode = 40000
        # ``find -exec chown`` runs a program, not the shell function below:
        # this one only logs, as the function does.
        self._fake_chown = self.bin / "fake-chown"
        self._fake_chown.write_text(
            f'#!/bin/sh\necho "$*" >> {shlex.quote(str(self.root.parent / "chown"))}\n'
        )
        self._fake_chown.chmod(0o755)
        # Model a root-owned cgroup-v2 delegation. The signals still use
        # K1's stand-in /proc; only kernel cgroup file semantics are stubbed.
        self.cgroups = tmp_path / "cgroup"
        self.freezer = self.cgroups / "benchflow-kept"
        self.freezer.mkdir(parents=True)
        (self.cgroups / "cgroup.procs").touch()
        (self.freezer / "cgroup.procs").touch()
        (self.freezer / "cgroup.freeze").write_text("1\n")
        (self.freezer / "cgroup.events").write_text("frozen 1\n")
        # The kernel's modes, whatever the umask that ran the tests.
        for path in (self.cgroups, self.freezer):
            path.chmod(0o755)
            for control in path.glob("cgroup.*"):
                control.chmod(0o644)
        self.freezer_overrides = ""
        # The kill runs at the root of the container's own cgroup namespace.
        if (self.root / "self").is_dir():
            (self.root / "self/cgroup").write_text("0::/\n")
            self.mount_cgroup("cgroup2")

    def mount_cgroup(self, fstype):
        """The stand-in's /proc/self/mountinfo, with the cgroup folder
        mounted as ``fstype`` (as Docker mounts it, an optional field and
        all)."""
        (self.root / "self/mountinfo").write_text(
            "498 400 0:60 / / rw,relatime master:1 - overlay overlay rw\n"
            f"512 511 0:30 / {self.cgroups} ro,nosuid shared:5 - {fstype} "
            "cgroup rw,nsdelegate\n"
        )

    def add(self, status, **kwargs):
        pid = super().add(status, **kwargs)
        if status and f"Uid:\t{UID}\t{UID}" in status:
            with (self.freezer / "cgroup.procs").open("a") as f:
                f.write(f"{pid}\n")
        return pid

    def listen(self, port, *, v6=False, state="0A"):
        self._inode += 1
        if v6:
            local = "0" * 32 + f":{port:04X}"
            remote = "0" * 32 + ":0000"
        else:
            local = f"00000000:{port:04X}"
            remote = "00000000:0000"
        line = (
            f"   0: {local} {remote} {state} 00000000:00000000 00:00000000 "
            f"00000000  {UID}        0 {self._inode} 1 0000000000000000 100 0 0 10 0\n"
        )
        with open(self.root / ("net/tcp6" if v6 else "net/tcp"), "a") as f:
            f.write(line)
        return self._inode

    def hold(self, pid, inode, *, fd=3, ptrace_only=False):
        fd_dir = self.root / str(pid) / "fd"
        fd_dir.mkdir(exist_ok=True)
        (fd_dir / str(fd)).symlink_to(f"socket:[{inode}]")
        if ptrace_only:
            (self.root / str(pid) / "ptrace_only").touch()

    def stopped(self, pid):
        return (self.root / str(pid) / "stopped").exists()

    def place(
        self, pid, *, cwd="/srv", fds=(), root="/", ns="1", tasks=None, maps=None
    ):
        """Give a process the links the release reads: per thread (``task/``)
        a working folder, a root folder, open files and its mount and user
        namespaces (the stand-in /proc/self is in namespace 1), and the files
        mapped into its memory (``maps``, paths)."""
        if maps is not None:
            (self.root / str(pid) / "maps").write_text(
                "".join(
                    f"7f0000{i:06x}000-7f0000{i:06x}fff rw-s 00000000 08:01 {900 + i}"
                    f"                    {path}\n"
                    for i, path in enumerate(maps)
                )
            )
        for name in ("mnt", "user"):
            link = self.root / "self/ns" / name
            if not link.is_symlink():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(f"{name}:[1]")
        for tid in tasks or (pid,):
            t = self.root / str(pid) / "task" / str(tid)
            (t / "fd").mkdir(parents=True)
            (t / "ns").mkdir()
            (t / "cwd").symlink_to(cwd)
            (t / "root").symlink_to(root)
            for name in ("mnt", "user"):
                (t / "ns" / name).symlink_to(f"{name}:[{ns}]")
            for i, target in enumerate(("socket:[1]", *fds)):
                (t / "fd" / str(i + 3)).symlink_to(target)

    def chowned(self):
        log = self.root.parent / "chown"
        return log.read_text().splitlines() if log.exists() else []

    def pgrp(self, pid, group, *, ppid=1, session=None, comm="py) S 7 7 (x"):
        """Give a process a ``stat`` line with its parent, process group and
        session (by default the group's number); the default name holds
        ``) `` and digits, as a name can."""
        sid = group if session is None else session
        (self.root / str(pid) / "stat").write_text(
            f"{pid} ({comm}) S {ppid} {group} {sid} 0 -1 4194560 0 0 0 0\n"
        )

    def _prelude(self):
        chown_log = shlex.quote(str(self.root.parent / "chown"))
        # A stop or a continue also shows in the stand-in's State line, as in
        # the kernel's (T while stopped); a zombie's line is left alone.
        base = super()._prelude().replace("_deliver() {", "_deliver0() {", 1)
        return (
            base
            + r"""
_deliver() { _deliver0 "$@"; _show_state "$1" "$2"; }
_show_state() { se="$PROC/$2"; [ -f "$se/status" ] || return 0
  case "$1" in STOP) ns='T (stopped)' ;; CONT) ns='S (sleeping)' ;; *) return 0 ;; esac
  awk -v ns="$ns" 'BEGIN { FS = OFS = "\t" } $1 == "State:" && $2 !~ /^Z/ { $2 = ns } { print }' "$se/status" > "$se/status.new" && command mv "$se/status.new" "$se/status"; }
"""
            + """
setpriv() { [ "$1" = --help ] && { echo "--reuid"; return 0; }; while [ $# -gt 0 ]; do [ "$1" = -c ] && { AS_USER=1; eval "$2"; AS_USER=; return; }; shift; done; }
readlink() { case "$1" in "$PROC"/[0-9]*/*) r=${1#"$PROC"/}; [ -e "$PROC/${r%%/*}/ptrace_only" ] && [ -z "$AS_USER" ] && return 1; [ -e "$PROC/${r%%/*}/ptrace_empty" ] && [ -z "$AS_USER" ] && { case "$1" in */ns/*) echo; return 0;; esac; return 1; };; esac; command readlink "$1"; }
"""
            + f'chown() {{ echo "$*" >> {chown_log}; }}\n'
            + f"CG={shlex.quote(str(self.cgroups))}\n"
            + r"""
cgroup_stat() {
    case "$*" in
        "-c %u %a $CG"*) printf '0 '; command stat -c %a "${3}" ;;
        *) command stat "$@" ;;
    esac
}
stat() { cgroup_stat "$@"; }
freezer_write() {
    printf '%s %s\n' "$1" "$2" >> "$CG/events"
    case "$1" in
        cgroup.procs) echo "$2" >> "$C/$1" ;;
        cgroup.freeze) echo "$2" > "$C/$1"; echo "frozen $2" > "$C/cgroup.events" ;;
    esac
}
freezer_members() {
    for p in $(sort -u "$C/cgroup.procs"); do
        [ -f "$P/$p/status" ] || continue
        grep -q '^State:.*[ZX]' "$P/$p/status" && continue
        echo "$p"
    done
}
"""
            + self.freezer_overrides
        )

    def run(self, command):
        """The stand-in's exec, synchronous, for a MagicMock env's side effect."""
        command = command.replace("/proc", str(self.root)).replace(
            "-exec chown ", f"-exec {self._fake_chown} "
        )
        command = (
            command.replace("/sys/fs/cgroup", str(self.cgroups))
            .replace("freezer_write() {", "kernel_freezer_write() {")
            .replace("freezer_members() {", "kernel_freezer_members() {")
        )
        result = subprocess.run(
            ["/bin/sh", "-c", self._prelude() + command],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin"},
        )
        return MagicMock(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
        )

    async def exec(self, command, **kwargs):
        return self.run(command)


def _env_on(proc):
    """A hardening env whose kills (and releases) run against the stand-in."""
    manifest = _manifest_env(_blank_manifest())

    def side_effect(cmd, **kwargs):
        if "BF killed" in cmd or "BF released" in cmd:
            return proc.run(cmd)
        return manifest(cmd, **kwargs)

    return _make_env(side_effect=side_effect)


def _task(keep):
    task = MagicMock()
    # Guards DT2-CONF-7 at b891c49a: model the pre-agent resolved policy.
    task._benchflow_service_identities = (Identity(int(UID), int(UID)), Identity(0, 0))
    task.config.verifier.env = None
    task.config.verifier.user = None
    task.config.verifier.sandbox_mode = None
    task.config.verifier.sandbox = None
    task.config.verifier.pytest_plugins = []
    task.config.verifier.service = "main"
    task.config.metadata = {"x-keep-services": keep}
    task.task_dir = None
    task.paths.uses_native_verifier_dir = False
    return task


def _logs(tmp_path, verifier_mode=0o777):
    """A stand-in /logs whose verifier folder is open, as the verifier's own
    clear leaves it on a sandbox whose logs are not mounted."""
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True, exist_ok=True)
    logs.chmod(0o755)
    (logs / "verifier").chmod(verifier_mode)
    return logs


class _Async:
    def __init__(self, proc):
        self._proc = proc

    async def exec(self, command, **kwargs):
        return self._proc.run(command)


def _signalled(proc, pid):
    return [s for s in proc.signalled() if s.endswith(f" {pid}")]


@pytest.mark.asyncio
async def test_the_kill_before_grading_keeps_named_listeners(tmp_path):
    """IPv4 and IPv6 listeners on named ports outlive the kill before grading
    and stay frozen; an unnamed port's listener, a child that is not
    listening (a connected socket on the kept port), a plain process and a
    setuid-root holder of a kept socket die; root is never signalled."""
    proc = _NetProc(tmp_path)
    v4 = proc.add(_LIVE, cmdline="python3 app.py --port 5000")
    proc.hold(v4, proc.listen(5000))
    v6 = proc.add(_LIVE, cmdline="pypi-server run -p 8080")
    proc.hold(v6, proc.listen(8080, v6=True))
    unnamed = proc.add(_LIVE, cmdline="python3 -m http.server 9000")
    proc.hold(unnamed, proc.listen(9000))
    child = proc.add(_LIVE, cmdline="curl localhost:5000")
    proc.hold(child, proc.listen(5000, state="01"))
    plain = proc.add(_LIVE, cmdline="sleep 1000")
    setuid = proc.add(_SETUID_ROOT, cmdline="sudo-like")
    proc.hold(setuid, proc.listen(5000))
    root = proc.add(_ROOT, cmdline="root-daemon")
    proc.hold(root, proc.listen(5000))

    task = _task([{"port": 5000, "name": "api"}, {"port": 8080}])
    record: dict = {}
    from benchflow._utils import kill_record

    with kill_record.recording_into(record):
        await harden_before_verify(_env_on(proc), task, "agent", workspace="/app")

    live = proc.live_pids()
    assert str(v4) in live and str(v6) in live
    assert proc.stopped(v4) and proc.stopped(v6)
    for dead in (unnamed, child, plain, setuid):
        assert str(dead) not in live, dead
    assert str(root) in live and _signalled(proc, root) == []
    assert not any(s.startswith("KILL -1") for s in proc.signalled())
    rows = {r["port"]: r for r in record["kept_services"]}
    assert rows[5000]["pid"] == v4 and rows[8080]["pid"] == v6
    assert rows[5000]["command"] == "python3 app.py --port 5000 "
    assert rows[5000]["source"] == "the task"
    assert record["kept_services_source"] == "the task"
    assert record["kept_services_no_listener"] == []
    assert record["killed"] == 4


@pytest.mark.asyncio
async def test_root_that_cannot_read_fd_links_still_keeps(tmp_path):
    """In Docker root cannot read another user's fd links; the agent user's
    own look (while its processes are frozen) finds the listener."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(pid, proc.listen(5000), ptrace_only=True)
    other = proc.add(_LIVE)
    await harden_before_verify(
        _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
    )
    assert str(pid) in proc.live_pids()
    assert str(other) not in proc.live_pids()


@pytest.mark.asyncio
async def test_a_named_port_with_no_listener_keeps_nothing_and_waits(tmp_path):
    """No listener on the named port: every agent process dies, the first
    kill waits for one (up to 30 s, timed by its own clock), and the record
    names the port with no listener."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    task = _task([{"port": 5000}])
    record: dict = {}
    from benchflow._utils import kill_record

    with kill_record.recording_into(record):
        await harden_before_verify(_env_on(proc), task, "agent", workspace="/app")
    assert proc.live_pids() == [] or str(pid) not in proc.live_pids()
    assert record["kept_services"] == []
    assert record["kept_services_no_listener"] == [5000]
    assert record["kept_services_waited_sec"] == 30.0


@pytest.mark.asyncio
async def test_a_task_without_the_key_kills_its_listeners(tmp_path):
    """Without x-keep-services the kill is K1's: a listener dies too."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    task = _task([])
    task.config.metadata = {}
    await harden_before_verify(_env_on(proc), task, "agent", workspace="/app")
    assert str(pid) not in proc.live_pids()


@pytest.mark.asyncio
async def test_the_turn_end_kill_keeps_named_listeners_frozen_for_the_capture(
    tmp_path, monkeypatch
):
    """The kill at the end of the agent's turn (before the evidence capture)
    spares the named listener and leaves it frozen while the submission and
    the evidence are read (K2-R1-A-3); the rest dies."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(pid, proc.listen(5000, v6=True))
    other = proc.add(_LIVE, regen=True)

    async def idle(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", idle)
    task = SimpleNamespace(
        _benchflow_service_identities=(Identity(int(UID), int(UID)), Identity(0, 0)),
        config=SimpleNamespace(
            metadata={"x-keep-services": [{"port": 5000}]},
            artifacts=[],
            verifier=SimpleNamespace(
                submission_files=[], user=None, sandbox_mode=None, sandbox=None
            ),
        ),
    )
    rollout = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=object(),
        _config=SimpleNamespace(purpose="task", sandbox_user="agent"),
        _env=proc,
        _agent_env={},
        _planes=DefaultRolloutPlanes(),
        _task=task,
        _agent_cwd="/app",
        disconnect=idle,
        _require_rollout_dir=lambda: tmp_path,
    )
    frozen_during_capture = []

    async def capture(*_args, **_kwargs):
        frozen_during_capture.append(proc.stopped(pid))

    monkeypatch.setattr(_review, "capture_task_evidence", capture)
    await _review.capture_terminal_workspace(rollout)
    assert str(pid) in proc.live_pids()
    assert frozen_during_capture == [True]
    assert proc.stopped(pid)
    assert str(other) not in proc.live_pids()
    assert proc.hoppers() == []


@pytest.mark.asyncio
async def test_kept_services_go_on_only_with_the_tests_closed(tmp_path):
    """The kept process stays frozen until the verifier's hook; the hook makes
    the verifier's code folder root's alone, then sends SIGCONT."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    proc.place(pid)
    env = _env_on(proc)
    await harden_before_verify(env, _task([{"port": 5000}]), "agent", workspace="/app")
    assert proc.stopped(pid)
    from benchflow.sandbox.lockdown import release_kept_services

    tests_dir = tmp_path / "verifier-code"
    (tests_dir / "sub").mkdir(parents=True)
    (tests_dir / "sub" / "test_x.py").write_text("x")
    (tests_dir / "sub" / "test_x.py").chmod(0o644)

    logs = _logs(tmp_path)
    released = await release_kept_services(
        _Async(proc), "agent", (pid,), (str(tests_dir),), logs_dir=str(logs)
    )
    assert released.released == 1 and released.dropped == ()
    assert not proc.stopped(pid)
    assert (tests_dir / "sub" / "test_x.py").stat().st_mode & 0o077 == 0
    assert tests_dir.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_hardening_closes_logs_and_the_workspace_while_keeping(tmp_path):
    """With kept services /logs becomes root's alone, /logs/verifier loses
    its write for others, and the workspace loses group and other write."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    env = _env_on(proc)
    await harden_before_verify(env, _task([{"port": 5000}]), "agent", workspace="/app")
    cmds = [c.args[0] for c in env.exec.call_args_list]
    assert any("chmod 700 /logs" in c and "chmod 755 /logs/verifier" in c for c in cmds)
    # K2-R2-B-2: what is in /logs/agent and /logs/artifacts becomes root's.
    assert any(
        "for d in /logs/agent /logs/artifacts" in c and "-exec chown root:root" in c
        for c in cmds
    )
    # K2-R3-B-1, K2-R3-B-2: links removed, files replaced by fresh copies.
    assert any(
        "! -type d ! -type f -exec rm -f" in c and "-type f -exec sh -c" in c
        for c in cmds
    )
    close = [c for c in cmds if _close_workspace_cmd("/app") in c]
    assert close
    # Ordering: /logs is closed after it is cleared; the workspace after it
    # is root's.
    clear = next(i for i, c in enumerate(cmds) if "find /logs/verifier" in c)
    logs = next(i for i, c in enumerate(cmds) if "chmod 700 /logs" in c)
    chown = next(
        i for i, c in enumerate(cmds) if "-exec chown -h root:root" in c and "/app" in c
    )
    ws = cmds.index(close[0])
    assert clear < logs and chown < ws


@pytest.mark.asyncio
async def test_without_kept_ports_the_kill_is_k1s_byte_for_byte(tmp_path):
    """Pin K1's plain kill after merging 6f9d482a with K2 at 7b4907c8."""
    from pathlib import Path

    golden = (
        Path(__file__).parent / "fixtures/keep_services/kill-before-grading-k1.sh"
    ).read_text()
    env = _make_env(side_effect=_manifest_env(_blank_manifest()))
    task = _task([])
    task.config.metadata = {}
    await harden_before_verify(env, task, "agent", workspace="/app")
    first = inner_command(env.exec.call_args_list[0].args[0])
    # No PATH prefix of its own any more: the exec route's PATH.
    assert first == golden.rstrip("\n")


@pytest.mark.asyncio
async def test_the_release_closes_logs_again_before_the_services_go_on(tmp_path):
    """K2-R1-A-1: on a sandbox whose logs are not mounted the verifier's clear
    leaves /logs/verifier 0777 after the hardening closed it; the release
    makes /logs root's 0700 and /logs/verifier 0755 again before SIGCONT."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.place(pid)
    logs = _logs(tmp_path)
    out = await release_kept_services(
        _Async(proc), "agent", (pid,), (), logs_dir=str(logs)
    )
    assert out.released == 1 and not proc.stopped(pid)
    assert logs.stat().st_mode & 0o777 == 0o700
    assert (logs / "verifier").stat().st_mode & 0o777 == 0o755
    assert f"root:root {logs}" in proc.chowned()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("links", "why"),
    [
        ({"cwd": "LOGS/verifier"}, "holds a folder or file in"),
        ({"fds": ("LOGS/verifier",)}, "holds a folder or file in"),
        ({"fds": ("LOGS/verifier/sub (deleted)",)}, "holds a folder or file in"),
        ({"root": "/srv/jail"}, "root folder or namespace of its own"),
        ({"ns": "2"}, "root folder or namespace of its own"),
        # K2-R2-B-2: the folders the agent user owns under /logs count too.
        ({"cwd": "LOGS/agent"}, "holds a folder or file in"),
        ({"fds": ("LOGS/artifacts/answer.txt",)}, "holds a folder or file in"),
        ({"cwd": "LOGS"}, "holds a folder or file in"),
        (
            {"maps": ("/usr/lib/libc.so.6", "LOGS/agent/acp_trajectory.jsonl")},
            "holds a folder or file in",
        ),
    ],
)
async def test_a_kept_process_holding_the_verifier_folder_is_killed(
    tmp_path, links, why
):
    """K2-R1-B-1: on Docker /logs/verifier keeps the host user's uid, which
    can be the agent user's, so a kept process that holds the folder (its
    working folder or an open descriptor, or another view of the files) is
    killed at the release and reported, while a clean one goes on. K2-R2-B-2:
    the same for anything under /logs (/logs/agent holds the trajectory
    published for the verifier, /logs/artifacts its answers), a file mapped
    into memory included."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    held = proc.add(_LIVE, cmdline="python3 app.py")
    kw = {
        k: (
            tuple(x.replace("LOGS", str(logs)) for x in v)
            if isinstance(v, tuple)
            else v.replace("LOGS", str(logs))
        )
        for k, v in links.items()
    }
    proc.place(held, **kw)
    clean = proc.add(_LIVE, cmdline="python3 other.py")
    proc.place(clean)
    for p in (held, clean):
        proc.run(f"kill -s STOP {p}")
    out = await release_kept_services(
        _Async(proc), "agent", (held, clean), (), logs_dir=str(logs)
    )
    assert str(held) not in proc.live_pids()
    assert [p for p, _ in out.dropped] == [held]
    assert why in out.dropped[0][1]
    assert out.released == 1
    assert str(clean) in proc.live_pids() and not proc.stopped(clean)


@pytest.mark.asyncio
async def test_a_second_thread_in_the_verifier_folder_is_found(tmp_path):
    """A thread can have a working folder of its own (unshare of CLONE_FS):
    every thread's links are read."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    pid = proc.add(_LIVE)
    proc.place(pid, tasks=(pid,))
    t = proc.root / str(pid) / "task" / str(pid + 1000)
    (t / "fd").mkdir(parents=True)
    (t / "ns").mkdir()
    (t / "cwd").symlink_to(str(logs / "verifier"))
    (t / "root").symlink_to("/")
    for name in ("mnt", "user"):
        (t / "ns" / name).symlink_to(f"{name}:[1]")
    (t / "fd" / "3").symlink_to("socket:[1]")
    out = await release_kept_services(
        _Async(proc), "agent", (pid,), (), logs_dir=str(logs)
    )
    assert out.released == 0 and [p for p, _ in out.dropped] == [pid]


@pytest.mark.asyncio
async def test_links_root_cannot_read_are_read_as_the_user(tmp_path):
    """Root in Docker cannot read another user's links; the agent user's own
    look decides, and a process neither can read is killed (fail closed)."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    readable = proc.add(_LIVE)
    proc.place(readable)
    (proc.root / str(readable) / "ptrace_only").touch()
    unreadable = proc.add(_LIVE)  # no task/ entries: nothing can be read
    out = await release_kept_services(
        _Async(proc), "agent", (readable, unreadable), (), logs_dir=str(logs)
    )
    assert out.released == 1
    assert str(readable) in proc.live_pids()
    assert out.dropped == (
        (unreadable, "its working folder and open files cannot be read"),
    )
    assert str(unreadable) not in proc.live_pids()


@pytest.mark.asyncio
async def test_a_namespace_link_root_reads_as_empty_is_read_as_the_user(tmp_path):
    """Guards the fix of the release on Linux 7.0 (7.0.0-1013-gcp):
    root without CAP_SYS_PTRACE reads another user's /proc/<pid>/ns/* links
    as an empty name with status 0, where 6.x refused them with EACCES. The
    release read that as a namespace of its own and killed every kept
    service (tests/test_identity_switch_respawn_container.py on real
    Docker). An empty read is an unreadable one: the agent user's own look
    decides, and a process in a namespace of its own is still killed."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    shared = proc.add(_LIVE)
    proc.place(shared)
    (proc.root / str(shared) / "ptrace_empty").touch()
    own = proc.add(_LIVE)
    proc.place(own, ns="2")
    (proc.root / str(own) / "ptrace_empty").touch()
    out = await release_kept_services(
        _Async(proc), "agent", (shared, own), (), logs_dir=str(logs)
    )
    assert out.released == 1
    assert str(shared) in proc.live_pids()
    assert out.dropped == ((own, "has a root folder or namespace of its own"),)
    assert str(own) not in proc.live_pids()


@pytest.mark.asyncio
async def test_the_kept_line_is_not_read_from_the_command_line(tmp_path):
    """K2-R1-A-6: the kept process's command line is printed with printf, so
    a backslash sequence in it adds no line to the kill's output."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline=r"server.py \nBF killed 0\nBF kept 22 1 sshd")
    proc.hold(pid, proc.listen(5000))
    record: dict = {}
    from benchflow._utils import kill_record

    with kill_record.recording_into(record):
        await harden_before_verify(
            _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
        )
    assert [r["pid"] for r in record["kept_services"]] == [pid]
    assert record["kept_services"][0]["command"].startswith(r"server.py \nBF")


@pytest.mark.asyncio
async def test_a_clean_kept_process_goes_on_only_after_the_holder_is_gone(tmp_path):
    """K2-R2-B-1: two kept processes, the clean one first (lower pid) and one
    whose working folder is /logs/verifier second. The holder is killed, and
    gone, before the clean one gets SIGCONT: a running process could
    otherwise enter the folder through /proc/<holder>/cwd, past /logs."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    clean = proc.add(_LIVE, cmdline="python3 master.py")
    proc.place(clean)
    held = proc.add(_LIVE, cmdline="python3 worker.py")
    proc.place(held, cwd=str(logs / "verifier"))
    assert clean < held
    for p in (clean, held):
        proc.run(f"kill -s STOP {p}")
    out = await release_kept_services(
        _Async(proc), "agent", (clean, held), (), logs_dir=str(logs)
    )
    assert [p for p, _ in out.dropped] == [held] and out.released == 1
    signals = proc.signalled()
    assert signals.index(f"KILL {held}") < signals.index(f"CONT {clean}"), signals
    assert str(held) not in proc.live_pids() and not proc.stopped(clean)


@pytest.mark.asyncio
async def test_a_dropped_process_that_does_not_exit_fails_the_release(tmp_path):
    """K2-R2-B-1: while a dropped process is still alive nothing goes on; if
    it does not exit, the release fails and every kept process stays
    frozen (the trial is not graded)."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    clean = proc.add(_LIVE)
    proc.place(clean)
    held = proc.add(_LIVE, stubborn=True)
    proc.place(held, cwd=str(logs / "verifier"))
    for p in (clean, held):
        proc.run(f"kill -s STOP {p}")
    with pytest.raises(RuntimeError, match="releasing the kept services"):
        await release_kept_services(
            _Async(proc), "agent", (clean, held), (), logs_dir=str(logs)
        )
    assert proc.stopped(clean)
    assert f"CONT {clean}" not in proc.signalled()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "goes_on"), [(_ZOMBIE, True), (_ZOMBIE_LEADER_LIVE, False)]
)
async def test_a_dropped_zombie_is_gone_unless_a_thread_lives(
    tmp_path, status, goes_on
):
    """A killed process its frozen parent has not reaped is a zombie with no
    working folder: gone. A zombie leader whose other threads still run is
    not."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    clean = proc.add(_LIVE)
    proc.place(clean)
    held = proc.add(status, stubborn=True)
    proc.place(held, cwd=str(logs / "verifier"))
    proc.run(f"kill -s STOP {clean}")
    if goes_on:
        out = await release_kept_services(
            _Async(proc), "agent", (clean, held), (), logs_dir=str(logs)
        )
        assert out.released == 1 and not proc.stopped(clean)
    else:
        with pytest.raises(RuntimeError):
            await release_kept_services(
                _Async(proc), "agent", (clean, held), (), logs_dir=str(logs)
            )
        assert proc.stopped(clean)


@pytest.mark.asyncio
async def test_the_release_closes_what_is_in_logs_agent_and_artifacts(tmp_path):
    """K2-R2-B-2: the agent user owns /logs/agent and /logs/artifacts. Before
    the kept services go on, their folders become root's and every file in
    them a fresh copy, all without group or other write (K2-R3-B-2); the
    two folders lose their write for others only until after grading (on
    Docker the host writes into agent/ then), and /logs at 0700 keeps them
    out of reach by path."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    (logs / "agent").mkdir()
    (logs / "artifacts" / "deep").mkdir(parents=True)
    traj = logs / "agent" / "acp_trajectory.jsonl"
    traj.write_text("{}\n")
    traj.chmod(0o666)
    answer = logs / "artifacts" / "deep" / "answer.txt"
    answer.write_text("42\n")
    answer.chmod(0o664)
    (logs / "artifacts" / "deep").chmod(0o777)
    (logs / "agent").chmod(0o777)
    pid = proc.add(_LIVE)
    proc.place(pid)
    inodes = {path: path.stat().st_ino for path in (traj, answer)}
    out = await release_kept_services(
        _Async(proc), "agent", (pid,), (), logs_dir=str(logs)
    )
    assert out.released == 1
    for path in (traj, answer, logs / "artifacts" / "deep"):
        assert path.stat().st_mode & 0o022 == 0, path
    # K2-R3-A-1: the folder loses its write for others until after grading.
    assert (logs / "agent").stat().st_mode & 0o777 == 0o755
    assert out.opened[0][0] == str(logs / "agent") and out.opened[0][2] == "777"
    chowned = {word for line in proc.chowned() for word in line.split()}
    for path in (logs / "agent", logs / "artifacts", logs / "artifacts" / "deep"):
        assert str(path) in chowned, (path, chowned)
    # K2-R3-B-2: each file is a fresh copy (root's, as root made it).
    for path, inode in inodes.items():
        assert path.stat().st_ino != inode, path
    assert traj.read_text() == "{}\n" and answer.read_text() == "42\n"


@pytest.mark.asyncio
async def test_the_close_leaves_nothing_a_kept_service_holds_or_points_at(tmp_path):
    """K2-R3-B-1: a symbolic link the agent planted in /logs/agent or
    /logs/artifacts into its home is removed, never followed (nor a FIFO
    read), so a kept service that rewrites the target changes nothing the
    verifier reads. K2-R3-B-2: a file there hard-linked into the agent's home
    and held open read-write by the kept service is replaced by a fresh copy,
    so writes through the held descriptor or the hard link land on an inode
    /logs no longer names. Content, mode (without write for others) and
    mtime carry over."""
    import os

    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    home = tmp_path / "home"
    home.mkdir()
    (logs / "agent").mkdir()
    (logs / "artifacts").mkdir()
    (home / "traj").write_text("REAL\n")
    (logs / "agent" / "acp_trajectory.jsonl").symlink_to(home / "traj")
    (home / "linked.txt").write_text("42\n")
    (logs / "artifacts" / "linked.txt").symlink_to(home / "linked.txt")
    os.mkfifo(logs / "artifacts" / "pipe")
    answer = logs / "artifacts" / "answer.txt"
    answer.write_text("42\n")
    answer.chmod(0o664)
    os.utime(answer, (1_000_000_000, 1_000_000_000))
    os.link(answer, home / "h")
    inode = answer.stat().st_ino
    pid = proc.add(_LIVE)
    proc.place(pid)
    with open(home / "h", "r+") as held:
        out = await release_kept_services(
            _Async(proc), "agent", (pid,), (), logs_dir=str(logs)
        )
        assert out.released == 1
        # The link targets in the home were not touched by the close.
        assert (home / "traj").read_text() == "REAL\n"
        assert (home / "linked.txt").read_text() == "42\n"
        # The kept service goes on and rewrites everything it can reach.
        held.write("99\n")
        held.flush()
    (home / "h").write_text("98\n")
    (home / "traj").write_text("FORGED\n")
    (home / "linked.txt").write_text("99\n")
    assert answer.read_text() == "42\n"
    st = answer.stat()
    assert st.st_ino != inode and st.st_nlink == 1
    assert st.st_mode & 0o777 == 0o644 and st.st_mtime == 1_000_000_000
    for gone in (
        "agent/acp_trajectory.jsonl",
        "artifacts/linked.txt",
        "artifacts/pipe",
    ):
        assert not os.path.lexists(logs / gone), gone
    assert sorted(os.listdir(logs / "artifacts")) == ["answer.txt"]
    assert sorted(os.listdir(logs / "agent")) == []


@pytest.mark.asyncio
async def test_with_nothing_else_left_the_kept_get_no_sigcont(tmp_path):
    """K2-R2-A-3: when the turn-end kill already took the rest, the kill
    before grading sends the kept process no SIGCONT (no kill(-1) CONT
    either), so nothing it does lands after the evidence capture and before
    the workspace is closed."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(pid, proc.listen(5000))
    await harden_before_verify(
        _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
    )
    signals = proc.signalled()
    assert f"CONT {pid}" not in signals and "CONT -1" not in signals, signals
    assert proc.stopped(pid)


def _last_before(signals, wanted, end):
    """Index of the last ``wanted`` signal before index ``end``, or -1."""
    hits = [i for i, s in enumerate(signals[:end]) if s == wanted]
    return hits[-1] if hits else -1


@pytest.mark.asyncio
async def test_a_kept_jobs_shell_and_session_die_before_the_grace(tmp_path):
    """K2p-R1-A-1, K2p-R2-A-1: a service that is a job of a terminal's
    interactive shell is in a process group of its own, a child of the shell
    in the shell's session. If the shell dies while a member of that group is
    stopped, the kernel sends the group SIGHUP and SIGCONT; if the terminal
    (a tmux server, its own session) ends in the grace first, the shell is
    hung up and passes SIGHUP on to its jobs. So before the SIGTERM grace the
    kept process and the rest of its group get SIGCONT, then every other
    process of its session (the shell, the helper in its group, another job)
    gets SIGKILL, and only then does anything get SIGTERM; the terminal gets
    the grace. Once the shell is gone nothing holds the group to the session,
    so the round that kills the terminal leaves the kept process frozen, and
    it is frozen when the kill returns."""
    proc = _NetProc(tmp_path)
    term = proc.add(_LIVE, cmdline="tmux new-session -d")
    proc.pgrp(term, term)
    shell = proc.add(_LIVE, cmdline="bash -i")
    proc.pgrp(shell, shell, ppid=term)
    kept = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(kept, proc.listen(5000))
    proc.pgrp(kept, kept, ppid=shell, session=shell)
    helper = proc.add(_LIVE, cmdline="python3 app.py --reloader")
    proc.pgrp(helper, kept, ppid=kept, session=shell)
    other = proc.add(_LIVE, cmdline="sleep 1000")
    proc.pgrp(other, other, ppid=shell, session=shell)
    record: dict = {}
    from benchflow._utils import kill_record

    with kill_record.recording_into(record):
        await harden_before_verify(
            _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
        )
    assert [r["pid"] for r in record["kept_services"]] == [kept]
    live = proc.live_pids()
    assert str(kept) in live and proc.stopped(kept)
    for dead in (term, shell, helper, other):
        assert str(dead) not in live, dead
    signals = proc.signalled()
    first_term = min(i for i, s in enumerate(signals) if s.startswith("TERM "))
    # The grace: the kept process and its group go on, then the rest of the
    # session is killed, all before the first SIGTERM.
    cont = _last_before(signals, f"CONT {kept}", first_term)
    assert cont > _last_before(signals, f"STOP {kept}", first_term), signals
    assert _last_before(signals, f"CONT {helper}", first_term) > _last_before(
        signals, f"STOP {helper}", first_term
    ), signals
    for pid in (shell, helper, other):
        assert _last_before(signals, f"KILL {pid}", first_term) > cont, (pid, signals)
    assert _last_before(signals, f"KILL {term}", first_term) == -1, signals
    assert f"TERM {term}" in signals
    # The round that kills the terminal: the kept process stays frozen.
    kill = signals.index(f"KILL {term}")
    assert _last_before(signals, f"CONT {kept}", kill) < _last_before(
        signals, f"STOP {kept}", kill
    ), signals
    last = max(i for i, s in enumerate(signals) if s.endswith(f" {kept}"))
    assert signals[last] == f"STOP {kept}", signals


@pytest.mark.asyncio
async def test_a_kept_process_with_no_parent_to_lose_stays_frozen(tmp_path):
    """K2p-R2-B-1: a kept master whose group no process still to be ended
    holds to its session (here a ``setsid``'d service, a session of its own,
    whose parent is a shell in another session) stays frozen while the kill
    ends its workers and everything else, so it cannot start new workers as
    fast as the rounds kill them; it gets no SIGCONT at all."""
    proc = _NetProc(tmp_path)
    shell = proc.add(_LIVE, cmdline="bash solve.sh")
    proc.pgrp(shell, shell)
    master = proc.add(_LIVE, cmdline="python3 pool_server.py 5000")
    proc.hold(master, proc.listen(5000))
    proc.pgrp(master, master, ppid=shell, session=master)
    workers = []
    for _ in range(2):
        w = proc.add(_LIVE, cmdline="python3 -c from multiprocessing.spawn")
        proc.pgrp(w, master, ppid=master, session=master)
        workers.append(w)
    sleeper = proc.add(_LIVE, cmdline="sleep 1000")
    proc.pgrp(sleeper, sleeper)
    await harden_before_verify(
        _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
    )
    live = proc.live_pids()
    assert str(master) in live and proc.stopped(master)
    for dead in (shell, sleeper, *workers):
        assert str(dead) not in live, dead
    signals = proc.signalled()
    assert f"CONT {master}" not in signals, signals
    assert "CONT -1" not in signals, signals


@pytest.mark.asyncio
async def test_the_release_names_kept_processes_lost_or_no_longer_frozen(tmp_path):
    """K2p-R1-A-1: at the release a kept process that has exited is listed
    as lost. One that is no longer stopped is not listed as unfrozen any
    more: the frozen cgroup holds it (K1-CONFIRM-1); it is checked and goes
    on like the stopped one."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    frozen = proc.add(_LIVE, cmdline="python3 a.py")
    proc.place(frozen)
    proc.run(f"kill -s STOP {frozen}")
    running = proc.add(_LIVE, cmdline="python3 b.py")
    proc.place(running)
    zombie = proc.add(_ZOMBIE)
    gone = 4999
    out = await release_kept_services(
        _Async(proc), "agent", (frozen, running, zombie, gone), (), logs_dir=str(logs)
    )
    assert out.lost == (zombie, gone)
    assert out.unfrozen == ()
    assert out.released == 2 and out.dropped == ()
    assert not proc.stopped(frozen) and not proc.stopped(running)


@pytest.mark.asyncio
async def test_a_launcher_above_a_kept_process_gets_sigkill_not_sigterm(tmp_path):
    """K2p-R3-A-1: a launcher (``npm start``, ``uv run``) that started the
    kept service and stays its parent passes the SIGTERM it gets in the grace
    on to the frozen service, which then dies when it is let go on. So the
    kept process's parents, as far up as each is to be ended, get SIGKILL
    and never SIGTERM; any other process still gets the grace."""
    proc = _NetProc(tmp_path)
    shell = proc.add(_LIVE, cmdline="sh -c npm start")
    proc.pgrp(shell, shell)
    launcher = proc.add(_LIVE, cmdline="npm start")
    proc.pgrp(launcher, shell, ppid=shell)
    kept = proc.add(_LIVE, cmdline="node server.js")
    proc.hold(kept, proc.listen(5000))
    proc.pgrp(kept, shell, ppid=launcher)
    other = proc.add(_LIVE, cmdline="sleep 1000")
    proc.pgrp(other, other)
    await harden_before_verify(
        _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
    )
    signals = proc.signalled()
    for pid in (launcher, shell):
        assert f"TERM {pid}" not in signals, (pid, signals)
        assert f"KILL {pid}" in signals, (pid, signals)
    assert f"TERM {other}" in signals, signals
    assert f"TERM {kept}" not in signals and f"KILL {kept}" not in signals
    live = proc.live_pids()
    assert str(kept) in live and proc.stopped(kept)
    for dead in (launcher, shell, other):
        assert str(dead) not in live, dead


@pytest.mark.asyncio
async def test_a_kept_process_that_dies_when_let_go_on_is_lost(tmp_path):
    """K2p-R3-A-1: a kept process with a SIGTERM left pending while it was
    frozen dies at the release's SIGCONT; the release names it lost, so the
    0 it scores has a record."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    doomed = proc.add(_LIVE, saver=True, cmdline="node server.js")
    proc.place(doomed)
    proc.run(f"kill -s STOP {doomed}; kill -s TERM {doomed}")
    fine = proc.add(_LIVE, cmdline="python3 app.py")
    proc.place(fine)
    proc.run(f"kill -s STOP {fine}")
    out = await release_kept_services(
        _Async(proc), "agent", (doomed, fine), (), logs_dir=str(logs)
    )
    assert out.lost == (doomed,)
    assert out.released == 2 and out.unfrozen == ()
    assert str(fine) in proc.live_pids() and not proc.stopped(fine)


@pytest.mark.asyncio
async def test_a_kept_process_writing_to_a_terminal_is_named(tmp_path):
    """K2p-R3-A-2: a job typed into a terminal tool's pane with no redirect
    writes to the pane's terminal, which is hung up once the terminal tool
    is gone, so its writes fail and it may answer nothing. The release names
    it with its terminal (read as the agent user where root cannot)."""
    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    on_pty = proc.add(_LIVE, cmdline="python3 -m http.server 5000")
    proc.place(on_pty)
    fd = proc.root / str(on_pty) / "fd"
    fd.mkdir(exist_ok=True)
    # Once the terminal has ended the link reads "/dev/pts/3 (deleted)".
    (fd / "1").symlink_to("/dev/pts/3 (deleted)")
    (proc.root / str(on_pty) / "ptrace_only").touch()
    to_file = proc.add(_LIVE, cmdline="python3 app.py")
    proc.place(to_file)
    out = await release_kept_services(
        _Async(proc), "agent", (on_pty, to_file), (), logs_dir=str(logs)
    )
    assert out.on_terminal == ((on_pty, "/dev/pts/3 (deleted)"),)
    assert out.released == 2


class _AgentOwnsVerifier(_Async):
    """The release's exec where /logs/verifier is owned by the agent's uid
    (Docker, a host user whose uid is the agent user's)."""

    async def exec(self, command, **kwargs):
        stat = (
            'stat() { case "$3" in */verifier) echo "'
            + UID
            + ':1002 755" ;; *) cgroup_stat "$@" ;; esac; }\n'
        )
        return self._proc.run(stat + command)


@pytest.mark.asyncio
async def test_the_release_closes_the_folders_a_descriptor_in_flight_reaches(
    tmp_path,
):
    """K2-R3-A-1: a kept service can hold a descriptor of /logs/agent or
    /logs/verifier in flight on a Unix socket, which no /proc link shows,
    and use it after the release past /logs at 0700. Permission is checked
    at each use, so the release takes group and other write off /logs/agent
    and /logs/artifacts, and makes /logs/verifier root's when its owner is
    the agent user's uid; each folder's owner and mode before is reported so
    they can be put back after grading."""
    import os

    from benchflow.sandbox.lockdown import release_kept_services

    proc = _NetProc(tmp_path)
    logs = _logs(tmp_path, verifier_mode=0o755)
    (logs / "agent").mkdir()
    (logs / "artifacts").mkdir()
    (logs / "agent").chmod(0o777)
    (logs / "artifacts").chmod(0o755)
    pid = proc.add(_LIVE)
    proc.place(pid)
    out = await release_kept_services(
        _AgentOwnsVerifier(proc), "agent", (pid,), (), logs_dir=str(logs)
    )
    assert out.released == 1
    assert (logs / "agent").stat().st_mode & 0o777 == 0o755
    owner = f"{os.getuid()}:{os.getgid()}"
    assert out.opened == (
        (str(logs / "agent"), owner, "777"),
        (str(logs / "verifier"), f"{UID}:1002", "755"),
    )
    assert f"0 {logs / 'verifier'}" in proc.chowned()


@pytest.mark.asyncio
async def test_after_grading_every_agent_process_dies_before_the_folders_reopen():
    """K2-R3-A-1: after the verifier every process of the agent user is
    killed (K1's kill), kept services and their descriptors in flight
    included, and only then do the folders the release closed get their
    owner and mode back; if a process survives they stay closed."""
    from benchflow.sandbox.lockdown import end_kept_services

    calls: list[str] = []

    async def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="BF killed 3 end\n", stderr="", exit_code=0)

    env = SimpleNamespace(exec=receipt_aware(run))
    killed = await end_kept_services(
        env,
        "agent",
        (("/logs/agent", "1001:1002", "777"), ("/logs/verifier", "1001:1002", "755")),
    )
    assert killed == 3 and len(calls) == 2
    assert "BF killed" in calls[0]
    assert calls[1] == (
        "chown 1001:1002 /logs/agent && chmod 777 /logs/agent && "
        "chown 1001:1002 /logs/verifier && chmod 755 /logs/verifier"
    )

    calls.clear()

    async def fail(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="", stderr="writers remain", exit_code=1)

    with pytest.raises(RuntimeError):
        await end_kept_services(
            SimpleNamespace(exec=receipt_aware(fail)),
            "agent",
            (("/logs/agent", "0:0", "777"),),
        )
    assert len(calls) == 1

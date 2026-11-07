"""The kill before grading (K1) and its GH #1078 / PR #1088 regressions.

Before the verifier runs, the sandbox user's leftover processes are stopped so
they cannot write while the trial is graded. A plain per-pid walk can be
outrun by a process that keeps forking or restarts what was just killed, and a
thread group whose leader thread exited (State Z) while its other threads run
was treated as dead. The kill now freezes (SIGSTOP) every live process of the
user, round after round until a round finds none new, then kills (SIGKILL) and
re-checks, and counts a State-Z pid with more than one thread as live
(P1-D8, Rust P1.1-r5-B-1). A pid hopper (a process that starts its next
generation and exits at once) is never listed by a walk of /proc, so each
freeze and kill is first sent with kill(-1) as the sandbox user (K1-R1-A-1).

These exercise :func:`_kill_sandbox_user_procs` through a fake sandbox against
a file-backed stand-in ``/proc``: ``id``, ``sleep``, ``kill``, ``pkill``,
``setpriv`` and ``su`` are shell functions (the last two run their command in
the prelude's shell, where ``kill -SIG -1`` reaches only the stand-in), so the
host's own processes are never signalled and nothing
is destroyed. The stand-in folder is never a real ``/proc`` path. The same
tests run on ``integ/py-0.8`` (its kill ignored a zombie leader and could be
outrun), where the fork-loop, respawner and zombie-leader cases fail.
"""

import os
import re
import shlex
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._runtime_tree import KIT_BIN
from benchflow.sandbox.lockdown import _SAFE_VERIFIER_PATH, _kill_sandbox_user_procs

UID = "1234"

# /proc/<pid>/status bodies.
_LIVE = f"State:\tS (sleeping)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t1\n"
_RUNNING = f"State:\tR (running)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t1\n"
_ZOMBIE = f"State:\tZ (zombie)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t1\n"
# A thread group whose leader thread exited (State Z) while its other threads
# keep running: live despite State Z (P1.1-r5-B-1).
_ZOMBIE_LEADER_LIVE = (
    f"State:\tZ (zombie)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t3\n"
)
_ROOT = "State:\tS (sleeping)\nUid:\t0\t0\t0\t0\nThreads:\t1\n"
_OTHER_UID = "State:\tS (sleeping)\nUid:\t5678\t5678\t5678\t5678\nThreads:\t1\n"


class _FakeProc:
    """A file-backed stand-in /proc the kill script reads and signals.

    Each process is a folder with ``status`` and ``cmdline``. Markers model
    behaviour: ``stubborn`` never dies on SIGKILL; ``regen`` spawns a fresh
    live process whenever it is killed *unless it is stopped first* (a fork
    loop or a self-respawning process — freezing it is what stops the escape).
    A SIGSTOP writes a ``stopped`` marker. ``saver`` saves its state (its pid
    into ``saved``) and exits on SIGTERM, at once or, when stopped, once it
    gets SIGCONT; SIGKILL ends it unsaved. Signals are recorded. A hopper
    lives under ``.hop/``, outside the walk's glob (a walk never lists one),
    and only ``kill -SIG -1`` reaches it: SIGSTOP freezes it into a normal
    entry, SIGKILL or SIGTERM ends it. ``id``, ``sleep``, ``kill``, ``pkill``,
    ``setpriv`` and ``su`` are shell functions, so only this stand-in is ever
    touched.
    """

    def __init__(
        self,
        tmp_path,
        process_table=True,
        has_pkill=True,
        self_status=_ROOT,
        overrides="",
    ):
        self.root = tmp_path / "proc"
        self.root.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.signals = tmp_path / "signals"
        self.nextpid = tmp_path / "nextpid"
        self.nextpid.write_text("500")
        self.saved = tmp_path / "saved"
        self.has_pkill = has_pkill
        self.overrides = overrides
        if process_table:
            (self.root / "self").mkdir()
            (self.root / "self/status").write_text(self_status)
        self._pid = 100

    def add(
        self,
        status,
        *,
        stubborn=False,
        regen=False,
        saver=False,
        reuse=False,
        pdeath=None,
        term_status=None,
        cmdline="sleep",
    ):
        pid = self._pid
        self._pid += 1
        entry = self.root / str(pid)
        if status is None:
            entry.mkdir()
            (entry / "status").mkdir()  # unreadable existing entry
            return pid
        entry.mkdir()
        (entry / "status").write_text(status)
        (entry / "cmdline").write_text(cmdline.replace(" ", "\x00") + "\x00")
        if stubborn:
            (entry / "stubborn").touch()
        if regen:
            (entry / "regen").touch()
        if saver:
            (entry / "saver").touch()
        if reuse:
            (entry / "reuse").touch()
        if pdeath is not None:
            # A child with a parent-death signal: killing this process kills
            # that pid too, and its pid goes to a new root process at once.
            (entry / "pdeath").write_text(str(pdeath))
        if term_status is not None:
            (entry / "term_status").write_text(term_status)
        return pid

    def add_init(self, status, cmdline, *, ptrace_scope=None):
        """PID 1, the PID namespace's init: kill(-1) skips it and it drops
        every signal sent from inside (a stand-in for the kernel's rule).
        *ptrace_scope* is Yama's setting, None for a kernel without Yama."""
        entry = self.root / "1"
        entry.mkdir()
        (entry / "status").write_text(status)
        (entry / "cmdline").write_text(cmdline.replace(" ", "\x00") + "\x00")
        (entry / "init").touch()
        if ptrace_scope is not None:
            yama = self.root / "sys/kernel/yama"
            yama.mkdir(parents=True)
            (yama / "ptrace_scope").write_text(f"{ptrace_scope}\n")

    def add_hopper(self):
        pid = self._pid
        self._pid += 1
        hop = self.root / ".hop"
        hop.mkdir(exist_ok=True)
        (hop / str(pid)).touch()
        return pid

    def hoppers(self):
        hop = self.root / ".hop"
        return sorted(p.name for p in hop.iterdir()) if hop.exists() else []

    def live_pids(self):
        return sorted(p.name for p in self.root.iterdir() if p.name.isdigit())

    def signalled(self):
        if not self.signals.exists():
            return []
        return [ln for ln in self.signals.read_text().split("\n") if ln]

    def _prelude(self):
        P = shlex.quote(str(self.root))
        SIG = shlex.quote(str(self.signals))
        NP = shlex.quote(str(self.nextpid))
        SV = shlex.quote(str(self.saved))
        LIVE = shlex.quote(_LIVE)
        ROOTB = shlex.quote(_ROOT)
        # ``regen`` children fork/respawn too, so a kill that hits them while
        # they are not stopped cannot catch up — only freezing them does.
        pkill = (
            f"""pkill() {{ sig=TERM; for a in "$@"; do case "$a" in -9) sig=KILL;; -STOP) sig=STOP;; esac; done
  for e in "$PROC"/[0-9]*; do p=${{e##*/}}; u=$(awk '/^Uid:/{{print $2}}' "$e/status" 2>/dev/null); [ "$u" = {UID} ] && _deliver "$sig" "$p"; done; }}"""
            if self.has_pkill
            else "pkill() { :; }"
        )
        return f"""
PROC={P}; SIGLOG={SIG}; NEXTPID={NP}; LIVEBODY={LIVE}; SAVED={SV}; ROOTBODY={ROOTB}
id() {{ case "$2" in root) echo 0;; *) echo {UID};; esac; }}
sleep() {{ :; }}
_newpid() {{ np=$(cat "$NEXTPID"); echo $((np + 1)) > "$NEXTPID"; echo "$np"; }}
_spawn() {{ np=$(_newpid); d="$PROC/$np"; mkdir -p "$d"; printf '%s' "$LIVEBODY" > "$d/status"; printf 'child\\000' > "$d/cmdline"; touch "$d/regen"; }}
_everyone() {{ bsig=$1; echo "$bsig -1" >> "$SIGLOG"
  for h in "$PROC"/.hop/*; do [ -e "$h" ] || continue; hp=${{h##*/}}
    case "$bsig" in
      STOP) rm -f "$h"; mkdir -p "$PROC/$hp"; printf '%s' "$LIVEBODY" > "$PROC/$hp/status"; printf 'hop\\000' > "$PROC/$hp/cmdline"; touch "$PROC/$hp/stopped" ;;
      KILL|TERM) rm -f "$h" ;;
    esac
  done
  for be in "$PROC"/[0-9]*; do bp=${{be##*/}}
    [ -e "$be/init" ] && continue
    bu=$(awk '/^Uid:/{{print $2}}' "$be/status" 2>/dev/null); [ "$bu" = {UID} ] || continue
    bs=$(awk '/^State:/{{print $2}}' "$be/status" 2>/dev/null); bt=$(awk '/^Threads:/{{print $2}}' "$be/status" 2>/dev/null)
    [ "$bs" = Z ] && [ "$bt" = 1 ] && continue
    _deliver "$bsig" "$bp" quiet
  done; }}
setpriv() {{ [ "$1" = --help ] && {{ echo "--reuid"; return 0; }}; while [ $# -gt 0 ]; do [ "$1" = -c ] && {{ AS_USER=1; eval "$2"; return; }}; shift; done; }}
su() {{ while [ $# -gt 0 ]; do [ "$1" = -c ] && {{ AS_USER=1; eval "$2"; return; }}; shift; done; }}
_deliver() {{ sig=$1; pid=$2; quiet=$3
  # The helper's temporary broadcast recipient is outside the stand-in.
  [ -n "$probe" ] && [ "$pid" = "$probe" ] && return 0
  if [ "$pid" = -1 ]; then
    if [ -n "$KILL_ALL_ERR" ] && {{ [ -z "$KILL_ALL_ERR_SIG" ] || [ "$sig" = "$KILL_ALL_ERR_SIG" ]; }}; then
      echo "kill: $KILL_ALL_ERR" >&2; return 1
    fi
    _everyone "$sig"; return 0
  fi
  e="$PROC/$pid"
  # The PID namespace's init drops every signal sent from inside it.
  [ -e "$e/init" ] && {{ echo "$sig $pid dropped" >> "$SIGLOG"; return 0; }}
  # As the user, only a process whose real or saved UID is the user's.
  if [ -n "$AS_USER" ] && [ -d "$e" ]; then
    set -- $(awk '/^Uid:/{{print $2, $4}}' "$e/status" 2>/dev/null)
    [ "$1" = {UID} ] || [ "$2" = {UID} ] || {{ echo "EPERM $sig $pid" >> "$SIGLOG"; return 1; }}
  fi
  [ -n "$quiet" ] || echo "$sig $pid" >> "$SIGLOG"; [ -d "$e" ] || return 0
  case "$sig" in
    STOP) touch "$e/stopped" ;;
    TERM)
      [ -e "$e/term_status" ] && touch "$e/uidpend"
      if [ -e "$e/saver" ]; then
        if [ -e "$e/stopped" ]; then touch "$e/termpend"; else echo "$pid" >> "$SAVED"; rm -rf "$e"; fi
      fi ;;
    CONT)
      rm -f "$e/stopped"
      [ -e "$e/uidpend" ] && cat "$e/term_status" > "$e/status"
      [ -e "$e/termpend" ] && {{ echo "$pid" >> "$SAVED"; rm -rf "$e"; }} ;;
    KILL)
      [ -e "$e/regen" ] && [ ! -e "$e/stopped" ] && _spawn
      [ -e "$e/stubborn" ] && return 0
      reuse=; [ -e "$e/reuse" ] && reuse=1
      if [ -e "$e/pdeath" ]; then c="$PROC/$(cat "$e/pdeath")"; rm -rf "$c"; mkdir "$c"; printf "%s" "$ROOTBODY" > "$c/status"; printf "rootd\\000" > "$c/cmdline"; fi
      rm -rf "$e"
      [ -n "$reuse" ] && {{ mkdir "$e"; printf "%s" "$ROOTBODY" > "$e/status"; printf "rootd\\000" > "$e/cmdline"; }} ;;
    *) : ;;
  esac
}}
kill() {{ case "$1" in -s) sig=$2; pid=$3;; -*) sig=${{1#-}}; pid=$2;; *) sig=$1; pid=$2;; esac
  case "$sig" in 9) sig=KILL;; 15) sig=TERM;; 19) sig=STOP;; 18) sig=CONT;; esac
  _deliver "$sig" "$pid"; }}
{pkill}
{self.overrides}
"""

    async def exec(self, command, **kwargs):
        # Point every /proc read at the stand-in, so the real process table is
        # never touched (works for both the base and the branch commands).
        # id/sleep/kill/pkill are shell functions in the prelude, so they
        # shadow the real tools; the real PATH stays for rm/cat/awk/printf.
        command = command.replace("/proc", str(self.root))
        result = subprocess.run(
            ["/bin/sh", "-c", self._prelude() + command],
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin"},
        )
        return SimpleNamespace(
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


@pytest.mark.asyncio
async def test_a_simple_live_process_is_killed(tmp_path):
    proc = _FakeProc(tmp_path)
    proc.add(_LIVE)
    await _kill_sandbox_user_procs(proc, "agent")
    assert proc.live_pids() == []


@pytest.mark.asyncio
async def test_a_fork_loop_ends_dead(tmp_path):
    """A process that keeps forking is frozen first, so no child escapes."""
    proc = _FakeProc(tmp_path)
    proc.add(_RUNNING, regen=True)
    await _kill_sandbox_user_procs(proc, "agent")
    assert proc.live_pids() == []


@pytest.mark.asyncio
async def test_a_self_respawning_process_ends_dead(tmp_path):
    """A process that restarts itself on death is gone once it is frozen."""
    proc = _FakeProc(tmp_path)
    proc.add(_LIVE, regen=True)
    await _kill_sandbox_user_procs(proc, "agent")
    assert proc.live_pids() == []


@pytest.mark.asyncio
async def test_a_pid_hopper_ends_dead(tmp_path):
    """A process that starts its next generation and exits at once is never
    listed by a walk of /proc; kill(-1) as the sandbox user freezes it in one
    pass, then the walk sees it, kills it and counts it (K1-R1-A-1)."""
    proc = _FakeProc(tmp_path)
    proc.add_hopper()
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert proc.hoppers() == []
    assert proc.live_pids() == []
    assert killed == 1


@pytest.mark.asyncio
async def test_a_process_that_saves_on_sigterm_still_saves(tmp_path):
    """The SIGTERM grace stays (K1-R1-A-2): a leftover that writes its state
    on SIGTERM (a trap, a database saving on shutdown) still writes it before
    grading, as with the 0.8 kill, and then is gone."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_LIVE, saver=True)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert proc.saved.read_text().split() == [str(pid)]
    assert proc.live_pids() == []
    assert killed == 1


@pytest.mark.asyncio
async def test_a_zombie_leader_with_live_threads_ends_dead(tmp_path):
    """State Z with Threads > 1 is a live thread group, not a dead process.

    On a slim image with no ``pkill`` the base relied on its quiescence probe,
    whose ``live()`` ignored State Z, so it declared quiescence while the
    thread group ran; the branch's kill counts it live and kills it."""
    proc = _FakeProc(tmp_path, has_pkill=False)
    pid = proc.add(_ZOMBIE_LEADER_LIVE)
    await _kill_sandbox_user_procs(proc, "agent")
    assert str(pid) not in proc.live_pids()
    assert any(s.endswith(f" {pid}") for s in proc.signalled())


@pytest.mark.asyncio
async def test_a_plain_zombie_is_not_signalled(tmp_path):
    """A zombie with one thread has no live writer: untouched, no signal."""
    proc = _FakeProc(tmp_path, has_pkill=False)
    pid = proc.add(_ZOMBIE)
    await _kill_sandbox_user_procs(proc, "agent")
    assert not any(s.endswith(f" {pid}") for s in proc.signalled())
    assert str(pid) in proc.live_pids()


@pytest.mark.asyncio
async def test_root_and_other_users_are_never_signalled(tmp_path):
    """Only the sandbox user's effective UID matches; root is untouched."""
    proc = _FakeProc(tmp_path)
    root_pid = proc.add(_ROOT)
    other_pid = proc.add(_OTHER_UID)
    proc.add(_LIVE)
    await _kill_sandbox_user_procs(proc, "agent")
    assert all(
        not s.endswith(f" {root_pid}") and not s.endswith(f" {other_pid}")
        for s in proc.signalled()
    )
    assert str(root_pid) in proc.live_pids()
    assert str(other_pid) in proc.live_pids()


@pytest.mark.asyncio
async def test_a_survivor_fails_the_check(tmp_path):
    """A process that never dies makes the kill raise, not pass silently."""
    proc = _FakeProc(tmp_path)
    proc.add(_LIVE, stubborn=True, cmdline="stubborn --x")
    with pytest.raises(RuntimeError):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_missing_process_table_is_not_quiescence(tmp_path):
    """GH #1078: no readable /proc is an unknown state, not an empty one."""
    proc = _FakeProc(tmp_path, process_table=False)
    proc.add(_LIVE, stubborn=True)
    with pytest.raises(RuntimeError):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_identity_check_rejects_root_before_any_signal(tmp_path):
    """Even a root alias must not have its processes killed before grading."""
    proc = _FakeProc(tmp_path)
    proc.add(_ROOT)
    with pytest.raises(RuntimeError, match="identity check"):
        await _kill_sandbox_user_procs(proc, "root")
    assert proc.signalled() == []


def test_kill_uses_no_procps_or_python():
    """busybox/slim images ship neither procps nor python3: the script reads
    /proc with shell builtins, so it must name none of them."""
    from benchflow.sandbox.lockdown import _KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE

    script = _KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE
    for tool in ("python3", "pgrep", "pkill", "ps"):
        assert not re.search(rf"(^|[\s;|(&]){tool}\s", script), tool


@pytest.mark.skipif(
    not os.path.exists("/proc/self/status") or os.geteuid() == 0,
    reason="needs Linux /proc and a non-root test user",
)
def test_real_process_is_killed_through_real_proc(tmp_path):
    """The real script against real kernel status files kills a real child of
    this (non-root) test user. No fakes: run only on Linux as non-root."""
    from benchflow.sandbox.lockdown import _KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE

    child = subprocess.Popen(["sleep", "60"])
    try:
        proc = tmp_path / "proc"
        proc.mkdir()
        (proc / "self").symlink_to("/proc/self", target_is_directory=True)
        (proc / str(child.pid)).symlink_to(
            f"/proc/{child.pid}", target_is_directory=True
        )
        script = (
            _KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE.replace("__USER__", "agent")
            .replace("__PROC__", shlex.quote(str(proc)))
            .replace("__NEED_ROOT__", "0")
        )
        result = subprocess.run(
            ["/bin/sh", "-c", f"id() {{ echo {os.getuid()}; }}\n" + script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        # ``sleep`` ends on the grace's SIGTERM, before any SIGKILL.
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


# A broadcast counts only when the helper really ran as the user.
_SETPRIV_FAILS = (
    'setpriv() { [ "$1" = --help ] && { echo "--reuid"; return 0; }; return 1; }'
)
_SU_FAILS = "su() { return 1; }"
# A helper that runs the command but never drops: it is still root inside.
_SETPRIV_NO_DROP = (
    'setpriv() { [ "$1" = --help ] && { echo "--reuid"; return 0; }; '
    'while [ $# -gt 0 ]; do [ "$1" = -c ] && { id() { echo 0; }; eval "$2"; return; }; '
    "shift; done; }"
)
_SU_NO_DROP = (
    'su() { while [ $# -gt 0 ]; do [ "$1" = -c ] && { id() { echo 0; }; eval "$2"; '
    "return; }; shift; done; }"
)


@pytest.mark.asyncio
async def test_no_working_broadcast_fails_hardening(tmp_path):
    """setpriv advertises --reuid but cannot drop and su is refused: no
    kill(-1) is sent, so a hopper may still run; the kill must fail, not
    report a confirmed kill."""
    proc = _FakeProc(tmp_path, overrides=f"{_SETPRIV_FAILS}\n{_SU_FAILS}")
    proc.add_hopper()
    with pytest.raises(RuntimeError, match="quiescence check"):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_a_helper_that_does_not_drop_is_not_trusted(tmp_path):
    """Both helpers run the command as root instead of the user: that is not
    a broadcast as the user, so the kill fails."""
    proc = _FakeProc(tmp_path, overrides=f"{_SETPRIV_NO_DROP}\n{_SU_NO_DROP}")
    proc.add(_LIVE)
    with pytest.raises(RuntimeError, match="quiescence check"):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_su_is_tried_when_setpriv_fails(tmp_path):
    """A setpriv that cannot drop falls through to su, whose broadcast
    freezes the hopper, so it is killed and counted."""
    proc = _FakeProc(tmp_path, overrides=_SETPRIV_FAILS)
    proc.add_hopper()
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert proc.hoppers() == []
    assert proc.live_pids() == []
    assert killed == 1


@pytest.mark.asyncio
async def test_the_kill_refuses_to_run_as_a_user_other_than_root(tmp_path):
    """Run as anyone but root the kill can send no kill(-1), so a hopper
    would escape: it fails instead of trusting the walk."""
    proc = _FakeProc(tmp_path, self_status=_OTHER_UID)
    proc.add(_LIVE)
    with pytest.raises(RuntimeError, match="quiescence check"):
        await _kill_sandbox_user_procs(proc, "agent")


# Per-pid signals never follow a broadcast that can free the pid.


@pytest.mark.asyncio
async def test_a_pid_reused_by_root_after_the_kill_is_not_signalled(tmp_path):
    """When the agent process dies its pid goes to a new root process (a
    forking daemon near pid wraparound). The per-pid SIGKILL is sent while the
    agent process is still frozen, before kill(-1), so the root process that
    takes the pid afterwards is never signalled."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_LIVE, reuse=True)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert killed == 1
    # The pid now belongs to root, and root's process is still there.
    assert str(pid) in proc.live_pids()
    assert (proc.root / str(pid) / "status").read_text() == _ROOT


@pytest.mark.asyncio
async def test_per_pid_signals_come_before_their_broadcast(tmp_path):
    """The frozen set gets its SIGCONT and SIGKILL by pid before kill(-1) of
    the same signal."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_LIVE)
    await _kill_sandbox_user_procs(proc, "agent")
    log = proc.signalled()
    for sig in ("CONT", "KILL"):
        assert log.index(f"{sig} {pid}") < log.index(f"{sig} -1"), log


@pytest.mark.asyncio
async def test_killing_one_frozen_process_never_kills_root_on_anothers_pid(tmp_path):
    """Two frozen agent processes, a parent and a child that set a
    parent-death signal. Killing the parent kills the child too, PID 1 reaps
    it and a root process takes its pid before the child's turn: root's
    per-pid SIGKILL then killed root's process. The user's own processes get
    their per-pid signals as the user, so that pid answers EPERM."""
    proc = _FakeProc(tmp_path)
    child = proc._pid + 1
    proc.add(_LIVE, pdeath=child)
    assert proc.add(_LIVE) == child
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert killed == 2
    # The child's pid is root's now, and root's process is still there.
    assert (proc.root / str(child) / "status").read_text() == _ROOT
    assert f"KILL {child}" not in proc.signalled()


# A broadcast that the kernel refused is not a broadcast.


@pytest.mark.asyncio
async def test_a_refused_broadcast_fails_hardening(tmp_path):
    """The helper drops to the user, but kill(-1) fails (a seccomp policy
    answers EPERM): the hopper it was for is never frozen, so the kill fails
    instead of reporting 'BF killed 0'."""
    proc = _FakeProc(tmp_path, overrides="KILL_ALL_ERR='Operation not permitted'")
    proc.add_hopper()
    with pytest.raises(RuntimeError, match="quiescence check"):
        await _kill_sandbox_user_procs(proc, "agent")
    assert proc.hoppers() != []


@pytest.mark.asyncio
async def test_a_broadcast_that_finds_no_process_is_fine(tmp_path):
    """An empty walk still permits a real broadcast.

    The helper supplies its own recipient, so ESRCH is never needed as
    success evidence, even with no agent processes left.
    """
    proc = _FakeProc(tmp_path)
    assert await _kill_sandbox_user_procs(proc, "agent") == 0
    assert proc.live_pids() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["No such process", "Operation not permitted"])
@pytest.mark.parametrize("sig", ["STOP", "CONT", "KILL"])
@pytest.mark.parametrize("helper", ["setpriv", "su", "both"])
async def test_denied_broadcast_never_confirms_delivery(tmp_path, error, sig, helper):
    """No errno proves delivery.

    A policy can fabricate ESRCH for either helper, in every broadcast
    phase, even with a writer that a walk cannot find.
    """
    unavailable = {"setpriv": _SU_FAILS, "su": _SETPRIV_FAILS, "both": ""}
    proc = _FakeProc(
        tmp_path,
        overrides=f"KILL_ALL_ERR={shlex.quote(error)}\nKILL_ALL_ERR_SIG={sig}\n"
        + unavailable[helper],
    )
    proc.add_hopper()
    with pytest.raises(RuntimeError, match=f"Cannot send SIG{sig}"):
        await _kill_sandbox_user_procs(proc, "agent")
    if sig == "STOP":
        assert proc.hoppers() != []


@pytest.mark.asyncio
async def test_su_retries_a_broadcast_denied_with_esrch(tmp_path):
    """Refused setpriv delivery must try su."""
    proc = _FakeProc(
        tmp_path,
        overrides='setpriv() { echo "kill: No such process"; echo BF kill-failed; }',
    )
    proc.add_hopper()
    assert await _kill_sandbox_user_procs(proc, "agent") == 1
    assert proc.hoppers() == []
    assert proc.live_pids() == []


# One target set, matched on the real, effective or saved UID.

# A setuid-root program the agent started: real UID the agent's, effective and
# saved UID root's.
_SETUID_HELPER = f"State:\tS (sleeping)\nUid:\t{UID}\t0\t0\t0\nThreads:\t1\n"
# A process root started with only its effective UID set to the agent's.
_EUID_ONLY = f"State:\tS (sleeping)\nUid:\t0\t{UID}\t0\t0\nThreads:\t1\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial,final",
    [(_EUID_ONLY, _LIVE), (_SETUID_HELPER, _EUID_ONLY)],
    ids=["root-to-user", "user-to-root"],
)
async def test_uid_classification_change_counts_the_process_once(
    tmp_path, initial, final
):
    """Signal tags are not identities.

    One process changes its real/saved UIDs during the SIGTERM grace and
    survives until SIGKILL. Both changes of signaling role count once.
    """
    proc = _FakeProc(tmp_path)
    pid = proc.add(initial, term_status=final)
    proc.add(_ROOT)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert killed == 1
    assert str(pid) not in proc.live_pids()
    assert f"KILL {pid}" in proc.signalled()


@pytest.mark.asyncio
async def test_a_setuid_program_the_agent_started_is_counted(tmp_path):
    """kill(-1) as the agent reaches a setuid program the agent started; the
    walk counts it too, so the record says one was killed."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_SETUID_HELPER)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert str(pid) not in proc.live_pids()
    assert killed == 1


@pytest.mark.asyncio
async def test_a_setuid_program_that_survives_fails_the_check(tmp_path):
    """A setuid program the agent started that does not die (stuck in
    uninterruptible I/O) is checked like any other target: the kill fails,
    naming it."""
    proc = _FakeProc(tmp_path)
    proc.add(_SETUID_HELPER, stubborn=True, cmdline="passwd")
    with pytest.raises(RuntimeError, match="passwd"):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_an_effective_uid_match_is_still_killed(tmp_path):
    """A process whose effective UID alone is the agent's (kill(-1) as the
    agent cannot reach it) is frozen and killed by pid, as ``pgrep -u`` would
    list it; root's own processes stay untouched."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_EUID_ONLY)
    root_pid = proc.add(_ROOT)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert str(pid) not in proc.live_pids()
    assert str(root_pid) in proc.live_pids()
    assert not any(s.endswith(f" {root_pid}") for s in proc.signalled())
    assert killed == 1


# Missing or malformed evidence is never a confirmed zero.


class _Canned:
    """A sandbox whose exec returns one fixed result, as its command's answer
    inside the receipt lockdown reads (tests/_receipt_fakes.py)."""

    def __init__(self, result):
        self.result = result

    async def exec(self, command, **kwargs):
        from tests._receipt_fakes import receipt_answer

        # A result without an integer status stands for missing evidence:
        # no receipt comes back, so the read fails.
        code = getattr(self.result, "return_code", None)
        if not isinstance(code, int) or isinstance(code, bool):
            return self.result
        return receipt_answer(command, self.result)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=0, stdout=None, stderr=""),
        SimpleNamespace(
            return_code=0, stdout="BF killed 2 end\nBF killed 0 end\n", stderr=""
        ),
        SimpleNamespace(return_code=0, stdout="BF killed x\n", stderr=""),
        # Cut short: "BF killed 12 end" read as 1.
        SimpleNamespace(return_code=0, stdout="BF killed 1", stderr=""),
        SimpleNamespace(return_code=0, stdout="BF killed 1\n", stderr=""),
        SimpleNamespace(
            return_code=0, stdout="BF killed 12 end\nBF killed 3", stderr=""
        ),
        SimpleNamespace(stdout="BF killed 0 end\n", stderr=""),
        SimpleNamespace(return_code=None, stdout="BF killed 0 end\n", stderr=""),
        SimpleNamespace(return_code="0", stdout="BF killed 0 end\n", stderr=""),
        None,
    ],
    ids=[
        "no-marker",
        "no-stdout",
        "two-markers",
        "bad-count",
        "cut-short",
        "no-end-word",
        "extra-marker",
        "no-status",
        "none-status",
        "str-status",
        "no-result",
    ],
)
@pytest.mark.asyncio
async def test_missing_kill_evidence_fails_hardening(tmp_path, monkeypatch, result):
    from benchflow._utils import kill_record

    recorded = []
    monkeypatch.setattr(
        kill_record,
        "record_kill_before_grading",
        lambda user, killed: recorded.append((user, killed)),
    )
    with pytest.raises(RuntimeError, match="Verifier hardening failed"):
        await _kill_sandbox_user_procs(_Canned(result), "agent")
    assert recorded == []


@pytest.mark.asyncio
async def test_a_confirmed_count_is_recorded(monkeypatch):
    from benchflow._utils import kill_record

    recorded = []
    monkeypatch.setattr(
        kill_record,
        "record_kill_before_grading",
        lambda user, killed: recorded.append((user, killed)),
    )
    result = SimpleNamespace(return_code=0, stdout="BF killed 3 end\n", stderr="")
    assert await _kill_sandbox_user_procs(_Canned(result), "agent") == 3
    assert recorded == [("agent", 3)]


# K1-R1-A-1: the quiesce before a workspace capture runs the same kill first,
# so the hardening kill usually finds nothing left. Its count must not hide
# the quiesce's.


def test_a_later_zero_never_replaces_an_earlier_count():
    from benchflow._utils import kill_record

    record: dict = {}
    with kill_record.recording_into(record):
        kill_record.record_kill_before_grading("agent", 3)
        kill_record.record_kill_before_grading("agent", 0)
    assert record == {"user": "agent", "killed": 0, "killed_total": 3}


def _capture_rollout(rollout_dir, quiesce_count: int):
    async def idle(*_args):
        return None

    async def quiesce(env, user):
        result = SimpleNamespace(
            return_code=0, stdout=f"BF killed {quiesce_count} end\n", stderr=""
        )
        await _kill_sandbox_user_procs(_Canned(result), user)

    return SimpleNamespace(
        _branch_child_active=False,
        _review_plan=None,
        _config=SimpleNamespace(
            purpose="reviewer", sandbox_user="agent", freeze_workspace=False
        ),
        _env=None,
        _agent_env={},
        _agent_cwd="/app",
        _planes=SimpleNamespace(quiesce_agent=quiesce),
        _task=SimpleNamespace(
            config=SimpleNamespace(
                artifacts=[], verifier=SimpleNamespace(submission_files=[])
            )
        ),
        disconnect=idle,
        _require_rollout_dir=lambda: rollout_dir,
    )


async def test_the_quiesce_before_a_capture_is_recorded(tmp_path, monkeypatch):
    import json

    from benchflow.rollout import _review

    async def no_evidence(*args, **kwargs):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", no_evidence)
    rollout = _capture_rollout(tmp_path, quiesce_count=4)
    await _review.capture_terminal_workspace(rollout)
    assert getattr(rollout, "_export_error", None) is None
    written = json.loads((tmp_path / "kill-before-grading.json").read_text())
    assert written == {
        "user": "agent",
        "killed": 4,
        "killed_total": 4,
        "quiesce_killed": 4,
    }


async def test_the_hardening_kill_adds_to_the_quiesce_count(tmp_path, monkeypatch):
    import json

    from benchflow.rollout import _review
    from benchflow.rollout._setup import _verify_rollout

    async def no_evidence(*args, **kwargs):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", no_evidence)
    rollout = _capture_rollout(tmp_path, quiesce_count=4)
    await _review.capture_terminal_workspace(rollout)

    async def harden(env, task, user, **kwargs):
        # The agent's processes are already dead: this kill finds none.
        result = SimpleNamespace(return_code=0, stdout="BF killed 0 end\n", stderr="")
        await _kill_sandbox_user_procs(_Canned(result), user)

    class _Verifier:
        execution_receipt = None

        async def verify(self):
            return SimpleNamespace(rewards={"reward": 1.0})

    planes = SimpleNamespace(
        harden_before_verify=harden, verifier=lambda **kwargs: _Verifier()
    )
    task = SimpleNamespace(
        name="t",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=60, reward_range=None, env={})
        ),
    )
    paths = SimpleNamespace(verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path)
    rewards, error, _ = await _verify_rollout(
        _Canned(SimpleNamespace(return_code=0, stdout="started", stderr="")),
        task,
        paths,
        {},
        planes,
        sandbox_user="agent",
        kill_record=rollout._kill_record,
    )
    assert (rewards, error) == ({"reward": 1.0}, None)
    written = json.loads((tmp_path / "kill-before-grading.json").read_text())
    assert written == {
        "user": "agent",
        "killed": 0,
        "killed_total": 4,
        "quiesce_killed": 4,
    }


def _verifying_rollout(tmp_path, monkeypatch, *, hardening_count: int):
    """A real ``Rollout`` whose verify runs the real ``_verify_rollout``.

    Only the sandbox is a stand-in: the hardening kill reads ``BF killed
    <hardening_count>`` and the verifier returns reward 1.
    """
    from datetime import datetime

    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task import RolloutPaths, Task

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Produce a file.")
    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=rollout._rollout_dir)
    rollout._env = _Canned(SimpleNamespace(return_code=0, stdout="started", stderr=""))
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]

    async def harden(env, task, user, **kwargs):
        result = SimpleNamespace(
            return_code=0, stdout=f"BF killed {hardening_count} end\n", stderr=""
        )
        await _kill_sandbox_user_procs(_Canned(result), user)

    class _Verifier:
        execution_receipt = None

        async def verify(self):
            return SimpleNamespace(rewards={"reward": 1.0})

    rollout._planes = SimpleNamespace(
        harden_before_verify=harden, verifier=lambda **kwargs: _Verifier()
    )

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("benchflow.rollout.collect_rollout_artifacts", noop)
    monkeypatch.setattr("benchflow.rollout._publish_trajectory_for_verifier", noop)
    return rollout


async def test_rollout_verify_hands_the_quiesce_count_to_the_hardening_kill(
    tmp_path, monkeypatch
):
    """``Rollout.verify`` carries the quiesce's count into the hardening record.

    The quiesce before a capture kills 4 and the hardening kill finds none, so
    the trial's record must still read 4 (K1-R2-A-2): dropping the hand-off in
    ``Rollout.verify`` would lose ``killed_total: 4``. The latest kill
    still has its own ``killed: 0`` count (K2, merge with 7b4907c8).
    """
    import json

    from benchflow.rollout import _review

    rollout = _verifying_rollout(tmp_path, monkeypatch, hardening_count=0)

    async def quiesce(env, user):
        result = SimpleNamespace(return_code=0, stdout="BF killed 4 end\n", stderr="")
        await _kill_sandbox_user_procs(_Canned(result), user)

    rollout._planes.quiesce_agent = quiesce

    async def capture(rollout):
        await _review._quiesce_and_record(rollout, "agent")

    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", capture)

    assert await rollout.verify() == {"reward": 1.0}
    written = json.loads((tmp_path / "trial" / "kill-before-grading.json").read_text())
    assert written == {
        "user": "agent",
        "killed": 0,
        "killed_total": 4,
        "quiesce_killed": 4,
    }


async def test_a_branch_childs_kill_record_lands_in_its_own_folder(
    tmp_path, monkeypatch
):
    """An in-place branch child's kill record goes to the child's folder.

    The branch points ``_rollout_dir`` (which host writers follow) at the
    child's folder and leaves ``_rollout_paths`` on the parent's root. The
    hardening kill wrote to the latter, so no child kept a record and the
    parent's was replaced by the last child's (K1-R2-A-1).
    """
    import json

    from benchflow.rollout import _review

    rollout = _verifying_rollout(tmp_path, monkeypatch, hardening_count=2)
    parent = rollout._rollout_dir
    parent_record = {"user": "agent", "killed": 5}
    (parent / "kill-before-grading.json").write_text(json.dumps(parent_record))
    child_dir = parent / "branches" / "f1" / "children" / "c1"
    child_dir.mkdir(parents=True)
    rollout._rollout_dir = child_dir
    rollout._branch_child_active = True
    # The real capture returns at once for a branch child (no quiesce).
    monkeypatch.setattr(
        "benchflow.rollout.capture_terminal_workspace",
        _review.capture_terminal_workspace,
    )

    assert await rollout.verify() == {"reward": 1.0}
    written = json.loads((child_dir / "kill-before-grading.json").read_text())
    assert written == {"user": "agent", "killed": 2, "killed_total": 2}
    assert json.loads((parent / "kill-before-grading.json").read_text()) == (
        parent_record
    )


# Restored from before K1 (GATE-noskip): the same ids, now against the one
# exec that freezes, kills and checks. Each keeps its old guard.


@pytest.mark.asyncio
async def test_failed_final_probe_prevents_verification():
    """GH #1078: an unsuccessful final process observation cannot admit
    scoring. The kill and its last check are one exec since K1; a survivor
    makes it exit 1."""
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="Sandbox-user writers remain: 1234 (sleep)",
            )
        )
    )
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill_sandbox_user_procs(env, "agent")


# How each stand-in /proc entry behaves: ``live`` never dies, ``dies_on_term``
# saves and exits on SIGTERM, ``dies_on_kill`` ignores SIGTERM; ``vanished``
# is a dangling link (a process that exited between the listing and the
# status read).
_KINDS = {
    "live": {"status": _LIVE, "stubborn": True},
    "dies_on_term": {"status": _LIVE, "saver": True},
    "dies_on_kill": {"status": _RUNNING},
    "zombie": {"status": _ZOMBIE},
    "other_effective_uid": {
        "status": f"State:\tS (sleeping)\nUid:\t{UID}\t5678\t{UID}\t{UID}\n"
        "Threads:\t1\n"
    },
    "no_state": {"status": f"Name:\tunknown\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\n"},
    "unreadable": {"status": None},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "processes,expected,signalled",
    [
        ([], 0, []),  # no process of the sandbox user
        (["dies_on_term"], 0, ["100"]),
        (["dies_on_kill"], 0, ["100"]),  # the second signal
        (["live"], 1, ["100"]),  # writer survived
        (["zombie"], 0, []),  # a zombie cannot write
        # Real and saved UID the user's: a target now (the
        # effective UID alone decided before K1).
        (["other_effective_uid"], 0, ["100"]),
        (["vanished", "zombie"], 0, []),  # exited before inspection
        (["no_state"], 2, []),  # existing entry without a state
        (["unreadable", "dies_on_term"], 2, []),  # existing but unreadable
    ],
)
async def test_real_shell_distinguishes_absence_failure_and_live_writers(
    tmp_path, processes, expected, signalled
):
    """GH #1078/PR #1088: absence, a failed observation and a live writer are
    three outcomes; a failed observation never passes as quiescence."""
    proc = _FakeProc(tmp_path)
    for kind in processes:
        if kind == "vanished":
            (proc.root / str(proc._pid)).symlink_to(tmp_path / "exited")
            proc._pid += 1
            continue
        spec = dict(_KINDS[kind])
        proc.add(spec.pop("status"), **spec)
    if expected == 0:
        await _kill_sandbox_user_procs(proc, "agent")
    else:
        with pytest.raises(RuntimeError, match=f"rc={expected}"):
            await _kill_sandbox_user_procs(proc, "agent")
    sent = {line.split()[1] for line in proc.signalled()} - {"-1"}
    assert sorted(sent) == signalled


@pytest.mark.asyncio
async def test_quiescence_needs_neither_procps_nor_python(tmp_path):
    """busybox/slim images without pgrep, pkill or python3 must be observed as
    quiescent, not fail with rc=127: the real script runs with a PATH that has
    only ``id``, ``sleep`` and ``setpriv`` (which runs its command with a
    ``kill`` that does nothing, so nothing on the host is signalled)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in {
        "id": "echo 1234",
        "sleep": "exit 0",
        "setpriv": 'while [ $# -gt 0 ]; do [ "$1" = -c ] && '
        'exec /bin/sh -c "kill() { :; }; $2"; shift; done; exit 1',
    }.items():
        (bindir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bindir / name).chmod(0o755)
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/status").write_text("State:\tR (running)\nUid:\t0\t0\t0\t0\n")
    (proc / "100").mkdir()
    (proc / "100/status").write_text(
        "State:\tZ (zombie)\nUid:\t1234\t1234\t1234\t1234\n"
    )
    commands = []

    class ShellEnv:
        async def exec(self, command, **kwargs):
            commands.append(command)
            command = command.replace(
                f"export PATH={shlex.quote(_SAFE_VERIFIER_PATH)}",
                f"export PATH={shlex.quote(str(bindir))}",
            ).replace("/proc", str(proc))
            result = subprocess.run(
                ["/bin/sh", "-c", "kill() { :; }\n" + command],
                # Root's PATH is the route's (C8): the exec starts with it.
                env={"PATH": str(bindir)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    assert await _kill_sandbox_user_procs(ShellEnv(), "agent") == 0
    assert len(commands) == 1
    assert "python3" not in commands[0] and "pgrep" not in commands[0]


@pytest.mark.asyncio
async def test_controller_uid_rejected_before_any_signal(tmp_path):
    """GH #1078: even a root alias must not kill the verifier/controller UID.
    The real ``id`` resolves root to 0; every signal command is a marker."""
    sentinel = tmp_path / "signalled"

    class ShellEnv:
        async def exec(self, command, **kwargs):
            mark = f"touch {shlex.quote(str(sentinel))}"
            stubs = (
                f"kill() {{ {mark}; }}\nsetpriv() {{ {mark}; }}\nsu() {{ {mark}; }}\n"
            )
            result = subprocess.run(
                ["/bin/sh", "-c", stubs + command],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    with pytest.raises(RuntimeError, match="identity check"):
        await _kill_sandbox_user_procs(ShellEnv(), "root")
    assert not sentinel.exists()


# PID 1 as the image's user. On Docker PID 1 runs as the image's USER, and a
# task image whose USER is the sandbox user's account (Terminal-Bench 4.1's
# rs-archive-clone: ``USER agent``) gives PID 1 that UID. Docker's default
# bounding set (CapBnd), which has no CAP_SYS_PTRACE (bit 19), and one with it.
_BOUND = "00000000a80425fb"
_BOUND_PTRACE = "00000000a80c25fb"


def _init_status(tracer="0", bound=_BOUND):
    return (
        f"State:\tS (sleeping)\nTracerPid:\t{tracer}\n"
        f"Uid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t1\nCapBnd:\t{bound}\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "bound", "keep_alive"),
    [
        (1, _BOUND, "sleep infinity"),
        (2, _BOUND, "sleep infinity"),
        (3, _BOUND_PTRACE, "sleep infinity"),
        (1, _BOUND, f"{KIT_BIN}/sleep infinity"),
    ],
    ids=["yama-1", "yama-2", "yama-3-ptrace-bounded", "kit-sleep-restored"],
)
async def test_the_keep_alive_init_is_left_alone_and_every_agent_process_dies(
    tmp_path, scope, bound, keep_alive
):
    """Guards the kill's PID 1 rule against the regression of the kill before
    grading (c5ab45031, 74c0e84eb) meeting the compose keep-alive (7a479e9f4,
    #294) in an image whose USER is the sandbox user: the kill ended the
    keep-alive's sleep, PID 1 (a shell waiting on it) exited, and the
    container went down with BenchFlow's root command helper (the trial
    errored: "the sandbox's command channel stopped: the helper closed its
    output"). With the keep-alive as PID 1 itself, the kill counted it, could
    not end it, and failed. Now an untraceable ``sleep infinity`` PID 1 is left
    alone and every other process of the user dies: one whose command line is
    the keep-alive's too, and an orphan (which the kernel reparents to PID 1)."""
    proc = _FakeProc(tmp_path)
    proc.add_init(_init_status(bound=bound), keep_alive, ptrace_scope=scope)
    mimic = proc.add(_LIVE, cmdline="sleep infinity")
    orphan = proc.add(_RUNNING, cmdline="python3 -c daemon")
    worker = proc.add(_LIVE, regen=True)
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert proc.live_pids() == ["1"]
    assert killed == 3
    assert not any(line.split()[1:2] == ["1"] for line in proc.signalled())
    assert {mimic, orphan, worker}.isdisjoint(int(p) for p in proc.live_pids())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "tracer", "bound", "cmdline"),
    [
        (0, "0", _BOUND, "sleep infinity"),
        (None, "0", _BOUND, "sleep infinity"),
        (1, "0", _BOUND_PTRACE, "sleep infinity"),
        (2, "0", _BOUND_PTRACE, "sleep infinity"),
        (1, "0", "a80425fb", "sleep infinity"),
        (1, "4242", _BOUND, "sleep infinity"),
        (1, "0", _BOUND, "sh -c sleep infinity"),
        (1, "0", _BOUND, "sleep infinity now"),
        (1, "0", _BOUND, "/bin/sleep infinity"),
    ],
    ids=[
        "yama-0",
        "no-yama",
        "ptrace-bounded-1",
        "ptrace-bounded-2",
        "short-mask",
        "traced",
        "shell-keep-alive",
        "other-command",
        "snapshot-sleep",
    ],
)
async def test_an_init_that_may_act_for_the_agent_still_fails_the_kill(
    tmp_path, scope, tracer, bound, cmdline
):
    """The other half: a PID 1 of the user's that some process of the user's
    could have traced (Yama off or absent, CAP_SYS_PTRACE in the bounding set
    where Yama lets its holder attach, a tracer now) or that is not the
    keep-alive is counted as before (so is a restored snapshot's own
    ``/bin/sleep``, a file of the image the agent worked in). No kill from
    inside can end it, so the kill fails and the trial is not graded; every
    other process still dies."""
    proc = _FakeProc(tmp_path)
    proc.add_init(_init_status(tracer, bound), cmdline, ptrace_scope=scope)
    proc.add(_LIVE)
    with pytest.raises(RuntimeError, match=r"writers remain: 1 \("):
        await _kill_sandbox_user_procs(proc, "agent")
    assert proc.live_pids() == ["1"]

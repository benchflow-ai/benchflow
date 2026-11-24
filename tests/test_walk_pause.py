"""C2-1: the agent's processes held still for each link walk after the turn.

The pause and the resume are root shells over /proc. These rows run them in a real ``sh`` over a stand-in /proc
folder, with ``kill`` replaced by a shell function that changes the stand-in's State as the kernel would (and logs
each signal), so the sweep-until-still logic is tested without stopping a real process. The last rows check that
each walk after the turn runs inside the pause: before test.sh (walk, removal and confirming look) and after
grading.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _verifier_links

# The pause itself is tested here: no stand-in answers it (tests/_fake_bfx.py).
pytestmark = pytest.mark.bfx_refusal

AGENT = 1000

#: The stand-in kernel: ``kill -SIG pid`` sets the pid's State (STOP: T, CONT: S) unless the pid is listed in
#: ``$FAKE/stubborn`` (it never stops: a task in uninterruptible sleep); the first STOP of a pid listed in
#: ``$FAKE/forks`` makes a new running child (a fork between the pass's read and its stop). Each signal is logged
#: to ``$FAKE/log``.
_KILL = r"""
kill() {
  _k_sig=${1#-}; _k_pid=$2
  [ -d "$FAKE/proc/$_k_pid" ] || return 1
  echo "$_k_sig $_k_pid" >> "$FAKE/log"
  case "$_k_sig" in
    STOP)
      if grep -qx "$_k_pid" "$FAKE/stubborn" 2>/dev/null; then return 0; fi
      if grep -q "^State:.Z" "$FAKE/proc/$_k_pid/status"; then return 0; fi
      sed -i "s/^State:.*/State:\tT (stopped)/" "$FAKE/proc/$_k_pid/status"
      if grep -qx "$_k_pid" "$FAKE/forks" 2>/dev/null; then
        sed -i "/^$_k_pid\$/d" "$FAKE/forks"
        _k_new=$((_k_pid + 500)); mkdir -p "$FAKE/proc/$_k_new"
        printf 'Pid:\t%s\nPPid:\t%s\nState:\tR (running)\nUid:\t%s\t%s\t%s\t%s\nThreads:\t1\n' \
          "$_k_new" "$_k_pid" 1000 1000 1000 1000 > "$FAKE/proc/$_k_new/status"
      fi ;;
    CONT) sed -i "s/^State:.*/State:\tS (sleeping)/" "$FAKE/proc/$_k_pid/status" ;;
  esac
  return 0
}
"""


def _proc(
    root: Path,
    pid: int,
    *,
    uid: int = AGENT,
    state: str = "S",
    threads: int = 1,
    ppid: int = 1,
) -> None:
    d = root / "proc" / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status").write_text(
        f"Pid:\t{pid}\nPPid:\t{ppid}\nState:\t{state} (x)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nThreads:\t{threads}\n"
    )


def _fake(tmp_path: Path) -> Path:
    root = tmp_path / "fake"
    (root / "proc" / "self").mkdir(parents=True)
    # The shell's own entry: a pid no stand-in process has, whose parent is 0 (an exec's).
    (root / "proc" / "self" / "status").write_text(
        "Pid:\t90000\nPPid:\t0\nState:\tR (running)\n"
    )
    _proc(
        root, 1, uid=AGENT
    )  # PID 1 as the agent's uid (an image whose USER is the agent): never stopped
    return root


def _run(root: Path, script: str) -> str:
    out = subprocess.run(
        ["sh", "-c", _KILL + script],
        env={"FAKE": str(root), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _state(root: Path, pid: int) -> str:
    for line in (root / "proc" / str(pid) / "status").read_text().splitlines():
        if line.startswith("State:"):
            return line.split()[1]
    return ""


def _signals(root: Path) -> list[str]:
    log = root / "log"
    return log.read_text().splitlines() if log.is_file() else []


def _pause(root: Path):
    from benchflow.sandbox import _walk_pause

    out = _run(root, _walk_pause.pause_command(AGENT, proc=str(root / "proc")))
    return _walk_pause.parse_pause(_walk_pause.Pause(user="agent", uid=AGENT), out, 0)


def _resume(root: Path, found):
    from benchflow.sandbox import _walk_pause

    out = _run(
        root, _walk_pause.resume_command(AGENT, found.pids, proc=str(root / "proc"))
    )
    return _walk_pause.parse_resume(found, out, 0)


def test_the_pause_stops_every_agent_process_and_the_resume_continues_them_in_reverse(
    tmp_path,
):
    """C2-1 (sweep until still). Two running processes of the agent's are stopped; root's process, PID 1 and a
    process the agent had stopped itself are left as they are. The pause shows every agent process stopped, the
    resume finds them all still stopped and continues the two in the reverse of the order they were stopped, never
    the one stopped before the pause.

    Mutants: a process of the agent's left running; the one stopped before the pause continued."""
    root = _fake(tmp_path)
    _proc(root, 10)
    _proc(root, 11, state="R")
    _proc(root, 12, uid=0)
    _proc(root, 13, state="T")
    found = _pause(root)
    assert found.still is True and found.why is None, found
    assert found.pids == [10, 11] and found.pre == [13], found
    assert [_state(root, p) for p in (1, 10, 11, 12, 13)] == ["S", "T", "T", "S", "T"]
    found = _resume(root, found)
    assert found.held is True and found.ok(), found
    assert [s for s in _signals(root) if s.startswith("CONT")] == ["CONT 11", "CONT 10"]
    assert [_state(root, p) for p in (10, 11, 13)] == ["S", "S", "T"]


def test_a_process_forked_between_passes_is_stopped_by_a_later_pass(tmp_path):
    """C2-1. A process of the agent's makes a child while the pass stops it: the pass that found it running isn't
    the last; the next one finds the child running and stops it, and only a pass that finds none running ends it.

    Mutant: one pass only (the child runs through the walk)."""
    root = _fake(tmp_path)
    _proc(root, 20)
    (root / "forks").write_text("20\n")
    found = _pause(root)
    assert found.still is True, found
    assert found.pids == [20, 520], found
    assert _state(root, 520) == "T"


@pytest.mark.parametrize("kind", ["never-stops", "zombie-with-threads"])
def test_a_process_that_cannot_be_shown_stopped_leaves_the_pause_unproven(
    tmp_path, kind
):
    """C2-1. A process of the agent's that never shows stopped (in uninterruptible sleep, or a zombie leader whose
    other threads still run): after the bounded passes the pause is not shown, and says which; the walk that
    needed it is incomplete. What the pause stopped is still continued.

    Mutant: the bounded passes taken as proof (still)."""
    root = _fake(tmp_path)
    _proc(root, 30)
    if kind == "never-stops":
        _proc(root, 31, state="D")
        (root / "stubborn").write_text("31\n")
    else:
        _proc(root, 31, state="Z", threads=3)
    found = _pause(root)
    assert found.still is False, found
    assert "31" in str(found.why) and "held still" in str(found.why), found
    found = _resume(root, found)
    assert "CONT 30" in _signals(root)
    assert found.ok() is False


def test_a_process_running_at_the_resume_means_the_pause_did_not_hold(tmp_path):
    """C2-1. Something continued a stopped process of the agent's during the walk: the resume's look finds it
    running, and the pause did not hold (the walk is incomplete).

    Mutant: the resume's look skipped."""
    root = _fake(tmp_path)
    _proc(root, 40)
    found = _pause(root)
    assert found.still is True, found
    _proc(root, 40, state="R")
    found = _resume(root, found)
    assert found.held is False and "40" in str(found.why), found


def test_a_pid_taken_by_another_account_is_never_continued(tmp_path):
    """C2-1. A stopped pid of the agent's that ended and was taken by root's process before the resume: it is not
    continued.

    Mutant: CONT by pid alone."""
    root = _fake(tmp_path)
    _proc(root, 50)
    _proc(root, 51)
    found = _pause(root)
    _proc(root, 51, uid=0, state="T")
    _resume(root, found)
    assert [s for s in _signals(root) if s.startswith("CONT")] == ["CONT 50"]


# -- each walk after the turn runs inside the pause --------------------------------------------------------------


def _paused_env(events):
    """A sandbox whose root execs (the pause, the resume, the kernel's resolution) and bfx calls are logged in
    order. The agent's link /app/x leads to the reward."""
    from benchflow.sandbox._identities import Identities

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        events.append(f"bfx {cmd}")
        if cmd == "walk":
            link = {
                "path": "app/x",
                "type": "link",
                "link": "/logs/verifier/reward.txt",
                "dev": 7,
                "ino": 42,
            }
            return {
                "entries": [link] if args.get("root") == "/" else [],
                "vanished": [],
                "deeper": [],
            }
        if cmd == "remove":
            return {"removed": True}
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}

    async def exec_(command, *args, **kwargs):
        text = str(command)
        if "BF pause" in text and "kill -STOP" in text:
            events.append("pause")
            return SimpleNamespace(
                return_code=0,
                stdout="BF pause pid 77\nBF pause at 5.0\nBF pause still 1\n",
                stderr="",
            )
        if "BF resume" in text:
            events.append("resume")
            return SimpleNamespace(
                return_code=0,
                stdout="BF resume at 6.0\nBF resume pid 77\nBF resume done\n",
                stderr="",
            )
        if "readlink" in text or "realpath" in text:
            events.append("resolve")
            return SimpleNamespace(
                return_code=0, stdout="/app/x\0/logs/verifier/reward.txt\0", stderr=""
            )
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    env = SimpleNamespace(
        bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock(side_effect=exec_)
    )
    env._benchflow_identities = Identities(
        "root:x:0:0::/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n",
        "root:x:0:\nagent:x:1000:\n",
        "0:0",
    )
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_each_walk_after_the_turn_runs_inside_the_pause(check):
    """C2-1. For a non-root agent the walk after the turn (before test.sh: with its removal and confirming look)
    runs between the pause and the resume, and the check's record holds the pause: its times and pids.

    Mutant: the walk not paused."""
    events: list[str] = []
    env = _paused_env(events)
    found = await getattr(_verifier_links, check)(
        env, timeout_sec=5, pause_user="agent"
    )
    resumed = events.index("resume")
    walks = [i for i, e in enumerate(events) if e == "bfx walk"]
    assert events[0] == "pause" and walks and walks[0] > 0 and walks[-1] < resumed, (
        events
    )
    if check == "remove_turn_links":
        # The removal inside the pause; the confirming look after the resume (813-M1, probe P5).
        assert events.index("bfx remove") < resumed, events
        assert events[resumed + 1 :] == ["bfx scan"], events
    else:
        assert events[-1] == "resume", events
    pause = found.get("pause") or {}
    assert (
        pause.get("pids") == [77]
        and pause.get("paused_at") == 5.0
        and pause.get("resumed_at") == 6.0
    ), found
    assert found["ran"] is True, found


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_a_pause_that_is_not_shown_leaves_the_walk_incomplete(check):
    """C2-1. The pause can't show every process of the agent's stopped: the walk still runs (a link it finds is
    still removed) but the check did not run, with the reason; the resume still continues what was stopped.

    Mutant: an unshown pause taken as a still sandbox."""
    events: list[str] = []
    env = _paused_env(events)
    exec_ = env.exec.side_effect

    async def unproven(command, *args, **kwargs):
        if "kill -STOP" in str(command) and "BF pause" in str(command):
            events.append("pause")
            return SimpleNamespace(
                return_code=0,
                stdout="BF pause pid 77\nBF pause running 78\nBF pause still 0\n",
                stderr="",
            )
        return await exec_(command, *args, **kwargs)

    env.exec = AsyncMock(side_effect=unproven)
    found = await getattr(_verifier_links, check)(
        env, timeout_sec=5, pause_user="agent"
    )
    assert found["ran"] is False and "held still" in str(found["why"]), found
    assert "resume" in events, events


@pytest.mark.asyncio
async def test_a_root_agent_s_walk_is_not_paused():
    """C2-1. A root agent's reward is already unverifiable: no pause is made for it (pause_user None)."""
    events: list[str] = []
    env = _paused_env(events)
    found = await _verifier_links.links_after_grading(env, timeout_sec=5)
    assert "pause" not in events and "pause" not in found, (events, found)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ["read-now", "unreadable"])
async def test_without_the_pre_agent_table_the_pause_reads_the_uid_itself(table):
    """C2-1. A trial that never made the identity table before the agent started (audit runs no kill): the pause
    reads the uid as root with ``id -u`` (the kill's own read; the account files are root's) and reports it. A uid
    it can't read (or root's) makes no pause, and the walk that needed it is incomplete, with the reason.

    Mutant: no table taken as no pause needed."""
    from benchflow.sandbox import _walk_pause

    commands = []

    async def exec_(command, *args, **kwargs):
        commands.append(str(command))
        said = (
            "BF pause uid 1000\nBF pause at 5.0\nBF pause still 1\n"
            if table == "read-now"
            else "BF pause nouid\nBF pause still 0\n"
        )
        return SimpleNamespace(return_code=0, stdout=said, stderr="")

    env = SimpleNamespace(exec=AsyncMock(side_effect=exec_))
    found = await _walk_pause.pause(env, "agent", timeout_sec=5)
    assert "id -u agent" in commands[0], commands
    # C3-6: found on the fixed PATH, as the pause's sleep and date are, never the exec's.
    from benchflow.task.verifier_freeze import _FIXED_PATH

    assert f"$({_FIXED_PATH} id -u agent" in commands[0], commands
    if table == "read-now":
        assert found.uid == 1000 and found.still is True, found
    else:
        assert (
            found.uid is None
            and found.still is False
            and "uid of agent" in str(found.why)
        ), found


def test_the_pause_s_own_uid_read_refuses_root(tmp_path):
    """C2-1. The shell's own read: a name that resolves to uid 0 (or to nothing) is no account to pause."""
    from benchflow.sandbox import _walk_pause

    out = subprocess.run(
        ["sh", "-c", _walk_pause.pause_command(None, user="root", proc=str(tmp_path))],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "BF pause nouid" in out and "BF pause still 0" in out, out


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_a_uid_the_pause_cannot_read_leaves_the_walk_incomplete(check):
    """C3-4. No identity table from before the turn, and the pause's own ``id -u`` can't read the uid (or reads
    root's): no process is held. The walk still runs, but the check did not run, with the reason, for both checks
    (the Q6 0 for a non-root agent).

    Mutant (the C3 review's d): the uid read failing open (the walk unpaused, reported complete)."""
    events: list[str] = []
    env = _paused_env(events)
    del env._benchflow_identities
    exec_ = env.exec.side_effect

    async def nouid(command, *args, **kwargs):
        if "kill -STOP" in str(command) and "BF pause" in str(command):
            events.append("pause")
            return SimpleNamespace(
                return_code=0, stdout="BF pause nouid\nBF pause still 0\n", stderr=""
            )
        return await exec_(command, *args, **kwargs)

    env.exec = AsyncMock(side_effect=nouid)
    found = await getattr(_verifier_links, check)(
        env, timeout_sec=5, pause_user="agent"
    )
    assert found["ran"] is False, found
    assert "the uid of agent could not be read" in str(found["why"]), found
    assert (found.get("pause") or {}).get("still") is False, found

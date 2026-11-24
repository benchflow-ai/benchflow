"""Rule S (sandbox/_agent_alive.py): whether a process of the agent's is alive when test.sh starts (C3).

The question runs in a real ``sh`` over a stand-in /proc (PID 1's uid, the listing) and stand-in readers on a PATH of
the test's own; the getpriority reader also runs on the real kernel, asked about this test's own uid (always found)
and a uid nothing runs as (never found).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _agent_alive

pytestmark = pytest.mark.bfx_refusal

AGENT = 1000


def _proc(root: Path, pid: int, uid: int, state: str = "S") -> None:
    d = root / "proc" / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status").write_text(
        f"Pid:\t{pid}\nState:\t{state} (x)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nThreads:\t1\n"
    )


#: A stand-in util-linux ionice: ``ionice -u <uid>`` answers from ``$FAKE/answers`` (lines ``<uid> found|none|eperm|<rc>``),
#: logging each uid it was asked about to ``$FAKE/asked``.
_IONICE = r"""#!/bin/sh
[ "$1" = -u ] || exit 64
echo "$2" >> "$FAKE/asked"
a=$(sed -n "s/^$2 //p" "$FAKE/answers")
case "$a" in
  found) echo "none: prio 4"; exit 0 ;;
  none|"") echo "ionice: ioprio_get failed: No such process" >&2; exit 1 ;;
  eperm) echo "ionice: ioprio_get failed: Operation not permitted" >&2; exit 1 ;;
  *) echo "ionice: unrecognized option: u" >&2; exit "$a" ;;
esac
"""


def _sandbox(
    tmp_path: Path,
    *,
    pid1: int = 0,
    answers: dict[int, str] | None = None,
    python: bool = False,
) -> Path:
    """A stand-in sandbox: /proc with PID 1 of *pid1*'s uid, a ``bin`` holding the stand-in ionice when *answers*
    is given (and this interpreter as ``python3`` when *python*), and ``id``."""
    root = tmp_path / "sb"
    _proc(root, 1, pid1)
    bin_ = root / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "id").write_text('#!/bin/sh\n[ "$2" = agent ] && echo 1000 || exit 1\n')
    (bin_ / "id").chmod(0o755)
    for tool in ("date", "sed"):
        (bin_ / tool).symlink_to(
            f"/usr/bin/{tool}" if Path(f"/usr/bin/{tool}").exists() else f"/bin/{tool}"
        )
    if answers is not None:
        (root / "answers").write_text("".join(f"{u} {a}\n" for u, a in answers.items()))
        (bin_ / "ionice").write_text(_IONICE)
        (bin_ / "ionice").chmod(0o755)
    if python:
        (bin_ / "python3").symlink_to(sys.executable)
    return root


def _ask(root: Path, uid: int | None = AGENT, kit: bool = True, **kw) -> dict:
    # The fake bin's python3 stands in for the kit's Python (S never runs the
    # image's own; test_without_a_kit_s_never_runs_the_image_s_python3).
    if kit and (root / "bin" / "python3").exists():
        kw.setdefault("kit_python", str(root / "bin" / "python3"))
    command = _agent_alive.alive_command(
        uid, user="agent", proc=str(root / "proc"), path=f"PATH={root / 'bin'}", **kw
    )
    out = subprocess.run(
        ["/bin/sh", "-c", command],
        env={"FAKE": str(root), "PATH": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    found = {
        "user": "agent",
        "uid": uid,
        "asked_at": None,
        "alive": None,
        "by": None,
        "why": None,
    }
    return _agent_alive.parse_alive(found, out.stdout, out.returncode)


def _asked(root: Path) -> list[str]:
    path = root / "asked"
    return path.read_text().split() if path.is_file() else []


def test_none_alive_is_the_kernel_s_none_with_both_controls(tmp_path):
    """The reader answers "none" for the agent's uid, finds root, and finds nothing for a uid no account holds."""
    root = _sandbox(
        tmp_path, answers={AGENT: "none", 0: "found", _agent_alive.NO_ONE: "none"}
    )
    found = _ask(root)
    assert (
        found["alive"] is False and found["by"] == "ionice" and found["why"] is None
    ), found
    assert _asked(root) == [str(AGENT), "0", str(_agent_alive.NO_ONE)], _asked(root)


def test_one_alive_is_found_and_listed(tmp_path):
    """Found: alive. The record lists the agent's processes from /proc (a zombie's state Z), for the reader only."""
    root = _sandbox(
        tmp_path, answers={AGENT: "found", 0: "found", _agent_alive.NO_ONE: "none"}
    )
    _proc(root, 40, AGENT, "S")
    _proc(root, 41, AGENT, "Z")
    _proc(root, 42, 0, "S")
    found = _ask(root)
    assert found["alive"] is True, found
    assert found["pids"] == [{"pid": 40, "state": "S"}, {"pid": 41, "state": "Z"}], (
        found
    )


def test_the_question_is_about_the_agent_s_uid(tmp_path):
    """The kernel is asked about the agent's uid (here read by root's ``id -u agent``), never root's, which is
    always found.

    Mutant: the account's uid replaced by root's."""
    root = _sandbox(
        tmp_path, answers={AGENT: "none", 0: "found", _agent_alive.NO_ONE: "none"}
    )
    found = _ask(root, None)
    assert found["uid"] == AGENT and found["alive"] is False, found
    assert _asked(root)[0] == str(AGENT), _asked(root)


@pytest.mark.parametrize(
    ("answers", "said"),
    [
        # A fabricated ESRCH: "none" for everyone, root included.
        ({AGENT: "none", 0: "none", _agent_alive.NO_ONE: "none"}, "controls failed"),
        # A stub that finds everyone (gVisor's getpriority(PRIO_USER) returns 0).
        ({AGENT: "found", 0: "found", _agent_alive.NO_ONE: "found"}, "controls failed"),
        # busybox's ionice, which has no -u.
        ({AGENT: "1", 0: "1", _agent_alive.NO_ONE: "1"}, "controls failed"),
    ],
    ids=["esrch-for-all", "found-for-all", "no-user-option"],
)
def test_an_answer_whose_controls_fail_is_no_answer(tmp_path, answers, said):
    """The analogue of the as-user helper's temporary child: an answer counts only when the same reader found root
    and found nothing for a uid no account holds. Otherwise the question went unanswered (never "none alive").

    Mutant: the controls ignored."""
    root = _sandbox(tmp_path, answers=answers)
    found = _ask(root)
    assert found["alive"] is None, found
    assert found["why"].startswith(_agent_alive.UNANSWERED) and said in found["why"], (
        found
    )


def test_the_next_reader_answers_when_the_first_fails_its_controls(tmp_path):
    """busybox's ionice fails its controls; python3 (here this interpreter, on the real kernel) answers."""
    root = _sandbox(
        tmp_path, answers={AGENT: "1", 0: "1", _agent_alive.NO_ONE: "1"}, python=True
    )
    found = _ask(root, os.getuid() if os.getuid() != 0 else AGENT)
    assert found["by"] == "getpriority" and isinstance(found["alive"], bool), found


def test_the_real_kernel_finds_a_live_uid_and_not_an_unused_one(tmp_path):
    """getpriority(PRIO_USER) on the real kernel: this test's own uid has a process (this one), and a uid nothing
    runs as has none."""
    if os.getuid() == 0:
        pytest.skip("the controls need a uid other than root's for this row")
    root = _sandbox(tmp_path, python=True)
    assert _ask(root, os.getuid())["alive"] is True
    found = _ask(root, 1999999)
    assert found["alive"] is False and found["by"] == "getpriority", found


@pytest.mark.parametrize("case", ["pid1-agent", "no-reader", "no-uid"])
def test_a_question_that_cannot_be_answered_says_why(tmp_path, case):
    """PID 1 of the agent's uid (the kernel's walk can't leave it out), no reader in the sandbox, or a uid that
    can't be read: unanswered, with the reason."""
    root = _sandbox(
        tmp_path,
        pid1=AGENT if case == "pid1-agent" else 0,
        answers=None
        if case == "no-reader"
        else {AGENT: "none", 0: "found", _agent_alive.NO_ONE: "none"},
    )
    if case == "no-uid":
        (root / "bin" / "id").write_text("#!/bin/sh\nexit 1\n")
    found = _ask(root, None if case == "no-uid" else AGENT)
    assert found["alive"] is None and found["why"].startswith(
        _agent_alive.UNANSWERED
    ), found
    said = {
        "pid1-agent": "PID 1 runs as the agent's uid",
        "no-reader": "no program",
        "no-uid": "could not be read",
    }
    assert said[case] in found["why"], found


@pytest.mark.asyncio
async def test_ask_is_one_root_exec_and_a_failed_one_is_unanswered():
    """``ask`` runs the question as one root exec; an exec that raises is a question that went unanswered."""
    calls = []

    async def exec_(command, *args, **kwargs):
        calls.append((str(command), kwargs.get("user")))
        return SimpleNamespace(
            return_code=0,
            stdout="BF alive uid 1000\nBF alive at 5.0\nBF alive by ionice none found:0 none\n",
            stderr="",
        )

    env = SimpleNamespace(exec=AsyncMock(side_effect=exec_))
    found = await _agent_alive.ask(env, "agent", timeout_sec=5)
    assert len(calls) == 1 and calls[0][1] == "root" and "BF alive" in calls[0][0], (
        calls
    )
    assert (
        found["alive"] is False and found["uid"] == 1000 and found["asked_at"] == 5.0
    ), found

    env = SimpleNamespace(exec=AsyncMock(side_effect=RuntimeError("sandbox gone")))
    found = await _agent_alive.ask(env, "agent", timeout_sec=5)
    assert found["alive"] is None and "sandbox gone" in found["why"], found


@pytest.mark.asyncio
async def test_ask_asks_about_the_agent_s_uid_from_the_table_or_its_name():
    """S-2. ``ask`` puts the agent's uid into the question: the one from the identity table taken before the
    turn, else root's ``id -u`` of the agent's name; never root's 0.

    Mutant: ``ask`` passes uid 0 to the question."""
    from benchflow.sandbox._identities import Identities

    def env_with(table):
        commands: list[str] = []

        async def exec_(command, *args, **kwargs):
            commands.append(str(command))
            return SimpleNamespace(
                return_code=0,
                stdout="BF alive uid 1000\nBF alive by ionice none found:0 none\n",
                stderr="",
            )

        env = SimpleNamespace(exec=AsyncMock(side_effect=exec_))
        if table:
            env._benchflow_identities = Identities(
                "root:x:0:0::/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n",
                "root:x:0:\nagent:x:1000:\n",
                "0:0",
            )
        return env, commands

    env, commands = env_with(table=True)
    await _agent_alive.ask(env, "agent", timeout_sec=5)
    assert len(commands) == 1 and "_bf_uid=1000; " in commands[0], commands
    env, commands = env_with(table=False)
    await _agent_alive.ask(env, "agent", timeout_sec=5)
    assert "id -u agent" in commands[0] and "_bf_uid=0;" not in commands[0], commands


def test_an_ionice_failure_other_than_no_such_process_is_no_answer(tmp_path):
    """S-3. Only ionice's "No such process" (the kernel's ESRCH) reads as none alive. Another failure for the
    agent's uid alone (EPERM, say), with both controls good, leaves the question unanswered, never "none".

    Mutant: every failed ``ionice -u`` read as none."""
    root = _sandbox(
        tmp_path, answers={AGENT: "eperm", 0: "found", _agent_alive.NO_ONE: "none"}
    )
    found = _ask(root)
    assert found["alive"] is None, found
    assert "ionice answered error:1" in found["why"], found


def test_the_python_readers_start_through_root_s_entry_point():
    """C8. Both Python readers (getpriority, and gVisor's kill reader) start as ``root_python_sh`` starts them:
    the kit's Python once the sandbox has a kit's RECORD, ``-I -S -B``, and the prelude that leaves the cwd and
    drops ``PYTHON*``; never an interpreter launched by hand.

    Mutant: a reader launched outside the entry point (the source scan in test_c8_r10_root_python does not see an
    interpreter given as a placeholder)."""
    from benchflow._runtime_tree import KIT_RECORD

    command = _agent_alive.alive_command(AGENT, user="agent")
    assert command.count(f"if [ -f {KIT_RECORD} ]") == 2, command
    assert command.count('exec "$bf_py" -I -S -B -c') == 2, command
    assert command.count("_bf_os.chdir(") == 2, command


def test_without_a_kit_s_never_runs_the_image_s_python3(tmp_path):
    """L-1 (late-delta review): S asks after the agent's turn, when a world-writable
    file in the image's standard library may hold the agent's code; run as root, it
    forged a `verified` 1.0. With no kit Python, S never starts the image's python3:
    it asks ionice, or nobody (unverifiable)."""
    root = tmp_path / "s"
    (root / "bin").mkdir(parents=True)
    (root / "proc" / "1").mkdir(parents=True)
    (root / "proc" / "1" / "status").write_text("Uid:\t0\t0\t0\t0\n")
    marker = tmp_path / "ran"
    (root / "bin" / "python3").write_text(
        f"#!/bin/sh\ntouch {marker}\necho BF alive by getpriority found:0 found:0 none\n"
    )
    (root / "bin" / "python3").chmod(0o755)
    command = _agent_alive.alive_command(
        AGENT, user="agent", proc=str(root / "proc"), path=f"PATH={root / 'bin'}"
    )
    out = subprocess.run(
        ["/bin/sh", "-c", command],
        env={"FAKE": str(root), "PATH": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert not marker.exists(), out.stdout
    assert "by getpriority" not in out.stdout, out.stdout

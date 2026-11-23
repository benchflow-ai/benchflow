"""Rule S on gVisor (Modal): the kill reader.

gVisor stubs ``getpriority(PRIO_USER)``: one value for every uid, so its controls fail and S went unanswered on every
Modal trial. gVisor's ``kill(-1, 0)`` sent as an account answers ESRCH unless a process it may signal exists, so on
gVisor alone (its kernel says so in /proc/version) root asks that too, from a child that has become the account, with
the same two controls made the kill's way: a throwaway uid holding one live process must be found, a uid no account
holds must not be. The question runs in a real ``sh`` over a stand-in /proc and a stand-in ``python3`` that answers
as gVisor does (the kill reader's answer from ``$FAKE/kill``); the live answers are in B2R-STATUS.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import _agent_alive

pytestmark = pytest.mark.bfx_refusal

AGENT = 1000
GVISOR = "Linux version 4.19.0-gvisor #1 SMP Sun Jan 10 15:06:54 PST 2016\n"
LINUX = "Linux version 7.0.0-1017-gcp (buildd@lcy02) #17 SMP PREEMPT_DYNAMIC\n"

#: A stand-in python3: the kill reader's source answers ``BF alive by kill <$FAKE/kill>``; any other source (the
#: getpriority reader) answers as gVisor's stub does, or ``$FAKE/gp`` when given. Each reader asked is logged to
#: ``$FAKE/asked``. Shell builtins only: the question's PATH holds nothing else.
_PYTHON = r"""#!/bin/sh
src=
while [ $# -gt 0 ]; do case "$1" in -c) src=$2; shift 2 ;; *) shift ;; esac; done
case "$src" in
  *"os.kill(-1"*) echo kill >> "$FAKE/asked"; read -r k < "$FAKE/kill"; echo "BF alive by kill $k" ;;
  *) echo getpriority >> "$FAKE/asked"; g="found:20 found:20 found:20"
     [ -f "$FAKE/gp" ] && read -r g < "$FAKE/gp"; echo "BF alive by getpriority $g" ;;
esac
"""


def _sandbox(
    tmp_path: Path, version: str | None, kill: str, gp: str | None = None
) -> Path:
    root = tmp_path / "sb"
    d = root / "proc" / "1"
    d.mkdir(parents=True)
    (d / "status").write_text("Pid:\t1\nState:\tS (x)\nUid:\t0\t0\t0\t0\nThreads:\t1\n")
    if version is not None:
        (root / "proc" / "version").write_text(version)
    bin_ = root / "bin"
    bin_.mkdir()
    for tool in ("date", "sed"):
        (bin_ / tool).symlink_to(
            f"/usr/bin/{tool}" if Path(f"/usr/bin/{tool}").exists() else f"/bin/{tool}"
        )
    (bin_ / "python3").write_text(_PYTHON)
    (bin_ / "python3").chmod(0o755)
    (root / "kill").write_text(kill + "\n")
    if gp is not None:
        (root / "gp").write_text(gp + "\n")
    return root


def _ask(root: Path) -> tuple[dict, list[str]]:
    command = _agent_alive.alive_command(
        AGENT,
        user="agent",
        proc=str(root / "proc"),
        path=f"PATH={root / 'bin'}",
        kit_python=str(root / "bin" / "python3"),  # stands in for the kit's
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
        "uid": AGENT,
        "asked_at": None,
        "alive": None,
        "by": None,
        "why": None,
    }
    asked = (root / "asked").read_text().split() if (root / "asked").is_file() else []
    return _agent_alive.parse_alive(found, out.stdout, out.returncode), asked


def test_on_gvisor_none_alive_is_the_kill_reader_s_none_with_both_controls(tmp_path):
    """(C3 part 5) getpriority's controls fail on gVisor; the kill reader answers ESRCH for the agent's
    uid, finds the throwaway process, and nothing for a uid no account holds: none alive, by kill.

    Mutant: no kill reader on gVisor (S unanswered on every Modal trial)."""
    found, asked = _ask(_sandbox(tmp_path, GVISOR, "none found:0 none"))
    assert found["alive"] is False and found["by"] == "kill" and found["why"] is None, (
        found
    )
    assert asked == ["getpriority", "kill"], asked


def test_on_gvisor_one_alive_is_found_by_the_kill_reader(tmp_path):
    """The agent's uid may signal one of its processes: alive, by kill.

    Mutant: the kill reader's answer and its found-control swapped."""
    found, _ = _ask(_sandbox(tmp_path, GVISOR, "found:0 found:0 none"))
    assert found["alive"] is True and found["by"] == "kill", found


@pytest.mark.parametrize(
    "kill",
    [
        "found:0 found:0 found:0",  # Linux's kill(-1): any other process counts, even root's
        "none error:control none",  # the throwaway process never became its uid
        "none none none",  # the throwaway process was not found
        "error:setuid found:0 none",  # the child could not become the account
    ],
)
def test_on_gvisor_a_kill_answer_whose_controls_fail_is_not_used(tmp_path, kill):
    """A kill answer whose controls fail is no answer: S stays unanswered (``unverifiable``, reward kept),
    as before.

    Mutant: the kill reader's controls ignored."""
    found, _ = _ask(_sandbox(tmp_path, GVISOR, kill))
    assert found["alive"] is None and found["by"] is None, found


@pytest.mark.parametrize("version", [LINUX, None])
def test_off_gvisor_the_kill_reader_is_never_asked(tmp_path, version):
    """On a kernel that does not say gVisor (or with no /proc/version) the kill reader is not run, even
    when getpriority's controls fail: Linux's kill(-1) answers for root's processes too.

    Mutant: the kill reader asked on every kernel."""
    found, asked = _ask(_sandbox(tmp_path, version, "none found:0 none"))
    assert found["alive"] is None and asked == ["getpriority"], (found, asked)


def test_on_gvisor_a_working_getpriority_still_answers_first(tmp_path):
    """Should gVisor's getpriority answer (its controls pass), it answers, ahead of the kill reader.

    Mutant: the kill reader placed first."""
    found, _ = _ask(
        _sandbox(tmp_path, GVISOR, "found:0 found:0 none", gp="none found:0 none")
    )
    assert found["alive"] is False and found["by"] == "getpriority", found

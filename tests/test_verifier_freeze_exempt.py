"""Which processes the freeze after test.sh's exit leaves running.

The stop is shell text the test command runs. It reads each process's command line with ``IFS= read -r`` (NULs
dropped) and leaves running BenchFlow's own exec channels, which carry the thaw and every later root step. These
rows run that very text, cut from :func:`verifier_freeze.freeze_step`, in a real shell over a fake ``/proc/<pid>``
folder (``cmdline``, ``status``):

- Z8: Daytona's channel helper (the kit's Python by its loader, the kit's Python, the channel interpreter) is left
  running; it matched neither kit pattern, so the bfx-off thaw could hang there (REV2 R2-2);
- Z9: a process is exempt only when its real uid is 0: any user can set its own argv, and a non-root agent's
  process named like a channel is stopped;
- Z13: the test command's own session machinery (the other children of the shells the stop runs under, below PID
  1) is left running. On Daytona those are the toolbox session's output copiers and its ``tail -f /dev/null``;
  stopped, the session never records the command's exit, so its exec waits out the whole verifier budget (the
  final dogfood, 11-18: frozen 119.7 s).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from benchflow.task import verifier_freeze

_SHELLS = [s for s in ("sh", "bash", "dash") if shutil.which(s)]
if shutil.which("busybox"):
    _SHELLS.append("busybox sh")


def _exemption(kit_channels: bool) -> str:
    """The freeze's own exemption text: from its read of the command line to its read of the state."""
    step = verifier_freeze.freeze_step("/run/bf-freeze-test", kit_channels=kit_channels)
    start = step.index("_bf_c=; IFS= read -r _bf_c")
    end = step.index("_bf_s=; ", start)
    return step[start:end]


def _kept(
    tmp_path, shell: str, argv: list[str], uid: int, *, kit_channels: bool
) -> bool:
    proc = tmp_path / "4242"
    proc.mkdir(exist_ok=True)
    (proc / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (proc / "status").write_text(
        f"Name:\tx\nState:\tS (sleeping)\nPid:\t4242\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t0\t0\t0\t0\n"
    )
    script = f"for _bf_d in {proc}; do {_exemption(kit_channels)} echo STOPPED; done; echo END\n"
    out = subprocess.run(
        [*shell.split(), "-c", script], capture_output=True, text=True, timeout=30
    )
    assert out.stdout.strip().endswith("END"), (out.stdout, out.stderr)
    return "STOPPED" not in out.stdout


_DAYTONA_HELPERS = {
    "kit-ld": [
        "/lib64/ld-linux-x86-64.so.2",
        "/opt/runtime/rt/libexec/python3",
        "-I",
        "-S",
        "-B",
        "/opt/runtime/run/channel-1a2b.py",
        "/opt/runtime/run/journal-1a2b",
    ],
    "kit-direct": [
        "/opt/runtime/rt/libexec/python3",
        "-I",
        "-S",
        "-B",
        "/opt/runtime/run/channel-1a2b.py",
        "/opt/runtime/run/journal-1a2b",
    ],
    "channel-python": [
        "/opt/runtime/rt/channel/python/bin/python3.12",
        "-I",
        "-S",
        "-B",
        "/opt/runtime/run/channel-1a2b.py",
        "/opt/runtime/run/journal-1a2b",
    ],
}


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize("helper", list(_DAYTONA_HELPERS))
def test_daytona_channel_helper_is_left_running(tmp_path, shell, helper):
    """Z8 (map-H2 §1.2f, R2-2). With the kit's channels (bfx not at /.bfx), Daytona's helper, root's, is left
    running through the freeze.

    Mutant: the Daytona patterns removed."""
    assert _kept(tmp_path, shell, _DAYTONA_HELPERS[helper], 0, kit_channels=True)


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize(
    ("argv", "kit"),
    [
        (["/.bfx/bfx", "serve", "--frames"], False),
        (["/.bfx/bfx", "scan"], False),
        (
            ["/opt/runtime/rt/libexec/python3", "-I", "-S", "-B", "-c", "\nimport os"],
            True,
        ),
        (_DAYTONA_HELPERS["kit-direct"], True),
    ],
    ids=["bfx-serve", "bfx-one-shot", "kit-python-c", "daytona-kit"],
)
def test_a_non_root_process_named_like_a_channel_is_stopped(tmp_path, shell, argv, kit):
    """Z9. Any user can set its own argv. A process named like one of BenchFlow's channels
    but running as uid 1000 is stopped; the same name as uid 0 is left running.

    Mutant: the uid check removed (the name alone exempts)."""
    assert not _kept(tmp_path, shell, argv, 1000, kit_channels=kit)
    assert _kept(tmp_path, shell, argv, 0, kit_channels=kit) is (argv[1:2] != ["scan"])


#: The chain the stop runs under on Daytona (live, 11-18): the toolbox daemon (PID 1), the test command's session
#: shell (3148), the route's dash, ``timeout`` and ``bash -c``, which runs test.sh and then the stop.
_DAYTONA_KEEP = " 1 3156 3155 3154 3148 "


def _kept_beside(
    tmp_path,
    shell: str,
    argv: list[str],
    uid: int,
    ppid: int,
    keep: str = _DAYTONA_KEEP,
) -> bool:
    """Whether the stop leaves running a process whose parent is *ppid*, with the stop's own chain set as on
    Daytona."""
    proc = tmp_path / "3151"
    proc.mkdir(exist_ok=True)
    (proc / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (proc / "status").write_text(
        f"Name:\tx\nState:\tS (sleeping)\nPid:\t3151\nPPid:\t{ppid}\n"
        f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t0\t0\t0\t0\n"
    )
    script = f'_bf_keep="{keep}"; for _bf_d in {proc}; do {_exemption(True)} echo STOPPED; done; echo END\n'
    out = subprocess.run(
        [*shell.split(), "-c", script], capture_output=True, text=True, timeout=30
    )
    assert out.stdout.strip().endswith("END"), (out.stdout, out.stderr)
    return "STOPPED" not in out.stdout


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize(
    "argv",
    [["/usr/bin/bash"], ["tail", "-f", "/dev/null"]],
    ids=["session-copier", "session-tail"],
)
def test_daytona_session_machinery_of_the_test_command_is_left_running(
    tmp_path, shell, argv
):
    """Z13 (the final dogfood's stall, 11-18). Root's children of the test command's session shell (Daytona's
    output copiers and its ``tail -f /dev/null``) are left running: stopped, the session never records the
    command's exit and the exec waits out the verifier budget.

    Mutant: the rule for the chain's other children removed."""
    assert _kept_beside(tmp_path, shell, argv, 0, 3148)


@pytest.mark.parametrize("shell", _SHELLS)
def test_init_children_and_non_root_children_are_still_stopped(tmp_path, shell):
    """Z13's controls. A root process under PID 1 (an agent's or test.sh's leftover, or another session) is
    stopped; so is a non-root child of the session shell, and, where the chain ends at an exec's PPid 0 (Docker,
    Modal), a root process another exec started.

    Mutants: PID 1 or PPid 0 counted among the chain's parents; the uid check dropped from the rule."""
    assert not _kept_beside(tmp_path, shell, ["/usr/bin/bash"], 0, 1)
    assert not _kept_beside(
        tmp_path, shell, ["/usr/bin/bash"], 0, 0, keep=" 1 77 76 0 "
    )
    assert not _kept_beside(tmp_path, shell, ["/usr/bin/bash"], 1000, 3148)
    assert not _kept_beside(tmp_path, shell, ["sleep", "100"], 0, 4242)

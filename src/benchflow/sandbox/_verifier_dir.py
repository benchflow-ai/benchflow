"""``/logs/verifier`` closed to the agent from sandbox setup (#1247).

Every backend's setup leaves the folder 0755: root's, or, where it is the host's own bind-mounted trial folder (local
Docker, Apple Container), the host user's. Nothing but root, and BenchFlow on the host, can create a name there at
any point of the agent's turn; a verifier user of the task's own gets the folder only in the close before test.sh
(``Verifier._close_output_dir``). The setup step judges the folder in the sandbox's own shell, in the same exec, and
says so (``BF verifier-dir-open <uid> <mode>``) when it isn't closed; that refuses the trial before any agent code
runs (``integrity_unavailable``), never an open folder.

A task's own start-up code (an entrypoint run as root that chowns the folder) may run after that close, since its
services are released just before it: on Docker the folder is closed the same way again right before each turn
(``DockerSandbox.close_verifier_dir_before_turn``). Where it is the host's mount, a folder another uid owns goes to
the host user as the sandbox sees it, never to root, so BenchFlow on the host can still write there.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Awaitable, Callable
from typing import Any

from benchflow.task.paths import SandboxPaths

VERIFIER_DIR = str(SandboxPaths.verifier_dir)
OPEN = "BF verifier-dir-open"
HOST_OWNER = "BF host-owner"


def _close_and_judge(folder: str, owners: str, on_open: str, give: str = "0:0") -> str:
    """sh, as root: *folder* exists and is 0755, root's, or, with *owners* other than ``0`` (the host's bind mount),
    kept when its owner matches *owners* (a ``case`` pattern) and given to *give* otherwise; then, unless its owner
    now matches *owners* and it has no group or other write, *on_open* runs with ``$bf_vd`` set to ``<uid> <mode>``
    (empty when unread)."""
    q = shlex.quote(folder)
    owner = (
        f"chown 0:0 {q}"
        if owners == "0"
        else f'case "$(stat -c %u {q})" in {owners}) ;; *) chown {give} {q};; esac'
    )
    return (
        f"{{ mkdir -p {q} && {owner} && chmod 755 {q}; }} 2>/dev/null; "
        f"bf_vd=$(stat -c '%u %a' {q} 2>/dev/null); bf_ok=; "
        f'case "${{bf_vd%% *}}" in {owners}) bf_m=${{bf_vd#* }}; case "$bf_m" in '
        "[0-7][0-7][0-7]|[0-7][0-7][0-7][0-7]) "
        "[ $(( 0$bf_m & 022 )) -eq 0 ] && bf_ok=1;; esac;; esac; "
        f'[ -n "$bf_ok" ] || {{ {on_open}; }}'
    )


def close_verifier_dir_cmd(
    *,
    host_uid: int | None = None,
    folder: str = VERIFIER_DIR,
    host_owner: tuple[int, int] | None = None,
) -> str:
    """sh, as root at sandbox setup: ``/logs/verifier`` (or *folder*, the VM folder bind-mounted there) closed, and
    ``BF verifier-dir-open <uid> <mode>`` printed when it isn't (:func:`check_verifier_dir`). Exits 0.

    With *host_uid*, the host's mount keeps that uid or root as its owner; any other owner makes it root's, or
    *host_owner*'s (the host user's uid and gid as the sandbox shows them, :func:`host_owner_shown`)."""
    uids: list[int] = [0] if host_uid is None else [0, int(host_uid)]
    give = "0:0"
    if host_owner is not None:
        uids.append(int(host_owner[0]))
        give = f"{int(host_owner[0])}:{int(host_owner[1])}"
    owners = "|".join(dict.fromkeys(str(uid) for uid in uids))
    return _close_and_judge(folder, owners, f'echo "{OPEN} ${{bf_vd:-unknown}}"', give)


def close_host_mount_cmd(probe: str, *, host_uid: int) -> str:
    """sh, as root, where *probe* is a file the host has just made in its bind-mounted ``/logs/verifier``: prints
    the probe's owner as this sandbox shows it (the host user: its uid on a rootful daemon, root on a rootless one) as
    ``BF host-owner <uid>:<gid>``, removes the probe, and closes the folder as :func:`close_verifier_dir_cmd` does,
    one owned by anyone but root, *host_uid* or that owner going to that owner. Neither depends on what the host
    may still do in a folder the task's own start-up code gave to another user."""
    p = shlex.quote(probe)
    return (
        f"bf_ho=$(stat -c %u:%g {p} 2>/dev/null); rm -f {p}; "
        f'case "$bf_ho" in [0-9]*:[0-9]*) echo "{HOST_OWNER} $bf_ho";; *) bf_ho=0:0;; esac; '
        + _close_and_judge(
            VERIFIER_DIR,
            f'0|{int(host_uid)}|"${{bf_ho%%:*}}"',
            f'echo "{OPEN} ${{bf_vd:-unknown}}"',
            '"$bf_ho"',
        )
    )


def host_owner_shown(stdout: object) -> tuple[int, int] | None:
    """The host user's uid and gid as :func:`close_host_mount_cmd` showed them in *stdout*, or None."""
    text = stdout if isinstance(stdout, str) else ""
    for line in text.splitlines():
        if line.startswith(HOST_OWNER + " "):
            uid, _, gid = line[len(HOST_OWNER) + 1 :].strip().partition(":")
            if uid.isdigit() and gid.isdigit():
                return int(uid), int(gid)
    return None


def close_verifier_dir_or_exit_cmd(mark: str) -> str:
    """The same, for a step whose own result line is judged elsewhere (Daytona's daemon-log exec): a folder not
    closed prints ``<mark> verifier-dir-open <uid> <mode>`` and exits 1, before the rest of the step runs."""
    return _close_and_judge(
        VERIFIER_DIR,
        "0",
        f'echo "{mark} verifier-dir-open ${{bf_vd:-unknown}}"; exit 1',
    )


def verifier_dir_problem(stdout: object) -> str | None:
    """Why the setup step found the folder open (its ``BF verifier-dir-open`` line in *stdout*), or None."""
    text = stdout if isinstance(stdout, str) else ""
    for line in text.splitlines():
        if line.startswith(OPEN + " "):
            shown = line[len(OPEN) + 1 :].split()
            if len(shown) != 2:
                return "its owner and mode could not be read"
            return f"it is uid {shown[0]}'s, mode {shown[1]}"
    return None


def check_verifier_dir(stdout: object) -> None:
    """Refuse the trial at sandbox setup when the setup step found ``/logs/verifier`` open to the agent."""
    problem = verifier_dir_problem(stdout)
    if problem is None:
        return
    from benchflow._utils.scoring import INTEGRITY_UNAVAILABLE_HEADER
    from benchflow.rollout._grading_outcome import IntegrityUnavailableError

    raise IntegrityUnavailableError(
        f"{INTEGRITY_UNAVAILABLE_HEADER}{VERIFIER_DIR} could not be closed to the agent at sandbox setup ({problem}): "
        "the agent could write test.sh's outputs folder. No agent code ran; the trial has no score."
    )


async def close_again(
    run: Callable[[str], Awaitable[Any]],
    command: str,
    *,
    first_turn: bool,
    logger: logging.Logger,
) -> None:
    """*command* (a :func:`close_verifier_dir_cmd`), run as root by *run*, up to three times, until it shows the
    folder closed. Still open before the first turn, where only the task's own start-up code has run, the trial is refused
    as at setup. Before a later turn something the agent left (a root process of an earlier turn) can be keeping it
    open: that is warned in the trial's result and the turn goes on; the close before test.sh still holds the
    score."""
    shown = ""
    for _ in range(3):
        shown = getattr(await run(command), "stdout", "") or ""
        if verifier_dir_problem(shown) is None:
            return
    if first_turn:
        check_verifier_dir(shown)
        return
    from benchflow._utils.trial_warnings import report_trial_warning

    text = (
        f"{VERIFIER_DIR} was still open to other users before this turn ({verifier_dir_problem(shown)}) after "
        "three closes: a process left from an earlier turn may be giving it away. The turn went on; the close "
        "before test.sh keeps test.sh's outputs to the verifier."
    )
    logger.warning(text)
    report_trial_warning(text)

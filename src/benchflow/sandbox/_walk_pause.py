"""The agent's processes held still for each link walk after the turn (C2-1).

In integrity off and audit the agent's processes live through grading. A
process that keeps renaming names in a big folder can move an entry from a
part of the folder the walk has not read yet to a part it has, between two
``getdents`` calls: the entry is never listed, nothing vanishes, and a
complete walk misses the link. So for a non-root agent each walk after the
turn (the one before test.sh with its removals, and the one after grading)
runs while every process whose real uid is the agent's is stopped, and they
are continued once it is done. The look that confirms each removal runs
after the resume, so a process that puts its link back at once is seen
(813-M1, probe P5).

The pause is a root shell of builtins over /proc. It stops every such
process but PID 1 and its own chain, pass after pass, until a pass finds
none of them running (each stopped, or a zombie with no thread left), within
:data:`_PASSES` passes. A pause that can't show that leaves the walk
incomplete. The resume first looks again (a process of the agent's running
then means the pause did not hold), then continues the stopped processes in
the reverse of the order it stopped them, each only while its real uid is
still the agent's. A process that was already stopped before the pause is
left stopped.

The record, ``pause`` in each check's record: ``{user, uid, paused_at,
resumed_at, pids, pre, still, held, why}``.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Any

from benchflow.task.verifier_freeze import _FIELD, _FIXED_PATH, _NOW, _SLEEP

logger = logging.getLogger(__name__)

#: Stop passes, 10 ms apart, before the pause gives up on showing every
#: process of the agent's stopped.
_PASSES = 200

#: Why a walk ran without the agent's processes held still.
NOT_PAUSED = "the agent's processes could not be held still for the walk"


@dataclass
class Pause:
    """One pause of the agent's processes, and its resume."""

    user: str
    uid: int | None = None
    paused_at: float | None = None
    resumed_at: float | None = None
    # Stopped by the pause, in the order it stopped them.
    pids: list[int] = field(default_factory=list)
    # Already stopped before the pause: left as they were.
    pre: list[int] = field(default_factory=list)
    # Every process of the agent's stopped before the walk.
    still: bool = False
    # None until the resume: whether every one was still stopped then.
    held: bool | None = None
    why: str | None = None
    # The pause ran but its answer could not be read: the resume continues
    # every stopped process of the agent's (the stop order is unknown).
    unknown: bool = False

    def ok(self) -> bool:
        return self.still and self.held is True

    def to_json(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "uid": self.uid,
            "paused_at": self.paused_at,
            "resumed_at": self.resumed_at,
            "pids": list(self.pids),
            "pre": list(self.pre),
            "still": self.still,
            "held": self.held,
            "why": self.why,
        }


def _chain(proc: str) -> str:
    """Shell: ``_bf_keep`` holds PID 1, this shell and its ancestors."""
    me = _FIELD.format(key="Pid:", var="_bf_me", path=f"{proc}/self/status")
    parent = _FIELD.format(key="PPid:", var="_bf_n", path=f"{proc}/$_bf_a/status")
    return (
        f"_bf_me=; {me}; "
        '_bf_keep=" 1 $_bf_me "; _bf_a=$_bf_me; '
        'while [ -n "$_bf_a" ] && [ "$_bf_a" != 0 ] && [ "$_bf_a" != 1 ]; do '
        f'_bf_n=; {parent}; _bf_keep="$_bf_keep$_bf_n "; _bf_a=$_bf_n; done; '
    )


def _facts(proc: str) -> str:
    """Shell, for ``$_bf_p``: its real uid, State and Threads."""
    path = f"{proc}/$_bf_p/status"
    return (
        "_bf_u=; _bf_s=; _bf_t=; "
        '{ while read -r _bf_k _bf_v _bf_r; do case "$_bf_k" in '
        "Uid:) _bf_u=$_bf_v ;; State:) _bf_s=$_bf_v ;; Threads:) _bf_t=$_bf_v ;; "
        f'esac; done < "{path}"; }} 2>/dev/null; '
    )


def _held() -> str:
    """Shell: ``$_bf_s`` names a process that runs no code: stopped, or a
    zombie with no other thread (a zombie leader's live threads still run)."""
    return (
        'case "$_bf_s" in T|t) _bf_h=1 ;; ""|X) _bf_h=1 ;; '
        'Z) if [ "${_bf_t:-1}" -le 1 ] 2>/dev/null; then _bf_h=1; else _bf_h=0; fi ;; '
        "*) _bf_h=0 ;; esac; "
    )


def _uid_head(uid: int | None, user: str | None, said: str) -> str:
    """Shell: ``_bf_uid`` is *uid*, or root's ``id -u`` of *user* (the
    kill's own read, of files a non-root agent can't write); a uid that
    can't be read, or is root's, prints ``BF <said> nouid`` and ends."""
    if uid is not None:
        return f"_bf_uid={int(uid)}; "
    return (
        f"_bf_uid=$({_FIXED_PATH} id -u {shlex.quote(str(user))} 2>/dev/null); "
        'case "$_bf_uid" in ""|*[!0-9]*|0) '
        f"echo 'BF {said} nouid'; echo 'BF {said} still 0'; exit 0 ;; esac; "
        f'echo "BF {said} uid $_bf_uid"; '
    )


def pause_command(
    uid: int | None, *, user: str | None = None, proc: str = "/proc"
) -> str:
    """The root shell that stops every process whose real uid is *uid*
    (read by ``id -u`` of *user* when *uid* is None), but PID 1 and its own
    chain, until a pass finds none running. Prints ``BF pause`` lines
    (:func:`parse_pause`). *proc* is for tests."""
    return (
        _uid_head(uid, user, "pause")
        + _chain(proc)
        + '_bf_ours=" "; _bf_pre=" "; _bf_i=0; _bf_still=0; _bf_left=; '
        f'while [ "$_bf_still" = 0 ] && [ "$_bf_i" -lt {_PASSES} ]; do '
        "_bf_i=$((_bf_i + 1)); _bf_still=1; _bf_left=; "
        f"for _bf_d in {proc}/[0-9]*; do _bf_p=${{_bf_d##*/}}; "
        'case "$_bf_keep" in *" $_bf_p "*) continue ;; esac; '
        + _facts(proc)
        + '[ "$_bf_u" = "$_bf_uid" ] || continue; '
        + _held()
        + 'if [ "$_bf_h" = 1 ]; then '
        # Stopped, and not by this pause: the agent's own (or a kept
        # service BenchFlow holds), left stopped at the resume.
        'case "$_bf_s" in T|t) case "$_bf_ours$_bf_pre" in *" $_bf_p "*) ;; '
        '*) _bf_pre="$_bf_pre$_bf_p " ;; esac ;; esac; continue; fi; '
        '_bf_still=0; _bf_left="$_bf_left $_bf_p"; '
        'if kill -STOP "$_bf_p" 2>/dev/null; then '
        'case "$_bf_ours" in *" $_bf_p "*) ;; *) _bf_ours="$_bf_ours$_bf_p " ;; esac; fi; '
        "done; "
        f'[ "$_bf_still" = 1 ] || {_SLEEP} 0.01 2>/dev/null || {_SLEEP} 1; '
        "done; "
        # A pid that ended and was taken by another account's process between
        # its read and its stop: continued at once, and never counted.
        '_bf_kept=" "; for _bf_p in $_bf_ours; do '
        + _facts(proc)
        + 'if [ -n "$_bf_s" ] && [ "$_bf_u" != "$_bf_uid" ]; then kill -CONT "$_bf_p" 2>/dev/null; '
        'else _bf_kept="$_bf_kept$_bf_p "; fi; done; '
        'for _bf_p in $_bf_kept; do echo "BF pause pid $_bf_p"; done; '
        'for _bf_p in $_bf_pre; do echo "BF pause pre $_bf_p"; done; '
        'if [ "$_bf_still" = 1 ]; then '
        f"printf 'BF pause at %s\\n' \"$({_NOW})\"; "
        'else echo "BF pause running$_bf_left"; fi; '
        'echo "BF pause still $_bf_still"'
    )


def resume_command(
    uid: int | None,
    pids: list[int] | None,
    *,
    user: str | None = None,
    proc: str = "/proc",
) -> str:
    """The root shell after the walk: a look at every process whose real uid
    is *uid* (one running means the pause did not hold), then SIGCONT to
    *pids* in reverse order, each only while its real uid is still *uid*.
    With *pids* None (the pause's answer was lost), every stopped process of
    *uid*'s is continued. Prints ``BF resume`` lines."""
    if pids is None:
        targets = (
            f"for _bf_d in {proc}/[0-9]*; do _bf_p=${{_bf_d##*/}}; "
            'case "$_bf_keep" in *" $_bf_p "*) continue ;; esac; '
            + _facts(proc)
            + '[ "$_bf_u" = "$_bf_uid" ] || continue; '
            'case "$_bf_s" in T|t) _bf_go="$_bf_p $_bf_go" ;; esac; done; '
        )
    else:
        order = " ".join(str(int(p)) for p in reversed(pids))
        targets = f'_bf_go="{order}"; '
    return (
        _uid_head(uid, user, "resume") + _chain(proc) + "_bf_ran=; _bf_go=; "
        f"for _bf_d in {proc}/[0-9]*; do _bf_p=${{_bf_d##*/}}; "
        'case "$_bf_keep" in *" $_bf_p "*) continue ;; esac; '
        + _facts(proc)
        + '[ "$_bf_u" = "$_bf_uid" ] || continue; '
        + _held()
        + '[ "$_bf_h" = 1 ] || _bf_ran="$_bf_ran $_bf_p"; done; '
        + targets
        + f"printf 'BF resume at %s\\n' \"$({_NOW})\"; "
        "for _bf_p in $_bf_go; do "
        + _facts(proc)
        + 'if [ "$_bf_u" = "$_bf_uid" ] && kill -CONT "$_bf_p" 2>/dev/null; then '
        'echo "BF resume pid $_bf_p"; fi; done; '
        '[ -z "$_bf_ran" ] || echo "BF resume ran$_bf_ran"; '
        'echo "BF resume done"'
    )


def _time(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_pause(found: Pause, stdout: str, return_code: int | None) -> Pause:
    """*found* with the pause's answer read into it."""
    still = None
    for line in (stdout or "").splitlines():
        parts = line.split()
        if parts[:2] != ["BF", "pause"] or len(parts) < 3:
            continue
        key, rest = parts[2], parts[3:]
        if key == "uid" and rest and rest[0].isdigit():
            found.uid = int(rest[0])
        elif key == "nouid":
            found.why = f"{NOT_PAUSED}: the uid of {found.user} could not be read"
            found.still = False
            return found
        elif key == "pid" and rest and rest[0].isdigit():
            found.pids.append(int(rest[0]))
        elif key == "pre" and rest and rest[0].isdigit():
            found.pre.append(int(rest[0]))
        elif key == "at" and rest:
            found.paused_at = _time(rest[0])
        elif key == "running":
            found.why = (
                f"{NOT_PAUSED}: still running after {_PASSES} stop passes "
                f"(pid {', '.join(rest) or '?'})"
            )
        elif key == "still" and rest:
            still = rest[0] == "1"
    if still is None or return_code not in (0, None):
        found.unknown = True
        found.why = f"{NOT_PAUSED}: the pause did not report (rc {return_code})"
        return found
    found.still = still and found.paused_at is not None
    if still and found.paused_at is None:
        found.why = f"{NOT_PAUSED}: the sandbox's clock could not be read"
    return found


def parse_resume(found: Pause, stdout: str, return_code: int | None) -> Pause:
    """*found* with the resume's answer read into it."""
    ran: list[str] = []
    done = False
    for line in (stdout or "").splitlines():
        parts = line.split()
        if parts[:2] != ["BF", "resume"] or len(parts) < 3:
            continue
        key, rest = parts[2], parts[3:]
        if key == "at" and rest:
            found.resumed_at = _time(rest[0])
        elif key == "ran":
            ran.extend(rest)
        elif key == "done":
            done = True
    if not done or return_code not in (0, None):
        found.held = False
        why = f"the resume did not report (rc {return_code})"
    elif ran:
        found.held = False
        why = f"a process of the agent's ran during the walk (pid {', '.join(ran)})"
    else:
        found.held = True
        return found
    found.why = f"{NOT_PAUSED}: {why}" if found.why is None else f"{found.why}; {why}"
    return found


def _uid(env: Any, user: str) -> int | None:
    """*user*'s uid from the identity table read before the agent started,
    where the trial made one; else None, and the pause reads it itself."""
    from benchflow.sandbox import _identities

    table = vars(env).get("_benchflow_identities") if hasattr(env, "__dict__") else None
    if not isinstance(table, _identities.Identities):
        return None
    try:
        return int(_identities.resolve(table, user).uid)
    except (RuntimeError, ValueError, TypeError):
        return None


async def _root(env: Any, command: str, timeout_sec: int) -> tuple[str, int | None]:
    from benchflow.sandbox._exec_output import exec_raw

    result = await exec_raw(
        env.exec, "sh -c " + _quote(command), user="root", timeout_sec=timeout_sec
    )
    return getattr(result, "stdout", "") or "", getattr(result, "return_code", None)


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


async def pause(env: Any, user: str, *, timeout_sec: int) -> Pause:
    """Stop *user*'s processes for a walk (:func:`pause_command`)."""
    found = Pause(user=user, uid=_uid(env, user))
    if found.uid == 0:
        found.why = f"{NOT_PAUSED}: {user} is root"
        return found
    try:
        stdout, code = await _root(
            env, pause_command(found.uid, user=user), timeout_sec
        )
    except Exception as exc:
        found.unknown = True
        found.why = f"{NOT_PAUSED}: the pause failed: {exc}"
        return found
    return parse_pause(found, stdout, code)


async def resume(env: Any, found: Pause, *, timeout_sec: int) -> Pause:
    """Continue what :func:`pause` stopped (:func:`resume_command`)."""
    if found.uid == 0 or (found.uid is None and not found.unknown):
        return found
    command = resume_command(
        found.uid, None if found.unknown else found.pids, user=found.user
    )
    for attempt in (1, 2):
        try:
            stdout, code = await _root(env, command, timeout_sec)
        except Exception as exc:
            if attempt == 2:
                found.held = False
                why = f"the resume failed: {exc}"
                found.why = (
                    f"{NOT_PAUSED}: {why}"
                    if found.why is None
                    else f"{found.why}; {why}"
                )
                logger.warning("the agent's processes may still be stopped: %s", exc)
                return found
            continue
        return parse_resume(found, stdout, code)
    return found

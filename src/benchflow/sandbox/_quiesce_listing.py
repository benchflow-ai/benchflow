"""The processes the quiesce after grading ends, listed by root before and after it.

In integrity audit and off the agent's processes live through grading, and the
rollout ends them after it (``planes.quiesce_agent``): the accounts the task
lets the agent switch to, then the agent user, each frozen, sent SIGTERM and
SIGCONT with up to 1 s to exit, then SIGKILL. The shell kill and ``bfx kill``
do the same and each return one count, not which processes ended or how. So
root lists the accounts' processes from /proc just before the quiesce and
again after it, and the trial's ``kill-after-grading.json`` names them
(:mod:`benchflow._utils.kill_record`).

Each row is a process whose real, effective or saved uid is one of the
accounts' (the kill's own rule; a zombie only while its thread group still
runs): its pid, account, state, command line (at most 200 characters, every
byte outside printable ASCII as ``?``: the process wrote it, and no byte of it
may end the listing's line), start time, and its SIGTERM disposition. The signal that ended it is inferred from
that disposition, not measured (``signal_source``): with the default action
SIGTERM ends it; ignored, or blocked in every thread, it lives until SIGKILL;
a process that handles SIGTERM may exit within the second or not, so either.

The listing is a shell loop over /proc with the verifier's fixed PATH, as the
kill's own walk is, so an image without Python (or bash) is listed too.
"""

from __future__ import annotations

import shlex
from datetime import UTC, datetime
from typing import Any

from benchflow.task.verifier_freeze import _FIXED_PATH

#: ``signal_source`` in the record: how the signal per process was found.
SIGNAL_SOURCE = "sigterm_disposition"
#: The quiesce's signals, in order, on both of its routes.
SIGNALS = (
    "SIGSTOP to freeze, SIGTERM and SIGCONT with up to 1 s to exit, "
    "then SIGKILL to whatever is left"
)
#: The signal inferred from each SIGTERM disposition.
ENDED_BY = {
    "default": "SIGTERM",
    "ignored": "SIGKILL",
    "blocked": "SIGKILL",
    "caught": "SIGTERM or SIGKILL",
}
_COMMAND_CHARS = 200

# Prints "BF clock <btime> <hz>", one "BF proc <pid> <account> <state>
# <start ticks> <disposition> <command>" per process, then "BF listed end".
# SIGTERM is signal 15: bit 0x4000 of the last four hex digits of a mask.
# Start times are in USER_HZ, 100 on Linux (x86_64 and arm64 alike): no
# getconf, so root runs nothing of the image's here. Besides the shell's
# builtins it runs tr and cut (and id for an account the trial has no uid
# for), which the runtime kit carries.
_LIST_BODY = r"""P=__PROC__
btime=
{ while read -r k v r; do [ "$k" = btime ] && { btime=$v; break; }; done < "$P/stat"; } 2>/dev/null
printf 'BF clock %s 100\n' "${btime:-0}"
term() { m=$1; m=${m#${m%????}}; case "$m" in ''|*[!0-9a-fA-F]*) return 1 ;; esac; [ $((0x$m & 0x4000)) -ne 0 ]; }
for entry in "$P"/[0-9]*; do
    pid=${entry##*/}
    name= state= ruid= euid= suid= threads= ign= cgt= who=
    # Another account's process is left at its Uid line: the shell reads a
    # byte at a time, and the signal masks are near the file's end.
    { while read -r k a b c r; do
        case "$k" in
            Name:) name=$a ;;
            State:) state=$a ;;
            Uid:)
                ruid=$a euid=$b suid=$c
                for pair in $accounts; do
                    u=${pair%%:*}
                    if [ "$ruid" = "$u" ] || [ "$euid" = "$u" ] || [ "$suid" = "$u" ]; then who=${pair#*:}; break; fi
                done
                [ -n "$who" ] || break ;;
            Threads:) threads=$a ;;
            SigIgn:) ign=$a ;;
            SigCgt:) cgt=$a ;;
        esac
    done < "$entry/status"; } 2>/dev/null || continue
    [ -n "$who" ] || continue
    if [ "$state" = Z ]; then
        case "$threads" in ''|*[!0-9]*) continue ;; esac
        [ "$threads" -gt 1 ] || continue
    fi
    seen=0 all=1
    for t in "$entry"/task/[0-9]*; do
        blk=
        { while read -r k a r; do [ "$k" = SigBlk: ] && { blk=$a; break; }; done < "$t/status"; } 2>/dev/null
        [ -n "$blk" ] || continue
        seen=$((seen + 1))
        term "$blk" || all=0
    done
    if term "$ign"; then disp=ignored
    elif [ "$seen" -gt 0 ] && [ "$all" = 1 ]; then disp=blocked
    elif term "$cgt"; then disp=caught
    else disp=default
    fi
    # stat holds the comm raw, newlines included (status escapes them): read
    # every line, and split the fields after the last ") " with globbing off,
    # so the comm can't move a field or expand to names in root's cwd.
    stat=
    { while read -r l || [ -n "$l" ]; do stat="$stat $l"; l=; done < "$entry/stat"; } 2>/dev/null
    rest=${stat##*) }
    set -f
    set -- $rest
    set +f
    start=${20:-0}
    case "$start" in *[!0-9]*) start=0 ;; esac
    # The process wrote its command line and name: every byte outside
    # printable ASCII becomes ?, so neither can start a line of its own.
    cmd=$(tr '\000' ' ' < "$entry/cmdline" 2>/dev/null | tr -c ' -~' '?' | cut -c1-__CHARS__)
    cmd=${cmd% }
    [ -n "$cmd" ] || cmd="[$(printf '%s' "$name" | tr -c ' -~' '?')]"
    # printf, never echo: dash's echo (Debian's and Ubuntu's sh) reads the
    # printable \n, \c and \0nnn of a command line as escapes.
    printf 'BF proc %s %s %s %s %s %s\n' "$pid" "$who" "${state:-?}" "$start" "$disp" "$cmd"
done
printf 'BF listed end\n'
"""


def list_command(accounts: list[tuple[str, int | None]], *, proc: str = "/proc") -> str:
    """Root's shell that lists the processes of *accounts* (name, uid; a uid
    of None is read with ``id -u``, root's being left out). *proc* is for
    tests: a stand-in ``/proc``."""
    heads = []
    for name, uid in accounts:
        quoted = shlex.quote(name)
        if uid is not None:
            heads.append(f'accounts="$accounts {int(uid)}:"{quoted}; ')
            continue
        heads.append(
            f"u=$({_FIXED_PATH} id -u {quoted} 2>/dev/null); "
            'case "$u" in ""|*[!0-9]*|0) ;; *) '
            f'accounts="$accounts $u:"{quoted} ;; esac; '
        )
    body = _LIST_BODY.replace("__CHARS__", str(_COMMAND_CHARS)).replace(
        "__PROC__", shlex.quote(proc)
    )
    return (
        f"{_FIXED_PATH}; export PATH; LC_ALL=C; export LC_ALL; accounts=; "
        + "".join(heads)
        + "\n"
        + body
    )


def parse_listing(stdout: str) -> list[dict[str, Any]]:
    """The rows of :func:`list_command`'s output; raises ValueError when the
    listing did not finish (no ``BF listed end``)."""
    # "\n" alone ends a line: the shell side leaves no other control byte,
    # and str.splitlines() would also break at \r, \x1c-\x1e, NEL, U+2028.
    lines = stdout.split("\n")
    if "BF listed end" not in lines:
        raise ValueError("the listing did not finish")
    btime, hz = 0, 100
    rows: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith("BF clock "):
            parts = line.split()
            if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
                btime, hz = int(parts[2]), max(int(parts[3]), 1)
            continue
        if not line.startswith("BF proc "):
            continue
        parts = line.split(" ", 7)
        if len(parts) < 8 or not parts[2].isdigit() or not parts[5].isdigit():
            continue
        _, _, pid, account, state, start, disposition, command = parts
        ticks = int(start)
        rows.append(
            {
                "pid": int(pid),
                "user": account,
                "command": command,
                "state": state,
                "started_at": _iso(btime + ticks / hz) if btime and ticks else None,
                "start_ticks": ticks,
                "sigterm": disposition,
                "ended_by": ENDED_BY.get(disposition),
            }
        )
    return rows


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


async def list_processes(
    env: Any, accounts: list[str], *, timeout_sec: int
) -> list[dict[str, Any]]:
    """The processes of *accounts* in *env* now, as root lists them."""
    from benchflow.sandbox import _walk_pause

    command = list_command([(name, _walk_pause._uid(env, name)) for name in accounts])
    stdout, code = await _walk_pause._root(env, command, timeout_sec)
    if code not in (0, None):
        raise RuntimeError(f"the listing exited with rc={code}")
    return parse_listing(stdout)


def mark_ended(
    before: list[dict[str, Any]] | None, after: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
    """*before*'s rows, each with ``ended``: whether its process (the same pid
    and start) is gone from *after*; None where *after* is unknown. A process
    that did not end has no ``ended_by``."""
    if before is None:
        return None
    alive = None if after is None else {(r["pid"], r["start_ticks"]) for r in after}
    for row in before:
        row["ended"] = (
            None if alive is None else (row["pid"], row["start_ticks"]) not in alive
        )
        if row["ended"] is False:
            row["ended_by"] = None
    return before

"""The sandbox held still from test.sh's exit until its outputs are copied
(Q3, ruled faithful).

In integrity off and audit the agent's processes live through grading (the
faithful default), so a process the agent left could change test.sh's
outputs between test.sh's exit and the verifier's copy of them
(``verifier_outputs.take``). So the test command's own shell, right after
test.sh returns and before it records test.sh's exit status, stops every
process in the sandbox but PID 1 and itself (and its ancestors): the
agent's, test.sh's leftovers, the services a task keeps and BenchFlow's own
(the 03:38Z ruling allows them all to be stopped after test.sh's exit). It
stops them pass after pass until a pass finds none new, waits until each is
stopped, and keeps each one's per-thread context-switch counts in a state
file only root writes. Once the copy is taken, a root exec
(:func:`thaw_command`) reads the counts again and continues every process
the stop stopped, never one that was stopped before it.

The record, ``freeze`` in the trial's provenance:
``{frozen_at, thawed_at, pids, ok, why}``. ``ok`` is true only when every
process was stopped and none of them ran while it was meant to be still:
each is still stopped, with the same counts and threads. A failed freeze
makes only a root agent's reward ``unverifiable``; a non-root agent is
unaffected, since it can't write the folder.

Residual: the microseconds between test.sh being reaped by the test
command's shell and the first stop. Only bfx as test.sh's parent, stopping
before it reaps, closes that.
"""

from __future__ import annotations

import shlex
from typing import Any
from uuid import uuid4

from benchflow._runtime_tree import KIT, KIT_BIN

#: The stop runs as root in test.sh's shell, whose PATH is the image's: a
#: folder on it the agent can write could hold a planted ``sleep`` or ``date``
#: They are looked up on a fixed PATH instead: the protected kit
#: where the shell sees it, then the system folders (in audit and off test.sh's
#: shell does not see the kit).
_FIXED_PATH = f"PATH={KIT_BIN}:/usr/bin:/bin"
_SLEEP = f"{_FIXED_PATH} sleep"
_DATE = f"{_FIXED_PATH} date"

#: Where the state file goes: a folder root owns in every image we run.
STATE_DIR = "/run"
#: Stop passes before the stop gives up on a sandbox that keeps starting
#: processes; checks (10 ms apart) before a stop that never took effect is
#: given up.
_PASSES = 64
_SETTLE_CHECKS = 200

# Shell-only reads of /proc (no process started while the stop runs): a
# field of /proc/<pid>/status, and a process's summed context switches and
# thread count. Every name starts with _bf_ so test.sh's variables are
# never touched.
_FIELD = (
    '{{ while read -r _bf_k _bf_v _bf_r; do case "$_bf_k" in {key}) '
    '{var}=$_bf_v; break ;; esac; done < "{path}"; }} 2>/dev/null'
)
_COUNT = (
    "_bf_c=0; _bf_n=0; "
    'for _bf_t in /proc/"$_bf_p"/task/[0-9]*; do '
    '[ -e "$_bf_t/status" ] || continue; _bf_n=$((_bf_n + 1)); '
    '{ while read -r _bf_k _bf_v _bf_r; do case "$_bf_k" in '
    "voluntary_ctxt_switches:|nonvoluntary_ctxt_switches:) "
    '_bf_c=$((_bf_c + _bf_v)) ;; esac; done < "$_bf_t/status"; } 2>/dev/null; '
    "done"
)

# The sandbox's clock: seconds with nanoseconds where its date has them,
# else whole seconds (BusyBox builds without %N).
_NOW = (
    f"_bf_ts=$({_DATE} +%s.%N 2>/dev/null); "
    f'case "$_bf_ts" in ""|*[!0-9.]*) _bf_ts=$({_DATE} +%s 2>/dev/null) ;; esac; '
    'echo "$_bf_ts"'
)


def state_path() -> str:
    """A fresh state file's path for one test command."""
    return f"{STATE_DIR}/bf-freeze-{uuid4().hex}"


def freeze_step(state: str, *, kit_channels: bool = False) -> str:
    """The shell the test command runs right after test.sh returns (before
    its exit record): stop everything but PID 1 and this shell's own chain,
    and keep the counts in *state*. It never changes ``$?``'s saved value
    (``_benchflow_status``) and prints nothing.

    *kit_channels*: bfx is not delivered, so BenchFlow's exec channels are
    the kit's Python helper or pre-started execs waiting on the kit's
    ``head`` (``sandbox/_exec_channel.py``), and those are left running too
    (the rc4 dogfood, 11-17: stopped, they held the thaw and every later
    root step until the trial timed out)."""
    f = shlex.quote(state)
    ours = "/.bfx/bfx|/.bfx/bfxserve*"
    if kit_channels:
        ours += f"|*{KIT}/libexec/python3-I-S-B-c*|*__bf_warm_n*"
    me = _FIELD.format(key="Pid:", var="_bf_me", path="/proc/self/status")
    parent = _FIELD.format(key="PPid:", var="_bf_n", path="/proc/$_bf_a/status")
    status = _FIELD.format(key="State:", var="_bf_s", path="$_bf_d/status")
    status_p = _FIELD.format(key="State:", var="_bf_s", path="/proc/$_bf_p/status")
    return (
        f"if [ ! -e {f}.cancel ]; then "
        f"_bf_me=; {me}; "
        '_bf_keep=" 1 $_bf_me "; _bf_a=$_bf_me; '
        'while [ -n "$_bf_a" ] && [ "$_bf_a" != 0 ] && [ "$_bf_a" != 1 ]; do '
        f'_bf_n=; {parent}; _bf_keep="$_bf_keep$_bf_n "; _bf_a=$_bf_n; done; '
        '_bf_ours=" "; _bf_pre=" "; _bf_bad=" "; _bf_i=0; _bf_new=1; '
        f'while [ "$_bf_new" = 1 ] && [ "$_bf_i" -lt {_PASSES} ]; do '
        "_bf_new=0; _bf_i=$((_bf_i + 1)); "
        "for _bf_d in /proc/[0-9]*; do _bf_p=${_bf_d#/proc/}; "
        'case "$_bf_keep$_bf_ours$_bf_pre$_bf_bad" in *" $_bf_p "*) continue ;; esac; '
        # BenchFlow's own exec channels (``/.bfx/bfx serve``) carry the thaw
        # and every later root step: stopping one would leave the thaw
        # waiting on it. Told by argv[0], read by the shell itself (root may
        # not read their /proc/<pid>/exe); a process named so to slip the stop
        # can only be a root agent's, whose trial is unverifiable anyway.
        '_bf_c=; IFS= read -r _bf_c < "$_bf_d/cmdline" 2>/dev/null; '
        f'case "$_bf_c" in {ours}) continue ;; esac; '
        f"_bf_s=; {status}; "
        'case "$_bf_s" in ""|Z|X) continue ;; T|t) _bf_pre="$_bf_pre$_bf_p "; continue ;; esac; '
        'if kill -STOP "$_bf_p" 2>/dev/null; then _bf_ours="$_bf_ours$_bf_p "; _bf_new=1; '
        'elif [ -e "$_bf_d" ]; then _bf_bad="$_bf_bad$_bf_p "; fi; '
        "done; done; "
        "_bf_j=0; "
        f'while [ "$_bf_j" -lt {_SETTLE_CHECKS} ]; do _bf_run=0; '
        f"for _bf_p in $_bf_ours; do _bf_s=; {status_p}; "
        'case "$_bf_s" in ""|T|t|Z|X) ;; *) _bf_run=1 ;; esac; done; '
        '[ "$_bf_run" = 0 ] && break; _bf_j=$((_bf_j + 1)); '
        f"{_SLEEP} 0.01 2>/dev/null || {_SLEEP} 1; done; "
        f"{_SLEEP} 0.05 2>/dev/null || {_SLEEP} 1; "
        f"if ( set -C; : > {f} ) 2>/dev/null; then {{ "
        f'[ "$_bf_new" = 1 ] && echo unsettled; '
        f'[ "$_bf_j" -ge {_SETTLE_CHECKS} ] && echo running; '
        'for _bf_p in $_bf_bad; do echo "unstopped $_bf_p"; done; '
        f'for _bf_p in $_bf_ours; do {_COUNT}; echo "ours $_bf_p $_bf_c $_bf_n"; done; '
        f'for _bf_p in $_bf_pre; do {_COUNT}; echo "pre $_bf_p $_bf_c $_bf_n"; done; '
        f"printf 'frozen_at %s\\n' \"$({_NOW})\"; "
        f"}} >> {f} 2>/dev/null; fi; "
        "fi; "
    )


def thaw_command(state: str) -> str:
    """The root exec after the copy: no later stop for *state*, the counts
    checked, every process the stop stopped continued. Prints ``BF freeze``
    lines (:func:`parse`)."""
    f = shlex.quote(state)
    status_p = _FIELD.format(key="State:", var="_bf_s", path="/proc/$_bf_p/status")
    return (
        f": > {f}.cancel 2>/dev/null; "
        f'if [ ! -f {f} ] || [ -h {f} ]; then echo "BF freeze none"; exit 0; fi; '
        "_bf_ok=1; "
        "while read -r _bf_kind _bf_p _bf_c0 _bf_n0; do "
        'case "$_bf_kind" in '
        'frozen_at) echo "BF freeze frozen_at $_bf_p" ;; '
        'unsettled|running) echo "BF freeze why $_bf_kind"; _bf_ok=0 ;; '
        'unstopped) echo "BF freeze unstopped $_bf_p"; _bf_ok=0 ;; '
        f"ours|pre) _bf_s=; {status_p}; {_COUNT}; "
        'case "$_bf_s" in ""|Z|X) echo "BF freeze gone $_bf_p"; _bf_ok=0; continue ;; esac; '
        '[ "$_bf_kind" = ours ] && echo "BF freeze pid $_bf_p"; '
        'case "$_bf_s" in T|t) ;; *) echo "BF freeze running $_bf_p"; _bf_ok=0 ;; esac; '
        'if [ "$_bf_c" != "$_bf_c0" ] || [ "$_bf_n" != "$_bf_n0" ]; then '
        'echo "BF freeze ran $_bf_p"; _bf_ok=0; fi ;; '
        f"esac; done < {f}; "
        f"printf 'BF freeze thawed_at %s\\n' \"$({_NOW})\"; "
        "while read -r _bf_kind _bf_p _bf_r; do "
        '[ "$_bf_kind" = ours ] && kill -CONT "$_bf_p" 2>/dev/null; '
        f"done < {f}; "
        'echo "BF freeze ok $_bf_ok"; '
        f"rm -f {f}"
    )


def _time(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def not_made(why: str) -> dict[str, Any]:
    """The record of a freeze that was not made, with why."""
    return {
        "frozen_at": None,
        "thawed_at": None,
        "pids": [],
        "ok": False,
        "why": why,
    }


def parse(stdout: str, return_code: int | None) -> dict[str, Any]:
    """The freeze record from :func:`thaw_command`'s output."""
    if return_code != 0:
        return not_made(f"the thaw exited {return_code}")
    frozen_at = thawed_at = None
    pids: list[int] = []
    problems: list[str] = []
    ok: bool | None = None
    for line in stdout.splitlines():
        parts = line.split()
        if parts[:2] != ["BF", "freeze"] or len(parts) < 3:
            continue
        key, rest = parts[2], parts[3:]
        if key == "none":
            return not_made(
                "the test command did not stop the sandbox's processes (it did "
                "not reach test.sh's exit, or the stop was cancelled)"
            )
        if key == "frozen_at" and rest:
            frozen_at = _time(rest[0])
        elif key == "thawed_at" and rest:
            thawed_at = _time(rest[0])
        elif key == "pid" and rest and rest[0].isdigit():
            pids.append(int(rest[0]))
        elif key == "ok" and rest:
            ok = rest[0] == "1"
        elif key == "why" and rest:
            problems.append(
                {
                    "unsettled": "processes kept starting while they were being stopped",
                    "running": "a process did not stop",
                }.get(rest[0], rest[0])
            )
        elif key in ("unstopped", "gone", "running", "ran") and rest:
            problems.append(
                {
                    "unstopped": "could not be stopped",
                    "gone": "ended while it was meant to be stopped",
                    "running": "was running at the thaw",
                    "ran": "ran while it was meant to be stopped",
                }[key]
                + f" (pid {rest[0]})"
            )
    if ok is None:
        return not_made("the thaw did not report")
    if frozen_at is None or thawed_at is None:
        problems.append("the sandbox's clock could not be read")
        ok = False
    return {
        "frozen_at": frozen_at,
        "thawed_at": thawed_at,
        "pids": pids,
        "ok": bool(ok) and not problems,
        "why": "; ".join(problems) or None,
    }

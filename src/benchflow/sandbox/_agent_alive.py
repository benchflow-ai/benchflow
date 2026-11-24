"""Rule S: whether a process of the agent's was alive when test.sh started.

In integrity off and audit the agent's processes live through grading. One
that is alive when test.sh starts can act while it runs: make a link once
test.sh runs and remove it after the write (C3-3), or outlast the walk's
pause, continued by its own timer (C3-1) or changing its pid faster than a
/proc listing reads it (C3-2). Neither walk can see a link that isn't there
while it runs. So for a non-root agent, once the check before test.sh is
done, root asks the kernel whether any thread whose real uid is the agent's
is alive in the sandbox: ``getpriority(PRIO_USER, uid)``, or
``ioprio_get(IOPRIO_WHO_USER, uid)`` through util-linux's ``ionice -u``. The
kernel walks its whole task list for either and answers ESRCH when it finds
none; a zombie counts until it is reaped. A new task joins the list's tail
and its parent leaves the list only once reaped, so a pid hopper is always
found, and a stopped or self-continuing process is alive either way. With
none alive, no process of the agent's runs during test.sh: only root's
could start one.

``kill(-1, 0)`` sent as the account can't answer this on Linux: the kernel
counts root's processes as recipients, and returns 0 whenever any other
process exists in the sandbox. gVisor (Modal) stubs ``getpriority(PRIO_USER)``
(the same value for every uid, so its controls fail), but its ``kill(-1)``
skips every target the sender may not signal and answers ESRCH when none is
left. So on gVisor alone root also asks that, from a child that has become
the account (the account's real uid: any of its processes, stopped ones
included, may be signalled by it), with the same two controls made the
kill's way: a throwaway uid that holds one live process of root's making
must be found, and a uid no account holds must not be.

Each answer comes with two controls from the same reader: root (the caller
itself, always found) and a uid no account holds (never found). An answer
whose controls fail (a kernel that stubs the call, as gVisor does) is not
used. The kernel's walk can't leave PID 1 out, so a PID 1 of the agent's
uid leaves the question unanswered. Unanswered, the reward's provenance
can't read ``verified`` (:mod:`benchflow.rollout._reward_provenance`).

The record, ``alive_at_test`` in the provenance: ``{user, uid, asked_at,
alive, by, why, pids}``; ``alive`` is None when the question went
unanswered. ``pids`` (at most 20, each ``{pid, state}``, a zombie's state
``Z``) is a /proc listing made after the answer, for the reader: it can
miss a pid hopper, so it decides nothing.
"""

from __future__ import annotations

import shlex
from typing import Any

from benchflow._root_python import root_python_sh
from benchflow.sandbox._walk_pause import _facts
from benchflow.task.verifier_freeze import _FIELD, _FIXED_PATH, _NOW

#: The check's name in the provenance record.
CHECK = "agent-processes-at-test"
#: The reason a reward is ``unverifiable`` when one was alive.
ALIVE = (
    "the agent's processes were still running when test.sh ran; BenchFlow "
    "can't confirm test.sh wrote this reward"
)
#: Why the question went unanswered (the head of ``why``).
UNANSWERED = (
    "whether the agent's processes were running when test.sh ran could not be told"
)
#: A uid no account holds: the control that must answer "none". ionice reads
#: its uid as a signed 32-bit number.
NO_ONE = 2147483646
#: The agent's processes listed in the record, at most.
_LISTED = 20
#: The kernel's nice values; a value outside them is no answer.
_NICE = range(-20, 20)

_ASK_PY = (
    "import errno,os,sys\n"
    "def a(u):\n"
    " try:\n"
    "  v=os.getpriority(os.PRIO_USER,u)\n"
    " except OSError as e:\n"
    "  return 'none' if e.errno==errno.ESRCH else 'error:%d'%e.errno\n"
    " return 'found:%d'%v\n"
    f"print('BF alive by getpriority',a(int(sys.argv[1])),a(0),a({NO_ONE}))\n"
)


#: The uid of the throwaway process the kill reader's found-control signals.
_THROWAWAY = NO_ONE - 1

_KILL_PY = (
    "import errno,os,sys\n"
    "def a(u):\n"
    " r,w=os.pipe();p=os.fork()\n"
    " if p==0:\n"
    "  os.close(r);o='error:setuid'\n"
    "  try:\n"
    "   os.setgroups([]);os.setgid(u);os.setuid(u)\n"
    "   try:\n"
    "    os.kill(-1,0);o='found:0'\n"
    "   except OSError as e:\n"
    "    o='none' if e.errno==errno.ESRCH else 'error:%d'%e.errno\n"
    "  finally:\n"
    "   os.write(w,o.encode());os._exit(0)\n"
    " os.close(w);d=os.read(r,64);os.close(r);os.waitpid(p,0)\n"
    " return d.decode() or 'error:empty'\n"
    "up,uw=os.pipe();hr,hw=os.pipe();s=os.fork()\n"
    "if s==0:\n"
    " os.close(up);os.close(hw)\n"
    " try:\n"
    f"  os.setgroups([]);os.setgid({_THROWAWAY});os.setuid({_THROWAWAY});os.write(uw,b'1')\n"
    "  os.read(hr,1)\n"
    " finally:\n"
    "  os._exit(0)\n"
    "os.close(uw);os.close(hr)\n"
    "if os.read(up,1)==b'1':\n"
    f" t=a({_THROWAWAY})\n"
    "else:\n"
    " t='error:control'\n"
    "v=a(int(sys.argv[1]));n=a(" + str(NO_ONE) + ")\n"
    "os.close(hw);os.waitpid(s,0)\n"
    "print('BF alive by kill',v,t,n)\n"
)


def alive_command(
    uid: int | None,
    *,
    user: str | None = None,
    proc: str = "/proc",
    path: str = _FIXED_PATH,
    kit_python: str | None = None,
) -> str:
    """The root shell that asks the kernel whether a thread of *uid*'s
    (read by ``id -u`` of *user* when *uid* is None) is alive. Prints ``BF
    alive`` lines (:func:`parse_alive`). *proc*, *path* and *kit_python* (an
    interpreter standing in for the kit's, in a sandbox with no kit) are for
    tests.

    The Python readers start through root's entry point
    (:func:`benchflow._root_python.root_python_sh`, ``-I -S -B``), on the
    kit's Python alone: a copy made before the agent's turn, where the agent
    can't write. Never the image's own ``python3``: by the time S asks, the
    agent may have written into its standard library (a world-writable
    ``encodings/__init__.py`` ran as root and forged a ``verified`` 1.0 in
    review, L-1). Without a kit Python, ``ionice`` answers, or nothing does
    (``unverifiable``)."""
    if uid is not None:
        head = f"_bf_uid={int(uid)}; "
    else:
        head = (
            f"_bf_uid=$({path} id -u {shlex.quote(str(user))} 2>/dev/null); "
            'case "$_bf_uid" in ""|*[!0-9]*|0) echo "BF alive nouid"; exit 0 ;; esac; '
        )
    pid1 = _FIELD.format(key="Uid:", var="_bf_1", path=f"{proc}/1/status")
    # No interpreter outside the kit: exec of /dev/null fails, prints nothing,
    # and the reader is not counted.
    no_kit = shlex.quote(kit_python) if kit_python else "/dev/null"

    def python(script: str) -> str:
        # Counted as a reader only when it answered: a kit without a Python
        # refuses, and a sandbox may have no python3 at all. ("$( (": not an
        # arithmetic "$((".)
        run = root_python_sh(script, raw_args='"$_bf_uid"', fallback=no_kit)
        return (
            f"_bf_o=$( {run} 2>/dev/null); "
            'if [ -n "$_bf_o" ]; then echo "$_bf_o"; _bf_r=1; fi; '
        )

    ionice = (
        # util-linux's ionice: ioprio_get(IOPRIO_WHO_USER), the same walk.
        f"if {path} command -v ionice >/dev/null 2>&1; then "
        "_bf_io() { _bf_o=$(LC_ALL=C " + path + ' ionice -u "$1" 2>&1); _bf_rc=$?; '
        'if [ "$_bf_rc" = 0 ] && [ -n "$_bf_o" ]; then echo found:0; '
        'elif [ "$_bf_rc" != 0 ]; then case "$_bf_o" in *"No such process"*) echo none ;; '
        '*) echo "error:$_bf_rc" ;; esac; else echo error:empty; fi; }; '
        f'echo "BF alive by ionice $(_bf_io "$_bf_uid") $(_bf_io 0) $(_bf_io {NO_ONE})"; '
        "_bf_r=1; fi; "
    )
    # gVisor alone (its kernel names itself in /proc/version, which no
    # process in the sandbox can change): the kill reader, after the others,
    # so a kernel whose getpriority answers is never asked this way.
    readers = (
        python(_ASK_PY)
        + ionice
        + 'if [ -n "$_bf_gv" ]; then '
        + python(_KILL_PY)
        + "fi; "
    )
    # Which processes, after the answer and for the record only: a /proc
    # listing can miss a hopper, so it never decides anything.
    listing = (
        f"_bf_n=0; for _bf_d in {proc}/[0-9]*; do _bf_p=${{_bf_d##*/}}; "
        + _facts(proc)
        + '[ "$_bf_u" = "$_bf_uid" ] || continue; _bf_n=$((_bf_n + 1)); '
        f'[ "$_bf_n" -le {_LISTED} ] || break; echo "BF alive pid $_bf_p ${{_bf_s:-?}}"; done; '
    )
    return (
        head
        + 'echo "BF alive uid $_bf_uid"; '
        + f"_bf_1=; {pid1}; "
        # PID 1 is in the kernel's walk: of the agent's uid, it would answer
        # for the agent.
        + 'if [ -z "$_bf_1" ]; then echo "BF alive nopid1"; exit 0; fi; '
        + 'if [ "$_bf_1" = "$_bf_uid" ]; then echo "BF alive pid1"; exit 0; fi; '
        + f"printf 'BF alive at %s\\n' \"$({_NOW})\"; "
        + f"_bf_gv=; _bf_v=; read -r _bf_v 2>/dev/null < {proc}/version; "
        + 'case "$_bf_v" in *gvisor*) _bf_gv=1 ;; esac; '
        + "_bf_r=; "
        + readers
        + '[ -n "$_bf_r" ] || echo "BF alive noreader"; '
        + listing
        + 'echo "BF alive done"'
    )


def _time(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _found(word: str) -> int | None:
    """The nice value of a ``found:<n>`` answer, else None."""
    if not word.startswith("found:"):
        return None
    try:
        value = int(word.split(":", 1)[1])
    except ValueError:
        return None
    return value if value in _NICE else None


def parse_alive(
    found: dict[str, Any], stdout: str, return_code: int | None
) -> dict[str, Any]:
    """*found* with the answer read into it."""
    why = None
    failed: list[str] = []
    for line in (stdout or "").splitlines():
        parts = line.split()
        if parts[:2] != ["BF", "alive"] or len(parts) < 3:
            continue
        key, rest = parts[2], parts[3:]
        if key == "uid" and rest and rest[0].isdigit():
            found["uid"] = int(rest[0])
        elif key == "nouid":
            why = f"the uid of {found.get('user')} could not be read"
        elif key == "nopid1":
            why = "PID 1's uid could not be read"
        elif key == "pid1":
            why = (
                "PID 1 runs as the agent's uid, and the kernel's answer would count it"
            )
        elif key == "noreader":
            why = "the sandbox has no program that can ask the kernel (BenchFlow's Python or ionice)"
        elif key == "at" and rest:
            found["asked_at"] = _time(rest[0])
        elif key == "pid" and len(rest) == 2 and rest[0].isdigit():
            found.setdefault("pids", []).append({"pid": int(rest[0]), "state": rest[1]})
        elif key == "by" and len(rest) == 4 and found.get("alive") is None:
            # Readers in order; the first whose controls pass answers.
            by, answer, root, no_one = rest
            if _found(root) is None or no_one != "none":
                failed.append(
                    f"{by}'s controls failed (root: {root}, a uid no account "
                    f"holds: {no_one}), so its answer can't be trusted"
                )
            elif answer == "none":
                found["alive"], found["by"] = False, by
            elif _found(answer) is not None:
                found["alive"], found["by"] = True, by
            else:
                failed.append(f"{by} answered {answer}")
    if found.get("alive") is None and why is None:
        why = "; ".join(failed) or f"the question did not report (rc {return_code})"
    if return_code not in (0, None) and found.get("alive") is not None:
        found["alive"] = None
        why = f"the question did not report (rc {return_code})"
    if why is not None:
        found["alive"] = None
        found["why"] = f"{UNANSWERED}: {why}"
    return found


async def ask(env: Any, user: str, *, timeout_sec: int) -> dict[str, Any]:
    """Whether a process of *user*'s is alive in *env* now (rule S)."""
    from benchflow.sandbox import _walk_pause

    found: dict[str, Any] = {
        "user": user,
        "uid": _walk_pause._uid(env, user),
        "asked_at": None,
        "alive": None,
        "by": None,
        "why": None,
    }
    if found["uid"] == 0:
        found["why"] = f"{UNANSWERED}: {user} is root"
        return found
    command = alive_command(found["uid"], user=user)
    try:
        stdout, code = await _walk_pause._root(env, command, timeout_sec)
    except Exception as exc:
        found["why"] = f"{UNANSWERED}: the question failed: {exc}"
        return found
    return parse_alive(found, stdout, code)

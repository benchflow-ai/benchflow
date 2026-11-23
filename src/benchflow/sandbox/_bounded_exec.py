"""Root's execs in main run bounded once the agent's turn has started.

From the agent's first process on, a root agent may have replaced any program
of the task image: RCI's run on #234 put wrappers in place of ``cat``, ``rm``,
``mkdir``, ``chmod`` and the rest, and every root step BenchFlow ran in main
afterwards (the proxy stop, the log publish, hardening, the CA trust shell,
the test upload) ran them with the container's full capabilities. So from
then on every root exec BenchFlow sends to main starts through the bounded
launch at the exec layer, before any program of the image runs:

- where bfx is delivered (``/.bfx/bfx``, read-only), the exec's first program
  is ``bfx launch --``: NET_ADMIN, NET_RAW and SYS_PTRACE leave the bounding
  set, the inheritable and ambient sets are emptied and no_new_privs is set,
  then the route's shell replaces it (:data:`BFX_LAUNCH`);
- elsewhere the route's shell starts a new one through util-linux
  ``setpriv`` (the kit's on the kit, else the system folders') or the kit
  Python's same drop (:func:`shell_launch`), and refuses (exit 97) with
  neither. Without bfx the outer shell and the setpriv it finds are the
  image's for a root agent: only bfx holds against one.

Two points of the trial decide what is bounded:

- main's first ingress (``benchflow.trust.state.admit``: the agent's launch,
  the oracle's, a script agent's, an SDK exec, a session factory, an
  agent-derived upload, a restore or branch from a snapshot), when the trust
  ledger marks main TAINTED: every root route but the task's, so all of
  BenchFlow's own root plumbing, from here on;
- :func:`end_agent_turn`, the turn is over (hardening, verify, clean-up): the
  task's own commands as root too (test.sh's launcher and its env preamble,
  the criteria runner, collect hooks). During the turn the task route runs
  the agent's own command (``solve.sh``), which is not BenchFlow's step. On a
  multi-step task the oracle's later steps follow a soft verify and so run
  bounded as well.

Only where BenchFlow's helper adds every rule of main's namespace
(``rules_by_helper``): there main holds neither NET_ADMIN nor NET_RAW and no
step of BenchFlow's in main needs them; the in-main egress proxy is unmarked.
With the helper off (``BENCHFLOW_METADATA_HELPER=0``,
``BENCHFLOW_MAIN_NET_ADMIN=1``, a recovery start on an old baseline)
BenchFlow still adds and removes main's rules from main, and nothing is
bounded here.

A bounded root process lacks capabilities PID 1 may hold (NET_RAW in Docker's
default set wherever the helper does not serve, SYS_PTRACE a task's compose
adds), so it cannot read ``/proc/1/environ``: the task route's copy of PID 1's
loader values is then empty and the inert bootstrap's receipt supplies them.
"""

from __future__ import annotations

import contextlib
import json
import shlex
from typing import Any

from benchflow._bfx.locate import SANDBOX_PATH

#: The agent's turn is over: the task's own commands as root are bounded too.
TURN_OVER = 2

#: The sandbox attribute that holds the level. It survives a new container
#: generation: a snapshot of the agent's sandbox keeps what the agent wrote.
_LEVEL = "_benchflow_root_exec_bound"

#: What goes before the route's shell in a bounded exec where bfx is delivered.
BFX_LAUNCH: tuple[str, ...] = (SANDBOX_PATH, "launch", "--")

#: The exit status of a launch that cannot hold the command
#: (``verifier_network.VERIFIER_LAUNCH_REFUSED``).
REFUSED = 97

_SETPRIV = ("/usr/bin/setpriv", "/bin/setpriv", "/usr/sbin/setpriv", "/sbin/setpriv")

# The kit Python's form of bfx launch (verifier_network._DROP_PY's drop), then
# argv[2] -c argv[1]: the route's shell, never one the image's PATH picks.
_DROP_EXEC_PY = (
    "import ctypes,os,sys;l=ctypes.CDLL(None,use_errno=True);"
    "r=os.getuid()==0 or os.geteuid()==0;"
    "[sys.exit(97) for c in ((12,13,19) if r else ()) if l.prctl(24,c,0,0,0)];"
    "l.prctl(47,4,0,0,0);"
    "h=(ctypes.c_uint32*2)(0x20080522,0);d=(ctypes.c_uint32*6)();"
    "l.capget(h,d);d[2]=d[5]=0;l.capset(h,d) and sys.exit(97);"
    "l.prctl(38,1,0,0,0) and sys.exit(97);"
    "os.execv(sys.argv[2],[sys.argv[2],'-c',sys.argv[1]])"
)


def _raise(env: Any, level: int) -> None:
    if env is None:
        return
    current = getattr(env, _LEVEL, 0)
    if isinstance(current, int) and current >= level:
        return
    with contextlib.suppress(AttributeError):
        setattr(env, _LEVEL, level)


def end_agent_turn(env: Any) -> None:
    """The agent's turn in *env* is over: every root exec into main runs
    bounded, the task's own commands as root included."""
    _raise(env, TURN_OVER)


def turn_over(env: Any) -> bool:
    """Whether :func:`end_agent_turn` was called for *env*."""
    value = getattr(env, _LEVEL, 0)
    return isinstance(value, int) and value >= TURN_OVER


def main_tainted(env: Any) -> bool:
    """Whether the trust ledger holds an admitted ingress into main."""
    from benchflow.trust.state import tainted

    return tainted(env)


def bounds(env: Any, route: Any, service: str = "main") -> bool:
    """Whether a root exec on *route* into *service* starts bounded."""
    if service != "main" or not getattr(route, "root", False):
        return False
    if getattr(route, "own_vm", False):
        return False
    if getattr(env, "rules_by_helper", False) is not True:
        return False
    if turn_over(env):
        return True
    return getattr(route, "pin", None) is not None and main_tainted(env)


def launch_for(env: Any, route: Any, service: str = "main") -> tuple[str, ...] | None:
    """None when the exec is not bounded; else what goes before the route's
    shell in its argv: :data:`BFX_LAUNCH` where bfx is delivered, or ``()``
    when the command must start the launch itself (:func:`shell_launch`)."""
    if not bounds(env, route, service):
        return None
    return BFX_LAUNCH if bfx_delivered(env) else ()


def shell_launch(shell: str, command: str, *, kit: bool = False) -> str:
    """Shell text for the route's shell: *command* in a new *shell* started
    bounded, as ``bfx launch`` would start it, through util-linux ``setpriv``
    with ``--bounding-set`` (the kit's on the kit, else the first in the system
    folders) or the kit Python; refuses (exit 97) with neither. A process that
    is not root keeps its (empty) capabilities and gets no_new_privs."""
    from benchflow._runtime_tree import KIT_BIN, KIT_PYTHON

    candidates = (f"{KIT_BIN}/setpriv",) if kit else ()
    candidates += _SETPRIV
    q = shlex.quote
    return (
        f"( bf_in={q(command)}; bf_sh={q(shell)}; "
        "bf_uid=; while read -r bf_k bf_v bf_rest; do "
        '[ "$bf_k" = Uid: ] && bf_uid=$bf_v; done < /proc/self/status; '
        f"bf_sp=; for bf_c in {' '.join(q(c) for c in candidates)}; do "
        '[ -x "$bf_c" ] && { bf_sp=$bf_c; break; }; done; '
        'bf_help=; [ -n "$bf_sp" ] && bf_help=$("$bf_sp" --help 2>&1); '
        "case $bf_help in *bounding-set*) ;; *) "
        f"[ -x {q(KIT_PYTHON)} ] && exec {q(KIT_PYTHON)} -I -S -B -c "
        f'{q(_DROP_EXEC_PY)} "$bf_in" "$bf_sh"; '
        "echo 'benchflow: root step refused: once the agent turn has started a "
        "root step in main starts only through the BenchFlow bfx, util-linux "
        "setpriv (with --bounding-set) or the kit Python, and this sandbox has "
        "none of them' >&2; "
        f"exit {REFUSED};; esac; "
        'if [ "$bf_uid" = 0 ]; then exec "$bf_sp" --no-new-privs '
        "--bounding-set=-net_admin,-net_raw,-sys_ptrace --inh-caps=-all "
        '--ambient-caps=-all -- "$bf_sh" -c "$bf_in"; '
        'else exec "$bf_sp" --no-new-privs -- "$bf_sh" -c "$bf_in"; fi )'
    )


def bfx_delivered(env: Any) -> bool:
    """Whether BenchFlow's pinned bfx is at ``/.bfx/bfx`` in *env*'s main
    for root's own scripts: delivered there, on a backend whose route is in
    its scripts (``bfx_in_scripts``; a route that serves the walker alone is
    not)."""
    return (
        getattr(env, "bfx_delivery", None) in ("bind", "volume", "upload")
        and getattr(env, "bfx_in_scripts", True) is not False
    )


def bfx_sh(cmd: str, path: str, **args: Any) -> str:
    """Shell text for one bfx file command (``mkdir``, ``chmod``,
    ``remove``) on the absolute *path*, by bfx's absolute path: no program
    of the image runs for it, and bfx follows no link. Its answer goes to
    stderr; its exit status is the command's (0 done, 1 refused)."""
    if not path.startswith("/") or path.rstrip("/") == "":
        raise ValueError(f"bfx_sh needs an absolute path beneath /, not {path!r}")
    body = {"root": "/", "path": path.strip("/"), **args}
    text = json.dumps(body, separators=(",", ":"), sort_keys=True)
    return f"{SANDBOX_PATH} {cmd} --json {shlex.quote(text)} >&2"

"""Structural evidence of what the agent user could reach: the access probe.

A trajectory records what the agent asked a shell to run, not what the shell
did. Every round of review found another shell form the command parser read
wrongly (``+n``, heredoc expansions, brace operands, process substitutions,
longer ``&&`` chains), in both directions: a hidden read it missed, or an
honest command it blamed. So the verdict no longer rests on parsing. It rests
on a fact the agent cannot change: whether each protected path was closed to
the agent user while the agent ran.

BenchFlow runs one probe as root at the end of ``install_agent`` (after
lockdown, before any agent turn) and one when the agent is done (before the
verifier, before any hardening). It runs only on root's plumbing route
(class C8: ``exec(..., user="root", system_shell=True)`` on a backend
with ``exec_route``, which starts ``/bin/sh`` by absolute path and sources the
task's persistent env without its PATH or loader variables). Nothing in it is
resolved from any PATH: the agent user's ids come from /etc/passwd and
/etc/group read with shell builtins, and ``stat`` and ``ls`` are taken by
absolute path from PT's protected kit (``/opt/benchflow/rt/bin``, root's
copies made before the agent's turn) when the sandbox has one, from fixed
system paths only when it has none, and started through ``env -i`` (an
environment built from scratch). A backend without that route is not probed: the run is unattested,
never Checked and never blamed.
The probe resolves each protected path as the kernel does and prints ``lstat``
facts for every entry it visits: the written path, every link's own entry,
and every component of each link's target, followed recursively. The
decision is made here on the host, from those facts, by :func:`assess_access`,
with one component rule for all of them (:meth:`_Facts.replaceable`). A root-owned folder the agent user cannot search, or a
root-owned file it cannot read, on the path means the agent could not have
read what lies behind it, whatever its commands said. Anything else (readable,
owned by the agent, behind an ACL or a symlink, not probed) is an exposure: a
static vector of the run, never blame.

The parser (``exec_decompose``) only annotates: what a command may have read,
written or dialed, shown next to the verdict, never deciding it.
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from benchflow._runtime_tree import KIT_BIN, KIT_RECORD
from benchflow.integrity.contract import (
    REFUSED_PATH_VIOLATIONS,
    IntegrityContract,
    posix_pathname,
)
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw

logger = logging.getLogger(__name__)

ACCESS_PROBE_FILE = "access_probe.json"

# Classes the agent may not observe (I1). RewardOutput is left out: the
# reward is written after the agent, so reading its folder leaks nothing.
READ_PROTECTED_CLASSES = frozenset({"VerifierOnly", "Hidden", "ResetState"})
# Classes the agent may not change (I2). Only checked inside the roots that
# reach the verifier (the workspace and declared artifacts): a change
# anywhere else in the agent's sandbox never reaches a separate verifier, and
# a shared verifier is a vector already.
WRITE_PROTECTED_CLASSES = frozenset(
    {"Trusted", "Public", "VerifierOnly", "Hidden", "ResetState"}
)

# Where the probe takes its tools from. A sandbox with PT's protected kit
# (C8, ``_runtime_tree``: its RECORD exists) uses only the kit's copies in
# KIT_BIN, which the route has checked against the kit's MANIFEST; a tool the
# kit lacks is no fact, never a fall back to the image. Without a kit (no
# sandbox user was set up) the absolute system paths below, in order.
PROBE_TOOL_PATHS = {
    "env": ("/usr/bin/env", "/bin/env"),
    "stat": ("/usr/bin/stat", "/bin/stat"),
    "ls": ("/usr/bin/ls", "/bin/ls"),
    "readlink": ("/usr/bin/readlink", "/bin/readlink"),
    "od": ("/usr/bin/od", "/bin/od"),
    "tr": ("/usr/bin/tr", "/bin/tr"),
}
# Symbolic links followed per protected path before it counts as unresolved.
PROBE_MAX_LINKS = 16

# Root's probe: shell builtins, and the tools above by absolute path with an
# empty environment. It prints facts, never decides. For every protected path
# it resolves the path as the kernel does, recording each directory entry it
# visits (lstat facts, real paths): the written path, every symbolic link's
# own entry and every component of each link's target, followed recursively.
# Every pathname field is the lowercase hex of its raw bytes, written from a
# pipe (never through a command substitution, which drops trailing newlines),
# so any name (newlines, tabs, backslashes, non-UTF-8 bytes) is exact. A link
# target is read with ``readlink -n`` and a sentinel byte for the same reason.
#   U <uid> <gids...>                           the agent user
#   P <hex path>                                a protected path follows
#   S <uid> <gid> <octal> <ls-mode> <hex path>  an entry that is not a link
#   L <uid> <gid> <octal> <ls-mode> <hex path>  an entry that is a link
#   F <hex path>                                the object the path resolves to
#   A <hex path>                                the first entry that is missing
#   X <hex path>                                unresolved (a loop, too many
#                                               links, an unreadable link)
# A protected path without an F, A or X record never completed: unresolved.
PROBE_SCRIPT = r"""
agent=$1; shift
bf_tool() {
  if [ -f KIT_RECORD_PATH ]; then
    bf_t=KIT_BIN_PATH/$1; [ -x "$bf_t" ] && { printf '%s' "$bf_t"; return 0; }; return 1
  fi
  shift
  for bf_t in "$@"; do [ -x "$bf_t" ] && { printf '%s' "$bf_t"; return 0; }; done; return 1
}
ENVT=$(bf_tool env ENV_PATHS) && STAT=$(bf_tool stat STAT_PATHS) && LS=$(bf_tool ls LS_PATHS) \
  && RL=$(bf_tool readlink READLINK_PATHS) && OD=$(bf_tool od OD_PATHS) && TR=$(bf_tool tr TR_PATHS) \
  || { printf 'E\tno-system-tools\n'; exit 0; }
uid=; gid=
while IFS=: read -r n _ u g _; do
  if [ "$n" = "$agent" ]; then uid=$u; gid=$g; break; fi
done < /etc/passwd
[ -n "$uid" ] || { printf 'E\tno-such-user\n'; exit 0; }
gids=$gid
while IFS=: read -r n _ g m; do
  case ",$m," in *",$agent,"*) gids="$gids $g" ;; esac
done < /etc/group
printf 'U\t%s\t%s\n' "$uid" "$gids"
bf_hex() { printf '%s' "$1" | "$ENVT" -i "$OD" -An -v -tx1 | "$ENVT" -i "$TR" -d ' \n'; }
bf_rec() { printf '%s\t' "$1"; bf_hex "$2"; printf '\n'; }
bf_entry() {
  bf_f=$("$ENVT" -i "$STAT" -c '%u %g %a' -- "$2" 2>/dev/null) || return 1
  bf_m=$("$ENVT" -i "$LS" -ldn -- "$2" 2>/dev/null); bf_m=${bf_m%% *}
  printf '%s\t%s\t%s\t' "$1" "$bf_f" "$bf_m"; bf_hex "$2"; printf '\n'
}
for p in "$@"; do
  bf_rec P "$p"
  cur=/; todo=${p#/}; hops=0; end=F
  bf_entry S / || { bf_rec X /; continue; }
  while [ -n "$todo" ]; do
    seg=${todo%%/*}
    if [ "$todo" = "$seg" ]; then todo=; else todo=${todo#*/}; fi
    case $seg in
      ''|.) continue ;;
      ..) if [ "$cur" != / ]; then cur=${cur%/*}; [ -n "$cur" ] || cur=/; fi; continue ;;
    esac
    if [ "$cur" = / ]; then next=/$seg; else next=$cur/$seg; fi
    if [ -L "$next" ]; then
      bf_entry L "$next" || { bf_rec X "$next"; end=; break; }
      hops=$((hops + 1))
      t=$("$ENVT" -i "$RL" -n -- "$next" 2>/dev/null && printf x) || t=
      t=${t%x}
      if [ "$hops" -gt MAX_LINKS ] || [ -z "$t" ]; then bf_rec X "$next"; end=; break; fi
      case $t in /*) cur=/; t=${t#/} ;; esac
      if [ -n "$todo" ]; then todo=$t/$todo; else todo=$t; fi
      continue
    fi
    if [ ! -e "$next" ]; then bf_rec A "$next"; end=; break; fi
    bf_entry S "$next" || { bf_rec X "$next"; end=; break; }
    cur=$next
  done
  if [ -n "$end" ]; then bf_rec F "$cur"; fi
done
"""
for _name, _paths in PROBE_TOOL_PATHS.items():
    PROBE_SCRIPT = PROBE_SCRIPT.replace(f"{_name.upper()}_PATHS", shlex.join(_paths))
PROBE_SCRIPT = PROBE_SCRIPT.replace("MAX_LINKS", str(PROBE_MAX_LINKS))
PROBE_SCRIPT = PROBE_SCRIPT.replace("KIT_RECORD_PATH", shlex.quote(KIT_RECORD))
PROBE_SCRIPT = PROBE_SCRIPT.replace("KIT_BIN_PATH", shlex.quote(KIT_BIN))


def probe_paths(
    contract: IntegrityContract, *, separate: bool = True
) -> tuple[list[str], list[str]]:
    """``(paths to probe, protected locators no probe can attest)``.

    Read-protected paths anywhere; write-protected paths only inside the
    workspace or a declared artifact. A protected route (a URL) has no
    filesystem fact behind it, so it is returned as unattestable.
    """

    checks, routes = protection_checks(contract)
    return list(checks), routes


def protection_checks(
    contract: IntegrityContract,
) -> tuple[dict[str, set[str]], list[str]]:
    """Each protected path with the operations it must be closed to
    (``read``, ``write`` or both, assessed independently), and the routes."""

    # Every resource path here is the contract's one typed canonical form
    # , so containment is a plain component-prefix test, and a
    # route is told by its kind, never by its spelling.
    crossing = [
        item.path
        for item in contract.resources
        if item.path
        and item.kind == "path"
        and item.resource_class in {"AgentWritable", "DeclaredArtifact"}
    ]
    checks: dict[str, set[str]] = {}
    routes: list[str] = []
    for item in contract.resources:
        if not item.path:
            continue
        read = item.resource_class in READ_PROTECTED_CLASSES
        write = (
            item.resource_class in WRITE_PROTECTED_CLASSES
            and item.kind == "path"
            and any(_under(item.path, root) for root in crossing)
        )
        if item.kind != "path":
            if (read or item.resource_class in WRITE_PROTECTED_CLASSES) and (
                item.path not in routes
            ):
                routes.append(item.path)
            continue
        if not (read or write):
            continue
        ops = checks.setdefault(item.path, set())
        if read:
            ops.add("read")
        if write:
            ops.add("write")
    return checks, routes


def probe_command(sandbox_user: str, paths: list[str]) -> str:
    """The probe as one system-shell command: no second shell is looked up."""
    return f"set -- {shlex.join([sandbox_user, *paths])}\n{PROBE_SCRIPT}"


async def run_access_probe(
    env: Any, sandbox_user: str | None, paths: list[str]
) -> dict[str, Any]:
    """One probe of ``paths`` as root on root's plumbing route.

    Returns ``{"status": "ok", "stdout": ...}`` or a status saying why there is
    no fact (``unsupported``: the backend has no root plumbing route, so a root
    process it starts could load what the task's persistent env names;
    ``failed``: the probe did not finish). Never raises.
    """

    if not sandbox_user or sandbox_user in {"root", "0"}:
        return {"status": "agent_root"}
    if getattr(env, "supports_system_shell", False) is not True or not callable(
        getattr(env, "exec_route", None)
    ):
        # C8 has one mechanism: without the backend's route, the
        # persistent env's PATH and loader values reach root's tools.
        return {
            "status": "unsupported",
            "reason": "no root plumbing route on this backend",
        }
    try:
        result = await exec_raw(
            env.exec,
            probe_command(sandbox_user, paths),
            user="root",
            timeout_sec=60,
            system_shell=True,
        )
    except Exception as exc:  # the probe never fails a trial
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
    if result.return_code != 0:
        # The record keeps the exit code; stderr is a display copy, logged.
        logger.warning(
            "access probe exited %s: %s",
            result.return_code,
            diagnostic_result(result).stderr.render()[-300:],
        )
        return {"status": "failed", "error": f"exit {result.return_code}"}
    return {"status": "ok", "stdout": result.stdout or ""}


@dataclass
class _Component:
    path: str
    uid: int | None = None
    gid: int | None = None
    mode: int = 0
    is_dir: bool = False
    acl: bool = False
    link: bool = False


@dataclass
class _Resolution:
    """One protected path as the probe resolved it."""

    # Every directory entry visited, in order (real paths; links included).
    entries: list[_Component] = field(default_factory=list)
    # found (``end_path`` is the object), absent (``end_path`` is the first
    # missing entry) or unresolved (a loop, too many links, unreadable).
    end: str = "unresolved"
    end_path: str | None = None


@dataclass
class _Probe:
    uid: int | None = None
    gids: set[int] = field(default_factory=set)
    paths: dict[str, _Resolution] = field(default_factory=dict)
    error: str | None = None


def decode_path(field: str) -> str | None:
    """A probe pathname field: lowercase hex of the raw bytes, decoded exactly
    (UTF-8 with surrogateescape, as the host spells paths). None for anything
    that is not an absolute path's even-length hex."""
    if not field or len(field) % 2 or not _HEX_RE.fullmatch(field):
        return None
    raw = bytes.fromhex(field)
    if not raw.startswith(b"/") or b"\0" in raw:
        return None
    return raw.decode("utf-8", "surrogateescape")


_HEX_RE = re.compile(r"[0-9a-f]+")


def _parse_entry(rest: str, *, link: bool) -> _Component | None:
    parts = rest.split("\t")
    if len(parts) != 3:
        return None
    facts, ls_mode, field = parts
    path = decode_path(field)
    if path is None:
        return None
    try:
        uid, gid, octal = facts.split()
        return _Component(
            path=path,
            uid=int(uid),
            gid=int(gid),
            mode=int(octal, 8),
            is_dir=ls_mode.startswith("d") and not link,
            acl=len(ls_mode) > 10 and ls_mode[10] == "+",
            link=link,
        )
    except ValueError:
        return None


_ENDS = {"F": "found", "A": "absent", "X": "unresolved"}


def parse_probe(stdout: str) -> _Probe:
    """The probe's records. A record that does not decode exactly leaves its
    protected path unresolved (open), and so does a path whose resolution
    never completed (no F, A or X record); one outside any path, or a
    protected path that does not decode, makes the whole probe unattested."""
    probe = _Probe()
    current: _Resolution | None = None
    broken: set[int] = set()
    for line in stdout.split("\n"):
        if not line:
            continue
        kind, _, rest = line.partition("\t")
        if kind == "E":
            probe.error = rest or "probe error"
        elif kind == "U":
            uid, _, gids = rest.partition("\t")
            if uid.strip().isdigit():
                probe.uid = int(uid)
            probe.gids = {int(g) for g in gids.split() if g.isdigit()}
        elif kind == "P":
            path = decode_path(rest)
            if path is None:
                probe.error = "undecodable protected path record"
                current = None
                continue
            current = probe.paths[path] = _Resolution()
        elif current is None:
            probe.error = f"record outside a protected path: {kind[:8]!r}"
        elif current.end_path is not None:
            broken.add(id(current))  # nothing may follow the end record
        elif kind in {"S", "L"}:
            entry = _parse_entry(rest, link=kind == "L")
            if entry is None:
                broken.add(id(current))
            else:
                current.entries.append(entry)
        elif kind in _ENDS:
            path = decode_path(rest)
            if path is None:
                broken.add(id(current))
            else:
                current.end, current.end_path = _ENDS[kind], path
        else:
            broken.add(id(current))
    for resolution in probe.paths.values():
        if id(resolution) in broken:
            resolution.end, resolution.end_path = "unresolved", None
    return probe


def _bits(component: _Component, uid: int, gids: set[int]) -> int:
    """The rwx bits that apply to the agent user (7 when it can change them)."""

    if component.acl or component.uid is None or component.uid == uid:
        # An ACL can grant what the mode does not, and an owner can chmod.
        return 7
    if component.gid in gids:
        return (component.mode >> 3) & 7
    return component.mode & 7


class _Facts:
    """The probe's facts about one resolution, by real path."""

    def __init__(self, probe: _Probe, resolution: _Resolution) -> None:
        assert probe.uid is not None
        self.uid, self.gids = probe.uid, probe.gids
        self.by_path = {entry.path: entry for entry in resolution.entries}

    def reachable(self, path: str) -> bool:
        """Whether the agent user can reach ``path`` by its real path: every
        folder above it searchable. A folder the probe has no fact for counts
        as searchable (it cannot close anything)."""
        for ancestor in _ancestors(path):
            entry = self.by_path.get(ancestor)
            if entry is None or entry.link:
                continue
            if not entry.is_dir or not _bits(entry, self.uid, self.gids) & 1:
                return False  # a path through a file, or an unsearchable folder
        return True

    def replaceable(self, entry: _Component) -> bool:
        """The one component rule: an entry is a replacement risk when the
        agent user can reach it and write the folder holding it (renaming,
        moving or deleting and recreating it), unless a sticky bit protects
        an entry it does not own in a folder it does not own."""
        if entry.path == "/" or not self.reachable(entry.path):
            return False
        parent = self.by_path.get(posixpath.dirname(entry.path))
        if parent is None:
            return True
        if not _bits(parent, self.uid, self.gids) & 2:
            return False
        sticky = parent.mode & 0o1000
        return not (sticky and entry.uid != self.uid and parent.uid != self.uid)


def _ancestors(path: str) -> list[str]:
    """The folders above ``path``, from the root down (``/a/b`` -> ``/``, ``/a``)."""
    parts = [part for part in path.split("/") if part]
    return ["/"] + ["/" + "/".join(parts[:i]) for i in range(1, len(parts))]


def path_state(probe: _Probe, path: str, *, write: bool = False) -> str:
    """``closed``, ``absent`` or ``open`` for the agent user, from one probe.

    Reading: the object the path resolves to, by its real path (the agent
    user can go there directly, whatever links led there), must deny it
    reading, or listing and search for a folder.

    Writing: every entry the resolution visited (the written path, every
    link's own entry, every component of every link target) must not be
    replaceable (:meth:`_Facts.replaceable`), and the object itself must deny
    the agent user writing it (or searching it, for a folder, whose entries
    it could otherwise change).
    """

    if probe.uid is None or path not in probe.paths or probe.uid == 0:
        return "open"
    resolution = probe.paths[path]
    if resolution.end == "unresolved" or resolution.end_path is None:
        return "open"
    facts = _Facts(probe, resolution)
    if write and any(facts.replaceable(entry) for entry in resolution.entries):
        return "open"
    target = resolution.end_path
    if resolution.end == "absent":
        if not write:
            return "absent"
        # Missing: open when the agent user could create it in its folder.
        parent = facts.by_path.get(posixpath.dirname(target))
        if facts.reachable(target) and (
            parent is None
            or (parent.is_dir and _bits(parent, facts.uid, facts.gids) & 2)
        ):
            return "open"
        return "absent"
    entry = facts.by_path.get(target)
    if entry is None:
        return "open"
    if not facts.reachable(target):
        return "closed"
    bits = _bits(entry, facts.uid, facts.gids)
    # Writing: a file's w; a folder the agent user can search may hold entries
    # it can change. Reading: a file's r; a folder's listing (r) shows its
    # names and search (x) reaches its contents.
    needed = _NEEDED_BITS[(write, entry.is_dir)]
    return "open" if bits & needed else "closed"


_NEEDED_BITS = {(True, True): 1, (True, False): 2, (False, True): 5, (False, False): 4}


@dataclass
class AccessAssessment:
    status: str  # attested | unattested | agent_root | scripted
    # Paths every required operation was proven closed for (a summary).
    closed: list[str] = field(default_factory=list)
    read_closed: list[str] = field(default_factory=list)
    write_closed: list[str] = field(default_factory=list)
    read_open: list[str] = field(default_factory=list)
    write_open: list[str] = field(default_factory=list)
    unattested_routes: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "closed": self.closed,
            "read_closed": self.read_closed,
            "write_closed": self.write_closed,
            "read_open": self.read_open,
            "write_open": self.write_open,
            "unattested_routes": self.unattested_routes,
            "reason": self.reason,
        }

    def closes(self, path: str, *, write: bool = False) -> bool:
        """Whether ``path`` lies under a protected path proven closed to the
        agent user for this operation. A read proof says nothing of writes."""

        normalized = _norm(path)
        roots = self.write_closed if write else self.read_closed
        return any(_under(normalized, root) for root in roots)


def assess_access(
    contract: IntegrityContract, record: dict[str, Any] | None
) -> AccessAssessment:
    """Decide exposure from the stored probes (``access_probe.json``).

    A path is closed only when it was closed (or absent) in the probe before
    the agent's first turn *and* in the probe after it: a path the agent
    could read at either point may have been read.
    """

    if record and record.get("pre", {}).get("status") == "agent_root":
        return AccessAssessment(status="agent_root", reason="the agent ran as root")
    refused = [
        violation
        for violation in contract.violations
        if violation.startswith(REFUSED_PATH_VIOLATIONS)
    ]
    if refused:
        # A path the contract could not type: nothing it protects is known
        # exactly, so nothing is certified.
        return AccessAssessment(
            status="unattested", reason=f"contract refused a path: {refused[0]}"
        )
    paths, routes = probe_paths(contract)
    if not record:
        return AccessAssessment(status="unattested", reason="no access probe")
    pre, post = record.get("pre") or {}, record.get("post") or {}
    for name, item in (("pre", pre), ("post", post)):
        if item.get("status") != "ok":
            return AccessAssessment(
                status="unattested",
                reason=f"{name}-turn probe {item.get('status') or 'missing'}"
                + (f": {item['reason']}" if item.get("reason") else ""),
            )
    probes = [parse_probe(pre["stdout"]), parse_probe(post["stdout"])]
    for probe in probes:
        if probe.error or probe.uid is None:
            return AccessAssessment(
                status="unattested", reason=f"probe: {probe.error or 'no agent uid'}"
            )
    checks, _ = protection_checks(contract)
    assessment = AccessAssessment(status="attested", unattested_routes=routes)
    for path in paths:
        ops = checks[path]
        proven = True
        for op in sorted(ops):
            write = op == "write"
            states = [path_state(probe, path, write=write) for probe in probes]
            if "open" in states:
                proven = False
                (assessment.write_open if write else assessment.read_open).append(path)
            else:
                (assessment.write_closed if write else assessment.read_closed).append(
                    path
                )
        if proven:
            assessment.closed.append(path)
    return assessment


def _norm(path: str) -> str:
    # The contract's own spelling, so the probe inspects exactly the bytes
    # the contract protects (no separator, case or ``..`` rewriting).
    return posix_pathname(path)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


__all__ = [
    "ACCESS_PROBE_FILE",
    "AccessAssessment",
    "assess_access",
    "parse_probe",
    "path_state",
    "probe_paths",
    "protection_checks",
    "run_access_probe",
]

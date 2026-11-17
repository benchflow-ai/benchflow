"""Links the agent made into the verifier's output folder ((c)3).

Root's test.sh follows a link. A link the agent left anywhere test.sh writes
(``ln -s /logs/verifier/reward.txt /app/grade.log``) turns test.sh's own
write of some other file into the reward: in integrity off and audit a task
whose test.sh wrote 0 to reward.txt and then ``echo 1 > /app/grade.log``
scored 1.0.

Nothing here runs a program of the image's:

- Before the agent's turn, root records every symlink in the sandbox with
  its target (:func:`record_before_turn`): through ``bfx scan`` where bfx is
  delivered (its own walk, never entering a mount point, so each mount but
  the kernel's and the output folder's is walked as its own root), else a
  walk of the shell's own (its globbing and ``[ -L ]``; only a link's target
  is read by the image's ``readlink``, and a link it cannot read leaves the
  walk incomplete).
- Before test.sh the same walk runs again: a link that is new since, or
  whose target changed, is the turn's; each turn link that resolves (through
  chains of links and links to folders, on the host, from the walk's own
  listing) to the output folder, into it, or to a folder above it is removed
  without following it, and flagged. The image's own links stay.
- After grading the walk runs once more: a turn link into the folder then
  leaves the reward's provenance unverifiable.

A walk that cannot run, or cannot read every link, is a check that did not
run: the reward's provenance is then unverifiable (:mod:`_reward_provenance`).
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import posixpath
import shlex
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: The folder test.sh writes its reward to.
OUTPUTS = "/logs/verifier"
#: The flag's category in the detector's record.
CATEGORY = "verifier-output-link"
#: Not walked: the kernel's own trees and the output folder itself (test.sh's
#: clear empties it; a link in it is a name test.sh replaces). /dev is walked,
#: and every mount under it (/dev/shm, /dev/mqueue) as a root of its own.
SKIP = ("/proc", "/sys", OUTPUTS)
#: Where the record from before the turn is kept on the sandbox object.
INVENTORY = "_benchflow_link_inventory"
#: bfx scan fails past this many names (a flood): the links-only walk follows.
SCAN_MAX_ENTRIES = 4_000_000
_HOPS = 40


@dataclass
class Walked:
    """One walk: every symlink's target by its absolute path, and whether
    the walk saw (and could read) all of them."""

    engine: str
    links: dict[str, str] = field(default_factory=dict)
    # The kernel's resolution of the turn's links (None: not asked).
    kernel: dict[str, str | None] | None = None
    roots: tuple[str, ...] = ("/",)
    complete: bool = True
    why: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "links": dict(self.links),
            "roots": list(self.roots),
            "complete": self.complete,
            "why": self.why,
        }

    @classmethod
    def from_json(cls, data: Any) -> Walked | None:
        if not isinstance(data, dict) or not isinstance(data.get("links"), dict):
            return None
        links = {
            str(k): str(v)
            for k, v in data["links"].items()
            if isinstance(k, str) and isinstance(v, str)
        }
        if len(links) != len(data["links"]):
            return None  # an entry the record cannot hold: no record at all
        roots = data.get("roots")
        return cls(
            str(data.get("engine") or "record"),
            links=links,
            roots=tuple(str(r) for r in roots) if isinstance(roots, list) else ("/",),
            complete=data.get("complete") is True,
            why=data.get("why") if isinstance(data.get("why"), str) else None,
        )


#: Why a walk after the turn did not run: the sandbox has no record of its
#: links from before the agent's turn, so no link can be told apart from the
#: image's own. BenchFlow's own gap, never the agent's.
NO_RECORD = "no record of the links before the turn"
#: The record's file in the trial's integrity folder, for a branch child or a
#: resume from this trial's checkpoint.
RECORD_FILE = "links-before-turn.json"


#: Why a walk did not run where bfx cannot (BenchFlow's gap, never the agent's).
BFX_UNAVAILABLE = "bfx cannot run in this sandbox"


def walker_unavailable(env: Any) -> str | None:
    """Why BenchFlow's walker (bfx) cannot run in *env*, or None when it can:
    a backend with no bfx route, or no pinned bfx delivered (delivery off,
    no binary for the sandbox's arch)."""
    if not callable(getattr(env, "bfx_call", None)):
        name = type(env).__name__.removesuffix("Sandbox") or type(env).__name__
        return (
            f"{BFX_UNAVAILABLE}: {name} can't run BenchFlow's integrity check yet; "
            "run the trial on Docker or Daytona"
        )
    if getattr(env, "bfx_delivery", None) is None:
        return (
            f"{BFX_UNAVAILABLE} (not delivered: BENCHFLOW_BFX_DELIVERY=off, or no "
            "pinned bfx for the sandbox's architecture)"
        )
    return None


def walker_record(env: Any) -> dict[str, Any]:
    """Which walker the link checks had (the trial's ``provenance.walker``):
    BenchFlow's bfx with its delivery, the sandbox's arch and that arch's
    pinned SHA-256; ``engine`` None, with why, where bfx could not run."""
    from benchflow._bfx import pins

    why = walker_unavailable(env)
    delivery = getattr(env, "bfx_delivery", None)
    arch = getattr(env, "_bfx_arch", None)
    arch = arch if isinstance(arch, str) else None
    return {
        "engine": None if why else "bfx",
        "delivery": delivery if isinstance(delivery, str) and not why else None,
        "arch": arch,
        "sha256": pins.SHA256.get(arch) if arch and not why else None,
        "why": why,
    }


def save_record(env: Any, rollout_dir: Any) -> None:
    """The sandbox's record before the turn, beside the trial (a branch child
    or a checkpoint resume of this trial inherits it)."""
    import json
    from pathlib import Path

    walked = getattr(env, INVENTORY, None)
    if not isinstance(walked, Walked) or rollout_dir is None:
        return
    folder = Path(rollout_dir) / "integrity"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / RECORD_FILE).write_text(json.dumps(walked.to_json()))
    except OSError as exc:
        # A full disk is said once, by the run (no traceback for it here).
        logger.warning("the record of the links before the turn was not saved: %s", exc)


def load_record(trial_dir: Any) -> Walked | None:
    """The record a trial saved before its turn (:func:`save_record`), or None."""
    import json
    from pathlib import Path

    if trial_dir is None:
        return None
    try:
        data = json.loads((Path(trial_dir) / "integrity" / RECORD_FILE).read_text())
    except (OSError, ValueError):
        return None
    return Walked.from_json(data)


def inherit_record(env: Any, record: Walked | None) -> None:
    """A sandbox restored from another's snapshot keeps that sandbox's record
    from before its turn: a link made before the snapshot is still the turn's."""
    if isinstance(record, Walked):
        with contextlib.suppress(AttributeError):
            setattr(env, INVENTORY, record)


def _name(entry: dict, key: str) -> str | None:
    """*key* of a bfx answer's entry as a str that keeps its bytes: bfx gives
    a name that is not UTF-8 as ``<key>_b64``, decoded here with
    surrogateescape, never lossily; None when the entry holds neither."""
    value = entry.get(key)
    if isinstance(value, str):
        return value
    raw = entry.get(key + "_b64")
    if not isinstance(raw, str):
        return None
    try:
        return base64.b64decode(raw, validate=True).decode("utf-8", "surrogateescape")
    except (ValueError, binascii.Error):
        return None


def _wire(args: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    """*name* into a bfx request as text, or as ``<key>_b64`` when its bytes
    are not UTF-8."""
    raw = name.encode("utf-8", "surrogateescape")
    try:
        args[key] = raw.decode("utf-8")
    except UnicodeDecodeError:
        args[key + "_b64"] = base64.b64encode(raw).decode("ascii")
    return args


def shown(name: Any) -> str:
    """*name* for text a person reads (a warning, a flag's reason): a byte
    that is not UTF-8 as ``\\xNN``. The record keeps the name itself."""
    if not isinstance(name, str):
        return str(name)
    return name.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


def _lossy(name: str) -> bool:
    """A name a text decode replaced bytes of: it names no file."""
    return "\ufffd" in name


def setup_timeout_sec() -> int:
    """The verifier-setup budget each walk and removal runs within."""
    from benchflow._verifier_setup import verifier_setup_timeout_sec

    return verifier_setup_timeout_sec()


def _skipped(path: str) -> bool:
    return any(path == s or path.startswith(s + "/") for s in SKIP)


def resolve(path: str, links: dict[str, str]) -> str | None:
    """Where *path* leads through the links of the listing (a link to a
    link, a link to a folder on the way), as the kernel would resolve it;
    None for a loop."""
    hops = 0
    parts = [p for p in path.split("/") if p]
    done = "/"
    while parts:
        part = parts.pop(0)
        if part == ".":
            continue
        if part == "..":
            done = posixpath.dirname(done) or "/"
            continue
        here = posixpath.join(done, part)
        target = links.get(here)
        if target is None:
            done = here
            continue
        hops += 1
        if hops > _HOPS:
            return None
        if target.startswith("/"):
            done = "/"
        parts = [p for p in target.split("/") if p] + parts
    return done


def into_outputs(resolved: str | None) -> bool:
    """A link that leads to the output folder, into it, or to a folder above
    it (through which a path reaches it); or anywhere under /proc, whose
    ``<pid>/root``, ``cwd`` and ``fd`` names lead back into the sandbox's own
    tree (``/proc/self/root/logs/verifier/…``)."""
    if resolved is None:
        return True  # a loop is no link test.sh can use; it goes too
    return (
        resolved == "/proc"
        or resolved.startswith("/proc/")
        or resolved == OUTPUTS
        or resolved.startswith(OUTPUTS + "/")
        or OUTPUTS.startswith(resolved.rstrip("/") + "/")
    )


# -- the walks --------------------------------------------------------------


async def _bfx_scan(env: Any, roots: list[str], timeout_sec: int) -> dict[str, Any]:
    return await env.bfx_call(
        "scan",
        {
            "walks": [{"root": r, "depth": 4096} for r in roots],
            "mounts": False,
            "acl": False,
            "max_bytes": 512 << 20,
            "max_entries": SCAN_MAX_ENTRIES,
        },
        timeout_sec=timeout_sec,
    )


async def _walk_bfx(
    env: Any, roots: tuple[str, ...] | None, timeout_sec: int
) -> Walked | None:
    from benchflow._bfx.locate import BfxUnavailable

    if not callable(getattr(env, "bfx_call", None)):
        return None
    # Every walk finds the mounts afresh from /: a record taken by another
    # walk (find, the shell) names / alone, and a mount the agent made would
    # be missed by an old list.
    roots = None
    try:
        answer = await _bfx_scan(env, ["/"], timeout_sec)
        entries = list(answer.get("entries") or ())
        found_roots = {"/"}
        if roots is None:
            # The mount points a walk listed but did not enter (a bind-mounted
            # workspace, /dev, then /dev/shm in it) are walked as roots, until
            # no new one appears.
            new = entries
            for _ in range(8):
                more = sorted(
                    {
                        name
                        for e in new
                        if e.get("mount")
                        and e.get("type") == "dir"
                        and (name := _name(e, "path")) is not None
                        and not _skipped(name)
                    }
                    - found_roots
                )
                if not more:
                    break
                extra = await _bfx_scan(env, more, timeout_sec)
                new = list(extra.get("entries") or ())
                entries += new
                for error in extra.get("errors") or ():
                    answer.setdefault("errors", []).append(error)
                found_roots.update(more)
    except BfxUnavailable:
        return None
    except Exception as exc:
        return Walked("bfx", complete=False, why=f"bfx scan failed: {exc}")
    walked = Walked("bfx", roots=tuple(sorted(found_roots)))
    for e in entries:
        if e.get("type") != "link":
            continue
        path = _name(e, "path")
        target = _name(e, "link")
        if path is None or target is None:
            # An entry the record cannot hold is never dropped (F40).
            walked.complete = False
            walked.why = f"a link the walk could not record: {sorted(e)}"
            continue
        walked.links[path] = target
    errors = [x for x in answer.get("errors") or () if isinstance(x, dict)]
    if errors:
        walked.complete = False
        walked.why = f"bfx scan could not read {len(errors)} path(s), e.g. {errors[0]}"
    return walked


# A walk of the shell's own: globbing and [ -L ] / [ -d ], no program of the
# image's but readlink for a link's target. NUL-separated records: L, path,
# target ("" and a U record when readlink failed).
_SHELL_WALK = r"""
bf_w() {
  for bf_e in "$1"/* "$1"/.[!.]* "$1"/..?*; do
    if [ -L "$bf_e" ]; then
      if bf_t=$(readlink -- "$bf_e" 2>/dev/null); then printf 'L\000%s\000%s\000' "$bf_e" "$bf_t"
      else printf 'U\000%s\000\000' "$bf_e"; fi
    elif [ -d "$bf_e" ]; then
      case "$bf_e" in /proc|/sys|/logs/verifier) ;; *) bf_w "$bf_e" ;; esac
    fi
  done
}
bf_w ""
"""


async def _walk_shell(env: Any, timeout_sec: int) -> Walked:
    from benchflow.sandbox._exec_output import exec_raw

    try:
        result = await exec_raw(
            env.exec, _SHELL_WALK, user="root", timeout_sec=timeout_sec
        )
    except Exception as exc:
        return Walked("shell", complete=False, why=f"the shell walk failed: {exc}")
    if getattr(result, "return_code", 1) not in (0, None):
        return Walked(
            "shell",
            complete=False,
            why=f"the shell walk failed (rc {getattr(result, 'return_code', None)})",
        )
    return _records("shell", getattr(result, "stdout", "") or "")


# Links only, by find, with no cap on the names it passes: a fallback for a
# bfx walk that failed or was capped (a flood of names). The kernel's trees
# and the output folder are pruned; mounts are walked. Same records as the
# shell walk.
_FIND_WALK = r"""
command -v find >/dev/null 2>&1 || exit 127
find / \( -path /proc -o -path /sys -o -path /logs/verifier \) -prune \
  -o -type l -exec sh -c '
for bf_l; do
  if bf_t=$(readlink -- "$bf_l" 2>/dev/null); then printf "L\000%s\000%s\000" "$bf_l" "$bf_t"
  else printf "U\000%s\000\000" "$bf_l"; fi
done' sh {} +
"""


def _records(engine: str, stdout: str) -> Walked:
    walked = Walked(engine)
    fields = (stdout or "").split("\0")
    for i in range(0, len(fields) - 2, 3):
        kind, path, target = fields[i], fields[i + 1], fields[i + 2]
        if _lossy(path) or _lossy(target):
            # The output's text decode replaced bytes of the name (F40).
            walked.complete = False
            walked.why = f"{engine} gave a name that is not text"
            continue
        if kind == "L" and path.startswith("/"):
            walked.links[path] = target
        elif kind == "U":
            walked.complete = False
            walked.why = f"the target of {path} could not be read (no readlink)"
    return walked


async def _walk_find(env: Any, timeout_sec: int) -> Walked | None:
    """The links by find; None where the image has no find."""
    from benchflow.sandbox._exec_output import exec_raw

    try:
        result = await exec_raw(
            env.exec, _FIND_WALK, user="root", timeout_sec=timeout_sec
        )
    except Exception as exc:
        return Walked("find", complete=False, why=f"the find walk failed: {exc}")
    code = getattr(result, "return_code", 1)
    if code == 127:
        return None
    walked = _records("find", getattr(result, "stdout", "") or "")
    if code not in (0, None):
        walked.complete = False
        walked.why = walked.why or f"the find walk failed (rc {code})"
    return walked


async def walk(env: Any, *, roots: tuple[str, ...] | None, timeout_sec: int) -> Walked:
    """Every symlink in the sandbox: bfx where delivered; where bfx is not,
    or its walk failed or was capped (a flood of names), a links-only find,
    else the shell's own walk. The links every walk found are kept; the walk
    is complete only when one of them was."""
    # BenchFlow's own binary alone: the image's find, readlink and sh can be
    # shadowed or replaced by the agent (F27), and a fallback kept a record
    # of its own (F15). A failed or capped bfx walk is an incomplete one.
    walked = await _walk_bfx(env, roots, timeout_sec)
    if walked is None:
        return Walked(
            "bfx", complete=False, why=walker_unavailable(env) or BFX_UNAVAILABLE
        )
    return walked


async def record_before_turn(env: Any, *, timeout_sec: int | None = None) -> None:
    """The record of every link before the agent's turn, kept on the host
    (``env._benchflow_link_inventory``). The first record of a sandbox is
    the one kept: a fresh verifier sandbox's, made before the frozen
    workspace arrives, or the one a branch child inherits, is never replaced
    by a walk that already sees the agent's links."""
    if isinstance(getattr(env, INVENTORY, None), Walked):
        return
    timeout_sec = timeout_sec or setup_timeout_sec()
    walked = await walk(env, roots=None, timeout_sec=timeout_sec)
    with contextlib.suppress(AttributeError):
        setattr(env, INVENTORY, walked)
    if not walked.complete:
        logger.warning(
            "the links before the agent's turn were not all recorded: %s", walked.why
        )


# The kernel's own answer for where each link leads (realpath: chains,
# relative targets, links to folders, /proc/<pid>/root), asked inside the
# sandbox as root: readlink -f, else realpath. NUL-separated path, answer
# ("" when it gave none).
_RESOLVE = r"""
if readlink -m / >/dev/null 2>&1; then bf_r="readlink -m --"
elif realpath -m / >/dev/null 2>&1; then bf_r="realpath -m --"
elif readlink -f / >/dev/null 2>&1; then bf_r="readlink -f --"
elif realpath / >/dev/null 2>&1; then bf_r="realpath --"
else exit 127; fi
for bf_l; do
  if bf_t=$($bf_r "$bf_l" 2>/dev/null); then printf "%s\000%s\000" "$bf_l" "$bf_t"
  else printf "%s\000\000" "$bf_l"; fi
done
"""


async def kernel_resolve(
    env: Any, paths: list[str], timeout_sec: int
) -> tuple[dict[str, str | None], str | None]:
    """Each path's kernel resolution (None: it gave none), and why the
    resolution did not complete (None when it did)."""
    from benchflow.sandbox._exec_output import exec_raw

    found: dict[str, str | None] = {}
    for i in range(0, len(paths), 200):
        batch = paths[i : i + 200]
        command = (
            "sh -c "
            + shlex.quote(_RESOLVE)
            + " sh "
            + " ".join(shlex.quote(p) for p in batch)
        )
        try:
            result = await exec_raw(
                env.exec, command, user="root", timeout_sec=timeout_sec
            )
        except Exception as exc:
            return found, f"the kernel's resolution failed: {exc}"
        code = getattr(result, "return_code", 1)
        if code not in (0, None):
            return found, f"the kernel's resolution failed (rc {code})"
        fields = (getattr(result, "stdout", "") or "").split("\0")
        for j in range(0, len(fields) - 1, 2):
            if fields[j]:
                found[fields[j]] = fields[j + 1] or None
    return found, None


#: The class of a flagged link, named in its flag.
INTO_OUTPUTS = "into the verifier's output folder"
THROUGH_PROC = "through /proc"
UNRESOLVED = "the check could not resolve it"


def _through_proc(path: str | None) -> bool:
    return path is not None and (path == "/proc" or path.startswith("/proc/"))


def _in_outputs(path: str | None) -> bool:
    return path is not None and (
        path == OUTPUTS
        or path.startswith(OUTPUTS + "/")
        or OUTPUTS.startswith(path.rstrip("/") + "/")
    )


def turn_links(before: Walked | None, now: Walked) -> list[dict[str, Any]]:
    """The links the turn made or changed that lead into the output folder
    or through /proc, each with its ``class``; a link no resolution can
    follow (a loop) is a row of class :data:`UNRESOLVED`. A dangling link is
    judged by where it leads as far as its target exists, the rest of the
    path appended as written; a dangling link elsewhere is no row."""
    rows = []
    for path, target in sorted(now.links.items()):
        if _skipped(path):
            continue
        if before is not None and before.links.get(path) == target:
            continue  # the image's own, as it was before the turn
        # The listing's resolution: every link it holds followed, a missing
        # part appended as written; None for a loop.
        resolved = resolve(path, now.links)
        # The kernel's (readlink -m): None when it gave no answer.
        kernel = now.kernel.get(path, "") if now.kernel is not None else ""
        if _in_outputs(resolved) or (kernel and _in_outputs(kernel)):
            kind = INTO_OUTPUTS
        elif (
            _through_proc(target)
            or _through_proc(resolved)
            or (kernel and _through_proc(kernel))
        ):
            kind = THROUGH_PROC
        elif resolved is None:
            kind = UNRESOLVED
        else:
            continue  # leads elsewhere, dangling or not: left as the agent left it
        row = {"path": path, "target": target, "resolved": resolved, "class": kind}
        if kernel:
            row["kernel"] = kernel
        rows.append(row)
    return rows


def _unresolved(rows: list[dict[str, Any]]) -> str | None:
    """Why the check is incomplete: a turn link it could not resolve."""
    left = [r for r in rows if r.get("class") == UNRESOLVED]
    if not left:
        return None
    return f"{UNRESOLVED}: {shown(left[0].get('path'))}" + (
        f" and {len(left) - 1} more" if len(left) > 1 else ""
    )


async def _with_kernel(
    env: Any, before: Walked | None, now: Walked, timeout_sec: int
) -> Walked:
    """*now*, with the kernel's resolution of each of the turn's links."""
    turn = [
        path
        for path, target in sorted(now.links.items())
        if not _skipped(path)
        and not (before is not None and before.links.get(path) == target)
    ]
    now.kernel = {}
    if turn:
        now.kernel, why = await kernel_resolve(env, turn, timeout_sec)
        if why:
            now.complete = False
            now.why = why
    return now


def _mount_of(path: str, roots: tuple[str, ...]) -> str:
    """The deepest mount point that holds *path* (bfx crosses no mount
    beneath the root it is given)."""
    best = "/"
    for root in roots:
        if path.startswith(root.rstrip("/") + "/") and len(root) > len(best):
            best = root
    return best


async def _mount_points(env: Any, timeout_sec: int) -> tuple[str, ...] | None:
    """Every mount point of the sandbox, from the kernel's mount table (bfx
    reads /proc/self/mountinfo), never a walk's own list (F15: a fallback
    walk's record named / alone); None where bfx cannot say."""
    try:
        answer = await env.bfx_call(
            "scan", {"mounts": True, "acl": False}, timeout_sec=timeout_sec
        )
    except Exception:
        return None
    rows = answer.get("mounts")
    if not isinstance(rows, list):
        return None
    points = {_name(m, "mount_point") for m in rows if isinstance(m, dict)}
    return tuple(sorted(p for p in points if p))


async def _still_links(
    env: Any, paths: list[str], timeout_sec: int
) -> tuple[set[str], str | None]:
    """Which of *paths* are links now, looked at component by component
    from / without following any (bfx ``scan`` paths); and why the look
    could not be made (None when it was)."""
    if not paths:
        return set(), None
    if callable(getattr(env, "bfx_call", None)):
        text = [p for p in paths if "paths" in _wire({}, "paths", p)]
        raw = [p for p in paths if p not in text]
        request: dict[str, Any] = {"acl": False, "mounts": False}
        if text:
            request["paths"] = text
        if raw:
            request["paths_b64"] = [
                base64.b64encode(p.encode("utf-8", "surrogateescape")).decode("ascii")
                for p in raw
            ]
        try:
            answer = await env.bfx_call("scan", request, timeout_sec=timeout_sec)
        except Exception as exc:
            return set(), f"the removal could not be confirmed: {exc}"
        links = {
            name
            for e in answer.get("entries") or ()
            if isinstance(e, dict)
            and e.get("type") == "link"
            and (name := _name(e, "path")) is not None
        }
        return {p for p in paths if p in links}, None
    from benchflow.sandbox._exec_output import exec_raw

    try:
        command = " ".join(
            f"[ -L {shlex.quote(p)} ] && printf '%s\\000' {shlex.quote(p)};"
            for p in paths
        )
        result = await exec_raw(
            env.exec, command + " true", user="root", timeout_sec=timeout_sec
        )
    except Exception as exc:
        return set(), f"the removal could not be confirmed: {exc}"
    if getattr(result, "return_code", 1) not in (0, None):
        return set(), "the removal could not be confirmed"
    said = set((getattr(result, "stdout", "") or "").split("\0"))
    return {p for p in paths if p in said}, None


async def _remove(
    env: Any, path: str, timeout_sec: int, roots: tuple[str, ...] = ("/",)
) -> tuple[bool, str | None]:
    """Remove the link *path* itself, never what it points to."""
    from benchflow._bfx.locate import BfxUnavailable

    if callable(getattr(env, "bfx_call", None)):
        mount = _mount_of(path, roots)
        try:
            await env.bfx_call(
                "remove",
                _wire(
                    _wire({}, "root", mount),
                    "path",
                    path[len(mount.rstrip("/")) :].lstrip("/"),
                ),
                timeout_sec=timeout_sec,
            )
            return True, None
        except BfxUnavailable:
            pass
        except Exception as exc:
            return False, str(exc)
    from benchflow.sandbox._exec_output import exec_raw

    quoted = shlex.quote(path)
    result = await exec_raw(
        env.exec,
        f"[ -L {quoted} ] && rm -f -- {quoted}",
        user="root",
        timeout_sec=timeout_sec,
    )
    code = getattr(result, "return_code", 1)
    return code in (0, None), None if code in (0, None) else f"rc {code}"


async def remove_turn_links(
    env: Any, *, timeout_sec: int | None = None
) -> dict[str, Any]:
    """Before test.sh: the turn's links into the output folder, removed.
    Returns the check's record: ``ran`` (False when no walk could be made
    whole, or no record from before the turn exists), ``why``, ``engine``,
    ``links`` (each removed or not)."""
    timeout_sec = timeout_sec or setup_timeout_sec()
    before = getattr(env, INVENTORY, None)
    if not isinstance(before, Walked):
        # No record: no link can be told from the image's own (/dev/stdin and
        # /etc/mtab lead into /proc), so nothing is removed and the check did
        # not run.
        return {"ran": False, "why": NO_RECORD, "engine": None, "links": []}
    unavailable = walker_unavailable(env)
    if unavailable is not None:
        return {"ran": False, "why": unavailable, "engine": None, "links": []}
    now = await walk(
        env, roots=before.roots if before else None, timeout_sec=timeout_sec
    )
    now = await _with_kernel(env, before, now, timeout_sec)
    rows = turn_links(before, now)
    unresolved = _unresolved(rows)
    if unresolved:
        now.complete = False
        now.why = unresolved if now.why is None else f"{now.why}; {unresolved}"
    rows = [r for r in rows if r.get("class") != UNRESOLVED]
    if rows:
        # Each link goes through its own mount, from the kernel's mount table.
        mounts = await _mount_points(env, timeout_sec) or now.roots
        for row in rows:
            row["removed"], error = await _remove(env, row["path"], timeout_sec, mounts)
            if error:
                row["error"] = error
        # The action took effect only when a look at each path, following no
        # link, finds no link there (F35).
        left, why = await _still_links(
            env, [r["path"] for r in rows if r.get("removed")], timeout_sec
        )
        for row in rows:
            if not row.get("removed"):
                continue
            if why is not None:
                row["removed"], row["error"] = False, why
            elif row["path"] in left:
                row["removed"] = False
                row["error"] = "the link is still there after its removal"
    return {
        "ran": now.complete,
        "why": None if now.complete else now.why,
        "engine": now.engine,
        "links": rows,
    }


async def links_after_grading(
    env: Any, *, timeout_sec: int | None = None
) -> dict[str, Any]:
    """After grading: the turn's links into the output folder there are."""
    timeout_sec = timeout_sec or setup_timeout_sec()
    before = getattr(env, INVENTORY, None)
    if not isinstance(before, Walked):
        # No record: no link can be told from the image's own (/dev/stdin and
        # /etc/mtab lead into /proc), so nothing is removed and the check did
        # not run.
        return {"ran": False, "why": NO_RECORD, "engine": None, "links": []}
    unavailable = walker_unavailable(env)
    if unavailable is not None:
        return {"ran": False, "why": unavailable, "engine": None, "links": []}
    now = await walk(
        env, roots=before.roots if before else None, timeout_sec=timeout_sec
    )
    now = await _with_kernel(env, before, now, timeout_sec)
    rows = turn_links(before, now)
    unresolved = _unresolved(rows)
    if unresolved:
        now.complete = False
        now.why = unresolved if now.why is None else f"{now.why}; {unresolved}"
    return {
        "ran": now.complete,
        "why": None if now.complete else now.why,
        "engine": now.engine,
        "links": [r for r in rows if r.get("class") != UNRESOLVED],
    }


def removed_flag(row: dict) -> str:
    """The flag's text for a link the walk before test.sh found."""
    if row.get("removed"):
        return (
            "removed before test.sh: a link the agent made during its turn "
            f"{row.get('class') or INTO_OUTPUTS} ({shown(row.get('path'))} -> "
            f"{shown(row.get('target'))}); test.sh's own writes are scored"
        )
    return (
        f"not removed before test.sh ({row.get('error') or 'unknown error'}): a "
        f"link the agent made during its turn {row.get('class') or INTO_OUTPUTS} "
        f"({shown(row.get('path'))} -> {shown(row.get('target'))})"
    )


def made_during_grading_reason(rows: list[dict]) -> str:
    listed = ", ".join(
        f"{shown(row.get('path'))} -> {shown(row.get('target'))}" for row in rows[:5]
    )
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return (
        f"a link into {OUTPUTS} was there after grading ({listed}{more}); "
        "BenchFlow can't confirm test.sh wrote this reward"
    )


def check_record(name: str, check: dict[str, Any]) -> dict[str, Any]:
    """What the provenance record lists for one link check."""
    return {
        "check": name,
        "ran": bool(check.get("ran")),
        "why": check.get("why"),
        "engine": check.get("engine"),
    }

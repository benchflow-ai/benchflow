"""Links the agent made into the verifier's output folder ((c)3).

Root's test.sh follows a link. A link the agent left anywhere test.sh writes
(``ln -s /logs/verifier/reward.txt /app/grade.log``) turns test.sh's own
write of some other file into the reward: in integrity off and audit a task
whose test.sh wrote 0 to reward.txt and then ``echo 1 > /app/grade.log``
scored 1.0.

The walks are BenchFlow's own bfx alone (F27: the image's ``find``,
``readlink`` and ``sh`` can be shadowed or replaced by the agent):

- Before the agent's turn, root records every symlink in the sandbox with
  its target and inode (:func:`record_before_turn`): bfx reads the kernel's
  mount table, then walks every mount as its own root, never entering a
  mount point or following a link. A record that is not complete is never
  used: the trial is refused before its turn (``pre_turn_refusal``),
  and a check that meets one anyway removes nothing. A name that vanishes
  while this walk reads it is no error: no agent code exists yet.
- Every walk after the record names each entry that vanished while it read
  it (bfx's ``walk``; its ``scan`` skips one without a word), and any one
  makes the walk incomplete: the agent's live process renaming a folder on
  the way to its link would hide the link (REVIEW-FINAL F2). For a non-root
  agent each such walk (and the removals before test.sh) runs with the
  agent's processes stopped (:mod:`_walk_pause`): a rename between two reads
  of a big folder hides an entry with nothing vanished (C2-1). The look that
  confirms each removal runs after they are continued.
- Before test.sh the same walk runs again: a link that is new since, or
  whose target changed, is the turn's; each turn link that resolves to the
  output folder, into it, to a folder above it, or through /proc is removed
  through its own mount, naming the entry the walk saw (``expect``), and the
  removal is confirmed by a look that follows no link. The image's own links
  stay.
- After grading the walk runs once more: a turn link into the folder there
  then is counted against a non-root agent (Q1), whoever owns it: test.sh's
  own ``chown -R`` can make the agent's link root's (REVIEW-FINAL F1).

Each of bfx's reads here (the mount table, ``scan``, ``walk``, the look at a
removed path) whose exec the sandbox's transport lost is asked again, at most
:data:`READ_ATTEMPTS` times in all (:func:`_bfx_read`; Daytona's proxy
can cut the command channel's websocket mid-answer). A read changes nothing
and only a whole answer is used; a removal is never asked again.

The one program of the image's left in the decision is ``kernel_resolve``'s
``sh`` with ``readlink -m``; its answer only adds links to remove,
never takes one away. A walk that cannot run, or cannot read every link, is
a check that did not run (:mod:`_reward_provenance`).
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
    # Each link's (dev, ino), as bfx saw it: a removal names the entry it
    # removes, and a link after grading that was a link before the turn,
    # moved, is told from a new one.
    ids: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Folders that vanished while the walk read them (a live process): no
    # error, recorded.
    vanished: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "links": dict(self.links),
            "ids": {path: list(ident) for path, ident in self.ids.items()},
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
        ids: dict[str, tuple[int, int]] = {}
        raw_ids = data.get("ids")
        if isinstance(raw_ids, dict):
            for k, v in raw_ids.items():
                if (
                    isinstance(k, str)
                    and isinstance(v, list)
                    and len(v) == 2
                    and all(isinstance(n, int) and not isinstance(n, bool) for n in v)
                ):
                    ids[k] = (v[0], v[1])
        roots = data.get("roots")
        return cls(
            str(data.get("engine") or "record"),
            links=links,
            ids=ids,
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
#: Why a check did not run with a record from before the turn that is not
# complete: its missing links can't be told from the turn's, so
#: nothing is acted on. BenchFlow's gap, never the agent's.
INCOMPLETE_RECORD = "the record of the links before the turn is incomplete"


#: Why a walk did not run where bfx cannot (BenchFlow's gap, never the agent's).
BFX_UNAVAILABLE = "bfx cannot run in this sandbox"


def walker_unavailable(env: Any) -> str | None:
    """Why BenchFlow's walker (bfx) cannot run in *env*, or None when it can:
    a backend with no bfx route, or no pinned bfx delivered (delivery off,
    no binary for the sandbox's arch). A sandbox that knows its own reason
    (``bfx_unavailable_why``: an opt-in route not asked for, a failed
    upload) has it named, under the same head."""
    own = getattr(env, "bfx_unavailable_why", None)
    if isinstance(own, str) and own and getattr(env, "bfx_delivery", None) is None:
        return f"{BFX_UNAVAILABLE}: {own}"
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
        "guarantee": None if why else _guarantee(env),
        "why": why,
    }


def _guarantee(env: Any) -> str | None:
    """Whom bfx holds against in *env* (map-H2 §1.2a): the sandbox's own
    ``bfx_guarantee`` (it knows how bfx is launched), else its delivery's."""
    said = getattr(env, "bfx_guarantee", None)
    if isinstance(said, str):
        return said
    from benchflow._bfx.delivery import GUARANTEE

    delivery = getattr(env, "bfx_delivery", None)
    return GUARANTEE.get(delivery) if isinstance(delivery, str) else None


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

#: How many times in all a bfx read is asked when the sandbox's transport lost
#: its exec (``TransientSandboxTransportError``). A failure that is not the
#: transport's (a cap, a refusal, a timeout) is not asked again.
READ_ATTEMPTS = 3


async def _bfx_read(
    env: Any, cmd: str, args: dict[str, Any], timeout_sec: int
) -> dict[str, Any]:
    """bfx's answer to the read *cmd*, asked again while the sandbox's
    transport loses its exec, :data:`READ_ATTEMPTS` times at most. A walk
    after the turn makes its reads inside the pause (:func:`_paused`), so
    every attempt runs with the agent's processes still stopped."""
    from benchflow.diagnostics import TransientSandboxTransportError

    attempt = 1
    while True:
        try:
            return await env.bfx_call(cmd, args, timeout_sec=timeout_sec)
        except TransientSandboxTransportError as exc:
            if attempt >= READ_ATTEMPTS:
                raise
            attempt += 1
            logger.warning(
                "bfx %s lost its exec to the sandbox's transport (%s); asking again",
                cmd,
                exc,
            )


def _after_turn(env: Any) -> bool:
    """A walk on a sandbox that holds a record from before the turn (its
    first walk's, kept by :func:`record_before_turn`, or one inherited)."""
    return isinstance(getattr(env, INVENTORY, None), Walked)


async def _bfx_scan(env: Any, roots: list[str], timeout_sec: int) -> dict[str, Any]:
    """bfx's reading of every entry under *roots*: ``scan`` for the record
    before the turn; after it, ``walk`` (:func:`_bfx_walk`), which names
    each entry that vanished mid-walk."""
    if _after_turn(env):
        return await _bfx_walk(env, roots, timeout_sec)
    return await _bfx_read(
        env,
        "scan",
        {
            "walks": [{"root": r, "depth": 4096} for r in roots],
            "mounts": False,
            "acl": False,
            "max_bytes": 512 << 20,
            "max_entries": SCAN_MAX_ENTRIES,
        },
        timeout_sec,
    )


async def _bfx_walk(env: Any, roots: list[str], timeout_sec: int) -> dict[str, Any]:
    """Every entry under *roots* by bfx's ``walk``, one root at a time, each
    folder it hands back (``deeper``) walked as a root of its own, in
    ``scan``'s shape with absolute paths, plus ``vanished``: every entry
    that was listed but gone when bfx looked at it. A file bind mount
    (/etc/hosts) among *roots* has nothing beneath it. Past
    :data:`SCAN_MAX_ENTRIES` names in all it fails, as ``scan`` does."""
    from benchflow._bfx.client import BfxError

    entries: list[dict[str, Any]] = []
    vanished: list[str] = []
    tops = list(roots)
    left = SCAN_MAX_ENTRIES
    while tops:
        top = tops.pop(0)
        try:
            answer = await _bfx_read(
                env,
                "walk",
                {**_wire({}, "root", top), "path": "", "max_entries": max(left, 0)},
                timeout_sec,
            )
        except BfxError as error:
            if top in roots and top != "/" and error.kind == "not-dir":
                continue
            raise
        rows = answer.get("entries") if isinstance(answer, dict) else None
        if not isinstance(rows, list):
            raise BfxError("bad-answer", f"the walk of {shown(top)} listed nothing")
        left -= len(rows)
        for e in rows:
            rel = _name(e, "path") if isinstance(e, dict) else None
            if rel is None:
                raise BfxError(
                    "bad-answer", f"an entry with no name under {shown(top)}"
                )
            row = {k: v for k, v in e.items() if k != "path_b64"}
            row["path"] = posixpath.join(top, rel)
            entries.append(row)
        for key, into in (("vanished", vanished), ("deeper", tops)):
            for e in answer.get(key) or ():
                rel = _name(e, "path") if isinstance(e, dict) else None
                if rel is None:
                    raise BfxError(
                        "bad-answer", f"a {key} entry with no name under {shown(top)}"
                    )
                into.append(posixpath.join(top, rel))
    return {"entries": entries, "errors": [], "vanished": vanished}


#: How a walk that bfx failed reads (the record's ``why``).
_SCAN_FAILED = "bfx scan failed: "
_WALK_FAILED = "bfx walk failed: "


def _mount_roots(table: Any) -> tuple[list[str], str | None]:
    """The mount points to walk, each as its own root, from bfx's reading of
    the kernel's mount table (``scan {mounts}``), and why that list is not
    whole (None when it is). /proc, /sys and the output folder, and every
    mount under them, are not walked."""
    rows = table.get("mounts") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return ["/"], "bfx gave no mount table"
    roots = {"/"}
    why = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        point = _name(row, "mount_point")
        if point is None or not point.startswith("/") or _lossy(point):
            # A mount point the table's text decode could not keep names no
            # folder the walk can open (F40).
            why = f"a mount point the walk cannot name ({shown(point)})"
            continue
        if not _skipped(point):
            roots.add(point)
    return sorted(roots), why


#: bfx's word for a directory read refused because the root is not a folder,
#: as gVisor gives it: its open of a file root with ``O_DIRECTORY`` succeeds
#: and the read fails (``getdents``), where Linux refuses the open itself.
_LIST_NOT_A_FOLDER = "reading a directory: ENOTDIR"


def _file_mount_root(error: dict[str, Any], path: str | None, roots: list[str]) -> bool:
    """A mount root the kernel's table names that is a file (/etc/hosts,
    /etc/resolv.conf): nothing beneath it. bfx reports it as Linux refuses it
    (``walk``, ``not-directory``) or as gVisor does (Modal: ``list``,
    ``reading a directory: ENOTDIR``). Any other error, or the same one
    beneath a root, is a folder the walk could not read."""
    if path not in roots:
        return False
    if error.get("op") == "walk" and error.get("error") == "not-directory":
        return True
    return (
        path != "/"
        and error.get("op") == "list"
        and error.get("error") == _LIST_NOT_A_FOLDER
    )


def _vanished(error: dict[str, Any]) -> bool:
    """A folder removed between its listing and its reading (a live process):
    bfx's own words, which no name in the sandbox can make."""
    said = str(error.get("error") or "")
    return error.get("op") == "list" and (said == "missing" or "ENOENT" in said)


async def _walk_bfx(env: Any, timeout_sec: int) -> Walked | None:
    """Every link, by bfx: the mount table first, then one walk with every
    mount as its own root. A name that vanished mid-walk is no error before
    the turn; after it, any one makes the walk incomplete. None where bfx is
    not there to run."""
    from benchflow._bfx.locate import BfxUnavailable

    if not callable(getattr(env, "bfx_call", None)):
        return None
    after_turn = _after_turn(env)
    try:
        table = await _bfx_read(
            env, "scan", {"mounts": True, "acl": False}, timeout_sec
        )
        roots, unnamed = _mount_roots(table)
        answer = await _bfx_scan(env, roots, timeout_sec)
    except BfxUnavailable:
        return None
    except Exception as exc:
        failed = _WALK_FAILED if after_turn else _SCAN_FAILED
        return Walked("bfx", complete=False, why=f"{failed}{exc}")
    walked = Walked("bfx", roots=tuple(roots))
    if unnamed:
        walked.complete = False
        walked.why = unnamed
    for e in answer.get("entries") or ():
        if not isinstance(e, dict) or e.get("type") != "link":
            continue
        path = _name(e, "path")
        target = _name(e, "link")
        if path is None or target is None:
            # An entry the record cannot hold is never dropped (F40).
            walked.complete = False
            walked.why = f"a link the walk could not record: {sorted(e)}"
            continue
        walked.links[path] = target
        dev, ino = e.get("dev"), e.get("ino")
        if isinstance(dev, int) and isinstance(ino, int):
            walked.ids[path] = (dev, ino)
    errors = []
    for error in answer.get("errors") or ():
        if not isinstance(error, dict):
            continue
        path = _name(error, "path")
        if _vanished(error) and path is not None and not after_turn:
            walked.vanished.append(path)
            continue
        if _file_mount_root(error, path, roots):
            continue  # a file bind mount (/etc/hosts): nothing beneath it
        errors.append(error)
    if errors:
        walked.complete = False
        walked.why = f"bfx scan could not read {len(errors)} path(s), e.g. {errors[0]}"
    gone = [p for p in answer.get("vanished") or () if isinstance(p, str)]
    if gone:
        walked.vanished.extend(gone)
        walked.complete = False
        said = (
            f"{len(gone)} name(s) vanished while the walk read them (a live "
            f"process), e.g. {shown(gone[0])}"
        )
        walked.why = said if walked.why is None else f"{walked.why}; {said}"
    return walked


async def walk(env: Any, *, roots: tuple[str, ...] | None, timeout_sec: int) -> Walked:
    """Every symlink in the sandbox, by BenchFlow's own bfx alone: the
    image's find, readlink and sh can be shadowed or replaced by the agent
    (F27). A failed or capped bfx walk is an incomplete one, and so is one
    after the record from before the turn that a name vanished from. *roots*
    is unused: every walk takes its roots from the kernel's mount table
    afresh (a mount made since is never missed)."""
    walked = await _walk_bfx(env, timeout_sec)
    if walked is None:
        return Walked(
            "bfx", complete=False, why=walker_unavailable(env) or BFX_UNAVAILABLE
        )
    return walked


def _transient(walked: Walked) -> bool:
    """A walk bfx failed on (a timeout, a lost exec), not one over a cap,
    which would only fail the same way again."""
    why = walked.why or ""
    return why.startswith(_SCAN_FAILED) and not any(
        f"bfx {kind}:" in why for kind in ("too-many", "too-large")
    )


async def record_before_turn(
    env: Any, *, timeout_sec: int | None = None
) -> Walked | None:
    """The record of every link before the agent's turn, kept on the host
    (``env._benchflow_link_inventory``) and returned. The first record of a
    sandbox is the one kept: a fresh verifier sandbox's, made before the
    frozen workspace arrives, or the one a branch child inherits, is never
    replaced by a walk that already sees the agent's links. A walk bfx
    failed on is tried once more (no agent process exists yet); a record
    still incomplete is refused by the caller."""
    kept = getattr(env, INVENTORY, None)
    if isinstance(kept, Walked):
        return kept
    timeout_sec = timeout_sec or setup_timeout_sec()
    walked = await walk(env, roots=None, timeout_sec=timeout_sec)
    if not walked.complete and _transient(walked):
        logger.warning(
            "the walk before the agent's turn failed (%s); trying it once more",
            walked.why,
        )
        walked = await walk(env, roots=None, timeout_sec=timeout_sec)
    with contextlib.suppress(AttributeError):
        setattr(env, INVENTORY, walked)
    if not walked.complete:
        logger.warning(
            "the links before the agent's turn were not all recorded: %s", walked.why
        )
    return walked


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
        answer = await _bfx_read(
            env, "scan", {"mounts": True, "acl": False}, timeout_sec
        )
    except Exception:
        return None
    rows = answer.get("mounts")
    if not isinstance(rows, list):
        return None
    points = {_name(m, "mount_point") for m in rows if isinstance(m, dict)}
    return tuple(sorted(p for p in points if p))


def _paths_request(paths: list[str]) -> dict[str, Any]:
    """A bfx ``scan`` request naming *paths* (text, or base64 for bytes that
    are not UTF-8)."""
    text = [p for p in paths if "paths" in _wire({}, "paths", p)]
    raw = [p for p in paths if p not in text]
    request: dict[str, Any] = {}
    if text:
        request["paths"] = text
    if raw:
        request["paths_b64"] = [
            base64.b64encode(p.encode("utf-8", "surrogateescape")).decode("ascii")
            for p in raw
        ]
    return request


async def _still_links(
    env: Any, paths: list[str], timeout_sec: int
) -> tuple[set[str], str | None]:
    """Which of *paths* are links now, looked at component by component
    from / without following any (bfx ``scan`` paths); and why the look
    could not be made (None when it was)."""
    if not paths:
        return set(), None
    if not callable(getattr(env, "bfx_call", None)):
        return set(), f"the removal could not be confirmed: {BFX_UNAVAILABLE}"
    request = {"acl": False, "mounts": False, **_paths_request(paths)}
    try:
        answer = await _bfx_read(env, "scan", request, timeout_sec)
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


async def _remove(
    env: Any,
    path: str,
    timeout_sec: int,
    roots: tuple[str, ...] = ("/",),
    expect: tuple[int, int] | None = None,
) -> tuple[bool, str | None]:
    """Remove the link *path* itself, never what it points to: bfx
    ``remove`` through the link's own mount, naming the entry the walk saw
    (*expect*, its dev and inode), so a name swapped in since is refused. No
    program of the image's removes it (F27)."""
    from benchflow._bfx.client import BfxError
    from benchflow._bfx.locate import BfxUnavailable

    if not callable(getattr(env, "bfx_call", None)):
        return False, BFX_UNAVAILABLE
    mount = _mount_of(path, roots)
    args = _wire(
        _wire({}, "root", mount), "path", path[len(mount.rstrip("/")) :].lstrip("/")
    )
    if expect is not None:
        args["expect"] = {"dev": expect[0], "ino": expect[1]}
    try:
        await env.bfx_call("remove", args, timeout_sec=timeout_sec)
        return True, None
    except BfxUnavailable as exc:
        return False, f"bfx became unavailable: {exc}"
    except BfxError as exc:
        if exc.kind == "changed":
            return False, f"the link changed before its removal ({exc.message})"
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def _not_whole(before: Walked) -> dict[str, Any]:
    """A check's record for a record from before the turn that is not
    complete: nothing is walked or removed."""
    return {
        "ran": False,
        "why": f"{INCOMPLETE_RECORD}: {before.why or 'unknown why'}",
        "engine": None,
        "links": [],
    }


async def _paused(
    env: Any, user: str | None, timeout_sec: int, check: Any
) -> dict[str, Any]:
    """*check()*, a walk after the turn, run while *user*'s processes are
    held still (:mod:`_walk_pause`, C2-1): a live process renaming names in a
    big folder can move a link past a walk's directory read with nothing
    vanished. With *user* None (a root agent) it runs as it is: a pause
    can't hold root's processes, which can continue anything. A root agent's
    reward is not unverifiable by that alone: on Docker the walker holds
    against root (``walker-holds-against-root`` passes), and what keeps the
    label from ``verified`` is ``root-agent-processes``, the inventory of
    the turn's processes still alive when grading starts (C3-8). A pause
    that can't be shown, or did not hold, leaves the check incomplete; what
    it stopped is continued either way. A process the pause could not hold
    (one its own timer continues, C3-1; one that changes its pid faster than
    a pass reads /proc, C3-2) is still alive when test.sh starts, so rule S
    (:mod:`_agent_alive`) keeps such a trial from reading ``verified``."""
    from benchflow.sandbox import _walk_pause

    if user is None:
        return await check()
    held = await _walk_pause.pause(env, user, timeout_sec=timeout_sec)
    try:
        found = await check()
    finally:
        held = await _walk_pause.resume(env, held, timeout_sec=timeout_sec)
    found["pause"] = held.to_json()
    if not held.ok():
        why = held.why or _walk_pause.NOT_PAUSED
        found["ran"] = False
        found["why"] = why if found.get("why") is None else f"{found['why']}; {why}"
    return found


async def remove_turn_links(
    env: Any, *, timeout_sec: int | None = None, pause_user: str | None = None
) -> dict[str, Any]:
    """Before test.sh: the turn's links into the output folder, removed.
    Returns the check's record: ``ran`` (False when no walk could be made
    whole, or no whole record from before the turn exists), ``why``,
    ``engine``, ``links`` (each removed or not), and ``pause``. The walk
    and the removals run while *pause_user*'s processes are held still
    (:func:`_paused`); the confirming look runs once they are continued, so
    a process that puts its link back the moment it is removed is seen
    (813-M1, probe P5), as before the pause."""
    timeout_sec = timeout_sec or setup_timeout_sec()
    before = getattr(env, INVENTORY, None)
    if not isinstance(before, Walked):
        # No record: no link can be told from the image's own (/dev/stdin and
        # /etc/mtab lead into /proc), so nothing is removed and the check did
        # not run.
        return {"ran": False, "why": NO_RECORD, "engine": None, "links": []}
    if not before.complete:
        # A record with links missing reads the image's own as the turn's
        # (probe P4: /dev/stdin removed before test.sh): never acted on.
        return _not_whole(before)
    unavailable = walker_unavailable(env)
    if unavailable is not None:
        return {"ran": False, "why": unavailable, "engine": None, "links": []}
    found = await _paused(
        env,
        pause_user,
        timeout_sec,
        lambda: _walk_and_remove(env, before, timeout_sec),
    )
    await _confirm_removals(env, found["links"], timeout_sec)
    return found


async def _walk_and_remove(
    env: Any, before: Walked, timeout_sec: int
) -> dict[str, Any]:
    now = await walk(env, roots=None, timeout_sec=timeout_sec)
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
            row["removed"], error = await _remove(
                env, row["path"], timeout_sec, mounts, expect=now.ids.get(row["path"])
            )
            if error:
                row["error"] = error
    return {
        "ran": now.complete,
        "why": None if now.complete else now.why,
        "engine": now.engine,
        "links": rows,
    }


async def _confirm_removals(
    env: Any, rows: list[dict[str, Any]], timeout_sec: int
) -> None:
    """The action took effect only when a look at each removed path,
    following no link, finds no link there (F35)."""
    paths = [r["path"] for r in rows if r.get("removed")]
    if not paths:
        return
    left, why = await _still_links(env, paths, timeout_sec)
    for row in rows:
        if not row.get("removed"):
            continue
        if why is not None:
            row["removed"], row["error"] = False, why
        elif row["path"] in left:
            row["removed"] = False
            row["error"] = "the link is still there after its removal"


async def links_after_grading(
    env: Any,
    *,
    timeout_sec: int | None = None,
    pause_user: str | None = None,
) -> dict[str, Any]:
    """After grading: the turn's links into the output folder there are.
    Every one counts, whoever made it: a link test.sh made, or one it
    locked down to root afterwards, can't be told from the agent's. The
    walk runs while *pause_user*'s processes are held still
    (:func:`_paused`)."""
    timeout_sec = timeout_sec or setup_timeout_sec()
    before = getattr(env, INVENTORY, None)
    if not isinstance(before, Walked):
        # No record: no link can be told from the image's own (/dev/stdin and
        # /etc/mtab lead into /proc), so nothing is removed and the check did
        # not run.
        return {"ran": False, "why": NO_RECORD, "engine": None, "links": []}
    if not before.complete:
        return _not_whole(before)
    unavailable = walker_unavailable(env)
    if unavailable is not None:
        return {"ran": False, "why": unavailable, "engine": None, "links": []}
    return await _paused(
        env, pause_user, timeout_sec, lambda: _walk_now(env, before, timeout_sec)
    )


async def _walk_now(env: Any, before: Walked, timeout_sec: int) -> dict[str, Any]:
    now = await walk(env, roots=None, timeout_sec=timeout_sec)
    now = await _with_kernel(env, before, now, timeout_sec)
    rows = turn_links(before, now)
    unresolved = _unresolved(rows)
    if unresolved:
        now.complete = False
        now.why = unresolved if now.why is None else f"{now.why}; {unresolved}"
    rows = [r for r in rows if r.get("class") != UNRESOLVED]
    return {
        "ran": now.complete,
        "why": None if now.complete else now.why,
        "engine": now.engine,
        "links": rows,
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

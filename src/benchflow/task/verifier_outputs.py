"""test.sh's outputs as they were the moment it exited (CIR's L14).

In integrity off and audit the agent's processes live through grading, and a mounted ``/logs/verifier`` stays writable from
the sandbox until it is gone. So the verifier copies the folder into the
trial, where no sandbox mount reaches (``verifier-at-exit/``), the moment the
test command has ended; the rewards are read from that copy, and once nothing
in the sandbox can write any more the rollout puts it back over ``verifier/``
(:func:`put_back`). Only the names test.sh left go back: each one is
recorded at the exit (files, folders, links and others, never followed).
BenchFlow's own later records are kept as BenchFlow wrote them: every
write BenchFlow makes there after the copy goes through a held capability
(``reachable_write``), which tells this module of it; any other change is
undone. A name nobody of those left (a file, a folder, a link a
process in the sandbox added after the exit) is removed from ``verifier/``,
a reward file of that kind moved to ``verifier-after-exit/``, and each one is
a flag in the trial's ``integrity/verifier-inputs.json``. Nothing here
changes the reward: it was read from the copy already.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import os
import stat
import threading
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from benchflow._utils.reachable_write import ReachablePath, observe_writes

logger = logging.getLogger(__name__)

AT_EXIT = "verifier-at-exit"
AFTER_EXIT = "verifier-after-exit"
#: Files copied: each up to the first limit, all together up to the second;
#: a larger one is kept by its sha256 alone.
FILE_MAX = 64 << 20
TOTAL_MAX = 512 << 20
REWARD_NAMES = ("reward.txt", "reward.json")
#: What the host reads to score, granted first by the read before the copy
#: the reward files, their details, the criteria runner's
#: manifest and the CTRF report.
PRIORITY = (
    *REWARD_NAMES,
    "reward-details.json",
    "criteria-runner-manifest.json",
    "ctrf.json",
)
#: The kinds of name a walk records.
FILE, FOLDER, LINK, OTHER = "file", "folder", "link", "other"
#: Folders in a removed tree counted for its flag, at most.
_COUNT_MAX = 10_000
#: BenchFlow's own writes to each trial's verifier/ since test.sh's exit
#: (:func:`wrote`), by the folder; the oldest go past this many trials.
_JOURNAL_MAX = 4096
_JOURNALS: OrderedDict[Path, dict[str, bytes | str | None]] = OrderedDict()
_JOURNAL_LOCK = threading.Lock()
#: Set while put_back writes: its own restores are not BenchFlow's later
#: records (they would keep a name a later put_back must check again).
_PUTTING_BACK: ContextVar[bool] = ContextVar("benchflow_putting_back", default=False)


def _private(rollout_dir: Path, name: str) -> ReachablePath:
    """A folder of the trial no sandbox mount reaches."""
    folder = Path(rollout_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return ReachablePath.directory(folder)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return FILE
    if stat.S_ISDIR(mode):
        return FOLDER
    if stat.S_ISLNK(mode):
        return LINK
    return OTHER


def _walk(
    root: ReachablePath, descend: Any = None
) -> dict[str, tuple[str, ReachablePath]]:
    """Every name under *root* by its relative path, with its kind, read
    through no-follow handles: a link is a name, never followed. A folder is
    walked into when *descend(rel)* says so (always, without one)."""
    found: dict[str, tuple[str, ReachablePath]] = {}
    if not root.is_dir():
        return found
    pending: list[tuple[str, ReachablePath]] = [("", root)]
    while pending:
        prefix, folder = pending.pop()
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            rel = prefix + child.name
            try:
                kind = _kind(child.stat().st_mode)
            except OSError:
                continue
            found[rel] = (kind, child)
            if kind == FOLDER and (descend is None or descend(rel)):
                pending.append((rel + "/", child))
    return found


def _files(root: ReachablePath) -> dict[str, ReachablePath]:
    """Every regular file under *root* by its relative path (links left out)."""
    return {rel: path for rel, (kind, path) in _walk(root).items() if kind == FILE}


def _spelled(folder: Path) -> Path:
    """*folder* spelled as a held anchor spells its path: absolute, the parent
    resolved (``ReachablePath.relative_to``)."""
    absolute = Path(os.path.abspath(folder))
    return absolute.parent.resolve() / absolute.name


def _open_journal(rollout_dir: Path) -> None:
    folder = _spelled(Path(rollout_dir) / "verifier")
    with _JOURNAL_LOCK:
        _JOURNALS.pop(folder, None)
        _JOURNALS[folder] = {}
        while len(_JOURNALS) > _JOURNAL_MAX:
            _JOURNALS.popitem(last=False)


def _journal(rollout_dir: Path) -> dict[str, bytes | str | None]:
    folder = _spelled(Path(rollout_dir) / "verifier")
    with _JOURNAL_LOCK:
        return dict(_JOURNALS.get(folder) or {})


class _Listed(str):
    """A name inside a tree BenchFlow published, by its kind when it was
    published: kept as BenchFlow left it, a name added there later is not."""


def _note(folder: Path, rel: str, data: bytes | None) -> None:
    """Note BenchFlow's write of *rel* under the trial folder *folder*. A
    tree written as a whole (*data* None: a publish, a rename) is listed as
    it is now, read through a held folder, so what appears in it later is
    told apart."""
    entries: dict[str, bytes | str | None] = {rel: data}
    if data is None:
        with contextlib.suppress(OSError, ValueError):
            target = ReachablePath.directory(folder) / rel
            kind = _kind(target.stat().st_mode)
            entries[rel] = _Listed(kind)
            if kind == FOLDER:
                entries.update(
                    {
                        f"{rel}/{sub}": _Listed(sub_kind)
                        for sub, (sub_kind, _path) in _walk(target).items()
                    }
                )
    with _JOURNAL_LOCK:
        journal = _JOURNALS.get(folder)
        if journal is None:
            return
        for name in [n for n in journal if n.startswith(rel + "/")]:
            del journal[name]
        journal.update(entries)


def _observed(folder: Path, parts: tuple[str, ...], data: bytes | None) -> None:
    """Every write through a held capability (reachable_write): one under a
    trial's verifier/ whose outputs were taken is BenchFlow's own, noted."""
    if _PUTTING_BACK.get():
        return
    with _JOURNAL_LOCK:
        if not _JOURNALS:
            return
        found = next(
            (
                (folder.joinpath(*parts[:i]), "/".join(parts[i:]))
                for i in range(len(parts) + 1)
                if folder.joinpath(*parts[:i]) in _JOURNALS
            ),
            None,
        )
    if found is not None and found[1]:
        _note(found[0], found[1], data)


observe_writes(_observed)


def wrote(path: ReachablePath | Path, data: bytes | None = None) -> None:
    """BenchFlow itself wrote *path*, a name in a trial's ``verifier/``,
    after test.sh's exit, other than through a held capability (whose writes
    are noted as they happen): :func:`put_back` keeps it. *data*, the bytes
    written, is compared and put back as the verifier's own records are;
    None keeps the name as it is (a tree BenchFlow published, listed as it is
    now; an entry it renamed; a file it appended to). A name in no trial
    whose outputs were taken is no one's here, and nothing is noted."""
    with _JOURNAL_LOCK:
        folders = list(_JOURNALS)
    for folder in folders:
        try:
            rel = (
                path.relative_to(folder)
                if isinstance(path, ReachablePath)
                else _spelled(path).relative_to(folder)
            )
        except ValueError:
            continue
        if rel.parts:
            _note(folder, rel.as_posix(), data)
        return


def unreadable_top(rollout_dir: Path) -> list[str]:
    """The regular files directly in the rollout's ``verifier/`` this host
    can see but not open, by name: on mounted outputs, root's 0600 files a
    test.sh left (``mkstemp`` and a rename, or ``umask 077``). The names the
    host reads to score (:data:`PRIORITY`) come first, the rest after them,
    sorted. Nothing in a subfolder: a subfolder may be another user's to
    change, and no reward is read from one."""
    root = ReachablePath.rollout(rollout_dir, "verifier")
    found: list[str] = []
    if not root.is_dir():
        return found
    for path in root.iterdir():
        try:
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb"):
                pass
        except PermissionError:
            found.append(path.name)
        except OSError:
            continue
    first = {name: i for i, name in enumerate(PRIORITY)}
    return sorted(found, key=lambda name: (first.get(name, len(first)), name))


def take(rollout_dir: Path) -> dict[str, Any]:
    """Copy the rollout's ``verifier/`` to ``verifier-at-exit/`` and return
    what was taken: ``{"files": {rel: {"sha256", "size", "copied"}},
    "special": [...], "names": {rel: {"kind", "target"?}}}``, every name
    test.sh left by its kind (a link's target with it). BenchFlow's own
    writes from here on are noted (:func:`wrote`)."""
    live = ReachablePath.rollout(rollout_dir, "verifier")
    snap = _private(rollout_dir, AT_EXIT)
    entries = _walk(live)
    names: dict[str, dict[str, Any]] = {}
    for rel, (kind, path) in entries.items():
        names[rel] = {"kind": kind}
        if kind == LINK:
            try:
                names[rel]["target"] = path.readlink_text()
            except OSError:
                names[rel]["target"] = None
    files: dict[str, dict[str, Any]] = {}
    unread: list[str] = []
    used = 0
    for rel, (kind, path) in sorted(entries.items()):
        if kind != FILE:
            continue
        try:
            size = path.stat().st_size
            if size > FILE_MAX or used + size > TOTAL_MAX:
                data = path.read_bytes()
                files[rel] = {"sha256": _sha(data), "size": len(data), "copied": False}
                unread.append(rel)
                continue
            data = path.read_bytes()
        except OSError:
            unread.append(rel)
            continue
        target = snap
        for part in rel.split("/")[:-1]:
            target = target / part
            target.mkdir(exist_ok=True)
        (target / rel.split("/")[-1]).write_bytes(data)
        files[rel] = {"sha256": _sha(data), "size": len(data), "copied": True}
        used += len(data)
    # A reward name test.sh left as something else (a folder, a link), or
    # one this copy could not read or keep: the rewards are read where it is,
    # so they fail (or are read) as they were before.
    special = [name for name in REWARD_NAMES if name in unread]
    for name in REWARD_NAMES:
        path = live / name
        try:
            there = path.is_symlink() or path.exists()
            regular = path.is_file() and not path.is_symlink()
        except OSError:
            there, regular = True, False
        if there and not regular and name not in special:
            special.append(name)
    _open_journal(Path(rollout_dir))
    return {"files": files, "special": special, "names": names}


def _parents(rel: str) -> list[str]:
    parts = rel.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def _under(rel: str, tops: set[str]) -> bool:
    return any(rel.startswith(top + "/") for top in tops)


def _held_facts(path: ReachablePath, kind: str) -> dict[str, Any]:
    """What a name to remove is, read through the held folder: never
    followed, never opened unless it is a regular file."""
    try:
        if kind == FILE:
            with path.open("rb") as handle:
                info = os.fstat(handle.fileno())
                facts: dict[str, Any] = {"type": FILE, "size": info.st_size}
                if info.st_size <= FILE_MAX:
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                    facts["sha256"] = digest.hexdigest()
            return facts
        if kind == LINK:
            return {"type": LINK, "target": path.readlink_text()[:512]}
        if kind == FOLDER:
            count = 0
            pending = [path]
            while pending and count < _COUNT_MAX:
                for child in pending.pop().iterdir():
                    count += 1
                    if child.is_dir():
                        pending.append(child)
            entries: int | str = count if count < _COUNT_MAX else f"{_COUNT_MAX}+"
            return {"type": FOLDER, "entries": entries}
        return {"type": OTHER, "mode": oct(path.stat().st_mode)}
    except OSError as exc:
        return {"type": kind, "unread": str(exc)[:200]}


def _remove_held(path: ReachablePath, kind: str) -> None:
    """Remove *path* through its held parent folder, never by a path name a
    sandbox process could swap: a name with ``unlinkat``, a folder by
    ``reachable_write``'s walk below it (each folder opened without following
    a link or crossing a mount, ``fchmod`` on its descriptor, entries removed
    relative to it). Whatever can't be removed that way raises, and stays."""
    if kind != FOLDER:
        path.unlink()
        return
    from benchflow._utils.reachable_write import _remove_beneath

    with path._folders() as handles:
        parent = handles[-1]
        if not isinstance(parent, int):
            raise OSError(errno.ENOTSUP, "no folder descriptor to remove through")
        _remove_beneath(parent, path.name, where=str(path))


def _keep_aside(rollout_dir: Path, rel: str, path: ReachablePath) -> None:
    """A reward file nobody of BenchFlow's or test.sh's wrote: a copy in
    ``verifier-after-exit/`` for the record (up to :data:`FILE_MAX`)."""
    if path.stat().st_size > FILE_MAX:
        return
    data = path.read_bytes()
    (_private(rollout_dir, AFTER_EXIT) / rel.replace("/", "__")).write_bytes(data)


def put_back(
    rollout_dir: Path,
    taken: dict[str, Any],
    written: dict[str, bytes | None],
    *,
    remove: bool = False,
) -> list[dict[str, Any]]:
    """Put test.sh's outputs at its exit back over ``verifier/``, and
    BenchFlow's own later records (*written*, and what :func:`wrote` noted:
    rel -> bytes, or None for one kept as it is) as it wrote them. Every
    other name is flagged; with *remove* (only once the sandbox is gone, so
    no process in it can still act on what is removed) it is also removed,
    through the held folder alone. Returns one flag per difference undone or
    found."""
    token = _PUTTING_BACK.set(True)
    try:
        return _put_back(Path(rollout_dir), taken, written, remove=remove)
    finally:
        _PUTTING_BACK.reset(token)


def _put_back(
    rollout_dir: Path,
    taken: dict[str, Any],
    written: dict[str, bytes | None],
    *,
    remove: bool,
) -> list[dict[str, Any]]:
    live = ReachablePath.rollout(rollout_dir, "verifier")
    snap = _private(rollout_dir, AT_EXIT)
    files = taken.get("files") or {}
    at_exit = taken.get("names")
    if not isinstance(at_exit, dict):
        # A copy taken before names were recorded: its files, in their folders.
        at_exit = {rel: {"kind": FILE} for rel in files}
    journal = _journal(Path(rollout_dir))
    later: dict[str, bytes | str | None] = {**written, **journal}
    # A name both the verifier's own record and another writer wrote: either
    # version is BenchFlow's (the order between the two is not recorded).
    versions = {
        rel: {
            _sha(d)
            for d in (written.get(rel), journal.get(rel))
            if isinstance(d, bytes)
        }
        for rel in later
    }
    flags: list[dict[str, Any]] = []

    def flag(rel: str, kind: str, before: Any, after: Any, **more: Any) -> None:
        flags.append(
            {
                "where": "verifier_output",
                "path": f"/logs/verifier/{rel}",
                "kind": kind,
                "category": "verifier-output",
                "before": before,
                "after": after,
                **more,
            }
        )

    # What may be there: test.sh's names at its exit, BenchFlow's later ones
    # and the folders they are in. A name BenchFlow keeps as it is (None)
    # keeps whatever is below it too.
    expected: dict[str, str | None] = {
        rel: entry.get("kind") for rel, entry in at_exit.items()
    }
    for rel, data in later.items():
        if isinstance(data, _Listed):
            expected[rel] = str(data)
        else:
            expected[rel] = FILE if data is not None else None
    # A tree BenchFlow published was listed then, so its names are checked
    # like the rest; only a name whose kind could not be read is kept as it
    # is, with whatever is below it.
    for rel in list(expected):
        for parent in _parents(rel):
            if expected.get(parent, FOLDER) == FOLDER:
                expected[parent] = FOLDER
    kept_trees = {rel for rel, data in later.items() if data is None}

    def descend(rel: str) -> bool:
        return expected.get(rel) == FOLDER and rel not in kept_trees

    current = _walk(live, descend)
    removed: set[str] = set()
    # Names test.sh or BenchFlow had that something else took the place of:
    # flagged once, here, and made again below.
    replaced: set[str] = set()
    # Names flagged and still there (the sandbox is up, or the removal
    # failed): nothing is put back at or below them.
    left: set[str] = set()
    for rel, (kind, path) in sorted(current.items()):
        if _under(rel, removed | replaced) or _under(rel, kept_trees):
            continue
        want = expected.get(rel, "")
        if want == kind or (want is None and kind != LINK):
            continue
        # A name BenchFlow keeps as it is (want None) is never a link of
        # BenchFlow's own: one there took its place.
        before: dict[str, Any] | None = (
            {"by": "benchflow"} if want is None else {"type": want} if want else None
        )
        (removed if before is None else replaced).add(rel)
        facts = _held_facts(path, kind)
        more: dict[str, Any] = {"removed": False}
        if not remove:
            # While the sandbox lives a process in it may still hold the name
            # (its cwd, an open folder): named now, removed once it is gone.
            more["pending"] = "removed once the sandbox is gone"
        else:
            try:
                if (
                    before is None
                    and kind == FILE
                    and rel.split("/")[-1] in REWARD_NAMES
                ):
                    _keep_aside(rollout_dir, rel, path)
                _remove_held(path, kind)
                more["removed"] = True
            except OSError as exc:
                more["error"] = str(exc)[:200]
        if not more["removed"]:
            left.add(rel)
        flag(rel, "planted" if before is None else "edited", before, facts, **more)

    # test.sh's files and BenchFlow's records back, as they were written.
    present = _walk(live, descend)
    restore = {r for r in files if r not in later and not _under(r, kept_trees)}
    restore |= {r for r, d in later.items() if isinstance(d, bytes)}
    restore = {r for r in restore if r not in left and not _under(r, left)}
    for rel in sorted(restore):
        data = later.get(rel)
        if isinstance(data, bytes):
            want_bytes: bytes | None = data
        elif files[rel].get("copied"):
            want_bytes = (snap / rel).read_bytes()
        else:
            want_bytes = None
        found = present.get(rel)
        try:
            have = found[1].read_bytes() if found and found[0] == FILE else None
        except OSError:
            have = None
        expected_sha = (
            _sha(want_bytes) if want_bytes is not None else files[rel]["sha256"]
        )
        if have is not None and _sha(have) in (versions.get(rel) or {expected_sha}):
            continue
        before = {"sha256": expected_sha}
        after = {"sha256": _sha(have), "size": len(have)} if have is not None else None
        kind = "edited" if have is not None else "removed"
        said = rel in replaced
        if want_bytes is None:
            if not said:
                flag(rel, kind, before, after)
            continue
        try:
            for parent in _parents(rel):
                (live / parent).mkdir(exist_ok=True)
            (live / rel).write_bytes(want_bytes)
        except OSError as exc:
            flag(rel, kind, before, after, restored=False, error=str(exc)[:200])
            continue
        if not said:
            flag(rel, kind, before, after)

    # test.sh's folders and links: a folder gone is made again (empty), a
    # link gone or pointing elsewhere is named.
    for rel, entry in sorted(at_exit.items()):
        kind = entry.get("kind")
        if kind not in (FOLDER, LINK) or rel in later or _under(rel, kept_trees):
            continue
        if rel in left or _under(rel, left):
            continue
        found = present.get(rel)
        if kind == FOLDER and found is None:
            with contextlib.suppress(OSError):
                (live / rel).mkdir(exist_ok=True)
            if rel not in replaced:
                flag(rel, "removed", {"type": FOLDER}, None)
        elif kind == LINK:
            if found is None:
                if rel not in replaced:
                    flag(
                        rel,
                        "removed",
                        {"type": LINK, "target": entry.get("target")},
                        None,
                    )
                continue
            try:
                now = found[1].readlink_text()
            except OSError:
                now = None
            if now != entry.get("target"):
                gone = False
                if remove:
                    with contextlib.suppress(OSError):
                        found[1].unlink()
                        gone = True
                flag(
                    rel,
                    "edited",
                    {"type": LINK, "target": entry.get("target")},
                    {"type": LINK, "target": now},
                    removed=gone,
                    **(
                        {}
                        if remove
                        else {"pending": "removed once the sandbox is gone"}
                    ),
                )
    return flags


def record_flags(rollout_dir: Path, flags: list[dict[str, Any]]) -> None:
    """Add *flags* to the trial's ``integrity/verifier-inputs.json`` (never
    in integrity off, whose record says the detector did not run). A flag
    already there is not added again; a later put_back that removed a name
    an earlier one could not says so in it."""
    if not flags:
        return
    path = Path(rollout_dir) / "integrity" / "verifier-inputs.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(record, dict) or record.get("status") == "off":
        return
    known = {
        (f.get("where"), f.get("path"), f.get("kind")): f
        for f in record.get("flags", [])
        if isinstance(f, dict)
    }
    new = []
    changed = False
    for f in flags:
        there = known.get((f["where"], f["path"], f["kind"]))
        if there is None:
            new.append(f)
            known[(f["where"], f["path"], f["kind"])] = f
        elif f.get("removed") is True and there.get("removed") is False:
            there["removed"] = True
            there.pop("error", None)
            there.pop("pending", None)
            there["after"] = f.get("after")
            changed = True
    if not new and not changed:
        return
    record.setdefault("flags", []).extend(new)
    section = record.setdefault("sections", {}).setdefault(
        "verifier_output", {"checked": 0, "flags": 0, "changes": 0}
    )
    section["flags"] = section.get("flags", 0) + len(new)
    record["status"] = "flagged"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def forget(holder: Any) -> None:
    """A new test command is issued in *holder*: the copy an earlier one
    left there (a multi-step task's earlier verify) is no longer test.sh's
    outputs, and is not put back."""
    for name in (
        "_benchflow_verifier_outputs_at_exit",
        "_benchflow_verifier_written_after_exit",
    ):
        with contextlib.suppress(AttributeError):
            setattr(holder, name, None)


def put_back_and_record(
    rollout_dir: str | Path | None, holder: Any, *, sandbox_gone: bool = False
) -> list[dict[str, Any]]:
    """The rollout's call: restore from what *holder* (the sandbox object the
    verifier ran in) keeps, and record the flags. *sandbox_gone*: the
    sandbox has been stopped and removed, so the names test.sh did not write
    are removed too (else only named). Never raises."""
    taken = getattr(holder, "_benchflow_verifier_outputs_at_exit", None)
    if not isinstance(taken, dict) or not isinstance(rollout_dir, (str, Path)):
        return []
    written = getattr(holder, "_benchflow_verifier_written_after_exit", None)
    try:
        flags = put_back(
            Path(rollout_dir),
            taken,
            written if isinstance(written, dict) else {},
            remove=sandbox_gone,
        )
        record_flags(Path(rollout_dir), flags)
        planted = [f["path"] for f in flags if f["kind"] == "planted"]
        if planted:
            # Said in every mode: off's record holds no flag.
            logger.warning(
                "names test.sh did not write %s the trial's verifier/ after "
                "grading: %s%s",
                "were removed from" if sandbox_gone else "are in",
                ", ".join(planted[:20]),
                " ..." if len(planted) > 20 else "",
            )
        return flags
    except Exception:
        logger.warning(
            "verifier outputs not restored from their copy at exit", exc_info=True
        )
        return []

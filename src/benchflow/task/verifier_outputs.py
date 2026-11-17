"""test.sh's outputs as they were the moment it exited (CIR's L14).

In integrity off and audit the agent's processes live through grading, and a mounted ``/logs/verifier`` stays writable from
the sandbox until it is gone. So the verifier copies the folder into the
trial, where no sandbox mount reaches (``verifier-at-exit/``), the moment the
test command has ended; the rewards are read from that copy, and once nothing
in the sandbox can write any more the rollout puts it back over ``verifier/``
(:func:`put_back`). BenchFlow's own later records are kept as BenchFlow wrote
them; any other change is undone, a reward file that appeared afterwards is
moved to ``verifier-after-exit/``, and each one is a flag in the trial's
``integrity/verifier-inputs.json``. Nothing here changes the reward: it was
read from the copy already.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow._utils.reachable_write import ReachablePath

logger = logging.getLogger(__name__)

AT_EXIT = "verifier-at-exit"
AFTER_EXIT = "verifier-after-exit"
#: Files copied: each up to the first limit, all together up to the second;
#: a larger one is kept by its sha256 alone.
FILE_MAX = 64 << 20
TOTAL_MAX = 512 << 20
REWARD_NAMES = ("reward.txt", "reward.json")


def _private(rollout_dir: Path, name: str) -> ReachablePath:
    """A folder of the trial no sandbox mount reaches."""
    folder = Path(rollout_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return ReachablePath.directory(folder)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(root: ReachablePath) -> dict[str, ReachablePath]:
    """Every regular file under *root* by its relative path (links left out)."""
    found: dict[str, ReachablePath] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
        except OSError:
            continue
        found[PurePosixPath(path.relative_to(root).as_posix()).as_posix()] = path
    return found


def unreadable(rollout_dir: Path) -> list[str]:
    """The regular files under the rollout's ``verifier/`` this host can see
    but not open, by their relative paths: on mounted outputs, root's 0600
    files a test.sh left (``mkstemp`` and a rename, or ``umask 077``)."""
    found: list[str] = []
    for rel, path in sorted(
        _files(ReachablePath.rollout(rollout_dir, "verifier")).items()
    ):
        try:
            with path.open("rb"):
                pass
        except PermissionError:
            found.append(rel)
        except OSError:
            continue
    return found


def take(rollout_dir: Path) -> dict[str, Any]:
    """Copy the rollout's ``verifier/`` to ``verifier-at-exit/`` and return
    what was taken: ``{"files": {rel: {"sha256", "size", "copied"}}}``."""
    live = ReachablePath.rollout(rollout_dir, "verifier")
    snap = _private(rollout_dir, AT_EXIT)
    files: dict[str, dict[str, Any]] = {}
    unread: list[str] = []
    used = 0
    for rel, path in sorted(_files(live).items()):
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
    return {"files": files, "special": special}


def put_back(
    rollout_dir: Path, taken: dict[str, Any], written: dict[str, bytes]
) -> list[dict[str, Any]]:
    """Put test.sh's outputs at its exit back over ``verifier/``, and
    BenchFlow's own later records (*written*: rel -> bytes) as it wrote them.
    Returns one flag per difference undone or found."""
    live = ReachablePath.rollout(rollout_dir, "verifier")
    snap = _private(rollout_dir, AT_EXIT)
    files = taken.get("files") or {}
    flags: list[dict[str, Any]] = []

    def flag(rel: str, kind: str, before: Any, after: Any) -> None:
        flags.append(
            {
                "where": "verifier_output",
                "path": f"/logs/verifier/{rel}",
                "kind": kind,
                "category": "verifier-output",
                "before": before,
                "after": after,
            }
        )

    current = _files(live)
    for rel in sorted(set(files) | set(written)):
        if rel in written:
            want: bytes | None = written[rel]
        elif files[rel].get("copied"):
            want = (snap / rel).read_bytes()
        else:
            want = None
        path = current.get(rel)
        have = path.read_bytes() if path is not None else None
        expected_sha = _sha(want) if want is not None else files[rel]["sha256"]
        if have is not None and _sha(have) == expected_sha:
            continue
        before = {"sha256": expected_sha}
        after = {"sha256": _sha(have), "size": len(have)} if have is not None else None
        if want is None:
            flag(rel, "edited" if have is not None else "removed", before, after)
            continue
        target = live / rel
        if target.is_symlink():
            target.unlink()
        target.write_bytes(want)
        flag(rel, "edited" if have is not None else "removed", before, after)
    for rel, path in sorted(current.items()):
        if rel in files or rel in written or rel.split("/")[-1] not in REWARD_NAMES:
            continue
        data = path.read_bytes()
        aside = _private(rollout_dir, AFTER_EXIT)
        (aside / rel.replace("/", "__")).write_bytes(data)
        path.unlink()
        flag(rel, "planted", None, {"sha256": _sha(data), "size": len(data)})
    return flags


def record_flags(rollout_dir: Path, flags: list[dict[str, Any]]) -> None:
    """Add *flags* to the trial's ``integrity/verifier-inputs.json`` (never
    in integrity off, whose record says the detector did not run)."""
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
        (f.get("where"), f.get("path"), f.get("kind")) for f in record.get("flags", [])
    }
    new = [f for f in flags if (f["where"], f["path"], f["kind"]) not in known]
    if not new:
        return
    record.setdefault("flags", []).extend(new)
    section = record.setdefault("sections", {}).setdefault(
        "verifier_output", {"checked": 0, "flags": 0, "changes": 0}
    )
    section["flags"] = section.get("flags", 0) + len(new)
    record["status"] = "flagged"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def put_back_and_record(
    rollout_dir: str | Path | None, holder: Any
) -> list[dict[str, Any]]:
    """The rollout's call: restore from what *holder* (the sandbox object the
    verifier ran in) keeps, and record the flags. Never raises."""
    taken = getattr(holder, "_benchflow_verifier_outputs_at_exit", None)
    if not isinstance(taken, dict) or not isinstance(rollout_dir, (str, Path)):
        return []
    written = getattr(holder, "_benchflow_verifier_written_after_exit", None)
    try:
        flags = put_back(
            Path(rollout_dir), taken, written if isinstance(written, dict) else {}
        )
        record_flags(Path(rollout_dir), flags)
        return flags
    except Exception:
        logger.warning(
            "verifier outputs not restored from their copy at exit", exc_info=True
        )
        return []

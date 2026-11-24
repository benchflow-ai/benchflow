"""Which BenchFlow build wrote a record: the ``benchflow_build`` field.

result.json, summary.json and run-card.json carry
``{"version": ..., "commit": ..., "dirty": ...}``: the installed version, the
git commit the code was built from, and whether the tree it was built from had
changes git had not committed. A version alone names a release, not a build:
every commit on a release branch reports the same version until the next bump.

The commit comes from the first of these that knows it:

1. the git checkout the package runs from (``uv sync``'s editable install),
   only when the package folder is that checkout's ``src/benchflow``: a virtual
   environment inside some other repository never lends it that repository's
   commit;
2. ``benchflow/_build.json`` in the installed package, which ``hatch_build.py``
   writes into the sdist and the wheel when they are built (an installed wheel
   has no ``.git`` to ask);
3. the ``vcs_info`` that ``pip install git+URL@COMMIT`` records in the
   distribution's ``direct_url.json``, with ``dirty`` unknown.

Otherwise ``commit`` and ``dirty`` are null. An unknown commit is never guessed.

The commit is the newest one on the first-parent line whose files, outside
``.github/release-acceptance``, are the build's. A release tag's commit adds
only its acceptance record to the accepted source commit
(docs/contributing/release.md), and the wheel built at the tag must hash the
same as the accepted one, so both name the source commit. ``dirty`` leaves
those records out for the same reason. Only first parents count: a merge
commit whose tree equals its PR side (GitHub's merge of a PR branched from the
base's head) names itself, never the PR head git's history simplification
would otherwise follow.

This module imports nothing from ``benchflow`` at load time: ``hatch_build.py``
loads it by path in the isolated build environment, where BenchFlow is not
installed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

# The file hatch_build.py bakes, beside this module in the installed package.
BAKED_FILE = "_build.json"
# What a release tag's commit adds to the accepted source commit.
RELEASE_RECORDS = ".github/release-acceptance"
# A SHA-1 commit, or a SHA-256 one.
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
# One git call's bound on a host whose load fits its CPUs (_git_bound()).
_GIT_TIMEOUT_SEC = 10
# A git that has not answered within its bound is asked again, up to this
# many calls in all: a stalled git never makes one tree read as two builds.
_GIT_TRIES = 3
_PACKAGE_DIR = Path(__file__).resolve().parent
_UNKNOWN: dict[str, Any] = {"commit": None, "dirty": None}


def benchflow_build() -> dict[str, Any]:
    """``{"version", "commit", "dirty"}`` of the BenchFlow that is running."""
    from benchflow import __version__

    return {"version": __version__, **_identity()}


@lru_cache(maxsize=1)
def _identity() -> dict[str, Any]:
    # Once per process: read_at_start() reads it when a run starts, so a
    # job's records name the build it started with though the checkout moves
    # under it. Stamping never fails a record: anything unexpected is unknown.
    try:
        return (
            checkout_identity(_PACKAGE_DIR)
            or baked_identity(_PACKAGE_DIR)
            or direct_url_identity(_direct_url_text())
            or dict(_UNKNOWN)
        )
    except Exception:
        return dict(_UNKNOWN)


def read_at_start() -> None:
    """Read the running build's identity now (the CLI and every Evaluation
    call this as they start), not at the first record."""
    _identity()


def baked_identity(package_dir: Path) -> dict[str, Any] | None:
    """What the build baked into ``package_dir``; None without a commit."""
    try:
        data = json.loads((package_dir / BAKED_FILE).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    commit, dirty = data.get("commit"), data.get("dirty")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    return {"commit": commit, "dirty": dirty if isinstance(dirty, bool) else None}


def checkout_identity(package_dir: Path) -> dict[str, Any] | None:
    """The identity of the git checkout whose ``src/benchflow`` is ``package_dir``."""
    root = checkout_root(package_dir)
    if root is None or (root / "src" / "benchflow") != package_dir.resolve():
        return None
    return git_identity(root)


def checkout_root(path: Path) -> Path | None:
    """The top folder of the git work tree ``path`` is in, or None."""
    top = _git(path, "rev-parse", "--show-toplevel")
    return Path(top).resolve() if top else None


def git_identity(root: Path) -> dict[str, Any] | None:
    """Commit and dirty state of the work tree at ``root``; None outside git.

    Untracked files count as changes (the build would package one under
    ``src/``); ignored files do not.
    """
    scope = ("--", ".", f":(exclude){RELEASE_RECORDS}")
    commit = _git(root, "log", "-1", "--first-parent", "--format=%H", *scope)
    if commit is None or not _COMMIT.fullmatch(commit):
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", *scope)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def direct_url_identity(text: str | None) -> dict[str, Any] | None:
    """The git commit a PEP 610 ``direct_url.json`` records; None otherwise."""
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    vcs = data.get("vcs_info") if isinstance(data, dict) else None
    if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
        return None
    commit = vcs.get("commit_id")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    return {"commit": commit, "dirty": None}


def _direct_url_text() -> str | None:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return distribution("benchflow").read_text("direct_url.json")
    except (PackageNotFoundError, OSError, UnicodeDecodeError):
        return None


def _git_bound() -> float:
    """One git call's bound: :data:`_GIT_TIMEOUT_SEC` while the 1-minute load
    fits the CPUs, in step with the load per CPU beyond that, up to 4 times
    (as the metadata helper's docker calls). A cold CLI reads its identity in
    the start burst, the runtime it would attach to read its own earlier:
    under load a fixed 10 s gave the two different records."""
    try:
        per_cpu = os.getloadavg()[0] / (os.cpu_count() or 1)
    except (AttributeError, OSError):
        return float(_GIT_TIMEOUT_SEC)
    return _GIT_TIMEOUT_SEC * min(max(per_cpu, 1.0), 4.0)


def _git(cwd: Path, *args: str) -> str | None:
    # No GIT_DIR or GIT_WORK_TREE from the caller's environment (a git hook,
    # say) may point the question at another repository; and no index lock,
    # so concurrent trials never collide in `git status`.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    for _ in range(_GIT_TRIES):
        try:
            proc = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                timeout=_git_bound(),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Stalled, not refused: the same question again.
            continue
        except (OSError, subprocess.SubprocessError):
            return None
        try:
            out = proc.stdout.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return out.strip() if proc.returncode == 0 else None
    return None

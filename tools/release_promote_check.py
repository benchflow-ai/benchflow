"""Whether one commit differs from another in the release version alone.

    python tools/release_promote_check.py [--repo .] OLD NEW

An rc becomes final by a version-only commit (docs/contributing/release.md, "Cadence"); so can a re-cut whose
source didn't change. Then the gate runs on the old commit stand for the new one
(tools/check_gate_receipts.py). The diff is version-only when, in every changed file, replacing OLD's version
(``pyproject.toml``'s ``[project].version``) with NEW's gives NEW's file, except for ``CHANGELOG.md`` and
``.github/release-acceptance/`` records, which may change freely. Prints each other changed path; exit 0 when
there is none.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

FREE = ("CHANGELOG.md",)
FREE_PREFIXES = (".github/release-acceptance/",)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120
    )
    if done.returncode:
        raise ValueError(f"git {' '.join(args[:2])}: {done.stderr.strip()[:200]}")
    return done.stdout


def _show(repo: Path, rev: str, path: str) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"],
        capture_output=True,
        timeout=120,
    )
    if done.returncode:
        return None
    return done.stdout.decode("utf-8", "surrogateescape")


def project_version(repo: Path, rev: str) -> str:
    text = _show(repo, rev, "pyproject.toml")
    if text is None:
        raise ValueError(f"{rev} has no pyproject.toml")
    return str(tomllib.loads(text)["project"]["version"])


def version_only_paths(repo: Path, old: str, new: str) -> list[str]:
    """The paths whose change between ``old`` and ``new`` is more than the version."""
    before, after = project_version(repo, old), project_version(repo, new)
    changed = [
        p for p in _git(repo, "diff", "--name-only", "-z", old, new).split("\0") if p
    ]
    other = []
    for path in sorted(changed):
        if path in FREE or path.startswith(FREE_PREFIXES):
            continue
        a, b = _show(repo, old, path), _show(repo, new, path)
        if a is None or b is None or before == after or a.replace(before, after) != b:
            other.append(path)
    return other


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("old")
    ap.add_argument("new")
    args = ap.parse_args(argv)
    try:
        other = version_only_paths(args.repo, args.old, args.new)
    except ValueError as exc:
        print(f"release_promote_check: {exc}", file=sys.stderr)
        return 2
    for path in other:
        print(path)
    print(
        "version-only"
        if not other
        else f"{len(other)} path(s) change more than the version",
        file=sys.stderr,
    )
    return 1 if other else 0


if __name__ == "__main__":
    sys.exit(main())

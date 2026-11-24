"""Fetch the sampled gate-1 tasks from their public sources, pinned by commit.

    python -m tools.parity.fetch --sample sample.tsv --dest DIR [--work DIR]

Each task is one folder of a public git repository at one commit (the corpus names both). The repository is
fetched once per commit, shallow and without blobs, and only the tasks' folders are checked out (a sparse
checkout), so a large dataset repository costs only the files the sample needs. Nothing comes from BenchFlow's
own storage. A task whose source isn't GitHub is refused here; the sample never draws one.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, timeout=1800
    )


def fetch(rows: list[dict[str, str]], dest: Path, work: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        if not r["repo"].startswith("https://github.com/"):
            raise SystemExit(
                f"fetch: {r['key']}: only GitHub sources are fetched in CI ({r['repo']})"
            )
        groups.setdefault((r["repo"], r["commit"]), []).append(r)
    for i, ((repo, commit), tasks) in enumerate(sorted(groups.items())):
        clone = work / f"src{i}"
        if clone.exists():
            shutil.rmtree(clone)
        clone.mkdir(parents=True)
        _git(clone, "init", "-q")
        _git(clone, "remote", "add", "origin", repo)
        _git(
            clone,
            "sparse-checkout",
            "set",
            "--no-cone",
            *[f"/{t['path'].strip('/')}/" for t in tasks],
        )
        _git(
            clone, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit
        )
        _git(clone, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
        for t in tasks:
            source = clone / t["path"].strip("/")
            target = dest / t["key"]
            if not source.is_dir():
                raise SystemExit(
                    f"fetch: {t['key']}: {t['path']} is not in {repo}@{commit[:12]}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, symlinks=True)
            print(f"{t['key']}\t{repo}@{commit[:12]}:{t['path']}")
        shutil.rmtree(clone)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    args = ap.parse_args(argv)
    with open(args.sample, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    fetch(rows, args.dest, args.work or args.dest.parent / "parity-src")
    return 0


if __name__ == "__main__":
    sys.exit(main())

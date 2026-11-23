"""Which gate-1 tasks a pull request runs (the oracle-parity CI job).

    python -m tools.parity.sample [--corpus tools/parity/corpus-g1.tsv] [--repo .] [--rev HEAD]
        [--base REF] [--k 12] --out sample.tsv

Gate 1 runs BenchFlow's oracle on 404 tasks and compares each reward with Harbor 0.26.0's; that is about 20
task-hours, a nightly's work. A pull request runs ``k`` of them (12 by default, 20 when it changes grading):

- the sentinels: cheap tasks whose outcome is a known rule (``SENTINELS``);
- one task from each of ``k - len(SENTINELS)`` families, families in an order and tasks within a family chosen
  by hashing the seed with their names. The seed is the tree hash of ``--rev``: a rerun, or a rebase onto the same
  tree, draws the same tasks, and another tree draws others, so over many pull requests every family is run;
- a family a changed adapter belongs to is drawn first.

Only tasks CI can fetch from their public source pinned by commit (the corpus) and that took at most
``MAX_SECS`` in the 0.10.0rc4 record are drawn. Tasks from our own storage never are.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus-g1.tsv"
KEYS = HERE / "g1-keys.txt"
MAX_SECS = 180
DEFAULT_K = 12
WIDE_K = 20
SENTINELS = (
    "hello-world/hello-world",
    "sldbench/easy_question_scaling_law",
    "researchcodebench/fractalgen_sample_orders",
    # Refused for a Harbor-format task.toml key: the accepted-unscored path.
    "bfcl/bfcl-simple-python-0",
)
# A change here can move any reward: the pull request runs the wider sample.
GRADING_PATHS = (
    "src/benchflow/task/verifier",
    "src/benchflow/rollout/_setup.py",
    "src/benchflow/rollout/_grading_outcome.py",
    "src/benchflow/rollout/_reward_provenance.py",
    "src/benchflow/_utils/scoring.py",
    "src/benchflow/sandbox/_verifier_links.py",
    "src/benchflow/sandbox/lockdown.py",
    "src/benchflow/_bfx/",
    "bfx/",
)
FIELDS = ("key", "family", "repo", "commit", "path", "rc4_secs", "rc4_end")


@dataclasses.dataclass(frozen=True)
class Task:
    key: str
    family: str
    repo: str
    commit: str
    path: str
    rc4_secs: int
    rc4_end: str


def load_corpus(path: Path = CORPUS) -> list[Task]:
    with open(path, newline="") as fh:
        rows = list(
            csv.DictReader(
                (line for line in fh if not line.startswith("#")), delimiter="\t"
            )
        )
    return [
        Task(
            r["key"],
            r["family"],
            r["repo"],
            r["commit"],
            r["path"],
            int(r["rc4_secs"]),
            r["rc4_end"],
        )
        for r in rows
    ]


def load_excluded(path: Path = HERE / "ledger" / "excluded.toml") -> dict[str, str]:
    """Tasks broken upstream, never drawn: key to reason."""
    import tomllib

    data = tomllib.loads(path.read_text()) if path.exists() else {}
    return {e["key"]: e["reason"] for e in data.get("excluded", [])}


def _rank(seed: str, name: str) -> str:
    return hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest()


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def forced_families(changed: Iterable[str], rows: Iterable[Task]) -> set[str]:
    """Families whose adapter a change touches (``src/benchflow/adapters/<name>...``)."""
    families = {_norm(r.family): r.family for r in rows}
    out = set()
    for path in changed:
        m = re.match(
            r"src/benchflow/adapters/_?([A-Za-z0-9_-]+?)(?:_services)?(?:\.py|/)", path
        )
        if m and _norm(m.group(1)) in families:
            out.add(families[_norm(m.group(1))])
    return out


def sample_size(changed: Iterable[str], k: int = DEFAULT_K) -> int:
    """``k``, or the wider sample when a change touches grading."""
    return max(k, WIDE_K) if any(p.startswith(GRADING_PATHS) for p in changed) else k


def select(
    rows: Iterable[Task], seed: str, k: int = DEFAULT_K, forced: Iterable[str] = ()
) -> list[str]:
    """The keys to run: the sentinels the corpus has, then one cheap task per family."""
    excluded = load_excluded()
    rows = [r for r in rows if r.key not in excluded]
    keys = {r.key for r in rows}
    picked = [s for s in SENTINELS if s in keys]
    pool = [
        r
        for r in rows
        if r.key not in picked
        and r.repo.startswith("https://github.com/")
        and r.rc4_secs <= MAX_SECS
    ]
    by_family: dict[str, list[Task]] = {}
    for r in pool:
        by_family.setdefault(r.family, []).append(r)
    for tasks in by_family.values():
        tasks.sort(key=lambda r: _rank(seed, r.key))
    forced = sorted(f for f in set(forced) if f in by_family)
    order = forced + sorted(
        (f for f in by_family if f not in forced), key=lambda f: _rank(seed, f)
    )
    want = max(0, k - len(picked))
    depth = 0
    while want and any(len(by_family[f]) > depth for f in order):
        for family in order:
            if want and len(by_family[family]) > depth:
                picked.append(by_family[family][depth].key)
                want -= 1
        depth += 1
    return picked


def tree_seed(repo: Path, rev: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{rev}^{{tree}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def changed_paths(repo: Path, base: str | None, rev: str) -> list[str]:
    if not base:
        return []
    done = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", f"{base}...{rev}"],
        capture_output=True,
        text=True,
    )
    return done.stdout.split() if done.returncode == 0 else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--base", help="the pull request's base, for the changed paths")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument(
        "--all",
        action="store_true",
        help="every GitHub task of the corpus (nightly), by shard",
    )
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument(
        "--every-key",
        action="store_true",
        help="all 404 gate-1 keys, for staged task folders",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = load_corpus(args.corpus)
    by_key = {r.key: r for r in rows}
    if args.every_key:
        keys = [k for k in KEYS.read_text().split() if k]
        seed = "every key"
    elif args.all:
        seed = "all"
        keys = [
            r.key
            for r in rows
            if r.repo.startswith("https://github.com/")
            and int(hashlib.sha256(r.key.encode()).hexdigest(), 16) % args.shards
            == args.shard - 1
        ]
    else:
        changed = changed_paths(args.repo, args.base, args.rev)
        seed = tree_seed(args.repo, args.rev)
        keys = select(
            rows, seed, sample_size(changed, args.k), forced_families(changed, rows)
        )
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(FIELDS)
        for key in keys:
            task = by_key.get(key)
            w.writerow(
                [getattr(task, f) for f in FIELDS]
                if task
                else [key] + [""] * (len(FIELDS) - 1)
            )
    print(
        f"oracle-parity sample: {len(keys)} tasks, seed tree {seed[:12]}",
        file=sys.stderr,
    )
    for key in keys:
        print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())

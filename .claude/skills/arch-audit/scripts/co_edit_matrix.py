#!/usr/bin/env python3
"""Rule 4 symbol co-edit clustering. TSV pairs ≥20% co-edit or `# status: ...`."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
CO_EDIT_THRESHOLD = 0.20
MIN_COMMITS = 10
MAX_SYMBOLS = 30


def symbol_ranges(src: str) -> list[tuple[str, int, int]]:
    """(name, start_line, end_line) for top-level defs + classes."""
    tree = ast.parse(src)
    out: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            out.append((node.name, start, end))
    return out


def commits_touching(file_rel: str) -> list[tuple[str, list[tuple[int, int]]]]:
    """For each commit modifying file, return (sha, list_of_hunk_line_ranges).
    Uses git log --since=6.months -U0 to extract hunks."""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--since=6.months",
                "--no-merges",
                "--format=---COMMIT %H",
                "-U0",
                "--",
                file_rel,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    commits: list[tuple[str, list[tuple[int, int]]]] = []
    cur_sha: str | None = None
    cur_hunks: list[tuple[int, int]] = []
    for line in out.stdout.splitlines():
        if line.startswith("---COMMIT "):
            if cur_sha is not None:
                commits.append((cur_sha, cur_hunks))
            cur_sha = line[len("---COMMIT ") :].strip()
            cur_hunks = []
        elif line.startswith("@@"):
            # @@ -oldstart,oldcount +newstart,newcount @@
            try:
                post = line.split("+", 1)[1].split(" ", 1)[0]
                if "," in post:
                    start_s, count_s = post.split(",", 1)
                    start, count = int(start_s), int(count_s)
                else:
                    start, count = int(post), 1
                if count == 0:
                    count = 1
                cur_hunks.append((start, start + count - 1))
            except (ValueError, IndexError):
                continue
    if cur_sha is not None:
        commits.append((cur_sha, cur_hunks))
    return commits


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: co_edit_matrix.py <file>\n")
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"# status: file_renamed_or_missing ({target})")
        return 0

    try:
        file_rel = str(target.resolve().relative_to(REPO))
    except ValueError:
        file_rel = str(target)

    src = target.read_text(encoding="utf-8", errors="replace")
    try:
        symbols = symbol_ranges(src)
    except SyntaxError:
        print("# status: parse_error")
        return 0

    if not symbols:
        print("# status: no_symbols")
        return 0
    if len(symbols) > MAX_SYMBOLS:
        print(f"# status: too_many_symbols ({len(symbols)})")
        return 0

    commits = commits_touching(file_rel)
    if len(commits) < MIN_COMMITS:
        print(f"# status: insufficient_commits ({len(commits)})")
        return 0

    # Per-symbol commit-appearance set.
    sym_commits: dict[str, set[str]] = defaultdict(set)
    boundary_drift = 0
    for sha, hunks in commits:
        touched_syms: set[str] = set()
        for hs, he in hunks:
            hit = False
            for name, s, e in symbols:
                if hs <= e and he >= s:
                    touched_syms.add(name)
                    hit = True
            if not hit:
                boundary_drift += 1
        for name in touched_syms:
            sym_commits[name].add(sha)

    # Pairwise co-edit ratio (Jaccard).
    pairs: list[tuple[str, str, float]] = []
    names = sorted(sym_commits.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = len(sym_commits[a] & sym_commits[b])
            union = len(sym_commits[a] | sym_commits[b])
            if union == 0:
                continue
            ratio = inter / union
            if ratio >= CO_EDIT_THRESHOLD:
                pairs.append((a, b, ratio))

    if not pairs:
        print("# status: no_clusters")
        if boundary_drift:
            print(f"# boundary_drift_hunks: {boundary_drift}")
        return 0

    pairs.sort(key=lambda t: -t[2])
    for a, b, r in pairs:
        print(f"{a}\t{b}\t{r:.2f}")
    if boundary_drift:
        print(f"# boundary_drift_hunks: {boundary_drift}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

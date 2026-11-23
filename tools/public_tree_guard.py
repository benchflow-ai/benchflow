"""Refuse a tree that would publish internal release material.

    python tools/public_tree_guard.py [ROOT] [--rev REV] [--json]

public-release.yml runs it on the tagged tree before anything is built: the
private release branches merge into the public main, so whatever their tree
holds goes public. It refuses:

- any path under ``release/`` (release records live in the internal lab
  repository, never here);
- any linear.app URL;
- any key of our tracker's teams (EN, ENG, BEN3, BD, CEA, OSS, then a hyphen
  and a number: the forms tools/strip_internal_refs.py strips), anywhere and
  in any case: a branch named for its issue (the team in lower case, a
  hyphen, the number, then the branch's words) names the key too. Only
  these teams count, as a whole word, then a hyphen and digits that end the
  number: ``CONF-1``, ``COVID-19``, ``[a-z]-9`` and ``gpt-oss-120b`` are not
  keys, nor is a zero-padded number (``bn-bd-00018``);
- in CHANGELOG.md and README.md, the public text a release ships and the
  docs site shows, any token shaped like a tracker key (capital letters, a
  hyphen, a number), unless it is a public name such as a CVE id, SHA-256 or
  UTF-8.

The key shape alone is not refused in code, tests or docs: there it also
spells review labels (CONF-1), regular-expression text, citation keys and
dataset names (COVID-19, CIFAR-10), none of them a tracker's.

It reads the files git tracks at ``--rev`` (default HEAD) in ROOT (default
the current folder), text files only. A finding names the file, the line and
the kind, and the token for a key or a path, never the URL. Exit 0: clean;
1: findings; 2: it could not read the tree.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RELEASE_DIR = "release/"
LINEAR_URL = re.compile(r"https?://(?:[\w-]+\.)?linear\.app\b", re.IGNORECASE)
# Our tracker's team keys (Linear), current and archived, in any case: a
# branch name spells the key in lower case (lead ruling on PR #146, M3).
# A tracker numbers issues from 1: a zero-padded number (a benchmark's
# task id such as bn-bd-00018) is not a key.
TRACKER_KEY = re.compile(r"\b(?i:EN|ENG|BEN3|BD|CEA|OSS)-[1-9]\d*\b")
KEY_SHAPED = re.compile(r"\b([A-Z][0-9A-Z]{1,7})-\d+\b")
# Where any key-shaped token is refused: the text a release ships for people
# to read (the same files as the CHANGELOG guard of the 0.9 line).
PUBLIC_TEXT = frozenset({"CHANGELOG.md", "README.md"})
# Public names that look like keys: CVE ids, hashes and ciphers, standards,
# model families.
PUBLIC_NAMES = frozenset({"AES", "CVE", "GPT", "ISO", "PEP", "RFC", "SHA", "UTF"})


def key_shaped_tokens(text: str) -> list[str]:
    return sorted(
        {
            m.group(0)
            for m in KEY_SHAPED.finditer(text)
            if m.group(1) not in PUBLIC_NAMES
        }
    )


def tracker_keys(path: str, text: str) -> list[str]:
    """The tracker keys in ``text``: our teams' anywhere, and in public text
    every key-shaped token that is not a public name."""
    found = {m.group(0) for m in TRACKER_KEY.finditer(text)}
    if path in PUBLIC_TEXT:
        found.update(key_shaped_tokens(text))
    return sorted(found)


def scan_text(path: str, text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if LINEAR_URL.search(line):
            findings.append({"path": path, "line": number, "kind": "linear-url"})
        for token in tracker_keys(path, line):
            findings.append(
                {"path": path, "line": number, "kind": "tracker-key", "token": token}
            )
    return findings


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, timeout=600
    ).stdout


def scan_tree(root: Path, rev: str = "HEAD") -> list[dict[str, object]]:
    """Every finding in the files git tracks at ``rev``."""
    listing = _git(root, "ls-tree", "-r", "-z", "--full-tree", rev)
    findings: list[dict[str, object]] = []
    blobs: list[tuple[str, str]] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, _, name = entry.partition(b"\t")
        mode, kind, sha = meta.decode().split()
        path = name.decode("utf-8", "surrogateescape")
        if path.startswith(RELEASE_DIR):
            findings.append({"path": path, "line": 0, "kind": "release-path"})
        if kind == "blob" and mode != "120000":
            blobs.append((path, sha))
    if not blobs:
        return findings
    # One cat-file process for every blob.
    batch = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=b"".join(f"{sha}\n".encode() for _, sha in blobs),
        check=True,
        capture_output=True,
        timeout=600,
    ).stdout
    offset = 0
    for path, _sha in blobs:
        header_end = batch.index(b"\n", offset)
        size = int(batch[offset:header_end].split()[2])
        data = batch[header_end + 1 : header_end + 1 + size]
        offset = header_end + 1 + size + 1
        if b"\0" in data[:8192]:
            continue  # binary
        findings.extend(scan_text(path, data.decode("utf-8", "replace")))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Refuse a tree that would publish internal release material."
    )
    ap.add_argument("root", nargs="?", type=Path, default=Path("."))
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--json", action="store_true", help="every finding, as JSON")
    args = ap.parse_args(argv)
    try:
        findings = scan_tree(args.root, args.rev)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"public_tree_guard: cannot read the tree: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(findings, sys.stdout, indent=1)
        print()
    else:
        for f in findings:
            where = f"{f['path']}:{f['line']}" if f["line"] else str(f["path"])
            print(f"{where}: {f['kind']}" + (f" {f['token']}" if "token" in f else ""))
    kinds: dict[str, int] = {}
    for f in findings:
        kinds[str(f["kind"])] = kinds.get(str(f["kind"]), 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "clean"
    print(f"public_tree_guard: {summary}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

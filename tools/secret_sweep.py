"""The push-protection sweep: what GitHub push protection would refuse, found before the push.

    python tools/secret_sweep.py --base BASE --tip TIP [--repo .] [--scanners DIR | --no-scanners]
        [--allowlist tools/secret_sweep_allow.toml] [--json FILE]

GitHub push protection reads every commit a push carries, tests included, and refuses a provider-shaped secret;
the release leak gate (tools/release_leak_gate.py) reads only the wheel, the sdist and the docs. 0.10.0rc4's public
push was refused on a synthetic key in a test. This sweep reads what push protection reads:

- the provider patterns of ``tools/ghpp_patterns.py`` over the lines each commit in ``BASE..TIP`` adds (a shape
  added and removed again inside the range is still pushed) and over ``TIP``'s tree;
- gitleaks and trufflehog (``--no-verification``: nothing is sent to any provider), pinned by
  tools/install_leak_scanners.sh, over the same range, when ``--scanners`` names their folder.

A provider-pattern hit can't be allowlisted: GitHub refuses it whatever this file says, and only a person's bypass
click lets it through. Fix it at the source (``tools/keyshape_fix.py`` splits a test literal so the value is built
at runtime). A scanner's generic-rule hit (gitleaks, trufflehog) passes only with an exception in the allowlist
that binds the tool, rule, path and the SHA-256 of the matched value, with a reason and an expiry.

Output names the tool, rule, file, line and commit, never the value. Exit 0: nothing open; 1: open hits; 2: the
sweep could not decide (a scanner missing or failing, an allowlist that is not valid).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import ghpp_patterns

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "tools" / "secret_sweep_allow.toml"
GHPP = "ghpp"
GITLEAKS_VERSION = "8.30.1"
TRUFFLEHOG_VERSION = "3.99.8"


class SweepError(Exception):
    """The sweep could not decide."""


@dataclasses.dataclass(frozen=True)
class Hit:
    tool: str
    rule: str
    path: str
    line: int
    commit: str = ""
    value_sha256: str = ""

    def row(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return "\t".join((self.tool, self.rule, where, self.commit[:12]))


@dataclasses.dataclass(frozen=True)
class Allow:
    tool: str
    rule: str
    path: str
    value_sha256: str
    reason: str
    expires: dt.date


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    if done.returncode:
        raise SweepError(
            f"git {' '.join(args[:2])} failed: {done.stderr.strip()[:300]}"
        )
    return done.stdout


def _is_text(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def scan_tree(repo: Path, rev: str = "HEAD") -> list[Hit]:
    """Provider-pattern hits in the text files git tracks at ``rev``."""
    listing = _git(repo, "ls-tree", "-r", "-z", "--full-tree", rev)
    blobs = []
    for entry in listing.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        mode, kind, sha = meta.split()
        if kind == "blob" and mode != "120000":
            blobs.append((path, sha))
    hits: list[Hit] = []
    if not blobs:
        return hits
    batch = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=b"".join(f"{sha}\n".encode() for _, sha in blobs),
        capture_output=True,
        check=True,
        timeout=600,
    ).stdout
    offset = 0
    for path, _sha in blobs:
        header_end = batch.index(b"\n", offset)
        size = int(batch[offset:header_end].split()[2])
        data = batch[header_end + 1 : header_end + 1 + size]
        offset = header_end + 1 + size + 1
        if not _is_text(data):
            continue
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for rule, _col, _n, value in ghpp_patterns.scan_line(line):
                hits.append(Hit(GHPP, rule, path, number, "", sha256_text(value)))
    return hits


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def scan_range(repo: Path, base: str, tip: str) -> list[Hit]:
    """Provider-pattern hits in the lines each commit of ``base..tip`` adds, with the commit that adds them."""
    log = _git(
        repo,
        "log",
        "--reverse",
        "--no-merges",
        "--no-renames",
        "--format=commit %H",
        "-p",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        f"{base}..{tip}",
    )
    hits: list[Hit] = []
    commit, path, number = "", "", 0
    for raw in log.splitlines():
        if raw.startswith("commit ") and len(raw) == 47:
            commit, path, number = raw[7:], "", 0
        elif raw.startswith("+++ "):
            target = raw[4:]
            path = target[2:] if target.startswith("b/") else ""
        elif raw.startswith("@@ "):
            m = _HUNK.match(raw)
            number = int(m.group(1)) if m else 0
        elif raw.startswith("+") and path:
            for rule, _col, _n, value in ghpp_patterns.scan_line(raw[1:]):
                hits.append(Hit(GHPP, rule, path, number, commit, sha256_text(value)))
            number += 1
    return hits


def _scanner(scanners: Path, name: str, version: str, args: list[str]) -> str:
    binary = scanners / name
    if not binary.is_file():
        raise SweepError(
            f"{name} is not in {scanners} (tools/install_leak_scanners.sh)"
        )
    out = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=120
    )
    if version not in out.stdout + out.stderr:
        raise SweepError(f"{name} is not version {version}")
    return str(binary)


def run_scanners(repo: Path, base: str, tip: str, scanners: Path) -> list[Hit]:
    """gitleaks and trufflehog over the commits of ``base..tip``; values are hashed, never kept."""
    gitleaks = _scanner(scanners, "gitleaks", GITLEAKS_VERSION, ["version"])
    trufflehog = _scanner(scanners, "trufflehog", TRUFFLEHOG_VERSION, ["--version"])
    hits: list[Hit] = []
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "gitleaks.json"
        done = subprocess.run(
            [gitleaks, "git", str(repo), f"--log-opts={base}..{tip}", "--no-banner",
             "--report-format", "json", "--report-path", str(report), "--exit-code", "0"],
            capture_output=True, text=True, timeout=1800,
        )  # fmt: skip
        if done.returncode:
            raise SweepError(f"gitleaks failed (rc {done.returncode})")
        for r in json.loads(report.read_text() or "[]"):
            hits.append(
                Hit("gitleaks", str(r.get("RuleID")), str(r.get("File")),
                    int(r.get("StartLine") or 0), str(r.get("Commit") or ""),
                    sha256_text(str(r.get("Secret") or "")))
            )  # fmt: skip
    _git(repo, "branch", "-f", "secret-sweep-tip", tip)
    try:
        done = subprocess.run(
            [trufflehog, "git", f"file://{repo.resolve()}", "--since-commit", base,
             "--branch", "secret-sweep-tip", "--no-verification", "--no-update", "--json"],
            capture_output=True, text=True, timeout=1800,
        )  # fmt: skip
    finally:
        _git(repo, "branch", "-D", "secret-sweep-tip")
    if done.returncode not in (0, 183):
        raise SweepError(f"trufflehog failed (rc {done.returncode})")
    for line in done.stdout.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "DetectorName" not in r:
            continue
        data = (r.get("SourceMetadata") or {}).get("Data") or {}
        git = data.get("Git") or {}
        hits.append(
            Hit("trufflehog", str(r["DetectorName"]), str(git.get("file") or ""),
                int(git.get("line") or 0), str(git.get("commit") or ""),
                sha256_text(str(r.get("Raw") or "")))
        )  # fmt: skip
    return hits


def load_allowlist(path: Path) -> list[Allow]:
    """The exceptions in ``path``; a provider-pattern exception is refused."""
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise SweepError(f"{path}: {exc}") from None
    out = []
    for i, entry in enumerate(data.get("allow", [])):
        missing = [
            f
            for f in ("tool", "rule", "path", "value_sha256", "reason", "expires")
            if not entry.get(f)
        ]
        if missing:
            raise SweepError(f"{path} entry {i}: missing {', '.join(missing)}")
        if entry["tool"] == GHPP:
            raise SweepError(
                f"{path} entry {i}: a provider pattern can't be allowlisted; push protection refuses it "
                "whatever this file says. Build the value at runtime (tools/keyshape_fix.py)."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["value_sha256"])):
            raise SweepError(f"{path} entry {i}: value_sha256 is not a SHA-256")
        expires = entry["expires"]
        if isinstance(expires, str):
            expires = dt.date.fromisoformat(expires)
        out.append(Allow(entry["tool"], entry["rule"], entry["path"], entry["value_sha256"],
                         entry["reason"], expires))  # fmt: skip
    return out


def apply_allowlist(
    hits: list[Hit], allow: list[Allow], today: dt.date | None = None
) -> tuple[list[Hit], list[Hit]]:
    """``(open, allowed)``: a hit is allowed by an unexpired exception binding its tool, rule, path and value."""
    today = today or dt.date.today()
    keys = {
        (a.tool, a.rule, a.path, a.value_sha256) for a in allow if a.expires >= today
    }
    open_hits, allowed = [], []
    for hit in hits:
        key = (hit.tool, hit.rule, hit.path, hit.value_sha256)
        (allowed if hit.tool != GHPP and key in keys else open_hits).append(hit)
    return open_hits, allowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument(
        "--base", required=True, help="the first commit not pushed (public main's head)"
    )
    ap.add_argument("--tip", default="HEAD")
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--scanners",
        type=Path,
        help="gitleaks and trufflehog, as install_leak_scanners.sh puts them",
    )
    group.add_argument(
        "--no-scanners", action="store_true", help="provider patterns only"
    )
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    ap.add_argument("--json", type=Path, help="write every hit (no values) as JSON")
    args = ap.parse_args(argv)
    try:
        allow = load_allowlist(args.allowlist)
        hits = scan_range(args.repo, args.base, args.tip)
        seen = {(h.rule, h.path, h.value_sha256) for h in hits}
        hits += [
            h
            for h in scan_tree(args.repo, args.tip)
            if (h.rule, h.path, h.value_sha256) not in seen
        ]
        if args.scanners:
            hits += run_scanners(args.repo, args.base, args.tip, args.scanners)
        elif not args.no_scanners:
            raise SweepError(
                "name --scanners DIR, or --no-scanners for the provider patterns alone"
            )
    except SweepError as exc:
        print(f"secret_sweep: {exc}", file=sys.stderr)
        return 2
    open_hits, allowed = apply_allowlist(hits, allow)
    for hit in open_hits:
        print(hit.row())
    if args.json:
        args.json.write_text(
            json.dumps(
                {"open": [dataclasses.asdict(h) for h in open_hits],
                 "allowed": [dataclasses.asdict(h) for h in allowed]},
                indent=1,
            )
            + "\n"
        )  # fmt: skip
    counts: dict[str, int] = {}
    for hit in open_hits:
        counts[f"{hit.tool}:{hit.rule}"] = counts.get(f"{hit.tool}:{hit.rule}", 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "nothing open"
    print(
        f"secret_sweep {args.base[:12]}..{args.tip}: {summary}; allowlisted {len(allowed)}",
        file=sys.stderr,
    )
    return 1 if open_hits else 0


if __name__ == "__main__":
    sys.exit(main())

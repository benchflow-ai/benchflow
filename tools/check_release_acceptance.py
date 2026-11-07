"""Require a reviewed acceptance record for the exact stable-release artifacts.

The record is committed after testing the frozen source and distributions. Only
that record may differ between the accepted source and the release tag. This
checks the binding; the referenced acceptance receipt and its results are
reviewed separately. Nothing here creates or approves an acceptance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA256 = re.compile(r"[0-9a-f]{64}")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
    )
    if result.returncode:
        raise ValueError(f"git validation failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate acceptance field: {key}")
        result[key] = value
    return result


def check_release_acceptance(repo: Path, dist: Path) -> list[str]:
    """Reject changed source, unaccepted artifacts and incomplete records."""
    project = tomllib.loads((repo / "pyproject.toml").read_text())
    version = project["project"]["version"]
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9][A-Za-z0-9.!+_-]*", version
    ):
        raise ValueError("invalid project version for an acceptance record")
    relative = f".github/release-acceptance/{version}.json"
    record_path = repo / relative
    record = json.loads(record_path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int:
        raise ValueError("acceptance record must have schema_version 1")
    if record["schema_version"] != 1 or record.get("version") != version:
        raise ValueError("acceptance schema or version does not match this release")
    if record.get("status") != "accepted":
        raise ValueError("release acceptance is not complete")
    receipt = record.get("acceptance_receipt_sha256")
    if not isinstance(receipt, str) or not SHA256.fullmatch(receipt):
        raise ValueError("acceptance receipt SHA-256 is missing or invalid")
    source = record.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("accepted source must be a full commit SHA")

    # Do not let an uncommitted replacement record or source tree certify HEAD.
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("release files have uncommitted changes")
    committed_record = _git(repo, "show", f"HEAD:{relative}")
    if committed_record != record_path.read_text().strip():
        raise ValueError("acceptance record differs from the release commit")
    _git(repo, "merge-base", "--is-ancestor", source, "HEAD")
    changed = _git(repo, "diff", "--name-only", "-z", source, "HEAD")
    unexpected = set(changed.split("\0")) - {"", relative}
    if unexpected:
        raise ValueError(
            "release source changed after acceptance: " + ", ".join(sorted(unexpected))
        )

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 2:
        raise ValueError("acceptance must name exactly one wheel and one sdist")
    artifact_hashes: dict[str, str] = {}
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError(
                "accepted artifact names and SHA-256 values must be strings"
            )
        if not SHA256.fullmatch(expected):
            raise ValueError(f"invalid accepted SHA-256 for {name}")
        artifact_hashes[name] = expected
    sdist = f"benchflow-{version}.tar.gz"
    wheel_pattern = re.compile(
        rf"benchflow-{re.escape(version)}-[A-Za-z0-9_.]+-"
        r"[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl"
    )
    wheels = [name for name in artifact_hashes if wheel_pattern.fullmatch(name)]
    if sdist not in artifact_hashes or len(wheels) != 1:
        raise ValueError("accepted artifact names do not match the release version")
    distribution_entries = {path.name for path in dist.iterdir()}
    # uv 0.13.4 creates this exact marker in its output directory. It is not
    # an upload artifact. Permit only its generated regular-file form; retain
    # the exact set check for every other entry, including other dotfiles.
    marker = dist / ".gitignore"
    if (
        ".gitignore" in distribution_entries
        and not marker.is_symlink()
        and marker.is_file()
        and marker.stat().st_size == 1
        and marker.read_bytes() == b"*"
    ):
        distribution_entries.remove(".gitignore")
    if distribution_entries != set(artifact_hashes):
        raise ValueError(
            "distribution directory differs from the accepted artifact set"
        )

    messages = []
    for name, expected in sorted(artifact_hashes.items()):
        path = dist / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"accepted artifact is not a regular file: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{name} differs from the accepted SHA-256")
        messages.append(f"{name}: matches accepted SHA-256 {actual}")
    return [f"Accepted source: {source}; receipt SHA-256: {receipt}", *messages]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        messages = check_release_acceptance(args.repo, args.dist)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"::error::Release acceptance check failed: {exc}", file=sys.stderr)
        return 1
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())

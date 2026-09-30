"""The vendored task.md reference tools and fixtures are the pinned commit's files, byte for byte."""

from __future__ import annotations

import hashlib
import json
import subprocess

from benchflow.taskmd._vendor import VENDOR_DIR, vendor_pin
from tests._taskmd_helpers import FIXTURES, require_taskmd_repo


def test_vendored_files_match_their_pinned_hashes() -> None:
    pin = vendor_pin()
    assert set(pin["files"]) == {"taskmd.py", "judgeprompt.py"}
    for name, entry in pin["files"].items():
        data = (VENDOR_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], (
            f"{name} was edited: re-vendor it from task-md {pin['commit']} "
            "instead, and update VENDOR.json"
        )


def test_vendored_files_equal_the_pinned_commit() -> None:
    repo = require_taskmd_repo()
    pin = vendor_pin()
    for name, entry in pin["files"].items():
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{pin['commit']}:{entry['source']}"],
            capture_output=True,
            check=True,
        ).stdout
        assert shown == (VENDOR_DIR / name).read_bytes(), name


def test_fixtures_equal_the_pinned_commit() -> None:
    repo = require_taskmd_repo()
    source = json.loads((FIXTURES / "SOURCE.json").read_text())
    assert source["commit"] == vendor_pin()["commit"]
    checked = 0
    for path in sorted((FIXTURES / "examples").rglob("*")) + sorted(
        (FIXTURES / "vectors").rglob("*")
    ):
        if not path.is_file():
            continue
        rel = path.relative_to(FIXTURES)
        parts = rel.parts
        if parts[0] == "vectors":
            upstream = "schema/vectors/" + "/".join(parts[1:])
        elif parts[1] == "regex-log":
            upstream = "examples/datasets/terminal-mini/tasks/" + "/".join(parts[1:])
        else:
            upstream = "examples/" + "/".join(parts[1:])
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{source['commit']}:{upstream}"],
            capture_output=True,
            check=True,
        ).stdout
        assert shown == path.read_bytes(), str(rel)
        checked += 1
    assert checked > 50
    # ... and nothing upstream is missing from the copies (.gitignore once dropped three).
    folders = {
        "schema/vectors/judge-prompt-1": FIXTURES / "vectors" / "judge-prompt-1",
        "examples/datasets/terminal-mini/tasks/regex-log": FIXTURES
        / "examples"
        / "regex-log",
    } | {
        f"examples/{p.name}": p
        for p in (FIXTURES / "examples").iterdir()
        if p.is_dir() and p.name != "regex-log"
    }
    for upstream, copy in folders.items():
        listed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-tree",
                "-r",
                "--name-only",
                source["commit"],
                upstream,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert listed, upstream
        missing = [f for f in listed if not (copy / f[len(upstream) + 1 :]).is_file()]
        assert missing == [], missing

"""Conversion-parity conformance over vendored public SkillsBench tasks.

The fixtures under ``tests/fixtures/skillsbench_slice/`` are vendored verbatim
from github.com/benchflow-ai/skillsbench at the commit pinned in
``manifest.json``, so the parity claim is reproducible from the public repo:
re-fetch the pinned commit and the per-file digests must match. Each runner
case feeds a real task through the migrate/export conversion path
(``build_harbor_roundtrip_conformance_report``: split -> task.md -> split) and
asserts every compared surface — canonical config, normalized prompt, and the
environment/solution/tests file-hash maps — survives with zero mismatches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchflow.task.export import build_harbor_roundtrip_conformance_report

SLICE_DIR = Path(__file__).parent / "fixtures" / "skillsbench_slice"
MANIFEST = json.loads((SLICE_DIR / "manifest.json").read_text())
VENDORED_TASKS = (
    "flood-risk-analysis",
    "lake-warming-attribution",
    "r2r-mpc-control",
    "suricata-custom-exfil",
    "syzkaller-ppdev-syzlang",
)
MAX_SLICE_BYTES = 300 * 1024


def _file_digests(task_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(task_dir).as_posix(): (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(task_dir.rglob("*"))
        if path.is_file()
    }


def test_manifest_pins_the_public_source_commit() -> None:
    assert MANIFEST["source_repo"] == "https://github.com/benchflow-ai/skillsbench"
    assert MANIFEST["source_path_prefix"] == "tasks"
    commit = MANIFEST["source_commit"]
    assert len(commit) == 40
    assert set(commit) <= set("0123456789abcdef")


def test_vendored_slice_matches_manifest_inventory_and_digests() -> None:
    on_disk = sorted(p.name for p in SLICE_DIR.iterdir() if p.is_dir())
    assert on_disk == sorted(VENDORED_TASKS)
    assert sorted(MANIFEST["tasks"]) == sorted(VENDORED_TASKS)
    for task in VENDORED_TASKS:
        assert _file_digests(SLICE_DIR / task) == MANIFEST["tasks"][task], task


def test_vendored_slice_stays_small_and_text_only() -> None:
    files = [
        path
        for path in SLICE_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    assert files
    assert sum(path.stat().st_size for path in files) <= MAX_SLICE_BYTES
    for path in files:
        path.read_text(encoding="utf-8")


@pytest.mark.parametrize("task_name", VENDORED_TASKS)
def test_harbor_roundtrip_is_lossless_for_vendored_task(task_name: str) -> None:
    report = build_harbor_roundtrip_conformance_report(SLICE_DIR / task_name)

    assert [(m.path, m.reason) for m in report.mismatches] == []
    assert report.status == "lossless"
    assert report.config_equal is True
    assert report.prompt_equal is True
    assert report.environment_file_map_equal is True
    assert report.solution_file_map_equal is True
    assert report.tests_file_map_equal is True
    assert report.restored_extension_paths == []

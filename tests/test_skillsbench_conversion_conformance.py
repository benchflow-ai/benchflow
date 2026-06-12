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

from benchflow.task import export as task_export
from benchflow.task.export import (
    HarborRoundTripMismatch,
    build_harbor_roundtrip_conformance_report,
)

SLICE_DIR = Path(__file__).parent / "fixtures" / "skillsbench_slice"
MANIFEST = json.loads((SLICE_DIR / "manifest.json").read_text())
TASK_NAMES = sorted(MANIFEST["tasks"])
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
    assert TASK_NAMES
    assert on_disk == TASK_NAMES
    for task in TASK_NAMES:
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


@pytest.mark.parametrize("task_name", TASK_NAMES)
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


def test_harbor_roundtrip_reports_tampered_environment_as_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silently disabled environment comparator must fail this test."""
    real_export = task_export.export_task_to_split_layout

    def tampering_export(task_dir, output_dir, **kwargs):
        report = real_export(task_dir, output_dir, **kwargs)
        victim = next(
            path
            for path in sorted((Path(output_dir) / "environment").rglob("*"))
            if path.is_file()
        )
        victim.write_bytes(victim.read_bytes() + b"# drift\n")
        return report

    monkeypatch.setattr(task_export, "export_task_to_split_layout", tampering_export)

    report = build_harbor_roundtrip_conformance_report(SLICE_DIR / TASK_NAMES[0])

    assert report.status == "drift"
    assert report.environment_file_map_equal is False
    assert report.mismatches == [
        HarborRoundTripMismatch(
            path="environment/",
            reason="environment file hashes differ after migrate/export",
        )
    ]
    assert report.config_equal is True
    assert report.prompt_equal is True
    assert report.solution_file_map_equal is True
    assert report.tests_file_map_equal is True

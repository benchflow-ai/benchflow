"""Symlink-defence regression test for LLM judge deliverable ingestion (#404).

`find_deliverables` is the sink that feeds the judge prompt. A deliverable
symlink to a host-side file must NOT be read or sent downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from benchflow.rewards.file_readers import find_deliverables

SECRET = "SECRET_FROM_HOST_JUDGE_LEAK"


def test_find_deliverables_skips_symlink_to_outside_secret(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = tmp_path / "host-secret.txt"
    secret.write_text(SECRET)

    deliverables_dir = tmp_path / "deliverables"
    deliverables_dir.mkdir()
    # Symlink uses .txt so it would otherwise be picked up.
    (deliverables_dir / "leak.txt").symlink_to(secret)
    # A real deliverable so we know we aren't just returning empty.
    (deliverables_dir / "real.txt").write_text("real-deliverable")

    with caplog.at_level(logging.WARNING):
        result = find_deliverables(deliverables_dir)

    assert "leak.txt" not in result, "symlinked deliverable was read"
    assert SECRET not in "\n".join(result.values())
    assert result.get("real.txt") == "real-deliverable"
    assert any("leak.txt" in r.message for r in caplog.records)


def test_find_deliverables_skips_symlinked_subfile_by_lstat(tmp_path: Path) -> None:
    """Even if a symlink target IS a regular file, the link itself is refused."""
    secret = tmp_path / "host-secret.md"
    secret.write_text(SECRET)

    deliverables_dir = tmp_path / "deliverables"
    deliverables_dir.mkdir()
    (deliverables_dir / "link.md").symlink_to(secret)

    result = find_deliverables(deliverables_dir)
    assert result == {}, f"expected empty dict, got {result}"


def test_find_deliverables_still_reads_regular_files(tmp_path: Path) -> None:
    """Sanity guard — the symlink refusal must not break the happy path."""
    deliverables_dir = tmp_path / "deliverables"
    deliverables_dir.mkdir()
    (deliverables_dir / "a.txt").write_text("alpha")
    (deliverables_dir / "b.md").write_text("# beta")
    (deliverables_dir / ".hidden").write_text("nope")  # dotfile filtered
    (deliverables_dir / "rubric.json").write_text("{}")  # rubric filtered

    result = find_deliverables(deliverables_dir)
    assert result == {"a.txt": "alpha", "b.md": "# beta"}

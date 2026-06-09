"""Path traversal regression tests for benchflow.learner_skills (issue #402).

Two attack surfaces:

* :func:`capture_skills` walks an export directory with ``is_dir()``, which
  follows symlinks — an attacker can place ``leaked-skill -> /some/outside``
  and have the symlink target's ``SKILL.md`` ingested.
* :func:`materialize_skills` uses skill names from ``LearnerState.skills`` as
  path segments without validation — a name like ``../escaped`` writes a
  SKILL.md outside the destination root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.learner_skills import capture_skills, materialize_skills
from benchflow.learner_store import LearnerState

# capture_skills — symlink-skip


def test_capture_skips_symlinked_pack(tmp_path: Path, caplog) -> None:
    # Build an "outside" SKILL.md the test wants to confirm is NOT ingested.
    outside = tmp_path / "outside-secret"
    outside.mkdir()
    (outside / "SKILL.md").write_text("# secret-token\n")

    # Export directory contains one legitimate pack and one symlinked pack
    # that points at the outside directory.
    export = tmp_path / "export"
    export.mkdir()
    real_pack = export / "real-skill"
    real_pack.mkdir()
    (real_pack / "SKILL.md").write_text("# real\n")
    (export / "leaked-skill").symlink_to(outside)

    with caplog.at_level("WARNING"):
        result = capture_skills(export)

    # Only the real pack came through; the symlinked pack was skipped.
    assert result == {"real-skill": "# real\n"}
    assert "leaked-skill" not in result
    # And a warning was logged so the skip is auditable.
    assert any("symlink" in r.message.lower() for r in caplog.records)


def test_capture_skips_symlinked_skill_md(tmp_path: Path, caplog) -> None:
    """A symlinked SKILL.md inside a real pack is also refused."""
    outside = tmp_path / "outside-secret.md"
    outside.write_text("# secret-payload\n")

    export = tmp_path / "export"
    pack = export / "trojan-skill"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").symlink_to(outside)

    with caplog.at_level("WARNING"):
        result = capture_skills(export)

    assert result == {}
    assert any("symlink" in r.message.lower() for r in caplog.records)


# materialize_skills — segment validation


@pytest.mark.parametrize(
    "bad_name",
    ["../escaped-materialized", "a/b", "..", ".", "", "/abs"],
)
def test_materialize_rejects_traversal_skill_name(
    tmp_path: Path, bad_name: str
) -> None:
    state = LearnerState(skills={bad_name: "body"})
    dest = tmp_path / "skills-root"
    with pytest.raises(ValueError, match="skill name"):
        materialize_skills(state, dest)
    # No SKILL.md landed at the sibling escape location.
    assert not (tmp_path / "escaped-materialized").exists()
    assert not (tmp_path / "escaped-materialized" / "SKILL.md").exists()


def test_materialize_accepts_valid_skill_names(tmp_path: Path) -> None:
    state = LearnerState(skills={"git-bisect": "do the bisect", "grep": "use rg"})
    dest = tmp_path / "skills-root"
    materialize_skills(state, dest)
    assert (dest / "git-bisect" / "SKILL.md").read_text() == "do the bisect"
    assert (dest / "grep" / "SKILL.md").read_text() == "use rg"

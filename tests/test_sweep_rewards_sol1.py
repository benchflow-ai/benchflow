"""Guards PY8 taskmd/rewards sweep fixes against base commit 363d815b."""

import pytest

from benchflow.rewards.file_readers import find_deliverables


@pytest.mark.parametrize("link_at", ["root", "output", "workspace", "workspace/output"])
def test_deliverable_directory_links_never_read_controller_files(tmp_path, link_at):
    """Every directory component must belong to the evidence tree."""
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "notes.txt").write_text("DUMMY_CONTROLLER_MARKER")
    (controller / "output").mkdir()
    (controller / "output" / "notes.txt").write_text("DUMMY_CONTROLLER_MARKER")
    evidence = tmp_path / "evidence"
    link = evidence if link_at == "root" else evidence / link_at
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(controller, target_is_directory=True)
    directory = (
        evidence
        if link_at == "root"
        else evidence / "workspace/output"
        if link_at.startswith("workspace")
        else link
    )
    assert find_deliverables(directory) == {}
    assert find_deliverables(directory, evidence_root=evidence) == {}


def test_controller_parent_alias_is_allowed_above_the_evidence_root(tmp_path):
    """Trusted OS aliases above the downloaded root remain usable."""
    parent = tmp_path / "physical"
    evidence = parent / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "answer.txt").write_text("correct")
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    assert find_deliverables(alias / "evidence", evidence_root=alias / "evidence") == {
        "answer.txt": "correct"
    }


def test_opened_deliverable_survives_a_leaf_name_swap(tmp_path, monkeypatch):
    """Reading uses the opened inode, not a name checked earlier."""
    import os

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    answer = evidence / "answer.txt"
    answer.write_text("correct")
    controller = tmp_path / "controller.txt"
    controller.write_text("DUMMY_CONTROLLER_MARKER")
    original = os.fstat
    swapped = []

    def stat_fd(fd):
        info = original(fd)
        if not swapped:
            answer.unlink()
            answer.symlink_to(controller)
            swapped.append(True)
        return info

    monkeypatch.setattr(os, "fstat", stat_fd)
    assert find_deliverables(evidence) == {"answer.txt": "correct"}

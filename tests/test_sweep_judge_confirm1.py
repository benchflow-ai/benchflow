"""Guard CONFIRM-taskmd-rewards-1 against reviewed commit 11fc9381."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.task import RolloutPaths, Verifier
from benchflow.task.verifier_errors import AgentJudgeInputError
from benchflow.task.verifier_judge_inputs import _read_agent_judge_input


@pytest.mark.parametrize("prefix", ["/logs/", ""])
@pytest.mark.parametrize(
    "link_at", ["artifacts", "artifacts/output", "artifacts/output/notes.txt"]
)
async def test_agent_judge_rejects_every_local_link_component(
    tmp_path, prefix, link_at
):
    """CONFIRM-1: trusted declarations cannot turn local links into judge text."""
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    controller = tmp_path / "controller"
    (controller / "output").mkdir(parents=True)
    (controller / "notes.txt").write_text("DUMMY_CONTROLLER_MARKER")
    (controller / "output/notes.txt").write_text("DUMMY_CONTROLLER_MARKER")
    link = paths.rollout_dir / link_at
    if link.exists():
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(
        controller / "notes.txt" if link_at.endswith(".txt") else controller
    )
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    strategy = SimpleNamespace(
        inputs=[prefix + "artifacts/output/notes.txt"], name="judge"
    )
    with pytest.raises(AgentJudgeInputError):
        await verifier._collect_agent_judge_inputs(strategy)


def test_agent_judge_open_inode_survives_name_swap(tmp_path, monkeypatch):
    """CONFIRM-1: a checked inode, rather than the checked name, is consumed."""
    answer = tmp_path / "answer.txt"
    answer.write_text("correct")
    controller = tmp_path / "controller.txt"
    controller.write_text("DUMMY_CONTROLLER_MARKER")
    original = Path.is_file

    def is_file(path):
        result = original(path)
        if path == answer:
            answer.unlink()
            answer.symlink_to(controller)
        return result

    monkeypatch.setattr(Path, "is_file", is_file)
    assert _read_agent_judge_input(answer) == ("correct", False)


@pytest.mark.parametrize("kind", ["fifo", "directory", "link"])
def test_agent_judge_requires_opened_regular_file(tmp_path, kind):
    """CONFIRM-1: special file spellings cannot block or redirect the reader."""
    import os

    path = tmp_path / "input.txt"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        (tmp_path / "target.txt").write_text("DUMMY_CONTROLLER_MARKER")
        path.symlink_to(tmp_path / "target.txt")
    with pytest.raises(AgentJudgeInputError):
        _read_agent_judge_input(path)


def test_agent_judge_regular_unicode_control(tmp_path):
    path = tmp_path / "answer.txt"
    path.write_text("correct \u2713")
    assert _read_agent_judge_input(path) == ("correct \u2713", False)

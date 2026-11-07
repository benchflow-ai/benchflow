"""Guards PY8 FF-sw-rollout fixes against the 363d815b artifact regressions."""

import os

import pytest

from benchflow.task.config import ArtifactConfig
from tests.test_artifacts_collection import LocalTransport, _collect, _layout


@pytest.mark.parametrize("link", ["absolute", "relative", "inside", "dangling", "loop"])
@pytest.mark.parametrize(
    "destination", ["reports/answer.txt", "nested/reports/answer.txt"]
)
async def test_publication_never_follows_destination_ancestors(
    tmp_path, link, destination
):
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "answer.txt").write_text("42")
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = (trial / "artifacts" / destination).parent
    parent.parent.mkdir(parents=True, exist_ok=True)
    targets = {
        "absolute": str(outside),
        "relative": os.path.relpath(outside, parent.parent),
        "inside": str(trial / "artifacts" / "other"),
        "dangling": "absent",
        "loop": parent.name,
    }
    (trial / "artifacts" / "other").mkdir(exist_ok=True)
    parent.symlink_to(targets[link])
    try:
        result = await _collect(
            LocalTransport(),
            logs,
            workspace,
            trial,
            artifacts=[ArtifactConfig(source="answer.txt", destination=destination)],
            mounted=True,
        )
    except (OSError, RuntimeError):
        pytest.fail("invalid mounted links must produce a durable collection record")
    assert not (outside / "answer.txt").exists()
    assert result["collections"][1]["status"] in {"refused", "error"}
    assert not (trial / "artifacts" / "other" / "answer.txt").exists()


@pytest.mark.parametrize("collision", ["file", "directory", "dangling-link"])
async def test_remote_logs_collision_does_not_publish_partial_collection(
    tmp_path, collision
):
    """Guards the all-or-nothing preflight removed by a0b0554f."""
    logs, workspace, trial = _layout(tmp_path)
    (logs / "answer.txt").write_text("new answer")
    (logs / "existing.txt").write_text("new collision")
    existing = trial / "artifacts" / "existing.txt"
    if collision == "file":
        existing.write_text("previous answer")
    elif collision == "directory":
        existing.mkdir()
    else:
        existing.symlink_to("absent")
    result = await _collect(LocalTransport(), logs, workspace, trial)
    assert result["collections"][0]["status"] == "error"
    if collision == "file":
        assert existing.read_text() == "previous answer"
    elif collision == "directory":
        assert existing.is_dir()
    else:
        assert existing.is_symlink()
    assert not (trial / "artifacts" / "answer.txt").exists()

"""Guards PY8 FF-sw-rollout fixes against the 363d815b artifact regressions."""

import json

import pytest

from benchflow.rollout import _artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    build_transfer_payload,
)
from tests.test_artifacts_collection import LocalTransport, _collect, _layout
from tests.test_separate_verifier import _agent_trial


async def test_mounted_unreadable_file_records_error(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (trial / "artifacts" / "answer.txt").write_text("42")
    blocked = trial / "artifacts" / "debug.bin"
    blocked.write_text("dummy")
    blocked.chmod(0)
    result = await _collect(LocalTransport(), logs, workspace, trial, mounted=True)
    assert result["collections"][0]["status"] == "error"
    assert json.loads((trial / _artifacts.MANIFEST_NAME).read_text()) == result


async def test_mounted_loop_is_excluded_without_losing_answer(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (trial / "artifacts" / "answer.txt").write_text("42")
    (trial / "artifacts" / "loop").symlink_to("loop")
    result = await _collect(LocalTransport(), logs, workspace, trial, mounted=True)
    assert result["collections"][0]["status"] == "collected"
    assert any(f["path"] == "answer.txt" for f in result["files"])
    assert result["collections"][0]["exclusions"]


@pytest.mark.parametrize(
    "contents", [None, "{}", "[]", '{"version": 1, "collections": []}']
)
async def test_separate_transfer_requires_completed_inventory(tmp_path, contents):
    _, _, trial = await _agent_trial(tmp_path)
    manifest = trial / _artifacts.MANIFEST_NAME
    if contents is None:
        manifest.unlink()
    else:
        manifest.write_text(contents)
    with pytest.raises(SeparateVerifierError, match=r"artifact.*manifest"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


async def test_unexpected_collection_failure_is_durable(tmp_path, monkeypatch):
    """Guards FF-sw-rollout against the swallowed collection error at 363d815b."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    _, workspace, trial = _layout(tmp_path)
    rollout = SimpleNamespace(
        _env=object(),
        _config=SimpleNamespace(sandbox_user="agent"),
        _agent_env={},
        _agent_cwd=str(workspace),
        _rollout_paths=SimpleNamespace(
            rollout_dir=trial, artifacts_dir=trial / "artifacts"
        ),
        _task=SimpleNamespace(config=SimpleNamespace(artifacts=[])),
    )
    monkeypatch.setattr(
        _artifacts,
        "collect_artifacts",
        AsyncMock(side_effect=RuntimeError("dummy scan failure")),
    )
    await _artifacts.collect_rollout_artifacts(rollout)
    manifest = json.loads((trial / _artifacts.MANIFEST_NAME).read_text())
    assert manifest["collections"][0]["status"] == "error"
    assert "dummy scan failure" in manifest["collections"][0]["reason"]

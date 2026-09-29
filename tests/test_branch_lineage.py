"""Persisted fork observation regressions for selective PR #1046."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow.review.persistence as persistence
from benchflow.branch_artifacts import MountedArtifacts
from benchflow.branch_lineage import UnscoredChildError
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.paths import RolloutPaths


def setup_rollout(tmp_path):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_paths = RolloutPaths(rollout._rollout_dir)
    rollout._rollout_paths.mkdir()
    rollout.disconnect = AsyncMock()
    rollout._environment = type(
        "Environment",
        (),
        {
            "snapshot": AsyncMock(
                side_effect=[
                    StateSnapshot(id="first", path="/sensitive/first"),
                    StateSnapshot(id="second", path="/sensitive/second"),
                ]
            ),
            "restore": AsyncMock(),
        },
    )()
    return rollout


def persisted(rollout):
    return json.loads((rollout._rollout_dir / "tree.json").read_text())


async def test_repeated_forks_use_only_their_children_and_retain_snapshot_refs(
    tmp_path,
):
    """PR #1046: repeat forks retain distinct observation sets, including real zero."""
    rollout = setup_rollout(tmp_path)
    rollout.tree.root.state["arbitrary"] = {"secret": "DO_NOT_SERIALIZE"}
    assert (
        await rollout.branch(
            2, AsyncMock(return_value=0), child_labels=["baseline", None]
        )
        == 0
    )
    assert await rollout.branch(2, AsyncMock(return_value=1)) == 1
    document = persisted(rollout)
    assert document["schema_version"] == 1
    first, second = document["forks"]
    assert first["id"] != second["id"]
    assert [first["value"], second["value"]] == [0, 1]
    assert [fork["snapshot"]["environment"]["id"] for fork in (first, second)] == [
        "first",
        "second",
    ]
    assert {child["node_id"] for child in first["children"]}.isdisjoint(
        child["node_id"] for child in second["children"]
    )
    assert first["children"][0]["intervention"] == {
        "label": "baseline",
        "requested": None,
        "execution": "unspecified",
        "evidence": None,
    }
    assert first["snapshot"]["restore_available"] is None
    assert "mounted_contents" in first["snapshot"]["excluded"]
    for fork in (first, second):
        assert fork["status"] == "completed"
        assert fork["parent_restore"] == "restored"
        for child in fork["children"]:
            assert child["reward_source"] == "runner_return"
            assert child["artifacts"]["status"] == "available"
            assert (
                rollout._rollout_dir / child["artifacts"]["path"] / "observation.json"
            ).is_file()
    assert "DO_NOT_SERIALIZE" not in json.dumps(document)
    assert "/sensitive" not in json.dumps(document)


@pytest.mark.parametrize(
    "outcome,status,code",
    [
        ("failure", "failed", "execution_failed"),
        ("cancel", "cancelled", "cancelled"),
        ("unscored", "unscored", "missing_verifier_reward"),
        ("nan", "failed", "nonfinite_reward"),
    ],
)
async def test_child_outcome_is_durable_without_invented_scores(
    tmp_path, outcome, status, code
):
    """PR #1046: failures/cancellation/unscored/nonfinite persist explicit outcomes."""
    rollout = setup_rollout(tmp_path)

    async def child(node):
        if outcome == "failure":
            raise ValueError("SECRET_MESSAGE_NOT_IN_TREE")
        if outcome == "cancel":
            raise asyncio.CancelledError("SECRET_MESSAGE_NOT_IN_TREE")
        if outcome == "unscored":
            raise UnscoredChildError("SECRET_MESSAGE_NOT_IN_TREE")
        return float("nan")

    with pytest.raises(BaseException) as caught:
        await rollout.branch(2, child)
    if outcome == "cancel":
        assert isinstance(caught.value, asyncio.CancelledError)
    document = persisted(rollout)
    fork = document["forks"][0]
    first, second = fork["children"]
    assert first["status"] == status
    assert first["error"]["code"] == code
    assert first["reward"] is None
    assert first["reward_source"] is None
    assert second["status"] == "not_started"
    assert second["node_id"] is None
    assert fork["value"] is None

    assert fork["parent_restore"] == "restored"
    assert fork["status"] == ("cancelled" if outcome == "cancel" else "partial")
    assert "SECRET_MESSAGE_NOT_IN_TREE" not in json.dumps(document)


async def test_parent_restore_failure_is_separate_from_scored_children(tmp_path):
    """PR #1046: scores do not imply successful parent world restoration."""
    rollout = setup_rollout(tmp_path)
    rollout._environment.restore = AsyncMock(
        side_effect=[None, None, OSError("restore secret")]
    )
    with pytest.raises(OSError):
        await rollout.branch(2, AsyncMock(return_value=1))
    fork = persisted(rollout)["forks"][0]
    assert [child["status"] for child in fork["children"]] == ["scored", "scored"]
    assert fork["parent_restore"] == "failed"
    assert fork["parent_restore_error"] == {
        "type": "OSError",
        "code": "execution_failed",
    }
    assert fork["value"] is None

    assert rollout._branch_world_unsafe
    with pytest.raises(RuntimeError, match="Branch world is unsafe"):
        await rollout.connect()


async def test_capture_failure_has_no_fabricated_snapshot_or_restore(tmp_path):
    """PR #1046: capture failure is recorded without fabricating durable handles."""
    rollout = setup_rollout(tmp_path)
    rollout._environment.snapshot = AsyncMock(side_effect=OSError("snapshot secret"))
    with pytest.raises(OSError):
        await rollout.branch(2, AsyncMock(return_value=1))
    fork = persisted(rollout)["forks"][0]
    assert fork["snapshot"]["captured_layers"] == []
    assert fork["snapshot"]["environment"] is None
    assert fork["parent_restore"] == "not_attempted"
    assert fork["status"] == "failed"


async def test_failed_custody_never_publishes_parent_or_partial_artifact_link(
    tmp_path, monkeypatch
):
    """PR #1046: clickable child links require completed observation and custody."""
    rollout = setup_rollout(tmp_path)

    def fail(self, target):
        raise OSError("handoff failed")

    monkeypatch.setattr(MountedArtifacts, "hand_off", fail)
    with pytest.raises(OSError):
        await rollout.branch(2, AsyncMock(return_value=1))
    child = persisted(rollout)["forks"][0]["children"][0]
    assert child["artifacts"] == {"status": "unavailable", "path": None}


async def test_atomic_write_failure_retains_previous_file_and_primary_error(
    tmp_path, monkeypatch
):
    """PR #1046: write failure leaves valid prior lineage and keeps primary exception."""
    rollout = setup_rollout(tmp_path)
    original = persistence.os.replace
    previous = None
    calls = 0

    def replace(source, target):
        nonlocal calls, previous
        if Path(target).name != "tree.json":
            return original(source, target)
        calls += 1
        if calls >= 4:
            raise OSError("lineage publication failed")
        original(source, target)
        previous = Path(target).read_bytes()

    monkeypatch.setattr(persistence.os, "replace", replace)

    async def child(node):
        raise ValueError("primary child error")

    with pytest.raises(BaseExceptionGroup) as caught:
        await rollout.branch(2, child)

    def flatten(exc):
        return [str(exc)] + [
            message
            for error in getattr(exc, "exceptions", ())
            for message in flatten(error)
        ]

    errors = flatten(caught.value)
    assert "primary child error" in errors
    assert "lineage publication failed" in errors
    assert (rollout._rollout_dir / "tree.json").read_bytes() == previous
    assert persisted(rollout)["forks"][0]["status"] == "running"
    assert rollout._branch_child_active is False


async def test_child_and_parent_restore_failures_have_separate_durable_fields(tmp_path):
    """PR #1046: a failed rollback must not erase the original child's failure."""
    rollout = setup_rollout(tmp_path)
    rollout._environment.restore = AsyncMock(
        side_effect=[None, OSError("restore failure")]
    )
    with pytest.raises(BaseExceptionGroup):
        await rollout.branch(2, AsyncMock(side_effect=ValueError("child failure")))
    fork = persisted(rollout)["forks"][0]
    assert fork["children"][0]["error"]["type"] == "ValueError"
    assert fork["parent_restore_error"]["type"] == "OSError"
    assert fork["children"][1]["status"] == "not_started"

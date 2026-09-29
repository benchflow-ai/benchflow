"""Branching and recovery refuse to claim restoration of a physical embodiment.

A container snapshot or a replayed LLM session cannot reset a real arm or
scene, and replay would repeat real motion. These tests prove the refusal
happens before anything is quiesced, checkpointed, restored or replayed.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.embodiment import (
    TRIAL_RECORD_FILENAME,
    Embodiment,
    PhysicalRestoreRefused,
    embodiment_from_metadata,
    recorded_embodiment,
)
from benchflow.environment.protocol import StateSnapshot
from benchflow.robotics.tasks import build_tasks
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.task import Task
from tests.continue_run._helpers import completion, exchange, write_run_folder


def test_metadata_resolution_defaults_and_fails_closed():
    assert embodiment_from_metadata(None) == Embodiment()
    assert embodiment_from_metadata({"category": "coding"}).world_restore
    tagged = embodiment_from_metadata({"tags": ["robotics", "physical"]})
    assert tagged.physical and tagged.source == "tags"
    assert not tagged.world_restore and not tagged.action_replay
    declared = embodiment_from_metadata({"embodiment": "physical"})
    assert declared.physical and not declared.world_restore
    simulated = embodiment_from_metadata(
        {"embodiment": {"kind": "simulated", "world_restore": False}}
    )
    assert simulated.kind == "simulated"
    assert not simulated.world_restore and simulated.action_replay
    assert declared.restoration_record() == {
        "kind": "physical",
        "world_restore": False,
        "action_replay": False,
        "physical_state": "not_restorable",
        "retry_requires": "new_qualified_episode",
    }


@pytest.mark.parametrize(
    "declared",
    [
        {"kind": "physical", "world_restore": True},
        {"kind": "physical", "action_replay": True},
        {"kind": "robot"},
        {"kind": "simulated", "world_restore": "yes"},
        {"kind": "simulated", "reset": "operator"},
        {"kind": ["physical"]},
        ["physical"],
    ],
)
def test_invalid_or_overclaiming_declarations_raise(declared):
    with pytest.raises(ValueError):
        embodiment_from_metadata({"embodiment": declared})


def test_generated_physical_task_declares_its_embodiment(tmp_path):
    for task_dir in build_tasks(tmp_path):
        metadata = Task(task_dir).config.metadata
        assert metadata["embodiment"] == "physical"
        assert embodiment_from_metadata(metadata).physical


def _rollout(tmp_path, metadata):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    rollout._task = SimpleNamespace(config=SimpleNamespace(metadata=metadata))
    rollout.disconnect = AsyncMock()
    rollout._environment = SimpleNamespace(
        snapshot=AsyncMock(return_value=StateSnapshot(id="s")),
        restore=AsyncMock(),
    )
    return rollout


@pytest.mark.parametrize(
    "metadata",
    [
        {"embodiment": "physical"},
        {"tags": ["robotics", "physical", "llm-as-policy"]},
        {"embodiment": {"kind": "simulated", "world_restore": False}},
    ],
)
async def test_branch_refuses_before_quiesce_or_checkpoint(tmp_path, metadata):
    rollout = _rollout(tmp_path, metadata)
    runner = AsyncMock(return_value=1.0)
    cursor = rollout._cursor
    with pytest.raises(PhysicalRestoreRefused, match="branch refused"):
        await rollout.branch(2, runner, snapshot_layers={"environment", "sandbox"})
    rollout.disconnect.assert_not_awaited()
    rollout._environment.snapshot.assert_not_awaited()
    rollout._environment.restore.assert_not_awaited()
    runner.assert_not_awaited()
    assert not (rollout._rollout_dir / "tree.json").exists()
    assert rollout._cursor is cursor and not cursor.children


async def test_physical_refusal_names_the_required_new_episode(tmp_path):
    rollout = _rollout(tmp_path, {"embodiment": "physical"})
    with pytest.raises(PhysicalRestoreRefused) as caught:
        await rollout.branch(2, AsyncMock(return_value=1.0))
    assert caught.value.embodiment.physical
    assert "operator-qualified physical reset" in str(caught.value)
    assert "never replays physical actions" in str(caught.value)


async def test_virtual_task_still_branches(tmp_path):
    rollout = _rollout(tmp_path, {"category": "coding"})
    assert await rollout.branch(2, AsyncMock(return_value=1.0)) == 1.0
    rollout._environment.snapshot.assert_awaited_once()


def _nested_rollout_dir(trial):
    """The layout ``benchflow.robotics.runner`` gives its BenchFlow rollout."""
    return trial / "benchflow" / "20260101T000000Z-0123456789" / "agent"


def test_continue_refuses_rollout_inside_legacy_physical_trial(tmp_path):
    trial = tmp_path / "20260101T000000Z-0123456789"
    trial.mkdir()
    (trial / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "physical_trial",
                "reset_id": "reset-1",
                "status": "awaiting_assessment",
            }
        )
    )
    folder = write_run_folder(
        _nested_rollout_dir(trial), exchanges=[exchange(completion(content="a"))]
    )
    with pytest.raises(RunFolderError, match="benchflow continue refused"):
        load_run_folder(folder)


def test_continue_refuses_before_reading_other_artifacts(tmp_path):
    trial = tmp_path / "trial"
    folder = _nested_rollout_dir(trial)
    folder.mkdir(parents=True)
    (trial / TRIAL_RECORD_FILENAME).write_text(
        json.dumps({"embodiment": {"kind": "physical"}})
    )
    (folder / "config.json").write_text(json.dumps({"agent": "openhands"}))
    # No llm_trajectory.jsonl: the physical refusal comes first.
    with pytest.raises(RunFolderError, match="physical embodiment"):
        load_run_folder(folder)


def test_continue_refuses_local_physical_task(tmp_path):
    task_dir = build_tasks(tmp_path / "tasks")[0]
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    config = json.loads((folder / "config.json").read_text())
    config["task_path"] = str(task_dir)
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises(RunFolderError, match="never replays physical actions"):
        load_run_folder(folder)


def test_unreadable_trial_record_is_not_a_restorable_world(tmp_path):
    (tmp_path / TRIAL_RECORD_FILENAME).write_text("{truncated")
    with pytest.raises(ValueError, match="unreadable embodiment evidence"):
        recorded_embodiment(tmp_path / "benchflow" / "job" / "agent")


def test_unrelated_manifest_does_not_mark_a_run_physical(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": "dataset"}))
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "manifest.json").write_text("not json")
    assert recorded_embodiment(tmp_path / "other") is None
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    assert load_run_folder(folder).agent == "openhands"

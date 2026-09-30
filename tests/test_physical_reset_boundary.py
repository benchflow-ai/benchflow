"""The robotics runner's embodiment records (``benchflow.embodiment``).

Branching, checkpoint restores and replay now take their rule from
``benchflow.embodied.spec`` (tests/test_embodied_restore_boundary.py), which
reads only ``metadata.embodied``. What is left here is the robotics runner's
own resolution of its ``metadata.embodiment`` key and trial records, and the
unified gate refusing the runner's tasks.
"""

import json

import pytest

from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.embodiment import (
    TRIAL_RECORD_FILENAME,
    Embodiment,
    embodiment_from_metadata,
    recorded_embodiment,
)
from benchflow.robotics.tasks import build_tasks
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


def test_continue_refuses_local_physical_task(tmp_path):
    task_dir = build_tasks(tmp_path / "tasks")[0]
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    config = json.loads((folder / "config.json").read_text())
    config["task_path"] = str(task_dir)
    (folder / "config.json").write_text(json.dumps(config))
    # The unified gate reads only metadata.embodied and refuses the retired key.
    with pytest.raises(RunFolderError, match=r"metadata\.embodiment is not read"):
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

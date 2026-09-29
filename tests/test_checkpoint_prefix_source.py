"""A trial started from a kept checkpoint records where its prefix is.

Children of a --from-checkpoint trial were exported
with an empty prefix although the source trial's conversation up to the
checkpoint is on disk. Checkpoint rows now record ``trajectory_events`` (the
length of the trajectory at the checkpoint) and ``checkpoint_source.json``
records the source trial's path and ``prefix_events``, which the branch-tree
exporter reads (tests/test_export_branch_tree_v2.py).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchflow.branch_run import load_checkpoint_source
from benchflow.checkpoints import after_prompt, parse_checkpoint_policy
from tests.test_auto_checkpoints import Sandbox


async def test_checkpoints_record_the_trajectory_position(tmp_path):
    run = tmp_path / "task__abc"
    run.mkdir()
    rollout = SimpleNamespace(
        _config=SimpleNamespace(checkpoints=parse_checkpoint_policy("every-prompt")),
        _env=Sandbox(),
        _rollout_dir=run,
        _rollout_name="task__abc",
        _cursor=SimpleNamespace(id="n3"),
        _session=None,
        _trajectory=[{"type": "user_message"}, {"type": "agent_message"}],
    )
    await after_prompt(rollout, 1)
    [row] = json.loads((run / "checkpoints.json").read_text())["checkpoints"]
    assert row["trajectory_events"] == 2


def _source(tmp_path, row_extra=None, tree=None):
    trial = tmp_path / "old" / "task__src"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task_path": "task"}))
    (trial / "checkpoints.json").write_text(
        json.dumps(
            {
                "kind": "benchflow-checkpoints",
                "checkpoints": [
                    {
                        "id": "prompt:1",
                        "after_prompt": 1,
                        "node_id": "n4",
                        "provider": "daytona",
                        "ref": "bf-snap-x",
                        "status": "kept",
                        **(row_extra or {}),
                    }
                ],
            }
        )
    )
    if tree is not None:
        (trial / "tree.json").write_text(json.dumps(tree))
    return trial


def test_source_records_path_and_prefix_length(tmp_path):
    trial = _source(tmp_path, {"trajectory_events": 4})
    record = load_checkpoint_source(trial, None).to_record()
    assert record["trial_path"] == str(trial.resolve())
    assert record["prefix_events"] == 4


def test_a_kept_fork_uses_its_parent_node_step(tmp_path):
    tree = {
        "kind": "benchflow-branch-tree",
        "schema_version": 1,
        "nodes": [{"id": "n4", "parent": "n3", "step_id": "step-3-agent_message"}],
        "forks": [
            {
                "id": "f1",
                "parent_node": "n4",
                "snapshot": {
                    "retention": "kept",
                    "captured_layers": ["sandbox"],
                    "sandbox": {"provider": "daytona", "ref": "bf-snap-f"},
                },
            }
        ],
    }
    record = load_checkpoint_source(_source(tmp_path, tree=tree), "f1").to_record()
    assert record["prefix_events"] == 4


def test_older_checkpoints_have_no_prefix_length(tmp_path):
    assert (
        load_checkpoint_source(_source(tmp_path), None).to_record()["prefix_events"]
        is None
    )

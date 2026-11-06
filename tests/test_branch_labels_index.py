"""Each fork folder has a labels.json index.

Child folders are named by node id (``children/n1``, ``children/n8``), not
by label; a reader had to open each ``observation.json`` to match them.
``branches/<fork>/labels.json`` now lists every child's label, node id,
folder, status and reward. The ``bench eval branch`` header also named the
kept checkpoint by its provider snapshot ref and said "after prompt 0" for
``--checkpoint prompt:1``; it now names the checkpoint and
the source trial, and never prints a snapshot ref.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _runner, _tree


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


async def test_fork_folder_indexes_children_by_label(tmp_path):
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"hint": "Do it.", "recheck": "Other."}),
        snapshot_layers={"sandbox"},
        child_labels=["hint", "recheck"],
        isolate_children=True,
    )
    fork = _tree(root)["forks"][0]
    index = json.loads(
        (root._rollout_dir / "branches" / fork["id"] / "labels.json").read_text()
    )
    assert index["kind"] == "benchflow-branch-labels"
    assert index["fork_id"] == fork["id"]
    rows = {row["label"]: row for row in index["children"]}
    assert set(rows) == {"hint", "recheck"}
    for child in fork["children"]:
        row = rows[child["intervention"]["label"]]
        assert row["node_id"] == child["node_id"]
        assert row["folder"] == Path(child["artifacts"]["path"]).name
        assert (row["status"], row["reward"]) == (child["status"], child["reward"])
        assert (root._rollout_dir / child["artifacts"]["path"]).is_dir()


def test_header_names_the_checkpoint_not_the_snapshot_ref(tmp_path):
    from benchflow.branch_run import CheckpointSource
    from benchflow.cli.branch import _trial_header

    source = CheckpointSource(
        trial_dir=tmp_path / "old" / "task__old",
        fork_id="prompt:1",
        provider="daytona",
        ref="bf-snap-secret-ref",
        task_name="task",
    )

    from types import SimpleNamespace

    Plan = SimpleNamespace(
        agent="claude-agent-acp",
        sandbox="daytona",
        children=[object(), object()],
        checkpoint_after=0,
        task_paths=[tmp_path / "task"],
    )
    text = _trial_header(Plan, source, tmp_path / "task", 1)
    assert "bf-snap-secret-ref" not in text
    assert "from checkpoint prompt:1 of task__old" in text
    assert "after prompt 0" not in text
    Plan.checkpoint_after = 1
    assert "2 children after prompt 1" in _trial_header(Plan, None, tmp_path / "t", 1)

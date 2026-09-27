"""The viewer's Lineage tab over rollout-branch ``tree.json`` files.

``tree.json`` and each child's ``observation.json`` are written here by the
real ``Rollout.branch`` engine (branch_lineage.py / branch_result.py, adapted
from PR #1046), so the viewer is pinned to the writer's
schema rather than to a hand-copied fixture.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.branch_lineage import UnscoredChildError
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.paths import RolloutPaths
from benchflow.trajectories.viewer import _HF_VIEWER_FILES, render_rollout
from benchflow.trajectories.viewer.catalog import _rollout_summary
from benchflow.trajectories.viewer.payload import (
    _build_acp_payload,
    _build_branch_child_payload,
)
from benchflow.trajectories.viewer.server import _serve_browse


async def branched_rollout(tmp_path: Path) -> Path:
    """A parent rollout with two forks: labelled 1/0 children, then 1/unscored."""
    run = tmp_path / "jobs" / "job" / "count-files__b1a2c3d4"
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )
    rollout._rollout_dir = run
    rollout._rollout_paths = RolloutPaths(run)
    rollout._rollout_paths.mkdir()
    rollout.disconnect = AsyncMock()
    rollout._environment = type(
        "Environment",
        (),
        {
            "snapshot": AsyncMock(
                side_effect=[
                    StateSnapshot(id="first", path="/state/first"),
                    StateSnapshot(id="second", path="/state/second"),
                ]
            ),
            "restore": AsyncMock(),
        },
    )()
    outcomes = iter([1.0, 0.0, 1.0, None])

    async def child(node):
        reward = next(outcomes)
        rollout._trajectory.append(
            {"type": "agent_message", "text": f"continuation of {node.id}"}
        )
        verifier = rollout._rollout_paths.verifier_dir
        verifier.mkdir(parents=True, exist_ok=True)
        if reward is None:
            rollout._verifier_error = "verifier timed out after 600s"
            raise UnscoredChildError("no canonical reward")
        (verifier / "reward.txt").write_text(str(reward))
        return reward

    await rollout.branch(2, child, child_labels=["baseline", "hint: off-by-one"])
    with pytest.raises(UnscoredChildError):
        await rollout.branch(2, child)
    (run / "trajectory").mkdir()
    (run / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": "parent"})
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "task_name": "count-files",
                "agent_name": "dummy",
                "model": "m",
                "rewards": {"reward": 1.0},
            }
        )
    )
    return run


async def test_lineage_lists_each_fork_parent_to_children(tmp_path):
    """Guards the Lineage tab's reading of solver-evidence preservation's tree.json: every child
    keeps its own reward (a real zero included), status and intervention."""
    run = await branched_rollout(tmp_path)
    lineage = _build_acp_payload(run, None).to_payload()["lineage"]
    assert lineage["nodes"] == 5 and lineage["error"] is None
    first, second = lineage["forks"]
    assert first["parent_node"] == "root"
    assert first["status"] == "completed" and first["value"] == 0.5
    assert first["captured_layers"] == ["environment"]
    assert first["parent_restore"] == "restored"
    assert [
        (child["node_id"], child["status"], child["reward"])
        for child in first["children"]
    ] == [("n1", "scored", 1.0), ("n2", "scored", 0.0)]
    assert [child["intervention"]["label"] for child in first["children"]] == [
        "baseline",
        "hint: off-by-one",
    ]
    assert first["children"][0]["intervention"]["execution"] == "unspecified"
    assert first["children"][0]["ref"] == f"{first['id']}/n1"

    assert second["status"] == "partial" and second["value"] is None
    unscored = second["children"][1]
    assert unscored["status"] == "unscored" and unscored["reward"] is None
    # Error records are type + code only; messages never enter tree.json.
    assert unscored["error"] == "UnscoredChildError (missing_verifier_reward)"


async def test_child_trajectory_opens_from_its_observation(tmp_path):
    """Guards the Lineage links: a child's continuation, verdict and mounted
    verifier sidecars load as their own trajectory page."""
    run = await branched_rollout(tmp_path)
    lineage = _build_acp_payload(run, None).lineage
    assert lineage is not None
    fork = lineage.forks[0]
    payload = _build_branch_child_payload(run, f"{fork.id}/n2")
    assert payload is not None
    wire = payload.to_payload()
    assert [step["text"] for step in wire["steps"]] == ["continuation of n2"]
    assert wire["meta"]["reward"] == 0.0
    assert wire["meta"]["task_name"] == "count-files"
    assert wire["meta"]["branch"]["node_id"] == "n2"
    assert wire["meta"]["branch"]["intervention"] == "hint: off-by-one"
    assert wire["verifier"]["reward"] == "0.0"
    assert wire["rollout_name"].endswith("/ branch n2")

    unscored = _build_branch_child_payload(run, f"{lineage.forks[1].id}/n4")
    assert unscored is not None
    status = unscored.to_payload()["meta"]["status"]
    # An unscored child finished its continuation; only the verdict is missing.
    assert status["summary"] == "completed but unscored"
    assert status["assessment_detail"] == "verifier timeout"


async def test_child_refs_resolve_only_through_the_tree(tmp_path):
    """Guards the child endpoint against crafted refs and escaping paths."""
    run = await branched_rollout(tmp_path)
    fork_id = json.loads((run / "tree.json").read_text())["forks"][0]["id"]
    for crafted in ("", "../x", f"{fork_id}/n9", f"{fork_id}/../n1", "branches"):
        assert _build_branch_child_payload(run, crafted) is None

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "observation.json").write_text(json.dumps({"trajectory": []}))
    document = json.loads((run / "tree.json").read_text())
    # tmp/jobs/job/<run>/../../../outside is a real child archive outside the run.
    document["forks"][0]["children"][0]["artifacts"]["path"] = "../../../outside"
    document["forks"][0]["children"][1]["artifacts"] = {
        "status": "unavailable",
        "path": None,
    }
    (run / "tree.json").write_text(json.dumps(document))
    children = _build_acp_payload(run, None).to_payload()["lineage"]["forks"][0][
        "children"
    ]
    assert [child["ref"] for child in children] == [None, None]
    assert children[1]["artifacts_status"] == "unavailable"
    assert _build_branch_child_payload(run, f"{fork_id}/n1") is None


@pytest.mark.parametrize(
    "document,error",
    [
        ("{not json", "not a benchflow-branch-tree document"),
        (json.dumps({"kind": "other"}), "not a benchflow-branch-tree document"),
        (
            json.dumps({"kind": "benchflow-branch-tree", "schema_version": 9}),
            "unsupported schema_version 9",
        ),
    ],
)
def test_unreadable_tree_is_reported_not_raised(tmp_path, document, error):
    """Guards the viewer against crashing on a malformed tree.json."""
    (tmp_path / "trajectory").mkdir()
    (tmp_path / "trajectory" / "acp_trajectory.jsonl").write_text("")
    (tmp_path / "tree.json").write_text(document)
    lineage = _build_acp_payload(tmp_path, None).to_payload()["lineage"]
    assert lineage["error"] == error and lineage["forks"] == []


def test_runs_without_a_tree_have_no_lineage(tmp_path):
    """Guards ordinary runs against an empty Lineage tab."""
    (tmp_path / "trajectory").mkdir()
    (tmp_path / "trajectory" / "acp_trajectory.jsonl").write_text("")
    assert _build_acp_payload(tmp_path, None).lineage is None


async def test_run_list_counts_the_forks_and_children_of_a_branched_run(tmp_path):
    """Regression test: the run list did not mark runs that have
    branches. Its row data now carries the fork and child counts read from
    the run's tree.json; runs without one carry none."""
    run = await branched_rollout(tmp_path)
    base = tmp_path / "jobs"
    row = _rollout_summary(base, f"job/{run.name}")
    assert (row["branch_forks"], row["branch_children"]) == (2, 4)

    plain = base / "job" / "plain__00000001"
    (plain / "trajectory").mkdir(parents=True)
    (plain / "trajectory" / "acp_trajectory.jsonl").write_text("")
    row = _rollout_summary(base, f"job/{plain.name}")
    assert (row["branch_forks"], row["branch_children"]) == (None, None)


async def test_single_page_embeds_every_openable_child(tmp_path):
    """Guards ``?branch=`` links on the server-less trajectory.html page."""
    run = await branched_rollout(tmp_path)
    page = render_rollout(run)
    data = page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    boot = json.loads(data)
    assert sorted(ref.rsplit("/", 1)[1] for ref in boot["branches"]) == [
        "n1",
        "n2",
        "n3",
        "n4",
    ]


async def test_browse_api_serves_children_by_ref(tmp_path):
    """Guards ``/api/rollout?id=…&branch=…``: known children load, others 404."""
    run = await branched_rollout(tmp_path)
    fork_id = json.loads((run / "tree.json").read_text())["forks"][0]["id"]
    base = tmp_path / "jobs"
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=_serve_browse, args=(base, port, 1), daemon=True).start()
    root = f"http://localhost:{port}/api/rollout?id=job/{run.name}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    with urllib.request.urlopen(f"{root}&branch={fork_id}/n1", timeout=5) as reply:
        assert json.loads(reply.read())["meta"]["branch"]["node_id"] == "n1"
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f"{root}&branch={fork_id}/n7", timeout=5)
    assert missing.value.code == 404


def test_dataset_slices_fetch_the_tree_and_the_recovery_pointer():
    """Guards hf:// sources: the small lineage/recovery JSON is fetched by
    exact name, never through a wildcard."""
    assert "tree.json" in _HF_VIEWER_FILES
    assert "verification.json" in _HF_VIEWER_FILES


async def test_isolated_and_nested_children_open_with_their_verifier(tmp_path):
    """Isolated children (branch(isolate_children=True))
    are full trial folders: their verifier output is at
    branches/<fork>/children/<node>/verifier, not under mounted/. A nested
    fork (a child branching again) is a second fork of the same tree whose
    children open the same way."""
    from tests.test_branch_isolated import _root

    root = await _root(tmp_path)

    async def leaf(node, *, child):
        sub = child.rollout
        await sub.connect()
        await sub.execute([child.label], node=node)
        verifier = sub._rollout_dir / "verifier"
        verifier.mkdir()
        reward = 1.0 if child.label in {"a", "a1"} else 0.0
        (verifier / "reward.txt").write_text(str(reward))
        return reward

    async def top(node, *, child):
        if child.label == "a":
            await child.rollout.branch(
                2,
                leaf,
                snapshot_layers={"sandbox"},
                child_labels=["a1", "a2"],
                isolate_children=True,
            )
        return await leaf(node, child=child)

    await root.branch(
        2,
        top,
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
        concurrency=2,
    )
    (root._rollout_dir / "result.json").write_text(
        json.dumps({"task_name": "t", "rewards": {"reward": 1.0}})
    )
    lineage = _build_acp_payload(root._rollout_dir, None).lineage
    assert lineage is not None and len(lineage.forks) == 2
    opened = {}
    for fork in lineage.forks:
        for child in fork.children:
            payload = _build_branch_child_payload(root._rollout_dir, child.ref)
            assert payload is not None, child.ref
            wire = payload.to_payload()
            opened[child.intervention["label"]] = wire["verifier"]["reward"]
    assert opened == {"a": "1.0", "b": "0.0", "a1": "1.0", "a2": "0.0"}

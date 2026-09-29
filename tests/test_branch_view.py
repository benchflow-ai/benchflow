"""The branch-view contract: one versioned JSON projection of a trial's branches.

A viewer needs one stable JSON per trial with, per fork and per child, what was requested, tokens, USD,
sandbox-seconds, timings and lineage. Until now a reader had to know every
tree.json field added over several releases and the result.json ``branches`` and
``retry`` blocks. ``benchflow.branch_view.load_branch_view(trial_dir)``
returns ``benchflow.branch-view/1`` (JSON Schema ``BRANCH_VIEW_SCHEMA``, doc
``docs/reference/branch-view.md``); ``bf.load_trial(...).branch_view`` and
``bf.load_job(...).branch_views()`` expose it; ``bench eval branches PATH
--json`` prints it. Older trees (fields missing) read as nulls, never errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.branch_view import BRANCH_VIEW_SCHEMA, load_branch_view
from benchflow.cli.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "branch_tree_claude" / "claude-continue"


def _cost(tokens, seconds, usd=None):
    return {"tokens": tokens, "usd": usd, "sandbox_seconds": seconds}


def _child(i, node, label, reward, path, *, requested=None, status="scored"):
    return {
        "index": i,
        "node_id": node,
        "status": status,
        "reward": reward,
        "reward_source": "verifier" if reward is not None else None,
        "error": None,
        "cleanup_error": None,
        "intervention": {
            "label": label,
            "requested": requested,
            "execution": "runner",
            "evidence": None,
        },
        "artifacts": {"status": "available", "path": path},
        "usage": {"total_tokens": 100},
        "timing_sec": {
            "agent_execution": 3.0,
            "verifier": 30.0,
            "sandbox_from_snapshot": 19.0,
            "install_agent": 5.0,
        },
        "cost": _cost(100, 60.0),
        "snapshot_start": {
            "agent": "reused",
            "verifier_baseline": "inherited",
            "setup_commands": "skipped",
        },
    }


def _fork(fid, rollout, parent_node, children, *, value, kind=None, **extra):
    record = {
        "id": fid,
        "rollout": rollout,
        "parent_node": parent_node,
        "requested_children": len(children),
        "snapshot": {
            "requested_layers": ["sandbox"],
            "captured_layers": ["sandbox"],
            "agent_session": "fresh",
            "retention": "deleted",
            "sandbox": {
                "provider": "daytona",
                "ref": "bf-snap-private",
                "digest": None,
            },
            "excluded": ["agent_session"],
        },
        "status": "completed",
        "parent_restore": "restored",
        "value": value,
        "children_mode": {"isolated": True, "concurrency": 2, "prewarm": 2},
        "timing_sec": {
            "checkpoint": 45.0,
            "child_restore": [19.0, 19.0],
            "children": 70.0,
            "parent_restore": 2.0,
        },
        "cost": {
            "tokens": 200,
            "usd": None,
            "usd_known": False,
            "wall_seconds": 120.0,
            "parent_sandbox_seconds": 120.0,
            "children_sandbox_seconds": 120.0,
            "sandbox_seconds": 240.0,
        },
        "children": children,
        **extra,
    }
    if kind:
        record["kind"] = kind
    return record


def _trial(tmp_path: Path) -> Path:
    trial = tmp_path / "job" / "task__abc"
    f1, f2, f3 = "a" * 32, "b" * 32, "c" * 32
    paths = {
        n: f"branches/{f}/children/{n}"
        for f, n in [(f1, "n5"), (f1, "n6"), (f2, "n9"), (f2, "n10"), (f3, "n12")]
    }
    for rel in paths.values():
        (trial / rel).mkdir(parents=True)
        (trial / rel / "observation.json").write_text("{}")
        (trial / rel / "result.json").write_text("{}")
    tree = {
        "schema_version": 1,
        "kind": "benchflow-branch-tree",
        "nodes": [
            {"id": n, "parent": None, "step_id": None}
            for n in ["root", "n4", "n5", "n6", "n8", "n9", "n10", "n11", "n12"]
        ],
        "forks": [
            _fork(
                f1,
                "task__abc",
                "n4",
                [
                    _child(
                        0,
                        "n5",
                        "hint",
                        0.0,
                        paths["n5"],
                        requested="own prompt (153 characters, sha256:000000000abc)",
                    ),
                    _child(
                        1,
                        "n6",
                        "baseline",
                        1.0,
                        paths["n6"],
                        requested="parent's remaining prompts (1)",
                    ),
                ],
                value=0.5,
            ),
            _fork(
                f2,
                "n5",
                "n8",
                [
                    _child(0, "n9", "fix", 1.0, paths["n9"]),
                    _child(1, "n10", "leave", None, paths["n10"], status="unscored"),
                ],
                value=None,
            ),
            _fork(
                f3,
                "task__abc",
                "n11",
                [_child(0, "n12", "retry", 1.0, paths["n12"])],
                value=1.0,
                kind="retry",
                reason="failure",
                checkpoint="prompt:1",
            ),
        ],
    }
    (trial / "tree.json").write_text(json.dumps(tree))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rollout_name": "task__abc",
                "rewards": {"reward": 0.0},
                "retry": {
                    "status": "completed",
                    "reason": "failure",
                    "checkpoint": "prompt:1",
                    "fork_id": f3,
                    "reward": 1.0,
                    "original_reward": 0.0,
                    "path": paths["n12"],
                },
            }
        )
    )
    return trial


def test_modern_tree_projects_every_field(tmp_path):
    view = load_branch_view(_trial(tmp_path))
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["schema"] == "benchflow.branch-view/1"
    assert view["trial"] == {
        "name": "task__abc",
        "task": "task",
        "reward": 0.0,
        "retry": {
            "status": "completed",
            "reason": "failure",
            "checkpoint": "prompt:1",
            "fork_id": "c" * 32,
            "reward": 1.0,
            "original_reward": 0.0,
            "path": "branches/" + "c" * 32 + "/children/n12",
        },
        # 1.1 (tests/test_branch_parent_discarded.py, test_branch_view_1_1.py)
        "checkpoint_source": None,
        "parent": "kept",
        "unscored_by_design": False,
    }
    outer, nested, retry = view["forks"]
    assert (outer["kind"], outer["depth"], outer["forked_by"]) == (
        "fork",
        1,
        {"rollout": "task__abc", "child_node": None, "child_label": None},
    )
    assert nested["depth"] == 2
    assert nested["forked_by"] == {
        "rollout": "n5",
        "child_node": "n5",
        "child_label": "hint",
    }
    assert (retry["kind"], retry["reason"], retry["checkpoint"]) == (
        "retry",
        "failure",
        "prompt:1",
    )
    assert outer["children_mode"] == {"isolated": True, "concurrency": 2, "prewarm": 2}
    assert outer["snapshot"] == {
        "layers_requested": ["sandbox"],
        "layers_captured": ["sandbox"],
        "agent_session": "fresh",
        "retention": "deleted",
        "provider": "daytona",
        "reused": False,  # 1.1
    }
    assert outer["cost"]["sandbox_seconds"] == 240.0
    hint, baseline = outer["children"]
    assert hint["requested"] == "own prompt (153 characters, sha256:000000000abc)"
    assert (hint["advantage"], baseline["advantage"]) == (-0.5, 0.5)
    assert hint["cost"] == _cost(100, 60.0)
    assert hint["timing_sec"]["sandbox_from_snapshot"] == 19.0
    assert hint["snapshot_start"]["verifier_baseline"] == "inherited"
    assert hint["nested_forks"] == ["b" * 32]
    # A tree written before child retries existed has no attempts: null.
    assert (hint["attempts"], hint["retried_after"]) == (None, None)
    assert baseline["nested_forks"] == []
    assert hint["archive"] == {
        "path": "branches/" + "a" * 32 + "/children/n5",
        "observation": "branches/" + "a" * 32 + "/children/n5/observation.json",
        "result": "branches/" + "a" * 32 + "/children/n5/result.json",
    }
    leave = nested["children"][1]
    assert leave["reward"] is None and leave["advantage"] is None
    assert view["totals"]["forks"] == 3 and view["totals"]["children"] == 5
    # Provider refs are capability handles: never in the view.
    assert "bf-snap-private" not in json.dumps(view)


def test_an_old_tree_reads_as_nulls(tmp_path):
    trial = next(FIXTURE.iterdir())
    view = load_branch_view(trial)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    [fork] = view["forks"]
    assert fork["kind"] == "fork" and fork["cost"] is None
    assert [c["label"] for c in fork["children"]] == ["baseline", "hint-reuse-draft"]
    assert all(c["cost"] is None and c["requested"] is None for c in fork["children"])
    assert fork["children"][0]["archive"]["result"] is None  # in-place child


def test_no_tree_is_an_empty_view(tmp_path):
    trial = tmp_path / "t"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_name": "t", "rewards": {"reward": 1.0}})
    )
    view = load_branch_view(trial)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["forks"] == [] and view["totals"]["children"] == 0


def test_sdk_accessors(tmp_path):
    trial = _trial(tmp_path)
    assert bf.load_trial(trial).branch_view == load_branch_view(trial)
    [view] = bf.load_job(trial.parent).branch_views()
    assert view["trial"]["name"] == "task__abc"
    loaded = bf.load_trial(trial)
    assert loaded.forks[0].cost["sandbox_seconds"] == 240.0
    assert loaded.forks[2].kind == "retry"
    assert loaded.forks[0].children[0].cost == _cost(100, 60.0)


@pytest.mark.parametrize("as_json", [True, False])
def test_cli_prints_the_view(tmp_path, as_json):
    trial = _trial(tmp_path)
    args = ["eval", "branches", str(trial)] + (["--json"] if as_json else [])
    result = CliRunner().invoke(app, args, terminal_width=200)
    assert result.exit_code == 0, result.output
    if as_json:
        [view] = json.loads(result.output)
        assert view == load_branch_view(trial)
    else:
        assert "hint" in result.output and "retry" in result.output


def test_the_builder_is_pure_and_vendorable(tmp_path):
    """A viewer with no benchflow dependency (reading files
    through its own storage layer) can copy benchflow/branch_view.py: the
    builder takes parsed documents and a file-exists callback, and the
    module imports nothing from benchflow."""
    import ast

    import benchflow.branch_view as module
    from benchflow.branch_view import build_branch_view

    tree_ast = ast.parse(Path(module.__file__).read_text())
    imported = {
        node.module or ""
        for node in ast.walk(tree_ast)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree_ast)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("benchflow") for name in imported), imported

    trial = _trial(tmp_path)
    tree = json.loads((trial / "tree.json").read_text())
    result = json.loads((trial / "result.json").read_text())
    view = build_branch_view(
        tree,
        result,
        trial_name=trial.name,
        file_exists=lambda rel: (trial / rel).is_file(),
    )
    assert view == load_branch_view(trial)

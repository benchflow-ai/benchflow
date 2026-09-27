"""``bench train convert --format branch-tree``: training rows from branch trees.

``bench train convert`` used to ignore ``tree.json``, so a branched run exported at most the
parent's own conversation (and native-subscription runs, the only ones that
can branch, have no LLM-proxy trajectory at all). The branch-tree format
exports one row per child: the sibling group's shared prefix (the parent's
events up to the fork, plus the enclosing child's events for a nested fork),
the child's own continuation, its reward, the fork's value V and the
advantage reward - V. ``--pairs`` adds one row per sibling pair whose rewards
differ (chosen / rejected, margin, whether both asked for the same thing).

Tests use a synthetic nested tree and a synthetic branched trial of the
bundled hello-world task (claude-agent-acp, baseline 1.0 / misleading hint 0.0)
in tests/fixtures/branch_tree_claude.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_branch import (
    events_to_messages,
    export_branch_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "branch_tree_claude"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_real_tree_exports_one_row_per_child(tmp_path):
    out = tmp_path / "children.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    # The two children were asked different things, so since the v2 rows
    # they pair only with any_request_pairs.
    stats = export_branch_jsonl(FIXTURE, out, pairs_out=pairs, any_request_pairs=True)
    assert (stats.trials, stats.forks, stats.child_rows, stats.pair_rows) == (
        1,
        1,
        2,
        1,
    )
    rows = _rows(out)
    assert [r["child"]["label"] for r in rows] == ["baseline", "hint-reuse-draft"]
    baseline, hint = rows
    # The shared prefix: the parent's draft turn, up to the fork at n4.
    assert baseline["prefix"] == hint["prefix"]
    assert baseline["prefix"][0]["role"] == "user"
    assert baseline["prefix"][0]["content"].startswith("This is step 1")
    assert baseline["parent_node"] == "n4"
    assert baseline["depth"] == 1
    # Each continuation starts with that child's own prompt.
    assert baseline["continuation"][0]["content"].startswith("Create a file called")
    assert hint["continuation"][0]["content"].startswith("The file draft.txt")
    assert (baseline["child"]["reward"], hint["child"]["reward"]) == (1.0, 0.0)
    assert baseline["value"] == 0.5
    assert (baseline["advantage"], hint["advantage"]) == (0.5, -0.5)
    assert baseline["messages"] == baseline["prefix"] + baseline["continuation"]
    tool_calls = [m for m in baseline["continuation"] if m.get("tool_calls")]
    # This run predates tool_name capture: the name is inferred.
    assert tool_calls[0]["tool_calls"][0]["function"]["name"] == "Write"
    assert baseline["tool_names"] == "inferred"
    [pair] = _rows(pairs)
    assert (pair["chosen_label"], pair["rejected_label"]) == (
        "baseline",
        "hint-reuse-draft",
    )
    assert pair["margin"] == 1.0
    assert pair["same_request"] is False
    assert pair["prompt"] == baseline["prefix"]
    assert pair["chosen"] == baseline["continuation"]


def _event(kind: str, text: str) -> dict:
    return {"type": kind, "text": text}


def _synthetic(tmp_path: Path) -> Path:
    """Parent: u0 a1 | fork F1 at n2 -> a (n3), b (n4); inside a: u, m |
    nested fork F2 at a's n6 -> a1 (n7), a2 (n8)."""
    trial = tmp_path / "jobs" / "job" / "task__x"
    (trial / "trajectory").mkdir(parents=True)
    parent = [_event("user_message", "draft"), _event("agent_message", "drafted")]
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in parent) + "\n"
    )

    def child(fork, node, events):
        path = f"branches/{fork}/children/{node}"
        (trial / path).mkdir(parents=True)
        (trial / path / "observation.json").write_text(
            json.dumps({"kind": "branch-child", "node_id": node, "trajectory": events})
        )
        return path

    a_events = [_event("user_message", "do a"), _event("agent_message", "did a")]
    paths = {
        "n3": child("f1", "n3", a_events),
        "n4": child("f1", "n4", [_event("user_message", "do b")]),
        "n7": child("f2", "n7", [_event("user_message", "do a1")]),
        "n8": child("f2", "n8", [_event("user_message", "do a2")]),
    }

    def kid(index, node, label, reward, status="scored", requested=None):
        return {
            "index": index,
            "node_id": node,
            "status": status,
            "reward": reward,
            "reward_source": "verifier",
            "intervention": {"label": label, "requested": requested},
            "artifacts": {"status": "available", "path": paths[node]},
        }

    tree = {
        "schema_version": 1,
        "kind": "benchflow-branch-tree",
        "nodes": [
            {"id": "root", "parent": None, "step_id": None},
            {"id": "n1", "parent": "root", "step_id": "step-0-user_message"},
            {"id": "n2", "parent": "n1", "step_id": "step-1-agent_message"},
            {"id": "n3", "parent": "n2", "step_id": "step-0-user_message"},
            {"id": "n4", "parent": "n2", "step_id": "step-0-user_message"},
            {"id": "n6", "parent": "n3", "step_id": "step-1-agent_message"},
            {"id": "n7", "parent": "n6", "step_id": "step-0-user_message"},
            {"id": "n8", "parent": "n6", "step_id": "step-0-user_message"},
        ],
        "forks": [
            {
                "id": "f1",
                "rollout": "task__x",
                "parent_node": "n2",
                "status": "completed",
                "value": 0.5,
                "children": [kid(0, "n3", "a", 1.0), kid(1, "n4", "b", 0.0)],
            },
            {
                "id": "f2",
                "rollout": "n3",
                "parent_node": "n6",
                "status": "partial",
                "value": None,
                "children": [
                    kid(0, "n7", "a1", 1.0, requested="same"),
                    kid(1, "n8", "a2", None, status="unscored", requested="same"),
                ],
            },
        ],
    }
    (trial / "tree.json").write_text(json.dumps(tree))
    (trial / "result.json").write_text(json.dumps({"task_name": "task"}))
    return tmp_path / "jobs"


def test_nested_fork_prefix_includes_the_enclosing_child(tmp_path):
    jobs = _synthetic(tmp_path)
    out = tmp_path / "out.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    stats = export_branch_jsonl(jobs, out, pairs_out=pairs, any_request_pairs=True)
    rows = {r["child"]["label"]: r for r in _rows(out)}
    assert set(rows) == {"a", "b", "a1", "a2"}
    contents = lambda msgs: [m["content"] for m in msgs]  # noqa: E731
    assert contents(rows["a"]["prefix"]) == ["draft", "drafted"]
    assert contents(rows["a1"]["prefix"]) == ["draft", "drafted", "do a", "did a"]
    assert rows["a1"]["prefix"] == rows["a2"]["prefix"]
    assert (rows["a"]["depth"], rows["a1"]["depth"]) == (1, 2)
    assert rows["a1"]["parent_child"] == "n3"
    # Unscored children keep their row but carry no reward or advantage, and
    # a fork without a value gives no advantage at all.
    assert rows["a2"]["child"]["reward"] is None
    assert rows["a2"]["advantage"] is None and rows["a1"]["advantage"] is None
    # Pairs only where both rewards exist and differ: (a, b) but not (a1, a2).
    assert [(p["chosen_label"], p["rejected_label"]) for p in _rows(pairs)] == [
        ("a", "b")
    ]
    assert stats.unscored_children == 1


def test_events_become_chat_messages():
    messages = events_to_messages(
        [
            {"type": "user_message", "text": "hi"},
            {"type": "agent_thought", "text": "hmm"},
            {
                "type": "tool_call",
                "tool_call_id": "t1",
                "title": "Write a.txt",
                "kind": "edit",
                "raw_input": {"file_path": "a.txt"},
                "raw_output": "ok",
            },
            {"type": "agent_message", "text": "done"},
            {"type": "oracle", "command": "oracle/solve.sh", "return_code": 0},
            {"type": "something_else"},
        ]
    )
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning": "hmm"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "Write a.txt",
                    "kind": "edit",
                    "arguments": {"file_path": "a.txt"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "[oracle] oracle/solve.sh exited 0"},
    ]


def test_cli_branch_tree_format(tmp_path):
    out = tmp_path / "rows.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "convert",
            str(FIXTURE),
            "--format",
            "branch-tree",
            "-o",
            str(out),
            "--pairs",
            str(pairs),
            "--pairs-any-request",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 child row(s)" in result.output
    assert "1 pair row(s)" in result.output
    assert len(_rows(out)) == 2


def test_cli_pairs_needs_branch_tree(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "train",
            "convert",
            str(FIXTURE),
            "-o",
            str(tmp_path / "x"),
            "--pairs",
            str(tmp_path / "p"),
        ],
    )
    assert result.exit_code == 1
    assert "--pairs" in result.output

"""Branch-tree export v2: rows a training pipeline can use as they are.

Gaps that were closed in ``bench train convert --format branch-tree``:

- tool calls were named by their display title ("Write hello.txt"),
  without the OpenAI ``function`` shape, a ``tools`` column or merged
  assistant turns, so ``apply_chat_template`` could not render them. Rows
  (``schema_version`` 2) now use the agent's tool name (inferred from the
  call when the agent did not send one, and flagged), ``{"type":
  "function", "function": {"name", "arguments": <JSON string>}}``, one
  assistant turn per reply, and a ``tools`` list inferred from the calls.
- pairs had no ``prompt`` and paired children that were asked different
  things. Pairs are now ``prompt`` / ``chosen`` / ``rejected`` message lists
  and, by default, only siblings asked the same thing (``any_request=True``
  keeps the rest, with the differing user turns inside chosen/rejected).
- children that resumed the parent's session were labelled
  ``session: "fresh"``, and children of a ``--from-checkpoint`` trial had an
  empty prefix. ``session`` now copies the fork's ``agent_session``, and the
  prefix of a trial started from a kept checkpoint begins with the source
  trial's conversation up to the checkpoint (``prefix_source``); when that
  cannot be found the row says ``prefix_complete: false``.
- oracle children became rows with a synthetic "[oracle] …" message.
  They are skipped by default (``include_oracle=True`` keeps them, tagged).
- ``--min-reward``, ``--expected-rows`` and ``--manifest`` were ignored
  for branch-tree; they now apply.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.trajectories.export_branch import (
    events_to_chat,
    export_branch_jsonl,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_trial(
    trial: Path,
    *,
    parent_events,
    children,
    agent_session="fresh",
    agent="claude-agent-acp",
    parent_node="n2",
    checkpoint_source=None,
):
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in parent_events)
    )
    fork_id = "f" * 32
    kids = []
    for index, (node, label, reward, events) in enumerate(children):
        path = f"branches/{fork_id}/children/{node}"
        (trial / path).mkdir(parents=True)
        (trial / path / "observation.json").write_text(
            json.dumps({"kind": "branch-child", "node_id": node, "trajectory": events})
        )
        kids.append(
            {
                "index": index,
                "node_id": node,
                "status": "scored",
                "reward": reward,
                "reward_source": "verifier",
                "intervention": {"label": label, "requested": None},
                "artifacts": {"status": "available", "path": path},
            }
        )
    rewards = [k["reward"] for k in kids]
    tree = {
        "schema_version": 1,
        "kind": "benchflow-branch-tree",
        "nodes": [
            {"id": "root", "parent": None, "step_id": None},
            {"id": "n1", "parent": "root", "step_id": "step-0-user_message"},
            {"id": "n2", "parent": "n1", "step_id": "step-1-agent_message"},
        ],
        "forks": [
            {
                "id": fork_id,
                "rollout": trial.name,
                "parent_node": parent_node,
                "status": "completed",
                "value": sum(rewards) / len(rewards),
                "snapshot": {"agent_session": agent_session},
                "children": kids,
            }
        ],
    }
    (trial / "tree.json").write_text(json.dumps(tree))
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task", "agent": agent, "model": "m"})
    )
    if checkpoint_source is not None:
        (trial / "checkpoint_source.json").write_text(json.dumps(checkpoint_source))
    return trial


def _user(text):
    return {"type": "user_message", "text": text}


def _say(text):
    return {"type": "agent_message", "text": text}


def _think(text):
    return {"type": "agent_thought", "text": text}


def _tool(call_id, title, kind, raw_input, output, tool_name=None):
    event = {
        "type": "tool_call",
        "tool_call_id": call_id,
        "title": title,
        "kind": kind,
        "raw_input": raw_input,
        "raw_output": output,
    }
    if tool_name:
        event["tool_name"] = tool_name
    return event


PARENT = [_user("Write a draft."), _say("Drafted.")]


def test_chat_template_shape():
    messages, tools, source = events_to_chat(
        [
            _user("Make hello.txt"),
            _think("I will write it."),
            _tool(
                "t1",
                "Write hello.txt",
                "edit",
                {"file_path": "/app/hello.txt", "content": "hi"},
                "ok",
                "Write",
            ),
            _tool(
                "t2",
                "cat /app/hello.txt",
                "execute",
                {"command": "cat /app/hello.txt"},
                "hi",
            ),
            _say("Done."),
        ]
    )
    assert messages == [
        {"role": "user", "content": "Make hello.txt"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I will write it.",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": json.dumps(
                            {"file_path": "/app/hello.txt", "content": "hi"}
                        ),
                    },
                },
                {
                    "id": "t2",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "cat /app/hello.txt"}),
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        {"role": "tool", "tool_call_id": "t2", "content": "hi"},
        {"role": "assistant", "content": "Done."},
    ]
    assert [t["function"]["name"] for t in tools] == ["Write", "Bash"]
    assert tools[0]["function"]["parameters"]["properties"] == {
        "file_path": {},
        "content": {},
    }
    assert source == "mixed"  # Write from the agent, Bash inferred


def test_rows_are_v2_with_session_agent_and_tools(tmp_path):
    trial = _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        agent_session="resumed",
        children=[
            (
                "n3",
                "a",
                1.0,
                [
                    _user("Go."),
                    _tool(
                        "t",
                        "Write x",
                        "edit",
                        {"file_path": "x", "content": "y"},
                        "ok",
                        "Write",
                    ),
                ],
            ),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
        ],
    )
    export_branch_jsonl(trial.parent, tmp_path / "out.jsonl")
    rows = _rows(tmp_path / "out.jsonl")
    assert {r["schema_version"] for r in rows} == {2}
    assert {r["session"] for r in rows} == {"resumed"}
    assert rows[0]["agent"] == "claude-agent-acp" and rows[0]["model"] == "m"
    assert rows[0]["tools"][0]["function"]["name"] == "Write"
    assert rows[0]["tool_names"] == "agent"
    assert rows[0]["messages"] == rows[0]["prefix"] + rows[0]["continuation"]
    assert rows[0]["prefix_complete"] is True


def test_pairs_are_dpo_ready_and_same_request_only_by_default(tmp_path):
    trial = _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[
            ("n3", "a", 1.0, [_user("Go."), _say("Yes.")]),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
            ("n5", "c", 0.5, [_user("Something else."), _say("Maybe.")]),
        ],
    )
    export_branch_jsonl(
        trial.parent, tmp_path / "o.jsonl", pairs_out=tmp_path / "p.jsonl"
    )
    [pair] = _rows(tmp_path / "p.jsonl")
    assert pair["schema_version"] == 2
    assert pair["prompt"] == [
        {"role": "user", "content": "Write a draft."},
        {"role": "assistant", "content": "Drafted."},
        {"role": "user", "content": "Go."},
    ]
    assert pair["chosen"] == [{"role": "assistant", "content": "Yes."}]
    assert pair["rejected"] == [{"role": "assistant", "content": "No."}]
    assert (pair["chosen_reward"], pair["rejected_reward"], pair["margin"]) == (
        1.0,
        0.0,
        1.0,
    )
    assert pair["same_request"] is True

    export_branch_jsonl(
        trial.parent,
        tmp_path / "o.jsonl",
        pairs_out=tmp_path / "all.jsonl",
        any_request_pairs=True,
    )
    cross = [p for p in _rows(tmp_path / "all.jsonl") if not p["same_request"]]
    assert len(cross) == 2
    # Different requests: the prompt stops at the shared prefix and each side
    # keeps its own user turn.
    assert cross[0]["prompt"][-1] == {"role": "assistant", "content": "Drafted."}
    assert cross[0]["chosen"][0]["role"] == "user"


def test_from_checkpoint_prefix_starts_with_the_source_conversation(tmp_path):
    source = tmp_path / "old" / "task__src"
    (source / "trajectory").mkdir(parents=True)
    (source / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in [
                _user("Remember BLUEBIRD."),
                _say("Noted."),
                _user("Later turn."),
                _say("x"),
            ]
        )
    )
    trial = _write_trial(
        tmp_path / "job" / "task__new",
        parent_events=[],
        parent_node="root",
        agent_session="resumed",
        checkpoint_source={
            "trial": "task__src",
            "trial_path": str(source),
            "fork_id": "prompt:1",
            "prefix_events": 2,
        },
        children=[
            ("n1", "a", 1.0, [_user("What was it?"), _say("BLUEBIRD")]),
            ("n2", "b", 0.0, [_user("What was it?"), _say("No idea")]),
        ],
    )
    export_branch_jsonl(trial.parent, tmp_path / "out.jsonl")
    row = _rows(tmp_path / "out.jsonl")[0]
    assert row["prefix"] == [
        {"role": "user", "content": "Remember BLUEBIRD."},
        {"role": "assistant", "content": "Noted."},
    ]
    assert row["prefix_source"] == {
        "trial": "task__src",
        "checkpoint": "prompt:1",
        "events": 2,
    }
    assert row["prefix_complete"] is True and row["session"] == "resumed"


def test_a_missing_source_prefix_is_flagged(tmp_path):
    trial = _write_trial(
        tmp_path / "job" / "task__new",
        parent_events=[],
        parent_node="root",
        checkpoint_source={"trial": "gone", "fork_id": "prompt:1"},
        children=[
            ("n1", "a", 1.0, [_user("Go."), _say("Yes.")]),
            ("n2", "b", 0.0, [_user("Go."), _say("No.")]),
        ],
    )
    export_branch_jsonl(trial.parent, tmp_path / "out.jsonl")
    assert {r["prefix_complete"] for r in _rows(tmp_path / "out.jsonl")} == {False}


def test_oracle_children_are_skipped_unless_asked(tmp_path):
    oracle = {"type": "oracle", "command": "solution/solve.sh", "return_code": 0}
    trial = _write_trial(
        tmp_path / "job" / "task__o",
        parent_events=[],
        parent_node="root",
        agent="oracle",
        children=[("n1", "a", 1.0, [oracle]), ("n2", "b", 1.0, [oracle])],
    )
    stats = export_branch_jsonl(trial.parent, tmp_path / "out.jsonl")
    assert _rows(tmp_path / "out.jsonl") == [] and stats.skipped_oracle == 2
    export_branch_jsonl(trial.parent, tmp_path / "all.jsonl", include_oracle=True)
    assert [r["agent"] for r in _rows(tmp_path / "all.jsonl")] == ["oracle", "oracle"]


def test_min_reward_expected_rows_and_manifest(tmp_path):
    trial = _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[
            ("n3", "a", 1.0, [_user("Go."), _say("Yes.")]),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
        ],
    )
    manifest = tmp_path / "m.json"
    stats = export_branch_jsonl(
        trial.parent, tmp_path / "out.jsonl", min_reward=1.0, manifest=manifest
    )
    assert [r["child"]["label"] for r in _rows(tmp_path / "out.jsonl")] == ["a"]
    assert stats.below_min_reward == 1
    recorded = json.loads(manifest.read_text())
    assert (recorded["child_rows"], recorded["below_min_reward"]) == (1, 1)
    with pytest.raises(ValueError, match="expected 5"):
        export_branch_jsonl(trial.parent, tmp_path / "x.jsonl", expected_rows=5)
    assert not (tmp_path / "x.jsonl").exists()


# ── the CLI ────────────────────────────────────────────


def _cli(*args):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    return CliRunner().invoke(app, ["train", "convert", *map(str, args)])


def _two_children(tmp_path):
    return _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[
            ("n3", "a", 1.0, [_user("Go."), _say("Yes.")]),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
        ],
    ).parent


def test_cli_applies_min_reward_expected_rows_and_manifest(tmp_path):
    jobs = _two_children(tmp_path)
    out, manifest = tmp_path / "o.jsonl", tmp_path / "m.json"
    result = _cli(
        jobs,
        "--format",
        "branch-tree",
        "-o",
        out,
        "--min-reward",
        "1.0",
        "--manifest",
        manifest,
    )
    assert result.exit_code == 0, result.output
    assert len(_rows(out)) == 1 and manifest.is_file()
    bad = _cli(
        jobs,
        "--format",
        "branch-tree",
        "-o",
        tmp_path / "x.jsonl",
        "--expected-rows",
        "5",
    )
    assert bad.exit_code == 1 and "expected 5" in bad.output


@pytest.mark.parametrize(
    "option",
    [
        ["--row-mode", "exchange"],
        ["--subagent-rows"],
        ["--context-policy", "message-window"],
        ["--canonical-selection", "sel.json"],
    ],
)
def test_cli_refuses_options_that_do_not_apply(tmp_path, option):
    result = _cli(
        _two_children(tmp_path),
        "--format",
        "branch-tree",
        "-o",
        tmp_path / "o.jsonl",
        *option,
    )
    assert result.exit_code == 1
    assert "branch-tree" in result.output and option[0] in result.output


def test_cli_include_oracle_and_any_request_need_branch_tree(tmp_path):
    result = _cli(
        _two_children(tmp_path), "-o", tmp_path / "o.jsonl", "--include-oracle"
    )
    assert result.exit_code == 1 and "--include-oracle" in result.output

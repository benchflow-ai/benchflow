"""``bench train validate --format branch-tree`` checks branch-tree rows.

Validating ``children.jsonl`` with
``--format branch-tree`` ran the TRL SFT validator and failed with
``row 1: prompt must be a non-empty message list``; the CLI reference also
listed a ``--pairs`` option that the command did not have. There is now a
branch-tree validator for child and pair rows: messages are prefix plus
continuation, siblings share their prefix, ``advantage == reward - value``,
every tool result answers an earlier call, tool calls are in the OpenAI
function shape and named in ``tools``, and a pair's chosen reward beats its
rejected one.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_branch import (
    export_branch_jsonl,
    validate_branch_jsonl,
)
from tests.test_export_branch_tree_v2 import PARENT, _say, _tool, _user, _write_trial


def _export(tmp_path):
    _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[
            (
                "n3",
                "a",
                1.0,
                [
                    _user("Go."),
                    _tool("c1", "Write x", "edit", {"file_path": "x"}, "ok"),
                    _say("Yes."),
                ],
            ),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
        ],
    )
    out, pairs = tmp_path / "children.jsonl", tmp_path / "pairs.jsonl"
    export_branch_jsonl(tmp_path / "job", out, pairs_out=pairs)
    return out, pairs


def _rewrite(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_exported_rows_validate(tmp_path):
    out, pairs = _export(tmp_path)
    result = validate_branch_jsonl(out, expected_rows=2)
    assert result["rows"] == 2
    assert result["kinds"] == {"branch_child": 2}
    assert result["forks"] == 1
    assert result["rows_with_tool_calls"] == 1
    assert result["prefix_incomplete"] == 0
    assert validate_branch_jsonl(pairs)["kinds"] == {"branch_pair": 1}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update(advantage=0.9), "advantage"),
        (lambda rows: rows[1]["prefix"].append(_say_msg()), "prefix"),
        (lambda rows: rows[0].update(messages=rows[0]["prefix"]), "messages"),
        (
            lambda rows: rows[0]["continuation"][2].update(tool_call_id="zz"),
            "tool_call_id",
        ),
        (lambda rows: rows[0].update(tools=[]), "not in tools"),
        (lambda rows: rows[0].update(kind="other"), "kind"),
    ],
)
def test_broken_child_rows_are_reported(tmp_path, mutate, message):
    out, _ = _export(tmp_path)

    def apply(rows):
        mutate(rows)
        for row in rows:
            if "messages" in row and row["messages"] is not row["prefix"]:
                row["messages"] = row["prefix"] + row["continuation"]

    _rewrite(out, apply if message != "messages" else mutate)
    with pytest.raises(ValueError, match=message):
        validate_branch_jsonl(out)


def _say_msg():
    return {"role": "assistant", "content": "extra"}


def test_broken_pair_rows_are_reported(tmp_path):
    _, pairs = _export(tmp_path)
    _rewrite(pairs, lambda rows: rows[0].update(chosen_reward=0.0))
    with pytest.raises(ValueError, match="chosen_reward"):
        validate_branch_jsonl(pairs)


def test_expected_rows(tmp_path):
    out, _ = _export(tmp_path)
    with pytest.raises(ValueError, match="expected 3"):
        validate_branch_jsonl(out, expected_rows=3)


def test_cli_validates_branch_tree(tmp_path):
    out, pairs = _export(tmp_path)
    runner = CliRunner()
    for path in (out, pairs):
        result = runner.invoke(
            app, ["train", "validate", str(path), "--format", "branch-tree"]
        )
        assert result.exit_code == 0, result.output
        assert "branch-tree" in result.output
    _rewrite(out, lambda rows: rows[0].update(advantage=0.9))
    result = runner.invoke(
        app, ["train", "validate", str(out), "--format", "branch-tree"]
    )
    assert result.exit_code == 1
    assert "advantage" in result.output


def test_cli_refuses_trl_options_for_branch_tree(tmp_path):
    out, _ = _export(tmp_path)
    result = CliRunner().invoke(
        app,
        ["train", "validate", str(out), "--format", "branch-tree", "--tokenizer", "x"],
    )
    assert result.exit_code != 0
    assert "branch-tree" in result.output

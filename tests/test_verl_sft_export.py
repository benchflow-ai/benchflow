"""verl SFT parquet from a recorded rollout.

verl's MultiTurnSFTDataset renders each row's messages and tools with the chat template and trains
every assistant message. Two things a plain parquet export of agent rollouts gets wrong, which
these tests guard:

- parquet gives nested values one schema per column, so a Bash call next to a Write call gains the
  other's argument keys as nulls and the template renders them: ``messages`` and ``tools`` are
  JSON strings, with tool arguments as objects inside;
- a call cut at the output cap, or one whose tool input was not a JSON object, would be trained:
  ``train_message_indices`` names the turns BenchFlow trains (TRL's per-call completions), and
  BenchFlowSFTDataset keeps the others at loss mask 0.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_verl_sft import (
    export_verl_sft_parquet,
    verl_dataset_path,
    verl_record,
)
from benchflow.trajectories.rl_conversations import build_conversations
from tests._rl_export_job import (
    BASH_ARGS,
    CONTEXT,
    TOOLS,
    TRAINED,
    WRITE_ARGS,
    write_job,
)

runner = CliRunner()


def _arguments(message):
    return message["tool_calls"][0]["function"]["arguments"]


def test_a_rollout_row_trains_only_the_calls_benchflow_trains(tmp_path):
    [conversation], _ = build_conversations(
        write_job(tmp_path), format_name="verl-sft", row_mode="rollout"
    )
    record = verl_record(conversation)
    messages = json.loads(record["messages"])

    assert [m["role"] for m in messages] == ["user"] + ["assistant", "tool"] * 4 + [
        "assistant"
    ]
    assert record["train_message_indices"] == TRAINED
    assert conversation.context == CONTEXT
    assert record["exchange_indices"] == [2, 3, 4]
    assert _arguments(messages[5]) == BASH_ARGS
    assert _arguments(messages[7]) == WRITE_ARGS
    trained = json.dumps([messages[i] for i in record["train_message_indices"]])
    assert "__unparsedToolInput" not in trained
    assert "_malformed_json_arguments" not in trained
    # The calls that are not trained stay as what happened, for the turns after them.
    assert "__unparsedToolInput" in _arguments(messages[3])
    assert json.loads(record["tools"]) == TOOLS
    assert record["reward"] == 1.0


def test_exchange_rows_train_their_own_call_only(tmp_path):
    conversations, _ = build_conversations(
        write_job(tmp_path), format_name="verl-sft", row_mode="exchange"
    )
    assert [c.trained for c in conversations] == [[5], [7], [9]]
    assert [c.calls[0]["exchange_index"] for c in conversations] == [2, 3, 4]
    assert conversations[2].context == {
        **CONTEXT,
        5: "trained_in_another_row",
        7: "trained_in_another_row",
    }


def test_parquet_keeps_each_argument_shape_exactly(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "train.parquet"
    stats = export_verl_sft_parquet(write_job(tmp_path), out, row_mode="exchange")
    rows = pq.read_table(out).to_pylist()

    assert stats.rows_written == len(rows) == 3
    bash = _arguments(json.loads(rows[0]["messages"])[-1])
    write = _arguments(json.loads(rows[1]["messages"])[-1])
    assert (bash, write) == (BASH_ARGS, WRITE_ARGS)
    assert [json.loads(row["tools"]) for row in rows] == [TOOLS] * 3
    assert [row["train_message_indices"] for row in rows] == [[5], [7], [9]]
    # What the JSON columns avoid: the same arguments as nested parquet values
    # take each other's keys as nulls.
    nested = pa.Table.from_pylist(
        [{"arguments": bash}, {"arguments": write}]
    ).to_pylist()
    assert nested[0]["arguments"] == {**BASH_ARGS, "file_path": None, "content": None}


def test_without_pyarrow_the_export_says_so_and_writes_nothing(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    out = tmp_path / "train.parquet"
    with pytest.raises(ValueError, match="pip install pyarrow"):
        export_verl_sft_parquet(write_job(tmp_path), out)
    assert not out.exists()


def test_cli_writes_parquet_and_prints_the_verl_flags(tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "train.parquet"
    manifest = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(write_job(tmp_path)),
            "--format",
            "verl-sft",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "3 assistant turn(s) trained, 2 kept as context" in output
    assert f"data.custom_cls.path={verl_dataset_path()}" in output
    assert "data.custom_cls.name=BenchFlowSFTDataset" in output
    stats = json.loads(manifest.read_text())
    assert (stats["format"], stats["row_mode"], stats["rows_written"]) == (
        "verl-sft",
        "rollout",
        1,
    )
    assert stats["context_turns"] == {"output_cap": 1, "unparsed_tool_input": 1}


def test_validate_refuses_the_format(tmp_path):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"")
    result = runner.invoke(
        app, ["train", "validate", str(path), "--format", "verl-sft"]
    )
    assert result.exit_code == 2
    assert "does not check --format verl-sft" in " ".join(result.output.split())


def test_dataset_class_imports_only_the_standard_library_and_verl():
    """verl loads the class by file path, where benchflow may not be installed."""
    tree = ast.parse(verl_dataset_path().read_text())
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots |= {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert roots - {"verl"} <= set(sys.stdlib_module_names)

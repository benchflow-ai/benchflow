"""SkyRL SFT rows from a recorded rollout.

SkyRL's tokenize_chat_example trains a row's last assistant message by default, or every one,
and cannot keep an assistant turn as context. These tests guard what each row mode promises:
exchange rows end on a call BenchFlow trains (never the call cut at the output cap or the one
with unparsed tool input), rollout rows end before the first turn BenchFlow does not train, and
tool arguments and tools are JSON strings, the shapes SkyRL parses after loading the file with
Hugging Face datasets.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_skyrl_sft import export_skyrl_sft_jsonl
from tests._rl_export_job import ANSWER, BASH_ARGS, TOOLS, WRITE_ARGS, write_job

runner = CliRunner()


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _arguments(message):
    arguments = message["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    return json.loads(arguments)


def test_exchange_rows_end_on_each_call_benchflow_trains(tmp_path):
    out = tmp_path / "skyrl" / "train.jsonl"
    stats = export_skyrl_sft_jsonl(write_job(tmp_path), out, row_mode="exchange")
    rows = _rows(out)

    assert [row["exchange_indices"] for row in rows] == [[2], [3], [4]]
    last = [row["messages"][-1] for row in rows]
    assert [_arguments(message) for message in last[:2]] == [BASH_ARGS, WRITE_ARGS]
    assert last[2] == {"role": "assistant", "content": ANSWER}
    assert all(json.loads(row["tools"]) == TOOLS for row in rows)
    # The unparsed call stays in later rows as what happened, never as the last turn.
    assert "__unparsedToolInput" in _arguments(rows[0]["messages"][3])
    assert (stats.rows_written, stats.trained_turns) == (3, 3)


def test_a_conversation_that_starts_with_an_untrained_turn_has_no_rollout_row(
    tmp_path,
):
    out = tmp_path / "skyrl" / "train.jsonl"
    stats = export_skyrl_sft_jsonl(write_job(tmp_path), out)

    assert out.read_text() == ""
    assert (stats.skipped_no_trained_turn, stats.trained_turns_left_out) == (1, 3)


def test_a_rollout_row_ends_before_the_first_untrained_turn(tmp_path):
    # Bash, Write, then a Write cut at the output cap, then the answer.
    jobs = write_job(tmp_path, order=(2, 3, 0, 4))
    out = tmp_path / "skyrl" / "train.jsonl"
    stats = export_skyrl_sft_jsonl(jobs, out)
    [row] = _rows(out)

    assert [m["role"] for m in row["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [_arguments(m) for m in row["messages"][1::2]] == [BASH_ARGS, WRITE_ARGS]
    assert row["exchange_indices"] == [0, 1]
    assert (stats.rows_cut, stats.trained_turns) == (1, 2)
    assert stats.trained_turns_left_out == 1


def test_cli_names_the_train_on_what_each_row_mode_is_for(tmp_path):
    jobs = write_job(tmp_path)
    for mode, setting in (
        ("exchange", "train_on_what=last_assistant_message"),
        ("rollout", "train_on_what=all_assistant_messages"),
    ):
        out = tmp_path / mode / "train.jsonl"
        result = runner.invoke(
            app,
            [
                "train",
                "convert",
                str(jobs),
                "--format",
                "skyrl-sft",
                "--row-mode",
                mode,
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        output = " ".join(result.output.split())
        assert f"train_datasets=[{out.parent}]" in output
        assert setting in output
    assert "--row-mode exchange writes them all" in output


def test_a_results_jsonl_input_is_refused(tmp_path):
    source = tmp_path / "results.jsonl"
    source.write_text("{}\n")
    out = tmp_path / "train.jsonl"
    result = runner.invoke(
        app,
        ["train", "convert", str(source), "--format", "skyrl-sft", "--out", str(out)],
    )
    assert result.exit_code == 1
    assert "reads a rollout or jobs directory" in " ".join(result.output.split())
    assert not out.exists()

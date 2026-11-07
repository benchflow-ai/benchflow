"""OpenEnv episode records from a recorded rollout.

Each line must construct OpenEnv's EpisodeRecord (its fields, nothing else) and be read back by
RolloutSerializer from <folder>/results.jsonl. The transcript is what happened: the main agent's
last model call, kept even when that call is one BenchFlow does not train; the calls cut at the
output cap or with tool input that was not a JSON object are marked as context, never as
trained turns. An unscored rollout is left out, since OpenEnv never counts a missing reward as 0.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_openenv_episodes import (
    EPISODE_FIELDS,
    export_openenv_episodes_jsonl,
)
from tests._rl_export_job import (
    ANSWER,
    BASH_ARGS,
    CONTEXT,
    TOOLS,
    TRAINED,
    UNPARSED,
    WRITE_ARGS,
    write_job,
)

runner = CliRunner()


def _episodes(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_an_episode_has_exactly_the_episode_record_fields(tmp_path):
    out = tmp_path / "episodes" / "results.jsonl"
    stats = export_openenv_episodes_jsonl(write_job(tmp_path), out)
    [episode] = _episodes(out)

    assert tuple(episode) == EPISODE_FIELDS
    assert episode["episode_id"] == "run/essay__abc123"
    assert (episode["reward"], episode["done"]) == (1.0, True)
    assert episode["messages"][-1] == {"role": "assistant", "content": ANSWER}
    calls = [m for m in episode["messages"] if m.get("tool_calls")]
    assert all(
        isinstance(m["tool_calls"][0]["function"]["arguments"], str) for m in calls
    )
    assert [m["name"] for m in episode["messages"] if m["role"] == "tool"] == [
        "Write",
        "Bash",
        "Bash",
        "Write",
    ]
    assert episode["extra"]["train_message_indices"] == TRAINED
    assert episode["extra"]["context_turns"] == {str(i): r for i, r in CONTEXT.items()}
    assert episode["extra"]["tools"] == TOOLS
    assert episode["verify_metrics"]["rewards"] == {"reward": 1.0}
    assert (stats.rows_written, stats.trained_turns) == (1, 3)


def test_tool_trace_pairs_each_call_with_its_result(tmp_path):
    out = tmp_path / "results.jsonl"
    export_openenv_episodes_jsonl(write_job(tmp_path), out)
    [episode] = _episodes(out)
    trace = episode["tool_trace"]

    assert [entry["tool_name"] for entry in trace] == ["Write", "Bash", "Bash", "Write"]
    assert trace[1]["arguments"] == UNPARSED  # what happened, marked as context
    assert [entry["arguments"] for entry in trace[2:]] == [BASH_ARGS, WRITE_ARGS]
    assert trace[2]["result"] == {
        "data": "ok",
        "done": False,
        "metadata": {"tool_call_id": "call_2"},
        "error": None,
    }


def test_a_last_call_benchflow_does_not_train_stays_in_the_transcript(tmp_path):
    # Bash, Write, then a Write cut at the output cap with nothing after it.
    out = tmp_path / "results.jsonl"
    export_openenv_episodes_jsonl(write_job(tmp_path, order=(2, 3, 0)), out)
    [episode] = _episodes(out)

    assert episode["messages"][-1]["tool_calls"][0]["id"] == "call_0"
    assert episode["extra"]["train_message_indices"] == [1, 3]
    assert episode["extra"]["context_turns"] == {"5": "output_cap"}


def test_an_unscored_rollout_is_left_out(tmp_path):
    write_job(tmp_path, reward=None, trial="essay__unscored")
    jobs = write_job(tmp_path, trial="essay__scored")
    out = tmp_path / "results.jsonl"
    stats = export_openenv_episodes_jsonl(jobs, out)

    assert [e["episode_id"] for e in _episodes(out)] == ["run/essay__scored"]
    assert stats.skipped_unscored == 1


def test_cli_points_at_the_serializer_and_refuses_exchange_rows(tmp_path):
    jobs = write_job(tmp_path)
    out = tmp_path / "episodes" / "results.jsonl"
    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(jobs),
            "--format",
            "openenv-episodes",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert f"RolloutSerializer({out.parent})" in output
    assert "3 assistant turn(s) listed in extra.train_message_indices" in output

    other = tmp_path / "x" / "results.jsonl"
    refused = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(jobs),
            "--format",
            "openenv-episodes",
            "--row-mode",
            "exchange",
            "--out",
            str(other),
        ],
    )
    assert refused.exit_code == 2
    assert "drop --row-mode exchange" in " ".join(refused.output.split())
    assert not other.exists()

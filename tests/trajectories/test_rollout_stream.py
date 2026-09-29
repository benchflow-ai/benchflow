"""Streaming finished rollouts of a running job (``benchflow.rollout-stream.v1``).

``stream_rollouts`` / ``astream_rollouts`` and ``bench train stream`` yield
each rollout as its ``result.json`` appears, with reward, group id and the
gateway's token ids and logprobs, so a trainer can consume a job while it
runs. Before this, token data could only be read after the job, one
``llm_trajectory.jsonl`` at a time.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.trajectories import rollout_stream as rs
from benchflow.trajectories.token_capture import TOKEN_CAPTURE_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]


def _capture(prompt: list[int], sampled: list[int], provider: str = "vllm") -> dict:
    return {
        "metadata": {
            "token_capture": {
                "schema_version": TOKEN_CAPTURE_SCHEMA_VERSION,
                "wire": "openai-chat",
                "provider": provider,
                "requested": {
                    "logprobs": True,
                    "top_logprobs": None,
                    "token_ids": True,
                },
                "prompt_token_ids": prompt,
                "completions": [
                    {
                        "index": 0,
                        "token_ids": sampled,
                        "tokens": [str(t) for t in sampled],
                        "logprobs": [-0.5] * len(sampled),
                        "top_logprobs": None,
                    }
                ],
                "unavailable": {},
            }
        }
    }


# Two calls, token-in/token-out: the second prompt is the first prompt, the
# sampled tokens, then a tool result [7, 8].
TITO_CALLS = [_capture([1, 2, 3], [10, 11]), _capture([1, 2, 3, 10, 11, 7, 8], [12])]


def write_rollout(
    job: Path,
    name: str,
    *,
    task: str = "hello",
    reward: float | None = 1.0,
    calls: list[dict] | None = None,
    error: str | None = None,
    model: str = "vllm/policy",
) -> Path:
    root = job / name
    (root / "trajectory").mkdir(parents=True, exist_ok=True)
    if calls is not None:
        (root / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in calls)
        )
    result = {
        "task_name": task,
        "rollout_name": name,
        "agent": "claude-agent-acp",
        "model": model,
        "rewards": None if reward is None else {"reward": reward},
        "error": error,
        "verifier_error": None,
        "finished_at": "2026-01-01T10:00:00",
    }
    (root / "result.json").write_text(json.dumps(result))
    return root


def start_job(job: Path, pid: int | None = None) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "evaluation.json").write_text("{}")
    (job / rs.JOB_LOCK).write_text(
        json.dumps(
            {"pid": pid or __import__("os").getpid(), "host": socket.gethostname()}
        )
    )


def finish_job(job: Path) -> None:
    (job / "summary.json").write_text("{}")
    (job / rs.JOB_LOCK).unlink()


def test_finished_job_streams_every_rollout_with_reward_group_and_tokens(tmp_path):
    job = tmp_path / "job"
    start_job(job)
    write_rollout(job, "hello__a", calls=TITO_CALLS)
    write_rollout(job, "hello__b", reward=None, error="agent timed out", calls=[])
    write_rollout(job, "other__c", task="other", reward=0.0)
    finish_job(job)

    records = {r.rollout: r for r in bf.stream_rollouts(job, follow=False)}

    assert set(records) == {"hello__a", "hello__b", "other__c"}
    a = records["hello__a"]
    assert a.job == "job"
    assert a.reward == 1.0 and a.scored and a.outcome == "passed"
    assert a.group_id == "task=hello|agent=claude-agent-acp|model=vllm/policy"
    assert a.token_capture["training_grade"] is True
    assert a.token_capture["path"] == "vllm"
    assert [c["prompt_token_ids"] for c in a.calls] == [
        [1, 2, 3],
        [1, 2, 3, 10, 11, 7, 8],
    ]
    assert a.sequences == [
        {
            "thread": 0,
            "kind": "chat",
            "calls": [0, 1],
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11, 7, 8, 12],
            "completion_mask": [1, 1, 0, 0, 1],
            "completion_logprobs": [-0.5, -0.5, 0.0, 0.0, -0.5],
        }
    ]
    b = records["hello__b"]
    assert b.reward is None and not b.scored and b.outcome == "errored"
    assert b.group_id == a.group_id
    assert b.token_capture["status"] == "capture_off" and b.sequences == []
    c = records["other__c"]
    assert c.token_capture["status"] == "no_gateway_capture"
    assert c.calls == [] and c.sequences == [] and c.outcome == "failed"


def test_prefix_break_gives_calls_but_no_merged_sequence(tmp_path):
    job = tmp_path / "job"
    calls = [_capture([1, 2, 3], [10]), _capture([9, 9, 9, 10], [11])]
    write_rollout(job, "t__a", calls=calls)

    [record] = rs.stream_rollouts(job, follow=False)

    assert record.token_capture["training_grade"] is False
    assert record.token_capture["prefix"]["breaks"] == [1]
    assert len(record.calls) == 2
    assert record.sequences == []


def test_merged_sequence_refuses_mismatched_logprobs():
    bad = _capture([1], [5, 6])
    bad["metadata"]["token_capture"]["completions"][0]["logprobs"] = [-1.0]
    calls = rs._calls([bad])
    assert rs.merged_sequence(calls) is None
    assert rs.merged_sequence([]) is None


def test_running_job_yields_rollouts_as_they_finish(tmp_path):
    """A trainer gets the first rollout while the job is still running."""
    job = tmp_path / "job"
    start_job(job)
    got: list[tuple[str, bool]] = []

    def run_job() -> None:
        for name in ("t__1", "t__2", "t__3"):
            time.sleep(0.3)
            write_rollout(job, name, calls=TITO_CALLS)
        time.sleep(0.3)
        finish_job(job)

    writer = threading.Thread(target=run_job)
    writer.start()
    for record in rs.stream_rollouts(job, poll_interval=0.05, timeout=30):
        got.append((record.rollout, (job / "summary.json").exists()))
    writer.join()

    assert [name for name, _ in got] == ["t__1", "t__2", "t__3"]
    assert got[0][1] is False  # streamed before the job finished


def test_stream_waits_for_a_job_folder_that_does_not_exist_yet(tmp_path):
    jobs = tmp_path / "jobs"

    def run_job() -> None:
        time.sleep(0.3)
        job = jobs / "2026-01-01__10-00-00"
        start_job(job)
        write_rollout(job, "t__1")
        finish_job(job)

    writer = threading.Thread(target=run_job)
    writer.start()
    records = list(rs.stream_rollouts(jobs, poll_interval=0.05, timeout=30))
    writer.join()

    assert [r.rollout for r in records] == ["t__1"]
    assert records[0].job == "2026-01-01__10-00-00"


def test_missing_job_without_follow_raises(tmp_path):
    with pytest.raises(rs.JobNotFound):
        list(rs.stream_rollouts(tmp_path / "nope", follow=False))


def test_dead_job_process_ends_the_stream_after_what_finished(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    dead_pid = int(proc.stdout)
    job = tmp_path / "job"
    start_job(job, pid=dead_pid)
    write_rollout(job, "t__1")
    got = []

    with pytest.raises(rs.JobProcessGone, match=str(dead_pid)):
        for record in rs.stream_rollouts(job, poll_interval=0.05, timeout=10):
            got.append(record.rollout)
    assert got == ["t__1"]


def test_timeout_ends_the_stream(tmp_path):
    job = tmp_path / "job"
    start_job(job)
    with pytest.raises(rs.StreamTimeout, match="0 rollouts streamed"):
        list(rs.stream_rollouts(job, poll_interval=0.05, timeout=0.2))


def test_unreadable_result_is_skipped_with_one_warning(tmp_path):
    job = tmp_path / "job"
    write_rollout(job, "t__good")
    (job / "t__bad").mkdir()
    (job / "t__bad" / "result.json").write_text("{not json")
    warnings: list[str] = []

    records = list(rs.stream_rollouts(job, follow=False, on_warning=warnings.append))

    assert [r.rollout for r in records] == ["t__good"]
    assert len(warnings) == 1 and "t__bad" in warnings[0]


def test_group_size_holds_records_until_the_group_fills(tmp_path):
    job = tmp_path / "job"
    for name, reward in (("t__1", 1.0), ("t__2", 0.0), ("t__3", 1.0)):
        write_rollout(job, name, reward=reward)
    write_rollout(job, "u__1", task="u", reward=1.0)

    records = list(rs.stream_rollouts(job, follow=False, group_size=3))

    by_name = {r.rollout: r for r in records}
    assert [r.rollout for r in records][-1] == "u__1"  # short group flushed last
    assert by_name["u__1"].group_complete is False
    assert by_name["u__1"].advantage is None
    assert by_name["t__1"].group_complete is True
    assert by_name["t__1"].advantage == pytest.approx(0.5773, abs=1e-3)
    assert by_name["t__2"].advantage == pytest.approx(-1.1546, abs=1e-3)


def test_group_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="group_size"):
        list(rs.stream_rollouts(tmp_path, follow=False, group_size=0))


def test_records_match_the_committed_schema(tmp_path):
    job = tmp_path / "job"
    write_rollout(job, "hello__a", calls=TITO_CALLS)
    write_rollout(job, "hello__b", reward=None, error="boom", calls=[])
    write_rollout(job, "other__c", task="other", reward=0.0)
    committed = json.loads(
        (REPO / "docs/reference/schemas" / rs.SCHEMA_FILE).read_text()
    )
    assert committed == rs.SCHEMA, (
        "regenerate with python -m benchflow.trajectories.rollout_stream "
        "docs/reference/schemas"
    )
    validator = jsonschema.Draft202012Validator(rs.SCHEMA)
    for record in rs.stream_rollouts(job, follow=False, group_size=2):
        document = json.loads(record.to_json())
        assert list(validator.iter_errors(document)) == []


def test_async_stream_yields_the_same_records(tmp_path):
    job = tmp_path / "job"
    start_job(job)
    write_rollout(job, "t__1", calls=TITO_CALLS)
    finish_job(job)

    async def collect() -> list[str]:
        return [r.rollout async for r in bf.astream_rollouts(job, poll_interval=0.01)]

    assert asyncio.run(collect()) == ["t__1"]


def test_evaluation_exposes_its_job_dir_before_running(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    evaluation = bf.Evaluation(
        tasks,
        tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle"),
        job_name="my-job",
    )
    assert evaluation.job_dir == tmp_path / "jobs" / "my-job"


# --- CLI ------------------------------------------------------------------


def _cli(*args: str):
    from benchflow.cli.main import app

    return CliRunner().invoke(app, ["train", "stream", *args])


def test_cli_prints_one_json_line_per_rollout(tmp_path):
    job = tmp_path / "job"
    start_job(job)
    write_rollout(job, "t__1", calls=TITO_CALLS)
    write_rollout(job, "t__2", reward=0.0)
    finish_job(job)

    result = _cli(str(job), "--format", "jsonl", "--poll-interval", "0.05")

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert [line["rollout"] for line in lines] == ["t__1", "t__2"]
    assert lines[0]["schema_version"] == rs.ROLLOUT_STREAM_SCHEMA_VERSION
    assert lines[0]["sequences"][0]["completion_mask"] == [1, 1, 0, 0, 1]


@pytest.mark.parametrize(
    ("setup", "args", "code"),
    [
        ("missing", ["--no-follow"], 2),
        ("running", ["--timeout", "0.2"], 3),
        ("dead", [], 1),
    ],
)
def test_cli_exit_codes(tmp_path, setup, args, code):
    job = tmp_path / "job"
    if setup == "running":
        start_job(job)
    elif setup == "dead":
        proc = subprocess.run(
            [sys.executable, "-c", "import os; print(os.getpid())"],
            capture_output=True,
            text=True,
            check=True,
        )
        start_job(job, pid=int(proc.stdout))

    result = _cli(str(job), "--poll-interval", "0.05", *args)

    assert result.exit_code == code, result.output


def test_cli_rejects_unknown_format(tmp_path):
    result = _cli(str(tmp_path), "--format", "csv")
    assert result.exit_code == 2


def test_agent_loop_and_helper_call_get_one_sequence_each(tmp_path):
    """A real agent run: the tool loop is one token stream, the tool-less
    helper call another (``conversation_threads``)."""
    tools = {"tools": [{"type": "function", "function": {"name": "Bash"}}]}
    loop = [dict(c, request={"body": tools}) for c in TITO_CALLS]
    helper = _capture([40, 41], [42])
    write_rollout(tmp_path / "job", "t__a", calls=[loop[0], loop[1], helper])

    [record] = rs.stream_rollouts(tmp_path / "job", follow=False)

    assert record.training_grade
    assert [(s["kind"], s["calls"]) for s in record.sequences] == [
        ("agent", [0, 1]),
        ("helper", [2]),
    ]
    assert record.sequences[1]["completion_ids"] == [42]

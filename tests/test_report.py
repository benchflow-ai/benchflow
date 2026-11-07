"""The report every job writes beside summary.json (benchflow.report)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchflow.model_settings import vendor_setting
from benchflow.report import REPORT, build_report, wilson, write_report


def _trial(
    job: Path,
    task: str,
    n: int,
    reward: float | None,
    *,
    sampling: dict | None = None,
    seconds: float = 60.0,
) -> None:
    trial = job / f"{task}__{n:04d}"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "rollout_name": trial.name,
                "rewards": None if reward is None else {"reward": reward},
                "error": None if reward is not None else "agent timed out",
                "agent_result": {"total_tokens": 1200, "cost_usd": 0.01},
                "started_at": "2026-11-04 12:00:00",
                "finished_at": f"2026-11-04 12:{int(seconds) // 60:02d}:{int(seconds) % 60:02d}",
            }
        )
    )
    if sampling is not None:
        (trial / "trajectory").mkdir()
        record = {"request": {"body": {"model": "qwen", "messages": [], **sampling}}}
        (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
            json.dumps(record) + "\n" + json.dumps(record) + "\n"
        )


def test_wilson_intervals_match_the_textbook_values():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    low, high = wilson(0, 10)
    assert low == 0.0 and round(high, 3) == 0.278
    low, high = wilson(10, 10)
    assert round(low, 3) == 0.722 and high == 1.0


def test_the_report_shows_each_task_s_tries_and_the_sampling_sent(tmp_path):
    job = tmp_path / "job"
    off = {"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    _trial(job, "dock-a", 1, 1.0, sampling=off, seconds=90)
    _trial(job, "dock-a", 2, 0.0, sampling=off, seconds=30)
    _trial(job, "dock-b", 1, None)
    (job / "run-card.json").write_text(
        json.dumps(
            {
                "agent": {"name": "mini-swe-agent"},
                "model": {
                    "id": "hosted_vllm/Qwen/Qwen3.8-27B",
                    "vendor_setting": vendor_setting("Qwen/Qwen3.8-27B").as_record(),
                },
            }
        )
    )
    text = build_report([job], "baseline")
    assert text.startswith("# Report: baseline\n")
    # The headline's interval is the scoring rule's, over tasks (here the
    # design effect is below 1, so it is Wilson's over the 3 trials).
    assert (
        "3 trials of 2 tasks, mini-swe-agent on hosted_vllm/Qwen/Qwen3.8-27B: 1 passed "
        "(33.3%; 95% interval 6.1% to 79.2%, clustered by task), mean reward "
        "0.33 (a trial with no score counts 0), 1 with no score."
    ) in text
    assert "| dock-a | 2 | 1 | 50% | 9% to 91% | 1, 0 | █████░░░░░ |" in text
    assert "| dock-b | 1 | 0 | 0% | 0% to 79% | no score | ░░░░░░░░░░ |" in text
    assert (
        "Tokens 3,600; cost $0.03; trial time 30 to 90 s, 3 trial-minutes in all."
        in text
    )
    # The requests carried temperature 1.0 with thinking off; the vendor line
    # beside it gives Qwen's setting for thinking off.
    assert "temperature 1.0 (4)" in text
    assert 'chat_template_kwargs {"enable_thinking": false} (4)' in text
    # top_p was never sent: the server's default decided it.
    assert "; no top_p sent, so the model server's default applies." in text
    assert "The run card's Vendor line: Qwen recommends for Qwen3.8-27B" in text


def test_a_retried_task_is_one_trial_and_a_review_pass_passes(tmp_path):
    """The report read every attempt folder (an error and its
    retry became two trials) and passed only reward 1, against summary.json."""
    from tests.test_python_sdk_load_job import _trial as sdk_trial

    job = tmp_path / "job"
    job.mkdir()
    (job / "evaluation.json").write_text(
        json.dumps({"planned_tasks": ["t", "r", "gone"]})
    )
    sdk_trial(
        job,
        "t",
        reward=None,
        error="sandbox lost",
        error_category="infra",
        suffix="aaaa0001",
    )
    sdk_trial(job, "t", reward=1.0, suffix="aaaa0002")
    review = sdk_trial(job, "r", reward=0.626, suffix="bbbb0001")
    result = json.loads((review / "result.json").read_text())
    result["scoring"] = {
        "status": "complete",
        "passed": True,
        "tests_pass": True,
        "all_blockers_pass": True,
        "verifier_reward": 1.0,
        "rubric_reward": 0.626,
        "reviewer_run": "review-r__bbbb0001",
    }
    (review / "result.json").write_text(json.dumps(result))
    text = build_report([job], "j")
    assert "2 trials of 2 tasks" in text
    assert "| t | 1 | 1 | 100% |" in text
    import benchflow as bf

    summary_passed = sum(t.passed for t in bf.load_job(job).trials)
    assert f": {summary_passed} passed" in text
    assert (
        "Not in these numbers: 1 planned task with no finished trial (the job stopped "
        "or was interrupted): gone."
    ) in text


def test_an_oracle_job_says_no_model_tokens_were_used(tmp_path):
    from tests.test_python_sdk_load_job import _trial as sdk_trial

    job = tmp_path / "oracle"
    sdk_trial(job, "t", agent="oracle", model=None, cost=None)
    result = json.loads((job / "t__abcd1234" / "result.json").read_text())
    result["agent_result"] = {"total_tokens": 0, "cost_usd": None}
    (job / "t__abcd1234" / "result.json").write_text(json.dumps(result))
    assert "No model tokens were used" in build_report([job], "oracle")


def test_a_job_with_no_trials_or_no_proxy_says_so(tmp_path):
    assert "No finished trials." in build_report([tmp_path / "empty"], "x")
    job = tmp_path / "job"
    _trial(job, "t", 1, 1.0)
    text = build_report([job], "x")
    assert "not recorded (no requests through BenchFlow's proxy)" in text
    assert "Vendor line" not in text


def test_a_report_that_cannot_be_written_never_fails_the_run(tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("")
    with caplog.at_level(logging.WARNING, logger="benchflow.report"):
        assert write_report([tmp_path], blocked, "x") is None
    assert "Could not write" in caplog.text
    out = tmp_path / "out"
    out.mkdir()
    assert write_report([tmp_path], out, "x") == out / REPORT


def test_a_trials_run_writes_one_report_over_its_repeats(tmp_path, monkeypatch):
    from tests.test_eval_runner_results_dx import _fake_rollouts, _invoke, _write_task

    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        _write_task(tasks / name)
    _fake_rollouts(monkeypatch)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--trials", "2", "--jobs-dir", str(jobs),
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    assert (jobs / "oracle" / REPORT).read_text().startswith("# Report: oracle")
    assert len(list((jobs / "oracle").glob(f"trial-0*/*/{REPORT}"))) == 2
    assert f"Report: {jobs / 'oracle' / REPORT}" in run.output

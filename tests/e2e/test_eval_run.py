"""bench eval run / bf.run* / Evaluation: single, batch, controls, pre-run checks, resume."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import benchflow as bf
from tests.e2e import harness as h


def test_cli_batch_outputs(batch_job: Path, jobs_root: Path, ledger: h.Ledger):
    """A 3-task oracle batch writes one trial per task with honest rewards."""
    summary = h.read_json(batch_job / "summary.json")
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["verifier_errored"] == 1
    rewards = {
        t.name.split("__")[0]: h.read_json(t / "result.json").get("rewards")
        for t in h.trial_dirs(batch_job)
    }
    assert rewards["e2e-pass"] == {"reward": 1.0}
    assert rewards["e2e-fail"] == {"reward": 0.0}
    # A verifier that crashed is unscored, never 0.
    assert not rewards["e2e-verifier-error"]
    rows = {
        r["info"]["task_name"]: r for r in h.read_jsonl(batch_job / "results.jsonl")
    }
    assert rows["e2e-verifier-error"]["reward"] is None
    run_summary = h.read_json(jobs_root / "batch-oracle.run-summary.json")
    h.validate(run_summary, "benchflow-run-summary.v1.schema.json")
    assert run_summary["exit_code"] == 1
    assert run_summary["gate"]["fail_on"] == ["verifier-error"]
    assert run_summary["budget"]["caps"]["max_sandbox_seconds"] == h.cap_seconds()
    # --freeze-workspace wrote evidence with a manifest for every trial.
    for trial in h.trial_dirs(batch_job):
        assert (trial / "evidence").is_dir(), trial
    ledger.record(
        "eval run batch outputs", surface="CLI", seconds=0, note="reads batch job"
    )


def test_cli_single_task_and_nop_control(
    sandbox: str, batch_tasks: Path, jobs_root: Path, ledger: h.Ledger
):
    """One task directory runs on its own; the nop agent is a control run scored 0."""
    task = batch_tasks / "e2e-pass"
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "nop", "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", "single-nop",
        "--max-sandbox-seconds", str(h.cap_seconds()), "--fail-under", "0.5", "--quiet",
        log=jobs_root / "single-nop.log",
    )  # fmt: skip
    ledger.record(
        "eval run single task, nop control, --fail-under",
        surface="CLI",
        seconds=run.seconds,
        job_dir=jobs_root / "single-nop",
    )
    # nop does nothing, so the pass rate 0 < 0.5 trips --fail-under.
    h.assert_exit(run, 1)
    trial = h.trial_of(jobs_root / "single-nop", "e2e-pass")
    assert h.read_json(trial / "result.json")["rewards"] == {"reward": 0.0}
    _, doc = h.bench_json("eval", "inspect", jobs_root / "single-nop", "--json")
    h.validate(doc, "benchflow-job.v1.schema.json")
    # Control runs are left out of the headline denominators.
    assert doc["denominators"]["attempted"] == 0, doc["denominators"]


@pytest.mark.parametrize(
    "args,needle",
    [
        (["--agent", "oracel"], "oracel"),
        (["--agent", "oracle", "--sandbox", "dayton"], "dayton"),
    ],
    ids=["misspelt-agent", "unknown-sandbox"],
)
def test_cli_prerun_checks_stop_before_a_job(
    sandbox: str, batch_tasks: Path, tmp_path: Path, args, needle, ledger: h.Ledger
):
    started = time.monotonic()
    run = h.bench(
        "eval", "run", "--tasks-dir", batch_tasks / "e2e-pass",
        *(["--sandbox", sandbox] if "--sandbox" not in args else []),
        *args, "--jobs-dir", tmp_path / "jobs",
    )  # fmt: skip
    ledger.record(
        f"pre-run check: {needle}", surface="CLI", seconds=time.monotonic() - started
    )
    assert run.returncode != 0, run.tail()
    assert needle in run.output
    assert not list((tmp_path / "jobs").glob("*/*/result.json"))


def test_cli_missing_tasks_dir(tmp_path: Path, sandbox: str):
    run = h.bench(
        "eval", "run", "--tasks-dir", tmp_path / "nope", "--agent", "oracle",
        "--sandbox", sandbox, "--jobs-dir", tmp_path / "jobs",
    )  # fmt: skip
    assert run.returncode != 0
    assert "nope" in run.output


def test_python_prerun_checks(sandbox: str, batch_tasks: Path, tmp_path: Path):
    """bf.run_sync refuses a misspelt agent and a missing task before a sandbox starts."""
    with pytest.raises(Exception, match="oracel"):
        bf.run_sync(
            bf.RolloutConfig(
                task_path=batch_tasks / "e2e-pass",
                agent="oracel",
                environment=sandbox,
                jobs_dir=tmp_path,
            )
        )
    with pytest.raises(Exception, match="nope"):
        bf.run_sync(
            bf.RolloutConfig(
                task_path=tmp_path / "nope",
                agent="oracle",
                environment=sandbox,
                jobs_dir=tmp_path,
            )
        )


def test_python_run_sync_and_run_batch(
    sandbox: str, batch_tasks: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    one = bf.run_sync(
        bf.RolloutConfig(
            task_path=batch_tasks / "e2e-pass",
            agent="oracle",
            environment=sandbox,
            jobs_dir=jobs_root,
            job_name="py-run-sync",
        )
    )
    assert one.reward == 1.0 and one.passed
    loaded = bf.RolloutResult.load(one.rollout_dir)
    assert loaded.reward == 1.0
    ledger.record(
        "bf.run_sync oracle", surface="Python", seconds=time.monotonic() - started
    )

    started = time.monotonic()
    configs = [
        bf.RolloutConfig(
            task_path=batch_tasks / name,
            agent="oracle",
            environment=sandbox,
            jobs_dir=jobs_root,
            job_name="py-run-batch",
        )
        for name in ("e2e-pass", "e2e-fail")
    ]
    results = bf.run_batch(configs, concurrency=2)
    ledger.record(
        "bf.run_batch oracle x2", surface="Python", seconds=time.monotonic() - started
    )
    assert len(results) == 2
    assert results.n_passed == 1
    assert results.mean_reward == pytest.approx(0.5)
    records = results.to_records()
    assert {r["reward"] for r in records} == {0.0, 1.0}


def test_python_evaluation_stream_and_resume(
    sandbox: str, batch_tasks: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    config = bf.EvaluationConfig(
        agent="oracle",
        environment=sandbox,
        concurrency=2,
        include_tasks={"e2e-pass", "e2e-fail"},
        budget=bf.Budget(max_sandbox_seconds=h.cap_seconds()),
    )
    ev = bf.Evaluation(
        tasks_dir=batch_tasks,
        jobs_dir=jobs_root,
        config=config,
        job_name="py-evaluation",
    )
    result = ev.run_sync()
    ledger.record(
        "Evaluation.run_sync oracle x2 with Budget",
        surface="Python",
        seconds=time.monotonic() - started,
        job_dir=result.job_dir,
    )
    assert result.total == 2 and result.passed == 1
    assert result.budget is not None
    job_dir = Path(result.job_dir)
    assert (job_dir / "evaluation.json").is_file()
    # Resume of a finished job reruns nothing.
    started = time.monotonic()
    again = bf.Evaluation.resume(job_dir).run_sync()
    ledger.record(
        "Evaluation.resume finished job",
        surface="Python",
        seconds=time.monotonic() - started,
    )
    assert again.ran == 0 and again.reused == 2
    # CLI resume of the same folder agrees.
    run = h.bench("eval", "resume", job_dir, log=jobs_root / "py-evaluation.resume.log")
    h.assert_exit(run, 0, 1)
    assert "resum" in run.output.lower()

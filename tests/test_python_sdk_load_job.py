"""Read finished jobs and trials into typed objects, and compare two jobs.

A Python user could load one rollout (``RolloutResult.load``) but not a job,
its verifier output, branch lineage or checkpoints, and had no fair way to
compare two jobs. ``bf.load_trial`` / ``bf.load_job`` read current and older
layouts; ``Job.denominators()`` and ``bf.compare`` count this way: attempted, scored, assessment errors and unscored are separate,
pass rates are given over scored and over attempted runs, and control runs
(oracle, empty/nop) are left out of agent denominators unless asked for.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import benchflow as bf


def _trial(
    job: Path,
    task: str,
    *,
    reward: float | None = 1.0,
    agent: str = "claude-agent-acp",
    model: str | None = "claude-haiku-4-5",
    error: str | None = None,
    error_category: str | None = None,
    verifier_error: str | None = None,
    suffix: str = "abcd1234",
    legacy: bool = False,
    cost: float | None = 0.01,
    mtime: float | None = None,
) -> Path:
    d = job / f"{task}__{suffix}"
    d.mkdir(parents=True)
    if legacy:
        # An older layout: top-level reward, trial_name, trajectory under agent/.
        result = {
            "task_name": task,
            "trial_name": d.name,
            "reward": reward,
            "agent": agent,
            "model": model,
            "error": error,
        }
        (d / "agent").mkdir()
        (d / "agent" / "acp_trajectory.jsonl").write_text(
            json.dumps({"type": "agent_message", "text": "old"}) + "\n"
        )
    else:
        result = {
            "task_name": task,
            "rollout_name": d.name,
            "rewards": None if reward is None else {"reward": reward},
            "agent": agent,
            "model": model,
            "error": error,
            "error_category": error_category,
            "verifier_error": verifier_error,
            "n_tool_calls": 2,
            "agent_result": {"total_tokens": 100, "cost_usd": cost},
            "started_at": "2026-01-01 12:00:00",
            "finished_at": "2026-01-01 12:01:30",
        }
        (d / "trajectory").mkdir()
        (d / "trajectory" / "acp_trajectory.jsonl").write_text(
            json.dumps({"type": "tool_call", "title": "Write hello.txt"}) + "\n"
        )
        (d / "verifier").mkdir()
        (d / "verifier" / "reward.txt").write_text(f"{reward}\n")
        (d / "verifier" / "test-stdout.txt").write_text("1 passed\n")
        (d / "config.json").write_text(json.dumps({"agent": agent, "timeout_sec": 300}))
        (d / "timing.json").write_text(json.dumps({"agent": 60.0, "total": 90.0}))
    (d / "result.json").write_text(json.dumps(result))
    if mtime is not None:
        os.utime(d / "result.json", (mtime, mtime))
    return d


def _tree(trial: Path) -> None:
    fork = "f1"
    children = []
    for i, (label, reward) in enumerate([("baseline", 1.0), ("hint", 0.0)]):
        node = trial / "branches" / fork / "children" / f"n{i + 6}"
        node.mkdir(parents=True)
        (node / "observation.json").write_text(json.dumps({"reward": reward}))
        children.append(
            {
                "index": i,
                "node_id": f"n{i + 6}",
                "status": "scored",
                "reward": reward,
                "reward_source": "verifier",
                "intervention": {"label": label, "requested": "x"},
                "artifacts": {
                    "status": "available",
                    "path": f"branches/{fork}/children/n{i + 6}",
                },
            }
        )
    (trial / "tree.json").write_text(
        json.dumps(
            {
                "kind": "benchflow-branch-tree",
                "schema_version": 1,
                "nodes": [],
                "forks": [
                    {
                        "id": fork,
                        "status": "completed",
                        "value": 0.5,
                        "parent_restore": "restored",
                        "snapshot": {"retention": "deleted"},
                        "children": children,
                    }
                ],
            }
        )
    )


def test_load_trial_reads_everything_the_viewer_shows(tmp_path: Path) -> None:
    d = _trial(tmp_path / "job", "hello")
    _tree(d)
    t = bf.load_trial(d)
    assert isinstance(t, bf.Trial) and isinstance(t.result, bf.RolloutResult)
    assert t.task_name == "hello" and t.reward == 1.0 and t.passed
    assert t.cost_usd == 0.01 and t.total_tokens == 100
    assert t.execution == "completed" and t.assessment == "scored"
    assert t.control is None
    assert t.verifier.reward_text == "1.0" and "1 passed" in t.verifier.stdout
    assert t.config["timeout_sec"] == 300 and t.timing["total"] == 90.0
    assert [e["type"] for e in t.trajectory] == ["tool_call"]
    (fork,) = t.forks
    assert fork.value == 0.5 and fork.status == "completed"
    assert [(c.label, c.reward) for c in fork.children] == [
        ("baseline", 1.0),
        ("hint", 0.0),
    ]
    assert (d / fork.children[0].path).is_dir()
    assert bf.load_trial(d / "result.json").task_name == "hello"


def test_load_trial_reads_an_older_layout(tmp_path: Path) -> None:
    d = _trial(tmp_path / "job", "old-task", legacy=True, reward=0.0)
    t = bf.load_trial(d)
    assert t.reward == 0.0 and t.assessment == "scored" and not t.passed
    assert t.result.rollout_name == d.name
    assert t.trajectory == [{"type": "agent_message", "text": "old"}]
    assert t.verifier.reward_text is None and t.forks == []


def test_load_trial_names_what_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"result\.json"):
        bf.load_trial(tmp_path)


def test_execution_and_assessment_states(tmp_path: Path) -> None:
    job = tmp_path / "job"
    timed = bf.load_trial(
        _trial(job, "a", reward=0.0, error="t", error_category="timeout")
    )
    crashed = bf.load_trial(
        _trial(job, "b", reward=None, error="boom", error_category="other")
    )
    verr = bf.load_trial(_trial(job, "c", reward=None, verifier_error="pytest crashed"))
    assert (timed.execution, timed.assessment) == ("timed_out", "scored")
    assert (crashed.execution, crashed.assessment) == ("errored", "unscored")
    assert (verr.execution, verr.assessment) == ("completed", "error")


def _job(tmp_path: Path, name: str, rewards: dict[str, float | None]) -> Path:
    job = tmp_path / name
    for task, reward in rewards.items():
        _trial(job, task, reward=reward, error=None if reward is not None else "boom")
    _trial(job, "zz-oracle", agent="oracle", model=None, suffix="0000oracl")
    _trial(job, "zz-empty", agent="nop", model=None, reward=0.0, suffix="0000empty")
    (job / "summary.json").write_text(json.dumps({"total": len(rewards) + 2}))
    return job


def test_load_job_counts_like_the_viewer(tmp_path: Path) -> None:
    job = bf.load_job(
        _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0, "t3": 0.5, "t4": None})
    )
    assert isinstance(job, bf.Job) and job.summary == {"total": 6}
    assert len(job.trials) == 6 and len(job.controls()) == 2
    d = job.denominators()
    assert (d.attempted, d.scored, d.unscored, d.passed) == (4, 3, 1, 1)
    assert d.controls_excluded == 2
    assert d.pass_rate_scored == pytest.approx(1 / 3)
    assert d.pass_rate_attempted == pytest.approx(1 / 4)
    assert d.mean_reward == pytest.approx(0.5)
    with_controls = job.denominators(include_controls=True)
    assert with_controls.attempted == 6 and with_controls.controls_excluded == 0
    rows = job.to_records()
    assert {r["control"] for r in rows} == {None, "oracle", "empty"}
    assert job.cost_usd == pytest.approx(0.06)


def test_retried_attempts_keep_the_best_by_default(tmp_path: Path) -> None:
    job = tmp_path / "job"
    now = time.time()
    _trial(job, "t", reward=None, error="infra", suffix="00000001", mtime=now - 100)
    _trial(job, "t", reward=1.0, suffix="00000002", mtime=now - 50)
    # Retries happen inside an Evaluation job, which records evaluation.json.
    (job / "evaluation.json").write_text("{}")
    assert [t.reward for t in bf.load_job(job).trials] == [1.0]
    assert len(bf.load_job(job, attempts="all").trials) == 2


def test_run_batch_rollouts_of_one_task_are_all_samples(tmp_path: Path) -> None:
    """Four rollouts of one task in one ``bf.run_batch`` folder are four trials.

    Guards the dx/sdk fix of the regression from bf6e8412 (SDK update):
    ``load_job`` kept one trial per task and folder, so a batch folder with
    rewards 1, 0, 1, 0 (newest last) gave a solve rate of 0.0 where
    ``attempts="all"`` gave 0.5, and ``bench eval metrics`` said 100% (it
    kept the best of the four) on the same folder.
    """
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.metrics import collect_metrics

    job = tmp_path / "2026-01-01__12-00-00"  # a batch writes no evaluation.json
    now = time.time()
    for i, reward in enumerate([1.0, 0.0, 1.0, 0.0]):
        _trial(job, "hello", reward=reward, suffix=f"0000000{i}", mtime=now + i)

    loaded = bf.load_job(job)
    assert len(loaded.trials) == 4
    assert loaded.solve_rates().solve_rate == 0.5
    assert bf.load_job(job, attempts="all").solve_rates().solve_rate == 0.5
    metrics = collect_metrics(job)
    assert (metrics.total, metrics.passed, metrics.score) == (4, 2, 0.5)
    out = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert out.exit_code == 0, out.output
    summary = json.loads(out.output)
    assert summary["score_ratio"] == summary["solve_rates"]["solve_rate"] == 0.5
    assert summary["passed_tasks"] == summary["failed_tasks"] == ["hello"]


def test_metrics_and_load_job_pick_the_same_attempt(tmp_path: Path) -> None:
    """A retried task whose two attempts are both scored counts as its newest.

    An idle-timeout attempt is scored (its verifier ran) and retried, so an
    Evaluation job can hold two scored attempts of one task. The job's own
    summary and ``bf.load_job`` keep the newest; ``collect_metrics`` (the
    ``bench eval metrics`` table) used to keep the passing one, so the two
    disagreed on the same job. Guards the same dx/sdk fix as above.
    """
    from benchflow.metrics import collect_metrics

    job = tmp_path / "job"
    now = time.time()
    _trial(job, "t", reward=1.0, error="idle", suffix="00000001", mtime=now - 100)
    _trial(job, "t", reward=0.0, suffix="00000002", mtime=now - 50)
    (job / "evaluation.json").write_text("{}")

    assert [t.reward for t in bf.load_job(job).trials] == [0.0]
    metrics = collect_metrics(job)
    assert (metrics.total, metrics.passed, metrics.failed) == (1, 0, 1)


def test_branch_children_are_lineage_not_extra_trials(tmp_path: Path) -> None:
    job = tmp_path / "job"
    d = _trial(job, "hello")
    _tree(d)
    child = d / "branches" / "f1" / "children" / "n6"
    (child / "result.json").write_text(
        json.dumps({"task_name": "hello", "rewards": {"reward": 1.0}})
    )
    loaded = bf.load_job(job)
    assert [t.task_name for t in loaded.trials] == ["hello"]
    assert loaded.trials[0].forks[0].children[1].label == "hint"


def test_compare_pairs_tasks_and_reports_fair_denominators(tmp_path: Path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0, "t3": None, "only-a": 1.0})
    b = _job(tmp_path, "b", {"t1": 1.0, "t2": 1.0, "t3": 0.0, "only-b": 0.0})
    cmp = bf.compare(a, b)
    rows = {r.task: r for r in cmp.rows}
    assert rows["t2"].delta == 1.0 and rows["t1"].delta == 0.0
    assert rows["t3"].delta is None and rows["t3"].status == "paired"
    assert rows["only-a"].status == "only_a" and "zz-oracle" not in rows
    s = cmp.summary
    assert (s.paired, s.only_a, s.only_b, s.both_scored) == (3, 1, 1, 2)
    assert (s.b_higher, s.b_lower, s.same_reward) == (1, 0, 1)
    assert s.mean_delta == pytest.approx(0.5)
    assert cmp.a.attempted == 4 and cmp.b.scored == 4
    assert any("n = 1" in c for c in cmp.caveats)
    assert "t2" in cmp.to_markdown()


def test_load_job_accepts_a_single_trial_dir(tmp_path: Path) -> None:
    d = _trial(tmp_path / "job", "hello")
    assert [t.task_name for t in bf.load_job(d).trials] == ["hello"]


def test_load_job_names_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no trial"):
        bf.load_job(tmp_path)


def test_staged_empty_and_oracle_copies_are_controls(tmp_path: Path) -> None:
    """A task copy suffixed ``__e`` (empty solution) is an empty control even
    under the oracle agent; one suffixed ``__o`` is an oracle control."""
    job = tmp_path / "job"
    empty = bf.load_trial(
        _trial(job, "demo-task__00000001__e", agent="oracle", model=None, reward=0.0)
    )
    oracle = bf.load_trial(
        _trial(job, "demo-task__00000001__o", agent="oracle", model=None)
    )
    agent = bf.load_trial(_trial(job, "demo-task__00000001", suffix="beefbeef"))
    assert (empty.control, oracle.control, agent.control) == ("empty", "oracle", None)


def test_load_job_merges_several_directories(tmp_path: Path) -> None:
    """A paired layout keeps each arm in per-task folders (<task>/a/<ts>,
    <task>/b/<ts>); one arm is then many directories."""
    for task in ("t1", "t2"):
        _trial(tmp_path / task / "a" / "ts", task)
        _trial(tmp_path / task / "b" / "ts", task, reward=0.0)
    arm_a = bf.load_job(sorted(tmp_path.glob("*/a")))
    arm_b = bf.load_job(sorted(tmp_path.glob("*/b")))
    assert [t.task_name for t in arm_a.trials] == ["t1", "t2"]
    cmp = bf.compare(arm_a, arm_b, labels=("a", "b"))
    assert cmp.summary.b_lower == 2
    assert "| Task | a | b |" in cmp.to_markdown()


def test_a_retried_trial_lists_its_attempts(tmp_path: Path) -> None:
    """``Trial.attempts`` exposes the rollouts an Evaluation's retries made.

    The hill-climb demo (docs/examples/hillclimb) had to count trials
    instead of rollouts because ``bf.load_job`` kept only each task's best
    attempt and gave no way to reach the attempts it retried.
    """
    job = tmp_path / "job"
    now = time.time()
    first = _trial(
        job, "t", reward=None, error="pipe closed", suffix="00000001", mtime=now - 30
    )
    second = _trial(
        job, "t", reward=None, error="pipe closed", suffix="00000002", mtime=now - 20
    )
    last = _trial(job, "t", reward=1.0, suffix="00000003", mtime=now - 10)
    _trial(job, "u", reward=0.0, suffix="00000004", mtime=now)
    (job / "evaluation.json").write_text("{}")

    loaded = bf.load_job(job)
    t, u = loaded.trials
    assert [a.path for a in t.attempts] == [first, second, last]
    assert t.attempts[-1] is t and t.reward == 1.0
    assert u.attempts == [u]
    assert sum(len(trial.attempts) for trial in loaded.trials) == 4
    assert {r["task_name"]: r["attempts"] for r in loaded.to_records()} == {
        "t": 3,
        "u": 1,
    }
    exported = {d["task_name"]: d["attempts"] for d in loaded.to_json_dict()["trials"]}
    assert exported == {"t": 3, "u": 1}

    every = bf.load_job(job, attempts="all")
    assert len(every.trials) == 4
    assert all(len(x.attempts) == 3 for x in every.trials if x.task_name == "t")
    assert bf.load_trial(last).attempts[0].path == last  # read alone: itself


def test_printing_a_job_summarises_it(tmp_path: Path) -> None:
    """``print(job)`` says what a notebook user asks first.

    ``repr(job)`` and ``str(job)`` were one line with the path and a trial
    count: no solve rate, no reason for unscored trials, no cost.
    """
    job = tmp_path / "job"
    _trial(job, "t1", reward=1.0, suffix="00000001")
    _trial(job, "t2", reward=0.0, suffix="00000002")
    crashed = _trial(
        job,
        "t3",
        reward=None,
        error="ACP error",
        error_category="acp_error",
        suffix="00000003",
    )
    _trial(job, "zz", agent="oracle", model=None, suffix="0000oracl")
    result = json.loads((crashed / "result.json").read_text())
    result["agent_result"]["price_source"] = "agent_session_log"
    (crashed / "result.json").write_text(json.dumps(result))

    loaded = bf.load_job(job)
    text = str(loaded)
    assert text == loaded.to_markdown() == loaded._repr_markdown_()
    assert "3 trial(s) of 3 task(s); claude-agent-acp · claude-haiku-4-5" in text
    assert "Solve rate: 50.0% (1 of 2 scored trials" in text
    assert "95% interval" in text
    assert "Unscored: 1 of 3 trial(s): acp_error x1 (t3)" in text
    assert "Control runs left out: 1 (oracle)" in text
    assert "Cost: $0.0400 over 4 rollout(s) (1 estimated" in text
    assert repr(loaded).startswith("Job(path=")

    oracle_only = tmp_path / "oracle"
    _trial(oracle_only, "t1", agent="oracle", model=None, cost=None)
    text = str(bf.load_job(oracle_only))
    assert "1 trial(s) of 1 task(s); oracle · no model (only control runs" in text
    assert "Solve rate: 100.0%" in text
    assert "Cost: none (control runs call no model)" in text


def test_an_evaluation_result_prints_on_one_line() -> None:
    from benchflow.evaluation import EvaluationConfig, EvaluationResult
    from benchflow.models import RolloutResult

    result = EvaluationResult(
        job_name="j",
        config=EvaluationConfig(agent="oracle"),
        total=2,
        passed=1,
        failed=1,
        mean_reward=0.5,
        results={
            "a": RolloutResult("a", rewards={"reward": 1.0}),
            "b": RolloutResult("b", rewards={"reward": 0.0}),
        },
    )
    assert repr(result) == (
        "EvaluationResult(job='j', passed=1/2 (50.0%), failed=1, errored=0, "
        "verifier_errored=0, mean_reward=0.500, job_dir=None)"
    )

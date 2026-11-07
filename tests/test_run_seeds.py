"""The seeds a job's numbers depend on are on its record, and a dropped one warns.

Of twenty published benchmark reports, seventeen give no seed, and where a
seed is set no record says whether the provider got it. BenchFlow records
the task order (no seed: tasks start in name order), the task sample's seed,
the rollout seeds, and the sampling seed: the one the run asked for
(``BENCHFLOW_MODEL_SEED``) beside the one each main call carried as it went
upstream, from the model proxy's own records. A requested seed that the
route dropped warns.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from benchflow.run_checks import RunChecks, read_calls, seeding, settings_sent
from benchflow.trajectories.model_calls import write_model_calls
from tests import _run_records as rr
from tests.test_litellm_logging import _callback_namespace

CARD = {
    "agent": {"name": "deepagents"},
    "model": {"id": "openai/gpt-oss-120b", "source": "run"},
    "settings": [
        {"name": "BENCHFLOW_MODEL_SEED", "value": "42", "source": "agent_env"}
    ],
}


def test_the_proxy_records_the_seed_that_went_upstream() -> None:
    """A seed in the request's parameters goes in the record; one the route's
    parameters dropped (only in the agent's own call) does not."""
    logger = _callback_namespace()["BenchFlowLiteLLMLogger"]()
    now = datetime.now()
    sent = logger._base_record(
        {"model": "m", "messages": [], "optional_params": {"seed": 42}}, now, now
    )
    assert sent["request"]["body"]["seed"] == 42
    dropped = logger._base_record(
        {"model": "m", "messages": [], "seed": 42, "optional_params": {}}, now, now
    )
    assert "seed" not in dropped["request"]["body"]


def _trial_with_calls(root: Path, name: str, seed: int | None) -> Path:
    folder = root / name
    (folder / "trajectory").mkdir(parents=True)
    calls = [
        rr.chat_call(
            model="gpt-oss-120b", served="gpt-oss-120b", route="openai/x", effort="high"
        )
        for _ in range(3)
    ]
    for call in calls:
        if seed is not None:
            call["request"]["body"]["seed"] = seed
    (folder / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in calls)
    )
    write_model_calls(folder)
    sent = settings_sent(read_calls(folder), "openai/gpt-oss-120b")
    (folder / "result.json").write_text(
        json.dumps({"rollout_name": name, "settings_sent": sent})
    )
    return folder


def test_each_call_s_record_has_its_seed_and_the_trial_counts_them(tmp_path) -> None:
    trial = _trial_with_calls(tmp_path, "t__1", 42)
    assert {call["seed"] for call in read_calls(trial)} == {42}
    sent = json.loads((trial / "result.json").read_text())["settings_sent"]
    assert sent["seed"] == {"42": 3} and sent["without"]["seed"] == 0


def test_a_requested_seed_the_route_dropped_warns(tmp_path) -> None:
    checks = RunChecks(CARD, planned=2)
    for name in ("a__1", "b__1"):
        checks.record(name, _trial_with_calls(tmp_path, name, None))
    summary = checks.summary()
    assert not summary["failed"]  # a seed warns; it does not fail the run
    (warning,) = [w for w in summary["warnings"] if w.startswith("seed:")]
    assert warning == (
        "seed: requested 42 (from --agent-env); no main call carried it (6 calls in 2 "
        "trial(s)), so the provider's default applied"
    )
    record = seeding(n_tasks=None, sample_seed=None, rollout_seeds=None, checks=summary)
    assert record["sampling_seed"] == {
        "requested": 42,
        "status": "dropped",
        "carried": {},
        "calls": 6,
        "without": 6,
    }
    assert record["warnings"] == [
        "sampling seed: 42 was asked for, but none of the 6 main calls carried it: the "
        "route dropped it, so the provider sampled freely"
    ]


def test_a_seed_the_calls_carried_and_the_task_order(tmp_path) -> None:
    checks = RunChecks(CARD, planned=1)
    checks.record("a__1", _trial_with_calls(tmp_path, "a__1", 42))
    record = seeding(
        n_tasks=20, sample_seed=7, rollout_seeds=[0, 1], checks=checks.summary()
    )
    assert record["sampling_seed"]["status"] == "carried" and not record["warnings"]
    assert (record["task_order"], record["task_order_seed"]) == ("name", None)
    assert record["task_sample"] == {"size": 20, "seed": 7}
    assert record["text"] == (
        "tasks: a random sample of 20 tasks drawn with seed 7, started in name order "
        "(no task-order seed; nothing shuffles them); rollout seeds 0, 1; sampling seed "
        "42 asked for and carried by all 3 main calls"
    )


def test_a_run_writes_its_seeds(tmp_path, monkeypatch) -> None:
    """summary.json, report.md and the run card's after-run lines give them."""
    from tests.test_eval_runner_results_dx import _fake_rollouts, _invoke, _write_task

    tasks = tmp_path / "tasks"
    for name in ("a", "b", "c"):
        _write_task(tasks / name)
    _fake_rollouts(monkeypatch)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--jobs-dir", str(jobs),
        "--n-tasks", "2", "--sample-seed", "7",
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    (summary_path,) = jobs.glob("*/summary.json")
    seeds = json.loads(summary_path.read_text())["scoring"]["seeding"]
    assert seeds["task_sample"] == {"size": 2, "seed": 7}
    assert seeds["sampling_seed"]["status"] == "not recorded"
    report = (summary_path.parent / "report.md").read_text()
    assert (
        "Seeds: tasks: a random sample of 2 tasks drawn with seed 7, started in name "
        "order (no task-order seed; nothing shuffles them); no rollout seeds; the "
        "sampling seed is not recorded (no call went through BenchFlow's model proxy)."
    ) in report
    card = json.loads((summary_path.parent / "run-card.json").read_text())
    assert card["after_run"]["scoring"]["seeding"] == seeds

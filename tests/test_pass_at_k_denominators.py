"""One denominator for pass@k, the solve rate and the mean reward.

BenchFlow 0.8 left errored and verifier-errored trials out of ``n`` for
pass@k, pass^k, the solve rate and the mean reward, so an agent that errored
scored higher than one that failed cleanly. Every figure now counts every
trial (an unscored one as a fail, reward 0), as the Rust ``bench`` 0.8.2
counts, and 0.8's figures stay beside them as ``*_scored_only``, each with
its own n, wherever metrics are shown: ``bench eval metrics`` and its
``--json``, ``summary.json``, the end-of-run line, ``Job.to_markdown``, the
job, comparison and run-summary JSON documents. Job folders written before
the change still read.

Guards the PY8 P4 change (one denominator for pass@k).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from tests.test_python_sdk_load_job import _trial

# A plain folder of rollouts: each is one trial. Two pass, one fails cleanly,
# one errored (no reward): every trial 2/4, scored only 2/3.
REWARDS = {"alpha": 1.0, "beta": None, "gamma": 0.0, "delta": 1.0}


def _errored_job(root: Path) -> Path:
    for task, reward in REWARDS.items():
        _trial(
            root,
            task,
            reward=reward,
            error=None if reward is not None else "agent crashed",
        )
    return root


def _rows() -> dict[str, dict]:
    return {
        task: {
            "task_name": task,
            "rewards": None if reward is None else {"reward": reward},
            "error": None if reward is not None else "agent crashed",
        }
        for task, reward in REWARDS.items()
    }


def test_bench_eval_metrics_json_has_both_figures_and_both_n(tmp_path: Path) -> None:
    job = _errored_job(tmp_path / "job")
    out = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    assert doc["score_ratio"] == 0.5 and doc["total"] == 4
    assert doc["score_excl_errors_ratio"] == pytest.approx(2 / 3)
    assert (doc["mean_reward"], doc["mean_reward_n"]) == (0.5, 4)
    assert doc["mean_reward_scored_only"] == pytest.approx(2 / 3)
    assert doc["mean_reward_scored_only_n"] == 3
    rates = doc["solve_rates"]
    assert (rates["solve_rate"], rates["trials"]) == (0.5, 4)
    assert rates["solve_rate_scored_only"] == pytest.approx(2 / 3)
    assert (rates["scored"], rates["unscored"]) == (3, 1)
    assert rates["pass_at_k"] == {"1": 0.5}
    assert rates["pass_at_k_scored_only"] == {"1": pytest.approx(2 / 3)}
    assert rates["pass_hat_k_scored_only"] == {"1": pytest.approx(2 / 3)}


def _plain(output: str) -> str:
    """The table's text without box-drawing characters, one space apart."""
    return " ".join(re.sub("[\u2500-\u257f]", " ", output).split())


def test_bench_eval_metrics_table_shows_both_figures_with_their_n(
    tmp_path: Path,
) -> None:
    job = _errored_job(tmp_path / "job")
    out = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    text = _plain(out.output)
    assert "Score 50.0% (2 of 4 trials)" in text
    assert "Score, scored only 66.7% (2 of 3 scored trials)" in text
    assert "Mean reward 0.500 (over 4 trials)" in text
    assert "Mean reward, scored only 0.667 (over 3 scored trials)" in text
    assert "Solve rate 50.0% (2 of 4 trials, 4 tasks)" in text
    assert "Solve rate, scored only 66.7% (2 of 3 scored trials, 3 tasks)" in text
    assert "pass@1 50.0%" in text and "pass@1, scored only 66.7%" in text


def test_bench_eval_inspect_headline_shows_both_rates(tmp_path: Path) -> None:
    job = _errored_job(tmp_path / "job")
    out = CliRunner().invoke(app, ["eval", "inspect", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    assert "2/4 attempted passed (50%), 2/3 scored passed (67%)" in out.output


def test_a_job_without_unscored_trials_shows_one_figure(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _trial(job, "alpha", reward=1.0)
    _trial(job, "gamma", reward=0.0)
    out = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    assert "scored only" not in out.output


def test_summary_json_counts_every_trial_and_keeps_the_scored_only_figures() -> None:
    from benchflow._utils.evaluation_results import solve_rate_summary
    from benchflow._utils.scoring import score_summary_fields

    fields = score_summary_fields(_rows().values())
    assert (fields["mean_reward"], fields["mean_reward_n"]) == (0.5, 4)
    assert fields["mean_reward_scored_only"] == pytest.approx(2 / 3)
    assert fields["mean_reward_scored_only_n"] == 3
    assert fields["score_ratio"] == 0.5
    assert fields["score_excl_errors_ratio"] == pytest.approx(2 / 3)
    rates = solve_rate_summary(_rows())["solve_rates"]
    assert rates["pass_at_k"] == {"1": 0.5}
    assert rates["pass_at_k_scored_only"] == {"1": pytest.approx(2 / 3)}
    assert (rates["trials"], rates["scored"]) == (4, 3)


def test_nothing_scored_is_a_zero_not_a_missing_mean() -> None:
    from benchflow._utils.scoring import score_summary_fields

    fields = score_summary_fields(
        [{"task_name": "a", "rewards": None, "error": "boom"}]
    )
    assert fields["mean_reward"] == 0.0 and fields["mean_reward_n"] == 1
    assert fields["mean_reward_scored_only"] is None
    assert fields["mean_reward_scored_only_n"] == 0
    assert score_summary_fields([])["mean_reward"] is None


def _report(result: SimpleNamespace) -> str:
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=200)
    original = shared.console
    shared.console = rec
    try:
        shared._report_eval_result(result)
    finally:
        shared.console = original
    return rec.file.getvalue()


def test_the_end_of_run_line_shows_both_figures() -> None:
    out = _report(
        SimpleNamespace(
            passed=2,
            failed=1,
            total=4,
            errored=1,
            verifier_errored=0,
            score=0.5,
            job_name="j",
            mean_reward=0.5,
            mean_reward_scored_only=2 / 3,
        )
    )
    assert (
        "Score: 2/4 (50.0%), mean reward 0.50 over all 4 trials (unscored as 0), "
        "errors=1; scored only: 2/3 (66.7%), mean reward 0.67 (0.7's mean_reward)"
    ) in out
    clean = _report(
        SimpleNamespace(
            passed=1,
            failed=1,
            total=2,
            errored=0,
            verifier_errored=0,
            score=0.5,
            job_name="j",
            mean_reward=0.5,
            mean_reward_scored_only=0.5,
        )
    )
    assert "scored only" not in clean


def test_evaluation_result_carries_both_means() -> None:
    from benchflow.evaluation import EvaluationResult

    fields = EvaluationResult.__dataclass_fields__
    assert "mean_reward" in fields and "mean_reward_scored_only" in fields


def test_job_markdown_denominators_and_documents_show_both(tmp_path: Path) -> None:
    job = bf.load_job(_errored_job(tmp_path / "job"))
    md = job.to_markdown()
    assert "- Solve rate: 50.0% (2 of 4 trials, passed (reward = 1))" in md
    assert "- Solve rate, scored only: 66.7% (2 of 3 scored trials)" in md

    d = job.denominators()
    assert d.mean_reward == 0.5 and d.attempted == 4
    assert d.mean_reward_scored_only == pytest.approx(2 / 3) and d.scored == 3

    doc = job.to_json_dict()
    assert doc["kind"] == "benchflow.job" and doc["schema_version"] == 2
    assert doc["denominators"]["mean_reward"] == 0.5
    assert doc["denominators"]["mean_reward_scored_only"] == pytest.approx(2 / 3)
    assert doc["solve_rates"]["pass_at_k"]["1"] == 0.5
    assert doc["solve_rates"]["pass_at_k_scored_only"]["1"] == pytest.approx(2 / 3)

    other = tmp_path / "other"
    for task in REWARDS:
        _trial(other, task, reward=1.0)
    cmp = bf.compare(job.path, other, on_mismatch="ignore")
    cmp_doc = cmp.to_json_dict()
    assert cmp_doc["schema_version"] == 2
    assert cmp_doc["a"]["solve_rates"]["solve_rate"] == 0.5
    assert cmp_doc["a"]["solve_rates"]["solve_rate_scored_only"] == pytest.approx(2 / 3)
    cmp_md = cmp.to_markdown()
    assert "2/4 attempted passed (50%), 2/3 scored passed (67%)" in cmp_md
    assert "Solve rate, scored only: 66.7% (2 of 3 scored trials, 3 tasks)" in cmp_md


def test_run_summary_document_has_both_means(tmp_path: Path) -> None:
    from benchflow.job_export import run_summary_export

    doc = run_summary_export(
        SimpleNamespace(
            job_name="j",
            total=4,
            passed=2,
            failed=1,
            errored=1,
            verifier_errored=0,
            score=0.5,
            mean_reward=0.5,
            mean_reward_scored_only=2 / 3,
        ),
        job_dir=tmp_path,
        timeouts=0,
        fail_under=None,
        fail_on=[],
        gate_failed=[],
        exit_code=0,
    ).model_dump(mode="json")
    assert doc["schema_version"] == 2
    assert doc["mean_reward"] == 0.5
    assert doc["mean_reward_scored_only"] == pytest.approx(2 / 3)


def test_batch_results_mean_reward_counts_every_result() -> None:
    from benchflow import RolloutResult
    from benchflow.batch import Results

    rs = Results(
        [
            RolloutResult("a", rewards={"reward": 1.0}),
            RolloutResult("b", error="agent crashed"),
        ]
    )
    assert rs.mean_reward == 0.5
    assert rs.mean_reward_scored_only == 1.0
    assert Results([]).mean_reward is None


# A job folder as BenchFlow 0.8 wrote it: summary.json with the scored-only
# mean_reward and a solve_rates block without the *_scored_only keys.
OLD_SUMMARY = {
    "job_name": "old",
    "agent": "claude-agent-acp",
    "model": "claude-haiku-4-5",
    "total": 3,
    "passed": 1,
    "failed": 1,
    "errored": 1,
    "verifier_errored": 0,
    "score": "33.3%",
    "score_ratio": 1 / 3,
    "score_excl_errors": "50.0%",
    "score_excl_errors_ratio": 0.5,
    "mean_reward": 0.5,
    "solve_rates": {
        "success_rule": "passed (reward = 1)",
        "solve_threshold": None,
        "tasks": 2,
        "trials": 2,
        "unscored": 1,
        "controls_excluded": 0,
        "min_trials_per_task": 1,
        "max_trials_per_task": 1,
        "solve_rate": 0.5,
        "nonbinary_rewards": 0,
        "ks": [1],
        "pass_at_k": {"1": 0.5},
        "pass_hat_k": {"1": 0.5},
        "tasks_at_k": {"1": 2},
        "tasks_short_at_k": {"1": 0},
        "caveats": [],
    },
}


def _old_job(root: Path) -> Path:
    job = root / "old"
    _trial(job, "alpha", reward=1.0)
    _trial(job, "beta", reward=None, error="agent crashed")
    _trial(job, "gamma", reward=0.0)
    (job / "summary.json").write_text(json.dumps(OLD_SUMMARY))
    return job


def test_a_job_folder_written_before_the_change_still_reads(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from benchflow import job_export

    job = _old_job(tmp_path)
    loaded = bf.load_job(job)
    assert loaded.kind == "evaluation" and len(loaded.trials) == 3
    rates = loaded.solve_rates()
    assert rates.solve_rate == pytest.approx(1 / 3) and rates.trials == 3
    assert rates.solve_rate_scored_only == 0.5 and rates.scored == 2

    out = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    assert doc["mean_reward"] == pytest.approx(1 / 3)
    assert doc["mean_reward_scored_only"] == 0.5
    assert doc["solve_rates"]["pass_at_k_scored_only"] == {"1": 0.5}

    exported = loaded.to_json_dict()
    schemas = Path(__file__).resolve().parents[1] / "docs/reference/schemas"
    schema = json.loads((schemas / job_export.schema_filename("job")).read_text())
    jsonschema.validate(exported, schema, cls=jsonschema.Draft202012Validator)
    # The job's own summary.json is carried as written.
    assert exported["summary"]["solve_rates"]["pass_at_k"] == {"1": 0.5}
    md = loaded.to_markdown()
    assert "- Solve rate: 33.3% (1 of 3 trials" in md
    assert "- Solve rate, scored only: 50.0% (1 of 2 scored trials)" in md

    listing = CliRunner().invoke(
        app, ["eval", "list", str(tmp_path)], terminal_width=200
    )
    assert listing.exit_code == 0, listing.output
    assert "1/3 (33.3%)" in listing.output


def test_the_embodied_seed_report_counts_every_trial() -> None:
    from benchflow.embodied.rollouts import format_report, seed_report

    def row(seed: int, reward: float | None, error: str | None = None) -> dict:
        return {
            "task": "t",
            "seed": seed,
            "reward": reward,
            "error": error,
            "outcome": None,
            "steps_used": None,
            "return": None,
            "initial_state_sha256": None,
            "trial": f"t-{seed}",
            "mode": "sim",
        }

    rep = seed_report(
        "unused", rows=[row(0, 1.0), row(1, 0.0), row(2, None, "sandbox died")]
    )
    task, s = rep["tasks"]["t"], rep["summary"]
    assert (task["n"], task["scored"], task["errored"]) == (3, 2, 1)
    assert task["pass_at_k"]["1"] == pytest.approx(1 / 3, abs=1e-6)
    assert task["pass_at_k_scored_only"]["1"] == 0.5
    assert task["mean"] == pytest.approx(1 / 3) and task["mean_scored_only"] == 0.5
    assert s["pass_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert s["pass_rate_scored_only"] == 0.5
    assert s["mean_reward_scored_only"] == 0.5
    assert s["by_mode"]["sim"]["trials"] == 3
    assert "scored only (2 of 3 trials): pass rate 0.500" in format_report(rep)


def test_the_version_1_schema_files_stay_published() -> None:
    schemas = Path(__file__).resolve().parents[1] / "docs/reference/schemas"
    for kind in ("job", "comparison", "run-summary"):
        assert (schemas / f"benchflow-{kind}.v1.schema.json").is_file()
        assert (schemas / f"benchflow-{kind}.v2.schema.json").is_file()


# A task may widen its reward_range below 0 ([-1, 1] scores deliberate harm).
# One trial scored -1.0 and one errored: counting the errored trial as 0 lifts
# the mean to -0.5 above the scored-only -1.0, so an error can outscore a
# trial scored below 0. Guards P4-R1-B-2 (round PY8): every surface that
# shows the every-trial mean says so with a caveat.
NEGATIVE = {"harm": -1.0, "crash": None}


def _negative_job(root: Path) -> Path:
    for task, reward in NEGATIVE.items():
        _trial(
            root,
            task,
            reward=reward,
            error=None if reward is not None else "agent crashed",
        )
    return root


def _says_errors_count_above(text: str | None) -> bool:
    return bool(text) and "below 0" in text and "1 unscored" in text


def test_a_reward_below_zero_with_an_unscored_trial_gets_a_caveat(
    tmp_path: Path,
) -> None:
    from benchflow._utils.scoring import score_summary_fields

    rows = [
        {"task_name": task, "rewards": None if r is None else {"reward": r}}
        | ({} if r is not None else {"error": "agent crashed"})
        for task, r in NEGATIVE.items()
    ]
    fields = score_summary_fields(rows)
    assert (fields["mean_reward"], fields["mean_reward_scored_only"]) == (-0.5, -1.0)
    assert _says_errors_count_above(fields.get("mean_reward_caveat"))

    job = _negative_job(tmp_path / "job")
    out = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    assert (doc["mean_reward"], doc["mean_reward_scored_only"]) == (-0.5, -1.0)
    assert _says_errors_count_above(doc.get("mean_reward_caveat"))
    table = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=300)
    assert table.exit_code == 0, table.output
    assert _says_errors_count_above(_plain(table.output))

    d = bf.load_job(job).denominators()
    assert (d.mean_reward, d.mean_reward_scored_only) == (-0.5, -1.0)
    assert _says_errors_count_above(getattr(d, "mean_reward_caveat", None))


def test_the_end_of_run_line_and_reports_carry_the_below_zero_caveat() -> None:
    from benchflow import RolloutResult
    from benchflow.batch import Results
    from benchflow.embodied.rollouts import format_report, seed_report

    caveat = "1 scored reward(s) are below 0 (lowest -1); ... 1 unscored trial(s)"
    out = _report(
        SimpleNamespace(
            passed=0,
            failed=1,
            total=2,
            errored=1,
            verifier_errored=0,
            score=0.0,
            job_name="j",
            mean_reward=-0.5,
            mean_reward_scored_only=-1.0,
            mean_reward_caveat=caveat,
        )
    )
    assert caveat in out

    rs = Results(
        [
            RolloutResult("harm", rewards={"reward": -1.0}),
            RolloutResult("crash", error="agent crashed"),
        ]
    )
    assert rs.mean_reward == -0.5
    assert _says_errors_count_above(getattr(rs, "mean_reward_caveat", None))

    def row(seed: int, reward: float | None, error: str | None = None) -> dict:
        return {
            "task": "t",
            "seed": seed,
            "reward": reward,
            "error": error,
            "outcome": None,
            "steps_used": None,
            "return": None,
            "initial_state_sha256": None,
            "trial": f"t-{seed}",
            "mode": "sim",
        }

    rep = seed_report("unused", rows=[row(0, -1.0), row(1, None, "sandbox died")])
    assert _says_errors_count_above(rep["summary"].get("mean_reward_caveat"))
    assert _says_errors_count_above(format_report(rep))


def test_no_caveat_when_rewards_are_at_least_zero_or_nothing_is_unscored(
    tmp_path: Path,
) -> None:
    """The every-trial mean never favours errors when rewards are at least 0,
    and with no unscored trial the two means agree."""
    from benchflow._utils.scoring import score_summary_fields

    assert score_summary_fields(_rows().values())["mean_reward_caveat"] is None
    scored = [
        {"task_name": "harm", "rewards": {"reward": -1.0}},
        {"task_name": "ok", "rewards": {"reward": 1.0}},
    ]
    assert score_summary_fields(scored)["mean_reward_caveat"] is None
    job = _errored_job(tmp_path / "job")
    out = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=300)
    assert "below 0" not in out.output
    assert bf.load_job(job).denominators().mean_reward_caveat is None

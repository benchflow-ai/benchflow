"""BenchFlow's one scoring rule, ``benchflow.score.v1``, its interval and its fixture.

Before it, one job could show two scores. summary.json counted a scored
trial whose verifier then recorded an error (a cleanup that failed after the
verify) as not passed, while the end-of-run line counted it passed; the
end-of-run line's error-excluding figure left out a refusal that
summary.json kept as a failure; and report.md's interval treated a task's
trials as independent while the solve rates clustered them by task. Every
headline now comes from ``benchflow.scoring``, and the conformance fixture
``tests/fixtures/scoring/benchflow.score.v1.json`` pins what it gives, so
another reader (BenchFlow Hub) can check it counts the same way.

The fixture's expected figures were computed apart from BenchFlow's code:
exact fractions for the score and SE², the Wilson interval written out from
its formula, pass@k from binomials.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.pass_at_k import Sample, solve_rates
from benchflow.scoring import (
    ENDINGS,
    OUTCOMES,
    RULE_ID,
    RULE_TEXT,
    Score,
    cluster_se,
    outcome,
    score,
    trial_record,
)

FIXTURE = Path(__file__).parent / "fixtures" / "scoring" / "benchflow.score.v1.json"
_REAL = ("score", "score_excl_errors", "se", "ci95")


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _case(name: str) -> dict:
    return next(c for c in _fixture()["cases"] if c["name"].startswith(name))


def _close(got, want, tolerance: float) -> bool:
    if want is None or got is None:
        return got is want
    if isinstance(want, list):
        return len(got) == len(want) and all(
            abs(g - w) <= tolerance for g, w in zip(got, want, strict=True)
        )
    return abs(got - want) <= tolerance


def _check(got: dict, expected: dict, tolerance: float) -> None:
    for key, want in expected.items():
        if key == "pass_at_k":
            continue
        if key in _REAL:
            assert _close(got[key], want, tolerance), (key, got[key], want)
        else:
            assert got[key] == want, (key, got[key], want)


@pytest.mark.parametrize("case", _fixture()["cases"], ids=lambda c: c["name"])
def test_benchflow_reproduces_the_conformance_fixture(case: dict) -> None:
    tolerance = _fixture()["tolerance"]
    _check(score(case["trials"]).to_dict(), case["expected"], tolerance)
    # pass@k over the same trials, from benchflow.pass_at_k.
    # Every trial is a sample, and only a pass is solved, as in the rule.
    samples = [
        Sample(t["task"], 1.0 if t["outcome"] == "passed" else 0.0)
        for t in case["trials"]
    ]
    want_k = case["expected"]["pass_at_k"]
    if want_k:
        rates = solve_rates(samples, ks=[int(k) for k in want_k])
        for k, value in want_k.items():
            assert rates.get(int(k)).pass_at_k == pytest.approx(value, abs=tolerance)
    # Each trial's outcome, who ended it and its flags, from its result.json.
    for trial in case["trials"]:
        if "result" not in trial:
            continue
        assert outcome(trial["result"]) == trial["outcome"], trial
        record = trial_record(trial["result"])
        assert record["outcome"] == trial["outcome"], trial
        assert record["ending"] == trial["ending"], (record, trial)
        assert record["timed_out"] is trial["timed_out"], (record, trial)
        assert record["rate_limited"] is trial["rate_limited"], (record, trial)


def test_the_fixture_is_the_rule_benchflow_applies() -> None:
    doc = _fixture()
    assert doc["rule"] == RULE_ID == "benchflow.score.v1"
    assert tuple(doc["outcomes"]) == OUTCOMES
    assert tuple(doc["endings"]) == ENDINGS
    seen = {t["outcome"] for c in doc["cases"] for t in c["trials"]}
    assert seen == set(OUTCOMES)
    assert {t["ending"] for c in doc["cases"] for t in c["trials"]} == {
        *ENDINGS,
        None,
    }
    # Every case with results gives each of its trials a result.
    assert "counts 1 only when it was scored and passed" in RULE_TEXT


def test_the_interval_is_the_solve_rates_interval() -> None:
    """One interval everywhere: the rule's is the solve rates' own
    (benchflow.pass_at_k.solve_rate_interval), method included."""
    for case in _fixture()["cases"]:
        if not case["trials"]:
            continue
        samples = [
            Sample(t["task"], 1.0 if t["outcome"] == "passed" else 0.0)
            for t in case["trials"]
        ]
        rates = solve_rates(samples, ks=[1])
        got = score(case["trials"])
        assert got.ci95 == rates.interval, case["name"]
        assert got.ci_method == rates.interval_method, case["name"]


def test_the_standard_error_matches_a_hand_computation() -> None:
    """11 trials of 6 tasks, 3 passed (p = 3/11). Each task's passes minus p
    times its trials: alpha (2 of 3) 13/11, beta (0 of 2) -6/11, gamma (0 of
    1) -3/11, delta (1 of 2) 5/11, epsilon (0 of 1) -3/11, zeta (0 of 2)
    -6/11; their squares add to 284/121. SE^2 = 6/5 * (284/121) / 11^2 =
    1704/73205."""
    case = _case("every outcome")
    got = score(case["trials"])
    se2 = Fraction(6, 5) * Fraction(284, 121) / 121
    assert se2 == Fraction(1704, 73205) == Fraction(case["se_squared"])
    assert got.se == pytest.approx(math.sqrt(se2), abs=1e-15)
    # The design effect is se^2 / (p(1-p)/n) = (1704/73205) / (24/1331) =
    # 71/55 > 1: the interval is Wilson's on 11 / (71/55) trials.
    p = 3 / 11
    deff = se2 / (Fraction(3, 11) * Fraction(8, 11) / 11)
    assert deff == Fraction(71, 55)
    n_eff = 11 / float(deff)
    z = 1.96
    centre = (p + z * z / (2 * n_eff)) / (1 + z * z / n_eff)
    half = (
        z
        * math.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff**2))
        / (1 + z * z / n_eff)
    )
    assert got.ci95 == pytest.approx((centre - half, centre + half), abs=1e-12)


def test_one_trial_per_task_is_the_binomial_error_times_the_correction() -> None:
    case = _case("one trial per task")
    got = score(case["trials"])
    p, tasks = got.score, got.n_tasks
    binomial = math.sqrt(p * (1 - p) / tasks)
    assert got.se == pytest.approx(binomial * math.sqrt(tasks / (tasks - 1)))
    assert got.ci_method == "wilson"


def _miller(trials: list[dict]) -> float:
    """Miller (2024), the clustered form: the variance over trials plus the
    covariance of every pair of trials in a task, over n^2, here with
    T/(T-1). Written apart from benchflow.scoring on purpose."""
    n = len(trials)
    values = [1.0 if t["outcome"] == "passed" else 0.0 for t in trials]
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values)
    cross = sum(
        (values[i] - mean) * (values[j] - mean)
        for i in range(n)
        for j in range(n)
        if i != j and trials[i]["task"] == trials[j]["task"]
    )
    tasks = len({t["task"] for t in trials})
    return math.sqrt(tasks / (tasks - 1) * (variance + cross)) / n


def test_the_standard_error_matches_the_pairwise_form_on_uneven_tasks() -> None:
    # 7 tasks with 1 to 7 trials each and a fixed pattern of passes.
    trials = [
        {"task": f"t{k}", "outcome": "passed" if (k * 3 + i) % 4 == 0 else "failed"}
        for k in range(7)
        for i in range(k + 1)
    ]
    assert score(trials).se == pytest.approx(_miller(trials), abs=1e-15)
    counts = [
        (k + 1, sum((k * 3 + i) % 4 == 0 for i in range(k + 1))) for k in range(7)
    ]
    assert cluster_se(counts) == pytest.approx(_miller(trials), abs=1e-15)


def test_an_unknown_outcome_ending_or_missing_task_is_refused() -> None:
    with pytest.raises(ValueError, match="outcome 'pass'"):
        score([{"task": "a", "outcome": "pass"}])
    with pytest.raises(ValueError, match="no task name"):
        score([{"outcome": "passed"}])
    with pytest.raises(ValueError, match="ending 'sandbox'"):
        score([{"task": "a", "outcome": "errored", "ending": "sandbox"}])
    assert score([("a", "passed"), ("b", "failed")]).score == 0.5


def test_a_score_survives_its_json_form() -> None:
    case = _case("provider and harness")
    got = score(case["trials"])
    again = Score.from_dict(json.loads(json.dumps(got.to_dict())))
    assert again.to_dict() == got.to_dict()
    assert got.endings == {"provider": 2, "harness": 1}


def _rows(case: dict) -> list[dict]:
    return [
        {"task_name": t["task"], "rollout_name": f"{t['task']}__{i:04d}", **t["result"]}
        for i, t in enumerate(case["trials"])
    ]


@pytest.mark.parametrize(
    "name", ["every outcome", "provider and harness", "a reward decides"]
)
def test_summary_json_counts_by_the_rule(name: str) -> None:
    """summary.json's scoring block is the fixture's figures, read from the
    trials' result.json fields as the run writes them."""
    from benchflow._utils.scoring import MEAN_REWARD_RULE, score_summary_fields

    case = _case(name)
    want = case["expected"]
    summary = score_summary_fields(_rows(case))
    assert summary["scoring_rule"]["id"] == RULE_ID
    assert summary["scoring_rule"]["text"] == RULE_TEXT
    assert summary["mean_reward_rule"] == MEAN_REWARD_RULE  # kept
    _check(summary["scoring"], want, _fixture()["tolerance"])
    assert summary["score_ratio"] == pytest.approx(want["score"])
    assert summary["score"] == f"{want['score']:.1%}"
    assert summary["score_excl_errors_ratio"] == pytest.approx(
        want["score_excl_errors"]
    )


def test_a_reward_after_a_verifier_error_counts_alike_everywhere() -> None:
    """summary.json counted a scored trial whose verifier then recorded an
    error (a cleanup that failed after the verify) as not passed, while the
    end-of-run line (EvaluationResult, count_score_outcomes) and
    ``bench eval metrics`` counted it passed: two scores for one job. The rule
    counts it by its score in all of them, and the audit counts beside it
    still show the verifier's error."""
    from benchflow._utils.scoring import count_score_outcomes, score_summary_fields

    rows = [
        {
            "task_name": "a",
            "rewards": {"reward": 1.0},
            "verifier_error": "[solver-preserved] post-solver cleanup failed: x",
        },
        {"task_name": "b", "rewards": {"reward": 0.0}},
    ]
    summary = score_summary_fields(rows)
    assert summary["score_ratio"] == 0.5 == count_score_outcomes(rows)["passed"] / 2
    assert summary["scoring"]["outcomes"]["passed"] == 1
    assert summary["verifier_errored"] == 1  # the audit view, unchanged


def test_one_task_and_no_trials() -> None:
    from benchflow._utils.scoring import score_summary_fields

    one = score_summary_fields(
        [{"task_name": "a", "rewards": {"reward": r}} for r in (1.0, 0.0)]
    )["scoring"]
    assert (one["n_tasks"], one["se"], one["ci_method"]) == (1, None, "wilson")
    assert one["score"] == 0.5 and one["ci95"] is not None
    empty = score_summary_fields([])
    assert empty["score_ratio"] == 0.0 and empty["scoring"]["ci95"] is None
    assert score([]).interval_text() == "no interval: no trial"


def _shown(result: object) -> str:
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=300)
    original = shared.console
    shared.console = rec
    try:
        shared._report_eval_result(result)
    finally:
        shared.console = original
    return " ".join(rec.file.getvalue().split())


def test_the_end_of_run_line_is_the_rule_s() -> None:
    """The line under the Score gives the rule's interval and its warnings,
    and the error-excluding figure keeps a refusal as a failure, as
    summary.json does: 2 passed, 1 failed, 1 refused, 1 errored is 2 of 4,
    not 2 of 3."""
    from benchflow._utils.scoring import score_summary_fields

    rows = [
        {"task_name": "a", "rewards": {"reward": 1.0}},
        {"task_name": "b", "rewards": {"reward": 1.0}},
        {"task_name": "c", "rewards": {"reward": 0.0}},
        {
            "task_name": "d",
            "rewards": None,
            "verifier_error": "verifier wrote no usable reward: empty reward.txt",
        },
        {
            "task_name": "e",
            "rewards": None,
            "error": "provider api error [api_error/transient]: HTTP 503",
            "error_category": "api_error",
        },
    ]
    fields = score_summary_fields(rows)
    out = _shown(
        SimpleNamespace(
            passed=2,
            failed=1,
            total=5,
            errored=1,
            verifier_errored=1,
            unusable_rewards=1,
            score=0.4,
            job_name="j",
            mean_reward=0.4,
            mean_reward_scored_only=2 / 3,
            scoring=fields["scoring"],
        )
    )
    assert "scored only: 2/4 (50.0%)" in out
    assert fields["score_excl_errors_ratio"] == 0.5
    low, high = fields["scoring"]["ci95"]
    assert (
        f"95% interval {low:.1%} to {high:.1%}, one trial per task, 5 tasks, "
        "standard error"
    ) in out
    assert "counted by benchflow.score.v1" in out
    assert "1 ended by the model provider (api_error 1)" in out
    assert "Score: k < 3: every task has 1 trial" in out
    # The refusal is a failure, not an error: 1 of 5 has an error.
    assert "error rate 20.0%: 1 of 5 trials ended with an error or no score" in out

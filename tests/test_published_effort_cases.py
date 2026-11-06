"""Effort cases from two published leaderboards, as BenchFlow reports them.

One published leaderboard (66 tasks, 5 trials each, 330 trials a row) lists
one model on Claude Code at five efforts, and another model at effort none:

    Opus 5   low 115 (34.85%)  medium 148 (44.85%)  high 166 (50.30%)
             xhigh 178 (53.94%)  max 171 (51.82%)
    GLM-5.3-Flash  none 118 (35.76%)

Its max run scores below its xhigh run, and Opus 5 at low ranks below
GLM-5.3-Flash. Pooled by model, Opus 5 reads 778/1650 = 47.15%, far above
GLM-5.3-Flash, and both facts disappear.

Another published leaderboard (7 runs of Opus 5, 2 of GPT-5.6 Sol) lists
Opus 5 low 22.32% < GPT-5.6 Sol max 27.34% < Opus 5 high 28.98%: effort alone
reorders the two vendors. Pooled by model, Opus 5 reads 25.65%, below GPT-5.6
Sol, though at high it is above. 384 tasks give those figures exactly: 600
and 779 passes of 2,688 trials, and 210 of 768.

The jobs are written as ``bench run --trials k`` writes them, each trial
from the model proxy's records of its calls (tests/_run_records.py), with
BenchFlow's own writers making the per-call records and result.json's
blocks. A row's passes are spread over its tasks as runs spread them: easy
tasks pass more often than hard ones, with some noise per row, and the total
is the published one. A few of each row's failures are trials that errored
before their first model call.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from benchflow.setting_groups import setting_rows
from tests import _run_records as rr

WIDE = {"COLUMNS": "300"}
OPUS, OPUS_SERVED = "anthropic/claude-opus-5", "claude-opus-5-20260801"
GLM = "glm-5.3-flash"
SOL = "openai/gpt-5.6-sol"
FIRST = {  # (model, effort): passes of 330
    (OPUS, "low"): 115,
    (OPUS, "medium"): 148,
    (OPUS, "high"): 166,
    (OPUS, "xhigh"): 178,
    (OPUS, "max"): 171,
    (GLM, "none"): 118,
}
SECOND = {  # (model, effort): (passes, trials per task)
    (OPUS, "low"): (600, 7),
    (OPUS, "high"): (779, 7),
    (SOL, "max"): (210, 2),
}
# Failures per row that errored before the first model call.
ERRORED = 3


def _spread(passes: int, tasks: int, runs: int, seed: str) -> list[int]:
    """Passes per task summing to ``passes``: task t is harder as t grows,
    the same order for every row, with +/-1 on a fifth of the tasks."""
    rng = random.Random(seed)
    level = passes / (tasks * runs)
    counts = [
        round(runs * min(1.0, max(0.0, level - 0.25 * (-1 + 2 * t / (tasks - 1)))))
        for t in range(tasks)
    ]
    for t in rng.sample(range(tasks), tasks // 5):
        counts[t] = min(runs, max(0, counts[t] + rng.choice((-1, 1))))
    while sum(counts) != passes:
        t = rng.randrange(tasks)
        if sum(counts) > passes and counts[t] > 0:
            counts[t] -= 1
        elif sum(counts) < passes and counts[t] < runs:
            counts[t] += 1
    return counts


def _calls(model: str, effort: str) -> list[dict]:
    if model == OPUS:
        return [
            rr.anthropic_call(model=OPUS, served=OPUS_SERVED, route=OPUS, effort=effort)
            for _ in range(2)
        ]
    if model == SOL:
        return [
            rr.responses_call(model=SOL, served="gpt-5.6-sol", route=SOL, effort=effort)
        ]
    return [rr.chat_call(model=GLM, served=GLM, route=f"openai/{GLM}", effort=effort)]


def _agent(model: str) -> str:
    return {OPUS: "claude-agent-acp", SOL: "codex-acp"}.get(model, "mini-swe-agent")


def _board(
    root: Path,
    rows: dict[tuple[str, str], tuple[int, int]],
    tasks: int,
    sent: dict[tuple[str, str], str] | None = None,
) -> Path:
    """One folder per row, ``<row>/trial-NN/job/<task>__<id>``. ``sent``
    names a row whose calls carried another effort than its label."""
    for (model, effort), (passes, runs) in rows.items():
        name = f"{model.rsplit('/', 1)[-1]}-{effort}"
        carried = (sent or {}).get((model, effort), effort)
        template = rr.Template(root / "templates" / name, _calls(model, carried), model)
        counts = _spread(passes, tasks, runs, name)
        failures = [
            (t, k)
            for t in range(tasks)
            for k in range(runs)
            if (k + t) % runs >= counts[t]
        ]
        errored = set(failures[:: max(1, len(failures) // ERRORED)][:ERRORED])
        for k in range(runs):
            job = root / name / f"trial-{k + 1:02d}" / "job"
            for t in range(tasks):
                passed = (k + t) % runs < counts[t]
                rr.trial(
                    job,
                    f"task-{t:03d}",
                    k * tasks + t,
                    reward=None if (t, k) in errored else float(passed),
                    model=model,
                    effort=effort,
                    agent=_agent(model),
                    template=None if (t, k) in errored else template,
                    error="sandbox lost before the agent started"
                    if (t, k) in errored
                    else None,
                    error_category="infra_failure" if (t, k) in errored else None,
                )
    return root


@pytest.fixture(scope="module")
def first(tmp_path_factory) -> Path:
    rows = {key: (passes, 5) for key, passes in FIRST.items()}
    return _board(tmp_path_factory.mktemp("first") / "jobs", rows, tasks=66)


@pytest.fixture(scope="module")
def second(tmp_path_factory) -> Path:
    return _board(tmp_path_factory.mktemp("second") / "jobs", SECOND, tasks=384)


def _by_key(settings) -> dict[tuple[str, str], object]:
    model = {OPUS_SERVED: OPUS, "gpt-5.6-sol": SOL, GLM: GLM}
    return {(model[r.setting.model], r.setting.effort): r for r in settings.rows}


def test_each_effort_is_its_own_row_in_the_published_order(first: Path) -> None:
    settings = setting_rows(bf.load_job(first).trials)
    rows = _by_key(settings)
    assert set(rows) == set(FIRST)
    for key, row in rows.items():
        figures = row.score()
        assert (figures.n, figures.n_tasks, figures.passed) == (330, 66, FIRST[key])
        # Its trials that errored before a model call stay in the row as 0s.
        assert figures.outcomes["errored"] == ERRORED
        assert (
            figures.ci95 is not None
            and figures.ci95[0] < figures.score < figures.ci95[1]
        )
    ranked = sorted(rows, key=lambda key: -(rows[key].score().score or 0.0))
    assert ranked == [
        (OPUS, "xhigh"),
        (OPUS, "max"),
        (OPUS, "high"),
        (OPUS, "medium"),
        (GLM, "none"),
        (OPUS, "low"),
    ]
    score = {key: row.score().score for key, row in rows.items()}
    assert score[(OPUS, "xhigh")] > score[(OPUS, "max")] > score[(OPUS, "low")]
    assert score[(OPUS, "low")] == pytest.approx(115 / 330)  # 34.85%
    assert score[(OPUS, "xhigh")] == pytest.approx(178 / 330)  # 53.94%
    assert score[(OPUS, "max")] == pytest.approx(171 / 330)  # 51.82%


def test_a_by_model_pooling_would_rank_wrongly_and_no_reader_pools(first: Path) -> None:
    pooled = sum(p for (m, _), p in FIRST.items() if m == OPUS) / (5 * 330)
    glm = FIRST[(GLM, "none")] / 330
    low = FIRST[(OPUS, "low")] / 330
    # Pooled, Opus 5 ranks far above GLM-5.3-Flash; at low it ranks below.
    assert pooled == pytest.approx(778 / 1650) and pooled > glm > low

    doc = json.loads(
        CliRunner().invoke(app, ["eval", "metrics", str(first), "--json"]).stdout
    )
    assert doc["score"] is None and doc["score_ratio"] is None
    assert doc["solve_rates"] is None
    assert "mixed settings: these trials ran 6 settings" in doc["headline_withheld"]
    rows = doc["scoring"]["settings"]["rows"]
    assert len(rows) == 6
    assert all(row["scoring"]["n"] == 330 for row in rows)
    assert not any(row["scoring"]["passed"] == 778 for row in rows)
    assert {(r["effort"], r["scoring"]["passed"]) for r in rows} == {
        (effort, passes) for (_, effort), passes in FIRST.items()
    }
    out = CliRunner().invoke(app, ["eval", "metrics", str(first)], env=WIDE)
    text = " ".join(re.sub("[─-╿]", " ", out.output).split())
    assert "47.1%" not in text and "47.2%" not in text and "1650" not in text
    assert "effort xhigh, BenchFlow 0.10.0: 178 of 330 trials passed (53.9%;" in text


def test_a_gap_inside_its_interval_is_flagged(first: Path) -> None:
    """xhigh 53.94% against max 51.82%: 2.1 points, inside the gap's 95%
    interval over the 66 tasks and under 3 points. low against xhigh, 19.1
    points, is outside it and is not flagged."""
    xhigh, maxed, low = (first / f"claude-opus-5-{e}" for e in ("xhigh", "max", "low"))
    close = bf.compare(xhigh, maxed, vary=("reasoning_effort",))
    scoring = close.scoring or {}
    assert scoring["gap"] == pytest.approx((171 - 178) / 330)
    assert scoring["gap_inside_ci"] is True and scoring["gap_under_3_points"] is True
    assert scoring["tasks"] == 66
    low_ci, high_ci = scoring["gap_ci95"]
    assert low_ci < 0 < high_ci
    warning = next(w for w in scoring["warnings"] if w.startswith("the gap"))
    assert "its 95% interval holds 0 and it is under 3 points" in warning
    markdown = close.to_markdown()
    assert "claude-opus-5-max - claude-opus-5-xhigh = -2.1 points" in markdown
    assert "Warning: the gap claude-opus-5-max - claude-opus-5-xhigh = -2.1 points" in (
        markdown
    )

    far = bf.compare(low, xhigh, vary=("reasoning_effort",))
    far_scoring = far.scoring or {}
    assert far_scoring["gap"] == pytest.approx((178 - 115) / 330)
    assert far_scoring["gap_inside_ci"] is False
    assert far_scoring["gap_under_3_points"] is False
    assert not any(w.startswith("the gap") for w in far_scoring["warnings"])
    # Both sides have 5 trials a task: no k < 3 warning.
    assert not any("k < 3" in w for w in far_scoring["warnings"])

    # The command line says it too.
    out = CliRunner().invoke(
        app,
        ["eval", "compare", str(xhigh), str(maxed), "--vary", "reasoning_effort"],
        env=WIDE,
    )
    assert out.exit_code == 0, out.output
    assert "its 95% interval holds 0 and it is under 3 points" in " ".join(
        out.output.split()
    )


def test_two_efforts_differ_in_what_ran_not_only_in_their_labels(first: Path) -> None:
    low, xhigh = first / "claude-opus-5-low", first / "claude-opus-5-xhigh"
    with pytest.warns(UserWarning, match="effort_sent"):
        result = bf.compare(low, xhigh)
    assert {m.setting for m in result.mismatches} == {"reasoning_effort", "effort_sent"}
    sent = next(m for m in result.mismatches if m.setting == "effort_sent")
    assert (sent.a, sent.b) == (["low"], ["xhigh"])


def test_the_second_board_keeps_its_order_across_vendors(second: Path) -> None:
    rows = _by_key(setting_rows(bf.load_job(second).trials))
    assert set(rows) == set(SECOND)
    score = {key: row.score().score for key, row in rows.items()}
    assert score[(OPUS, "low")] == pytest.approx(600 / 2688)  # 22.32%
    assert score[(SOL, "max")] == pytest.approx(210 / 768)  # 27.34%
    assert score[(OPUS, "high")] == pytest.approx(779 / 2688)  # 28.98%
    assert score[(OPUS, "high")] > score[(SOL, "max")] > score[(OPUS, "low")]
    assert rows[(SOL, "max")].setting.agent == "codex-acp"
    assert rows[(SOL, "max")].score().tries == (2, 2)
    # Pooled by model, Opus 5 ranks below GPT-5.6 Sol: wrong for high.
    pooled = (600 + 779) / (2 * 2688)
    assert pooled < 210 / 768 < 779 / 2688

    # Sol has 2 runs a task (k < 3), and the gap to Opus at high is 1.6
    # points: both are flagged.
    result = bf.compare(
        second / "gpt-5.6-sol-max", second / "claude-opus-5-high", vary=("model",)
    )
    scoring = result.scoring or {}
    assert scoring["gap"] == pytest.approx(779 / 2688 - 210 / 768)
    assert scoring["gap_under_3_points"] is True
    assert any(w.startswith("gpt-5.6-sol-max: k < 3") for w in scoring["warnings"])


def test_a_row_labelled_max_whose_calls_carried_high_is_reported_as_high(
    tmp_path: Path,
) -> None:
    """The max row of the first board, its calls carrying high: reported
    under the effort sent, apart from the row that asked for high, and
    warned about."""
    rows = {(OPUS, e): (FIRST[(OPUS, e)], 5) for e in ("high", "max")}
    board = _board(tmp_path / "jobs", rows, tasks=66, sent={(OPUS, "max"): "high"})
    settings = setting_rows(bf.load_job(board).trials)
    assert [(r.setting.effort, r.setting.effort_requested) for r in settings.rows] == [
        ("high", "high"),
        ("high", "max"),
    ]
    labelled = settings.rows[1]
    assert labelled.setting.effort_mismatch and labelled.score().passed == 171
    assert "effort high (requested max)" in labelled.setting.label()
    (warning,) = settings.effort_warnings()
    assert warning.startswith(
        "effort label: 330 trials labelled max, but the calls carried high; they "
        "are reported under the effort that ran"
    )
    doc = json.loads(
        CliRunner().invoke(app, ["eval", "metrics", str(board), "--json"]).stdout
    )
    assert doc["scoring"]["settings"]["effort_warnings"] == [warning]
    out = CliRunner().invoke(app, ["eval", "metrics", str(board)], env=WIDE)
    assert "effort label: 330 trials labelled max" in " ".join(out.output.split())

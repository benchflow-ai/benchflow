"""Golden snapshots of a run's final screen and its --json document.

The default end of `bench run` is one row per trial, one footer (score, model
cost, tokens, wall time) and the results folder (cli/_output/run_summary.py).
Each fixture is a run as an EvaluationResult carries it, built from real
RolloutResult rows, and its screen must match tests/cli_golden/<name>.txt byte
for byte; BENCHFLOW_UPDATE_GOLDEN=1 rewrites them. The --json document is
checked as data, not text. Spec: run/hub/pm-views-20261117/G-cli-ux-vs-harbor.md
("Output contract", "Golden output and acceptance tests").
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from benchflow.cli._output.run_summary import (
    SCHEMA,
    cost_cell,
    summary_document,
    summary_lines,
    trial_rows,
)
from benchflow.models import RolloutResult

GOLDEN = Path(__file__).parent / "cli_golden"
START = datetime(2026, 11, 18, 3, 30, 13)


def _trial(
    task: str,
    *,
    reward: float | None = 1.0,
    tokens: int | None = None,
    cost: float | None = None,
    source: str | None = "list_price_estimate",
    partial: bool | None = None,
    error: str | None = None,
    category: str | None = None,
    seconds: float = 58,
    judge: dict | None = None,
) -> RolloutResult:
    return RolloutResult(
        judge_usage=judge,
        task_name=task,
        rollout_name=f"{task}__abc123",
        rewards=None if reward is None else {"reward": reward},
        total_tokens=tokens,
        cost_usd=cost,
        cost_source=source if cost is not None else None,
        usage_source="proxy" if tokens is not None else "unavailable",
        usage_incomplete=partial,
        error=error,
        error_category=category,
        started_at=START,
        finished_at=START + timedelta(seconds=seconds),
    )


def _run(
    trials, *, agent="claude-agent-acp", model="deepseek/deepseek-flash", **fields
):
    results = {t.rollout_name: t for t in trials}
    passed = sum(t.score_outcome == "passed" for t in trials)
    failed = sum(t.score_outcome == "failed" for t in trials)
    errored = sum(t.score_outcome == "errored" for t in trials)
    rewards = [t.reward or 0.0 for t in trials]
    base = {
        "job_name": "2026-11-18__03-30-13",
        "config": SimpleNamespace(agent=agent, model=model),
        "results": results,
        "total": len(trials),
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "verifier_errored": 0,
        "not_run": 0,
        "reused": 0,
        "ran": len(trials),
        "elapsed_sec": 120.4,
        "score": passed / len(trials),
        "mean_reward": sum(rewards) / len(rewards),
    }
    base.update(fields)
    return SimpleNamespace(**base)


FIXTURES = {
    "oracle": _run(
        [_trial("0", seconds=31), _trial("1", seconds=33)],
        agent="oracle",
        model=None,
        elapsed_sec=68.0,
    ),
    "paid-estimate": _run(
        [
            _trial("0", tokens=33_351, cost=0.0054),
            _trial("1", tokens=33_351, cost=0.0054, seconds=107),
        ]
    ),
    "paid-provider": _run(
        [
            _trial("0", tokens=33_351, cost=0.0054, source="provider"),
            _trial("1", tokens=33_351, cost=0.0054, source="provider"),
        ]
    ),
    "mixed": _run(
        [
            _trial("hello-pass", tokens=12_000, cost=0.002),
            _trial("hello-wrong", reward=0.0, tokens=11_500, cost=0.0019),
            _trial(
                "hello-timeout",
                reward=None,
                error="agent timed out after 180s",
                category="timeout",
                seconds=180,
            ),
        ]
    ),
    "partial-cost": _run(
        [
            _trial("0", tokens=22_862, cost=0.003922284),
            _trial("1", tokens=11_290, cost=0.0036237, partial=True),
        ]
    ),
    "resumed": _run(
        [_trial("0", seconds=40), _trial("1", seconds=41)],
        agent="oracle",
        model=None,
        reused=1,
        ran=1,
        elapsed_sec=70.2,
    ),
    "nonbinary": _run(
        [
            _trial("unit-distance", reward=0.5, tokens=40_000, cost=0.03),
            _trial("binary-fuse", reward=1.0, tokens=30_000, cost=0.03),
        ]
    ),
    "narrow": _run(
        [
            _trial(
                "datasets/gpqa-diamond/a-very-long-task-name-that-overflows-80-columns",
                tokens=33_351,
                cost=0.0054,
            )
        ]
    ),
}
# The 2 judge calls of docs/examples/llm-judge/sea-haiku's verifier on an
# oracle run (the part-2 review's B2 repro, result.json judge_usage).
SEA_HAIKU_JUDGE = {
    "calls": 2,
    "calls_without_usage": 0,
    "total_tokens": 995,
    "cost_usd": 0.0006747,
    "cost_source": "list_price_estimate",
}
FIXTURES["oracle-judge"] = _run(
    [_trial("sea-haiku", seconds=21, judge=SEA_HAIKU_JUDGE)],
    agent="oracle",
    model=None,
    elapsed_sec=22.0,
)
# The part-2 review's B1 repro: a claude trial killed mid tool call (no result,
# no receipt), then `bench resume latest`; the kept trial counted some calls.
FIXTURES["resumed-salvaged"] = _run(
    [_trial("state", tokens=22_845, cost=0.003906384, partial=True, seconds=147)],
    elapsed_sec=148.0,
)
RESUMED_SPEND = {
    "attempts": 1,
    "stopped_attempts": 1,
    "cost_usd": 0.007953684,
    "tokens": 34_488,
    "judge_cost_usd": None,
    "judge_tokens": None,
    "attempts_without_cost": 0,
    "attempts_without_tokens": 0,
    "attempts_without_model_call": 0,
    "stopped_attempts_without_usage": 0,
    "attempts_usage_salvaged": 1,
    "stopped_attempts_without_cost": 0,
    "attempts_cost_incomplete": 1,
}
#: summary.json's spend_all_attempts per fixture (none: no summary.json).
SPEND = {"resumed-salvaged": RESUMED_SPEND}
WIDTHS = {"narrow": 80}


def _screen(name: str) -> str:
    from benchflow.cli._output import run_summary

    with mock.patch.object(
        run_summary, "every_attempt_spend", lambda _job_dir: SPEND.get(name)
    ):
        lines = summary_lines(
            FIXTURES[name],
            Path("jobs/2026-11-18__03-30-13"),
            width=WIDTHS.get(name, 120),
        )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_final_screen_matches_its_golden(name: str) -> None:
    path = GOLDEN / f"{name}.txt"
    got = _screen(name)
    if os.environ.get("BENCHFLOW_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(got)
    assert got == path.read_text()


def test_the_screen_is_plain_text_within_its_width() -> None:
    for name in FIXTURES:
        screen = _screen(name)
        assert "\x1b" not in screen
        width = WIDTHS.get(name, 120)
        assert all(len(line) <= width for line in screen.splitlines()), name


def test_a_missing_cost_is_never_a_zero() -> None:
    rows = trial_rows(FIXTURES["mixed"])
    assert cost_cell(rows) == "≥~$0.0039 · 2/3 trials"
    assert cost_cell([r for r in rows if r.task == "hello-timeout"]) == "—"
    assert cost_cell(trial_rows(FIXTURES["oracle"])) == "$0"


def test_the_json_document_carries_the_screen_s_figures() -> None:
    document = json.loads(
        json.dumps(summary_document(FIXTURES["mixed"], Path("jobs/x")))
    )
    assert document["schema"] == SCHEMA
    assert document["counts"] == {
        "trials": 3,
        "passed": 1,
        "failed": 1,
        "unscored": 1,
        "not_run": 0,
        "reused": 0,
    }
    assert document["cost"] == {
        "usd": pytest.approx(0.0039),
        "source": "estimate",
        "trials_priced": 2,
        "partial": True,
        "judge_calls": 0,
        "judge_cost_usd": None,
        "judge_tokens": None,
    }
    assert document["tokens"] == 23_500
    assert document["spend_all_attempts"] is None  # no summary.json
    timeout = next(t for t in document["trials"] if t["task"] == "hello-timeout")
    assert timeout["outcome"] == "unscored"
    assert timeout["cost_usd"] is None and timeout["tokens"] is None
    assert timeout["error_category"] == "timeout"


def test_a_control_spends_nothing_and_counts_no_tokens() -> None:
    document = summary_document(FIXTURES["oracle"], None)
    assert document["cost"]["usd"] == 0.0 and document["cost"]["source"] == "no_model"
    assert document["tokens"] is None


@pytest.mark.parametrize(
    ("reward", "said"),
    [
        (0.0, "nop control: 0/1, as expected: doing nothing passes none of the tasks"),
        (1.0, "nop control: 1/1 passed with nothing done: those tasks can be passed"),
    ],
)
def test_the_default_screen_gives_the_nop_control_s_verdict(
    reward, said, capsys
) -> None:
    """A nop run's 0 is the good result; the new screen dropped the line that
    says so (the first-run dogfood's DF-p1-first-run-13), and the docs' own
    journeys check for it."""
    from benchflow.cli._output.report import print_run_result

    nop = _run([_trial("hello", reward=reward)], agent="nop", model=None)
    nop.task_failures = []
    print_run_result(nop, Path("jobs/nop"))
    out = " ".join(capsys.readouterr().out.split())
    assert said in out


def test_an_oracle_whose_judge_called_a_model_is_not_free() -> None:
    """B2: a control's $0 and no_model hold only with no model call at all,
    its verifier's LLM judge included."""
    (row,) = trial_rows(FIXTURES["oracle-judge"])
    assert cost_cell([row]) == "~$0.0007" and row.tokens == 995
    document = summary_document(FIXTURES["oracle-judge"], None)
    assert document["cost"]["source"] == "estimate"
    assert document["cost"]["usd"] == pytest.approx(0.0006747)
    assert document["cost"]["judge_calls"] == 2
    assert document["cost"]["judge_cost_usd"] == pytest.approx(0.0006747)
    assert document["tokens"] == 995
    (trial,) = document["trials"]
    assert trial["cost_usd"] == pytest.approx(0.0006747) and trial["judge_calls"] == 2
    # With no judge call it stays $0 and no_model.
    assert summary_document(FIXTURES["oracle"], None)["cost"]["source"] == "no_model"


def test_a_judge_without_a_price_makes_a_floor() -> None:
    judge = {**SEA_HAIKU_JUDGE, "cost_usd": None, "cost_source": None}
    run = _run(
        [_trial("0", tokens=1_000, cost=0.001, judge=judge)], model="deepseek-flash"
    )
    (row,) = trial_rows(run)
    assert cost_cell([row]) == "≥~$0.0010" and row.tokens == 1_995


def test_a_resumed_job_names_every_attempt_it_paid_for(tmp_path) -> None:
    """B1: the stopped, salvaged attempt the resume left behind is in the
    screen and the document, as summary.json counts it."""
    (tmp_path / "summary.json").write_text(
        json.dumps({"spend_all_attempts": RESUMED_SPEND})
    )
    lines = summary_lines(FIXTURES["resumed-salvaged"], tmp_path)
    assert (
        "Every attempt: ≥~$0.0080, ≥34,488 tokens (2 attempts, 1 stopped, 1 counted "
        "from its logs)"
    ) in lines
    document = summary_document(FIXTURES["resumed-salvaged"], tmp_path)
    assert document["spend_all_attempts"] == {
        "usd": 0.007953684,
        "tokens": 34_488,
        "judge_usd": None,
        "attempts": 2,
        "stopped": 1,
        "salvaged": 1,
        "partial": True,
    }


def test_a_job_with_one_attempt_per_trial_adds_no_line(tmp_path) -> None:
    spend = {**RESUMED_SPEND, "attempts": 2, "stopped_attempts": 0}
    spend.update(attempts_usage_salvaged=0, attempts_cost_incomplete=0)
    (tmp_path / "summary.json").write_text(json.dumps({"spend_all_attempts": spend}))
    lines = summary_lines(FIXTURES["paid-estimate"], tmp_path)
    assert not any(line.startswith("Every attempt") for line in lines)
    assert (
        summary_document(FIXTURES["paid-estimate"], tmp_path)["spend_all_attempts"][
            "attempts"
        ]
        == 2
    )


def test_a_refused_json_run_says_why_in_its_document(monkeypatch, capsys) -> None:
    """A `bench run --json` that refuses a resume put "error": null in its
    document, the reason only on stderr (the part-2 review, round 2)."""
    from benchflow.cli import _errors
    from benchflow.cli._output import report
    from benchflow.cli._shared import print_error

    monkeypatch.setattr(_errors, "_JSON_RUN", _errors._JsonRun())
    monkeypatch.setattr(
        report, "_state", {"json": True, "stdout": None, "written": False}
    )
    print_error(
        "Refusing to resume job trial: 1 trial in it ran prompts=['x'] "
        "(this run: None).",
        next_step="pass the same --prompt, or --fresh",
    )
    report.write_failure(1)
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "error" and document["exit_code"] == 1
    assert document["error"].startswith("Refusing to resume job trial:")
    assert document["next_step"] == "pass the same --prompt, or --fresh"

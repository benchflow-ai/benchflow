"""The run's closing ``Cost:`` line names a floor once (EN-624).

The partial-count note EN-624 added to the closing line ("N trial(s) counted
only some of their calls, so the total is a floor") and EN-648's per-cause
floor notes (``floor_causes``) each said it, so the merged closing line said
it twice. The shape is a real mixed-floor run's through the model proxy: two
trials whose stream was cut during a held call (``usage_incomplete``, each
priced from the calls that were counted) and one whole trial.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

import benchflow.cli._shared as shared
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.models import RolloutResult

CUT = (
    "stream interrupted before final usage: 1 call(s) through the model proxy "
    "ended before the provider reported their usage"
)


def _agent(tokens: int, cost: float, *, cut: bool) -> dict:
    agent = {
        "usage_source": "proxy",
        "n_input_tokens": tokens - 50 * (tokens // 1050),
        "n_output_tokens": 50 * (tokens // 1050),
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": tokens,
        "cost_usd": cost,
        "cost_source": "list_price_estimate",
        "price_source": "litellm",
    }
    if cut:
        agent.update(
            usage_incomplete=True, usage_incomplete_reason=CUT, cost_incomplete=True
        )
    return agent


def _job(tmp_path: Path, trials: dict[str, dict]) -> tuple[Path, EvaluationResult]:
    job = tmp_path / "jobs" / "s12-mixed"
    job.mkdir(parents=True)
    (job / "summary.json").write_text("{}")
    results = {}
    for name, agent in trials.items():
        folder = job / f"{name}__0a1b2c3d"
        folder.mkdir()
        payload = {
            "task_name": name,
            "rollout_name": folder.name,
            "agent": "codex-acp",
            "model": "gpt-5.5",
            "rewards": {"reward": 1.0},
            "agent_result": agent,
        }
        (folder / "result.json").write_text(json.dumps(payload))
        results[name] = RolloutResult.from_dict(payload, rollout_dir=folder)
    return job, EvaluationResult(
        job_name=job.name,
        config=EvaluationConfig(),
        total=len(results),
        passed=len(results),
        elapsed_sec=219.0,
        job_dir=job,
        results=results,
    )


def _closing_line(monkeypatch, tmp_path: Path, trials: dict[str, dict]) -> str:
    job, result = _job(tmp_path, trials)
    rec = Console(file=io.StringIO(), width=400)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(result, job)
    return next(row for row in rec.file.getvalue().splitlines() if "Cost:" in row)


def test_two_cut_trials_and_a_whole_one_name_the_floor_once(tmp_path, monkeypatch):
    line = _closing_line(
        monkeypatch,
        tmp_path,
        {
            "t1-warmup": _agent(2100, 0.013, cut=False),
            "t2-cut": _agent(1050, 0.0065, cut=True),
            "t3-cut": _agent(1050, 0.0065, cut=True),
        },
    )
    assert (
        "2 trial(s) counted only some of their calls, so the total is a floor" in line
    )
    assert line.count("counted only some of their calls") == 1, line
    assert line.count("so the total is a floor") == 1, line


def test_whole_trials_name_no_floor(tmp_path, monkeypatch):
    line = _closing_line(
        monkeypatch,
        tmp_path,
        {
            "t1-warmup": _agent(2100, 0.013, cut=False),
            "t4-whole": _agent(2100, 0.013, cut=False),
        },
    )
    assert "floor" not in line, line

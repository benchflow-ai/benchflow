"""#261's floors in the readers #260 and #262 bring (R5 merge, rel 0.9.3).

- The live footer marks every floor: a count that missed calls (#260), and
  #261's cost-only floors, an unpriced Codex warm-up and cache writes a
  priced route left unreported (a review B).
- ``bench eval metrics --json`` counts the trials without a cache-write
  count over summary.json's rows, stopped attempts included, as summary.json
  does (a review A).
- A cut trial is a floor and is never called a warm-up: report.md's Cost
  line and the closing Cost line name each floor by its own cause (F2b).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from typer.testing import CliRunner

import benchflow as bf
import benchflow.cli._shared as shared
from benchflow._utils.result_metadata import codex_warmup
from benchflow.cli._live_progress import LiveEvalProgress
from benchflow.cli.main import app
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.jobs import _cost_line
from benchflow.models import RolloutResult

_PRICED = {
    "usage_source": "proxy",
    "usage_source_legacy": "provider_response",
    "cost_source": "list_price_estimate",
    "price_source": "litellm",
    "no_usage_reason": None,
}
_COUNTS = {
    "n_input_tokens": 1626,
    "n_output_tokens": 60,
    "n_cache_read_tokens": 11264,
    "n_cache_creation_tokens": 0,
    "total_tokens": 12950,
    "cost_usd": 0.5,
}
_CACHE_WRITES = {
    "tokens": None,
    "named_tokens": 0,
    "inside_input": True,
    "models": ["openai/gpt-6.1-sol"],
    "priced_models": ["openai/gpt-6.1-sol"],
    "cost_floor": True,
}


def _agent(kind: str) -> dict[str, Any]:
    agent = {**_COUNTS, **_PRICED}
    if kind == "warmup":
        agent.update(
            usage_source="session_log", usage_details={"warmup": codex_warmup()}
        )
    elif kind == "cache_writes":
        agent.update(
            n_cache_creation_tokens=None, usage_details={"cache_writes": _CACHE_WRITES}
        )
    elif kind == "cut":
        agent.update(
            usage_incomplete=True,
            usage_incomplete_reason="a stream was cut before the provider's usage",
        )
    return agent


def _payload(name: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "task_name": name,
        "rollout_name": f"{name}__0a1b2c3d",
        "agent": "codex-acp",
        "model": "gpt-6.1-sol",
        "rewards": {"reward": 1.0},
        "started_at": "2026-11-15 09:00:00.000000",
        "finished_at": "2026-11-15 09:01:00.000000",
        "agent_result": _agent(kind),
        **fields,
    }


def _footer(*results: RolloutResult) -> str:
    d = LiveEvalProgress(
        Console(), label="bench", agent="codex-acp", model="m", sandbox="docker"
    )
    d.on_plan(total=len(results), done=0, remaining=len(results))
    for index, result in enumerate(results):
        d.on_result(f"t{index}", result)
    out = Console(file=io.StringIO(), width=200)
    out.print(d.__rich__())
    return out.file.getvalue()


def test_the_footer_marks_each_cost_only_floor():
    for kind in ("warmup", "cache_writes", "cut"):
        result = RolloutResult.from_dict(_payload("t", kind))
        assert "$0.50+ (floor)" in _footer(result), kind
    whole = RolloutResult.from_dict(_payload("t", "whole"))
    text = _footer(whole)
    assert "$0.50" in text and "(floor)" not in text


def _job(tmp_path: Path, kinds: dict[str, str]) -> Path:
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    for name, kind in kinds.items():
        trial = job / f"{name}__0a1b2c3d"
        trial.mkdir()
        (trial / "result.json").write_text(json.dumps(_payload(name, kind)))
    return job


def test_a_cut_trial_is_a_floor_and_never_called_a_warmup(tmp_path, monkeypatch):
    job = _job(tmp_path, {"cut1": "cut", "cut2": "cut", "whole": "whole"})
    line = _cost_line(bf.load_job(job).trials)
    assert "2 rollout(s) counted only some of their calls" in line
    assert "so the total is a floor" in line
    assert "warm-up" not in line and "generating calls only" not in line

    results = {
        name: RolloutResult.load(job / f"{name}__0a1b2c3d")
        for name in ("cut1", "cut2", "whole")
    }
    rec = Console(file=io.StringIO(), width=300)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(
        EvaluationResult(
            job_name=job.name,
            config=EvaluationConfig(),
            total=3,
            passed=3,
            elapsed_sec=10.0,
            job_dir=job,
            results=results,
        ),
        job,
    )
    closing = next(
        " ".join(line.split())
        for line in rec.file.getvalue().splitlines()
        if "Cost:" in line
    )
    assert closing.count("2 trial(s) counted only some of their calls") == 1
    assert "warm-up" not in closing and "generating calls only" not in closing


def test_a_warmup_trial_is_called_a_warmup_and_a_cut_one_is_not(tmp_path):
    job = _job(tmp_path, {"warm": "warmup", "cut": "cut"})
    line = _cost_line(bf.load_job(job).trials)
    assert "1 rollout(s) counted only some of their calls" in line
    assert "1 rollout(s) count generating calls only" in line


def test_bench_eval_metrics_counts_a_stopped_attempt_s_unknown_writes(tmp_path):
    """A SIGINT-stopped Codex attempt whose write count is unknown: summary.json
    counts it in trials_without_cache_write_count, and so does the metrics JSON."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_paths import interrupted_trial_results

    job = tmp_path / "jobs" / "cancel"
    (tmp_path / "tasks" / "cancel").mkdir(parents=True)
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "tasks_dir": str(tmp_path / "tasks")})
    )
    kept = job / "cancel__23af19de"
    kept.mkdir()
    (kept / "result.json").write_text(json.dumps(_payload("cancel", "whole")))
    stopped = job / "cancel__8affcac8"
    stopped.mkdir()
    payload = _payload(
        "cancel",
        "whole",
        rewards=None,
        interrupted=True,
        started_at="2026-11-15 08:58:00.000000",
        finished_at="2026-11-15 08:59:00.000000",
    )
    payload["agent_result"]["n_cache_creation_tokens"] = None
    (stopped / "result.json").write_text(json.dumps(payload))
    receipt = {
        "version": 1,
        "reason": "interrupted",
        "timing": {"total": 25.6, "verifier_sandbox_total": 0},
        "agent_result": payload["agent_result"],
    }
    (stopped / "budget-cancelled.json").write_text(json.dumps(receipt))
    (job / "summary.json").write_text("{}")

    # summary.json's rule (evaluation.py): the kept results and every attempt
    # stopped before it finished.
    own = {"cancel": json.loads((kept / "result.json").read_text())}
    rows = {**own, **interrupted_trial_results(job)}
    expected = usage_summary(rows)["trials_without_cache_write_count"]
    assert expected == 1

    result = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["trials_without_cache_write_count"] == expected

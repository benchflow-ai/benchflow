"""Every reasoning effort a trial's requests used is recorded (benchflow#1246, J9).

With the effort pin off (the default) nothing is rewritten, so a sub-agent at
another level runs at it. ``model-calls.json`` then says which levels the
requests went upstream at and how many at each (``efforts_used``), in
BenchFlow's level names whichever wire carried them (Anthropic
``output_config.effort``, OpenAI ``reasoning_effort``, Responses
``reasoning.effort``); an Anthropic thinking budget with no level is the
agent's own and is kept as ``budget_tokens=N``. With the pin on, it says the levels as asked
(``efforts_requested``) beside those sent. The job's summary shows a line when
its requests used more than one level.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchflow.trajectories.model_calls import write_model_calls


def _trial(tmp_path: Path, run_effort: str | None, ends: list[dict]) -> Path:
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"reasoning_effort": run_effort}))
    row = {
        "request": {"method": "POST", "path": "/v1/messages", "body": {}},
        "response": {
            "status_code": 200,
            "body": {"model": "m", "stop_reason": "end_turn", "usage": {}},
        },
        "duration_ms": 1.0,
        "metadata": {"request_model": "m", "call_purpose": "agent"},
    }
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(json.dumps(row) + "\n")
    (trial / "trajectory" / "provider_requests.jsonl").write_text(
        "".join(
            json.dumps({"event": "request_end", "request_id": str(i), **end}) + "\n"
            for i, end in enumerate(ends)
        )
    )
    return trial


def test_unpinned_trial_records_every_level_its_requests_used(tmp_path):
    trial = _trial(
        tmp_path,
        "high",
        [
            *({"status": 200, "effort_asked": "high"} for _ in range(10)),
            {"status": 200, "effort_asked": "max"},
            {"status": 200, "effort_asked": "max"},
            # A thinking budget with no level is the agent's own, kept as is.
            {"status": 200, "effort_asked": "budget_tokens=1024"},
            {"status": 200, "effort_asked": "budget_tokens=3000"},
            # Refused before any provider: it used no level.
            {
                "status": 400,
                "effort_asked": "low",
                "refused": {"rule": "model-not-served", "model": "x"},
            },
            {"status": 200},
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["efforts_used"] == {
        "high": 10,
        "max": 2,
        "budget_tokens=1024": 1,
        "budget_tokens=3000": 1,
    }
    assert summary["efforts_requested"] == summary["efforts_used"]


def test_pinned_trial_records_the_levels_asked_and_sent(tmp_path):
    trial = _trial(
        tmp_path,
        "high",
        [
            {
                "status": 200,
                "effort_pin": "high",
                "effort_asked": "low",
                "effort_sent": "high",
            },
            {
                "status": 200,
                "effort_pin": "high",
                "effort_asked": "max",
                "effort_sent": "high",
            },
            {"status": 200, "effort_pin": "high", "effort_asked": "high"},
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["efforts_requested"] == {"low": 1, "max": 1, "high": 1}
    assert summary["efforts_used"] == {"high": 3}


def _job(tmp_path: Path, *figures: dict) -> SimpleNamespace:
    trials = []
    for i, figure in enumerate(figures):
        trial = tmp_path / f"trial-{i}"
        trial.mkdir(parents=True)
        (trial / "model-calls.json").write_text(
            json.dumps({"calls": 1, "source": "proxy", **figure})
        )
        trials.append(SimpleNamespace(path=trial))
    return SimpleNamespace(trials=trials)


def test_the_summary_shows_a_line_when_more_than_one_level_was_used(tmp_path):
    from benchflow.cli.main import _efforts_row

    row = _efforts_row(
        _job(
            tmp_path,
            {"efforts_used": {"high": 10, "max": 2}},
            {"efforts_used": {"high": 4}},
        )
    )
    assert row is not None
    assert row.startswith("high 14, max 2"), row
    assert _efforts_row(_job(tmp_path / "one", {"efforts_used": {"high": 3}})) is None


def test_the_run_card_after_the_run_shows_the_levels_when_there_are_several(
    tmp_path,
):
    from benchflow.run_checks import RunChecks, after_run_lines

    checks = RunChecks({}, 2)
    for i, used in enumerate(({"max": 3, "low": 2}, {"max": 1})):
        trial = tmp_path / f"trial-{i}"
        trial.mkdir()
        (trial / "result.json").write_text("{}")
        (trial / "model-calls.json").write_text(json.dumps({"efforts_used": used}))
        checks.record(f"trial-{i}", trial)
    lines = after_run_lines(checks.summary())
    assert [line for line in lines if line.startswith("  Efforts:")] == [
        "  Efforts:   requests went upstream at max (4), low (2): 2 levels; each "
        "trial's model-calls.json (efforts_used)"
    ], lines

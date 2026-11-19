"""A trial the backfill finds nothing to count keeps no 0 tokens (review of
PR #122, I2: missing is not 0).

0.10.0rc3 wrote 0 for every token field of a trial nothing counted: an
oracle (no model call), a native Codex run refused before launch
(``NativeRouteRefused``), a Gemini trial whose first call failed with a 400.
summary.json already said null for them; their result.json kept the zeros.
The backfill now writes them null, with ``no_usage_reason`` saying why
(``no_model_call``, or the reason it found nothing to count), the old record
under ``backfill.original``; a second run changes nothing.
"""

from __future__ import annotations

import json

import benchflow.cost_backfill as cb
from tests.test_cost_backfill import _agent_result, _job, _snapshot, _trial

TOKENS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_write_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)


def _rc3_zeros(job):
    oracle = _trial(
        job, "answer__oracle", agent="oracle", model="", agent_result=_agent_result()
    )
    refused = _trial(
        job,
        "answer__refused",
        agent="codex-acp",
        model="gpt-6.1-sol",
        harness="native",
        agent_result=_agent_result(),
        error="NativeRouteRefused: native Codex runs only through BenchFlow's proxy",
        error_category="agent_integration",
    )
    gemini = _trial(
        job,
        "answer__gemini",
        agent="gemini",
        model="gemini-4-argon",
        agent_result=_agent_result(),
        error='ACP error 400: {"error": {"message": "bad request"}}',
        error_category="api_error",
    )
    return oracle, refused, gemini


def _agent(trial):
    return json.loads((trial / "result.json").read_text())["agent_result"]


def test_zeros_nothing_counted_become_unknown_with_why(tmp_path):
    job = _job(tmp_path)
    oracle, refused, gemini = _rc3_zeros(job)

    report = cb.backfill_job(job)

    for trial, status, reason in (
        (oracle, "not_applicable", "no_model_call"),
        (refused, "uncovered", "no_logs"),
        (gemini, "uncovered", "no_logs"),
    ):
        line = next(line for line in report["trials"] if line["trial"] == trial.name)
        assert (line["status"], line["reason"]) == (status, reason)
        assert line["changes_record"] is True
        after = _agent(trial)
        assert {key: after[key] for key in TOKENS} == dict.fromkeys(TOKENS)
        assert after["no_usage_reason"] == reason
        assert after["usage_source"] is None
        assert after["usage_source_legacy"] == "unavailable"
        assert after["cost_usd"] is None
        assert after["backfill"]["original"]["total_tokens"] == 0
        final = json.loads((trial / "result.json").read_text())["final_metrics"]
        assert final["total_prompt_tokens"] is None
        assert final["total_completion_tokens"] is None
    summary = json.loads((job / "summary.json").read_text())
    assert summary["total_tokens"] is None
    assert summary["trials_without_model_call"] == 1


def test_a_second_run_changes_nothing(tmp_path):
    job = _job(tmp_path)
    _rc3_zeros(job)
    cb.backfill_job(job)
    first = _snapshot(job)

    report = cb.backfill_job(job)

    assert _snapshot(job) == first
    assert report["job"]["files_written"] == []
    assert not any(line.get("changes_record") for line in report["trials"])


def test_a_counted_zero_is_kept(tmp_path):
    """A 0 a trusted source counted (an agent's end-of-turn report) is left
    as it is."""
    job = _job(tmp_path)
    trial = _trial(
        job,
        "answer__reported",
        agent="gemini",
        model="gemini-4-argon",
        agent_result=_agent_result(usage_source="agent_native_acp"),
    )
    cb.backfill_job(job)
    assert _agent(trial)["total_tokens"] == 0

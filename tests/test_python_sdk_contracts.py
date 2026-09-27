"""Contract fields that viewers and notebooks read.

Each field is optional and bumps the document's
``schema_minor``:

- trial 1.1: ``usage.cost_status`` (priced / subscription / unpriced /
  unavailable, so a blank cost is explained),
  ``sandbox`` (sandbox.json: id, provider, created),
  ``verifier.reward_details`` (reward-details.json).
- job 1.1: ``groups`` (denominators per agent and model; the notebook facet
  table), ``interrupted`` (attempt folders without result.json, flagged when
  a sandbox was created), ``error_categories`` and
  ``timing_totals``.
- comparison 1.1: ``by``, ``rows[].group`` and ``a_paired``/``b_paired``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow as bf
from benchflow import job_export
from tests.test_python_sdk_load_job import _job, _trial

jsonschema = pytest.importorskip("jsonschema")
SCHEMAS = Path(__file__).resolve().parents[1] / "docs/reference/schemas"


def _valid(doc: dict, kind: str) -> dict:
    schema = json.loads((SCHEMAS / job_export.schema_filename(kind)).read_text())
    jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)
    return doc


def _set(trial: Path, **fields) -> None:
    result = json.loads((trial / "result.json").read_text())
    result.update(fields)
    (trial / "result.json").write_text(json.dumps(result))


def test_cost_status_explains_a_missing_cost(tmp_path: Path) -> None:
    job = tmp_path / "j"
    priced = _trial(job, "a", cost=0.02)
    subscription = _trial(job, "b", cost=None)
    _set(
        subscription,
        agent_result={"total_tokens": 9, "usage_source": "agent_native_acp"},
    )
    unpriced = _trial(job, "c", cost=None)
    _set(
        unpriced, agent_result={"total_tokens": 9, "usage_source": "provider_response"}
    )
    unavailable = _trial(job, "d", cost=None)
    _set(unavailable, agent_result={"usage_source": "unavailable"})
    status = {
        t.name: _valid(bf.load_trial(t).to_json_dict(), "trial")["usage"]["cost_status"]
        for t in (priced, subscription, unpriced, unavailable)
    }
    assert list(status.values()) == [
        "priced",
        "subscription",
        "unpriced",
        "unavailable",
    ]


def test_the_trial_carries_its_sandbox_and_reward_details(tmp_path: Path) -> None:
    trial = _trial(tmp_path / "j", "a")
    (trial / "sandbox.json").write_text(
        json.dumps(
            {
                "sandbox_id": "00000001",
                "provider": "DaytonaSandbox",
                "created_at": "2026-01-01 12:00:00.000000",
            }
        )
    )
    (trial / "verifier" / "reward-details.json").write_text(
        json.dumps({"checks": [{"name": "hello", "passed": False}]})
    )
    doc = _valid(bf.load_trial(trial).to_json_dict(), "trial")
    assert doc["schema_minor"] >= 1
    assert doc["sandbox"] == {
        "sandbox_id": "00000001",
        "provider": "DaytonaSandbox",
        "created_at": "2026-01-01 12:00:00.000000",
    }
    assert doc["verifier"]["reward_details"]["checks"][0]["passed"] is False
    bare = _valid(bf.load_trial(_trial(tmp_path / "k", "a")).to_json_dict(), "trial")
    assert bare["sandbox"] is None and bare["verifier"]["reward_details"] is None


def test_the_job_document_has_groups_interruptions_errors_and_time(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, "a", {"t1": 1.0, "t2": None})
    _trial(
        job,
        "t3",
        agent="codex-acp",
        model="gpt-5.5",
        reward=0.0,
        error="agent timed out",
        error_category="timeout",
    )
    # An attempt that crashed after its sandbox was created, and one before.
    crashed = job / "t4__deadbeef"
    crashed.mkdir()
    (crashed / "config.json").write_text(
        json.dumps(
            {
                "agent": "codex-acp",
                "model": "gpt-5.5",
                "task_path": "/tasks/t4",
                "started_at": "2026-01-01 12:00:00",
            }
        )
    )
    (crashed / "sandbox.json").write_text(json.dumps({"sandbox_id": "sb-1"}))
    early = job / "t5__cafef00d"
    early.mkdir()
    (early / "config.json").write_text(json.dumps({"agent": "codex-acp"}))

    doc = _valid(bf.load_job(job).to_json_dict(), "job")
    assert doc["schema_minor"] == 1
    groups = {(g["agent"], g["model"]): g["denominators"] for g in doc["groups"]}
    assert set(groups) == {
        ("claude-agent-acp", "claude-haiku-4-5"),
        ("codex-acp", "gpt-5.5"),
    }
    assert groups[("codex-acp", "gpt-5.5")]["attempted"] == 1
    assert doc["interrupted"] == [
        {
            "path": str(crashed),
            "task_name": "t4",
            "agent": "codex-acp",
            "model": "gpt-5.5",
            "started_at": "2026-01-01 12:00:00",
            "sandbox_id": "sb-1",
        },
        {
            "path": str(early),
            "task_name": "t5",
            "agent": "codex-acp",
            "model": None,
            "started_at": None,
            "sandbox_id": None,
        },
    ]
    assert doc["error_categories"] == {"timeout": 1, "unknown": 1}
    assert doc["timing_totals"] == {"agent": 180.0, "total": 270.0}
    assert bf.load_job(job).interrupted == [crashed, early]


def test_the_comparison_document_has_groups_and_paired_denominators(
    tmp_path: Path,
) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    b = _job(tmp_path, "b", {"t1": 0.0})
    doc = _valid(bf.compare(a, b, by=("agent",)).to_json_dict(), "comparison")
    assert doc["schema_minor"] == 1 and doc["by"] == ["agent"]
    row = next(r for r in doc["rows"] if r["task"] == "t1")
    assert row["group"] == {"agent": "claude-agent-acp"}
    assert doc["a_paired"]["attempted"] == 1 and doc["b_paired"]["attempted"] == 1
    plain = _valid(bf.compare(a, b).to_json_dict(), "comparison")
    assert plain["by"] == [] and plain["rows"][0]["group"] == {}

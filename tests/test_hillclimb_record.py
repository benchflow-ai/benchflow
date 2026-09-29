"""``hillclimb.json``'s committed JSON Schema stays current and forward-compatible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.hillclimbing import record

jsonschema = pytest.importorskip("jsonschema")

SCHEMAS = Path(__file__).resolve().parents[1] / "docs/reference/schemas"


def _schema() -> dict:
    return json.loads((SCHEMAS / record.SCHEMA_FILENAME).read_text())


def test_the_committed_schema_file_is_current():
    """Regenerate with ``python -m benchflow.hillclimbing.record docs/reference/schemas``."""
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema == json.loads(json.dumps(record.json_schema()))


def test_the_schema_accepts_fields_it_does_not_know():
    """A newer writer's optional fields must not fail an older reader
    (docs/reference/json-export.md: a new optional field keeps the version)."""
    assert '"additionalProperties": false' not in json.dumps(_schema())


def _minimal() -> dict:
    config = {
        "tasks": ["/t/a"],
        "surfaces": [{"kind": "skills", "source": "/s", "name": "skills"}],
        "agent": "claude-agent-acp",
        "model": "m",
        "reasoning_effort": None,
        "environment": "docker",
        "concurrency": 4,
        "objective": "score",
        "rounds": 5,
        "trials": 2,
        "min_gain": 0.05,
        "max_cost_usd": None,
        "stall_rounds": 3,
        "candidates": 1,
        "max_infra_error_rate": 0.25,
        "leak_check": "reject",
        "bootstrap_samples": 2000,
        "seed": 0,
        "force": False,
        "proposer": {
            "agent": "claude-agent-acp",
            "model": None,
            "reasoning_effort": None,
            "environment": "docker",
            "timeout_sec": 1800,
            "image": "python",
            "open_network": False,
            "max_failures": 24,
        },
    }
    return {
        "kind": "benchflow.hillclimb",
        "schema_version": 1,
        "status": "running",
        "benchflow_version": "0",
        "created_at": "2026-09-29T00:00:00+00:00",
        "updated_at": "2026-09-29T00:00:00+00:00",
        "config": config,
        "split": {
            "method": "random",
            "seed": 0,
            "test_frac": 0.3,
            "stratify_by": None,
            "train": ["a"],
            "test": ["b"],
        },
        "cost": {
            "total_usd": 0.0,
            "agent_usd": 0.0,
            "proposer_usd": 0.0,
            "usd_unknown_rollouts": 0,
            "max_cost_usd": None,
        },
        "paths": {
            "report": "report.html",
            "split": "split.json",
            "surfaces": "surfaces",
            "surface_history": None,
            "evals": "evals",
            "proposer": "proposer",
        },
        "rounds": [],
    }


def test_a_document_round_trips_through_the_models_and_the_schema():
    raw = _minimal()
    doc = record.HillclimbDoc.model_validate(raw)
    dumped = record.dump(doc)
    jsonschema.validate(dumped, _schema(), cls=jsonschema.Draft202012Validator)
    # Unknown fields: the models refuse to build them, the schema tolerates them.
    with pytest.raises(ValueError):
        record.HillclimbDoc.model_validate({**raw, "surprise": 1})
    jsonschema.validate({**dumped, "surprise": 1}, _schema())
    # The schema refuses another kind or version.
    for key, value in (("kind", "benchflow.job"), ("schema_version", 2)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**dumped, key: value}, _schema())


def test_write_record_is_atomic_and_readable(tmp_path):
    doc = record.HillclimbDoc.model_validate(_minimal())
    path = record.write_record(doc, tmp_path)
    assert path.name == "hillclimb.json"
    assert record.load_record(tmp_path) == doc
    assert [p.name for p in tmp_path.iterdir()] == ["hillclimb.json"]

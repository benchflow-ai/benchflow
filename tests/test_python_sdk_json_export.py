"""A stable, versioned JSON export of load_trial/load_job/compare.

Viewers and other tools read BenchFlow
results through this format rather than the on-disk layout. The documents are
built from pydantic models that also generate the JSON Schemas committed in
docs/reference/schemas/; these tests keep the files current and validate real
exports against them with an independent validator (jsonschema).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import benchflow as bf
from benchflow import job_export
from tests.test_python_sdk_load_job import _job, _tree, _trial
from tests.test_python_sdk_load_job_more import _jsonl_job, _review

# jsonschema is present through litellm, not a declared dependency.
jsonschema = pytest.importorskip("jsonschema")

SCHEMAS = Path(__file__).resolve().parents[1] / "docs/reference/schemas"


def _validate(document: dict, kind: str) -> None:
    schema = json.loads((SCHEMAS / job_export.schema_filename(kind)).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(document, schema, cls=jsonschema.Draft202012Validator)
    json.dumps(document, allow_nan=False)  # strict JSON, no NaN/inf


@pytest.mark.parametrize("kind", ["trial", "job", "comparison", "run-summary"])
def test_the_committed_schema_files_are_current(kind: str) -> None:
    path = SCHEMAS / job_export.schema_filename(kind)
    assert json.loads(path.read_text()) == job_export.json_schema(kind)  # type: ignore[arg-type]


def test_a_full_trial_exports_and_validates(tmp_path: Path) -> None:
    d = _trial(tmp_path / "jobs" / "j", "hello")
    _tree(d)
    _review(tmp_path / "jobs", d)
    doc = bf.load_trial(d).to_json_dict()
    _validate(doc, "trial")
    assert doc["kind"] == "benchflow.trial" and doc["schema_version"] == 1
    assert doc["reward"] == 1.0 and doc["assessment"] == "scored"
    assert doc["forks"][0]["children"][1]["label"] == "hint"
    assert doc["review"]["reviewer_model"] == "gpt-5.5"
    assert doc["verifier"]["stdout"].startswith("1 passed")
    assert doc["trajectory"][0]["type"] == "tool_call"
    assert doc["usage"]["cost_usd"] == 0.01
    slim = bf.load_trial(d).to_json_dict(
        include_trajectory=False, include_verifier=False
    )
    assert slim["trajectory"] is None and slim["verifier"] is None
    _validate(slim, "trial")


def test_jobs_and_comparisons_export_and_validate(tmp_path: Path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": None})
    b = _job(tmp_path, "b", {"t1": 0.0, "t2": 1.0})
    doc = bf.load_job(a).to_json_dict()
    _validate(doc, "job")
    assert doc["denominators"]["attempted"] == 2
    assert doc["denominators_with_controls"]["attempted"] == 4
    assert all(t["trajectory"] is None for t in doc["trials"])
    cmp = bf.compare(a, b, labels=("a", "b")).to_json_dict()
    _validate(cmp, "comparison")
    assert cmp["a"]["label"] == "a" and cmp["summary"]["paired"] == 2
    rows = {r["task"]: r for r in cmp["rows"]}
    assert rows["t1"]["delta"] == -1.0


def test_results_jsonl_trials_export_and_validate(tmp_path: Path) -> None:
    doc = bf.load_job(_jsonl_job(tmp_path / "exported")).to_json_dict()
    _validate(doc, "job")
    assert {t["source"] for t in doc["trials"]} == {"results.jsonl"}


def test_non_finite_values_become_null(tmp_path: Path) -> None:
    d = _trial(tmp_path / "job", "odd")
    data = json.loads((d / "timing.json").read_text())
    data["agent"] = math.nan
    (d / "timing.json").write_text(json.dumps(data))
    doc = bf.load_trial(d).to_json_dict()
    assert doc["timing"]["agent"] is None
    _validate(doc, "trial")


def test_to_json_writes_a_file(tmp_path: Path) -> None:
    job = bf.load_job(_job(tmp_path, "a", {"t1": 1.0}))
    out = job.to_json(tmp_path / "out" / "job.json")
    assert json.loads(out.read_text())["kind"] == "benchflow.job"


def test_fork_kind_and_costs_are_exported(tmp_path: Path) -> None:
    """Fork kind (fork / retry), parent_node and cost,
    and a per-child cost are in tree.json and Fork; the export carries them as
    optional v1 fields."""
    d = _trial(tmp_path / "job", "hello")
    _tree(d)
    tree = json.loads((d / "tree.json").read_text())
    fork = tree["forks"][0]
    fork.update(kind="retry", parent_node="n5", cost={"usd": 0.02, "tokens": 900})
    fork["children"][0]["cost"] = {"usd": 0.01}
    (d / "tree.json").write_text(json.dumps(tree))
    doc = bf.load_trial(d).to_json_dict()
    _validate(doc, "trial")
    f = doc["forks"][0]
    assert (f["kind"], f["parent_node"], f["cost"]["usd"]) == ("retry", "n5", 0.02)
    assert (
        f["children"][0]["cost"] == {"usd": 0.01} and f["children"][1]["cost"] is None
    )


def _with_later_fields(document: object, schema: dict, root: dict) -> object:
    """``document`` with a new optional field in every record object (a
    schema node with ``properties``), as a later minor release of the same
    schema_version may write. Maps (``additionalProperties`` schemas) keep
    their typed values."""
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    for branch in schema.get("anyOf") or schema.get("oneOf") or []:
        if isinstance(document, dict) and ("properties" in branch or "$ref" in branch):
            return _with_later_fields(document, branch, root)
        if isinstance(document, list) and branch.get("type") == "array":
            return _with_later_fields(document, branch, root)
    if isinstance(document, dict) and "properties" in schema:
        props = schema["properties"]
        out = {
            k: _with_later_fields(v, props[k], root) if k in props else v
            for k, v in document.items()
        }
        return {**out, "added_in_a_later_minor": {"any": "value"}}
    if isinstance(document, list) and isinstance(schema.get("items"), dict):
        return [_with_later_fields(v, schema["items"], root) for v in document]
    return document


def test_a_later_minor_still_validates_against_the_committed_schemas(
    tmp_path: Path,
) -> None:
    """A reader's copy of a v1 schema accepts optional fields added later.

    The docs promise that a new optional field keeps ``schema_version``;
    guards the dx/sdk fix of the schemas committed since bf6e8412 (SDK
    update), which set ``additionalProperties: false`` at every level, so a
    reader validating with them refused every document that gained a field.
    """
    from benchflow.trajectories import rollout_stream as rs
    from tests.trajectories.test_rollout_stream import TITO_CALLS, write_rollout

    d = _trial(tmp_path / "jobs" / "j", "hello")
    _tree(d)
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": None})
    b = _job(tmp_path, "b", {"t1": 0.0, "t2": 1.0})
    documents = {
        "trial": bf.load_trial(d).to_json_dict(),
        "job": bf.load_job(a).to_json_dict(),
        "comparison": bf.compare(a, b).to_json_dict(),
    }
    for kind, document in documents.items():
        schema = json.loads((SCHEMAS / job_export.schema_filename(kind)).read_text())
        later = _with_later_fields(document, schema, schema)
        assert later != document
        _validate(later, kind)  # type: ignore[arg-type]

    stream_job = tmp_path / "stream"
    write_rollout(stream_job, "hello__a", calls=TITO_CALLS)
    committed = json.loads((SCHEMAS / rs.SCHEMA_FILE).read_text())
    for record in rs.stream_rollouts(stream_job, follow=False):
        later = _with_later_fields(record.to_json_dict(), committed, committed)
        assert later != record.to_json_dict()
        jsonschema.validate(later, committed, cls=jsonschema.Draft202012Validator)

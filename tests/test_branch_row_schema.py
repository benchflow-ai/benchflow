"""JSON Schemas for branch-tree rows.

``docs/reference/schemas/`` had job, trial and comparison schemas but none
for the ``branch_child`` / ``branch_pair`` rows. They are committed now, kept
current by this test, and real exported rows validate against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.trajectories import branch_row_schema
from benchflow.trajectories.export_branch import export_branch_jsonl
from tests.test_export_branch_tree_v2 import PARENT, _say, _tool, _user, _write_trial

jsonschema = pytest.importorskip("jsonschema")

SCHEMAS = Path(__file__).resolve().parents[1] / "docs/reference/schemas"


@pytest.mark.parametrize("name", sorted(branch_row_schema.SCHEMAS))
def test_the_committed_schema_files_are_current(name):
    schema = json.loads((SCHEMAS / name).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema == json.loads(json.dumps(branch_row_schema.SCHEMAS[name]))


def test_exported_rows_validate(tmp_path):
    _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[
            (
                "n3",
                "a",
                1.0,
                [
                    _user("Go."),
                    _tool("c1", "Write x", "edit", {"file_path": "x"}, "ok"),
                    _say("Yes."),
                ],
            ),
            ("n4", "b", 0.0, [_user("Go."), _say("No.")]),
            ("n5", "c", 0.5, [_user("Other."), _say("Maybe.")]),
        ],
    )
    out, pairs = tmp_path / "c.jsonl", tmp_path / "p.jsonl"
    export_branch_jsonl(tmp_path / "job", out, pairs_out=pairs, any_request_pairs=True)
    for path, name in (
        (out, "benchflow-branch-child-row.v2.schema.json"),
        (pairs, "benchflow-branch-pair-row.v2.schema.json"),
    ):
        schema = json.loads((SCHEMAS / name).read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows
        for row in rows:
            jsonschema.validate(row, schema, cls=jsonschema.Draft202012Validator)

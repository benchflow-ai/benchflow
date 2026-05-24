"""Non-finite float handling in trainer JSONL export.

``json.dumps`` happily emits the bare tokens ``NaN``, ``Infinity``, and
``-Infinity`` for non-finite floats — accepted by Python's permissive
parser, rejected by strict JSON parsers (jq, serde, Node ``JSON.parse``).
The trainer export normalizes non-finite floats to ``null`` and passes
``allow_nan=False`` so any regression raises rather than silently
producing invalid JSONL. See issue #409.
"""

from __future__ import annotations

import json
import math

import pytest

from benchflow.trajectories.export import export_trajectories_to_jsonl


def test_nan_reward_emits_null_not_nan_token(tmp_path):
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl([{"example_id": 1, "reward": math.nan}], out)
    raw = out.read_text()
    # Bare NaN token would be invalid JSON for strict parsers.
    assert "NaN" not in raw
    assert "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["example_id"] == 1
    assert parsed["reward"] is None


def test_positive_infinity_metric_becomes_null(tmp_path):
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl(
        [{"example_id": 2, "metrics": {"latency_ms": math.inf}}], out
    )
    parsed = json.loads(out.read_text())
    assert parsed["metrics"]["latency_ms"] is None


def test_negative_infinity_in_nested_list_becomes_null(tmp_path):
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl(
        [{"example_id": 3, "info": {"values": [1.0, -math.inf, 2.5]}}], out
    )
    parsed = json.loads(out.read_text())
    assert parsed["info"]["values"] == [1.0, None, 2.5]


def test_finite_floats_pass_through_unchanged(tmp_path):
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl(
        [{"example_id": 4, "reward": 0.75, "metrics": {"a": 0.0, "b": -1.5}}],
        out,
    )
    parsed = json.loads(out.read_text())
    assert parsed["reward"] == 0.75
    assert parsed["metrics"] == {"a": 0.0, "b": -1.5}


def test_emitted_lines_are_strict_json(tmp_path):
    """``json.loads`` with default args is strict-enough for the regression."""
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl(
        [
            {"example_id": 0, "reward": math.nan},
            {"example_id": 1, "reward": 1.0, "metrics": {"x": math.inf}},
        ],
        out,
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        # Would raise on bare NaN / Infinity tokens — that's the bug we
        # are fixing.
        json.loads(line)


def test_scrubbed_record_does_not_mutate_input(tmp_path):
    rec = {"example_id": 5, "reward": math.nan, "metrics": {"x": math.inf}}
    export_trajectories_to_jsonl([rec], tmp_path / "out.jsonl")
    # Caller's dict is unchanged — scrubbing returns a fresh structure.
    assert math.isnan(rec["reward"])
    assert math.isinf(rec["metrics"]["x"])


def test_allow_nan_false_is_active_defense(monkeypatch, tmp_path):
    """If the scrubber is ever bypassed, serialization must still raise.

    Patch ``_scrub_non_finite`` to a no-op so a NaN reaches ``json.dumps``;
    ``allow_nan=False`` should turn that into ``ValueError`` rather than
    silently writing an invalid JSONL line.
    """
    from benchflow.trajectories import export as export_mod

    monkeypatch.setattr(export_mod, "_scrub_non_finite", lambda v: v)
    with pytest.raises(ValueError):
        export_trajectories_to_jsonl(
            [{"example_id": 6, "reward": math.nan}], tmp_path / "out.jsonl"
        )

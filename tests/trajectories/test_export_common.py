"""Shared export plumbing: content rendering, thought buffering, aggregation."""

import json
import logging

from benchflow.trajectories._export_common import (
    ThoughtBuffer,
    aggregate_rollout_jsonl,
    content_blocks_to_text,
)


def test_content_blocks_string_passes_through():
    assert content_blocks_to_text("plain output") == "plain output"


def test_content_blocks_non_list_yields_empty():
    assert content_blocks_to_text(None) == ""
    assert content_blocks_to_text({"text": "x"}) == ""


def test_content_blocks_mixed_shapes_joined():
    blocks = [
        {"text": "flat"},
        {"type": "content", "content": {"type": "text", "text": "nested"}},
        {"type": "image", "data": "..."},
        "not-a-dict",
    ]
    assert content_blocks_to_text(blocks) == "flat\nnested"


def test_thought_buffer_joins_then_clears():
    buf = ThoughtBuffer()
    assert buf.take() is None
    buf.push("first")
    buf.push("second")
    assert buf.take() == "first\n\nsecond"
    assert buf.take() is None


def _write_artifact(job_dir, rollout, text):
    path = job_dir / rollout / "trainer" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(text)


def test_aggregate_normalizes_missing_trailing_newline(tmp_path):
    _write_artifact(tmp_path, "r1", '{"id": 1}')
    _write_artifact(tmp_path, "r2", '{"id": 2}\n')
    out = aggregate_rollout_jsonl(
        tmp_path,
        rollout_relpath="trainer/records.jsonl",
        out_filename="records.jsonl",
    )
    assert out == tmp_path / "records.jsonl"
    lines = out.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [1, 2]


def test_aggregate_skips_unreadable_artifact(tmp_path, caplog):
    # A directory where the artifact file should be: read_text raises OSError.
    (tmp_path / "r1" / "trainer" / "records.jsonl").mkdir(parents=True)
    _write_artifact(tmp_path, "r2", '{"id": 2}\n')
    with caplog.at_level(logging.WARNING):
        out = aggregate_rollout_jsonl(
            tmp_path,
            rollout_relpath="trainer/records.jsonl",
            out_filename="records.jsonl",
        )
    assert out is not None
    lines = out.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [2]
    assert "Skipping unreadable trainer artifact" in caplog.text


def test_aggregate_returns_none_without_rollouts(tmp_path):
    assert (
        aggregate_rollout_jsonl(
            tmp_path, rollout_relpath="trainer/records.jsonl", out_filename="r.jsonl"
        )
        is None
    )
    assert (
        aggregate_rollout_jsonl(
            tmp_path / "missing",
            rollout_relpath="trainer/records.jsonl",
            out_filename="r.jsonl",
        )
        is None
    )
    assert not (tmp_path / "r.jsonl").exists()

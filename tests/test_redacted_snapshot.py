"""Guards capture caching against full re-redaction in commit 4196c4a."""

from __future__ import annotations

import copy
import json
from unittest.mock import Mock

import pytest

from benchflow.trajectories import _snapshot
from benchflow.trajectories._capture import TrajectoryWriter
from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from benchflow.trajectories._snapshot import RedactedJSONLSnapshot
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
    redact_acp_trajectory_jsonl,
)


@pytest.fixture
def redactor(monkeypatch):
    spy = Mock(wraps=_snapshot.redact_trajectory_obj)
    monkeypatch.setattr(_snapshot, "redact_trajectory_obj", spy)
    return spy


def test_snapshot_reuses_equal_records_and_detects_nested_same_length_edit(redactor):
    """Guards 4196c4a: streamed updates must not rescan unchanged tool output."""
    snapshot = RedactedJSONLSnapshot()
    records = [
        {"type": "tool_call", "content": [{"text": "large prior output"}]},
        {"type": "agent_message", "text": "first"},
    ]
    assert snapshot.serialize(records) == redact_acp_trajectory_jsonl(records)
    assert redactor.call_count == 2
    assert snapshot.serialize(copy.deepcopy(records)) == redact_acp_trajectory_jsonl(
        records
    )
    assert redactor.call_count == 2

    records[0]["content"][0]["text"] = "TOKEN=hidden-data!"
    assert snapshot.serialize(records) == redact_acp_trajectory_jsonl(records)
    assert redactor.call_count == 3
    assert "hidden-data!" not in snapshot.serialize(records)


def test_snapshot_invalidates_shared_objects_reorder_truncation_and_new_secrets(
    redactor,
):
    """Guards 4196c4a: immutable keys prevent stale or aliased redacted rows."""
    snapshot = RedactedJSONLSnapshot()
    shared = {"api_key": "first-synthetic-value"}
    records = [shared, shared, {"text": "last"}]
    snapshot.serialize(records)

    shared["api_key"] = "other-synthetic-value"
    payload = snapshot.serialize(records)
    assert payload == redact_acp_trajectory_jsonl(records)
    assert redactor.call_count == 5
    records.reverse()
    assert snapshot.serialize(records) == redact_acp_trajectory_jsonl(records)
    assert redactor.call_count == 7

    records.clear()
    assert snapshot.serialize(records) == ""
    assert snapshot._entries == []
    records.append({"raw_output": {"authorization": "Bearer new-synthetic-value"}})
    payload = snapshot.serialize(records)
    assert payload == redact_acp_trajectory_jsonl(records)
    assert "new-synthetic-value" not in payload
    assert redactor.call_count == 8


def test_snapshot_preserves_canonical_normalization_and_escaped_secrets(tmp_path):
    """Guards 4196c4a and PR #849: caching must retain valid escaped JSON."""
    records = [
        {
            "path": tmp_path / "artifact.txt",
            "raw_output": json.dumps({"api_key": "synthetic-value"}),
            "text": 'raise ValueError("TOKEN=synthetic\\\\value")',
        }
    ]
    snapshot = RedactedJSONLSnapshot()
    for _ in range(2):
        payload = snapshot.serialize(records)
        assert payload == redact_acp_trajectory_jsonl(records)
        assert json.loads(payload)["path"] == str(tmp_path / "artifact.txt")
        assert "synthetic-value" not in payload


def test_snapshot_failed_serialization_does_not_commit_partial_cache(monkeypatch):
    """Guards 4196c4a: a failed update must retain the last complete cache."""
    snapshot = RedactedJSONLSnapshot()
    records = [{"text": "old one"}, {"text": "old two"}]
    previous = snapshot.serialize(records)
    original = _snapshot.redact_trajectory_obj

    def fail_second(record):
        if record["text"] == "new two":
            raise ValueError("synthetic redaction failure")
        return original(record)

    monkeypatch.setattr(_snapshot, "redact_trajectory_obj", fail_second)
    with pytest.raises(ValueError, match="synthetic redaction failure"):
        snapshot.serialize([{"text": "new one"}, {"text": "new two"}])
    spy = Mock(wraps=original)
    monkeypatch.setattr(_snapshot, "redact_trajectory_obj", spy)
    assert snapshot.serialize(records) == previous
    spy.assert_not_called()


def test_acp_writer_updates_tool_completion_and_forces_complete_final_write(
    tmp_path, redactor
):
    """Guards 4196c4a: cached ACP updates and final writes keep complete evidence."""
    path = tmp_path / "acp_trajectory.jsonl"
    writer = TrajectoryWriter(path)
    records = [
        {"type": "user_message", "text": "task"},
        {"type": "tool_call", "status": "in_progress", "content": []},
    ]
    writer.write_events(records)
    writer.write_events(records)
    assert redactor.call_count == 2
    records[1]["status"] = "completed"
    records[1]["content"].append({"api_key": "new-synthetic-secret"})
    writer.write_events(records)
    assert redactor.call_count == 3
    assert path.read_text() == redact_acp_trajectory_jsonl(records)
    assert "new-synthetic-secret" not in path.read_text()

    path.write_text("external incomplete write")
    writer.write_final(records)
    assert path.read_text() == redact_acp_trajectory_jsonl(records)
    assert redactor.call_count == 3
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_llm_writer_reuses_exchanges_and_reconciles_nested_changes(tmp_path, redactor):
    """Guards 4196c4a: new LLM exchanges must not re-redact the whole history."""
    path = tmp_path / "llm_trajectory.jsonl"
    writer = LiveLLMTrajectoryWriter(path)
    first = LLMExchange(
        request=LLMRequest(body={"messages": [{"content": "question"}]}),
        response=LLMResponse(body={"content": "first answer"}),
    )
    trajectory = Trajectory(session_id="cache-test", exchanges=[first])
    assert writer.write(trajectory)
    assert not writer.write(trajectory)
    assert redactor.call_count == 1
    trajectory.exchanges.append(first.model_copy(deep=True))
    assert writer.write(trajectory)
    assert redactor.call_count == 2
    trajectory.exchanges[1].response.body["content"] = "TOKEN=synthetic-secret"
    assert writer.reconcile(trajectory)
    assert redactor.call_count == 3
    assert path.read_text() == trajectory.to_jsonl(redact_keys=True)
    assert "synthetic-secret" not in path.read_text()
    assert first.response.body["content"] == "first answer"

    trajectory.exchanges.reverse()
    assert writer.reconcile(trajectory)
    trajectory.exchanges.pop()
    assert writer.reconcile(trajectory)
    assert path.read_text() == trajectory.to_jsonl(redact_keys=True)
    assert len(path.read_text().splitlines()) == 1
    assert not path.with_suffix(".jsonl.tmp").exists()


@pytest.mark.parametrize("kind", ["acp", "llm"])
def test_writer_retries_failed_atomic_replace_using_cached_records(
    kind, tmp_path, monkeypatch, redactor
):
    """Guards 4196c4a: a cached failed write must still publish on retry."""
    path = tmp_path / f"{kind}_trajectory.jsonl"
    content = {"text": "original output"}
    if kind == "acp":
        writer = TrajectoryWriter(path)
        records = [{"type": "tool_call", "raw_output": content}]

        def write():
            writer.write_events(records)

        def expected():
            return redact_acp_trajectory_jsonl(records)

    else:
        writer = LiveLLMTrajectoryWriter(path)
        exchange = LLMExchange(request=LLMRequest(), response=LLMResponse(body=content))
        content = exchange.response.body
        trajectory = Trajectory(session_id="retry-test", exchanges=[exchange])

        def write():
            writer.write(trajectory)

        def expected():
            return trajectory.to_jsonl(redact_keys=True)

    write()
    previous = path.read_text()
    content["text"] = "TOKEN=new-synthetic-secret"

    def fail_replace(*args):
        raise OSError("synthetic atomic replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr("os.replace", fail_replace)
        with pytest.raises(OSError, match="synthetic atomic replacement failure"):
            write()
    assert path.read_text() == previous
    assert json.loads(previous)
    assert redactor.call_count == 2

    write()
    assert path.read_text() == expected()
    assert "new-synthetic-secret" not in path.read_text()
    assert redactor.call_count == 2
    assert not path.with_suffix(".jsonl.tmp").exists()

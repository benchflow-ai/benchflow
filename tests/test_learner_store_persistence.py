"""Persistence tests for the LearnerStore — guards issue #394.

A continual-learning (``sequential-shared``) Job must survive process death:
the LearnerStore is the agent's accumulating memory and skills, and a resumed
job that silently starts from an empty store would mix old result rows with
new ones whose learning curve never inherited the earlier rollouts' work.

These tests pin the serialization round-trip and the atomic-save invariant
the orchestrator depends on.
"""

from __future__ import annotations

import json

import pytest

from benchflow.learner_store import LearnerState, LearnerStore


def test_to_dict_round_trips_through_from_dict():
    """A snapshot must rebuild a byte-identical store — same generation, same
    history, same next-number counter."""
    store = LearnerStore()
    store.commit(LearnerState(memory={"a": 1}, skills={"s1": "v1"}), metric=0.5)
    store.commit(LearnerState(memory={"a": 2}, skills={"s1": "v2"}), metric=0.8)

    restored = LearnerStore.from_dict(store.to_dict())

    assert restored.generation == store.generation
    assert restored.learning_curve() == store.learning_curve()
    assert restored.current().memory == {"a": 2}
    assert restored.current().skills == {"s1": "v2"}
    # next_number must survive so a revert + commit on the restored store
    # never reuses a number from a dropped generation.
    assert restored._next_number == store._next_number


def test_save_and_load_through_disk(tmp_path):
    """``LearnerStore.save`` then ``LearnerStore.load`` must rebuild the same
    history end-to-end — the orchestrator depends on this for resume (#394)."""
    store = LearnerStore()
    store.commit(LearnerState(skills={"k": "v"}), metric=1.0)
    snapshot = tmp_path / "snap" / "learner_store.json"

    store.save(snapshot)
    assert snapshot.is_file(), "save must create the parent directory"

    loaded = LearnerStore.load(snapshot)
    assert loaded.generation == 1
    assert loaded.learning_curve() == [1.0]
    assert loaded.current().skills == {"k": "v"}


def test_save_is_atomic_under_crash(tmp_path, monkeypatch):
    """A crash mid-save must leave the previous snapshot intact — a partial
    JSON file would silently corrupt the next resume."""
    snapshot = tmp_path / "learner_store.json"

    good = LearnerStore()
    good.commit(LearnerState(skills={"keep": "me"}), metric=0.9)
    good.save(snapshot)
    original = snapshot.read_text()

    # Simulate a crash *after* the temp file is written but *before* the
    # rename completes — the live snapshot must still be the prior one.
    from pathlib import Path

    real_replace = Path.replace

    def boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)

    later = LearnerStore()
    later.commit(LearnerState(skills={"newer": "skill"}), metric=0.95)
    with pytest.raises(OSError):
        later.save(snapshot)

    # Restore real behavior for the assert read.
    monkeypatch.setattr(Path, "replace", real_replace)
    assert snapshot.read_text() == original, (
        "atomic-save invariant broken: a failed save corrupted the snapshot"
    )


def test_from_dict_rejects_unknown_version():
    """An unknown snapshot version must hard-fail rather than silently load
    a partial or future format."""
    with pytest.raises(ValueError, match="version"):
        LearnerStore.from_dict({"version": 999, "history": []})


def test_from_dict_rejects_dangling_pointer():
    """A snapshot whose ``generation`` pointer is not in ``history`` is
    structurally corrupt — fail closed so the next rollout does not silently
    inherit from generation 0."""
    bad = {
        "version": 1,
        "generation": 5,
        "next_number": 6,
        "history": [
            {"number": 0, "metric": None, "memory": {}, "skills": {}},
        ],
    }
    with pytest.raises(ValueError, match="generation"):
        LearnerStore.from_dict(bad)


def test_from_dict_restores_generation_zero_when_missing():
    """Generation 0 is the empty starting state — its absence in a snapshot
    must not break revert/learning_curve callers."""
    restored = LearnerStore.from_dict(
        {
            "version": 1,
            "generation": 0,
            "next_number": 1,
            "history": [],
        }
    )
    assert restored.generation == 0
    assert 0 in restored.history


def test_snapshot_is_pretty_json(tmp_path):
    """The on-disk snapshot is human-auditable JSON — readers may diff it
    across resumes to verify which generations rolled forward."""
    store = LearnerStore()
    store.commit(LearnerState(skills={"a": "b"}), metric=1.0)
    snap = tmp_path / "s.json"
    store.save(snap)

    data = json.loads(snap.read_text())
    assert data["version"] == 1
    assert "history" in data
    # Indented (non-minified) so it diffs cleanly.
    assert "\n" in snap.read_text()

"""The assessment event log: envelope, ordering, idempotency and optimistic append."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow import assessment as a
from benchflow.assessment import records as r
from benchflow.assessment.store import EVENTS_FILE, Sidecar
from tests.assessment._fixtures import OPERATOR, REVIEWER, world


def _lines(directory: Path) -> list[dict]:
    return [
        json.loads(line) for line in (directory / EVENTS_FILE).read_text().splitlines()
    ]


def test_a_full_assessment_writes_one_verified_chain(tmp_path: Path) -> None:
    w = world(tmp_path)
    created = w.create()
    assert created.event.kind == "assessment.created" and not created.replayed
    assert created.assessment.state == "awaiting_release"
    assert w.release().assessment.state == "awaiting_assignment"
    assert w.assign().assessment.state == "awaiting_human"
    w.vote("left", reason="clearer hierarchy", tags=("hierarchy", "layout"))
    answered = w.import_vote()
    assert answered.assessment.state == "answered"
    reduced = a.reduce(w.sidecar)
    assert reduced.label["status"] == "ready" and reduced.label["value"] == 1
    assert reduced.assessment.state == "label_ready"

    records = _lines(w.sidecar)
    assert [rec["kind"] for rec in records] == [
        "assessment.created",
        "compute.released",
        "assignment.issued",
        "review.submitted",
        "label.reduced",
    ]
    previous = None
    for seq, record in enumerate(records, start=1):
        assert record["schema_version"] == r.EVENT_SCHEMA
        assert record["event_seq"] == seq
        assert record["previous_event_sha256"] == previous
        assert record["expected_previous_event_sha256"] == previous
        assert record["event_sha256"] == r.event_digest(record)
        assert record["assessment_id"] == records[0]["assessment_id"]
        previous = record["event_sha256"]
    # Every line is exactly JCS(record) + "\n".
    raw = (w.sidecar / EVENTS_FILE).read_bytes()
    assert raw == b"".join(a.canonicalize(rec) + b"\n" for rec in records)


def test_release_comes_before_assignment(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.create()
    w.collector_assignment()
    with pytest.raises(a.TransitionRefused, match=r"compute\.released first"):
        w.assign()
    assert len(_lines(w.sidecar)) == 1


def test_an_assignment_made_before_the_release_stays_history(tmp_path: Path) -> None:
    """A legacy collector assignment (made before this assessment's release) is not bound."""
    w = world(tmp_path)
    w.create()
    w.collector_assignment()  # created now, before the release below
    w.release()
    with pytest.raises(a.EvidenceRejected, match="predates the recorded release"):
        w.assign()
    assert len(_lines(w.sidecar)) == 2


def test_identical_retry_returns_the_original_event(tmp_path: Path) -> None:
    w = world(tmp_path)
    first = w.create()
    again = w.create()
    assert again.replayed and again.event.id == first.event.id
    w.release()
    assert w.release().replayed
    assigned = w.assign()
    assert w.assign().event.id == assigned.event.id
    w.vote("left")
    submitted = w.import_vote()
    replay = w.import_vote()
    assert replay.replayed and replay.event.id == submitted.event.id
    assert len(_lines(w.sidecar)) == 4


def test_idempotent_retry_wins_over_a_stale_expectation(tmp_path: Path) -> None:
    w = world(tmp_path)
    created = w.create()
    w.release()
    sidecar = Sidecar(w.sidecar)
    retry = sidecar.append(
        kind="assessment.created",
        actor=created.event.actor,
        payload=created.event.payload,
        idempotency_key=created.event.key,
        expected_event_seq=0,
        expected_previous_event_sha256=None,
    )
    assert retry.replayed and retry.event.id == created.event.id


def test_same_key_with_another_payload_conflicts_and_appends_nothing(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    w.answered("left")
    state = a.load(w.sidecar)
    original = state.answers[state.effective].event
    payload = original.to_dict()["payload"]
    payload["choice"] = "right"
    before = (w.sidecar / EVENTS_FILE).read_bytes()
    with pytest.raises(a.Conflict, match="different payload"):
        Sidecar(w.sidecar).append(
            kind="review.submitted",
            actor=original.actor,
            payload=payload,
            idempotency_key=original.key,
            expected_event_seq=state.head_seq,
            expected_previous_event_sha256=state.head_sha256,
        )
    assert (w.sidecar / EVENTS_FILE).read_bytes() == before


def test_a_second_answer_under_a_new_key_conflicts(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    state = a.load(w.sidecar)
    original = state.answers[state.effective].event
    with pytest.raises(a.Conflict, match="already has an answer"):
        Sidecar(w.sidecar).append(
            kind="review.submitted",
            actor=original.actor,
            payload=original.payload,
            idempotency_key="another-key",
            expected_event_seq=state.head_seq,
            expected_previous_event_sha256=state.head_sha256,
        )


def test_stale_expectation_appends_nothing(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    state = a.load(w.sidecar)
    target = state.effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="tie",
        reason="equal",
        correction_reason="misclick",
        actor=REVIEWER,
    )
    before = (w.sidecar / EVENTS_FILE).read_bytes()
    with pytest.raises(a.StaleExpectation):
        Sidecar(w.sidecar).append(
            kind="review.retracted",
            actor=REVIEWER,
            payload={
                "target_event_id": target,
                "assignment_id": w.assignment_id,
                "reviewer_subject": "held-out-reviewer-01",
                "reason": "withdrawn",
                "retracted_at": r.utc_now(),
            },
            idempotency_key="retract-1",
            expected_event_seq=state.head_seq,
            expected_previous_event_sha256=state.head_sha256,
        )
    assert (w.sidecar / EVENTS_FILE).read_bytes() == before


def test_only_the_reviewer_or_an_operator_changes_an_answer(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    target = a.load(w.sidecar).effective
    with pytest.raises(a.TransitionRefused, match="only the reviewer"):
        a.correct(
            w.sidecar,
            supersedes_event_id=target,
            choice="right",
            reason="",
            correction_reason="not mine",
            actor={"role": "reviewer", "subject": "someone-else"},
        )
    corrected = a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="right",
        reason="",
        correction_reason="operator fix",
        actor=OPERATOR,
    )
    assert corrected.assessment.effective == corrected.event.id


def test_a_correction_must_name_the_current_answer(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    first = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=first,
        choice="right",
        reason="",
        correction_reason="x",
        actor=REVIEWER,
    )
    with pytest.raises(a.TransitionRefused, match="current effective answer"):
        a.correct(
            w.sidecar,
            supersedes_event_id=first,
            choice="tie",
            reason="",
            correction_reason="y",
            actor=REVIEWER,
        )


def test_unknown_payload_fields_and_versions_are_refused(tmp_path: Path) -> None:
    w = world(tmp_path)
    created = w.create()
    payload = created.event.to_dict()["payload"]
    payload["extra"] = True
    with pytest.raises(a.InvalidRecord, match="unknown field extra"):
        Sidecar(tmp_path / "other").append(
            kind="assessment.created",
            actor=created.event.actor,
            payload=payload,
            idempotency_key="k",
            expected_event_seq=0,
            expected_previous_event_sha256=None,
        )
    with pytest.raises(a.InvalidRecord, match="unknown assessment schema version"):
        r.check_envelope(
            dict(
                created.event.to_dict(), schema_version="benchflow.assessment-event.v2"
            )
        )
    with pytest.raises(a.InvalidRecord, match="must be one of"):
        r.check_envelope(dict(created.event.to_dict(), kind="review.deleted"))

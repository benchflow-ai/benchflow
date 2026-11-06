"""The frozen assessment-sidecar-v1 contract's acceptance cases, made executable.

``tests/fixtures/assessment/contract-92341d97/`` holds the contract's
``acceptance-cases.json`` and ``assessment-event.v1.schema.json`` exactly as
lab commit 92341d97 froze them (``MANIFEST.json`` pins their git blob ids).
Each case below runs the implementation through the case's steps and checks
the case's ``expect`` block; every event written must also validate against
the frozen schema with an independent validator. Votes are synthetic
fixtures: they exercise mechanics, not human preference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchflow import assessment as a
from benchflow.assessment.store import EVENTS_FILE
from tests.assessment._fixtures import OPERATOR, REVIEWER, World, world

CONTRACT = Path(__file__).resolve().parents[1] / "fixtures/assessment/contract-92341d97"
REVIEWER_SUBJECT = "collector-subject:held-out-reviewer-01"


def _cases() -> dict[str, dict[str, Any]]:
    document = json.loads((CONTRACT / "acceptance-cases.json").read_text())
    assert document["schema_version"] == "benchflow.assessment.acceptance-fixtures.v1"
    return {case["id"]: case for case in document["cases"]}


def test_the_vendored_contract_is_the_frozen_one() -> None:
    manifest = json.loads((CONTRACT / "MANIFEST.json").read_text())
    assert manifest["commit"] == "92341d974f84c4f7f3f40ef87d075a78080c23a6"
    for name, pinned in manifest["files"].items():
        data = (CONTRACT / name).read_bytes()
        blob = hashlib.sha1(
            b"blob %d\0" % len(data) + data
        ).hexdigest()  # git's blob id
        assert blob == pinned["git_blob"], name
        assert hashlib.sha256(data).hexdigest() == pinned["sha256"], name


def test_every_frozen_case_is_executed() -> None:
    assert set(_cases()) == set(CASES)


def _frozen_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((CONTRACT / "assessment-event.v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )


def _setup(
    tmp_path: Path, steps: list[dict[str, Any]], shared: dict[str, Any]
) -> World:
    """Run the case's created/released/assigned steps."""
    w = world(tmp_path)
    kinds = [step["kind"] for step in steps]
    assert kinds[0] == "assessment.created"
    w.create()
    assert a.load(w.sidecar).state == "awaiting_release"
    assert "compute.released" in kinds
    w.release()
    if "assignment.issued" in kinds:
        w.collector_assignment(reviewer=REVIEWER_SUBJECT)
        w.assign()
    return w


def _answer(w: World, step: dict[str, Any]) -> a.Appended:
    collector_choice = {"abstain": "skip"}.get(step["choice"], step["choice"])
    w.vote(collector_choice, reviewer=REVIEWER_SUBJECT, reason=step.get("reason", ""))
    return a.import_vote(
        w.sidecar,
        collector_db=w.collector.db,
        collector_code=w.collector.code,
        idempotency_key=step["idempotency_key"],
    )


def _snapshot(w: World) -> dict[str, bytes]:
    """The saved attempts' bytes outside the sidecar: result.json and outputs."""
    files = {}
    for rollout in (w.first, w.second):
        for path in sorted(rollout.root.rglob("*")):
            if (
                path.is_file()
                and "assessment" not in path.relative_to(rollout.root).parts
            ):
                files[str(path)] = path.read_bytes()
    return files


def case_restart(tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    issued = a.load(w.sidecar).assignment
    assert issued is not None
    # Restart: a fresh read of the sidecar and the collector's persisted row.
    again = w.assign(w.assignment_id)
    assert again.replayed and again.event.id == issued.id
    assert again.event.payload["presentation"] == issued.payload["presentation"]
    assert again.event.payload["collector"]["assignment_id"] == w.assignment_id
    state = a.load(w.sidecar)
    assert state.state == case["expect"]["state"] == "awaiting_human"
    assert [e.kind for e in state.events].count("compute.released") == 1
    assert case["expect"]["new_compute_allocations"] == 0
    assert state.next_reduction() is None and state.label() is None
    assert case["expect"]["reduction"] is None


def case_duplicate(
    tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]
) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    submitted = _answer(w, case["steps"][3])
    replay = a.import_vote(
        w.sidecar,
        collector_db=w.collector.db,
        collector_code=w.collector.code,
        idempotency_key=case["steps"][4]["idempotency_key"],
    )
    assert replay.replayed and replay.event.id == submitted.event.id
    reduced = a.reduce(w.sidecar)
    kinds = [e.kind for e in reduced.assessment.events]
    assert kinds.count("review.submitted") == case["expect"]["answer_events"] == 1
    assert reduced.label["status"] == case["expect"]["label_status"]
    assert reduced.label["value"] == case["expect"]["value"]


def case_conflict(tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    submitted = _answer(w, case["steps"][3])
    refusal = case["steps"][4]
    before = (w.sidecar / EVENTS_FILE).read_bytes()
    payload = submitted.event.to_dict()["payload"]
    payload["choice"] = refusal["choice"]
    state = a.load(w.sidecar)
    with pytest.raises(a.Conflict):
        a.Sidecar(w.sidecar).append(
            kind="review.submitted",
            actor=submitted.event.actor,
            payload=payload,
            idempotency_key=refusal["idempotency_key"],
            expected_event_seq=state.head_seq,
            expected_previous_event_sha256=state.head_sha256,
        )
    assert refusal["http_status"] == 409 and refusal["appended"] is False
    assert (w.sidecar / EVENTS_FILE).read_bytes() == before
    reduced = a.reduce(w.sidecar)
    kinds = [e.kind for e in reduced.assessment.events]
    assert kinds.count("review.submitted") == case["expect"]["answer_events"]
    assert reduced.label["value"] == case["expect"]["value"]


def case_tie(tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    before = _snapshot(w)
    _answer(w, case["steps"][3])
    reduced = a.reduce(w.sidecar)
    assert reduced.label["status"] == case["expect"]["label_status"]
    assert reduced.label["value"] == case["expect"]["value"] == 0.5
    assert _snapshot(w) == before  # no reward, result or stream file written
    assert case["expect"]["legacy_reward_write"] is False
    assert case["expect"]["rollout_stream_write"] is False


def case_abstain(tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    before = _snapshot(w)
    step = case["steps"][3]
    answered = _answer(w, step)
    assert answered.event.payload["choice"] == "abstain"
    assert answered.event.payload["reason"] == step["reason"]
    reduced = a.reduce(w.sidecar)
    assert reduced.label["status"] == case["expect"]["label_status"] == "withheld"
    assert reduced.label["withheld_reason"] == case["expect"]["withheld_reason"]
    assert ("value" in reduced.label) is case["expect"]["has_numeric_value"] is False
    assert reduced.assessment.state == "withheld"
    assert _snapshot(w) == before
    assert case["expect"]["legacy_reward_write"] is False


def _consume(w: World, step: dict[str, Any], revision: int) -> a.Appended:
    return a.consume(
        w.sidecar,
        label_revision=revision,
        consumer_id=step["consumer_id"],
        consumer_receipt=step["consumer_receipt"],
    )


def case_correction(
    tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]
) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    steps = case["steps"]
    submitted = _answer(w, steps[3])
    first = a.reduce(w.sidecar)
    assert (
        first.label["revision"] == steps[4]["revision"]
        and first.label["value"] == steps[4]["value"]
    )
    consumed = _consume(w, steps[5], first.label["revision"])
    consumed_line = consumed.event.line
    correction = steps[6]
    a.correct(
        w.sidecar,
        supersedes_event_id=submitted.event.id,
        choice=correction["choice"],
        reason="",
        correction_reason="reviewer corrected the answer",
        actor=REVIEWER | {"subject": REVIEWER_SUBJECT},
        idempotency_key=correction["idempotency_key"],
    )
    second = a.reduce(w.sidecar)
    label = second.label
    assert (
        label["revision"]
        == case["expect"]["current_label_revision"]
        == steps[7]["revision"]
    )
    assert label["value"] == case["expect"]["current_value"]
    assert (
        label["revisions"][1]["supersedes_revision"] == steps[7]["supersedes_revision"]
    )
    state = second.assessment
    # The consumption is untouched, byte for byte, and still names revision 1.
    original = next(e for e in state.events if e.id == consumed.event.id)
    assert original.line == consumed_line and original.payload["label_revision"] == 1
    assert case["expect"]["old_consumption_mutated"] is False
    assert case["expect"]["consumer_disposition_required"] is True
    assert label["resolved"] is False
    assert label["unresolved_revisions"] == [
        {"label_revision": 1, "consumption_event_ids": [consumed.event.id]}
    ]
    a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="superseded by a correction",
        disposition="queued_compensation",
        actor=OPERATOR,
    )
    assert a.load(w.sidecar).label()["resolved"] is True


def case_retraction(
    tmp_path: Path, case: dict[str, Any], shared: dict[str, Any]
) -> None:
    w = _setup(tmp_path, case["steps"], shared)
    steps = case["steps"]
    submitted = _answer(w, steps[3])
    first = a.reduce(w.sidecar)
    consumed = _consume(w, steps[5], first.label["revision"])
    a.retract(
        w.sidecar,
        target_event_id=submitted.event.id,
        reason=steps[6]["reason"],
        actor=REVIEWER | {"subject": REVIEWER_SUBJECT},
    )
    withdrawn = a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="answer retracted",
        disposition=steps[7]["consumer_disposition"],
        actor=OPERATOR,
    )
    state = withdrawn.assessment
    assert state.state == case["expect"]["state"] == "withdrawn_for_future_use"
    with pytest.raises(a.TransitionRefused, match="withdrawn"):
        a.consume(
            w.sidecar,
            label_revision=1,
            consumer_id="trainer-ledger:new",
            consumer_receipt="r",
        )
    assert case["expect"]["new_consumers_refused"] is True
    original = next(e for e in state.events if e.id == consumed.event.id)
    assert original.line == consumed.event.line
    assert case["expect"]["old_consumption_mutated"] is False
    label = state.label()
    assert label["status"] == "withdrawn" and "value" not in label
    assert case["expect"]["numeric_label_available"] is False
    assert state.consumable() is None
    # History stays: the answer, its retraction and the consumed revision.
    kinds = [e.kind for e in state.events]
    assert kinds[-3:] == ["label.consumed", "review.retracted", "label.withdrawn"]


CASES = {
    "restart_preserves_assignment_after_compute_release": case_restart,
    "identical_duplicate_is_idempotent": case_duplicate,
    "conflicting_duplicate_is_rejected_without_append": case_conflict,
    "tie_reduces_to_half_without_scalar_reward_change": case_tie,
    "abstention_is_withheld_not_zero_or_pending": case_abstain,
    "correction_creates_new_revision_and_preserves_consumed_label": case_correction,
    "retraction_withdraws_future_use_without_deleting_history": case_retraction,
}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_frozen_acceptance_case(case_id: str, tmp_path: Path) -> None:
    document = json.loads((CONTRACT / "acceptance-cases.json").read_text())
    case = _cases()[case_id]
    CASES[case_id](tmp_path, case, document["shared"])
    w_sidecar = next(tmp_path.rglob(EVENTS_FILE)).parent
    validator = _frozen_validator()
    for line in (w_sidecar / EVENTS_FILE).read_text().splitlines():
        record = json.loads(line)
        assert list(validator.iter_errors(record)) == [], record["kind"]

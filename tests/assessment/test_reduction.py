"""Deterministic reduction: the label is a pure function of the sidecar's log.

Left is 1 (A preferred), right 0, tie 0.5; an abstention and a retraction
are withheld, never a number. A correction makes a new revision; a consumed
revision is never rewritten, and needs a disposition once superseded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchflow import assessment as a
from benchflow.assessment.jcs import canonicalize
from benchflow.assessment.store import LABEL_FILE
from tests.assessment._fixtures import OPERATOR, REVIEWER, world


@pytest.mark.parametrize(
    ("choice", "status", "value"),
    [
        ("left", "ready", 1),
        ("right", "ready", 0),
        ("tie", "ready", 0.5),
        ("skip", "withheld", None),
    ],
)
def test_each_answer_reduces_to_its_label(
    tmp_path: Path, choice: str, status: str, value
) -> None:
    w = world(tmp_path)
    w.answered(choice)
    label = a.reduce(w.sidecar).label
    assert label["status"] == status
    assert label.get("value") == value
    if value is None:
        assert label["withheld_reason"] == "human_abstention" and "value" not in label


def test_the_label_names_every_identity_it_used(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    reduced = a.reduce(w.sidecar)
    label, state = reduced.label, reduced.assessment
    created = state.created.payload
    assignment = state.assignment.payload
    assert label["presentation"] == assignment["presentation"]
    by_review = {x["review_artifact"]["sha256"]: x for x in created["attempts"]}
    for side in ("A", "B"):
        attempt = by_review[label["presentation"][side]]
        assert label["attempts"][side] == {
            "attempt_id": attempt["attempt_id"],
            "rollout_id": attempt["rollout_id"],
            "result_sha256": attempt["result_sha256"],
            "output_manifest_sha256": attempt["output_manifest_sha256"],
            "review_artifact_sha256": attempt["review_artifact"]["sha256"],
        }
    assert label["rubric"]["sha256"] == created["rubric"]["sha256"]
    assert label["judge_protocol"]["sha256"] == created["judge_protocol"]["sha256"]
    assert (
        label["collector"]["assignment_export_sha256"]
        == assignment["collector"]["export_sha256"]
    )
    answer = state.answers[state.effective].event
    assert label["source"] == {
        "event_id": answer.id,
        "event_sha256": answer.sha256,
        "kind": "review.submitted",
    }
    assert label["event_chain"]["head_event_sha256"] == state.head_sha256
    assert label["revisions"][0]["reduced_event_sha256"] == state.current.event.sha256


def test_a_fresh_process_rebuilds_the_same_label_bytes(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("right")
    a.reduce(w.sidecar)
    a.consume(
        w.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row-9"
    )
    here = canonicalize(a.load(w.sidecar).label())
    script = (
        "import sys; from benchflow import assessment as a; "
        "sys.stdout.buffer.write(a.canonicalize(a.load(sys.argv[1]).label()))"
    )
    there = subprocess.run(
        [sys.executable, "-c", script, str(w.sidecar)], check=True, capture_output=True
    ).stdout
    assert there == here
    assert (w.sidecar / LABEL_FILE).read_bytes() == here + b"\n"


def test_reduce_is_idempotent_and_refuses_without_an_answer(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.create()
    with pytest.raises(a.TransitionRefused, match="needs an answer"):
        a.reduce(w.sidecar)
    w.release()
    w.assign()
    w.vote("tie")
    w.import_vote()
    first = a.reduce(w.sidecar)
    second = a.reduce(w.sidecar)
    assert first.event is not None and second.event is None
    assert canonicalize(first.label) == canonicalize(second.label)


def test_correction_before_reduction_is_the_first_revision(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    target = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="tie",
        reason="",
        correction_reason="misclick",
        actor=REVIEWER,
    )
    label = a.reduce(w.sidecar).label
    assert label["revision"] == 1 and label["value"] == 0.5
    assert label["source"]["kind"] == "review.corrected"


def test_a_correction_after_reduction_makes_a_new_revision(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    first = a.reduce(w.sidecar).label
    target = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="right",
        reason="",
        correction_reason="c",
        actor=REVIEWER,
    )
    stale = a.load(w.sidecar).label()
    assert (
        stale["status"] == "stale" and "value" not in stale and not stale["consumable"]
    )
    second = a.reduce(w.sidecar).label
    assert (second["revision"], second["value"], second["label_id"]) == (
        2,
        0,
        first["label_id"],
    )
    assert second["revisions"][0]["value"] == 1  # history stays
    assert second["resolved"]  # nobody consumed revision 1


def test_retraction_withholds_the_label(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    a.reduce(w.sidecar)
    target = a.load(w.sidecar).effective
    a.retract(
        w.sidecar, target_event_id=target, reason="reviewer withdrew", actor=REVIEWER
    )
    assert a.load(w.sidecar).state == "retracted"
    label = a.reduce(w.sidecar).label
    assert (
        label["status"] == "withheld" and label["withheld_reason"] == "answer_retracted"
    )
    assert label["revision"] == 2 and label["source"]["kind"] == "review.retracted"
    with pytest.raises(a.TransitionRefused, match="no effective answer"):
        a.correct(
            w.sidecar,
            supersedes_event_id=target,
            choice="tie",
            reason="",
            correction_reason="c",
            actor=REVIEWER,
        )


def test_a_reduced_abstention_takes_only_a_replacement(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("skip")
    a.reduce(w.sidecar)
    target = a.load(w.sidecar).effective
    with pytest.raises(a.TransitionRefused, match="replacement assignment"):
        a.correct(
            w.sidecar,
            supersedes_event_id=target,
            choice="left",
            reason="",
            correction_reason="c",
            actor=REVIEWER,
        )
    with pytest.raises(a.TransitionRefused, match="replacement assignment"):
        a.retract(w.sidecar, target_event_id=target, reason="r", actor=REVIEWER)


def test_consumption_needs_the_current_ready_revision(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("skip")
    a.reduce(w.sidecar)
    with pytest.raises(a.TransitionRefused, match="withheld"):
        a.consume(
            w.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row"
        )

    w2 = world(tmp_path / "two")
    w2.answered("left")
    a.reduce(w2.sidecar)
    a.consume(
        w2.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row"
    )
    with pytest.raises(a.Conflict, match="different payload"):
        a.consume(
            w2.sidecar,
            label_revision=1,
            consumer_id="ledger",
            consumer_receipt="another row",
        )
    with pytest.raises(a.Conflict, match="already recorded using"):
        a.consume(
            w2.sidecar,
            label_revision=1,
            consumer_id="ledger",
            consumer_receipt="another row",
            idempotency_key="a-second-use",
        )
    target = a.load(w2.sidecar).effective
    a.correct(
        w2.sidecar,
        supersedes_event_id=target,
        choice="right",
        reason="",
        correction_reason="c",
        actor=REVIEWER,
    )
    with pytest.raises(a.TransitionRefused, match="awaits reduction"):
        a.consume(
            w2.sidecar, label_revision=1, consumer_id="ledger-2", consumer_receipt="row"
        )
    a.reduce(w2.sidecar)
    with pytest.raises(a.TransitionRefused, match="supersedes it"):
        a.consume(
            w2.sidecar, label_revision=1, consumer_id="ledger-2", consumer_receipt="row"
        )
    assert (
        a.consume(
            w2.sidecar, label_revision=2, consumer_id="ledger-2", consumer_receipt="row"
        ).assessment.state
        == "consumed"
    )


def test_withdrawal_rules(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    a.reduce(w.sidecar)
    with pytest.raises(a.TransitionRefused, match="is current"):
        a.withdraw(
            w.sidecar,
            label_revision=1,
            reason="r",
            disposition="not_applicable",
            actor=OPERATOR,
        )
    target = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="tie",
        reason="",
        correction_reason="c",
        actor=REVIEWER,
    )
    with pytest.raises(a.TransitionRefused, match="no consumer to compensate"):
        a.withdraw(
            w.sidecar,
            label_revision=1,
            reason="r",
            disposition="compensated",
            actor=OPERATOR,
        )
    withdrawn = a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="superseded",
        disposition="not_applicable",
        actor=OPERATOR,
    )
    assert withdrawn.assessment.state == "withdrawn_for_future_use"
    label = a.reduce(w.sidecar).label  # the correction still reduces
    assert (label["revision"], label["status"], label["value"]) == (2, "ready", 0.5)
    assert label["state"] == "label_ready" and label["resolved"]


def test_a_label_never_reaches_result_json(tmp_path: Path) -> None:
    w = world(tmp_path)
    result = (w.first.root / "result.json").read_bytes()
    w.answered("left")
    a.reduce(w.sidecar)
    a.consume(w.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row")
    assert (w.first.root / "result.json").read_bytes() == result
    files = sorted(p.name for p in w.sidecar.iterdir())
    assert files == [
        "assessment-events.v1.jsonl",
        "assessment-events.v1.lock",
        "assessment-label.v1.json",
    ]

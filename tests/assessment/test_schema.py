"""The published event and label schemas: current, frozen-compatible, and agreeing.

docs/reference/schemas/ holds generated files; the event schema's envelope
must equal the frozen contract's schema exactly. Real events and labels must
validate with an independent validator (jsonschema), and mutated events the
Python checks refuse must fail it too.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchflow import assessment as a
from benchflow.assessment import records as r
from benchflow.assessment import schema as s
from benchflow.assessment.store import EVENTS_FILE
from tests.assessment._fixtures import OPERATOR, REVIEWER, world

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "docs/reference/schemas"
FROZEN = (
    REPO / "tests/fixtures/assessment/contract-92341d97/assessment-event.v1.schema.json"
)


def test_the_committed_schemas_are_current() -> None:
    for name, schema in (
        (s.EVENT_FILE, s.event_schema()),
        (s.LABEL_FILE, s.label_schema()),
    ):
        committed = json.loads((SCHEMAS / name).read_text())
        assert committed == schema, (
            "regenerate with python -m benchflow.assessment.schema docs/reference/schemas"
        )
        jsonschema.Draft202012Validator.check_schema(schema)


def test_the_envelope_is_the_frozen_contract_schema() -> None:
    frozen = json.loads(FROZEN.read_text())
    assert frozen == s.EVENT_ENVELOPE
    assert list(frozen) == list(s.EVENT_ENVELOPE)  # key order too
    published = s.event_schema()
    published.pop("allOf")
    for name in list(published["$defs"]):
        if name not in frozen["$defs"]:
            del published["$defs"][name]
    assert published == frozen


@pytest.fixture(scope="module")
def full_log(tmp_path_factory) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every event kind, from a run with a correction, consumption and withdrawal."""
    w = world(tmp_path_factory.mktemp("schema"))
    w.answered("left", reason="clearer", tags=("layout",))
    a.reduce(w.sidecar)
    a.consume(w.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row")
    target = a.load(w.sidecar).effective
    corrected = a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="abstain",
        reason="",
        correction_reason="unsure",
        actor=REVIEWER,
    )
    a.reduce(w.sidecar)
    a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="superseded",
        disposition="queued_compensation",
        actor=OPERATOR,
    )
    w2 = world(tmp_path_factory.mktemp("retracted"))
    w2.answered("right")
    a.reduce(w2.sidecar)
    a.retract(
        w2.sidecar,
        target_event_id=a.load(w2.sidecar).effective,
        reason="withdrawn",
        actor=REVIEWER,
    )
    a.reduce(w2.sidecar)
    events = []
    for directory in (w.sidecar, w2.sidecar):
        events += [
            json.loads(line)
            for line in (directory / EVENTS_FILE).read_text().splitlines()
        ]
    assert {e["kind"] for e in events} == set(r.KINDS), corrected
    return events, a.load(w.sidecar).label()


def test_real_events_and_labels_validate(full_log) -> None:
    events, label = full_log
    validator = jsonschema.Draft202012Validator(s.event_schema())
    frozen = jsonschema.Draft202012Validator(json.loads(FROZEN.read_text()))
    for event in events:
        assert list(validator.iter_errors(event)) == [], event["kind"]
        assert list(frozen.iter_errors(event)) == [], event["kind"]
    labels = jsonschema.Draft202012Validator(s.label_schema())
    assert list(labels.iter_errors(label)) == []


def _mutations(event: dict[str, Any]):
    """Edits the Python checks must refuse; each must also fail the published schema."""
    payload = event["payload"]
    yield "extra-payload-field", lambda e: e["payload"].update(unexpected=1)
    yield "missing-payload-field", lambda e: e["payload"].pop(next(iter(payload)))
    yield "wrong-actor-role", lambda e: e["actor"].update(role="nobody")
    yield "uppercase-digest", lambda e: e.update(event_sha256=e["event_sha256"].upper())
    if event["kind"] == "label.reduced":
        yield "value-not-a-label", lambda e: e["payload"].update(value=0.25)
        if payload["status"] == "ready":
            yield (
                "withheld-with-value",
                lambda e: e["payload"].update(status="withheld"),
            )
    if event["kind"] in ("review.submitted", "review.corrected"):
        yield "unknown-choice", lambda e: e["payload"].update(choice="maybe")
        yield "unsorted-tags", lambda e: e["payload"].update(tags=["z", "a", "z"])
    if event["kind"] == "label.withdrawn":
        yield (
            "unknown-disposition",
            lambda e: e["payload"].update(consumer_disposition="ignored"),
        )


def test_the_python_checks_and_the_schema_agree_on_refusals(full_log) -> None:
    events, _ = full_log
    validator = jsonschema.Draft202012Validator(s.event_schema())
    checked = 0
    for event in events:
        for name, mutate in _mutations(event):
            broken = copy.deepcopy(event)
            mutate(broken)
            with pytest.raises(a.InvalidRecord):
                r.check_envelope(broken)
            assert list(validator.iter_errors(broken)), (event["kind"], name)
            checked += 1
    assert checked > 50

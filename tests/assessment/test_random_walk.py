"""Seeded random walks over the assessment's operations keep its invariants.

Each walk starts from an answered assessment and applies random corrections,
retractions, reductions, consumptions, withdrawals and replays, some aimed at
stale or wrong targets. After every step, accepted or refused:

* the log only grows: the old bytes are a prefix of the new ones;
* a refusal changes no byte;
* every consumption line, once written, stays byte for byte;
* the label rebuilds identically from the log's bytes alone;
* only a current, up-to-date, ready revision is consumable, and a withheld,
  withdrawn or stale label carries no number;
* an unresolved revision is a consumed one that is no longer current.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from benchflow import assessment as a
from benchflow.assessment.jcs import canonicalize
from benchflow.assessment.store import EVENTS_FILE, parse_log
from tests.assessment._fixtures import OPERATOR, REVIEWER, world

REFUSALS = (a.TransitionRefused, a.Conflict, a.EvidenceRejected)
CHOICES = ("left", "right", "tie", "abstain")


def _step(rng: random.Random, sidecar: Path, index: int) -> str:
    state = a.load(sidecar)
    revisions = len(state.revisions)
    answers = list(state.answers)
    target = rng.choice([state.effective, *answers, None] if answers else [None])
    op = rng.choice(["reduce", "correct", "retract", "consume", "withdraw", "replay"])
    if op == "reduce":
        a.reduce(sidecar)
    elif op == "correct" and target is not None:
        a.correct(
            sidecar,
            supersedes_event_id=target,
            choice=rng.choice(CHOICES),
            reason=f"walk {index}",
            correction_reason=f"walk {index}",
            actor=rng.choice([REVIEWER, OPERATOR]),
        )
    elif op == "retract" and target is not None:
        a.retract(
            sidecar, target_event_id=target, reason=f"walk {index}", actor=REVIEWER
        )
    elif op == "consume" and revisions:
        a.consume(
            sidecar,
            label_revision=rng.randint(1, revisions),
            consumer_id=f"ledger-{rng.randint(1, 3)}",
            consumer_receipt=f"row-{index}",
            idempotency_key=f"consume-{index}",
        )
    elif op == "withdraw" and revisions:
        a.withdraw(
            sidecar,
            label_revision=rng.randint(1, revisions),
            reason=f"walk {index}",
            disposition=rng.choice(
                ["not_applicable", "queued_compensation", "compensated"]
            ),
            actor=OPERATOR,
        )
    elif op == "replay" and state.events:
        event = rng.choice(state.events)
        result = a.Sidecar(sidecar).append(
            kind=event.kind,
            actor=event.actor,
            payload=event.payload,
            idempotency_key=event.key,
            expected_event_seq=rng.randint(0, state.head_seq),
            expected_previous_event_sha256=None,
        )
        assert result.replayed and result.event.id == event.id
    return op


def _check_invariants(
    sidecar: Path, before: bytes, consumed: dict[str, bytes]
) -> bytes:
    data = (sidecar / EVENTS_FILE).read_bytes()
    assert data.startswith(before), "the log was rewritten"
    state = parse_log(data)
    lines = {event.id: event.line for event in state.events}
    for event_id, line in consumed.items():
        assert lines[event_id] == line, "a consumption changed"
    for revision in state.revisions:
        for consumption in revision.consumptions:
            consumed.setdefault(consumption.id, consumption.line)
    label = state.label()
    rebuilt = parse_log(data).label()
    assert (label is None) == (rebuilt is None)
    if label is not None:
        assert canonicalize(label) == canonicalize(rebuilt)
        assert ("value" in label) == (label["status"] == "ready")
        assert label["consumable"] == (label["status"] == "ready")
        if label["status"] == "ready":
            current = state.current
            assert current is state.consumable()
            assert (
                not current.withdrawn
                and current.status == "ready"
                and not state.stale()
            )
    for revision in state.unresolved():
        assert revision.consumptions and not revision.withdrawals
        assert revision is not state.current or state.stale()
    return data


@pytest.mark.parametrize("seed", range(16))
def test_random_walks_keep_the_invariants(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    w = world(tmp_path)
    w.answered(rng.choice(["left", "right", "tie", "skip"]))
    data = (w.sidecar / EVENTS_FILE).read_bytes()
    consumed: dict[str, bytes] = {}
    accepted = refused = 0
    for index in range(30):
        try:
            _step(rng, w.sidecar, index)
            accepted += 1
        except REFUSALS:
            refused += 1
            assert (w.sidecar / EVENTS_FILE).read_bytes() == data, (
                "a refusal wrote bytes"
            )
        data = _check_invariants(w.sidecar, data, consumed)
    assert accepted and refused  # the walk exercised both paths

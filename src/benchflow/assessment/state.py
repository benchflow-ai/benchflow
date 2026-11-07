"""Replaying an assessment's events: its state machine and deterministic reduction.

:class:`Assessment` folds verified events, in order, into the assessment's
state. Every event is checked against the state before it, the same way when
a writer appends it and when a reader replays the log, so a log can only hold
transitions this module allows. A writer that wants to append an event calls
:meth:`Assessment.apply` on the current state first; a reader that finds an
event :meth:`~Assessment.apply` refuses treats the whole log as corrupt.

States (the frozen contract's, plus ``awaiting_assignment`` between release
and assignment, and ``retracted`` while a retraction awaits reduction):

==========================  ===================================================
``awaiting_release``        created; both attempts' hashes, rubric, protocol frozen
``awaiting_assignment``     every recorded runtime released; no collector assignment
``awaiting_human``          assignment, reviewer and A/B order frozen; no answer
``answered``                an effective answer not yet reduced into the label
``retracted``               the answer was retracted; reduction not yet recorded
``label_ready``             the current revision is ready (1, 0 or 0.5)
``consumed``                ... and a consumer recorded using it
``withheld``                the current revision is withheld (abstention, retraction)
``withdrawn_for_future_use``  the current revision was withdrawn from future use
==========================  ===================================================

Reduction (:meth:`Assessment.next_reduction`) is a pure function of the log:
the effective answer (the latest non-retracted one in the correction chain)
maps ``left`` to 1.0 (A preferred), ``right`` to 0.0, ``tie`` to 0.5 and
``abstain`` to ``withheld`` (``human_abstention``); a retraction with no
replacement is ``withheld`` (``answer_retracted``). A ``label.reduced`` event
must equal that derivation exactly, ``label_id`` included, so a replay
rebuilds the same label byte for byte.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from benchflow.assessment import records as r
from benchflow.assessment.collector import check_assignment_binding, check_vote_binding
from benchflow.assessment.errors import (
    Conflict,
    EvidenceRejected,
    TransitionRefused,
)

__all__ = ["Assessment", "Event", "Revision"]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class Event:
    """One verified event: its record and canonical line (no newline)."""

    __slots__ = ("_record", "line")

    def __init__(self, record: dict[str, Any], line: bytes) -> None:
        self._record = record
        self.line = line

    @property
    def id(self) -> str:
        return self._record["event_id"]

    @property
    def seq(self) -> int:
        return self._record["event_seq"]

    @property
    def kind(self) -> str:
        return self._record["kind"]

    @property
    def sha256(self) -> str:
        return self._record["event_sha256"]

    @property
    def assessment_id(self) -> str:
        return self._record["assessment_id"]

    @property
    def key(self) -> str:
        return self._record["idempotency_key"]

    @property
    def actor(self) -> Mapping[str, Any]:
        return self._record["actor"]

    @property
    def payload(self) -> Mapping[str, Any]:
        return self._record["payload"]

    @property
    def created_at(self) -> str:
        return self._record["created_at"]

    def semantic(self) -> bytes:
        return r.semantic_payload(self.kind, self.actor, self.payload)

    def to_dict(self) -> dict[str, Any]:
        """A deep copy of the record."""
        return copy.deepcopy(self._record)


@dataclass
class _Answer:
    event: Event
    superseded_by: str | None = None
    retracted_by: str | None = None

    @property
    def choice(self) -> str:
        return self.event.payload["choice"]


@dataclass
class Revision:
    """One ``label.reduced`` revision and what happened to it since."""

    event: Event
    consumptions: list[Event] = field(default_factory=list)
    withdrawals: list[Event] = field(default_factory=list)

    @property
    def number(self) -> int:
        return self.event.payload["revision"]

    @property
    def status(self) -> str:
        return self.event.payload["status"]

    @property
    def source_event_id(self) -> str:
        return self.event.payload["source_event_id"]

    @property
    def withdrawn(self) -> bool:
        return bool(self.withdrawals)

    @property
    def disposition(self) -> str | None:
        return (
            self.withdrawals[-1].payload["consumer_disposition"]
            if self.withdrawals
            else None
        )


class Assessment:
    """The replayed state of one sidecar."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.keys: dict[str, Event] = {}
        self.created: Event | None = None
        self.release: Event | None = None
        self.assignment: Event | None = None
        self.answers: dict[str, _Answer] = {}
        self.effective: str | None = None
        self.last_retraction: Event | None = None
        self.revisions: list[Revision] = []

    # -- identity and head -------------------------------------------------

    @property
    def assessment_id(self) -> str | None:
        return self.created.assessment_id if self.created else None

    @property
    def head_seq(self) -> int:
        return self.events[-1].seq if self.events else 0

    @property
    def head_sha256(self) -> str | None:
        return self.events[-1].sha256 if self.events else None

    @property
    def attempts(self) -> list[Mapping[str, Any]]:
        return list(self._created_payload()["attempts"])

    @property
    def protocol_collector(self) -> Mapping[str, Any]:
        return self._created_payload()["judge_protocol"]["collector"]

    def _created_payload(self) -> Mapping[str, Any]:
        if self.created is None:
            raise TransitionRefused("the sidecar holds no assessment yet")
        return self.created.payload

    # -- derived state -----------------------------------------------------

    @property
    def current(self) -> Revision | None:
        return self.revisions[-1] if self.revisions else None

    def label_source(self) -> Event | None:
        """What the label must reflect: the effective answer, else the retraction."""
        if self.effective is not None:
            return self.answers[self.effective].event
        return self.last_retraction

    def stale(self) -> bool:
        """True when the latest answer or retraction is not yet the current revision's source."""
        source = self.label_source()
        if source is None:
            return False
        current = self.current
        return current is None or current.source_event_id != source.id

    @property
    def state(self) -> str:
        if self.created is None:
            return "empty"
        if self.release is None:
            return "awaiting_release"
        if self.assignment is None:
            return "awaiting_assignment"
        if not self.answers:
            return "awaiting_human"
        current = self.current
        if current is not None and current.withdrawn:
            return "withdrawn_for_future_use"
        if current is None or self.stale():
            return "answered" if self.effective is not None else "retracted"
        if current.status == "withheld":
            return "withheld"
        return "consumed" if current.consumptions else "label_ready"

    def unresolved(self) -> list[Revision]:
        """Consumed revisions no longer current and still without a disposition."""
        current = self.current
        return [
            revision
            for revision in self.revisions
            if revision.consumptions
            and not revision.withdrawals
            and (revision is not current or self.stale())
        ]

    def consumable(self) -> Revision | None:
        """The revision a new consumer may use: current, up to date, ready."""
        current = self.current
        if (
            current is None
            or self.stale()
            or current.withdrawn
            or current.status != "ready"
        ):
            return None
        return current

    # -- applying events ---------------------------------------------------

    def apply(self, event: Event) -> None:
        """Check ``event`` against the state and fold it in, or raise.

        Raises :class:`TransitionRefused`, :class:`Conflict` or
        :class:`EvidenceRejected`; on a raise the state is unchanged.
        """
        if self.created is None:
            if event.kind != "assessment.created":
                raise TransitionRefused("an assessment starts with assessment.created")
        else:
            if event.kind == "assessment.created":
                raise Conflict(
                    f"the sidecar already holds assessment {self.assessment_id}",
                    hint="create a new assessment in a new sidecar directory",
                )
            if event.assessment_id != self.assessment_id:
                raise TransitionRefused(
                    f"event names assessment {event.assessment_id}, the sidecar holds "
                    f"{self.assessment_id}"
                )
        if event.key in self.keys:
            raise Conflict(
                f"idempotency key {event.key!r} is already used by event {self.keys[event.key].id}"
            )
        handler = _HANDLERS[event.kind]
        handler(self, event)
        self.keys[event.key] = event
        self.events.append(event)

    def _on_created(self, event: Event) -> None:
        self.created = event

    def _on_released(self, event: Event) -> None:
        if self.release is not None:
            raise TransitionRefused(
                f"compute was already released by event {self.release.id}"
            )
        payload = event.payload
        expected = self.expected_runtime_refs()
        if payload["runtime_refs"] != expected:
            raise EvidenceRejected(
                "the release must list exactly the runtimes the two attempts recorded "
                f"({len(expected)}), in (attempt_id, provider, runtime_id) order",
            )
        wanted = {(ref["provider"], ref["runtime_id"]) for ref in expected}
        observed: set[tuple[str, str]] = set()
        latest: datetime | None = None
        for observation in payload["release_proof"]["observations"]:
            key = (observation["provider"], observation["runtime_id"])
            if key not in wanted:
                raise EvidenceRejected(
                    f"the release proof observes {key[0]}:{key[1]}, which neither attempt recorded"
                )
            observed.add(key)
            moment = r.timestamp(observation["observed_at"], "observed_at")
            latest = moment if latest is None or moment > latest else latest
        missing = sorted(wanted - observed)
        if missing:
            names = ", ".join(f"{p}:{i}" for p, i in missing)
            raise EvidenceRejected(
                f"no release observation for {names}; every execution resource must be "
                "observed released before a human waits",
                hint="finished_at or a missing log line is not a release; observe each runtime",
            )
        released_at = r.timestamp(payload["released_at"], "released_at")
        if latest is not None and released_at < latest:
            raise EvidenceRejected(
                "released_at is earlier than an observation it relies on"
            )
        if released_at > r.timestamp(event.created_at, "created_at"):
            raise EvidenceRejected(
                "released_at is later than the event that records it"
            )
        self.release = event

    def expected_runtime_refs(self) -> list[dict[str, Any]]:
        refs = [
            {
                "attempt_id": attempt["attempt_id"],
                "provider": runtime["provider"],
                "runtime_id": runtime["runtime_id"],
                "role": runtime["role"],
            }
            for attempt in self.attempts
            for runtime in attempt["runtime_refs"]
        ]
        return sorted(
            refs,
            key=lambda ref: (ref["attempt_id"], ref["provider"], ref["runtime_id"]),
        )

    def _on_assigned(self, event: Event) -> None:
        if self.release is None:
            raise TransitionRefused(
                "assignment.issued needs compute.released first: every execution "
                "resource is released before an assignment can wait for a person",
                hint="record the release with `bench assess release <sidecar> --proof <file>`",
            )
        if self.assignment is not None:
            raise Conflict(
                "the assessment already has collector assignment "
                f"{self.assignment.payload['collector']['assignment_id']}",
                hint="a restarted collector must return that same assignment",
            )
        check_assignment_binding(self, event.payload)
        self.assignment = event

    def _assignment_payload(self) -> Mapping[str, Any]:
        if self.assignment is None:
            raise TransitionRefused("the assessment has no collector assignment yet")
        return self.assignment.payload

    def _check_answer_target(self, payload: Mapping[str, Any]) -> None:
        assignment = self._assignment_payload()
        if payload["assignment_id"] != assignment["collector"]["assignment_id"]:
            raise EvidenceRejected(
                f"answer names assignment {payload['assignment_id']!r}; this assessment's is "
                f"{assignment['collector']['assignment_id']!r}"
            )
        if payload["reviewer_subject"] != assignment["reviewer_subject"]:
            raise EvidenceRejected(
                f"answer names reviewer {payload['reviewer_subject']!r}; the assignment's is "
                f"{assignment['reviewer_subject']!r}"
            )

    def _on_submitted(self, event: Event) -> None:
        self._assignment_payload()
        if self.answers:
            raise Conflict(
                "this assignment already has an answer; replay it with its own "
                "idempotency key, or change it with review.corrected",
            )
        self._check_answer_target(event.payload)
        check_vote_binding(self, event)
        self.answers[event.id] = _Answer(event)
        self.effective = event.id

    def _check_change(self, event: Event, target: str, verb: str) -> _Answer:
        if self.effective is None:
            raise TransitionRefused(
                f"there is no effective answer to {verb}"
                + (" (it was retracted)" if self.last_retraction else "")
            )
        if target != self.effective:
            raise TransitionRefused(
                f"a {verb} must name the current effective answer {self.effective}; "
                f"{target} is superseded, retracted or unknown"
            )
        current = self.current
        if current is not None and not self.stale() and current.status == "withheld":
            raise TransitionRefused(
                "the label is withheld; only a separately issued replacement assignment "
                "(a new assessment) can follow"
            )
        self._check_answer_target(event.payload)
        actor = event.actor
        if (
            actor["role"] == "reviewer"
            and actor["subject"] != event.payload["reviewer_subject"]
        ):
            raise TransitionRefused(
                f"only the reviewer {event.payload['reviewer_subject']!r} or an operator may {verb} this answer"
            )
        return self.answers[target]

    def _on_corrected(self, event: Event) -> None:
        previous = self._check_change(
            event, event.payload["supersedes_event_id"], "correct"
        )
        previous.superseded_by = event.id
        self.answers[event.id] = _Answer(event)
        self.effective = event.id

    def _on_retracted(self, event: Event) -> None:
        previous = self._check_change(
            event, event.payload["target_event_id"], "retract"
        )
        previous.retracted_by = event.id
        self.effective = None
        self.last_retraction = event

    def next_reduction(self) -> dict[str, Any] | None:
        """The ``label.reduced`` payload the log calls for now, or None when up to date."""
        source = self.label_source()
        if source is None or not self.stale():
            return None
        current = self.current
        payload: dict[str, Any] = {
            "label_id": self.label_id(),
            "revision": current.number + 1 if current else 1,
        }
        if current is not None:
            payload["supersedes_revision"] = current.number
        payload["source_event_id"] = source.id
        payload["source_event_sha256"] = source.sha256
        if self.effective is None:
            payload.update(status="withheld", withheld_reason="answer_retracted")
        else:
            choice = self.answers[self.effective].choice
            if choice == "abstain":
                payload.update(status="withheld", withheld_reason="human_abstention")
            else:
                payload.update(status="ready", value=r.CHOICE_VALUES[choice])
        payload["reducer"] = dict(r.REDUCER)
        return payload

    def label_id(self) -> str:
        """The label lineage's id: a UUIDv7 derived from the log, the same on every replay.

        Its time is the first answer's receipt time; its other bits come from
        SHA-256 of the assessment id.
        """
        if self.revisions:
            return self.revisions[0].event.payload["label_id"]
        first = next(iter(self.answers.values())).event
        since_epoch = r.timestamp(first.created_at, "created_at") - _EPOCH
        millis = since_epoch // timedelta(milliseconds=1)
        assert self.assessment_id is not None
        bits = int.from_bytes(
            hashlib.sha256(f"{self.assessment_id}:label".encode()).digest()[:10], "big"
        )
        value = (millis & ((1 << 48) - 1)) << 80
        value |= 0x7 << 76
        value |= (bits >> 68) << 64
        value |= 0b10 << 62
        value |= bits & ((1 << 62) - 1)
        hexed = f"{value:032x}"
        return f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"

    def _on_reduced(self, event: Event) -> None:
        expected = self.next_reduction()
        if expected is None:
            raise TransitionRefused(
                "nothing to reduce: "
                + (
                    "no answer yet"
                    if self.label_source() is None
                    else "the label is up to date"
                )
            )
        if dict(event.payload) != expected:
            raise TransitionRefused(
                "label.reduced does not match the deterministic reduction of the log "
                f"(expected revision {expected['revision']} {expected['status']} from event "
                f"{expected['source_event_id']})"
            )
        self.revisions.append(Revision(event))

    def _revision(self, label_id: str, number: int) -> Revision:
        if not self.revisions:
            raise TransitionRefused("the assessment has no label yet")
        if label_id != self.revisions[0].event.payload["label_id"]:
            raise TransitionRefused(f"label {label_id} is not this assessment's label")
        if not 1 <= number <= len(self.revisions):
            raise TransitionRefused(f"label revision {number} does not exist")
        return self.revisions[number - 1]

    def _on_consumed(self, event: Event) -> None:
        payload = event.payload
        revision = self._revision(payload["label_id"], payload["label_revision"])
        if self.consumable() is not revision:
            raise TransitionRefused(
                f"label revision {revision.number} may not be consumed: "
                + _why_not_consumable(self, revision),
            )
        if event.actor["subject"] != payload["consumer_id"]:
            raise TransitionRefused(
                "a consumption is recorded by its consumer (actor.subject = consumer_id)"
            )
        if any(
            c.payload["consumer_id"] == payload["consumer_id"]
            for c in revision.consumptions
        ):
            raise Conflict(
                f"consumer {payload['consumer_id']!r} already recorded using revision {revision.number}"
            )
        revision.consumptions.append(event)

    def _on_withdrawn(self, event: Event) -> None:
        payload = event.payload
        revision = self._revision(payload["label_id"], payload["label_revision"])
        if revision is self.current and not self.stale():
            raise TransitionRefused(
                f"label revision {revision.number} is current; only a revision whose answer "
                "was corrected or retracted can be withdrawn"
            )
        consumed = [c.id for c in revision.consumptions]
        if list(payload["consumption_event_ids"]) != consumed:
            raise TransitionRefused(
                "consumption_event_ids must list every consumption of the revision, in order"
            )
        disposition = payload["consumer_disposition"]
        if not consumed and disposition != "not_applicable":
            raise TransitionRefused(
                "an unconsumed revision has no consumer to compensate: use not_applicable"
            )
        if revision.withdrawals:
            previous = revision.disposition
            if not (previous == "queued_compensation" and disposition == "compensated"):
                raise Conflict(
                    f"revision {revision.number} was already withdrawn ({previous}); a later "
                    "withdrawal may only move queued_compensation to compensated"
                )
        revision.withdrawals.append(event)

    # -- the label record --------------------------------------------------

    def label(self) -> dict[str, Any] | None:
        """The ``benchflow.assessment-label.v1`` record, or None before the first reduction.

        A pure function of the log: no clock, no file other than the log.
        """
        current = self.current
        if current is None or self.created is None or self.assignment is None:
            return None
        created = self.created.payload
        assignment = self.assignment.payload
        by_review = {a["review_artifact"]["sha256"]: a for a in created["attempts"]}
        sides = {}
        for side in ("A", "B"):
            attempt = by_review[assignment["presentation"][side]]
            sides[side] = {
                "attempt_id": attempt["attempt_id"],
                "rollout_id": attempt["rollout_id"],
                "result_sha256": attempt["result_sha256"],
                "output_manifest_sha256": attempt["output_manifest_sha256"],
                "review_artifact_sha256": attempt["review_artifact"]["sha256"],
            }
        if current.withdrawn:
            status = "withdrawn"
        elif self.stale():
            status = "stale"
        else:
            status = current.status
        label: dict[str, Any] = {
            "schema_version": r.LABEL_SCHEMA,
            "assessment_id": self.assessment_id,
            "label_id": current.event.payload["label_id"],
            "revision": current.number,
            "status": status,
        }
        if status == "ready":
            label["value"] = current.event.payload["value"]
        if status == "withheld":
            label["withheld_reason"] = current.event.payload["withheld_reason"]
        source = self._event(current.source_event_id)
        unresolved = self.unresolved()
        label.update(
            {
                "state": self.state,
                "consumable": self.consumable() is current,
                "resolved": not unresolved,
                "unresolved_revisions": [
                    {
                        "label_revision": revision.number,
                        "consumption_event_ids": [c.id for c in revision.consumptions],
                    }
                    for revision in unresolved
                ],
                "task_digest": created["attempts"][0]["task_digest"],
                "presentation": dict(assignment["presentation"]),
                "attempts": sides,
                "rubric": {
                    k: created["rubric"][k] for k in ("id", "version", "sha256")
                },
                "judge_protocol": {
                    k: created["judge_protocol"][k] for k in ("id", "version", "sha256")
                },
                "collector": {
                    "implementation": assignment["collector"]["implementation"],
                    "commit": assignment["collector"]["commit"],
                    "assignment_id": assignment["collector"]["assignment_id"],
                    "assignment_export_sha256": assignment["collector"][
                        "export_sha256"
                    ],
                },
                "reviewer_subject": assignment["reviewer_subject"],
                "source": {
                    "event_id": source.id,
                    "event_sha256": source.sha256,
                    "kind": source.kind,
                },
                "revisions": [_revision_view(revision) for revision in self.revisions],
                "event_chain": {
                    "length": len(self.events),
                    "head_event_seq": self.head_seq,
                    "head_event_sha256": self.head_sha256,
                },
                "reducer": dict(r.REDUCER),
            }
        )
        return label

    def _event(self, event_id: str) -> Event:
        for event in self.events:
            if event.id == event_id:
                return event
        raise KeyError(event_id)

    def summary(self) -> dict[str, Any]:
        """A short JSON view for ``bench assess show``."""
        current = self.current
        return {
            "assessment_id": self.assessment_id,
            "state": self.state,
            "events": len(self.events),
            "head_event_seq": self.head_seq,
            "head_event_sha256": self.head_sha256,
            "assignment_id": (
                self.assignment.payload["collector"]["assignment_id"]
                if self.assignment
                else None
            ),
            "effective_answer_event_id": self.effective,
            "label_revision": current.number if current else None,
            "reduction_pending": self.stale(),
            "resolved": not self.unresolved(),
        }


def _why_not_consumable(assessment: Assessment, revision: Revision) -> str:
    if revision is not assessment.current:
        return f"revision {assessment.current.number if assessment.current else '?'} supersedes it"
    if revision.withdrawn:
        return "it was withdrawn from future use"
    if assessment.stale():
        return "a correction or retraction awaits reduction"
    return "it is withheld (no numeric label)"


def _revision_view(revision: Revision) -> dict[str, Any]:
    payload = revision.event.payload
    view: dict[str, Any] = {
        "revision": revision.number,
        "status": revision.status,
    }
    if "value" in payload:
        view["value"] = payload["value"]
    if "withheld_reason" in payload:
        view["withheld_reason"] = payload["withheld_reason"]
    if "supersedes_revision" in payload:
        view["supersedes_revision"] = payload["supersedes_revision"]
    view.update(
        {
            "source_event_id": payload["source_event_id"],
            "source_event_sha256": payload["source_event_sha256"],
            "reduced_event_id": revision.event.id,
            "reduced_event_sha256": revision.event.sha256,
            "consumptions": [
                {
                    "event_id": c.id,
                    "event_sha256": c.sha256,
                    "consumer_id": c.payload["consumer_id"],
                    "consumer_receipt": c.payload["consumer_receipt"],
                }
                for c in revision.consumptions
            ],
            "withdrawals": [
                {
                    "event_id": w.id,
                    "event_sha256": w.sha256,
                    "consumer_disposition": w.payload["consumer_disposition"],
                    "reason": w.payload["reason"],
                }
                for w in revision.withdrawals
            ],
        }
    )
    return view


_HANDLERS = {
    "assessment.created": Assessment._on_created,
    "compute.released": Assessment._on_released,
    "assignment.issued": Assessment._on_assigned,
    "review.submitted": Assessment._on_submitted,
    "review.corrected": Assessment._on_corrected,
    "review.retracted": Assessment._on_retracted,
    "label.reduced": Assessment._on_reduced,
    "label.consumed": Assessment._on_consumed,
    "label.withdrawn": Assessment._on_withdrawn,
}

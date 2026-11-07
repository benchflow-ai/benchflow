"""Assessment operations: from two saved attempts to a reduced, consumable label.

Each operation reads the sidecar, re-verifies whatever evidence its boundary
needs, builds one event and appends it with the head it read as its
expectation. Each has a default idempotency key derived from its inputs, so
running the same command twice returns the first event instead of writing a
second one; a different input under the same key is a conflict.

==================  =========================================================
:func:`create`      two attempt refs + frozen rubric + frozen judge protocol
:func:`release`     a release proof observing every runtime the attempts name
:func:`assign`      bind the collector's persisted blind assignment
:func:`import_vote` bind the collector's vote (a skip becomes ``abstain``)
:func:`correct`     replace the effective answer (reviewer or operator)
:func:`retract`     withdraw the effective answer from future reduction
:func:`reduce`      append the deterministic ``label.reduced`` the log calls for
:func:`consume`     record a consumer's use of the current ready revision
:func:`withdraw`    bar a superseded revision, with the consumer disposition
:func:`verify`      re-verify the chain, the label file and optionally evidence
==================  =========================================================

None of them writes ``result.json``, ``regrade.json``, a rollout stream or a
reward: the sidecar is the only thing they change.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.assessment import records as r
from benchflow.assessment.collector import export_record
from benchflow.assessment.errors import (
    Conflict,
    CorruptSidecar,
    EvidenceRejected,
    InvalidRecord,
    StaleExpectation,
    TransitionRefused,
)
from benchflow.assessment.evidence import (
    frozen_file,
    read_regular,
    verify_attempt,
    verify_frozen_file,
)
from benchflow.assessment.jcs import CanonicalizationError, canonicalize, loads_strict
from benchflow.assessment.state import Assessment, Event
from benchflow.assessment.store import Appended, Sidecar

__all__ = [
    "BRIDGE",
    "REDUCER_ACTOR",
    "Reduced",
    "assign",
    "consume",
    "correct",
    "create",
    "import_vote",
    "load",
    "reduce",
    "release",
    "retract",
    "verify",
    "withdraw",
]

BRIDGE = {"role": "bridge", "subject": "benchflow.assessment.bridge"}
REDUCER_ACTOR = {"role": "reducer", "subject": "benchflow.assessment.reduce"}
Clock = Callable[[], str]


def _no_log(directory: str | Path) -> TransitionRefused:
    return TransitionRefused(
        f"{directory} holds no assessment log",
        hint="`bench assess create <sidecar> --attempt <ref> --attempt <ref> ...` starts one",
    )


def load(directory: str | Path) -> Assessment:
    """Verify a sidecar's whole log and replay it (no evidence access)."""
    sidecar = Sidecar(directory)
    if not sidecar.exists():
        raise _no_log(directory)
    return sidecar.load()


def _json_file(path: str | Path, what: str) -> tuple[bytes, Any]:
    data = read_regular(Path(path), what)
    try:
        return data, loads_strict(data)
    except CanonicalizationError as exc:
        raise InvalidRecord(f"{what} {path}: {exc}") from None


def _append(
    sidecar: Sidecar,
    state: Assessment,
    *,
    kind: str,
    actor: Mapping[str, str],
    payload: Mapping[str, Any],
    key: str,
    now: Clock,
) -> Appended:
    return sidecar.append(
        kind=kind,
        actor=actor,
        payload=payload,
        idempotency_key=key,
        expected_event_seq=state.head_seq,
        expected_previous_event_sha256=state.head_sha256,
        now=now,
    )


def _verify_evidence(state: Assessment) -> None:
    """Re-hash both attempts, the rubric and the judge protocol."""
    created = state.created
    if created is None:
        raise TransitionRefused("the sidecar holds no assessment")
    for attempt in created.payload["attempts"]:
        verify_attempt(attempt)
    verify_frozen_file(created.payload["rubric"], "rubric")
    verify_frozen_file(created.payload["judge_protocol"], "judge protocol")


def create(
    directory: str | Path,
    attempts: Sequence[Mapping[str, Any]],
    *,
    rubric: str | Path,
    rubric_id: str,
    rubric_version: str,
    protocol: str | Path,
    idempotency_key: str | None = None,
    actor: Mapping[str, str] = BRIDGE,
    now: Clock = r.utc_now,
) -> Appended:
    """Start an assessment of two saved attempts of one task.

    Both attempts are hashed again here: a changed, missing or cross-task
    attempt is refused before anything is written, so no collector
    assignment can exist for it.
    """
    if len(attempts) != 2:
        raise InvalidRecord("an assessment compares exactly two attempts")
    refs = [r.check_attempt(dict(a), f"attempts[{i}]") for i, a in enumerate(attempts)]
    refs.sort(key=lambda ref: ref["attempt_id"])
    if refs[0]["task_digest"] != refs[1]["task_digest"]:
        raise EvidenceRejected(
            "the attempts come from different tasks "
            f"({refs[0]['task_digest']} and {refs[1]['task_digest']}); an assessment "
            "compares two outputs of one task"
        )
    if refs[0]["attempt_id"] == refs[1]["attempt_id"]:
        raise EvidenceRejected("both references name the same attempt")
    for ref in refs:
        verify_attempt(ref)
    rubric_ref = frozen_file(rubric, file_id=rubric_id, version=rubric_version)
    data, document = _json_file(protocol, "judge protocol")
    document = r.check_protocol_document(document)
    # The digest is of the very bytes parsed above.
    protocol_ref = r.check_frozen_file(
        {
            "id": document["id"],
            "version": document["version"],
            "sha256": r.sha256_hex(data),
            "bytes": len(data),
            "locator": os.path.abspath(protocol),
            "collector": document["collector"],
        },
        "judge_protocol",
        extra=("collector",),
    )
    payload = {"attempts": refs, "rubric": rubric_ref, "judge_protocol": protocol_ref}
    key = idempotency_key or "assessment.created:" + r.jcs_sha256(
        {
            "attempts": [ref["attempt_id"] for ref in refs],
            "rubric": rubric_ref["sha256"],
            "judge_protocol": protocol_ref["sha256"],
        }
    )
    sidecar = Sidecar(directory)
    state = sidecar.load()
    if state.created is not None and state.created.key != key:
        raise Conflict(
            f"{directory} already holds assessment {state.assessment_id}",
            hint="use a new sidecar directory for another assessment",
        )
    return sidecar.append(
        kind="assessment.created",
        actor=actor,
        payload=payload,
        idempotency_key=key,
        expected_event_seq=0,
        expected_previous_event_sha256=None,
        now=now,
    )


def release(
    directory: str | Path,
    proof: str | Path,
    *,
    idempotency_key: str | None = None,
    actor: Mapping[str, str] = BRIDGE,
    now: Clock = r.utc_now,
) -> Appended:
    """Record that every runtime of both attempts was observed released.

    ``proof`` is a ``benchflow.assessment-release-proof.v1`` file of
    observations (provider, runtime, ``deleted``/``absent``, when, by whom,
    how). The bridge checks it names every recorded runtime and nothing else;
    it never starts, stops or reserves compute itself.
    """
    sidecar = Sidecar(directory)
    state = sidecar.load()
    data, document = _json_file(proof, "release proof")
    document = r.check_release_proof(document)
    proof_sha256 = r.sha256_hex(data)
    moment = now()
    payload = {
        "runtime_refs": state.expected_runtime_refs(),
        "released_at": moment,
        "release_proof": {
            "sha256": proof_sha256,
            "observations": document["observations"],
        },
    }
    return _append(
        sidecar,
        state,
        kind="compute.released",
        actor=actor,
        payload=payload,
        key=idempotency_key or f"compute.released:{proof_sha256}",
        now=lambda: moment,
    )


def assign(
    directory: str | Path,
    *,
    collector_db: str | Path,
    collector_code: str | Path,
    assignment_id: str,
    idempotency_key: str | None = None,
    actor: Mapping[str, str] = BRIDGE,
    now: Clock = r.utc_now,
) -> Appended:
    """Bind the collector's persisted blind assignment (after the release).

    Re-run after a collector restart, it returns the original event when the
    collector still holds the same assignment, sides and identities.
    """
    sidecar = Sidecar(directory)
    state = sidecar.load()
    _verify_evidence(state)
    export = export_record(
        collector_db,
        collector_code,
        assignment_id,
        pin=state.protocol_collector,
        record="assignment",
    )
    row = export["assignment"]
    payload = {
        "collector": {
            "implementation": export["collector"]["implementation"],
            "commit": export["collector"]["commit"],
            "assignment_id": row["id"],
            "export_sha256": r.jcs_sha256(export),
        },
        "presentation": {"A": row["left_hash"], "B": row["right_hash"]},
        "reviewer_subject": row["reviewer"],
        "collector_export": export,
    }
    return _append(
        sidecar,
        state,
        kind="assignment.issued",
        actor=actor,
        payload=payload,
        key=idempotency_key or f"assignment.issued:{row['id']}",
        now=now,
    )


def import_vote(
    directory: str | Path,
    *,
    collector_db: str | Path,
    collector_code: str | Path,
    idempotency_key: str | None = None,
    now: Clock = r.utc_now,
) -> Appended:
    """Bind the collector's vote on this assessment's assignment."""
    sidecar = Sidecar(directory)
    state = sidecar.load()
    if state.assignment is None:
        raise TransitionRefused("the assessment has no collector assignment yet")
    _verify_evidence(state)
    assignment_id = state.assignment.payload["collector"]["assignment_id"]
    export = export_record(
        collector_db,
        collector_code,
        assignment_id,
        pin=state.protocol_collector,
        record="vote",
    )
    vote = export["vote"]
    implementation = export["collector"]["implementation"]
    payload = {
        "assignment_id": assignment_id,
        "reviewer_subject": vote["reviewer"],
        "choice": r.COLLECTOR_CHOICES[vote["choice"]],
        "reason": vote["reason"],
        "tags": vote["tags"],
        "submitted_at": r.format_timestamp(
            r.parse_collector_time(vote["created"], "vote.created")
        ),
        "imported_from": {
            "collector": implementation,
            "record_id": vote["id"],
            "export_sha256": r.jcs_sha256(export),
        },
        "collector_export": export,
    }
    return _append(
        sidecar,
        state,
        kind="review.submitted",
        actor={"role": "collector", "subject": implementation},
        payload=payload,
        key=idempotency_key or f"review.submitted:{vote['id']}",
        now=now,
    )


def _answer_target(state: Assessment) -> tuple[str, str]:
    if state.assignment is None:
        raise TransitionRefused("the assessment has no collector assignment yet")
    payload = state.assignment.payload
    return payload["collector"]["assignment_id"], payload["reviewer_subject"]


def correct(
    directory: str | Path,
    *,
    supersedes_event_id: str,
    choice: str,
    reason: str,
    correction_reason: str,
    actor: Mapping[str, str],
    tags: Sequence[str] = (),
    idempotency_key: str | None = None,
    now: Clock = r.utc_now,
) -> Appended:
    """Replace the current effective answer with a complete new one."""
    sidecar = Sidecar(directory)
    state = sidecar.load()
    assignment_id, reviewer = _answer_target(state)
    moment = now()
    tag_list = sorted(set(tags))
    payload = {
        "supersedes_event_id": supersedes_event_id,
        "assignment_id": assignment_id,
        "reviewer_subject": reviewer,
        "choice": choice,
        "reason": reason,
        "tags": tag_list,
        "correction_reason": correction_reason,
        "corrected_at": moment,
    }
    key = idempotency_key or (
        f"review.corrected:{supersedes_event_id}:"
        + r.jcs_sha256(
            {
                "choice": choice,
                "reason": reason,
                "tags": tag_list,
                "correction_reason": correction_reason,
            }
        )[:32]
    )
    return _append(
        sidecar,
        state,
        kind="review.corrected",
        actor=actor,
        payload=payload,
        key=key,
        now=lambda: moment,
    )


def retract(
    directory: str | Path,
    *,
    target_event_id: str,
    reason: str,
    actor: Mapping[str, str],
    idempotency_key: str | None = None,
    now: Clock = r.utc_now,
) -> Appended:
    """Retract the current effective answer; history stays."""
    sidecar = Sidecar(directory)
    state = sidecar.load()
    assignment_id, reviewer = _answer_target(state)
    moment = now()
    payload = {
        "target_event_id": target_event_id,
        "assignment_id": assignment_id,
        "reviewer_subject": reviewer,
        "reason": reason,
        "retracted_at": moment,
    }
    return _append(
        sidecar,
        state,
        kind="review.retracted",
        actor=actor,
        payload=payload,
        key=idempotency_key or f"review.retracted:{target_event_id}",
        now=lambda: moment,
    )


@dataclass(frozen=True)
class Reduced:
    """The label after :func:`reduce`; ``event`` is None when nothing was appended."""

    label: dict[str, Any]
    event: Event | None
    assessment: Assessment


def reduce(
    directory: str | Path, *, now: Clock = r.utc_now, retries: int = 3
) -> Reduced:
    """Append the ``label.reduced`` the log calls for, if any, and refresh the label file.

    Reduction is derived from the log alone, so a stale expectation (another
    writer came first) is retried from a fresh read.
    """
    sidecar = Sidecar(directory)
    for attempt in range(retries + 1):
        state = sidecar.load()
        plan = state.next_reduction()
        if plan is None:
            label = state.label()
            if label is None:
                raise TransitionRefused(
                    f"nothing to reduce yet (state {state.state}): the label needs an answer"
                )
            if sidecar.read_label_file() != canonicalize(label) + b"\n":
                sidecar.write_label()
            return Reduced(label, None, state)
        try:
            result = _append(
                sidecar,
                state,
                kind="label.reduced",
                actor=REDUCER_ACTOR,
                payload=plan,
                key=f"label.reduced:{plan['source_event_id']}",
                now=now,
            )
        except StaleExpectation:
            if attempt == retries:
                raise
            continue
        label = result.assessment.label()
        assert label is not None
        return Reduced(
            label, None if result.replayed else result.event, result.assessment
        )
    raise AssertionError("unreachable")


def consume(
    directory: str | Path,
    *,
    label_revision: int,
    consumer_id: str,
    consumer_receipt: str,
    idempotency_key: str | None = None,
    now: Clock = r.utc_now,
) -> Appended:
    """Record that ``consumer_id`` used one exact ready label revision."""
    sidecar = Sidecar(directory)
    state = sidecar.load()
    if not state.revisions:
        raise TransitionRefused("the assessment has no label yet")
    moment = now()
    payload = {
        "label_id": state.revisions[0].event.payload["label_id"],
        "label_revision": label_revision,
        "consumer_id": consumer_id,
        "consumer_receipt": consumer_receipt,
        "consumed_at": moment,
    }
    return _append(
        sidecar,
        state,
        kind="label.consumed",
        actor={"role": "consumer", "subject": consumer_id},
        payload=payload,
        key=idempotency_key or f"label.consumed:{consumer_id}:{label_revision}",
        now=lambda: moment,
    )


def withdraw(
    directory: str | Path,
    *,
    label_revision: int,
    reason: str,
    disposition: str,
    actor: Mapping[str, str],
    idempotency_key: str | None = None,
    now: Clock = r.utc_now,
) -> Appended:
    """Bar a corrected or retracted revision from future use; record its consumers' disposition."""
    sidecar = Sidecar(directory)
    state = sidecar.load()
    if not state.revisions:
        raise TransitionRefused("the assessment has no label yet")
    if not 1 <= label_revision <= len(state.revisions):
        raise TransitionRefused(f"label revision {label_revision} does not exist")
    revision = state.revisions[label_revision - 1]
    moment = now()
    payload = {
        "label_id": state.revisions[0].event.payload["label_id"],
        "label_revision": label_revision,
        "reason": reason,
        "consumer_disposition": disposition,
        "consumption_event_ids": [c.id for c in revision.consumptions],
        "withdrawn_at": moment,
    }
    return _append(
        sidecar,
        state,
        kind="label.withdrawn",
        actor=actor,
        payload=payload,
        key=idempotency_key or f"label.withdrawn:{label_revision}:{disposition}",
        now=lambda: moment,
    )


def verify(
    directory: str | Path, *, evidence: bool = False, expect_event: str | None = None
) -> dict[str, Any]:
    """Verify the log (always), the label file, and optionally the evidence.

    ``expect_event`` is an event SHA-256 a receipt holds: the log must still
    contain it, so a log truncated or replaced since is refused.
    """
    sidecar = Sidecar(directory)
    if not sidecar.exists():
        raise _no_log(directory)
    state = sidecar.load()
    if expect_event is not None and expect_event not in {
        e.sha256 for e in state.events
    }:
        raise CorruptSidecar(
            f"the log does not contain event {expect_event}: it was truncated or replaced"
        )
    label = state.label()
    stored = sidecar.read_label_file()
    if label is None:
        label_file = "absent" if stored is None else "unexpected"
    elif stored is None:
        label_file = "missing"
    else:
        label_file = "current" if stored == canonicalize(label) + b"\n" else "stale"
    if evidence:
        _verify_evidence(state)
    report = state.summary()
    report.update(
        {
            "chain_verified": True,
            "label_file": label_file,
            "evidence_verified": evidence,
        }
    )
    return report

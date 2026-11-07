"""Why an assessment write or read was refused.

Every refusal appends nothing. ``code`` is stable for scripts (``bench
assessment ... --json`` prints it); the CLI exits 3 for a :class:`Conflict`
and 1 for any other refusal.
"""

from __future__ import annotations

from typing import ClassVar

from benchflow.errors import UserError

__all__ = [
    "AssessmentError",
    "Conflict",
    "CorruptSidecar",
    "EvidenceRejected",
    "InvalidRecord",
    "StaleExpectation",
    "TransitionRefused",
]


class AssessmentError(ValueError, UserError):
    """Base class: an assessment operation was refused and nothing was appended."""

    code: ClassVar[str] = "assessment.refused"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class InvalidRecord(AssessmentError):
    """A record, payload or input document breaks its schema."""

    code = "assessment.invalid_record"


class CorruptSidecar(AssessmentError):
    """The stored event log does not verify; nothing is materialized from it.

    A torn last line, a broken hash chain, a gap in ``event_seq``, a line that
    is not canonical JSON, an unknown schema version or an event the state
    machine forbids. The log is left exactly as found.
    """

    code = "assessment.corrupt_sidecar"


class EvidenceRejected(AssessmentError):
    """A saved result, output, review artifact, rubric, protocol or collector
    record is missing, changed, or belongs to another task or assessment."""

    code = "assessment.evidence_rejected"


class TransitionRefused(AssessmentError):
    """The event is not allowed in the assessment's current state."""

    code = "assessment.transition_refused"


class Conflict(AssessmentError):
    """The write conflicts with what the log already holds (HTTP 409).

    An idempotency key reused with another payload, a second answer for one
    assignment, or a different assignment for an assessment that has one.
    """

    code = "assessment.conflict"


class StaleExpectation(Conflict):
    """The writer's ``expected_event_seq``/``expected_previous_event_sha256``
    no longer names the log's last event: another write came first."""

    code = "assessment.stale_expectation"

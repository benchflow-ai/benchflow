"""Blind pairwise human assessments of saved outputs, kept beside them in a sidecar.

Two completed attempts of one task become one durable, hash-bound
assessment: a frozen rubric and judge protocol, a recorded release of every
execution resource, the review collector's persisted blind assignment and
vote, corrections and retractions, and a deterministic, separately versioned
label (``benchflow.assessment-label.v1``).

The sidecar is opt-in and separate. Nothing here writes or reads a reward:
``result.json`` (including its own ``assessment`` field, the outcome
assessment status of a trial), ``regrade.json`` and ``benchflow.rollout-stream.v1``
keep their meaning, and no 0.9 reader scans a sidecar. A missing sidecar is
not a pending review; a tie is ``0.5`` in the label only; an abstention is
``withheld``, never a zero.

Start with :func:`attempt_ref` and :func:`create`; ``bench assess``
wraps the same calls. docs/tasks/human-preferences.md walks through a full run.
"""

from benchflow.assessment.bridge import (
    Reduced,
    assign,
    consume,
    correct,
    create,
    import_vote,
    load,
    reduce,
    release,
    retract,
    verify,
    withdraw,
)
from benchflow.assessment.collector import export_record
from benchflow.assessment.errors import (
    AssessmentError,
    Conflict,
    CorruptSidecar,
    EvidenceRejected,
    InvalidRecord,
    StaleExpectation,
    TransitionRefused,
)
from benchflow.assessment.evidence import attempt_ref, parse_runtime_ref, verify_attempt
from benchflow.assessment.jcs import canonicalize, loads_strict
from benchflow.assessment.state import Assessment, Event, Revision
from benchflow.assessment.store import Appended, Sidecar

__all__ = [
    "Appended",
    "Assessment",
    "AssessmentError",
    "Conflict",
    "CorruptSidecar",
    "Event",
    "EvidenceRejected",
    "InvalidRecord",
    "Reduced",
    "Revision",
    "Sidecar",
    "StaleExpectation",
    "TransitionRefused",
    "assign",
    "attempt_ref",
    "canonicalize",
    "consume",
    "correct",
    "create",
    "export_record",
    "import_vote",
    "load",
    "loads_strict",
    "parse_runtime_ref",
    "reduce",
    "release",
    "retract",
    "verify",
    "verify_attempt",
    "withdraw",
]

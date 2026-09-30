"""Typed lifecycle events and their hash chain.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/events.py``). Every event carries the hash of the
one before it, and ``audit_manifest.json`` records the root, so an edited,
dropped or reordered event is detected when the stream is replayed.

The chain is unkeyed: it proves the stream is internally consistent, not who
wrote it. What keeps the agent from writing it is where it lives: BenchFlow
writes the events on the host, after the sandbox is gone, into the trial
folder the agent never had access to.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from benchflow.integrity.constants import (
    AUDIT_MANIFEST_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    PHASES,
    ActionClass,
    ActorClass,
    Phase,
    ResourceClass,
)

Decision = Literal["Allow", "Deny", "ObserveOnly", "Normalize", "N/A"]
AuditStatus = Literal["accepted", "rejected"]
TamperStatus = Literal["untampered", "tampered", "unknown"]

# The paper's reward-relevant event alphabet (section 6.5). Concrete events
# roll up to one of these symbols; ``Lifecycle`` covers trusted framework
# transitions that carry no authority crossing of their own.
AbstractEventType = Literal[
    "Expose",
    "Mutate",
    "Handoff",
    "Verify",
    "Reward",
    "Release",
    "SemanticWitness",
    "Lifecycle",
]

_ACTION_TO_ABSTRACT: dict[str, AbstractEventType] = {
    "Observe": "Expose",
    "Read": "Expose",
    "Download": "Handoff",
    "Write": "Mutate",
    "Upload": "Handoff",
    "ExtractArtifact": "Handoff",
    "BuildVerifierWorkspace": "Verify",
    "StartVerifier": "Verify",
    "NormalizeFailure": "Verify",
    "CollectReward": "Reward",
    "NetworkRequest": "Expose",
    "ReleaseFeedback": "Release",
    "SanitizeLog": "Release",
    "Reset": "Lifecycle",
    "CanonicalizePath": "Handoff",
    "PolicyDecision": "Lifecycle",
    "Execute": "Lifecycle",
}
_EVENT_TYPE_TO_ABSTRACT: dict[str, AbstractEventType] = {
    "ResetFromTrustedSnapshot": "Lifecycle",
    "ExtractArtifacts": "Handoff",
    "StartVerifier": "Verify",
    "StartSeparateVerifier": "Verify",
    "NormalizeFailure": "Verify",
    "CollectReward": "Reward",
    "ReleaseFeedback": "Release",
    "SanitizeLog": "Release",
}


def abstract_event_type(
    *,
    action_class: str,
    event_type: str,
    phase: str | None = None,
) -> AbstractEventType:
    """Map a concrete event onto the paper's alphabet.

    Transfer verbs describe backend operations, not authority crossings by
    themselves; the lifecycle phase tells setup uploads from declared
    extraction and verifier preparation.
    """

    if event_type in _EVENT_TYPE_TO_ABSTRACT:
        return _EVENT_TYPE_TO_ABSTRACT[event_type]
    if action_class == "Upload":
        if phase == "Verify":
            return "Verify"
        if phase == "Extract":
            return "Handoff"
        return "Handoff" if phase is None else "Lifecycle"
    if action_class == "Download":
        return "Handoff" if phase in {None, "Extract"} else "Lifecycle"
    if action_class == "Execute" and phase == "Verify":
        return "Verify"
    return _ACTION_TO_ABSTRACT.get(action_class, "Lifecycle")


REQUIRED_LIFECYCLE_EVENTS: tuple[str, ...] = (
    "InitProfile",
    "ValidateManifest",
    "CompilePolicy",
    "ValidatePackage",
    "SealRefinementMap",
    "ResetFromTrustedSnapshot",
    "StartAgentPhase",
    "EndAgentPhase",
    "ExtractArtifacts",
    "BuildVerifierWorkspace",
    "StartVerifier",
    "NormalizeFailure",
    "CollectReward",
    "ReleaseFeedback",
    "SanitizeLog",
    "AuditSeal",
    "ClaimVerdict",
)
REQUIRED_LIFECYCLE_PHASES: dict[str, Phase] = {
    "InitProfile": "Init",
    "ValidateManifest": "Init",
    "CompilePolicy": "Init",
    "ValidatePackage": "Init",
    "SealRefinementMap": "Init",
    "ResetFromTrustedSnapshot": "Reset",
    "StartAgentPhase": "Agent",
    "EndAgentPhase": "Agent",
    "ExtractArtifacts": "Extract",
    "BuildVerifierWorkspace": "Verify",
    "StartVerifier": "Verify",
    "NormalizeFailure": "Verify",
    "CollectReward": "Reward",
    "ReleaseFeedback": "Reward",
    "SanitizeLog": "Cleanup",
    "AuditSeal": "Done",
    "ClaimVerdict": "Done",
}
_REQUIRED_LIFECYCLE_ORDER = {
    event_type: index for index, event_type in enumerate(REQUIRED_LIFECYCLE_EVENTS)
}


class BenchGuardEvent(BaseModel):
    schema_version: Literal["benchguard.event.v0"] = EVENT_SCHEMA_VERSION
    event_id: str
    parent_event_id: str | None = None
    run_id: str
    sequence: int
    timestamp_unix_ms: int
    phase: Phase
    actor: str
    actor_class: ActorClass
    trust_domain: str
    event_type: str
    action_class: ActionClass
    # Rollup onto the paper's alphabet. Always re-derived, never trusted from
    # input, and excluded from the hash chain: BenchGuard found that honoring
    # a stamped value let one edited field flip a rejected trace to accepted
    # while every hash still matched.
    abstract_event_type: AbstractEventType | None = None
    resource: str | None = None
    resource_class: ResourceClass | None = None
    decision: Decision = "N/A"
    effect: str = "N/A"
    policy_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    prev_hash: str | None = None
    event_hash: str | None = None

    @model_validator(mode="after")
    def _derive_abstract_event_type(self) -> BenchGuardEvent:
        object.__setattr__(
            self,
            "abstract_event_type",
            abstract_event_type(
                action_class=self.action_class,
                event_type=self.event_type,
                phase=self.phase,
            ),
        )
        return self


class AuditManifest(BaseModel):
    schema_version: Literal["benchguard.audit_manifest.v0"] = (
        AUDIT_MANIFEST_SCHEMA_VERSION
    )
    run_id: str
    events_file: str = "events.jsonl"
    event_count: int
    first_event_id: str | None = None
    last_event_id: str | None = None
    hash_chain_root: str | None = None
    writer_identity: str = "TrustedHost"
    tamper_status: TamperStatus = "unknown"
    missing_required_events: list[str] = Field(default_factory=list)
    unknown_event_count: int = 0
    phase_summary: dict[str, str] = Field(default_factory=dict)


class AuditValidationReport(BaseModel):
    status: AuditStatus
    violations: list[str] = Field(default_factory=list)
    missing_required_events: list[str] = Field(default_factory=list)
    unknown_event_count: int = 0
    event_count: int = 0
    hash_chain_root: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def canonical_event_payload(event: BenchGuardEvent) -> str:
    """Canonical JSON used for the hash chain."""

    data = event.model_dump(mode="json", exclude={"event_hash", "abstract_event_type"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_event_hash(event: BenchGuardEvent) -> str:
    digest = hashlib.sha256(canonical_event_payload(event).encode()).hexdigest()
    return f"sha256:{digest}"


def chain_events(events: list[BenchGuardEvent]) -> list[BenchGuardEvent]:
    """Fill ``prev_hash`` and ``event_hash`` for an ordered event list."""

    chained: list[BenchGuardEvent] = []
    prev_hash: str | None = None
    for event in events:
        prepared = event.model_copy(update={"prev_hash": prev_hash, "event_hash": None})
        hashed = prepared.model_copy(
            update={"event_hash": compute_event_hash(prepared)}
        )
        chained.append(hashed)
        prev_hash = hashed.event_hash
    return chained


def validate_event_stream(
    events: list[BenchGuardEvent],
    *,
    audit_manifest: AuditManifest | None = None,
    require_lifecycle_events: bool = True,
) -> AuditValidationReport:
    """Validate ordering, hash-chain integrity and lifecycle completeness."""

    violations: list[str] = []
    if not events:
        violations.append("empty_event_stream")

    event_ids: set[str] = set()
    run_ids = {event.run_id for event in events}
    if len(run_ids) > 1:
        violations.append("mixed_run_ids")

    prev_hash: str | None = None
    lifecycle_counts: dict[str, int] = {}
    last_lifecycle_index = -1
    for index, event in enumerate(events):
        if event.event_id in event_ids:
            violations.append(f"duplicate_event_id:{event.event_id}")
        event_ids.add(event.event_id)
        if event.sequence != index + 1:
            violations.append(f"non_contiguous_sequence:{event.event_id}")
        if event.prev_hash != prev_hash:
            violations.append(f"prev_hash_mismatch:{event.event_id}")
        if event.event_hash != compute_event_hash(event):
            violations.append(f"event_hash_mismatch:{event.event_id}")
        prev_hash = event.event_hash
        if require_lifecycle_events and event.event_type in _REQUIRED_LIFECYCLE_ORDER:
            lifecycle_counts[event.event_type] = (
                lifecycle_counts.get(event.event_type, 0) + 1
            )
            expected_phase = REQUIRED_LIFECYCLE_PHASES[event.event_type]
            if event.phase != expected_phase:
                violations.append(
                    "lifecycle_phase_mismatch:"
                    f"{event.event_id}:{event.event_type}:{event.phase}!={expected_phase}"
                )
            lifecycle_index = _REQUIRED_LIFECYCLE_ORDER[event.event_type]
            if lifecycle_index < last_lifecycle_index:
                violations.append(
                    f"lifecycle_order_violation:{event.event_id}:{event.event_type}"
                )
            last_lifecycle_index = max(last_lifecycle_index, lifecycle_index)

    observed_types = {event.event_type for event in events}
    missing = (
        sorted(set(REQUIRED_LIFECYCLE_EVENTS) - observed_types)
        if require_lifecycle_events
        else []
    )
    violations.extend(f"missing_required_event:{event_type}" for event_type in missing)
    if require_lifecycle_events:
        for event_type, count in sorted(lifecycle_counts.items()):
            if count > 1:
                violations.append(f"duplicate_required_event:{event_type}")
        if events and events[-1].event_type != "ClaimVerdict":
            violations.append("terminal_event_not_claim_verdict")

    if audit_manifest is not None:
        if len(run_ids) == 1 and audit_manifest.run_id != next(iter(run_ids)):
            violations.append("audit_manifest_run_id_mismatch")
        if audit_manifest.event_count != len(events):
            violations.append("audit_manifest_event_count_mismatch")
        if events and audit_manifest.first_event_id != events[0].event_id:
            violations.append("audit_manifest_first_event_id_mismatch")
        if events and audit_manifest.last_event_id != events[-1].event_id:
            violations.append("audit_manifest_last_event_id_mismatch")
        if audit_manifest.writer_identity != "TrustedHost":
            violations.append("audit_manifest_writer_identity")
        if audit_manifest.tamper_status != "untampered":
            violations.append("audit_manifest_tamper_status")
        if events and audit_manifest.hash_chain_root != events[-1].event_hash:
            violations.append("audit_manifest_hash_chain_root_mismatch")
        if audit_manifest.missing_required_events:
            violations.append("audit_manifest_missing_required_events")
        if (
            require_lifecycle_events
            and sorted(audit_manifest.missing_required_events) != missing
        ):
            violations.append("audit_manifest_missing_required_events_mismatch")
        if audit_manifest.unknown_event_count:
            violations.append("audit_manifest_unknown_events")

    status: AuditStatus = "rejected" if violations else "accepted"
    return AuditValidationReport(
        status=status,
        violations=violations,
        missing_required_events=missing,
        unknown_event_count=audit_manifest.unknown_event_count
        if audit_manifest is not None
        else 0,
        event_count=len(events),
        hash_chain_root=events[-1].event_hash if events else None,
    )


def build_audit_manifest(
    events: list[BenchGuardEvent],
    *,
    tamper_status: TamperStatus = "untampered",
    unknown_event_count: int = 0,
) -> AuditManifest:
    """A compact audit summary for an already chained event stream."""

    observed_types = {event.event_type for event in events}
    missing = sorted(set(REQUIRED_LIFECYCLE_EVENTS) - observed_types)
    observed_phases = {event.phase for event in events}
    return AuditManifest(
        run_id=events[0].run_id if events else "",
        event_count=len(events),
        first_event_id=events[0].event_id if events else None,
        last_event_id=events[-1].event_id if events else None,
        hash_chain_root=events[-1].event_hash if events else None,
        tamper_status=tamper_status,
        missing_required_events=missing,
        unknown_event_count=unknown_event_count,
        phase_summary={
            phase: "observed" if phase in observed_phases else "missing"
            for phase in PHASES
        },
    )


__all__ = [
    "AbstractEventType",
    "AuditManifest",
    "AuditValidationReport",
    "BenchGuardEvent",
    "REQUIRED_LIFECYCLE_EVENTS",
    "abstract_event_type",
    "build_audit_manifest",
    "chain_events",
    "compute_event_hash",
    "validate_event_stream",
]

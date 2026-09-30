"""Replay a lifecycle event stream into violation flags.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/trace_checker.py``); only the imports changed. The
checker is a pure function of the events: it keeps state across phases (the
handoff set built during Extract is consulted during Verify), which is what
makes the I3 rule a relation between events rather than a per-event test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from benchflow.integrity.authority import (
    authority_of,
    crossing_is_forbidden,
)
from benchflow.integrity.claim import FinalFlags
from benchflow.integrity.constants import (
    CONFORMANCE_SCHEMA_VERSION,
    FEEDBACK_CLASSES,
    LOG_CLASSES,
    NETWORK_CLASSES,
    REWARD_SOURCES,
    SAFE_FEEDBACK_CLASSES,
    SAFE_LOG_CLASSES,
    TRACE_CHECK_SCHEMA_VERSION,
    VERIFIER_FAILURE_STATUSES,
    VERIFIER_STATUSES,
    ActionClass,
    RewardSource,
    VerifierStatus,
)
from benchflow.integrity.events import (
    AuditManifest,
    AuditStatus,
    BenchGuardEvent,
    abstract_event_type,
    validate_event_stream,
)

# Authority domains whose state the outcome procedure may legitimately read as
# input (paper Algorithm 2: outcome_input is a subset of handoff-set plus trusted state). Agent-owned
# state must instead arrive through a declared handoff, tracked across phases.
_TRUSTED_OUTCOME_AUTHORITIES: frozenset[str] = frozenset(
    {"OutcomeOwned", "TrustedInput", "DeclaredHandoff"}
)

ACCESS_ACTIONS: frozenset[ActionClass] = frozenset(
    {"Observe", "Read", "Execute", "Download"}
)
MUTATION_ACTIONS: frozenset[ActionClass] = frozenset({"Write", "Upload"})
PATH_CLASSIFICATION_ACTIONS: frozenset[ActionClass] = frozenset(
    {
        "CanonicalizePath",
        "ExtractArtifact",
        "BuildVerifierWorkspace",
        "Upload",
        "Download",
    }
)
REWARD_ACTORS = frozenset({"TrustedHost", "Verifier"})


@dataclass
class _ReplayState:
    """Running state the conformance replay carries across lifecycle phases.

    This is what makes the checker stronger than per-event auditing (paper
    §6.7): the ``handoff`` set is built during the Extract/Handoff phase and
    consulted during the Verify phase, so a cross-phase property like
    "outcome input ⊆ handoff ∪ trusted state" (I3) is decided by the *relation*
    between events, not by any single event in isolation.
    """

    handoff: set[str] = field(default_factory=set)
    outcome_inputs: set[str] = field(default_factory=set)
    exposed: set[str] = field(default_factory=set)
    mutated: set[str] = field(default_factory=set)
    verifier_status: VerifierStatus = "NotRun"
    reward_source: RewardSource = "None"


def _abstract_of(event: BenchGuardEvent) -> str:
    """The event's paper-alphabet symbol, deriving it if not stamped."""

    stamped = getattr(event, "abstract_event_type", None)
    if stamped:
        return stamped
    return abstract_event_type(
        action_class=event.action_class,
        event_type=event.event_type,
        phase=event.phase,
    )


def _resource_key(event: BenchGuardEvent) -> str | None:
    if event.resource:
        return event.resource
    if event.resource_class:
        return f"class:{event.resource_class}"
    return None


class TraceCheckReport(BaseModel):
    schema_version: str = TRACE_CHECK_SCHEMA_VERSION
    conformance_schema_version: str = CONFORMANCE_SCHEMA_VERSION
    status: AuditStatus
    reasons: list[str] = Field(default_factory=list)
    agent_violation_evidence: list[str] = Field(default_factory=list)
    final_flags: FinalFlags = Field(default_factory=FinalFlags)
    event_count: int = 0
    verifier_status: VerifierStatus = "NotRun"
    reward_source: RewardSource = "None"

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def check_benchguard_trace(
    events: Iterable[BenchGuardEvent | Mapping[str, Any]],
    *,
    audit_manifest: AuditManifest | Mapping[str, Any] | None = None,
    require_lifecycle_events: bool = False,
) -> TraceCheckReport:
    """Classify a BenchGuard event stream into final conformance flags.

    The checker accepts typed ``BenchGuardEvent`` objects or event dictionaries.
    Hash-chain/lifecycle validation is delegated to ``validate_event_stream``
    when an audit manifest is supplied, lifecycle validation is requested, or
    the event stream already carries hash-chain fields.
    """

    flags = FinalFlags()
    reasons: list[str] = []
    normalized_events = _coerce_events(events, flags=flags, reasons=reasons)

    audit = _coerce_audit_manifest(audit_manifest, flags=flags, reasons=reasons)
    if (
        audit is not None
        or require_lifecycle_events
        or _has_hash_chain_fields(normalized_events)
    ):
        audit_report = validate_event_stream(
            normalized_events,
            audit_manifest=audit,
            require_lifecycle_events=require_lifecycle_events,
        )
        if not audit_report.accepted:
            _set_flag(
                flags,
                reasons,
                "audit_tampering",
                "audit:validation_failed",
            )
            reasons.extend(
                f"audit:{violation}" for violation in audit_report.violations
            )
        if audit_report.unknown_event_count:
            _set_flag(
                flags,
                reasons,
                "unknown_event",
                f"audit:unknown_event_count:{audit_report.unknown_event_count}",
            )

    state = _ReplayState()
    agent_violation_evidence: list[str] = []
    for event in normalized_events:
        observed_status = _verifier_status_from_event(event)
        if observed_status is not None:
            state.verifier_status = observed_status

        observed_source = _reward_source_from_event(event)
        if observed_source is not None:
            state.reward_source = observed_source

        _check_agent_protected_access(
            event,
            state=state,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _check_agent_protected_mutation(
            event,
            state=state,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _check_path_escape(
            event,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _replay_handoff_and_outcome_input(
            event,
            state=state,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _check_forbidden_network(
            event,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _check_reward_event(
            event,
            verifier_status=state.verifier_status,
            flags=flags,
            reasons=reasons,
            agent_violation_evidence=agent_violation_evidence,
        )
        _check_logs_and_feedback(event, flags=flags, reasons=reasons)
        _check_unknown_reward_relevant_object(event, flags=flags, reasons=reasons)

    status: AuditStatus = "rejected" if flags.true_flags() else "accepted"
    if status == "accepted" and not reasons:
        reasons.append("checked")
    return TraceCheckReport(
        status=status,
        reasons=reasons,
        agent_violation_evidence=agent_violation_evidence,
        final_flags=flags,
        event_count=len(normalized_events),
        verifier_status=state.verifier_status,
        reward_source=state.reward_source,
    )


def check_trace(
    events: Iterable[BenchGuardEvent | Mapping[str, Any]],
    *,
    audit_manifest: AuditManifest | Mapping[str, Any] | None = None,
    require_lifecycle_events: bool = False,
) -> TraceCheckReport:
    """Short alias for ``check_benchguard_trace``."""

    return check_benchguard_trace(
        events,
        audit_manifest=audit_manifest,
        require_lifecycle_events=require_lifecycle_events,
    )


def _coerce_events(
    events: Iterable[BenchGuardEvent | Mapping[str, Any]],
    *,
    flags: FinalFlags,
    reasons: list[str],
) -> list[BenchGuardEvent]:
    normalized: list[BenchGuardEvent] = []
    for index, raw_event in enumerate(events, start=1):
        if isinstance(raw_event, BenchGuardEvent):
            normalized.append(raw_event)
            continue
        if isinstance(raw_event, Mapping):
            try:
                normalized.append(BenchGuardEvent.model_validate(raw_event))
            except ValidationError as exc:
                _set_flag(
                    flags,
                    reasons,
                    "unknown_event",
                    f"event:{_event_label(raw_event, index)}:validation_error:{_validation_summary(exc)}",
                )
            continue
        _set_flag(
            flags,
            reasons,
            "unknown_event",
            f"event:index-{index}:unsupported_type:{type(raw_event).__name__}",
        )
    return normalized


def _coerce_audit_manifest(
    audit_manifest: AuditManifest | Mapping[str, Any] | None,
    *,
    flags: FinalFlags,
    reasons: list[str],
) -> AuditManifest | None:
    if audit_manifest is None:
        return None
    if isinstance(audit_manifest, AuditManifest):
        return audit_manifest
    try:
        return AuditManifest.model_validate(audit_manifest)
    except ValidationError as exc:
        _set_flag(
            flags,
            reasons,
            "audit_tampering",
            f"audit:manifest_validation_error:{_validation_summary(exc)}",
        )
        return None


def _has_hash_chain_fields(events: list[BenchGuardEvent]) -> bool:
    return any(
        event.prev_hash is not None or event.event_hash is not None for event in events
    )


def _check_agent_protected_access(
    event: BenchGuardEvent,
    *,
    state: _ReplayState,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    # Authority-first (paper §6.2): an agent role observing a resource whose
    # authority domain it does not own is a forbidden crossing (I1). Applies
    # to any protected authority (outcome-owned state incl. other roles'
    # state, reward output), not the verifier alone. Trusted/public inputs are
    # agent-observable and correctly do not flag here.
    if not (
        event.action_class in ACCESS_ACTIONS
        and event.resource_class is not None
        and _not_explicitly_denied(event)
    ):
        return
    key = _resource_key(event)
    if event.actor_class == "Agent" and key is not None:
        state.exposed.add(key)
    if crossing_is_forbidden(
        actor_class=event.actor_class,
        resource_class=event.resource_class,
        write=False,
    ):
        _set_flag(
            flags,
            reasons,
            "hidden_observation",
            f"event:{event.event_id}:agent_accessed_protected_resource:"
            f"{event.resource_class}:{authority_of(event.resource_class)}",
        )
        _add_agent_violation_evidence(
            agent_violation_evidence,
            event,
            f"accessed_protected_resource:{event.resource_class}",
        )


def _check_agent_protected_mutation(
    event: BenchGuardEvent,
    *,
    state: _ReplayState,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    if (
        event.action_class in MUTATION_ACTIONS
        and event.actor_class == "Agent"
        and _not_explicitly_denied(event)
    ):
        key = _resource_key(event)
        if key is not None:
            state.mutated.add(key)
    if not (
        event.action_class in MUTATION_ACTIONS
        and event.resource_class is not None
        and crossing_is_forbidden(
            actor_class=event.actor_class,
            resource_class=event.resource_class,
            write=True,
        )
        and _not_explicitly_denied(event)
    ):
        return

    if event.resource_class == "VerifierOnly":
        _set_flag(
            flags,
            reasons,
            "verifier_mutation",
            f"event:{event.event_id}:agent_mutated_verifier_resource",
        )
        _add_agent_violation_evidence(
            agent_violation_evidence,
            event,
            "mutated_verifier_resource",
        )
        return

    if event.resource_class == "RewardOutput":
        _set_flag(
            flags,
            reasons,
            "untrusted_reward_source",
            f"event:{event.event_id}:agent_mutated_reward_output",
        )
        _add_agent_violation_evidence(
            agent_violation_evidence,
            event,
            "mutated_reward_output",
        )
        return

    _set_flag(
        flags,
        reasons,
        "trusted_mutation",
        f"event:{event.event_id}:agent_mutated_protected_resource:{event.resource_class}",
    )
    _add_agent_violation_evidence(
        agent_violation_evidence,
        event,
        f"mutated_protected_resource:{event.resource_class}",
    )
    if event.resource_class == "ResetState":
        _set_flag(
            flags,
            reasons,
            "reset_contamination",
            f"event:{event.event_id}:agent_mutated_reset_state",
        )


def _check_unknown_reward_relevant_object(
    event: BenchGuardEvent,
    *,
    flags: FinalFlags,
    reasons: list[str],
) -> None:
    """Fail closed on a reward-relevant object the contract cannot classify.

    The event synthesizer marks a reward-relevant action whose resource does
    not resolve to any declared class (paper I5: unknown ⇒ fail closed). We
    surface it on the ``unknown_event`` evidence-gap flag so a checked claim
    is blocked rather than the object being treated as benign.
    """

    if _payload_bool(event.payload, ("unknown_reward_relevant_object",)):
        _set_flag(
            flags,
            reasons,
            "unknown_event",
            f"event:{event.event_id}:unknown_reward_relevant_object",
        )


def _replay_handoff_and_outcome_input(
    event: BenchGuardEvent,
    *,
    state: _ReplayState,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    """Stateful I3 (paper Algorithm 2): outcome input is a subset of handoff-set plus trusted state.

    A declared ``Handoff`` event admits its object into the running handoff
    set. A ``Handoff`` or ``Verify`` event that routes an *agent-owned* object
    into the outcome procedure is a violation unless that object first crossed
    a declared handoff — a relation between two events across phases, decided
    from the carried set rather than from either event alone.
    """

    if not _not_explicitly_denied(event):
        return
    abstract = _abstract_of(event)
    # Only Extract/Verify-phase handoffs and reads route state *into* the
    # outcome procedure. Setup/agent-phase uploads (instruction, workspace,
    # oracle) are also ``Handoff``-abstract but flow host→agent, not into the
    # verifier, so they are not outcome inputs.
    if abstract not in {"Handoff", "Verify"} or event.phase not in {
        "Extract",
        "Verify",
    }:
        return

    key = _resource_key(event)
    resource_class = event.resource_class
    authority = authority_of(resource_class) if resource_class else None
    undeclared = (
        resource_class == "UndeclaredArtifact"
        or _payload_has_undeclared_artifact(event.payload)
    )

    # A trusted-host (or verifier) declared extraction admits its object into
    # the handoff set — this is the framework moving a declared verifier-input
    # into the outcome workspace. The declaration is at the artifact level, so
    # any non-undeclared Handoff extraction qualifies regardless of the
    # resource's own class (a declared artifact may be an agent-writable file).
    if (
        abstract == "Handoff"
        and key is not None
        and not undeclared
        and event.actor_class in {"TrustedHost", "Verifier"}
    ):
        state.handoff.add(key)

    if key is not None:
        state.outcome_inputs.add(key)

    agent_owned = authority == "AgentOwned"
    crossed_declared = key is not None and key in state.handoff
    trusted_input = authority in _TRUSTED_OUTCOME_AUTHORITIES
    agent_routed = event.actor_class == "Agent"

    # I3: the object reaches outcome computation without crossing a declared
    # handoff — either an explicitly undeclared artifact, an agent-owned object
    # the trusted host never extracted, or the agent routing its own state in.
    if (
        undeclared
        or (agent_owned and not crossed_declared and not trusted_input)
        or (agent_routed and not crossed_declared)
    ):
        _set_flag(
            flags,
            reasons,
            "undeclared_artifact_cross",
            f"event:{event.event_id}:outcome_input_not_in_handoff_set:"
            f"{key or resource_class}",
        )
        if event.actor_class == "Agent":
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                "undeclared_artifact_crossed",
            )


def _check_path_escape(
    event: BenchGuardEvent,
    *,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    if (
        event.action_class not in PATH_CLASSIFICATION_ACTIONS
        and event.event_type not in {"CanonicalizePath", "ExtractArtifacts"}
    ):
        return
    if _payload_bool(event.payload, ("path_escape_detected", "path_escape")):
        _set_flag(
            flags,
            reasons,
            "path_escape",
            f"event:{event.event_id}:path_escape_detected",
        )
        if event.actor_class == "Agent" and _not_explicitly_denied(event):
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                "path_escape_detected",
            )
        return
    if _payload_has_rejection_reason(event.payload, "path_escape"):
        _set_flag(
            flags,
            reasons,
            "path_escape",
            f"event:{event.event_id}:path_escape_rejected",
        )
        if event.actor_class == "Agent" and _not_explicitly_denied(event):
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                "path_escape_rejected",
            )


def _check_forbidden_network(
    event: BenchGuardEvent,
    *,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    if event.action_class != "NetworkRequest" or event.decision != "Allow":
        return
    if _forbidden_network_represented(event):
        _set_flag(
            flags,
            reasons,
            "forbidden_network",
            f"event:{event.event_id}:allowed_forbidden_network",
        )
        if event.actor_class == "Agent":
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                "allowed_forbidden_network",
            )


def _check_reward_event(
    event: BenchGuardEvent,
    *,
    verifier_status: VerifierStatus,
    flags: FinalFlags,
    reasons: list[str],
    agent_violation_evidence: list[str],
) -> None:
    if event.action_class != "CollectReward" and event.event_type != "CollectReward":
        return

    if event.actor_class not in REWARD_ACTORS and _not_explicitly_denied(event):
        _set_flag(
            flags,
            reasons,
            "untrusted_reward_source",
            f"event:{event.event_id}:reward_collected_by:{event.actor_class}",
        )
        if event.actor_class == "Agent":
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                "agent_collected_reward",
            )

    raw_source = _payload_string(event.payload, ("reward_source", "source"))
    if raw_source is not None and raw_source != "TrustedVerifier":
        _set_flag(
            flags,
            reasons,
            "untrusted_reward_source",
            f"event:{event.event_id}:reward_source:{raw_source}",
        )
        if event.actor_class == "Agent":
            _add_agent_violation_evidence(
                agent_violation_evidence,
                event,
                f"reward_source:{raw_source}",
            )

    status = _verifier_status_from_event(event) or verifier_status
    if status in VERIFIER_FAILURE_STATUSES and _event_reward_is_pass(event):
        _set_flag(
            flags,
            reasons,
            "fail_open",
            f"event:{event.event_id}:reward_pass_after_verifier_failure:{status}",
        )


def _check_logs_and_feedback(
    event: BenchGuardEvent,
    *,
    flags: FinalFlags,
    reasons: list[str],
) -> None:
    if event.action_class == "SanitizeLog" or event.event_type == "SanitizeLog":
        log_class = _payload_string(
            event.payload,
            ("log_class", "released_log_class", "log_release", "released_class"),
        )
        if log_class is not None and not _safe_log_class(log_class):
            _set_flag(
                flags,
                reasons,
                "log_leak",
                f"event:{event.event_id}:released_log_class:{log_class}",
            )

    if event.action_class == "ReleaseFeedback" or event.event_type == "ReleaseFeedback":
        feedback_class = _payload_string(
            event.payload,
            ("feedback_class", "feedback_mode", "released_feedback_class"),
        )
        if feedback_class is not None and not _safe_feedback_class(feedback_class):
            _set_flag(
                flags,
                reasons,
                "feedback_policy_violation",
                f"event:{event.event_id}:feedback_class:{feedback_class}",
            )


def _reward_source_from_event(event: BenchGuardEvent) -> RewardSource | None:
    source = _payload_string(event.payload, ("reward_source", "source"))
    if source in REWARD_SOURCES:
        return cast("RewardSource", source)
    return None


def _verifier_status_from_event(event: BenchGuardEvent) -> VerifierStatus | None:
    status = _payload_string(
        event.payload,
        (
            "normalized_verifier_status",
            "verifier_status",
            "normalized_status",
            "raw_verifier_status",
            "raw_status",
        ),
    )
    if status in VERIFIER_STATUSES:
        return cast("VerifierStatus", status)
    if event.effect in VERIFIER_STATUSES:
        return cast("VerifierStatus", event.effect)
    return None


def _event_reward_is_pass(event: BenchGuardEvent) -> bool:
    payload = event.payload
    for key in ("reward_verdict", "verdict", "outcome", "reward"):
        if key not in payload:
            continue
        value = payload[key]
        if _value_is_pass(value):
            return True
        if isinstance(value, Mapping):
            nested_reward = value.get("reward")
            if _value_is_pass(nested_reward):
                return True
    rewards = payload.get("rewards")
    if isinstance(rewards, Mapping) and _value_is_pass(rewards.get("reward")):
        return True
    return _value_is_pass(event.effect)


def _value_is_pass(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "success", "true", "accepted"}:
            return True
        try:
            return float(normalized) > 0
        except ValueError:
            return False
    return False


def _payload_has_undeclared_artifact(payload: Mapping[str, Any]) -> bool:
    direct_keys = (
        "artifact_class",
        "artifact_classes",
        "resource_class",
        "resource_classes",
        "source_class",
        "source_classes",
        "included_resource_classes",
        "verifier_workspace_classes",
        "extracted_artifact_classes",
    )
    if any(
        _contains_value(payload.get(key), "UndeclaredArtifact") for key in direct_keys
    ):
        return True

    undeclared_artifacts = payload.get("undeclared_artifacts")
    if isinstance(undeclared_artifacts, list | tuple | set | frozenset):
        return bool(undeclared_artifacts)
    return isinstance(undeclared_artifacts, Mapping) and bool(undeclared_artifacts)


def _forbidden_network_represented(event: BenchGuardEvent) -> bool:
    network_class = _payload_string(
        event.payload,
        (
            "network_class",
            "egress_class",
            "task_egress",
            "policy_network_class",
            "destination_class",
        ),
    )
    if network_class == "NoEgress":
        return True
    if network_class is not None and network_class not in NETWORK_CLASSES:
        lowered = network_class.lower()
        if lowered in {"forbidden", "blocked", "no-egress", "no_egress", "noegress"}:
            return True

    resource = (event.resource or "").lower().replace("_", "-")
    return "noegress" in resource or "no-egress" in resource or "forbidden" in resource


def _payload_string(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _payload_bool(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    return False


def _payload_has_rejection_reason(payload: Mapping[str, Any], reason: str) -> bool:
    rejections = payload.get("rejections")
    if rejections is None:
        rejections = payload.get("rejected_paths")
    if not isinstance(rejections, list | tuple):
        return False
    for item in rejections:
        if item == reason:
            return True
        if isinstance(item, Mapping) and item.get("reason") == reason:
            return True
    return False


def _contains_value(value: object, needle: str) -> bool:
    if value == needle:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, needle) for item in value.values())
    if isinstance(value, list | tuple | set | frozenset):
        return any(_contains_value(item, needle) for item in value)
    return False


def _safe_log_class(value: str) -> bool:
    return value in LOG_CLASSES and value in SAFE_LOG_CLASSES


def _safe_feedback_class(value: str) -> bool:
    return value in FEEDBACK_CLASSES and value in SAFE_FEEDBACK_CLASSES


def _not_explicitly_denied(event: BenchGuardEvent) -> bool:
    return event.decision != "Deny"


def _set_flag(
    flags: FinalFlags,
    reasons: list[str],
    flag: str,
    reason: str,
) -> None:
    setattr(flags, flag, True)
    if reason not in reasons:
        reasons.append(reason)


def _add_agent_violation_evidence(
    evidence: list[str],
    event: BenchGuardEvent,
    detail: str,
) -> None:
    item = f"event:{event.event_id}:{detail}"
    if item not in evidence:
        evidence.append(item)


def _event_label(raw_event: Mapping[str, Any], index: int) -> str:
    event_id = raw_event.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id
    return f"index-{index}"


def _validation_summary(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error.get("loc", ())) or "event"
    return f"{location}:{first_error.get('msg', 'invalid')}"

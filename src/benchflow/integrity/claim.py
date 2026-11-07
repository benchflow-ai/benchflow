"""Claim verdicts: the flags a trace raised, and the four core verdicts.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/conformance.py`` and ``claim_engine.py``), trimmed
to the runtime lane: no semantic witnesses and no static taint report. Static
vectors here are the run-profile vectors BenchFlow knows for certain (for
example a verifier that shares the agent's sandbox).

The four core verdicts (paper section 6.6):

- ``Checked``: the contract is accepted, the trace conforms, and the evidence
  a certified claim needs is present. Open review points keep the finer
  certification at ``Conditional``.
- ``Rejected``: a violation or an evidence gap without agent attribution;
  the claim is refused rather than overclaimed (the paper's "Inconclusive").
- ``VectorExposed``: the run's profile has a known path to the reward, and
  nothing shows this run used it. Not agent blame.
- ``AgentViolation``: direct agent-attributed evidence of a forbidden
  crossing.

Precedence is AgentViolation > Rejected > VectorExposed > Checked.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from benchflow.integrity.constants import (
    CLAIM_VERDICT_SCHEMA_VERSION,
    CORE_CLAIM_SCHEMA_VERSION,
    ClaimLabel,
    CoreVerdict,
    EventSeverity,
    RejectionCategory,
    RewardSource,
    TaskOutcome,
    VerifierStatus,
    ViolationSource,
)


class FinalFlags(BaseModel):
    hidden_observation: bool = False
    trusted_mutation: bool = False
    verifier_mutation: bool = False
    undeclared_artifact_cross: bool = False
    verifier_input_missing: bool = False
    artifact_policy_rejected: bool = False
    path_escape: bool = False
    untrusted_reward_source: bool = False
    fail_open: bool = False
    reset_contamination: bool = False
    forbidden_network: bool = False
    network_policy_gap: bool = False
    agent_monitor_gap: bool = False
    log_leak: bool = False
    feedback_policy_violation: bool = False
    unknown_event: bool = False
    audit_tampering: bool = False

    def true_flags(self) -> list[str]:
        return [
            name
            for name, value in self.model_dump().items()
            if isinstance(value, bool) and value
        ]

    def true_vector_flags(self) -> list[str]:
        """Exercised-vector flags, without evidence-integrity gaps."""
        return [flag for flag in self.true_flags() if flag in _VECTOR_FLAGS]


_VECTOR_FLAGS = frozenset(
    {
        "hidden_observation",
        "trusted_mutation",
        "verifier_mutation",
        "undeclared_artifact_cross",
        "artifact_policy_rejected",
        "path_escape",
        "untrusted_reward_source",
        "fail_open",
        "reset_contamination",
        "forbidden_network",
        "log_leak",
        "feedback_policy_violation",
    }
)
_EVIDENCE_FLAGS = frozenset(
    {
        "verifier_input_missing",
        "network_policy_gap",
        "agent_monitor_gap",
        "unknown_event",
        "audit_tampering",
    }
)


class ClaimVerdict(BaseModel):
    schema_version: str = CLAIM_VERDICT_SCHEMA_VERSION
    claim_label: ClaimLabel
    task_outcome: TaskOutcome = "TaskError"
    certification: ClaimLabel
    reasons: list[str] = Field(default_factory=list)
    rejection_categories: list[RejectionCategory] = Field(default_factory=list)
    source_labels: list[ViolationSource] = Field(default_factory=list)
    event_severity: EventSeverity = "None"
    final_flags: FinalFlags = Field(default_factory=FinalFlags)
    verifier_status: VerifierStatus = "NotRun"
    reward_source: RewardSource = "None"


def decide_claim_verdict(
    *,
    contract_label: ClaimLabel,
    contract_violations: list[str],
    audit_violations: list[str],
    final_flags: FinalFlags | None = None,
    agent_violation_evidence: list[str] | None = None,
    verifier_status: VerifierStatus = "NotRun",
    reward_source: RewardSource = "None",
) -> ClaimVerdict:
    """Combine contract, event-stream and trace evidence into a claim label."""

    flags = final_flags or FinalFlags()
    direct_agent_evidence = list(agent_violation_evidence or [])
    reasons: list[str] = []
    label: ClaimLabel = contract_label
    task_outcome = _task_outcome(verifier_status)

    if label in {"Rejected", "AuditOnly", "OutsideClaim"}:
        reasons.append(f"manifest_label:{label}")
    reasons.extend(f"manifest:{code}" for code in contract_violations)
    if audit_violations:
        label = _downgrade_to_rejected(label)
        reasons.extend(f"audit:{violation}" for violation in audit_violations)
    true_flags = flags.true_flags()
    if true_flags:
        label = _downgrade_to_rejected(label)
        reasons.extend(f"flag:{flag}" for flag in true_flags)
    if direct_agent_evidence:
        label = _downgrade_to_rejected(label)
        reasons.extend(f"agent_violation:{item}" for item in direct_agent_evidence)
    if task_outcome == "TaskError":
        label = _downgrade_to_rejected(label)
        reasons.append(f"verifier_status:{verifier_status}")
    if reward_source != "TrustedVerifier":
        label = _downgrade_to_rejected(label)
        reasons.append(f"reward_source:{reward_source}")
    if label == "Checked" and not reasons:
        reasons.append("checked")

    categories = _rejection_categories(
        label=label, reasons=reasons, flags=true_flags, reward_source=reward_source
    )
    sources = _source_labels(
        reasons=reasons,
        flags=true_flags,
        reward_source=reward_source,
        agent_violation=bool(direct_agent_evidence),
    )
    return ClaimVerdict(
        claim_label=label,
        task_outcome=task_outcome,
        certification=label,
        reasons=reasons,
        rejection_categories=categories,
        source_labels=sources,
        event_severity=_event_severity(
            categories=categories, task_outcome=task_outcome, sources=sources
        ),
        final_flags=flags,
        verifier_status=verifier_status,
        reward_source=reward_source,
    )


class CoreClaim(BaseModel):
    """The single-axis verdict plus its refinements."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CORE_CLAIM_SCHEMA_VERSION
    core_verdict: CoreVerdict
    certification: ClaimLabel
    task_outcome: TaskOutcome = "TaskError"
    static_vector_classes: list[str] = Field(default_factory=list)
    agent_evidence: list[str] = Field(default_factory=list)
    i7_obligations: list[str] = Field(default_factory=list)
    rejection_categories: list[RejectionCategory] = Field(default_factory=list)
    source_labels: list[ViolationSource] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def core_verdict_from_labels(
    *,
    certification: ClaimLabel,
    static_vector_classes: list[str] | None = None,
    agent_evidence: list[str] | None = None,
    violation_flags: list[str] | None = None,
    structural_rejection: bool = False,
    evidence_gap: bool = False,
) -> CoreVerdict:
    """Map the finer certification vocabulary onto the four-verdict axis.

    A run-time violation flag means a vector was exercised in this run
    (``Rejected``, or ``AgentViolation`` with agent attribution); a static
    vector with no run-time flag stays ``VectorExposed``. A rejection rooted
    in the contract itself stays ``Rejected`` even beside static vectors.

    Deviation from BenchGuard: an evidence gap (``evidence_gap``: a missing,
    partial or scraped agent trajectory, an unclassifiable reward-relevant
    object, a broken event stream) is ``Rejected`` even beside static
    vectors. ``VectorExposed`` says the run shows no use of the vector, which
    means nothing when the run's evidence is incomplete; BenchGuard let the
    static vector win there.
    """

    vectors = static_vector_classes or []
    if agent_evidence:
        return "AgentViolation"
    if violation_flags or evidence_gap:
        return "Rejected"
    if certification == "Rejected":
        if vectors and not structural_rejection:
            return "VectorExposed"
        return "Rejected"
    if certification in {"AuditOnly", "OutsideClaim"}:
        return "Rejected"
    if vectors:
        return "VectorExposed"
    return "Checked"


def decide_core_claim(
    *,
    claim_verdict: ClaimVerdict,
    static_vector_classes: list[str],
    i7_obligations: list[str],
) -> CoreClaim:
    """Combine run-time conformance with the run's static vectors.

    Static vectors are not certifiable, so they join the certification as
    ``Rejected`` (BenchGuard's static report with a counterexample does the
    same); open I7 obligations join it as ``Conditional``.
    """

    agent_evidence = _dedupe(
        [
            reason.removeprefix("agent_violation:")
            for reason in claim_verdict.reasons
            if reason.startswith("agent_violation:")
        ]
    )
    certification: ClaimLabel = claim_verdict.certification
    if static_vector_classes:
        certification = _combine(certification, "Rejected")
    if i7_obligations:
        certification = _combine(certification, "Conditional")
    structural_rejection = (
        claim_verdict.certification == "Rejected"
        and "TaskContractViolation" in claim_verdict.rejection_categories
    )
    evidence_flags = [
        flag
        for flag in claim_verdict.final_flags.true_flags()
        if flag in _EVIDENCE_FLAGS
    ]
    core = core_verdict_from_labels(
        certification=certification,
        static_vector_classes=static_vector_classes,
        agent_evidence=agent_evidence,
        violation_flags=claim_verdict.final_flags.true_vector_flags(),
        structural_rejection=structural_rejection,
        evidence_gap=bool(evidence_flags),
    )
    reasons = _dedupe(
        [
            *claim_verdict.reasons,
            *(f"static_vector:{vector}" for vector in static_vector_classes),
        ]
    )
    return CoreClaim(
        core_verdict=core,
        certification=certification,
        task_outcome=claim_verdict.task_outcome,
        static_vector_classes=list(static_vector_classes),
        agent_evidence=agent_evidence,
        i7_obligations=list(i7_obligations),
        rejection_categories=list(claim_verdict.rejection_categories),
        source_labels=list(claim_verdict.source_labels),
        reasons=reasons,
    )


def _combine(left: ClaimLabel, right: ClaimLabel) -> ClaimLabel:
    order: dict[ClaimLabel, int] = {
        "Checked": 0,
        "Conditional": 1,
        "OutsideClaim": 2,
        "AuditOnly": 3,
        "Rejected": 4,
    }
    return left if order[left] >= order[right] else right


def _downgrade_to_rejected(label: ClaimLabel) -> ClaimLabel:
    if label in {"AuditOnly", "OutsideClaim"}:
        return label
    return "Rejected"


def _task_outcome(verifier_status: VerifierStatus) -> TaskOutcome:
    if verifier_status == "Pass":
        return "TaskPass"
    if verifier_status == "Fail":
        return "TaskFail"
    return "TaskError"


def _dedupe[T](items: list[T]) -> list[T]:
    deduped: list[T] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _rejection_categories(
    *,
    label: ClaimLabel,
    reasons: list[str],
    flags: list[str],
    reward_source: RewardSource,
) -> list[RejectionCategory]:
    categories: list[RejectionCategory] = []
    if label == "OutsideClaim" or "manifest_label:OutsideClaim" in reasons:
        categories.append("UnsupportedProfile")
    if any(reason.startswith("manifest:") for reason in reasons):
        categories.append("TaskContractViolation")
    if (
        any(flag in _VECTOR_FLAGS for flag in flags)
        or reward_source == "UntrustedAgent"
    ):
        categories.append("VectorExposed")
    if any(flag in _EVIDENCE_FLAGS for flag in flags) or any(
        reason.startswith(("audit:", "verifier_status:")) for reason in reasons
    ):
        categories.append("EvidenceGap")
    return _dedupe(categories)


def _source_labels(
    *,
    reasons: list[str],
    flags: list[str],
    reward_source: RewardSource,
    agent_violation: bool,
) -> list[ViolationSource]:
    sources: list[ViolationSource] = []
    if agent_violation:
        sources.append("AgentViolation")
    if any(reason.startswith("manifest:") for reason in reasons):
        sources.append("TaskContractViolation")
    if (
        any(flag in _VECTOR_FLAGS for flag in flags)
        or reward_source == "UntrustedAgent"
    ):
        sources.append("FrameworkViolation")
    if any(flag in _EVIDENCE_FLAGS for flag in flags) or any(
        reason.startswith("audit:") for reason in reasons
    ):
        sources.append("EvidenceMissing")
    if any(reason.startswith("verifier_status:") for reason in reasons):
        sources.append("VerifierAssumptionGap")
    if not sources and any(reason != "checked" for reason in reasons):
        sources.append("Unknown")
    return _dedupe(sources)


def _event_severity(
    *,
    categories: list[RejectionCategory],
    task_outcome: TaskOutcome,
    sources: list[ViolationSource],
) -> EventSeverity:
    if "AgentViolation" in sources and task_outcome == "TaskPass":
        return "RewardRelevant"
    if "AgentViolation" in sources:
        return "Delivered"
    if "VectorExposed" in categories and task_outcome == "TaskPass":
        return "RewardRelevant"
    if "VectorExposed" in categories:
        return "Delivered"
    return "None"


__all__ = [
    "ClaimVerdict",
    "CoreClaim",
    "FinalFlags",
    "core_verdict_from_labels",
    "decide_claim_verdict",
    "decide_core_claim",
]

"""The finite vocabulary of BenchFlow's reward-integrity layer.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/constants.py``), the implementation of BenchShield
(arXiv 2609.11028). The labels and schema ids are BenchGuard's, so evidence
written by BenchFlow can be read by BenchGuard's own tools.
"""

from __future__ import annotations

from typing import Literal

PROFILE_ID = "BenchGuard-SeparatedVerifier-v0"
MANIFEST_SCHEMA_VERSION = "benchguard.manifest.v0"
EVENT_SCHEMA_VERSION = "benchguard.event.v0"
AUDIT_MANIFEST_SCHEMA_VERSION = "benchguard.audit_manifest.v0"
CONFORMANCE_SCHEMA_VERSION = "benchguard.conformance.v0"
CLAIM_VERDICT_SCHEMA_VERSION = "benchguard.claim_verdict.v0"
CORE_CLAIM_SCHEMA_VERSION = "benchguard.core_claim.v0"
ACTION_RECORD_SCHEMA_VERSION = "benchguard.action_record.v1"
TRACE_CHECK_SCHEMA_VERSION = "benchguard.trace_check.v0"
REWARD_PROVENANCE_SCHEMA_VERSION = "benchguard.reward_provenance.v0"
HANDOFF_SCHEMA_VERSION = "benchflow.integrity.handoff.v0"

type IntegrityMode = Literal["off", "audit", "strict"]
INTEGRITY_MODES: tuple[IntegrityMode, ...] = ("off", "audit", "strict")

type ActorClass = Literal["Agent", "TrustedHost", "Verifier"]
type ResourceClass = Literal[
    "Public",
    "AgentWritable",
    "DeclaredArtifact",
    "UndeclaredArtifact",
    "Trusted",
    "VerifierOnly",
    "Hidden",
    "ResetState",
    "RewardOutput",
]
type ActionClass = Literal[
    "Observe",
    "Read",
    "Write",
    "Execute",
    "Upload",
    "Download",
    "CanonicalizePath",
    "ExtractArtifact",
    "BuildVerifierWorkspace",
    "StartVerifier",
    "NormalizeFailure",
    "CollectReward",
    "NetworkRequest",
    "ReleaseFeedback",
    "SanitizeLog",
    "Reset",
    "PolicyDecision",
]
type Phase = Literal[
    "Init", "Reset", "Agent", "Extract", "Verify", "Reward", "Cleanup", "Done"
]
type NetworkClass = Literal[
    "NoEgress", "InternalOnly", "ProviderEgress", "Allowlist", "Full", "LoggedOnly"
]
type RewardSource = Literal["None", "TrustedVerifier", "UntrustedAgent"]
type VerifierStatus = Literal["NotRun", "Pass", "Fail", "Crash", "Timeout", "Malformed"]
type FeedbackClass = Literal[
    "None", "TerminalOnly", "Scalar", "Summary", "FullPrivate", "FullPublic"
]
type LogClass = Literal["None", "InternalOnly", "SanitizedPublic", "RawPublic"]
type ClaimLabel = Literal[
    "Checked", "Conditional", "AuditOnly", "OutsideClaim", "Rejected"
]
# One certification axis, four core verdicts (paper section 6.6).
type CoreVerdict = Literal["Checked", "Rejected", "VectorExposed", "AgentViolation"]
type TaskOutcome = Literal["TaskPass", "TaskFail", "TaskError"]
type RejectionCategory = Literal[
    "VectorExposed",
    "EvidenceGap",
    "VerifierAssumptionGap",
    "TaskContractViolation",
    "UnsupportedProfile",
]
type ViolationSource = Literal[
    "AgentViolation",
    "FrameworkViolation",
    "TaskContractViolation",
    "VerifierAssumptionGap",
    "EvidenceMissing",
    "Unknown",
]
type EventSeverity = Literal["None", "BlockedAttempt", "Delivered", "RewardRelevant"]

ACTOR_CLASSES: tuple[ActorClass, ...] = ("Agent", "TrustedHost", "Verifier")
RESOURCE_CLASSES: tuple[ResourceClass, ...] = (
    "Public",
    "AgentWritable",
    "DeclaredArtifact",
    "UndeclaredArtifact",
    "Trusted",
    "VerifierOnly",
    "Hidden",
    "ResetState",
    "RewardOutput",
)
ACTION_CLASSES: tuple[ActionClass, ...] = (
    "Observe",
    "Read",
    "Write",
    "Execute",
    "Upload",
    "Download",
    "CanonicalizePath",
    "ExtractArtifact",
    "BuildVerifierWorkspace",
    "StartVerifier",
    "NormalizeFailure",
    "CollectReward",
    "NetworkRequest",
    "ReleaseFeedback",
    "SanitizeLog",
    "Reset",
    "PolicyDecision",
)
PHASES: tuple[Phase, ...] = (
    "Init",
    "Reset",
    "Agent",
    "Extract",
    "Verify",
    "Reward",
    "Cleanup",
    "Done",
)
NETWORK_CLASSES: tuple[NetworkClass, ...] = (
    "NoEgress",
    "InternalOnly",
    "ProviderEgress",
    "Allowlist",
    "Full",
    "LoggedOnly",
)
REWARD_SOURCES: tuple[RewardSource, ...] = ("None", "TrustedVerifier", "UntrustedAgent")
VERIFIER_STATUSES: tuple[VerifierStatus, ...] = (
    "NotRun",
    "Pass",
    "Fail",
    "Crash",
    "Timeout",
    "Malformed",
)
FEEDBACK_CLASSES: tuple[FeedbackClass, ...] = (
    "None",
    "TerminalOnly",
    "Scalar",
    "Summary",
    "FullPrivate",
    "FullPublic",
)
LOG_CLASSES: tuple[LogClass, ...] = (
    "None",
    "InternalOnly",
    "SanitizedPublic",
    "RawPublic",
)
CORE_VERDICTS: tuple[CoreVerdict, ...] = (
    "Checked",
    "Rejected",
    "VectorExposed",
    "AgentViolation",
)

PROTECTED_RESOURCE_CLASSES: frozenset[ResourceClass] = frozenset(
    {"Hidden", "VerifierOnly", "Trusted", "ResetState", "RewardOutput"}
)
VERIFIER_FAILURE_STATUSES: frozenset[VerifierStatus] = frozenset(
    {"Fail", "Crash", "Timeout", "Malformed"}
)
SAFE_FEEDBACK_CLASSES: frozenset[FeedbackClass] = frozenset(
    {"None", "TerminalOnly", "Scalar", "Summary", "FullPrivate"}
)
SAFE_LOG_CLASSES: frozenset[LogClass] = frozenset(
    {"None", "InternalOnly", "SanitizedPublic"}
)

"""Build one trial's integrity evidence and verdict, and write ``integrity/``.

The pipeline is BenchGuard's runtime lane (arXiv 2609.11028;
``src/benchflow/benchguard/runtime_verification.py``), cut down to the
evidence BenchFlow 0.8 has on every backend:

    contract + host-recorded ACP trajectory + host lane (verifier outcome,
    separate-verifier transfer record, egress proxy denials)
        -> action records -> typed, hash-chained events
        -> trace check -> claim verdict -> core verdict

The record-to-event helpers (resource resolution, derived exec targets,
loopback reads, the fail-closed opaque-exec rule) are ported from BenchGuard;
what is new is where the evidence comes from.

Everything here reads files and objects the host wrote. Nothing is read back
from the sandbox, and nothing here changes a reward.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from benchflow.integrity.agent_actions import (
    action_records_from_trajectory,
    redacted_preview,
)
from benchflow.integrity.claim import (
    decide_claim_verdict,
    decide_core_claim,
)
from benchflow.integrity.classify import (
    classify_resource_class,
    normalize_manifest_resources,
)
from benchflow.integrity.constants import (
    ACTION_CLASSES,
    ACTION_RECORD_SCHEMA_VERSION,
    ACTOR_CLASSES,
    CONFORMANCE_SCHEMA_VERSION,
    HANDOFF_SCHEMA_VERSION,
    PHASES,
    PROFILE_ID,
    RESOURCE_CLASSES,
    REWARD_PROVENANCE_SCHEMA_VERSION,
    ActionClass,
    ActorClass,
    Phase,
    ResourceClass,
    VerifierStatus,
)
from benchflow.integrity.contract import (
    IntegrityContract,
    contract_label,
    i7_obligations,
)
from benchflow.integrity.events import (
    BenchGuardEvent,
    build_audit_manifest,
    chain_events,
    validate_event_stream,
)
from benchflow.integrity.exec_decompose import is_loopback_host
from benchflow.integrity.trace_checker import check_benchguard_trace

INTEGRITY_DIRNAME = "integrity"
CLAIM_VERDICT_FILE = "claim_verdict.json"
SCRIPTED_AGENTS = frozenset({"oracle", "nop"})

# Outcome-owned and reward classes: an action on their roots is
# reward-relevant even when it resolves to no declared resource (fail closed).
_PROTECTED_OUTCOME_CLASSES = frozenset(
    {"VerifierOnly", "Hidden", "ResetState", "RewardOutput"}
)
# Static vectors of the run's profile, with the dimensions they expose.
PROFILE_VECTORS = {
    "shared_verifier": "I2,I3: the verifier runs in the sandbox the agent used; "
    "pre-verifier hardening is a temporal cleanup boundary, not an independent "
    "verifier environment",
    "agent_root": "I1,I2: the agent runs as root, so the locked task paths are "
    "not a boundary",
}


@dataclass
class TrialEvidence:
    """Everything the host knows about one finished trial."""

    trial_dir: Path
    run_id: str
    mode: str
    contract: IntegrityContract
    agent: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    trajectory_source: str | None = None
    rewards: dict[str, Any] | None = None
    verifier_error: str | None = None
    verifier_error_category: str | None = None
    error: str | None = None
    sandbox_user: str | None = "agent"
    environment: str = "docker"
    separate_verifier: bool = False
    separate_verifier_record: dict[str, Any] | None = None
    egress_denials: list[dict[str, Any]] = field(default_factory=list)
    source: str = "rollout"


def emit_integrity(evidence: TrialEvidence) -> dict[str, Any]:
    """Write ``integrity/`` for one trial and return the claim verdict."""

    out = evidence.trial_dir / INTEGRITY_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    contract = evidence.contract
    _write_json(out / "manifest.json", contract.model_dump(mode="json", by_alias=True))

    reward_provenance = _reward_provenance(evidence)
    _write_json(out / "reward_provenance.json", reward_provenance)
    handoff = _handoff_record(evidence)
    if handoff is not None:
        _write_json(out / "handoff.json", handoff)

    scripted = evidence.agent in SCRIPTED_AGENTS
    agent_lane = (
        action_records_from_trajectory(
            evidence.trajectory,
            agent_cwd=contract.workspace,
            manifest_resources=contract.resource_pairs(),
            network_class=contract.network.task_egress,
        )
        if not scripted
        else None
    )
    host_records = _egress_denial_records(evidence.egress_denials)
    records = _normalize_action_records(
        [*(agent_lane.records if agent_lane else []), *host_records]
    )
    _write_jsonl(out / "action_records.jsonl", records)

    events = _runtime_events(
        run_id=evidence.run_id,
        evidence=evidence,
        reward_provenance=reward_provenance,
        handoff=handoff,
        action_records=records,
    )
    _write_jsonl(
        out / "events.jsonl", [event.model_dump(mode="json") for event in events]
    )
    audit_manifest = build_audit_manifest(events)
    _write_json(out / "audit_manifest.json", audit_manifest.model_dump(mode="json"))
    audit_report = validate_event_stream(events, audit_manifest=audit_manifest)
    trace = check_benchguard_trace(
        events, audit_manifest=audit_manifest, require_lifecycle_events=True
    )
    _write_json(out / "trace_check.json", trace.model_dump(mode="json"))

    gaps = _evidence_gaps(evidence, scripted=scripted, handoff=handoff)
    flags = trace.final_flags.model_copy(
        update={
            "unknown_event": trace.final_flags.unknown_event or bool(gaps["unknown"]),
            "verifier_input_missing": bool(gaps["verifier_input"]),
            "artifact_policy_rejected": bool(gaps["artifact_policy"]),
            "network_policy_gap": bool(gaps["network"]),
            "audit_tampering": trace.final_flags.audit_tampering
            or not audit_report.accepted,
        }
    )
    label, violations = contract_label(contract)
    verdict = decide_claim_verdict(
        contract_label=label,
        contract_violations=violations,
        audit_violations=[] if audit_report.accepted else audit_report.violations,
        final_flags=flags,
        agent_violation_evidence=trace.agent_violation_evidence,
        verifier_status=cast("VerifierStatus", reward_provenance["verifier_status"]),
        reward_source=reward_provenance["reward_source"],
    )
    for reason in (
        *gaps["unknown"],
        *gaps["verifier_input"],
        *gaps["artifact_policy"],
        *gaps["network"],
    ):
        if reason not in verdict.reasons:
            verdict.reasons.append(reason)
    vectors = profile_vectors(evidence)
    core = decide_core_claim(
        claim_verdict=verdict,
        static_vector_classes=vectors,
        i7_obligations=i7_obligations(contract),
    )
    exploited = core.core_verdict == "AgentViolation"
    reason = _headline_reason(
        core.core_verdict, core.agent_evidence, events, gaps, vectors
    )
    claim = {
        **verdict.model_dump(mode="json"),
        "core": core.model_dump(mode="json"),
        "core_verdict": core.core_verdict,
        "exploited": exploited,
        "reason": reason,
        "mode": evidence.mode,
        # Strict runs the verifier where the agent never was; audit only
        # observes the run as the task ships it.
        "claim_posture": "enforced" if evidence.mode == "strict" else "observed",
        "separate_verifier": evidence.separate_verifier,
        "profile_vectors": {name: PROFILE_VECTORS[name] for name in vectors},
        "reward": _scalar_reward(evidence.rewards),
        # BenchFlow never changes a reward because of this verdict.
        "reward_effect": "none",
        "source": evidence.source,
    }
    conformance = {
        "schema_version": CONFORMANCE_SCHEMA_VERSION,
        "profile": PROFILE_ID,
        "run_id": evidence.run_id,
        "mode": evidence.mode,
        "claim_posture": claim["claim_posture"],
        "status": "accepted" if audit_report.accepted else "rejected",
        "claim_label": verdict.claim_label,
        "task_outcome": verdict.task_outcome,
        "certification": core.certification,
        "core_verdict": core.core_verdict,
        "core": core.model_dump(mode="json"),
        "verifier_status": reward_provenance["verifier_status"],
        "reward_source": reward_provenance["reward_source"],
        "contract": {"claim_label": label, "violations": violations},
        "audit": audit_report.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "final_flags": flags.model_dump(mode="json"),
        "agent_violation_evidence": trace.agent_violation_evidence,
        "evidence_gaps": gaps,
        "agent_action_lane": agent_lane.summary if agent_lane else {"scripted": True},
        "host_lane": {"egress_denials": len(host_records)},
        "trajectory_source": evidence.trajectory_source,
        "handoff": handoff,
    }
    _write_json(out / "conformance.json", conformance)
    _write_json(out / CLAIM_VERDICT_FILE, claim)
    return claim


def emit_integrity_error(
    trial_dir: Path, *, mode: str, error: BaseException, source: str = "rollout"
) -> dict[str, Any]:
    """Fail closed when building the evidence itself fails."""

    out = trial_dir / INTEGRITY_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    reason = f"integrity_error:{type(error).__name__}"
    claim = {
        "schema_version": "benchguard.claim_verdict.v0",
        "claim_label": "Rejected",
        "certification": "Rejected",
        "task_outcome": "TaskError",
        "reasons": [reason],
        "rejection_categories": ["EvidenceGap"],
        "source_labels": ["EvidenceMissing"],
        "core_verdict": "Rejected",
        "core": {
            "core_verdict": "Rejected",
            "certification": "Rejected",
            "task_outcome": "TaskError",
            "reasons": [reason],
        },
        "exploited": False,
        "reason": f"the integrity audit failed ({type(error).__name__}: {error}); "
        "no claim is made",
        "mode": mode,
        "claim_posture": "enforced" if mode == "strict" else "observed",
        "reward_effect": "none",
        "error": {"type": type(error).__name__, "message": str(error)[:500]},
        "source": source,
    }
    _write_json(out / CLAIM_VERDICT_FILE, claim)
    return claim


def profile_vectors(evidence: TrialEvidence) -> list[str]:
    """Static vectors of the run's profile that BenchFlow knows for certain."""

    vectors: list[str] = []
    if not evidence.separate_verifier:
        vectors.append("shared_verifier")
    if evidence.sandbox_user in {None, "", "root", "0"}:
        vectors.append("agent_root")
    return vectors


# --- evidence pieces ---------------------------------------------------------


def verifier_status(rewards: dict[str, Any] | None, verifier_error: str | None) -> str:
    """BenchGuard's normalized verifier status."""

    if verifier_error:
        lowered = verifier_error.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return "Timeout"
        if "malformed" in lowered:
            return "Malformed"
        return "Crash"
    if rewards is None:
        return "NotRun"
    reward = rewards.get("reward")
    if isinstance(reward, int | float) and not isinstance(reward, bool) and reward > 0:
        return "Pass"
    return "Fail"


def _reward_provenance(evidence: TrialEvidence) -> dict[str, Any]:
    status = verifier_status(evidence.rewards, evidence.verifier_error)
    verifier_dir = evidence.trial_dir / "verifier"
    reward_file = next(
        (
            verifier_dir / name
            for name in ("reward.txt", "reward.json")
            if (verifier_dir / name).is_file()
        ),
        None,
    )
    trusted = (
        evidence.rewards is not None
        and evidence.verifier_error is None
        and reward_file is not None
    )
    return {
        "schema_version": REWARD_PROVENANCE_SCHEMA_VERSION,
        "profile": PROFILE_ID,
        "run_id": evidence.run_id,
        "verifier_status": status,
        "reward_source": "TrustedVerifier" if trusted else "None",
        "rewards": evidence.rewards,
        "reward_file": str(reward_file.relative_to(evidence.trial_dir))
        if reward_file is not None
        else None,
        "reward_file_sha256": _sha256_file(reward_file),
        "verifier_error": evidence.verifier_error,
        "verifier_error_category": evidence.verifier_error_category,
        "separate_verifier": evidence.separate_verifier,
    }


def _handoff_record(evidence: TrialEvidence) -> dict[str, Any] | None:
    """What crossed into the separate verifier, from the host's own record."""

    if not evidence.separate_verifier:
        return None
    record = evidence.separate_verifier_record or {}
    transfer = (
        record.get("transfer") if isinstance(record.get("transfer"), dict) else {}
    )
    transfer = transfer or {}
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "source": "verifier-sandbox/verifier-sandbox.json",
        "record_present": bool(record),
        "status": record.get("status"),
        "image_source": record.get("image_source"),
        "refusal": record.get("refusal"),
        "error": record.get("error"),
        "workspace": transfer.get("workspace"),
        "agent_paths": list(transfer.get("agent_paths") or []),
        "declared_artifacts": transfer.get("declared_artifacts"),
        "missing_artifacts": transfer.get("missing_artifacts"),
        "files": transfer.get("files"),
        "bytes": transfer.get("bytes"),
        "exclusions": transfer.get("exclusions"),
        "logs_artifacts_files": transfer.get("logs_artifacts_files"),
    }


def _evidence_gaps(
    evidence: TrialEvidence, *, scripted: bool, handoff: dict[str, Any] | None
) -> dict[str, list[str]]:
    """Evidence the claim needs and this run does not have, by flag."""

    unknown: list[str] = []
    verifier_input: list[str] = []
    artifact_policy: list[str] = []
    network: list[str] = []
    if not scripted:
        # TaskRuntime records each bash call on the host as it runs, so an
        # empty list there means the policy did nothing, not a lost stream.
        host_recorded = evidence.agent == "task-runtime"
        if evidence.trajectory_source == "scraped":
            # Read back from the agent's own files: forgeable, so no evidence.
            unknown.append("gap:agent_trajectory_untrusted_scraped")
        elif not evidence.trajectory and not host_recorded:
            # Nothing was recorded (the run failed before the agent, or the
            # stream was lost): there is nothing to judge the agent by.
            unknown.append("gap:agent_trajectory_missing")
        elif evidence.trajectory_source == "partial_acp":
            unknown.append("gap:agent_trajectory_partial")
    if evidence.separate_verifier:
        status = (handoff or {}).get("status")
        if not (handoff or {}).get("record_present"):
            verifier_input.append("gap:separate_verifier_record_missing")
        elif status == "refused":
            artifact_policy.append("handoff:refused")
        elif status in {"transfer_failed", "sandbox_failed", "interrupted", "pending"}:
            verifier_input.append(f"gap:separate_verifier_{status}")
    contract = evidence.contract
    if (
        contract.network.task_egress == "NoEgress"
        and contract.network.agent_network_mode != "no-network"
    ):
        # The contract forbids agent egress but the sandbox allowed it: a
        # NoEgress claim with no boundary behind it (BenchGuard's
        # no_container_network_boundary_for_task_egress).
        network.append("gap:no_network_boundary_for_declared_no_egress")
    return {
        "unknown": unknown,
        "verifier_input": verifier_input,
        "artifact_policy": artifact_policy,
        "network": network,
    }


def _egress_denial_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent network attempts the egress proxy refused (host-side log).

    The proxy filters the agent's uid only, so each row is an agent attempt.
    A denied attempt is evidence, never a violation by itself.
    """

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or row.get("server_name") or ""
        records.append(
            {
                "schema_version": ACTION_RECORD_SCHEMA_VERSION,
                "operation": "egress_proxy.refused",
                "event_type": "AgentNetworkDenied",
                "action_class": "NetworkRequest",
                "phase": "Agent",
                "actor": "actor.agent",
                "actor_class": "Agent",
                "trust_domain": "Untrusted",
                "attribution_source": "egress_proxy_log",
                "status": "denied",
                "resource": "network:web",
                "url": redacted_preview(str(url), limit=240),
                "rule": row.get("rule"),
                "method": row.get("method"),
            }
        )
    return records


# --- events ------------------------------------------------------------------

_LIFECYCLE_PREFIX: list[tuple[Phase, str, ActorClass, str, ActionClass]] = [
    ("Init", "actor.host", "TrustedHost", "InitProfile", "PolicyDecision"),
    ("Init", "actor.host", "TrustedHost", "ValidateManifest", "PolicyDecision"),
    ("Init", "actor.host", "TrustedHost", "CompilePolicy", "PolicyDecision"),
    ("Init", "actor.host", "TrustedHost", "ValidatePackage", "PolicyDecision"),
    ("Init", "actor.host", "TrustedHost", "SealRefinementMap", "PolicyDecision"),
    ("Reset", "actor.host", "TrustedHost", "ResetFromTrustedSnapshot", "Reset"),
    ("Agent", "actor.agent", "Agent", "StartAgentPhase", "Execute"),
    ("Agent", "actor.agent", "Agent", "EndAgentPhase", "Execute"),
    ("Extract", "actor.host", "TrustedHost", "ExtractArtifacts", "ExtractArtifact"),
    (
        "Verify",
        "actor.host",
        "TrustedHost",
        "BuildVerifierWorkspace",
        "BuildVerifierWorkspace",
    ),
    ("Verify", "actor.verifier", "Verifier", "StartVerifier", "StartVerifier"),
    ("Verify", "actor.host", "TrustedHost", "NormalizeFailure", "NormalizeFailure"),
    ("Reward", "actor.host", "TrustedHost", "CollectReward", "CollectReward"),
    ("Reward", "actor.host", "TrustedHost", "ReleaseFeedback", "ReleaseFeedback"),
    ("Cleanup", "actor.host", "TrustedHost", "SanitizeLog", "SanitizeLog"),
]
_LIFECYCLE_SUFFIX: list[tuple[Phase, str, ActorClass, str, ActionClass]] = [
    ("Done", "actor.host", "TrustedHost", "AuditSeal", "PolicyDecision"),
    ("Done", "actor.host", "TrustedHost", "ClaimVerdict", "PolicyDecision"),
]


def _runtime_events(
    *,
    run_id: str,
    evidence: TrialEvidence,
    reward_provenance: dict[str, Any],
    handoff: dict[str, Any] | None,
    action_records: list[dict[str, Any]],
) -> list[BenchGuardEvent]:
    contract = evidence.contract
    now_ms = int(time.time() * 1000)
    events: list[BenchGuardEvent] = []

    def lifecycle(spec: tuple[Phase, str, ActorClass, str, ActionClass]) -> None:
        phase, actor, actor_class, event_type, action_class = spec
        events.append(
            BenchGuardEvent(
                event_id=f"evt-{len(events) + 1:06d}",
                parent_event_id=f"evt-{len(events):06d}" if events else None,
                run_id=run_id,
                sequence=len(events) + 1,
                timestamp_unix_ms=now_ms + len(events) + 1,
                phase=phase,
                actor=actor,
                actor_class=actor_class,
                trust_domain="Untrusted" if actor_class == "Agent" else "Trusted",
                event_type=event_type,
                action_class=action_class,
                resource=_lifecycle_resource(event_type, contract),
                resource_class=_lifecycle_resource_class(event_type, contract),
                decision="Normalize" if event_type == "NormalizeFailure" else "Allow",
                effect=_lifecycle_effect(
                    event_type, evidence, reward_provenance, handoff
                ),
                payload=_lifecycle_payload(
                    event_type, evidence, reward_provenance, handoff
                ),
            )
        )

    for spec in _LIFECYCLE_PREFIX:
        lifecycle(spec)
    for record in action_records:
        events.append(
            _action_record_event(
                record=record,
                run_id=run_id,
                sequence=len(events) + 1,
                now_ms=now_ms,
                contract=contract,
            )
        )
        # Each decomposed exec target becomes its own Read/Write event, so
        # `cat > /tests/x <<EOF` is a classified write, not an opaque exec.
        for sub_record in (
            _derived_target_records(record)
            + _derived_network_records(record)
            + _derived_loopback_records(record)
        ):
            events.append(
                _action_record_event(
                    record=sub_record,
                    run_id=run_id,
                    sequence=len(events) + 1,
                    now_ms=now_ms,
                    contract=contract,
                )
            )
    for spec in _LIFECYCLE_SUFFIX:
        lifecycle(spec)
    return chain_events(events)


def _lifecycle_resource(event_type: str, contract: IntegrityContract) -> str | None:
    if event_type in {"ExtractArtifacts", "BuildVerifierWorkspace"}:
        return contract.artifacts[0].resource if contract.artifacts else None
    if event_type == "CollectReward":
        for resource in contract.resources:
            if resource.resource_class == "RewardOutput":
                return resource.id
    return None


def _lifecycle_resource_class(
    event_type: str, contract: IntegrityContract
) -> ResourceClass | None:
    resource_id = _lifecycle_resource(event_type, contract)
    for resource in contract.resources:
        if resource.id == resource_id:
            return resource.resource_class
    return None


def _lifecycle_effect(
    event_type: str,
    evidence: TrialEvidence,
    reward_provenance: dict[str, Any],
    handoff: dict[str, Any] | None,
) -> str:
    if event_type == "InitProfile":
        return "separate_verifier" if evidence.separate_verifier else "shared_verifier"
    if event_type == "ExtractArtifacts":
        return str((handoff or {}).get("status") or "shared_workspace")
    if event_type == "NormalizeFailure":
        return str(reward_provenance.get("verifier_status", "NotRun"))
    if event_type == "CollectReward":
        return str(evidence.rewards)
    return "recorded"


def _lifecycle_payload(
    event_type: str,
    evidence: TrialEvidence,
    reward_provenance: dict[str, Any],
    handoff: dict[str, Any] | None,
) -> dict[str, Any]:
    if event_type == "InitProfile":
        return {
            "integrity_mode": evidence.mode,
            "environment": evidence.environment,
            "sandbox_user": evidence.sandbox_user,
            "agent": evidence.agent,
        }
    if event_type == "ExtractArtifacts" and handoff is not None:
        return {
            "handoff_status": handoff.get("status"),
            "transferred_paths": handoff.get("agent_paths"),
            "excluded": handoff.get("exclusions"),
        }
    if event_type == "NormalizeFailure":
        return {
            "normalized_verifier_status": reward_provenance.get("verifier_status"),
            "verifier_error_category": reward_provenance.get("verifier_error_category"),
        }
    if event_type == "CollectReward":
        return {
            "reward_source": reward_provenance.get("reward_source"),
            "verifier_status": reward_provenance.get("verifier_status"),
            "rewards": evidence.rewards or {},
        }
    if event_type == "ReleaseFeedback":
        # BenchFlow releases the verifier's output only after the agent stops.
        return {"feedback_class": "TerminalOnly"}
    if event_type == "SanitizeLog":
        return {"released_log_class": "InternalOnly"}
    return {}


def _action_record_event(
    *,
    record: dict[str, Any],
    run_id: str,
    sequence: int,
    now_ms: int,
    contract: IntegrityContract,
) -> BenchGuardEvent:
    resource = _action_record_resource(record, contract)
    resource_class = _action_record_resource_class(record, contract, resource)
    payload = _action_record_payload(record)
    # Fail closed (I5): a reward-relevant action the contract cannot classify
    # blocks a checked claim instead of reading as benign.
    if resource_class is None and _action_record_is_reward_relevant(record, contract):
        payload = {
            **payload,
            "unknown_reward_relevant_object": True,
            "unclassified_path": str(
                record.get("target_path") or record.get("source_path") or ""
            ),
        }
    if record.get("attribution_gap") and _action_record_is_reward_relevant(
        record, contract
    ):
        payload = {**payload, "unknown_reward_relevant_object": True}
    opaque_reference = _opaque_exec_protected_reference(record, contract)
    if opaque_reference is not None:
        payload = {
            **payload,
            "unknown_reward_relevant_object": True,
            "opaque_exec_protected_reference": opaque_reference,
        }
    return BenchGuardEvent(
        event_id=f"evt-{sequence:06d}",
        parent_event_id=f"evt-{sequence - 1:06d}" if sequence > 1 else None,
        run_id=run_id,
        sequence=sequence,
        timestamp_unix_ms=int(record.get("timestamp_unix_ms") or now_ms + sequence),
        phase=_safe_phase(record.get("phase")),
        actor=str(record.get("actor") or "actor.host"),
        actor_class=_safe_actor_class(record.get("actor_class")),
        trust_domain=str(record.get("trust_domain") or "Trusted"),
        event_type=str(record.get("event_type") or "SandboxAction"),
        action_class=_safe_action_class(record.get("action_class")),
        resource=resource,
        resource_class=resource_class,
        decision="Allow" if record.get("status") == "ok" else "Deny",
        effect=str(record.get("status") or "recorded"),
        payload=payload,
        evidence_refs=["action_records"],
    )


def _normalize_resource_locator(value: str) -> str:
    """Lower-case, slash-normalized locator; URLs fold to matchable structure.

    A declared service route is a loopback URL prefix; the agent may dial it
    as any loopback spelling, over https, or with a query string. Userinfo,
    query and fragment are dropped, https folds to http, loopback hosts fold
    to ``localhost``.
    """

    normalized = value.replace("\\", "/").lower()
    if "://" in normalized:
        parts = urlsplit(normalized)
        scheme = "http" if parts.scheme == "https" else parts.scheme
        host = parts.hostname or ""
        if is_loopback_host(host):
            host = "localhost"
        try:
            port = parts.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        return f"{scheme}://{netloc}{parts.path}"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def _action_record_resource(
    record: dict[str, Any], contract: IntegrityContract
) -> str | None:
    resource = record.get("resource")
    resource_ids = {item.id for item in contract.resources}
    if isinstance(resource, str) and resource in resource_ids:
        return resource
    path_hint = str(record.get("target_path") or record.get("source_path") or "")
    if path_hint:
        record_cwd = record.get("cwd")
        is_url = "://" in path_hint
        if (
            not is_url
            and not path_hint.startswith("/")
            and isinstance(record_cwd, str)
            and record_cwd
        ):
            path_hint = posixpath.normpath(posixpath.join(record_cwd, path_hint))
        normalized = _normalize_resource_locator(path_hint)
        best: tuple[str, str, str] | None = None
        for item in contract.resources:
            resource_path = _normalize_resource_locator(item.path or "").rstrip("/")
            if not resource_path:
                continue
            # Anchored at a path root; the most specific declaration wins, and
            # at equal specificity a declared artifact wins over the workspace
            # it lives in.
            if normalized == resource_path or normalized.startswith(
                f"{resource_path}/"
            ):
                candidate = (item.id, resource_path, item.resource_class)
                if (
                    best is None
                    or len(resource_path) > len(best[1])
                    or (
                        len(resource_path) == len(best[1])
                        and item.resource_class == "DeclaredArtifact"
                        and best[2] != "DeclaredArtifact"
                    )
                ):
                    best = candidate
        if best is not None:
            return best[0]
    resource_class = record.get("resource_class")
    if isinstance(resource_class, str):
        for item in contract.resources:
            if item.resource_class == resource_class:
                return item.id
    return None


def _action_record_resource_class(
    record: dict[str, Any], contract: IntegrityContract, resource: str | None
) -> ResourceClass | None:
    # Contract first: a resource resolved against the contract's paths is
    # authoritative; the record's own class is only a fallback.
    if resource is not None:
        for item in contract.resources:
            if item.id == resource:
                return item.resource_class
    resource_class = record.get("resource_class")
    if resource_class in RESOURCE_CLASSES:
        return cast("ResourceClass", resource_class)
    return None


def _action_record_is_reward_relevant(
    record: dict[str, Any], contract: IntegrityContract
) -> bool:
    action_class = record.get("action_class")
    if action_class not in {"Upload", "Download", "Execute", "StartVerifier", "Write"}:
        return False
    path_hint = str(record.get("target_path") or record.get("source_path") or "")
    untrusted = bool(
        record.get("actor_class") == "Agent" or record.get("attribution_gap")
    )
    if record.get("phase") == "Verify" or record.get("actor_class") == "Verifier":
        return bool(path_hint) or untrusted
    if not path_hint:
        return False
    normalized = path_hint.replace("\\", "/").lower()
    for item in contract.resources:
        if item.resource_class not in _PROTECTED_OUTCOME_CLASSES:
            continue
        resource_path = (item.path or "").replace("\\", "/").lower().rstrip("/")
        if resource_path and (
            normalized == resource_path or normalized.startswith(resource_path + "/")
        ):
            return True
    return False


_INHERITED_KEYS = (
    "schema_version",
    "timestamp_unix_ms",
    "operation",
    "phase",
    "actor",
    "actor_class",
    "trust_domain",
    "attribution_source",
    "attribution_gap",
    "status",
    "service",
    "cwd",
    "tool_call_id",
)


def _derived_target_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    """A command's decomposed file targets as Read/Write sub-records.

    Sub-records inherit attribution and phase, not the parent's opacity
    markers: the fail-closed opacity check belongs to the parent exec alone.
    """

    derived = record.get("derived_targets")
    if not isinstance(derived, list):
        return []
    out: list[dict[str, Any]] = []
    for target in derived:
        if not isinstance(target, dict) or not target.get("path"):
            continue
        # The container root is a mount point, not an object (`find / ...`).
        if str(target["path"]).replace("\\", "/").rstrip("/") == "":
            continue
        write = target.get("mode") == "write"
        sub = {key: record[key] for key in _INHERITED_KEYS if key in record}
        sub.update(
            {
                "event_type": "ExecDerivedWrite" if write else "ExecDerivedRead",
                "action_class": "Write" if write else "Read",
                "target_path" if write else "source_path": target["path"],
                "mechanism": target.get("mechanism"),
                "derived_from_sequence": record.get("sequence"),
            }
        )
        if target.get("resource_class") in RESOURCE_CLASSES:
            sub["resource_class"] = target["resource_class"]
        out.append(sub)
    return out


def _derived_network_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    """A command's egress URLs as NetworkRequest sub-records."""

    derived = record.get("derived_network")
    if not isinstance(derived, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in derived:
        if not isinstance(entry, dict) or not entry.get("url"):
            continue
        sub = {key: record[key] for key in _INHERITED_KEYS if key in record}
        if "network_class" in record:
            sub["network_class"] = record["network_class"]
        sub.update(
            {
                "event_type": "ExecDerivedNetworkRequest",
                "action_class": "NetworkRequest",
                "resource": "network:web",
                "url": entry["url"],
                "mechanism": "exec_command_text",
                "derived_from_sequence": record.get("sequence"),
            }
        )
        out.append(sub)
    return out


def _derived_loopback_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Reads of the sandbox's own services, so a declared route can classify."""

    derived = record.get("derived_loopback")
    if not isinstance(derived, list) or not derived:
        return []
    out: list[dict[str, Any]] = []
    for target in derived:
        if not isinstance(target, dict) or not target.get("url"):
            continue
        sub = {key: record[key] for key in _INHERITED_KEYS if key in record}
        sub.update(
            {
                "event_type": "ExecDerivedRead",
                "action_class": "Read",
                "source_path": str(target["url"]),
                "mechanism": "loopback_http",
                "derived_from_sequence": record.get("sequence"),
            }
        )
        out.append(sub)
    return out


def _opaque_exec_protected_reference(
    record: dict[str, Any], contract: IntegrityContract
) -> str | None:
    """The first protected root an unparseable agent exec mentions.

    Abstains on commands recovered from a tool title (they may be prose) and
    on paths the decomposition already turned into classified targets.
    """

    if record.get("action_class") != "Execute" or not record.get("exec_opaque"):
        return None
    if record.get("actor_class") != "Agent" and not record.get("attribution_gap"):
        return None
    if record.get("command_from_title"):
        return None
    referenced = record.get("referenced_paths")
    if not isinstance(referenced, list):
        return None
    decomposed = {
        str(target.get("path")).replace("\\", "/").rstrip("/")
        for target in (record.get("derived_targets") or [])
        if isinstance(target, dict) and target.get("path")
    }
    resources = normalize_manifest_resources(contract.resource_pairs())
    for path in referenced:
        if not isinstance(path, str):
            continue
        if path.replace("\\", "/").rstrip("/") in decomposed:
            continue
        if classify_resource_class(path, resources) in _PROTECTED_OUTCOME_CLASSES:
            return path
    return None


def _action_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "schema_version",
        "sequence",
        "timestamp_unix_ms",
        "phase",
        "actor",
        "actor_class",
        "trust_domain",
        "event_type",
        "action_class",
        "resource",
        "resource_class",
    }
    return {
        key: value
        for key, value in record.items()
        if key not in skip and value is not None
    }


def _normalize_action_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        item = dict(record)
        item["sequence"] = index
        item.setdefault("schema_version", ACTION_RECORD_SCHEMA_VERSION)
        item.setdefault("operation", "sandbox.unknown")
        item.setdefault("event_type", "SandboxAction")
        item.setdefault("action_class", "Observe")
        item.setdefault("phase", "Agent")
        item.setdefault("actor", "actor.host")
        # A record without a valid actor is an attribution gap, never a
        # trusted host action: it fails closed when reward-relevant.
        if item.get("actor_class") not in ACTOR_CLASSES:
            item["actor_class"] = "TrustedHost"
            item["attribution_gap"] = True
        item.setdefault(
            "trust_domain", "Untrusted" if item["actor_class"] == "Agent" else "Trusted"
        )
        item.setdefault("status", "unknown")
        normalized.append(item)
    return normalized


def _safe_phase(value: object) -> Phase:
    return cast("Phase", value) if value in PHASES else "Agent"


def _safe_actor_class(value: object) -> ActorClass:
    return cast("ActorClass", value) if value in ACTOR_CLASSES else "TrustedHost"


def _safe_action_class(value: object) -> ActionClass:
    return cast("ActionClass", value) if value in ACTION_CLASSES else "Observe"


def _headline_reason(
    core_verdict: str,
    agent_evidence: list[str],
    events: list[BenchGuardEvent],
    gaps: dict[str, list[str]],
    vectors: list[str],
) -> str:
    """One sentence a person can act on, pointing at the evidence."""

    if core_verdict == "AgentViolation" and agent_evidence:
        by_id = {event.event_id: event for event in events}
        parts: list[str] = []
        for item in agent_evidence[:3]:
            _, event_id, detail = [*item.split(":", 2), "", ""][:3]
            event = by_id.get(event_id)
            where = ""
            if event is not None:
                path = (
                    event.payload.get("target_path")
                    or event.payload.get("source_path")
                    or event.payload.get("url")
                )
                command = event.payload.get("command_preview")
                where = f" {path}" if path else (f" `{command}`" if command else "")
            parts.append(f"{detail}{where} ({event_id})")
        more = f" and {len(agent_evidence) - 3} more" if len(agent_evidence) > 3 else ""
        return "agent " + "; ".join(parts) + more
    if core_verdict == "Rejected":
        found = [reason for values in gaps.values() for reason in values]
        return "no claim: " + (
            ", ".join(found) if found else "violation flags without agent attribution"
        )
    if core_verdict == "VectorExposed":
        return "no violation observed; exposed by " + ", ".join(vectors)
    return "no violation observed; the separate verifier received only the transferred outputs"


def _scalar_reward(rewards: dict[str, Any] | None) -> float | None:
    if not isinstance(rewards, dict):
        return None
    value = rewards.get("reward")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: Any) -> None:
    """Replace the document once it is complete (never a half-written claim)."""

    text = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    )


__all__ = [
    "CLAIM_VERDICT_FILE",
    "INTEGRITY_DIRNAME",
    "PROFILE_VECTORS",
    "SCRIPTED_AGENTS",
    "TrialEvidence",
    "emit_integrity",
    "emit_integrity_error",
    "profile_vectors",
    "verifier_status",
]

"""The integrity contract: which resources are whose, derived from the task.

A BenchGuard contract (arXiv 2609.11028;
``src/benchflow/benchguard/manifest.py``, ``task_toml.py``,
``task_binding.py``) classifies every reward-relevant resource into one of
nine classes and says what may cross into the verifier. Most of it follows
from facts BenchFlow already has: the workspace the rollout resolved, the
roots BenchFlow locks away from the agent, the reward folder, the declared
artifacts, the effective agent network mode. That is the derived contract.

Semantics BenchFlow cannot infer come from an optional ``benchguard.yaml`` in
the task folder, in BenchGuard's own binding format
(``benchguard.task_binding.v1``): extra resources (for example a trusted
dependency tree inside the workspace), handoffs, and task semantics such as
``measurement_mode: closed_book``, which makes any agent egress forbidden. A
binding can add or narrow; it cannot make a protected root agent-visible.

The written ``manifest.json`` keeps BenchGuard's field names.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from benchflow.integrity.constants import (
    MANIFEST_SCHEMA_VERSION,
    PROFILE_ID,
    ClaimLabel,
    NetworkClass,
    ResourceClass,
)

BINDING_FILENAMES = ("benchguard.yaml", "benchguard.yml")
BINDING_SCHEMA_VERSION = "benchguard.task_binding.v1"

MeasurementMode = Literal[
    "closed_book", "open_book", "tool_use", "interactive", "custom"
]
AnswerBearingDisposition = Literal[
    "hard_fail_on_use", "outside_claim", "allowed_by_task"
]
PublicAnswerRisk = Literal["none", "low", "medium", "high", "unknown"]
AssumptionStatus = Literal["None", "Assumed", "Reviewed", "Proved"]
ArtifactContentKind = Literal["data", "code"]
ResourceKind = Literal["path", "env", "service", "network", "log", "other"]

# Roots a binding may never reclassify as something the agent may see or write.
_FIXED_PROTECTED_ROOTS = (
    "/tests",
    "/verifier",
    "/testbed_verify",
    "/solution",
    "/oracle",
    "/logs/verifier",
)
_AGENT_VISIBLE_CLASSES = frozenset(
    {"Public", "AgentWritable", "DeclaredArtifact", "UndeclaredArtifact", "Trusted"}
)
_EXECUTABLE_ARTIFACT_SUFFIXES = frozenset(
    {".joblib", ".pickle", ".pkl", ".pt", ".pth", ".py", ".sh"}
)

_MODEL = ConfigDict(extra="forbid", populate_by_name=True)


# --- the task-supplied binding (BenchGuard's format) --------------------------


class BindingTaskSemantics(BaseModel):
    model_config = _MODEL

    intended_property: str | None = None
    intended_skill: str | None = None
    measurement_mode: MeasurementMode | None = None
    answer_bearing_channel_disposition: AnswerBearingDisposition = "hard_fail_on_use"
    public_answer_risk: PublicAnswerRisk = "unknown"


class BindingResource(BaseModel):
    model_config = _MODEL

    id: str
    selector: str
    class_: ResourceClass = Field(alias="class")
    task_use: Literal["allowed", "forbidden", "review"] = "review"
    reason: str


class BindingHandoff(BaseModel):
    model_config = _MODEL

    id: str
    path: str
    content_kind: ArtifactContentKind


class BindingEndpoint(BaseModel):
    model_config = _MODEL

    id: str
    target: str
    task_required: bool
    answer_bearing: bool | None = None
    reason: str


class BindingSemanticObligation(BaseModel):
    model_config = _MODEL

    id: str
    subject: str
    question: str


class TaskBinding(BaseModel):
    """The small author- or auditor-supplied part of a contract."""

    model_config = _MODEL

    schema_version: Literal["benchguard.task_binding.v1"] = BINDING_SCHEMA_VERSION
    task: BindingTaskSemantics = Field(default_factory=BindingTaskSemantics)
    resources: list[BindingResource] = Field(default_factory=list)
    handoffs: list[BindingHandoff] = Field(default_factory=list)
    external_endpoints: list[BindingEndpoint] = Field(default_factory=list)
    semantic_obligations: list[BindingSemanticObligation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> TaskBinding:
        for name in (
            "resources",
            "handoffs",
            "external_endpoints",
            "semantic_obligations",
        ):
            ids = [item.id for item in getattr(self, name)]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} ids must be unique")
        return self


@dataclass(frozen=True)
class LoadedBinding:
    path: Path
    sha256: str
    binding: TaskBinding | None
    error: str | None = None


def find_binding(task_dir: Path | None) -> Path | None:
    if task_dir is None:
        return None
    for name in BINDING_FILENAMES:
        candidate = Path(task_dir) / name
        if candidate.is_file():
            return candidate
    return None


def load_binding(path: Path) -> LoadedBinding:
    """Read a binding; a malformed one is kept as an error, never ignored."""

    raw = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        payload = yaml.safe_load(raw.decode()) or {}
        if not isinstance(payload, dict):
            raise ValueError("a task binding must be a YAML mapping")
        return LoadedBinding(path, digest, TaskBinding.model_validate(payload))
    except (UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        return LoadedBinding(path, digest, None, f"{type(exc).__name__}: {exc}")


# --- the contract written to manifest.json -----------------------------------


class ContractTask(BaseModel):
    model_config = _MODEL

    task_id: str = ""
    intended_property: str = ""
    verifier_soundness_status: AssumptionStatus = "Assumed"


class ContractTaskValidity(BaseModel):
    model_config = _MODEL

    intended_skill: str = "unspecified"
    measurement_mode: MeasurementMode = "custom"
    allowed_resources: list[str] = Field(default_factory=list)
    forbidden_resources: list[str] = Field(default_factory=list)
    answer_bearing_channel_disposition: AnswerBearingDisposition = "hard_fail_on_use"
    public_answer_risk: PublicAnswerRisk = "unknown"


class ContractResource(BaseModel):
    model_config = _MODEL

    id: str
    class_: ResourceClass = Field(alias="class")
    kind: ResourceKind = "path"
    path: str | None = None
    source: Literal["derived", "binding"] = "derived"

    @property
    def resource_class(self) -> ResourceClass:
        return self.class_


class ContractArtifact(BaseModel):
    model_config = _MODEL

    id: str
    resource: str
    declared_path: str
    verifier_input: bool = True
    content_kind: ArtifactContentKind = "data"


class ContractNetwork(BaseModel):
    model_config = _MODEL

    task_egress: NetworkClass = "Full"
    provider_egress: NetworkClass = "ProviderEgress"
    allowlist: list[str] = Field(default_factory=list)
    agent_network_mode: str = "public"


class ContractReward(BaseModel):
    model_config = _MODEL

    source: Literal["TrustedVerifier"] = "TrustedVerifier"
    output_class: Literal["RewardOutput"] = "RewardOutput"
    score_range: list[float] = Field(default_factory=lambda: [0.0, 1.0])


class IntegrityContract(BaseModel):
    """What the claim is checked against (``integrity/manifest.json``)."""

    model_config = _MODEL

    schema_version: Literal["benchguard.manifest.v0"] = MANIFEST_SCHEMA_VERSION
    profile: str = PROFILE_ID
    claim_mode: Literal["enforce", "audit_only", "outside_claim"] = "enforce"
    task: ContractTask = Field(default_factory=ContractTask)
    task_validity: ContractTaskValidity = Field(default_factory=ContractTaskValidity)
    workspace: str = "/app"
    resources: list[ContractResource] = Field(default_factory=list)
    artifacts: list[ContractArtifact] = Field(default_factory=list)
    network: ContractNetwork = Field(default_factory=ContractNetwork)
    reward: ContractReward = Field(default_factory=ContractReward)
    binding: dict[str, Any] | None = None
    violations: list[str] = Field(default_factory=list)

    def resource_pairs(self) -> list[tuple[str, str]]:
        """``(path, class)`` for every filesystem resource, for the classifier.

        Service routes (URLs) are matched separately, by URL structure.
        """
        return [
            (item.path, item.resource_class)
            for item in self.resources
            if item.path and item.kind == "path"
        ]


@dataclass
class ContractFacts:
    """What BenchFlow knows about a trial that the contract is built from."""

    task_id: str
    workspace: str
    agent_network_mode: str = "public"
    allowed_hosts: list[str] = field(default_factory=list)
    skills_dir: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    state_paths: list[str] = field(default_factory=list)
    reward_range: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def derive_contract(
    facts: ContractFacts, *, binding: LoadedBinding | None = None
) -> IntegrityContract:
    """Build the contract for one trial: derived facts, then the binding."""

    semantic = binding.binding if binding is not None else None
    violations: list[str] = []
    if binding is not None and binding.error is not None:
        violations.append("binding_invalid")

    workspace = _normalize_root(facts.workspace) or "/app"
    resources: list[ContractResource] = [
        _resource("resource.workspace", "AgentWritable", workspace),
        _resource("resource.tests", "VerifierOnly", "/tests"),
        _resource("resource.verifier_code", "VerifierOnly", "/verifier"),
        _resource("resource.testbed_verify", "VerifierOnly", "/testbed_verify"),
        _resource("resource.solution", "Hidden", "/solution"),
        _resource("resource.oracle", "Hidden", "/oracle"),
        _resource("resource.verifier_logs", "RewardOutput", "/logs/verifier"),
        _resource("resource.agent_logs", "Trusted", "/logs/agent"),
        # BenchFlow collects /logs/artifacts for the verifier: an agent-written
        # channel into outcome computation by design, i.e. a declared handoff.
        _resource("resource.logs_artifacts", "DeclaredArtifact", "/logs/artifacts"),
    ]
    if facts.skills_dir:
        # Skills are mounted for the agent to read: Public (a trusted input the
        # agent may observe but not change).
        resources.append(_resource("resource.skills", "Public", facts.skills_dir))
    for index, path in enumerate(sorted(set(facts.state_paths)), start=1):
        # A service's backing store: state the agent must reach through the
        # service's interface, not directly (BenchGuard's ClawsBench finding).
        resources.append(
            _resource(f"resource.environment_state_{index}", "Hidden", path)
        )

    artifacts: list[ContractArtifact] = [
        ContractArtifact(
            id="artifact.logs_artifacts",
            resource="resource.logs_artifacts",
            declared_path="/logs/artifacts",
        )
    ]
    handoff_paths = (
        {h.path for h in semantic.handoffs} if semantic is not None else set()
    )
    for index, path in enumerate(facts.artifact_paths, start=1):
        if path in handoff_paths or not path.startswith("/"):
            continue
        resource_id = f"resource.config_artifact_{index}"
        resources.append(_resource(resource_id, "DeclaredArtifact", path))
        artifacts.append(
            ContractArtifact(
                id=f"artifact.config_{index}",
                resource=resource_id,
                declared_path=path,
                content_kind="code"
                if Path(path).suffix.lower() in _EXECUTABLE_ARTIFACT_SUFFIXES
                else "data",
            )
        )

    if semantic is not None:
        for resource in semantic.resources:
            if _reclassifies_protected_root(resource.selector, resource.class_):
                violations.append(f"binding_protected_root:{resource.id}")
                continue
            resources.append(
                ContractResource.model_validate(
                    {
                        "id": resource.id,
                        "class": resource.class_,
                        "path": _normalize_root(resource.selector)
                        if not _is_url(resource.selector)
                        else resource.selector,
                        "kind": "service" if _is_url(resource.selector) else "path",
                        "source": "binding",
                    }
                )
            )
        for handoff in semantic.handoffs:
            resource_id = f"resource.{handoff.id}"
            path = _absolute(handoff.path, workspace)
            resources.append(
                ContractResource.model_validate(
                    {
                        "id": resource_id,
                        "class": "DeclaredArtifact",
                        "path": path,
                        "source": "binding",
                    }
                )
            )
            artifacts.append(
                ContractArtifact(
                    id=f"artifact.{handoff.id}",
                    resource=resource_id,
                    declared_path=path,
                    content_kind=handoff.content_kind,
                )
            )
    # With no declared artifact, the verifier reads the final workspace: the
    # workspace root is the one declared crossing (BenchGuard's final_state).
    if len(artifacts) == 1:
        artifacts.append(
            ContractArtifact(
                id="artifact.final_state",
                resource="resource.workspace",
                declared_path=workspace,
            )
        )

    metadata = facts.metadata or {}
    task_semantics = semantic.task if semantic is not None else None
    measurement_mode: MeasurementMode = (
        task_semantics.measurement_mode
        if task_semantics is not None and task_semantics.measurement_mode
        else _metadata_choice(
            metadata,
            "measurement_mode",
            {"closed_book", "open_book", "tool_use", "interactive", "custom"},
            "custom",
        )
    )
    contract = IntegrityContract(
        task=ContractTask(
            task_id=facts.task_id,
            intended_property=(
                task_semantics.intended_property
                if task_semantics is not None and task_semantics.intended_property
                else str(metadata.get("intended_property") or "")
            ),
            verifier_soundness_status=_soundness(metadata),
        ),
        task_validity=ContractTaskValidity(
            intended_skill=(
                task_semantics.intended_skill
                if task_semantics is not None and task_semantics.intended_skill
                else str(metadata.get("intended_skill") or "unspecified")
            ),
            measurement_mode=measurement_mode,
            allowed_resources=["resource.workspace"],
            forbidden_resources=[
                "resource.tests",
                "resource.solution",
                "resource.oracle",
                "resource.verifier_logs",
            ],
            answer_bearing_channel_disposition=(
                task_semantics.answer_bearing_channel_disposition
                if task_semantics is not None
                else "hard_fail_on_use"
            ),
            public_answer_risk=(
                task_semantics.public_answer_risk
                if task_semantics is not None
                else _metadata_choice(
                    metadata,
                    "public_answer_risk",
                    {"none", "low", "medium", "high", "unknown"},
                    "unknown",
                )
            ),
        ),
        workspace=workspace,
        resources=_dedupe_resources(resources),
        artifacts=artifacts,
        network=ContractNetwork(
            task_egress=authorized_task_egress(
                facts.agent_network_mode, facts.allowed_hosts, semantic
            ),
            allowlist=list(facts.allowed_hosts),
            agent_network_mode=facts.agent_network_mode,
        ),
        reward=ContractReward(
            score_range=[float(v) for v in (facts.reward_range or (0.0, 1.0))]
        ),
        binding=(
            {
                "path": binding.path.name,
                "sha256": binding.sha256,
                "error": binding.error,
            }
            if binding is not None
            else None
        ),
        violations=violations,
    )
    return contract


def authorized_task_egress(
    agent_network_mode: str,
    allowed_hosts: Iterable[str],
    binding: TaskBinding | None,
) -> NetworkClass:
    """Task-authorized egress, apart from what the network happens to allow.

    Ported from BenchGuard's ``_authorized_task_egress``: without a binding
    the class follows the enforced mode; a binding's measurement mode can
    narrow it (closed-book tasks forbid all agent egress, even when the
    sandbox is public).

    Deviation from BenchGuard: a binding that leaves ``measurement_mode``
    unset keeps the enforced class. BenchGuard treated an unset mode like
    ``custom`` (no egress), so a ``benchguard.yaml`` written only to mark a
    trusted tree turned every download on a public task into a violation.
    An explicit ``custom`` still means no egress, as in BenchGuard.
    """

    mode = agent_network_mode or "public"
    if mode == "no-network":
        return "NoEgress"
    if binding is None or binding.task.measurement_mode is None:
        return "Allowlist" if mode == "allowlist" else "Full"
    measurement_mode = binding.task.measurement_mode
    if measurement_mode == "closed_book":
        return "NoEgress"
    if mode == "allowlist":
        endpoints = {
            endpoint.target: endpoint for endpoint in binding.external_endpoints
        }
        if all(
            host in endpoints
            and endpoints[host].task_required
            and endpoints[host].answer_bearing is False
            for host in allowed_hosts
        ):
            return "Allowlist"
        return "NoEgress"
    if measurement_mode in {"open_book", "tool_use", "interactive"}:
        return "Full"
    return "NoEgress"


def contract_label(contract: IntegrityContract) -> tuple[ClaimLabel, list[str]]:
    """The claim label the contract alone allows, and its violation codes."""

    violations = list(contract.violations)
    if contract.claim_mode == "audit_only":
        return "AuditOnly", violations
    if contract.claim_mode == "outside_claim":
        return "OutsideClaim", violations
    if contract.task_validity.answer_bearing_channel_disposition == "outside_claim":
        return "OutsideClaim", violations
    if violations:
        return "Rejected", violations
    if contract.reward.score_range[0] < 0.0 or contract.reward.score_range[1] > 1.0:
        # Penalty rewards are a limit of the v0 reward model: flagged for
        # review, not expelled (BenchGuard's reward_score_range).
        return "Conditional", ["reward_score_range"]
    if contract.task.verifier_soundness_status in {"None", "Assumed"}:
        return "Conditional", violations
    if contract.task_validity.public_answer_risk in {"medium", "high", "unknown"}:
        return "Conditional", violations
    return "Checked", violations


def i7_obligations(contract: IntegrityContract) -> list[str]:
    """Semantic-adequacy review points the environment cannot enforce (I7)."""

    obligations: list[str] = []
    status = contract.task.verifier_soundness_status
    if status in {"None", "Assumed"}:
        obligations.append(f"verifier_soundness:{status.lower()}")
    risk = contract.task_validity.public_answer_risk
    if risk in {"medium", "high", "unknown"}:
        obligations.append(f"public_answer_risk:{risk}")
    if contract.task_validity.answer_bearing_channel_disposition == "outside_claim":
        obligations.append("answer_bearing_channel:outside_claim")
    if not contract.task.intended_property.strip():
        obligations.append("intended_property:missing")
    return obligations


_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def dockerfile_workdir(task_dir: Path | None) -> str | None:
    """The last ``WORKDIR`` of the task's ``environment/Dockerfile``, if any.

    Used only when re-verdicting a stored trial that did not record the
    workspace the rollout resolved.
    """

    if task_dir is None:
        return None
    dockerfile = Path(task_dir) / "environment" / "Dockerfile"
    try:
        text = dockerfile.read_text(errors="replace")
    except OSError:
        return None
    matches = _WORKDIR_RE.findall(text)
    if not matches:
        return None
    value = matches[-1].strip().strip("\"'")
    return value if value.startswith("/") and "$" not in value else None


def _resource(
    resource_id: str, resource_class: ResourceClass, path: str
) -> ContractResource:
    return ContractResource.model_validate(
        {"id": resource_id, "class": resource_class, "path": path}
    )


def _dedupe_resources(resources: list[ContractResource]) -> list[ContractResource]:
    seen: set[str] = set()
    out: list[ContractResource] = []
    for resource in resources:
        if resource.id in seen:
            continue
        seen.add(resource.id)
        out.append(resource)
    return out


def _normalize_root(path: str | None) -> str:
    if not path:
        return ""
    normalized = str(path).replace("\\", "/").rstrip("/")
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized or "/"


def _absolute(path: str, workspace: str) -> str:
    if path.startswith("/"):
        return _normalize_root(path)
    return _normalize_root(f"{workspace.rstrip('/')}/{path}")


def _is_url(value: str) -> bool:
    return "://" in value


def _reclassifies_protected_root(selector: str, resource_class: str) -> bool:
    if resource_class not in _AGENT_VISIBLE_CLASSES or _is_url(selector):
        return False
    normalized = _normalize_root(selector)
    return any(
        normalized == root or normalized.startswith(f"{root}/")
        for root in _FIXED_PROTECTED_ROOTS
    )


def _soundness(metadata: dict[str, Any]) -> AssumptionStatus:
    raw = str(
        metadata.get("verifier_soundness_status")
        or metadata.get("benchguard_verifier_soundness_status")
        or "assumed"
    ).strip()
    table: dict[str, AssumptionStatus] = {
        "none": "None",
        "assumed": "Assumed",
        "reviewed": "Reviewed",
        "proved": "Proved",
    }
    return table.get(raw.lower(), "Assumed")


def _metadata_choice(
    metadata: dict[str, Any], key: str, allowed: set[str], default: str
) -> Any:
    value = metadata.get(key) or metadata.get(f"benchguard_{key}")
    return value if isinstance(value, str) and value in allowed else default


__all__ = [
    "BINDING_FILENAMES",
    "ContractFacts",
    "IntegrityContract",
    "LoadedBinding",
    "TaskBinding",
    "authorized_task_egress",
    "contract_label",
    "derive_contract",
    "dockerfile_workdir",
    "find_binding",
    "i7_obligations",
    "load_binding",
]

"""Typed view models for the Python-to-JavaScript viewer boundary.

Raw rollout artifacts are normalized before they enter these models.  Each
``to_payload`` method is therefore a small, explicit declaration of the wire
shape instead of a second validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]

ToolHue = Literal[
    "read", "edit", "execute", "fetch", "search", "think", "skill", "other"
]
ToolStatus = Literal[
    "pending", "in_progress", "completed", "failed", "cancelled", "unknown"
]
VerifierStatus = Literal["passed", "failed", "skipped", "pending", "unknown"]

_KNOWN_HUES: frozenset[str] = frozenset(
    {"read", "edit", "execute", "fetch", "search", "think", "skill"}
)
# Viewer-assigned kinds that are not ACP kinds, with their fixed hue.
_KIND_HUES: dict[str, ToolHue] = {"agent": "skill"}
# Harnesses report the tool that spawns a subagent with ACP kind "think":
# claude-agent-acp's Task/Agent (title "Task" before the input arrives, the
# description after) and opencode's task. Its first title word, or a
# Task-shaped input, identifies it.
_SPAWN_TOOL_NAMES: frozenset[str] = frozenset({"task", "agent"})
_HUE_INFER: tuple[tuple[ToolHue, tuple[str, ...]], ...] = (
    ("search", ("web", "search", "fetch", "grep", "glob", "browser")),
    ("execute", ("bash", "shell", "exec", "terminal", "command")),
    ("edit", ("write", "edit", "patch", "delete", "move", "notebook")),
    ("read", ("read", "cat", "view", "ls", "list")),
    ("skill", ("agent", "task", "skill", "oracle")),
)
_TOOL_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed", "failed", "cancelled"}
)
_VERIFIER_STATUSES: frozenset[str] = frozenset(
    {"passed", "failed", "skipped", "pending"}
)


def tool_hue(kind: str, title: str = "") -> ToolHue:
    """Classify a tool once for both modern and legacy renderers."""
    normalized_kind = kind.strip().lower()
    if normalized_kind in _KNOWN_HUES:
        return cast(ToolHue, normalized_kind)
    if normalized_kind in _KIND_HUES:
        return _KIND_HUES[normalized_kind]
    haystack = f"{kind} {title}".lower()
    for hue, needles in _HUE_INFER:
        if any(needle in haystack for needle in needles):
            return hue
    return "other"


def tool_kind(kind: str, title: str = "", raw_input: object = None) -> str:
    """The kind the badge shows: ``agent`` for a subagent spawn, else ``kind``.

    The ACP kind of a spawning call is ``think`` (internal reasoning), which
    names the wrong thing; every other kind passes through unchanged.
    """
    if kind.strip().lower() != "think":
        return kind
    words = title.split(None, 1)
    named_spawn = bool(words) and words[0].lower() in _SPAWN_TOOL_NAMES
    task_input = (
        isinstance(raw_input, dict)
        and "prompt" in raw_input
        and ("description" in raw_input or "subagent_type" in raw_input)
    )
    return "agent" if named_spawn or task_input else kind


def normalize_tool_status(value: object) -> ToolStatus:
    """Return a CSS-safe ACP tool status, never untrusted source text."""
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _TOOL_STATUSES:
            return cast(ToolStatus, normalized)
    return "unknown"


def normalize_verifier_status(value: object) -> VerifierStatus:
    """Return the small verifier status vocabulary understood by the UI."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VERIFIER_STATUSES:
            return cast(VerifierStatus, normalized)
    return "unknown"


@dataclass(frozen=True)
class ToolCall:
    id: str
    kind: str
    title: str
    status: ToolStatus
    content: list[str]
    hue: ToolHue

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "content": self.content,
            "hue": self.hue,
        }


@dataclass(frozen=True)
class TimeoutInfo:
    reason: str
    timeout_sec: float | None
    pending: list[str]
    complete: bool | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "timeout_sec": self.timeout_sec,
            "pending": self.pending,
            "complete": self.complete,
        }


def _step_payload(i: int, kind: str, *, t: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"i": i, "kind": kind}
    if t is not None:
        payload["t"] = t
    return payload


@dataclass(frozen=True)
class PromptStep:
    i: int
    label: str
    text: str
    t: float | None = None
    kind: Literal["prompt"] = field(default="prompt", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload.update(label=self.label, text=self.text)
        return payload


@dataclass(frozen=True)
class MessageStep:
    i: int
    text: str
    t: float | None = None
    kind: Literal["message"] = field(default="message", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload["text"] = self.text
        return payload


@dataclass(frozen=True)
class ThoughtStep:
    i: int
    text: str
    t: float | None = None
    kind: Literal["thought"] = field(default="thought", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload["text"] = self.text
        return payload


@dataclass(frozen=True)
class SubagentTrace:
    """Events a subagent emitted, attributed through ``parent_tool_call_id``.

    ``depth`` counts levels below the main agent (1 = spawned by the main
    agent), capped at the ATIF export's ``MAX_SUBAGENT_DEPTH``.
    """

    parent_tool_call_id: str
    depth: int
    steps: list[Step]
    subagent_type: str | None = None
    description: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_tool_call_id": self.parent_tool_call_id,
            "depth": self.depth,
            "subagent_type": self.subagent_type,
            "description": self.description,
            "steps": [step.to_payload() for step in self.steps],
        }


@dataclass(frozen=True)
class ToolStep:
    i: int
    tool: ToolCall
    t: float | None = None
    dur: float | None = None
    subagent: SubagentTrace | None = None
    kind: Literal["tool"] = field(default="tool", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload["tool"] = self.tool.to_payload()
        if self.dur is not None:
            payload["dur"] = self.dur
        if self.subagent is not None:
            payload["subagent"] = self.subagent.to_payload()
        return payload


@dataclass(frozen=True)
class TimeoutStep:
    i: int
    timeout: TimeoutInfo
    t: float | None = None
    kind: Literal["timeout"] = field(default="timeout", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload["timeout"] = self.timeout.to_payload()
        return payload


@dataclass(frozen=True)
class UnknownStep:
    i: int
    type: str
    text: str
    t: float | None = None
    kind: Literal["unknown"] = field(default="unknown", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload.update(type=self.type, text=self.text)
        return payload


SubagentGroupReason = Literal["parent_not_captured", "too_deep", "cyclic"]


@dataclass(frozen=True)
class SubagentGroupStep:
    """A labelled group for subagent events with no spawning tool card.

    ``parent_not_captured``: the attributed tool call never appears in the
    capture (the ATIF export embeds these at the root without a reference).
    ``too_deep`` / ``cyclic``: the spawning call exists but is not reachable
    from the main agent within ``MAX_SUBAGENT_DEPTH`` levels; the ATIF export
    leaves these out, the viewer still shows every recorded event.
    """

    gid: str
    reason: SubagentGroupReason
    subagent: SubagentTrace
    kind: Literal["subagent"] = field(default="subagent", init=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "gid": self.gid,
            "reason": self.reason,
            "subagent": self.subagent.to_payload(),
        }


type Step = (
    PromptStep
    | MessageStep
    | ThoughtStep
    | ToolStep
    | TimeoutStep
    | UnknownStep
    | SubagentGroupStep
)


def iter_event_steps(steps: list[Step]) -> list[Step]:
    """Every event step, nested subagent steps included, in capture order.

    Group containers are not events and are skipped; ``i`` is the capture
    order, so sorting by it restores the flat sequence.
    """
    found: list[Step] = []
    pending: list[list[Step]] = [steps]
    while pending:
        for step in pending.pop():
            if isinstance(step, SubagentGroupStep):
                pending.append(step.subagent.steps)
                continue
            found.append(step)
            if isinstance(step, ToolStep) and step.subagent is not None:
                pending.append(step.subagent.steps)
    return sorted(found, key=lambda step: getattr(step, "i", 0))


@dataclass(frozen=True)
class ErrorBanner:
    label: str
    text: str
    level: Literal["error", "info"] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label, "text": self.text}
        if self.level is not None:
            payload["level"] = self.level
        return payload


@dataclass(frozen=True)
class StepCounts:
    prompts: int = 0
    messages: int = 0
    thoughts: int = 0
    tools: int = 0
    # Subagent traces and the events inside them (already included in the
    # per-kind counts above); omitted from the wire when there are none.
    subagents: int = 0
    subagent_events: int = 0

    def to_payload(self) -> dict[str, int]:
        payload = {
            "prompts": self.prompts,
            "messages": self.messages,
            "thoughts": self.thoughts,
            "tools": self.tools,
        }
        if self.subagents:
            payload["subagents"] = self.subagents
            payload["subagent_events"] = self.subagent_events
        return payload


@dataclass(frozen=True)
class Usage:
    """Normalized ``agent_result`` plus typed fields used by the catalog."""

    values: JsonObject = field(default_factory=dict)
    n_tool_calls: int | None = None
    n_skill_invocations: int | None = None
    n_prompts: int | None = None
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    n_cache_read_tokens: int | None = None
    n_cache_creation_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    usage_source: str | None = None
    price_source: str | None = None

    def to_payload(self) -> JsonObject:
        return dict(self.values)


@dataclass(frozen=True)
class Timing:
    """Finite phase timings loaded from the preferred timing artifact."""

    values: dict[str, float | None]
    total: float | None = None

    def to_payload(self) -> dict[str, float | None]:
        return dict(self.values)


ExecutionStatus = Literal["completed", "errored", "timed_out"]
AssessmentStatus = Literal["scored", "unscored"]


@dataclass(frozen=True)
class RunStatus:
    """Execution and assessment, reported separately.

    Execution answers "did the solver run finish?" (the agent ``error``);
    assessment answers "was a verdict recorded?" (reward, integrated
    ``scoring`` block, ``verifier_error``). A verifier failure after a clean
    run is therefore "completed but unscored", never an agent failure.
    """

    execution: ExecutionStatus
    execution_detail: str | None
    assessment: AssessmentStatus
    assessment_detail: str | None
    summary: str
    scoring: JsonObject | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "execution": self.execution,
            "execution_detail": self.execution_detail,
            "assessment": self.assessment,
            "assessment_detail": self.assessment_detail,
            "summary": self.summary,
            "scoring": dict(self.scoring) if self.scoring is not None else None,
        }


@dataclass(frozen=True)
class BranchOrigin:
    """Where a branch-child trajectory sits in its parent's ``tree.json``."""

    parent_rollout: str
    fork_id: str
    node_id: str
    index: int | None
    parent_node: str | None
    status: str | None
    intervention: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_rollout": self.parent_rollout,
            "fork_id": self.fork_id,
            "node_id": self.node_id,
            "index": self.index,
            "parent_node": self.parent_node,
            "status": self.status,
            "intervention": self.intervention,
        }


@dataclass(frozen=True)
class RolloutMetadata:
    """Canonical normalized projection of result.json and timing.json."""

    task_name: str | None = None
    agent_name: str | None = None
    model: str | None = None
    skill_mode: str | None = None
    reward: float | None = None
    usage: Usage = field(default_factory=Usage)
    timing: Timing | None = None
    n_tool_calls: int | None = None
    errors: tuple[ErrorBanner, ...] = ()
    has_error: bool = False
    trajectory_source: str | None = None
    partial_trajectory: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    status: RunStatus | None = None


@dataclass(frozen=True)
class Meta:
    task_name: str | None = None
    agent_name: str | None = None
    model: str | None = None
    skill_mode: str | None = None
    reward: float | None = None
    usage: Usage = field(default_factory=Usage)
    counts: StepCounts = field(default_factory=StepCounts)
    timing: Timing | None = None
    duration_sec: float | None = None
    errors: tuple[ErrorBanner, ...] = ()
    trajectory_source: str | None = None
    partial_trajectory: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    status: RunStatus | None = None
    branch: BranchOrigin | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "agent_name": self.agent_name,
            "model": self.model,
            "skill_mode": self.skill_mode,
            "reward": self.reward,
            "usage": self.usage.to_payload(),
            "counts": self.counts.to_payload(),
            "timing": self.timing.to_payload() if self.timing is not None else None,
            "duration_sec": self.duration_sec,
            "errors": [error.to_payload() for error in self.errors],
            "trajectory_source": self.trajectory_source,
            "partial_trajectory": self.partial_trajectory,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.to_payload() if self.status is not None else None,
            "branch": self.branch.to_payload() if self.branch is not None else None,
        }


@dataclass(frozen=True)
class VerifierTest:
    name: str
    status: VerifierStatus
    duration: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration": self.duration,
        }


@dataclass(frozen=True)
class RecoveryAttempt:
    """One ``verifier-recovery/<id>/recovery.json`` receipt."""

    attempt: str
    admitted: bool
    status: str | None = None
    original_error: str | None = None
    error: str | None = None
    evidence: str | None = None
    solver_replayed: bool | None = None
    publication_error: str | None = None
    cleanup_error: str | None = None
    admission: str | None = None
    reward: float | None = None
    verifier_sec: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "admitted": self.admitted,
            "status": self.status,
            "original_error": self.original_error,
            "error": self.error,
            "evidence": self.evidence,
            "solver_replayed": self.solver_replayed,
            "publication_error": self.publication_error,
            "cleanup_error": self.cleanup_error,
            "admission": self.admission,
            "reward": self.reward,
            "verifier_sec": self.verifier_sec,
        }


@dataclass(frozen=True)
class VerifierRecovery:
    """Verifier-only recovery evidence (docs/verifier-recovery.md).

    ``pointer`` is ``verification.json``'s admitted attempt; ``attempts``
    lists the admitted receipt first, then any other receipts.
    """

    pointer: str | None
    pointer_error: str | None
    attempts: list[RecoveryAttempt]

    def to_payload(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "pointer_error": self.pointer_error,
            "attempts": [attempt.to_payload() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class VerifierArtifacts:
    reward: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    ctrf: list[VerifierTest] | None = None
    recovery: VerifierRecovery | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ctrf": [test.to_payload() for test in self.ctrf]
            if self.ctrf is not None
            else None,
            "recovery": self.recovery.to_payload()
            if self.recovery is not None
            else None,
        }


@dataclass(frozen=True)
class RubricCriterion:
    """One rubric criterion as the reviewer judged it."""

    name: str
    # True for a v0.2 blocker, False for a v0.2 scored criterion, None for a
    # legacy v0.1 criterion, which only carries an outcome.
    blocker: bool | None
    weight: int | None
    outcome: str | None
    score: int | None
    explanation: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocker": self.blocker,
            "weight": self.weight,
            "outcome": self.outcome,
            "score": self.score,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class RubricReview:
    """The ``bench review`` verdict for one rollout, from ``review_report.json``."""

    reviewer_model: str | None
    review_valid: bool
    scoring: JsonObject
    summary: str
    criteria: list[RubricCriterion]
    source: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "reviewer_model": self.reviewer_model,
            "review_valid": self.review_valid,
            "scoring": dict(self.scoring),
            "summary": self.summary,
            "criteria": [criterion.to_payload() for criterion in self.criteria],
            "source": self.source,
        }


@dataclass(frozen=True)
class BranchChild:
    """One child row of a fork in ``tree.json`` (branch_lineage.ForkRecord).

    ``ref`` (``<fork id>/<node id>``) is set only when the child's evidence
    was published (``artifacts.status == "available"``) and its
    ``observation.json`` is present, so the viewer can open its trajectory.
    """

    index: int | None
    node_id: str | None
    status: str | None
    reward: float | None
    reward_source: str | None
    error: str | None
    cleanup_error: str | None
    intervention: dict[str, str | None]
    artifacts_status: str | None
    artifacts_path: str | None
    ref: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node_id": self.node_id,
            "status": self.status,
            "reward": self.reward,
            "reward_source": self.reward_source,
            "error": self.error,
            "cleanup_error": self.cleanup_error,
            "intervention": dict(self.intervention),
            "artifacts_status": self.artifacts_status,
            "artifacts_path": self.artifacts_path,
            "ref": self.ref,
        }


@dataclass(frozen=True)
class BranchFork:
    id: str
    parent_node: str | None
    status: str | None
    value: float | None
    requested_children: int | None
    requested_layers: list[str]
    captured_layers: list[str]
    parent_restore: str | None
    error: str | None
    parent_restore_error: str | None
    artifact_error: str | None
    children: list[BranchChild]

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_node": self.parent_node,
            "status": self.status,
            "value": self.value,
            "requested_children": self.requested_children,
            "requested_layers": list(self.requested_layers),
            "captured_layers": list(self.captured_layers),
            "parent_restore": self.parent_restore,
            "error": self.error,
            "parent_restore_error": self.parent_restore_error,
            "artifact_error": self.artifact_error,
            "children": [child.to_payload() for child in self.children],
        }


@dataclass(frozen=True)
class Lineage:
    """Rollout-branch lineage read from the run's ``tree.json``."""

    nodes: int
    forks: list[BranchFork]
    error: str | None = None
    source: str = "tree.json"

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "nodes": self.nodes,
            "forks": [fork.to_payload() for fork in self.forks],
            "error": self.error,
        }


@dataclass(frozen=True)
class ViewerPayload:
    rollout_name: str
    meta: Meta
    steps: list[Step]
    verifier: VerifierArtifacts
    rubric: RubricReview | None = None
    lineage: Lineage | None = None
    schema_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_name": self.rollout_name,
            "meta": self.meta.to_payload(),
            "steps": [step.to_payload() for step in self.steps],
            "verifier": self.verifier.to_payload(),
            "rubric": self.rubric.to_payload() if self.rubric is not None else None,
            "lineage": self.lineage.to_payload() if self.lineage is not None else None,
        }


@dataclass(frozen=True)
class RunSummary:
    id: str
    name: str
    task_name: str
    agent_name: str | None
    model: str | None
    reward: float | None
    has_error: bool
    skill_mode: str | None
    duration_sec: float | None = None
    cost_usd: float | None = None
    total_tokens: int | None = None
    n_tool_calls: int | None = None
    status: RunStatus | None = None
    # Forks and children recorded in the run's tree.json; None when the run
    # has no tree.json (it never branched).
    branch_forks: int | None = None
    branch_children: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_name": self.task_name,
            "agent_name": self.agent_name,
            "model": self.model,
            "reward": self.reward,
            "has_error": self.has_error,
            "skill_mode": self.skill_mode,
            "duration_sec": self.duration_sec,
            "cost_usd": self.cost_usd,
            "total_tokens": self.total_tokens,
            "n_tool_calls": self.n_tool_calls,
            "execution": self.status.execution if self.status else None,
            "execution_detail": self.status.execution_detail if self.status else None,
            "assessment": self.status.assessment if self.status else None,
            "assessment_detail": self.status.assessment_detail if self.status else None,
            "status_summary": self.status.summary if self.status else None,
            "branch_forks": self.branch_forks,
            "branch_children": self.branch_children,
        }

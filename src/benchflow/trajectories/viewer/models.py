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
    haystack = f"{kind} {title}".lower()
    for hue, needles in _HUE_INFER:
        if any(needle in haystack for needle in needles):
            return hue
    return "other"


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
class ToolStep:
    i: int
    tool: ToolCall
    t: float | None = None
    dur: float | None = None
    kind: Literal["tool"] = field(default="tool", init=False)

    def to_payload(self) -> dict[str, Any]:
        payload = _step_payload(self.i, self.kind, t=self.t)
        payload["tool"] = self.tool.to_payload()
        if self.dur is not None:
            payload["dur"] = self.dur
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


type Step = (
    PromptStep | MessageStep | ThoughtStep | ToolStep | TimeoutStep | UnknownStep
)


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

    def to_payload(self) -> dict[str, int]:
        return {
            "prompts": self.prompts,
            "messages": self.messages,
            "thoughts": self.thoughts,
            "tools": self.tools,
        }


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
class VerifierArtifacts:
    reward: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    ctrf: list[VerifierTest] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ctrf": [test.to_payload() for test in self.ctrf]
            if self.ctrf is not None
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
class ViewerPayload:
    rollout_name: str
    meta: Meta
    steps: list[Step]
    verifier: VerifierArtifacts
    rubric: RubricReview | None = None
    schema_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_name": self.rollout_name,
            "meta": self.meta.to_payload(),
            "steps": [step.to_payload() for step in self.steps],
            "verifier": self.verifier.to_payload(),
            "rubric": self.rubric.to_payload() if self.rubric is not None else None,
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
        }

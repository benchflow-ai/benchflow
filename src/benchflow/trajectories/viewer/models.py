"""Typed contract for the viewer payload — the Python ↔ JavaScript boundary.

Every field the template's JavaScript reads is declared here, once. The
``to_payload()`` methods are the only place view-model objects become JSON:
they preserve the exact wire shape the renderer and its tests pin, including
which optional keys are *omitted* (not nulled) when absent.

Tool-kind classification also lives here as the single source: Python
computes each tool call's display ``hue`` and ships it in the payload, so
the JavaScript never re-derives (and can never drift from) it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

StepKind = Literal["prompt", "message", "thought", "tool", "timeout", "unknown"]
ToolHue = Literal[
    "read", "edit", "execute", "fetch", "search", "think", "skill", "other"
]

# ACP kinds that are already display hues.
_KNOWN_HUES: frozenset[str] = frozenset(
    {"read", "edit", "execute", "fetch", "search", "think", "skill"}
)

# Fallback classification for coarse kinds ("other" covers Skill/Task/...,
# and kinds like "delete"/"move" fall outside the canonical set): infer the
# hue from kind+title needles. Order matters — earlier entries win (e.g.
# "websearch" hits search before read). Mirrors the legacy renderer's
# _tool_accent_class chain.
_HUE_INFER: tuple[tuple[ToolHue, tuple[str, ...]], ...] = (
    ("search", ("web", "search", "fetch", "grep", "glob", "browser")),
    ("execute", ("bash", "shell", "exec", "terminal", "command")),
    ("edit", ("write", "edit", "patch", "delete", "move", "notebook")),
    ("read", ("read", "cat", "view", "ls", "list")),
    ("skill", ("agent", "task", "skill", "oracle")),
)


def tool_hue(kind: str, title: str) -> ToolHue:
    """Display hue for a tool call — the one classification site."""
    if kind in _KNOWN_HUES:
        return cast(ToolHue, kind)
    hay = f"{kind} {title}".lower()
    for hue, needles in _HUE_INFER:
        if any(needle in hay for needle in needles):
            return hue
    return "other"


@dataclass
class ToolCall:
    """One tool invocation as the trace stream renders it."""

    id: str
    kind: str
    title: str
    status: str
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


@dataclass
class TimeoutInfo:
    """agent_timeout event details."""

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


@dataclass
class Step:
    """One renderer step. Optional fields are omitted from the wire when
    absent — the template (and the timestamp tests) rely on key presence,
    not null checks."""

    i: int
    kind: StepKind
    label: str | None = None
    text: str | None = None
    type: str | None = None  # original event type, unknown steps only
    tool: ToolCall | None = None
    timeout: TimeoutInfo | None = None
    t: float | None = None  # epoch seconds, when the capture has timestamps
    dur: float | None = None  # tool duration seconds

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"i": self.i, "kind": self.kind}
        if self.label is not None:
            out["label"] = self.label
        if self.text is not None:
            out["text"] = self.text
        if self.type is not None:
            out["type"] = self.type
        if self.tool is not None:
            out["tool"] = self.tool.to_payload()
        if self.timeout is not None:
            out["timeout"] = self.timeout.to_payload()
        if self.t is not None:
            out["t"] = self.t
        if self.dur is not None:
            out["dur"] = self.dur
        return out


@dataclass
class ErrorBanner:
    """One header banner. ``level`` is present only for registry diagnostics
    ("error" riding along a real failure, "info" for behavior flags on an
    otherwise-clean rollout); explicit error channels carry no level and
    always render as errors."""

    label: str
    text: str
    level: Literal["error", "info"] | None = None

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label, "text": self.text}
        if self.level is not None:
            out["level"] = self.level
        return out


@dataclass
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


@dataclass
class Meta:
    """Header metadata from result.json/timing.json. Every key ships (null
    when unknown) — the template hides absent stats itself."""

    task_name: str | None = None
    agent_name: str | None = None
    model: str | None = None
    skill_mode: str | None = None
    reward: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    counts: StepCounts = field(default_factory=StepCounts)
    timing: dict[str, Any] | None = None
    duration_sec: float | None = None
    errors: list[ErrorBanner] = field(default_factory=list)
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
            "usage": self.usage,
            "counts": self.counts.to_payload(),
            "timing": self.timing,
            "duration_sec": self.duration_sec,
            "errors": [e.to_payload() for e in self.errors],
            "trajectory_source": self.trajectory_source,
            "partial_trajectory": self.partial_trajectory,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class VerifierArtifacts:
    reward: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    ctrf: list[dict[str, Any]] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ctrf": self.ctrf,
        }


@dataclass
class ViewerPayload:
    """The complete single-run payload embedded in (or fetched by) the page."""

    rollout_name: str
    meta: Meta
    steps: list[Step]
    verifier: VerifierArtifacts
    schema_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_name": self.rollout_name,
            "meta": self.meta.to_payload(),
            "steps": [s.to_payload() for s in self.steps],
            "verifier": self.verifier.to_payload(),
        }


@dataclass
class RunSummary:
    """One browse-mode sidebar row."""

    id: str
    name: str
    task_name: str
    agent_name: str | None
    model: str | None
    reward: float | None
    has_error: bool
    skill_mode: str | None

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
        }

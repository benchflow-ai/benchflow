"""ATIF trajectory model — Agent Trajectory Interchange Format.

Canonical schema for recording a complete agent session as a sequence of
steps (system/user/agent), each with optional tool calls, observations,
and per-step token metrics. Designed to be agent-agnostic and exportable
as JSON for cross-tool analysis.

Status: backlog — implemented but not wired into SDK trajectory capture yet.
See STATUS.md "Later" section (ATIF export).
Related: trajectories/claude_code.py (converts Claude Code output → ATIF).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    """A text or image fragment within a step message or observation."""

    type: Literal["text", "image"] = "text"
    text: str | None = None
    source: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """Record of a single tool invocation by the agent."""

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ObservationResult(BaseModel):
    """Output returned by a tool call (linked via source_call_id)."""

    source_call_id: str | None = None
    content: str | list[ContentPart] = ""


class Observation(BaseModel):
    """Collection of tool-call outputs observed after a step's tool calls."""

    results: list[ObservationResult] = Field(default_factory=list)


class Metrics(BaseModel):
    """Token usage and cost for a single LLM call or aggregated over a session."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_tokens: int | None = None
    cost_usd: float | None = None


class Step(BaseModel):
    """One turn in the trajectory: a system, user, or agent action.

    Agent steps may include tool_calls + observation; user steps carry the
    prompt text; system steps carry setup or environment messages.
    """

    step_id: int = Field(ge=1)
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    message: str | list[ContentPart] = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None
    extra: dict[str, Any] | None = None


class Agent(BaseModel):
    """Identity and configuration of the agent that produced the trajectory."""

    name: str
    version: str = ""
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFTrajectory(BaseModel):
    """Top-level ATIF document: one agent, one session, ordered steps."""

    schema_version: str = "ATIF-v1.6"
    session_id: str
    agent: Agent
    steps: list[Step] = Field(default_factory=list)
    notes: str | None = None
    final_metrics: Metrics | None = None
    extra: dict[str, Any] | None = None

    def add_step(
        self,
        source: Literal["system", "user", "agent"],
        message: str,
        **kwargs: Any,
    ) -> Step:
        step = Step(
            step_id=len(self.steps) + 1,
            source=source,
            message=message,
            **kwargs,
        )
        self.steps.append(step)
        return step

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

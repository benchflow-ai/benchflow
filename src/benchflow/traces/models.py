"""Intermediate representation for parsed agent traces.

These types are format-agnostic — both Claude Code JSONL and opentraces
records are normalized into ``ParsedTrace`` before task generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCall:
    """A single tool invocation within a trace step."""

    name: str
    input: dict[str, object] = field(default_factory=dict)
    output: str | None = None


@dataclass
class TraceStep:
    """One turn in a conversation (user prompt or assistant response)."""

    role: str  # "user" | "assistant" | "system" | "tool_result"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str | None = None


@dataclass
class GitContext:
    """Git metadata extracted from a trace."""

    repo: str | None = None
    branch: str | None = None
    commit_before: str | None = None
    commit_after: str | None = None


@dataclass
class ParsedTrace:
    """Normalized representation of an agent coding session.

    This is the common intermediate format that both Claude Code sessions
    and opentraces records are parsed into before task generation.
    """

    trace_id: str
    session_id: str
    agent_name: str = "claude-code"
    model: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    steps: list[TraceStep] = field(default_factory=list)
    git: GitContext = field(default_factory=GitContext)
    cwd: str | None = None
    outcome: str | None = None  # "success" | "failure" | "unknown"
    tags: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def first_user_prompt(self) -> str | None:
        """Extract the first user message as the task instruction seed."""
        for step in self.steps:
            if step.role == "user" and step.content.strip():
                return step.content.strip()
        return None

    @property
    def tool_names_used(self) -> list[str]:
        """Unique tool names invoked across all steps."""
        seen: set[str] = set()
        names: list[str] = []
        for step in self.steps:
            for tc in step.tool_calls:
                if tc.name not in seen:
                    seen.add(tc.name)
                    names.append(tc.name)
        return names

    @property
    def files_edited(self) -> list[str]:
        """File paths that were written/edited based on tool calls."""
        paths: list[str] = []
        write_tools = {"Write", "Edit", "MultiEdit", "write_to_file", "edit_file"}
        for step in self.steps:
            for tc in step.tool_calls:
                if tc.name in write_tools:
                    path = tc.input.get("file_path") or tc.input.get("path")
                    if isinstance(path, str) and path not in paths:
                        paths.append(path)
        return paths

    @property
    def n_tool_calls(self) -> int:
        """Total number of tool invocations."""
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def duration_sec(self) -> float | None:
        """Session duration in seconds, if timestamps are available."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

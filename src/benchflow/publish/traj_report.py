"""Format-aware summaries for locally staged trajectory JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from benchflow.publish.traj_capture import strict_json_loads

DEFAULT_PREVIEW_STEPS = 5
MAX_PREVIEW_STEPS = 20
PREVIEW_WORD_LIMIT = 100
_PREVIEW_CHAR_SAFETY_LIMIT = 4000


class TrajectoryFormat(Enum):
    BENCHFLOW_ACP = "BenchFlow ACP"
    CLAUDE_CODE = "Claude Code"
    CODEX = "Codex"
    LLM_EXCHANGE = "LLM exchanges"
    OPENTRACES = "OpenTrace"
    GENERIC = "Generic JSONL"


@dataclass(frozen=True)
class TrajectoryPreviewStep:
    number: int
    kind: str
    summary: str


@dataclass(frozen=True)
class TrajectoryReport:
    primary_file: str
    format: TrajectoryFormat
    file_count: int
    size_bytes: int
    total_steps: int
    thinking_steps: int
    tool_call_steps: int
    human_steps: int
    created_at: datetime
    created_at_source: str
    masked_values: int
    preview: tuple[TrajectoryPreviewStep, ...]


class TrajectoryArtifact(Protocol):
    relname: str
    local_path: Path
    size_bytes: int
    created_at: datetime | None


@dataclass(frozen=True)
class _Step:
    kind: str
    summary: str
    thinking: bool = False
    tool_call: bool = False
    human_steps: int = 0


@dataclass
class _Analysis:
    preview_limit: int
    total_steps: int = 0
    thinking_steps: int = 0
    tool_call_steps: int = 0
    human_steps: int = 0
    preview: list[TrajectoryPreviewStep] = field(default_factory=list)
    previous_llm_messages: list[Any] | None = None

    def add(self, step: _Step) -> None:
        self.total_steps += 1
        self.thinking_steps += int(step.thinking)
        self.tool_call_steps += int(step.tool_call)
        self.human_steps += step.human_steps
        if len(self.preview) < self.preview_limit:
            self.preview.append(
                TrajectoryPreviewStep(
                    number=self.total_steps,
                    kind=step.kind,
                    summary=_preview_text(step.summary),
                )
            )


def build_trajectory_report(
    artifacts: tuple[TrajectoryArtifact, ...],
    *,
    masked_values: int,
    preview_steps: int = DEFAULT_PREVIEW_STEPS,
) -> TrajectoryReport:
    """Summarize one canonical trajectory view from staged, redacted artifacts."""
    if not artifacts:
        raise ValueError("trajectory report requires at least one JSONL artifact")
    if not 0 <= preview_steps <= MAX_PREVIEW_STEPS:
        raise ValueError(f"trajectory preview must contain 0-{MAX_PREVIEW_STEPS} steps")

    primary = _primary_artifact(artifacts)
    analysis = _Analysis(preview_limit=preview_steps)
    trajectory_format = _detect_file_format(primary.local_path)
    earliest_timestamp: datetime | None = None

    with primary.local_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = strict_json_loads(line)
            if not isinstance(record, dict):  # staging already enforces this
                continue
            timestamp = _record_timestamp(record)
            if timestamp is not None and (
                earliest_timestamp is None or timestamp < earliest_timestamp
            ):
                earliest_timestamp = timestamp
            for step in _record_steps(trajectory_format, record, analysis):
                analysis.add(step)

    created_at = earliest_timestamp or primary.created_at or datetime.now(UTC)
    created_at_source = (
        "trajectory timestamp" if earliest_timestamp is not None else "file timestamp"
    )
    return TrajectoryReport(
        primary_file=primary.relname,
        format=trajectory_format,
        file_count=len(artifacts),
        size_bytes=sum(item.size_bytes for item in artifacts),
        total_steps=analysis.total_steps,
        thinking_steps=analysis.thinking_steps,
        tool_call_steps=analysis.tool_call_steps,
        human_steps=analysis.human_steps,
        created_at=created_at,
        created_at_source=created_at_source,
        masked_values=masked_values,
        preview=tuple(analysis.preview),
    )


def _primary_artifact(
    artifacts: tuple[TrajectoryArtifact, ...],
) -> TrajectoryArtifact:
    def priority(item: TrajectoryArtifact) -> tuple[int, str]:
        name = Path(item.relname).name.casefold()
        if name == "acp_trajectory.jsonl":
            return 0, item.relname
        if name == "llm_trajectory.jsonl":
            return 2, item.relname
        return 1, item.relname

    return min(artifacts, key=priority)


def _detect_file_format(path: Path) -> TrajectoryFormat:
    """Find the first recognized trajectory record after optional metadata."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = strict_json_loads(line)
            if not isinstance(record, dict):
                continue
            detected = _detect_format(record)
            if detected is not TrajectoryFormat.GENERIC:
                return detected
    return TrajectoryFormat.GENERIC


def _detect_format(record: dict[str, Any]) -> TrajectoryFormat:
    record_type = _lower(record.get("type"))
    if record_type in {
        "session_meta",
        "event_msg",
        "response_item",
        "turn_context",
        "world_state",
    } and isinstance(record.get("payload"), dict):
        return TrajectoryFormat.CODEX
    if record_type in {"user", "assistant", "system", "progress"} and isinstance(
        record.get("message"), dict
    ):
        return TrajectoryFormat.CLAUDE_CODE
    if record_type in {
        "user_message",
        "agent_message",
        "agent_thought",
        "tool_call",
        "agent_timeout",
    }:
        return TrajectoryFormat.BENCHFLOW_ACP
    if isinstance(record.get("request"), dict) and isinstance(
        record.get("response"), dict
    ):
        return TrajectoryFormat.LLM_EXCHANGE
    if isinstance(record.get("steps"), list) and (
        "trace_id" in record or "schema_version" in record
    ):
        return TrajectoryFormat.OPENTRACES
    return TrajectoryFormat.GENERIC


def _record_steps(
    trajectory_format: TrajectoryFormat,
    record: dict[str, Any],
    analysis: _Analysis,
) -> tuple[_Step, ...]:
    if trajectory_format is TrajectoryFormat.CODEX:
        return _codex_steps(record)
    if trajectory_format is TrajectoryFormat.CLAUDE_CODE:
        return _claude_steps(record)
    if trajectory_format is TrajectoryFormat.BENCHFLOW_ACP:
        return _benchflow_steps(record)
    if trajectory_format is TrajectoryFormat.LLM_EXCHANGE:
        return _llm_exchange_steps(record, analysis)
    if trajectory_format is TrajectoryFormat.OPENTRACES:
        return _opentraces_steps(record)
    return (_generic_step(record),)


def _codex_steps(record: dict[str, Any]) -> tuple[_Step, ...]:
    if record.get("type") != "response_item":
        return ()
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return ()
    payload_type = _lower(payload.get("type"))
    if payload_type == "message":
        role = _lower(payload.get("role"))
        if role not in {"user", "assistant"}:
            return ()
        return (
            _Step(
                kind="Human" if role == "user" else "Assistant",
                summary=_content_text(payload.get("content")) or f"{role} message",
                human_steps=int(role == "user"),
            ),
        )
    if payload_type == "reasoning":
        return (
            _Step(
                kind="Thinking",
                summary=_content_text(payload.get("summary"))
                or _content_text(payload.get("content"))
                or "Reasoning step",
                thinking=True,
            ),
        )
    if payload_type.endswith("_call") or payload_type in {
        "function_call",
        "custom_tool_call",
    }:
        name = str(payload.get("name") or payload_type.replace("_", " "))
        summary = _tool_text(name, payload.get("arguments") or payload.get("input"))
        return (_Step(kind="Tool call", summary=summary, tool_call=True),)
    if payload_type.endswith("_call_output"):
        return (
            _Step(
                kind="Tool result",
                summary=_content_text(payload.get("output")) or "Tool result returned",
            ),
        )
    return ()


def _claude_steps(record: dict[str, Any]) -> tuple[_Step, ...]:
    record_type = _lower(record.get("type"))
    if record_type not in {"user", "assistant"}:
        return ()
    message = record.get("message")
    if not isinstance(message, dict):
        return ()
    content = message.get("content")
    block_types = _block_types(content)
    if record_type == "user":
        only_tool_results = bool(block_types) and block_types <= {"tool_result"}
        return (
            _Step(
                kind="Tool result" if only_tool_results else "Human",
                summary=_content_text(content)
                or ("Tool result returned" if only_tool_results else "Human prompt"),
                human_steps=int(not only_tool_results),
            ),
        )
    thinking = bool(block_types & {"thinking", "redacted_thinking"})
    tool_call = bool(block_types & {"tool_use", "server_tool_use"})
    return (
        _Step(
            kind=_assistant_kind(thinking=thinking, tool_call=tool_call),
            summary=_content_text(content)
            or _first_tool_name(content)
            or "Assistant response",
            thinking=thinking,
            tool_call=tool_call,
        ),
    )


def _benchflow_steps(record: dict[str, Any]) -> tuple[_Step, ...]:
    record_type = _lower(record.get("type"))
    if record_type == "user_message":
        return (
            _Step(
                kind="Human",
                summary=_content_text(record.get("text")) or "Human prompt",
                human_steps=1,
            ),
        )
    if record_type == "agent_thought":
        return (
            _Step(
                kind="Thinking",
                summary=_content_text(record.get("text")) or "Reasoning step",
                thinking=True,
            ),
        )
    if record_type == "tool_call":
        title = str(record.get("title") or record.get("kind") or "Tool call")
        content = _content_text(record.get("content"))
        summary = f"{title}: {content}" if content else title
        return (_Step(kind="Tool call", summary=summary, tool_call=True),)
    if record_type == "agent_message":
        return (
            _Step(
                kind="Assistant",
                summary=_content_text(record.get("text")) or "Assistant response",
            ),
        )
    if record_type == "agent_timeout":
        return (_Step(kind="Status", summary="Agent timeout"),)
    return ()


def _llm_exchange_steps(
    record: dict[str, Any], analysis: _Analysis
) -> tuple[_Step, ...]:
    request = record.get("request")
    response = record.get("response")
    request_body = request.get("body") if isinstance(request, dict) else None
    response_body = response.get("body") if isinstance(response, dict) else None
    request_body = request_body if isinstance(request_body, dict) else {}
    response_body = response_body if isinstance(response_body, dict) else {}
    messages = request_body.get("messages")
    current_messages = messages if isinstance(messages, list) else []
    previous = analysis.previous_llm_messages or []
    new_messages = (
        current_messages[len(previous) :]
        if current_messages[: len(previous)] == previous
        else current_messages
    )
    analysis.previous_llm_messages = list(current_messages)
    human_steps = sum(
        1
        for message in new_messages
        if isinstance(message, dict) and _lower(message.get("role")) == "user"
    )
    thinking, tool_call = _response_signals(response_body)
    summary = (
        _response_text(response_body)
        or _last_message_text(new_messages)
        or "Model exchange"
    )
    return (
        _Step(
            kind=_assistant_kind(thinking=thinking, tool_call=tool_call),
            summary=summary,
            thinking=thinking,
            tool_call=tool_call,
            human_steps=human_steps,
        ),
    )


def _opentraces_steps(record: dict[str, Any]) -> tuple[_Step, ...]:
    steps: list[_Step] = []
    task = record.get("task")
    if isinstance(task, dict) and (prompt := _content_text(task.get("input"))):
        steps.append(_Step(kind="Human", summary=prompt, human_steps=1))
    for item in record.get("steps") or []:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        action = action if isinstance(action, dict) else {}
        tool = action.get("tool_call")
        tool = tool if isinstance(tool, dict) else {}
        thought = _content_text(item.get("thought"))
        tool_name = str(tool.get("name") or "")
        thinking = bool(thought)
        tool_call = bool(tool)
        steps.append(
            _Step(
                kind=_assistant_kind(thinking=thinking, tool_call=tool_call),
                summary=thought or tool_name or "Trace step",
                thinking=thinking,
                tool_call=tool_call,
            )
        )
    return tuple(steps)


def _generic_step(record: dict[str, Any]) -> _Step:
    record_type = _lower(record.get("type") or record.get("kind"))
    role = _lower(record.get("role"))
    human = role in {"user", "human"} or record_type in {"user", "human"}
    thinking = record_type in {
        "thinking",
        "reasoning",
        "thought",
        "agent_thought",
    } or any(
        record.get(key) not in (None, "", [], {}) for key in ("thinking", "thought")
    )
    tool_call = record_type in {"tool_call", "function_call", "tool_use"} or bool(
        record.get("tool_calls")
    )
    summary = (
        _content_text(record.get("text"))
        or _content_text(record.get("content"))
        or _content_text(record.get("message"))
        or _content_text(record.get("payload"))
        or str(record.get("title") or record_type or "JSONL event")
    )
    kind = "Human" if human else _assistant_kind(thinking=thinking, tool_call=tool_call)
    return _Step(
        kind=kind,
        summary=summary,
        thinking=thinking,
        tool_call=tool_call,
        human_steps=int(human),
    )


def _assistant_kind(*, thinking: bool, tool_call: bool) -> str:
    if thinking and tool_call:
        return "Thinking + tool"
    if thinking:
        return "Thinking"
    if tool_call:
        return "Tool call"
    return "Assistant"


def _record_timestamp(record: dict[str, Any]) -> datetime | None:
    candidates = [
        record.get("timestamp"),
        record.get("created_at"),
        record.get("timestamp_start"),
    ]
    for container_name in ("request", "response", "payload"):
        container = record.get(container_name)
        if isinstance(container, dict):
            candidates.extend((container.get("timestamp"), container.get("created_at")))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _block_types(content: Any) -> set[str]:
    if not isinstance(content, list):
        return set()
    return {
        _lower(block.get("type"))
        for block in content
        if isinstance(block, dict) and block.get("type")
    }


def _first_tool_name(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, dict) and _lower(block.get("type")) in {
            "tool_use",
            "server_tool_use",
        }:
            return str(block.get("name") or "Tool call")
    return ""


def _response_signals(value: dict[str, Any]) -> tuple[bool, bool]:
    """Find reasoning and tool-call signals in one response traversal."""
    thinking = False
    tool_call = False
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            thinking = thinking or any(
                item.get(key) not in (None, "", [], {})
                for key in ("reasoning", "thinking")
            )
            tool_call = tool_call or bool(
                item.get("tool_calls") or item.get("function_call")
            )
            item_type = _lower(item.get("type"))
            thinking = thinking or item_type in {
                "thinking",
                "reasoning",
                "reasoning_content",
            }
            tool_call = tool_call or item_type in {
                "tool_use",
                "server_tool_use",
                "function_call",
                "tool_call",
            }
            if thinking and tool_call:
                break
            stack.extend(
                child
                for key, child in item.items()
                if key in {"choices", "content", "message", "output"}
            )
        elif isinstance(item, list):
            stack.extend(item)
    return thinking, tool_call


def _response_text(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and (
                text := _content_text(message.get("content"))
            ):
                return text
    return _content_text(response_body.get("content"))


def _last_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and (
            text := _content_text(message.get("content"))
        ):
            return text
    return ""


def _content_text(value: Any) -> str:
    return _compact(" ".join(_content_parts(value)))


def _content_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        compact = _compact(value)
        return [compact] if compact else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_content_parts(item))
        return parts
    if isinstance(value, dict):
        item_type = _lower(value.get("type"))
        if item_type in {
            "tool_use",
            "server_tool_use",
            "function_call",
            "tool_call",
        }:
            name = str(value.get("name") or item_type.replace("_", " "))
            return [_tool_text(name, value.get("input") or value.get("arguments"))]
        parts = []
        for key in (
            "text",
            "thinking",
            "reasoning",
            "summary",
            "content",
            "output",
            "message",
            "prompt",
        ):
            if key in value:
                parts.extend(_content_parts(value[key]))
        return parts
    return []


def _tool_text(name: str, details: Any) -> str:
    if details in (None, "", [], {}):
        return name
    if isinstance(details, str):
        rendered = _compact(details)
    else:
        rendered = json.dumps(details, ensure_ascii=False, sort_keys=True)
    return f"{name}: {rendered}"


def _compact(value: str) -> str:
    return " ".join(value.split())


def _preview_text(value: str) -> str:
    words = _compact(value).split()
    preview = " ".join(words[:PREVIEW_WORD_LIMIT])
    truncated = len(words) > PREVIEW_WORD_LIMIT
    if len(preview) > _PREVIEW_CHAR_SAFETY_LIMIT:
        preview = preview[:_PREVIEW_CHAR_SAFETY_LIMIT].rstrip()
        truncated = True
    return f"{preview}…" if truncated else preview


def _lower(value: Any) -> str:
    return value.casefold() if isinstance(value, str) else ""

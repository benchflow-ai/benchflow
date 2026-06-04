"""ACP session lifecycle management."""

import logging
from datetime import datetime

from benchflow.trajectories.metrics import is_skill_invocation_event

from .types import (
    AgentCapabilities,
    AgentInfo,
    StopReason,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


def _is_skill_tool_call(
    kind: object, title: object = "", content: object = None
) -> bool:
    """Classify a live ACP tool call via the shared trajectory classifier.

    Builds a synthetic trajectory event so live capture and historical rescans
    apply one identical definition of "skill invocation". Crucially, the tool's
    own ``kind`` gates content sniffing, so a ``read`` / ``execute`` / ``search``
    tool whose output quotes a legacy ``invoke_skill`` envelope is not
    reclassified as a skill.
    """
    return is_skill_invocation_event(
        {"type": "tool_call", "kind": kind, "title": title, "content": content}
    )


def _canonical_tool_kind(kind: object, title: object = "") -> str:
    raw_kind = kind if isinstance(kind, str) and kind else "other"
    if _is_skill_tool_call(kind, title):
        return "skill"
    return raw_kind


class ToolCallRecord:
    """Record of a single tool call within a session.

    Tracks identity (tool_call_id, title, kind), lifecycle status, captured
    content blocks, and wall-clock timing.
    """

    def __init__(self, tool_call_id: str, title: str, kind: str):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = ToolCallStatus.PENDING
        self.content: list[dict] = []
        self.started_at = datetime.now()
        self.finished_at: datetime | None = None

    def update_status(
        self, status: ToolCallStatus, content: list[dict] | None = None
    ) -> None:
        self.status = status
        if content:
            self.content.extend(content)
        if status in (
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
        ):
            self.finished_at = datetime.now()


class ACPSession:
    """Tracks mutable state for one ACP session.

    Accumulates streaming chunks (message_chunks, thought_chunks) and
    tool-call records as session/update notifications arrive.  Use
    ``full_message`` / ``full_thought`` to read the assembled text.

    The ``events`` list records every significant event in chronological
    order (user prompts, tool calls, message/thought boundaries) so that
    ``_capture_session_trajectory`` can produce a faithful interleaved
    trajectory instead of a flat blob.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent_info: AgentInfo | None = None
        self.agent_capabilities: AgentCapabilities | None = None
        self.model_state: dict | None = None
        self.message_chunks: list[str] = []
        self.thought_chunks: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self._tool_call_map: dict[str, ToolCallRecord] = {}
        self.stop_reason: StopReason | None = None
        self.created_at = datetime.now()
        self.events: list[dict] = []
        self._pending_text: list[dict] = []
        self._events_active: bool = False

    def record_user_prompt(self, text: str) -> None:
        """Record a user prompt. Call before sending each ACP prompt."""
        self._events_active = True
        self._flush_agent_text()
        self.events.append({"type": "user_message", "text": text})

    def mark_prompt_end(self) -> None:
        """Flush pending agent text after a prompt completes."""
        self._flush_agent_text()

    def _flush_agent_text(self) -> None:
        """Flush pending text events, merging consecutive same-type chunks."""
        if not self._pending_text:
            return
        current = self._pending_text[0].copy()
        for event in self._pending_text[1:]:
            if event["type"] == current["type"]:
                current["text"] += event["text"]
            else:
                self.events.append(current)
                current = event.copy()
        self.events.append(current)
        self._pending_text.clear()

    def handle_update(self, update: dict) -> None:
        """Process a session/update notification."""
        self._events_active = True
        update_type = update.get("sessionUpdate")

        if update_type == "tool_call":
            self._flush_agent_text()
            record = ToolCallRecord(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=_canonical_tool_kind(
                    update.get("kind", "other"), update.get("title", "")
                ),
            )
            self.tool_calls.append(record)
            self._tool_call_map[record.tool_call_id] = record
            self.events.append({"type": "tool_call", "record": record})

        elif update_type == "tool_call_update":
            tc_id = update.get("toolCallId", "")
            record = self._tool_call_map.get(tc_id)
            if not record:
                self._flush_agent_text()
                record = ToolCallRecord(
                    tool_call_id=tc_id,
                    title=update.get("title", ""),
                    kind=_canonical_tool_kind(
                        update.get("kind", "tool"), update.get("title", "")
                    ),
                )
                self.tool_calls.append(record)
                self._tool_call_map[tc_id] = record
                self.events.append({"type": "tool_call", "record": record})
            try:
                status = ToolCallStatus(update.get("status", "in_progress"))
            except ValueError:
                logger.warning(f"Unknown tool call status: {update.get('status')}")
                status = ToolCallStatus.IN_PROGRESS
            content = update.get("content")
            record.update_status(status, content)
            # Canonicalize legacy OpenHands invoke_skill calls using the same
            # classifier the rescan path uses. Only upgrade to "skill"; never
            # downgrade, and never reclassify a tool that already has a real
            # ACP kind (its output may merely quote a skill envelope).
            if record.kind != "skill" and _is_skill_tool_call(
                record.kind, record.title, record.content
            ):
                record.kind = "skill"

        elif update_type == "agent_message_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                text = content.get("text", "")
                self.message_chunks.append(text)
                self._pending_text.append({"type": "agent_message", "text": text})

        elif update_type == "text_update":
            # Used by openclaw shim — full text (not chunked)
            text = update.get("text", "")
            if text:
                self.message_chunks.append(text)
                self._pending_text.append({"type": "agent_message", "text": text})

        elif update_type == "agent_thought":
            # Used by openclaw shim — full thought (not chunked)
            text = update.get("text", "")
            if text:
                self.thought_chunks.append(text)
                self._pending_text.append({"type": "agent_thought", "text": text})

        elif update_type == "agent_thought_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                text = content.get("text", "")
                self.thought_chunks.append(text)
                self._pending_text.append({"type": "agent_thought", "text": text})

    @property
    def full_message(self) -> str:
        """Concatenated agent message text from all received chunks."""
        return "".join(self.message_chunks)

    @property
    def full_thought(self) -> str:
        """Concatenated agent thought/reasoning text from all received chunks."""
        return "".join(self.thought_chunks)

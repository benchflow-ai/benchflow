"""ACP session lifecycle management."""

import logging
from datetime import datetime

from .types import (
    AgentCapabilities,
    AgentInfo,
    StopReason,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent_info: AgentInfo | None = None
        self.agent_capabilities: AgentCapabilities | None = None
        self.message_chunks: list[str] = []
        self.thought_chunks: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self._tool_call_map: dict[str, ToolCallRecord] = {}
        self.stop_reason: StopReason | None = None
        self.created_at = datetime.now()

    def handle_update(self, update: dict) -> None:
        """Process a session/update notification."""
        update_type = update.get("sessionUpdate")

        if update_type == "tool_call":
            record = ToolCallRecord(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=update.get("kind", "other"),
            )
            self.tool_calls.append(record)
            self._tool_call_map[record.tool_call_id] = record

        elif update_type == "tool_call_update":
            tc_id = update.get("toolCallId", "")
            record = self._tool_call_map.get(tc_id)
            if not record:
                # Auto-create record for agents that skip the initial tool_call
                # notification (e.g. Gemini CLI sends only tool_call_update)
                record = ToolCallRecord(
                    tool_call_id=tc_id,
                    title=update.get("title", ""),
                    kind=update.get("kind", "tool"),
                )
                self.tool_calls.append(record)
                self._tool_call_map[tc_id] = record
            try:
                status = ToolCallStatus(update.get("status", "in_progress"))
            except ValueError:
                logger.warning(f"Unknown tool call status: {update.get('status')}")
                status = ToolCallStatus.IN_PROGRESS
            record.update_status(status, update.get("content"))

        elif update_type == "agent_message_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                self.message_chunks.append(content.get("text", ""))

        elif update_type == "text_update":
            # Used by openclaw shim — full text (not chunked)
            text = update.get("text", "")
            if text:
                self.message_chunks.append(text)

        elif update_type == "agent_thought":
            # Used by openclaw shim — full thought (not chunked)
            text = update.get("text", "")
            if text:
                self.thought_chunks.append(text)

        elif update_type == "agent_thought_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                self.thought_chunks.append(content.get("text", ""))

    @property
    def full_message(self) -> str:
        """Concatenated agent message text from all received chunks."""
        return "".join(self.message_chunks)

    @property
    def full_thought(self) -> str:
        """Concatenated agent thought/reasoning text from all received chunks."""
        return "".join(self.thought_chunks)

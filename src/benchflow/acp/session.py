"""ACP session lifecycle management."""

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from benchflow.trajectories.metrics import is_skill_invocation_event

from .types import (
    AgentCapabilities,
    AgentInfo,
    StopReason,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)

# Console progress heartbeat contract. ``BENCHFLOW_PROGRESS`` is the operator
# switch (on/off); unset defers to ``BENCHFLOW_PROGRESS_AUTO``, which the
# Evaluation layer sets to "0" for multi-concurrency jobs where interleaved
# per-task lines would be noise. ``bench eval run --quiet`` sets the explicit
# off value.
_PROGRESS_ENV = "BENCHFLOW_PROGRESS"
_PROGRESS_AUTO_ENV = "BENCHFLOW_PROGRESS_AUTO"
_PROGRESS_INTERVAL_SEC = 45.0
_PROGRESS_OFF_VALUES = frozenset({"off", "0", "false", "none"})
_PROGRESS_ON_VALUES = frozenset({"on", "1", "true"})


def _console_progress_enabled() -> bool:
    raw = os.environ.get(_PROGRESS_ENV, "").strip().lower()
    if raw in _PROGRESS_OFF_VALUES:
        return False
    if raw in _PROGRESS_ON_VALUES:
        return True
    return os.environ.get(_PROGRESS_AUTO_ENV, "1").strip() != "0"


ACPUsageSnapshot = dict[str, int | None]

_ACP_USAGE_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_read_tokens",
    "cached_write_tokens",
    "thought_tokens",
)


def _coerce_usage_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _usage_mapping(usage: object) -> dict[str, Any]:
    if isinstance(usage, dict):
        return {str(key): value for key, value in usage.items()}
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        data = dump(by_alias=False, exclude_none=True)
        if isinstance(data, dict):
            alias_data = dump(by_alias=True, exclude_none=True)
            if isinstance(alias_data, dict):
                data = {**alias_data, **data}
            return data
    return {
        field: getattr(usage, field)
        for field in _ACP_USAGE_FIELDS
        if hasattr(usage, field)
    }


def normalize_acp_usage(usage: object | None) -> ACPUsageSnapshot | None:
    """Normalize SDK ACP usage into BenchFlow's snake_case token counters."""
    if usage is None:
        return None
    raw = _usage_mapping(usage)
    if not raw:
        return None
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "total_tokens": ("total_tokens", "totalTokens"),
        "cached_read_tokens": ("cached_read_tokens", "cachedReadTokens"),
        "cached_write_tokens": ("cached_write_tokens", "cachedWriteTokens"),
        "thought_tokens": ("thought_tokens", "thoughtTokens"),
    }
    snapshot: ACPUsageSnapshot = {}
    for field, names in aliases.items():
        value = None
        for name in names:
            if name in raw:
                value = raw[name]
                break
        snapshot[field] = _coerce_usage_int(value)
    if all(value is None for value in snapshot.values()):
        return None
    return snapshot


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


def _tool_display_title(title: str, kind: str) -> str:
    """A tool call's display title: first line of ``title``, else ``kind``.

    ``split`` not ``splitlines`` so a whitespace-only title strips to ``""``
    instead of raising on an empty line list. Shared by the heartbeat/dashboard
    snapshot and the distinct-title tracker so "distinct" means exactly
    "renders differently".
    """
    return (title or kind or "").strip().split("\n", 1)[0]


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
        # ACP ``rawInput`` / ``rawOutput``: the agent's own view of the call.
        # codex-acp puts the command and its output here and nowhere else.
        self.raw_input: object | None = None
        self.raw_output: object | None = None
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

    def absorb_raw_io(self, update: dict) -> None:
        """Keep the latest ``rawInput`` / ``rawOutput`` an update carries."""
        if update.get("rawInput") is not None:
            self.raw_input = update["rawInput"]
        if update.get("rawOutput") is not None:
            self.raw_output = update["rawOutput"]


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
        self.config_options: list[dict] = []
        self.message_chunks: list[str] = []
        self.thought_chunks: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self._tool_call_map: dict[str, ToolCallRecord] = {}
        # Total update count is diagnostic. Per-call pending versions let the
        # idle watchdog distinguish relevant streaming progress from repeated
        # terminal updates for another call (PR #1066).
        self.tool_call_update_count: int = 0
        self._pending_tool_call_update_counts: dict[str, int] = {}
        # Distinct display titles across recorded tool calls, maintained
        # incrementally at record creation so the eval dashboard can detect
        # single-tool agents (prime-agent funnels everything through one
        # IPython tool — a constant "last: IPython cell" carries no
        # information) in O(1) at render time.
        self._seen_tool_titles: set[str] = set()
        self.stop_reason: StopReason | None = None
        self.usage_snapshots: list[ACPUsageSnapshot] = []
        self.created_at = datetime.now()
        self.events: list[dict] = []
        self._pending_text: list[dict] = []
        self._events_active: bool = False
        # Optional sink invoked after every public state mutation so callers
        # can stream a trajectory snapshot to disk without polling.
        self.on_change: Callable[[ACPSession], None] | None = None
        # Console progress heartbeat. Without it a first-run user stares at
        # total silence between "Prompt 1/1: ..." and the verifier — observed
        # 18 minutes on a passing rollout (fresh-user dogfood 2026-08-09) with
        # no way to distinguish "working" from "hung". The throttle keys off
        # _notify_change, which fires on every streamed update.
        self._progress_enabled = _console_progress_enabled()
        self._prompt_started_at: float | None = None
        self._last_progress_at = 0.0

    def _notify_change(self) -> None:
        self._maybe_log_progress()
        if self.on_change is None:
            return
        try:
            self.on_change(self)
        except Exception as e:
            # error (not warning): a failing callback means trajectory
            # streaming is silently degraded for the rest of the run,
            # which is otherwise easy to miss in a 64-concurrency log.
            logger.error(f"ACPSession on_change callback failed: {e}")

    def progress_snapshot(self) -> tuple[int, str]:
        """(tool-call count, last tool title) — the console heartbeat's
        counters, also polled by the live eval dashboard's activity cell.

        The title is the raw first line (untruncated — display width belongs
        to each render site; see :func:`_tool_display_title`).
        """
        title = ""
        if self.tool_calls:
            last = self.tool_calls[-1]
            title = _tool_display_title(last.title, last.kind)
        return len(self.tool_calls), title

    @property
    def distinct_tool_titles(self) -> int:
        """Count of distinct display titles across recorded tool calls.

        O(1) read for the eval dashboard: ``1`` with several calls means a
        single-tool agent, whose activity cell drops the constant ``last:``
        suffix in favour of the token count.
        """
        return len(self._seen_tool_titles)

    def _maybe_log_progress(self) -> None:
        if not self._progress_enabled or self._prompt_started_at is None:
            return
        now = time.monotonic()
        if now - self._last_progress_at < _PROGRESS_INTERVAL_SEC:
            return
        self._last_progress_at = now
        elapsed_min = (now - self._prompt_started_at) / 60.0
        calls, title = self.progress_snapshot()
        line = f"  … {elapsed_min:.1f}min, {calls} tool calls"
        if title:
            line += f" (last: {title[:60]})"
        logger.info(line)

    def record_user_prompt(self, text: str) -> None:
        """Record a user prompt. Call before sending each ACP prompt."""
        self._events_active = True
        self._prompt_started_at = time.monotonic()
        # Grace period: the first heartbeat waits a full interval so short
        # prompts stay single-line.
        self._last_progress_at = time.monotonic()
        self._flush_agent_text()
        self.events.append({"type": "user_message", "text": text})
        self._notify_change()

    def mark_prompt_end(self) -> None:
        """Flush pending agent text after a prompt completes."""
        self._prompt_started_at = None
        self._flush_agent_text()
        self._notify_change()

    def pending_tool_call_ids(self) -> list[str]:
        """Return tool calls that have not reached a terminal status."""
        pending_statuses = {ToolCallStatus.PENDING, ToolCallStatus.IN_PROGRESS}
        return [
            record.tool_call_id
            for record in self.tool_calls
            if record.status in pending_statuses
        ]

    def pending_tool_call_state(self) -> tuple[tuple[str, int], ...]:
        """Return stable pending-call identities with progress versions."""
        pending_statuses = {ToolCallStatus.PENDING, ToolCallStatus.IN_PROGRESS}
        return tuple(
            sorted(
                (
                    record.tool_call_id,
                    self._pending_tool_call_update_counts.get(record.tool_call_id, 0),
                )
                for record in self.tool_calls
                if record.status in pending_statuses
            )
        )

    def record_agent_timeout(
        self,
        *,
        timeout_sec: float,
        pending_tool_call_ids: list[str],
        terminal_trajectory_complete: bool,
    ) -> None:
        """Append BenchFlow's terminal timeout marker to the ACP event stream."""
        self._events_active = True
        self._flush_agent_text()
        self.events.append(
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": timeout_sec,
                "pending_tool_call_ids": list(pending_tool_call_ids),
                "terminal_trajectory_complete": terminal_trajectory_complete,
            }
        )
        self._notify_change()

    def record_prompt_usage(self, usage: object | None) -> None:
        """Record cumulative ACP token usage returned by session/prompt."""
        snapshot = normalize_acp_usage(usage)
        if snapshot is None:
            return
        self.usage_snapshots.append(snapshot)
        self._notify_change()

    def latest_usage_totals(self) -> ACPUsageSnapshot | None:
        """Return the latest cumulative ACP usage snapshot, if any."""
        if not self.usage_snapshots:
            return None
        return dict(self.usage_snapshots[-1])

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

    def _record_tool_call(self, record: ToolCallRecord) -> None:
        """Register a newly created tool-call record in every live structure.

        Single bookkeeping site for the two creation paths in
        :meth:`handle_update` (``tool_call``, and the ``tool_call_update``
        fallback for unseen ids), so the distinct-title tracker can never
        drift from the record list. ``title`` itself is never rewritten after
        creation, but an empty-title record's display title follows ``kind``,
        which the legacy-skill upgrade can rewrite — that site re-registers
        the new display title (over-counting fails safe).
        """
        self.tool_calls.append(record)
        self._tool_call_map[record.tool_call_id] = record
        self._seen_tool_titles.add(_tool_display_title(record.title, record.kind))
        self.events.append({"type": "tool_call", "record": record})

    _RECOGNIZED_UPDATE_TYPES = frozenset(
        {
            "tool_call",
            "tool_call_update",
            "agent_message_chunk",
            "text_update",
            "agent_thought",
            "agent_thought_chunk",
        }
    )

    def handle_update(self, update: dict) -> None:
        """Process a session/update notification."""
        self._events_active = True
        update_type = update.get("sessionUpdate")
        # Unknown update types (future ACP versions, agent-specific
        # extensions) mutate no state and must not trigger a no-op
        # snapshot. Mark events_active so the snapshot path stays on
        # the modern branch, but skip _notify_change for unrecognized
        # types.
        if update_type not in self._RECOGNIZED_UPDATE_TYPES:
            return

        if update_type == "tool_call":
            self._flush_agent_text()
            record = ToolCallRecord(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=_canonical_tool_kind(
                    update.get("kind", "other"), update.get("title", "")
                ),
            )
            # The opening notification may already carry content (codex-acp
            # sends file-change diffs this way) and raw I/O.
            initial_content = update.get("content")
            if isinstance(initial_content, list) and initial_content:
                record.content.extend(initial_content)
            record.absorb_raw_io(update)
            # An opening call may already be terminal (codex-acp file edits
            # arrive completed with no later update); honor its status so it
            # never lingers in pending_tool_call_ids().
            if update.get("status") is not None:
                try:
                    record.update_status(ToolCallStatus(update["status"]))
                except ValueError:
                    logger.warning(f"Unknown tool call status: {update.get('status')}")
            self._record_tool_call(record)

        elif update_type == "tool_call_update":
            self.tool_call_update_count += 1
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
                self._record_tool_call(record)
            try:
                status = ToolCallStatus(update.get("status", "in_progress"))
            except ValueError:
                logger.warning(f"Unknown tool call status: {update.get('status')}")
                status = ToolCallStatus.IN_PROGRESS
            content = update.get("content")
            record.update_status(status, content)
            record.absorb_raw_io(update)
            if status in (ToolCallStatus.PENDING, ToolCallStatus.IN_PROGRESS):
                self._pending_tool_call_update_counts[tc_id] = (
                    self._pending_tool_call_update_counts.get(tc_id, 0) + 1
                )
            # Canonicalize legacy OpenHands invoke_skill calls using the same
            # classifier the rescan path uses. Only upgrade to "skill"; never
            # downgrade, and never reclassify a tool that already has a real
            # ACP kind (its output may merely quote a skill envelope).
            if record.kind != "skill" and _is_skill_tool_call(
                record.kind, record.title, record.content
            ):
                record.kind = "skill"
                # An empty-title record's DISPLAY title falls back to kind, so
                # this upgrade can change what renders. Re-register the new
                # display title: the set may now over-count (creation-time
                # fallback + "skill"), which fails safe — the cell keeps the
                # last: suffix instead of falsely claiming a single tool.
                self._seen_tool_titles.add(
                    _tool_display_title(record.title, record.kind)
                )

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

        self._notify_change()

    @property
    def full_message(self) -> str:
        """Concatenated agent message text from all received chunks."""
        return "".join(self.message_chunks)

    @property
    def full_thought(self) -> str:
        """Concatenated agent thought/reasoning text from all received chunks."""
        return "".join(self.thought_chunks)

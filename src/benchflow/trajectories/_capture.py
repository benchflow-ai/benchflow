"""Trajectory capture and parsing utilities."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchflow.acp.session import ACPSession
from benchflow.trajectories.types import redact_acp_trajectory_jsonl

logger = logging.getLogger(__name__)


def make_trajectory_sink(
    writer: "TrajectoryWriter",
    prior_trajectory: list[dict],
) -> Callable[[ACPSession], None]:
    """Build an ``on_change`` sink that writes ``prior + current session`` to disk.

    ``prior_trajectory`` is captured by value (a shallow copy taken at
    wire-up time), so subsequent mutations by the caller (e.g. a Rollout
    extending its cumulative ``_trajectory`` after ``execute_prompts``
    returns) cannot cause the current session's events to be
    double-counted on disk.

    Used across multi-scene rollouts so each scene's streaming writer
    sees prior scenes' events instead of overwriting them.
    """
    prior_snapshot = list(prior_trajectory)

    def sink(session: ACPSession) -> None:
        writer.write_events(prior_snapshot + _snapshot_session_trajectory(session))

    return sink


def _merge_pending_text(pending: list[dict]) -> list[dict]:
    """Merge consecutive same-type pending text events without mutating `pending`.

    Mirrors the merging done by ``ACPSession._flush_agent_text`` but is
    side-effect-free so a live snapshot can include in-flight chunks while
    leaving the session free to keep streaming.
    """
    if not pending:
        return []
    merged: list[dict] = []
    current = dict(pending[0])
    for event in pending[1:]:
        if event["type"] == current["type"]:
            current["text"] += event["text"]
        else:
            merged.append(current)
            current = dict(event)
    merged.append(current)
    return merged


def _events_to_trajectory(events: list[dict]) -> list[dict]:
    """Convert ``ACPSession.events`` records into the JSONL event format.

    Single canonical conversion used by both the destructive
    end-of-run :func:`_capture_session_trajectory` and the non-destructive
    live :func:`_snapshot_session_trajectory`, so streaming-format =
    final-format is a structural invariant rather than a copy/paste
    discipline (PR #566 review finding #3).
    """
    out: list[dict] = []
    for event in events:
        if event["type"] == "tool_call":
            tc = event["record"]
            out.append(
                {
                    "type": "tool_call",
                    "tool_call_id": tc.tool_call_id,
                    "kind": tc.kind,
                    "title": tc.title,
                    "status": tc.status.value,
                    "content": tc.content,
                }
            )
        elif event["type"] in ("user_message", "agent_message", "agent_thought"):
            out.append({"type": event["type"], "text": event["text"]})
        elif event["type"] == "agent_timeout":
            out.append(
                {
                    "type": "agent_timeout",
                    "reason": event["reason"],
                    "timeout_sec": event["timeout_sec"],
                    "pending_tool_call_ids": event["pending_tool_call_ids"],
                    "terminal_trajectory_complete": event[
                        "terminal_trajectory_complete"
                    ],
                }
            )
    return out


def _snapshot_session_trajectory(session: ACPSession | None) -> list[dict]:
    """Non-destructive trajectory snapshot — safe to call mid-prompt.

    Equivalent to ``_capture_session_trajectory`` except it does not call
    ``session._flush_agent_text()``: pending streamed chunks are merged
    into the returned snapshot but remain in ``session._pending_text``
    until the prompt completes. Use this from the live ``on_change``
    sink; use ``_capture_session_trajectory`` for the end-of-run capture.
    """
    if session is None:
        return []
    if not session._events_active:
        # Legacy path — no event log, fall back to flat capture which has
        # no pending-text bookkeeping anyway.
        return _capture_session_trajectory(session)
    return _events_to_trajectory(session.events) + _merge_pending_text(
        session._pending_text
    )


class TrajectoryWriter:
    """Streams ACP trajectory snapshots to ``acp_trajectory.jsonl`` on demand.

    Wire as ``session.on_change`` so each ACP update flushes the current
    trajectory to disk. Each flush rewrites the file atomically (tmp +
    ``os.replace``) so a concurrent reader (a ``cat``, a follower script)
    never sees a partial line. The on-disk format matches what
    ``_capture_session_trajectory`` produces at end-of-run, so the viewer
    and downstream consumers do not need to change.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        # Sweep any stale .tmp left behind by a previous crashed writer
        # so a follow-up reader can't pick up an orphaned partial file.
        self._tmp.unlink(missing_ok=True)
        self._last_payload: str | None = None

    def __call__(self, session: ACPSession) -> None:
        self.flush(session)

    def flush(self, session: ACPSession) -> None:
        """Re-snapshot ``session`` and rewrite the file if anything changed.

        Single-session use — ignores any prior scenes' events. For
        cumulative multi-scene streaming, use ``make_trajectory_sink``.
        """
        self.write_events(_snapshot_session_trajectory(session))

    def write_events(self, events: list[dict]) -> None:
        """Atomically write a fully-formed event list, deduped.

        Skips the disk write if the serialized payload is byte-identical
        to the previous one — keeps a no-op chunk (an unchanged
        tool_call status poll) from churning the filesystem.
        """
        payload = redact_acp_trajectory_jsonl(events)
        if payload == self._last_payload:
            return
        self._tmp.write_text(payload)
        os.replace(self._tmp, self.path)
        self._last_payload = payload

    def write_final(self, trajectory: list[dict]) -> None:
        """Overwrite the file with a fully-formed trajectory list, no dedup.

        Used by the end-of-run code path (oracle mode, scraped fallback,
        and the final batch write) so the canonical final state always
        lands on disk even if the live streaming writer had already
        written the same content.
        """
        payload = redact_acp_trajectory_jsonl(trajectory)
        self._tmp.write_text(payload)
        os.replace(self._tmp, self.path)
        self._last_payload = payload


def _capture_session_trajectory(session: ACPSession | None) -> list[dict]:
    """Extract trajectory data from an ACP session.

    Produces a chronologically ordered list of events: user_message,
    tool_call, agent_message, and agent_thought — interleaved in the
    order they actually occurred during the session.

    Safe to call even if the session is None or in a partial state (e.g. after timeout).
    """
    if session is None:
        return []

    if session._events_active:
        # Flush any trailing agent text that hasn't been recorded yet.
        session._flush_agent_text()
        return _events_to_trajectory(session.events)

    # Legacy fallback: session has no event log (e.g. older agent shims
    # that manipulate session.tool_calls directly without going through
    # handle_update). Preserves the old flat behaviour.
    trajectory = []
    for tc in session.tool_calls:
        trajectory.append(
            {
                "type": "tool_call",
                "tool_call_id": tc.tool_call_id,
                "kind": tc.kind,
                "title": tc.title,
                "status": tc.status.value,
                "content": tc.content,
            }
        )
    if session.full_message:
        trajectory.append({"type": "agent_message", "text": session.full_message})
    if session.full_thought:
        trajectory.append({"type": "agent_thought", "text": session.full_thought})
    return trajectory


async def _scrape_agent_trajectory(
    env: Any, agent: str, sandbox_user: str | None
) -> list[dict]:
    """Fallback: read agent-native trajectory files from the container."""
    home = f"/home/{sandbox_user}" if sandbox_user else "/root"

    # Gemini CLI: writes ~/.gemini/sessions/*/gemini-cli.trajectory.json
    if "gemini" in agent:
        result = await env.exec(
            f"cat $(find {home}/.gemini -name 'gemini-cli.trajectory.json' 2>/dev/null | head -1) 2>/dev/null",
            timeout_sec=10,
        )
        if result.return_code == 0 and result.stdout and result.stdout.strip():
            try:
                return _parse_gemini_trajectory(json.loads(result.stdout))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse gemini trajectory: {e}")

    return []


def _parse_gemini_trajectory(data: dict) -> list[dict]:
    """Convert gemini-cli.trajectory.json → ACP trajectory event format."""
    events = []
    for msg in data.get("messages", []):
        if msg.get("type") == "user":
            continue
        for tc in msg.get("toolCalls", []):
            events.append(
                {
                    "type": "tool_call",
                    "tool_call_id": tc.get("id", ""),
                    "kind": tc.get("name", ""),
                    "title": tc.get("args", {}).get("command", tc.get("name", "")),
                    "status": "completed"
                    if tc.get("status") == "success"
                    else "failed",
                    "content": tc.get("result", []),
                }
            )
        content = msg.get("content", "")
        if content:
            events.append({"type": "agent_message", "text": content})
        for thought in msg.get("thoughts", []):
            if thought:
                events.append({"type": "agent_thought", "text": thought})
    return events

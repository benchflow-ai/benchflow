"""Trajectory capture and parsing utilities."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchflow.acp.session import ACPSession
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
)

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


def _redact_session_trajectory(session: object, trajectory: list[dict]) -> list[dict]:
    """Use real session redactors without accepting permissive mock attributes."""
    redactor = getattr(type(session), "redact_for_persistence", None)
    if not callable(redactor):
        return redact_trajectory_obj(trajectory)
    redacted = redactor(session, trajectory)
    return redacted if isinstance(redacted, list) else []


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
    # Steps-only session-factory Session (benchflow.agents.protocol.Session):
    # exposes .steps + .on_change but NOT the ACP streaming bookkeeping
    # (_events_active / _pending_text live only on ACPSession). Duck-type off
    # .steps so the live on_change sink works for both planes; the ACP paths
    # below stay byte-identical when those attrs ARE present (#825 BLOCKER 8).
    if not hasattr(session, "_events_active") or not hasattr(session, "_pending_text"):
        return list(getattr(session, "steps", []))
    if not session._events_active:
        # Legacy path — no event log, fall back to flat capture which has
        # no pending-text bookkeeping anyway.
        return _capture_session_trajectory(session)
    trajectory = _events_to_trajectory(session.events) + _merge_pending_text(
        session._pending_text
    )
    return _redact_session_trajectory(session, trajectory)


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
        return _redact_session_trajectory(
            session, _events_to_trajectory(session.events)
        )
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
    return _redact_session_trajectory(session, trajectory)


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


def _reconcile_tool_evidence(
    trajectory: list[dict], provider_evidence: list[dict]
) -> tuple[list[dict], int]:
    """Repair lossy ACP tool events from exact-ID provider evidence.

    ACP remains the canonical ordering/status source. A non-empty provider
    result fills only an empty ACP observation, and a provider command title
    replaces only a generic ACP title equal to the tool name. The normal
    trajectory writer applies credential redaction before merged events reach
    disk.

    Returns the merged trajectory and the number of repaired observations.
    """

    evidence_by_id = {
        event.get("tool_call_id"): event
        for event in provider_evidence
        if event.get("type") == "tool_call" and event.get("tool_call_id")
    }
    if not evidence_by_id:
        return trajectory, 0

    merged: list[dict] = []
    repaired = 0
    for event in trajectory:
        tool_call_id = event.get("tool_call_id")
        evidence = evidence_by_id.get(tool_call_id)
        if event.get("type") == "tool_call" and evidence:
            updated = dict(event)
            if not event.get("content") and evidence.get("content"):
                updated["content"] = evidence["content"]
            provider_title = evidence.get("title")
            provider_tool = evidence.get("provider_tool")
            if (
                isinstance(provider_title, str)
                and provider_title
                and event.get("title") in {None, "", provider_tool}
            ):
                updated["title"] = provider_title
            if updated != event:
                event = updated
                repaired += 1
        merged.append(event)
    return merged, repaired


def _provider_tool_title(name: str, arguments: Any) -> str | None:
    """Render the most useful bounded title from captured tool arguments."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments[:4096] or None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command")
    if isinstance(command, str) and command:
        return command[:4096]
    rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return f"{name} {rendered}"[:4096] if rendered != "{}" else None


def _parse_provider_tool_evidence(exchanges: list[Any]) -> list[dict]:
    """Extract exact-ID tool commands/results from trusted provider requests.

    OpenAI-style messages carry ordinary ``tool_calls`` and ``tool`` results.
    Gemini CLI additionally embeds its native ``contents`` conversation as
    JSON inside one of those messages. Later requests repeat the growing
    conversation, so evidence is deduplicated by exact tool-call ID. This
    source is captured by BenchFlow's host/sandbox-local proxy, unlike the
    agent-writable native trajectory fallback.
    """

    evidence: dict[str, dict] = {}
    order: list[str] = []

    def record_call(tool_call_id: Any, name: Any, arguments: Any) -> None:
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return
        if tool_call_id not in evidence:
            order.append(tool_call_id)
            evidence[tool_call_id] = {
                "type": "tool_call",
                "tool_call_id": tool_call_id,
            }
        if isinstance(name, str) and name:
            evidence[tool_call_id]["provider_tool"] = name
            title = _provider_tool_title(name, arguments)
            if title:
                evidence[tool_call_id]["title"] = title

    def record_result(tool_call_id: Any, output: Any) -> None:
        if not isinstance(tool_call_id, str) or not tool_call_id or output is None:
            return
        if tool_call_id not in evidence:
            order.append(tool_call_id)
            evidence[tool_call_id] = {
                "type": "tool_call",
                "tool_call_id": tool_call_id,
            }
        text = output if isinstance(output, str) else json.dumps(output)
        evidence[tool_call_id]["content"] = [
            {
                "type": "content",
                "content": {"type": "text", "text": text},
            }
        ]

    for exchange in exchanges:
        request = getattr(exchange, "request", None)
        body = getattr(request, "body", None)
        if not isinstance(body, dict):
            continue
        messages = body.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    function = function if isinstance(function, dict) else {}
                    record_call(
                        tool_call.get("id") or tool_call.get("tool_call_id"),
                        function.get("name") or tool_call.get("name"),
                        function.get("arguments", tool_call.get("arguments")),
                    )
            if message.get("role") == "tool":
                record_result(
                    message.get("tool_call_id") or message.get("id"),
                    message.get("content"),
                )
            raw_content = message.get("content")
            if not isinstance(raw_content, str):
                continue
            try:
                nested = json.loads(raw_content.lstrip())
            except json.JSONDecodeError:
                continue
            contents = nested.get("contents") if isinstance(nested, dict) else None
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    call = part.get("functionCall") if isinstance(part, dict) else None
                    if isinstance(call, dict):
                        record_call(call.get("id"), call.get("name"), call.get("args"))
                    response = (
                        part.get("functionResponse") if isinstance(part, dict) else None
                    )
                    if not isinstance(response, dict):
                        continue
                    payload = response.get("response")
                    if not isinstance(payload, dict):
                        continue
                    record_result(response.get("id"), payload.get("output"))
    return [evidence[tool_call_id] for tool_call_id in order]

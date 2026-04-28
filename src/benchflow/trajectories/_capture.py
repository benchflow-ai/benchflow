"""Trajectory capture and parsing utilities."""

import json
import logging
from typing import Any

from benchflow.acp.session import ACPSession

logger = logging.getLogger(__name__)


def _capture_session_trajectory(session: ACPSession | None) -> list[dict]:
    """Extract trajectory data from an ACP session.

    Safe to call even if the session is None or in a partial state (e.g. after timeout).
    """
    if session is None:
        return []
    trajectory: list[dict] = []
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
        trajectory.append(
            {
                "type": "agent_message",
                "text": session.full_message,
            }
        )
    if session.full_thought:
        trajectory.append(
            {
                "type": "agent_thought",
                "text": session.full_thought,
            }
        )
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

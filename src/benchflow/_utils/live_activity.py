"""Process-global registry of live rollouts for the eval dashboard.

The Rich dashboard (``cli/_live_progress.py``) renders in the same process as
the engine's rollouts, so surfacing per-task activity needs no new event
plumbing: the engine registers each live Rollout under its task name and the
dashboard polls the rollout's ACP session counters — the exact ones the
console heartbeat (``ACPSession._maybe_log_progress``) logs — at render time.
Best-effort by contract: every reader failure degrades to "no activity",
never to a render error.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_live: dict[str, Any] = {}  # task name -> live Rollout


def register(task_name: str, rollout: Any) -> None:
    """Expose a live rollout to the dashboard under its task name."""
    with _lock:
        _live[task_name] = rollout


def unregister(task_name: str) -> None:
    with _lock:
        _live.pop(task_name, None)


def activity(task_name: str) -> tuple[int, str, int | None] | None:
    """(tool calls, last tool title, total tokens) for a running task.

    None until the task's agent session exists — including non-ACP
    (session-factory) agents, which never grow an ACP client.
    """
    with _lock:
        rollout = _live.get(task_name)
    try:
        session = getattr(getattr(rollout, "acp_client", None), "session", None)
        if session is None:
            return None
        calls, last_title = session.progress_snapshot()
        usage = session.latest_usage_totals()
        tokens = usage.get("total_tokens") if usage else None
        return calls, last_title, tokens
    except Exception:
        # Dashboard reads race a live rollout; degrading to "no activity" is
        # always preferable to perturbing the render.
        return None

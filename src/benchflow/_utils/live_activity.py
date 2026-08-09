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
from typing import Any, NamedTuple


class SessionCounters(NamedTuple):
    """The live ACP session counters the console heartbeat logs.

    ``total_tokens`` is None until the session has a usage snapshot (i.e.
    after a completed prompt). ``distinct_tools`` counts distinct tool
    display titles seen this session; ``0`` means unknown, and renderers must
    treat unknown like varied (keep the ``last:`` cell), so a producer that
    omits it degrades to the always-``last:`` behaviour instead of mislabeling.
    """

    tool_calls: int
    last_tool: str
    total_tokens: int | None
    distinct_tools: int = 0


class ActivitySnapshot(NamedTuple):
    """One dashboard poll of a live rollout.

    ``phase`` is the rollout's lifecycle phase (``Rollout._phase``) and is
    always present, so the activity cell can label the long non-agent
    stretches — sandbox create, agent install, verify — instead of blanking
    in a way that is indistinguishable from a hang. ``counters`` is None
    until the agent session exists (session-factory agents never grow one).
    """

    phase: str
    counters: SessionCounters | None


_lock = threading.Lock()
# Task name -> live Rollout. Task basenames are unique keys by engine-wide
# invariant: the whole run pipeline (resume filtering, result keying — see
# Evaluation's `d.name not in completed` plan filter) is keyed by basename,
# so each name is scheduled at most once at a time.
_live: dict[str, Any] = {}


def register(task_name: str, rollout: Any) -> None:
    """Expose a live rollout to the dashboard under its task name."""
    with _lock:
        _live[task_name] = rollout


def unregister(task_name: str) -> None:
    with _lock:
        _live.pop(task_name, None)


def activity(task_name: str) -> ActivitySnapshot | None:
    """The :class:`ActivitySnapshot` for a running task, or None when the
    rollout isn't registered (pre-create, between retries, teardown).

    ``counters`` inside the snapshot is None until the task's agent session
    exists — including non-ACP (session-factory) agents, which never grow an
    ACP client — but ``phase`` is always present. The dig into
    client/session state lives on ``Rollout.activity_snapshot()`` (typed,
    owner-side); the except here only guards renders racing rollout teardown.
    """
    with _lock:
        rollout = _live.get(task_name)
    if rollout is None:
        return None
    try:
        return rollout.activity_snapshot()
    except Exception:
        # Dashboard reads race a live rollout; degrading to "no activity" is
        # always preferable to perturbing the render.
        return None

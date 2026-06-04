"""Live SkillsBench experiment-fill status for the dashboard's Experiments panel.

Mirrors ``daytona_status``: ``serve.py`` is the thin HTTP layer and imports
:func:`snapshot` from here. :func:`snapshot` reads the generated ledger JSON
(``experiments_ledger.json`` — produced by the fill orchestrator's
``build_ledger`` step and synced into ``dashboard/``) and returns a plain dict
with an ``error`` field instead of raising, so the panel renders a failure
inline rather than 500.

Ledger schema (one row per matrix cell)::

    {"as_of": ISO8601, "target": int, "rows": [
        {"cell_id", "model", "model_slug", "effort", "skill_mode", "task",
         "trial_slot", "status", "sandbox", "sandbox_id", "run_root", "reward",
         "health", "review_verdict", "task_skills_loading", "tokens": {...},
         "timing_total_s", "hf_path", "note", "started_at", "updated_at"}, ...]}

``status`` is one of :data:`STATES`; every other field is optional and rendered
best-effort by the panel.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

_DASH = Path(__file__).resolve().parent
LEDGER_PATH = _DASH / "experiments_ledger.json"

# Canonical pipeline states, in flow order.
STATES = (
    "queued",
    "running",
    "completed",
    "run_failed",
    "review_fail",
    "quarantined",
    "review_pass",
    "published",
)

# Progressive, non-overlapping rollup into the four dashboard panels.
# "completed" panel = ran-but-not-passed (awaiting review, infra-failed, or failed review).
BUCKETS = {
    "queue": ("queued",),
    "running": ("running",),
    "completed": ("completed", "run_failed", "review_fail", "quarantined"),
    "reviewed": ("review_pass", "published"),
}


def _resolve(ledger_path: str | Path | None) -> Path:
    """Ledger location: explicit arg > ``EXPERIMENTS_LEDGER`` env > default."""
    if ledger_path is not None:
        return Path(ledger_path)
    return Path(os.environ.get("EXPERIMENTS_LEDGER") or LEDGER_PATH)


def snapshot(ledger_path: str | Path | None = None) -> dict:
    """Summarize the experiment-fill ledger for the dashboard.

    Returns ``{as_of, target, summary:{by_state, by_bucket, total}, rows}``. On a
    missing or unreadable ledger returns that same shape plus ``error`` (with
    empty containers) so the panel renders the message instead of 500.
    """
    empty = {
        "as_of": "",
        "target": 0,
        "summary": {
            "by_state": {s: 0 for s in STATES},
            "by_bucket": {b: 0 for b in BUCKETS},
            "total": 0,
        },
        "rows": [],
    }
    path = _resolve(ledger_path)
    if not path.is_file():
        return {**empty, "error": f"no ledger at {path.name} yet — start the fill orchestrator"}
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001 - render the failure inline
        return {**empty, "error": f"ledger unreadable: {e}"}

    rows = data.get("rows") if isinstance(data, dict) else None
    rows = rows if isinstance(rows, list) else []
    by_state = {s: 0 for s in STATES}
    for r in rows:
        st = str((r or {}).get("status") or "queued")
        by_state[st] = by_state.get(st, 0) + 1
    by_bucket = {b: sum(by_state.get(s, 0) for s in states) for b, states in BUCKETS.items()}
    return {
        "as_of": (data.get("as_of") if isinstance(data, dict) else "")
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "target": (data.get("target") if isinstance(data, dict) else 0) or len(rows),
        "summary": {"by_state": by_state, "by_bucket": by_bucket, "total": len(rows)},
        "rows": rows,
    }

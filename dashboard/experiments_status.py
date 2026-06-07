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

The dashboard only marks a row as reviewed after the review health gate passes:
healthy review, skill/no-skill evidence, token usage, timing, and HF path.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DASH = Path(__file__).resolve().parent
LEDGER_PATH = _DASH / "experiments_ledger.json"
EXPECTED_TARGET = 3 * 88 * 2 * 3

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
BUCKETS = {
    "queue": "queued",
    "running": "running",
    "completed": "ran but not yet health-gated",
    "reviewed": "reviewed and health-gated",
}

_REVIEWED_STATES = {"review_pass", "published"}
_PASS_VALUES = {"1", "true", "yes", "ok", "pass", "passed", "healthy", "approved"}
_FAIL_VALUES = {"0", "false", "no", "fail", "failed", "missing", "leaked", "error"}
_TOKEN_KEYS = (
    "total",
    "input",
    "output",
    "n_input_tokens",
    "n_output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _flag_true(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if _truthy(row.get(key)):
            return True
    checklist = row.get("review_checklist") or row.get("review_checks") or {}
    if isinstance(checklist, dict):
        for key in keys:
            if _truthy(checklist.get(key)):
                return True
    return False


def _resolve(ledger_path: str | Path | None) -> Path:
    """Ledger location: explicit arg > ``EXPERIMENTS_LEDGER`` env > default."""
    if ledger_path is not None:
        return Path(ledger_path)
    return Path(os.environ.get("EXPERIMENTS_LEDGER") or LEDGER_PATH)


def _expected_target() -> int:
    raw = os.environ.get("EXPERIMENTS_TARGET")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return EXPECTED_TARGET


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in _PASS_VALUES:
        return True
    if text in _FAIL_VALUES:
        return False
    return bool(text)


def _checklist_passes(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if not value:
            return False
        return all(_checklist_passes(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return False
        return all(_checklist_passes(v) for v in value)
    return _truthy(value)


def _review_passes(row: dict[str, Any]) -> bool:
    checklist = row.get("review_checklist") or row.get("review_checks")
    if checklist is not None:
        return _checklist_passes(checklist)
    if row.get("review_verdict") is not None:
        return _truthy(row.get("review_verdict"))
    return (
        str(row.get("status") or "").strip().lower() == "published"
        and str(row.get("health") or "").strip().lower() == "healthy"
        and bool(row.get("hf_path"))
    )


def _skills_pass(row: dict[str, Any]) -> bool:
    skill_mode = str(row.get("skill_mode") or "").strip().lower()
    loading = row.get("task_skills_loading")
    leaked = row.get("skill_leakage", row.get("skill_files_accessed"))
    if "no_skill_leakage" in row:
        no_leak = _truthy(row.get("no_skill_leakage"))
    else:
        no_leak = not _truthy(leaked)
    if skill_mode in {"with", "with-skills", "skills", "skill"}:
        return _truthy(loading)
    if skill_mode in {
        "without",
        "without-skills",
        "no",
        "none",
        "no-skills",
        "noskills",
    }:
        return not _truthy(loading) and no_leak
    return False


def _has_tokens(row: dict[str, Any]) -> bool:
    def present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "[REDACTED]":
                return False
            try:
                return float(text) > 0
            except ValueError:
                return True
        return bool(value)

    tokens = row.get("tokens")
    if isinstance(tokens, dict):
        return any(present(tokens.get(k)) for k in _TOKEN_KEYS)
    if present(tokens):
        return True
    return any(present(row.get(k)) for k in _TOKEN_KEYS)


def _has_timing(row: dict[str, Any]) -> bool:
    for key in ("timing_total_s", "duration_s"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return bool(value)
    return False


def _has_reward(row: dict[str, Any]) -> bool:
    value = row.get("reward")
    if value is None:
        return False
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= reward <= 1.0


def _is_partial(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    summary = row.get("trajectory_summary")
    return any(
        _truthy(value)
        for value in (
            row.get("partial"),
            row.get("partial_trajectory"),
            row.get("trajectory_summary_partial"),
            metadata.get("partial_trajectory") if isinstance(metadata, dict) else None,
            summary.get("partial_trajectory") if isinstance(summary, dict) else None,
        )
    )


def _accepted_normal_timeout(row: dict[str, Any]) -> bool:
    return _flag_true(
        row,
        "accepted_normal_timeout",
        "timeout_accepted",
        "strict_timeout_accepted",
    ) and _flag_true(row, "timeout_complete_artifacts")


def _error_ok(row: dict[str, Any]) -> bool:
    error = row.get("error")
    if error in (None, "", False):
        return True
    text = str(error).lower()
    return _accepted_normal_timeout(row) and ("timeout" in text or "timed out" in text)


def _sandbox_daytona(row: dict[str, Any]) -> bool:
    return str(row.get("sandbox") or "").strip().lower() == "daytona"


def _opus_max_evidence_ok(row: dict[str, Any]) -> bool:
    model = str(row.get("model") or row.get("model_slug") or "").lower()
    if "opus-4.8" not in model and "claude-opus-4-8" not in model:
        return True
    return _flag_true(row, "effort_ok") or (
        _flag_true(row, "opus_adaptive_thinking")
        and _flag_true(row, "opus_output_config_effort_max")
    )


def _health_notes(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if str(row.get("health") or "").strip().lower() != "healthy":
        notes.append("health is not healthy")
    if _is_partial(row) and not _accepted_normal_timeout(row):
        notes.append("partial trajectory not accepted by strict timeout overlay")
    if not _error_ok(row):
        notes.append("run error is not an accepted normal timeout")
    if not _has_reward(row):
        notes.append("reward missing or invalid")
    if not _review_passes(row):
        notes.append("review checklist/verdict missing or failed")
    if not _sandbox_daytona(row):
        notes.append("sandbox is not Daytona")
    if not _opus_max_evidence_ok(row):
        notes.append("Opus MAX adaptive-thinking evidence missing")
    if not _skills_pass(row):
        notes.append("skill/no-skill detection missing or failed")
    if not _has_tokens(row):
        notes.append("token usage missing")
    if not _has_timing(row):
        notes.append("timing missing")
    if not row.get("hf_path"):
        notes.append("HF path missing")
    return notes


def _dashboard_bucket(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "queued")
    if status == "queued":
        return "queue"
    if status == "running":
        return "running"
    if status in _REVIEWED_STATES and not _health_notes(row):
        return "reviewed"
    return "completed"


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
            "missing": 0,
        },
        "rows": [],
    }
    path = _resolve(ledger_path)
    if not path.is_file():
        return {
            **empty,
            "error": f"no ledger at {path.name} yet — start the fill orchestrator",
        }
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return {**empty, "error": f"ledger unreadable: {e}"}

    rows = data.get("rows") if isinstance(data, dict) else None
    rows = rows if isinstance(rows, list) else []
    by_state = {s: 0 for s in STATES}
    rendered_rows: list[dict[str, Any]] = []
    for raw in rows:
        r = raw if isinstance(raw, dict) else {}
        st = str(r.get("status") or "queued")
        by_state[st] = by_state.get(st, 0) + 1
        notes = _health_notes(r) if st in _REVIEWED_STATES else []
        row = {
            **r,
            "status": st,
            "review_health_ok": st in _REVIEWED_STATES and not notes,
            "review_health_notes": notes,
        }
        row["dashboard_bucket"] = _dashboard_bucket(row)
        rendered_rows.append(row)

    by_bucket = {b: 0 for b in BUCKETS}
    for row in rendered_rows:
        by_bucket[row["dashboard_bucket"]] = (
            by_bucket.get(row["dashboard_bucket"], 0) + 1
        )
    target = (data.get("target") if isinstance(data, dict) else 0) or _expected_target()
    return {
        "as_of": (data.get("as_of") if isinstance(data, dict) else "")
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "target": target,
        "expected_target": _expected_target(),
        "summary": {
            "by_state": by_state,
            "by_bucket": by_bucket,
            "total": len(rendered_rows),
            "missing": max(0, int(target) - len(rendered_rows)),
        },
        "rows": rendered_rows,
    }

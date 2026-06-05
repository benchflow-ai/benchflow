"""Live Daytona sandbox status for the dashboard's Daytona panel.

Mirrors the ``jobs_visibility`` / ``roadmap`` split: ``serve.py`` is the thin
HTTP layer and imports :func:`snapshot` from here. ``snapshot`` returns a plain
dict (with an ``error`` field instead of raising) so the panel can render a
failure inline rather than 500.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime

# Sandbox attribute names vary across daytona SDK versions; probe in order.
_CREATED_FIELDS = ("created_at", "createdAt", "created", "start_time", "started_at")
_STATE_FIELDS = ("state", "status")
_TARGET_FIELDS = ("target", "region", "node")


def _attr(obj: object, names: tuple[str, ...]) -> object:
    """First present, non-empty attribute among ``names`` (also checks ``.info``)."""
    for n in names:
        v = getattr(obj, n, None)
        if v not in (None, ""):
            return v
    info = getattr(obj, "info", None)
    if callable(info):
        with contextlib.suppress(Exception):
            info = info()
    if info is not None:
        for n in names:
            v = getattr(info, n, None)
            if v not in (None, ""):
                return v
    return None


def _parse_dt(v: object) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    s = str(v).strip().replace("Z", "+00:00")
    with contextlib.suppress(Exception):
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def _age(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    secs = int((datetime.now(UTC) - dt).total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _row(sb: object) -> dict:
    created = _parse_dt(_attr(sb, _CREATED_FIELDS))
    return {
        "id": str(_attr(sb, ("id",)) or "?"),
        "state": str(_attr(sb, _STATE_FIELDS) or "?").split(".")[-1],
        "created": created.isoformat() if created else "?",
        "age": _age(created),
        "target": str(_attr(sb, _TARGET_FIELDS) or ""),
    }


def snapshot(api_key: str | None) -> dict:
    """Active Daytona sandboxes: count, state breakdown, and per-sandbox age.

    ``api_key`` is passed explicitly to the SDK (no env mutation). Falls back to
    the ``DAYTONA_API_KEY`` env var the SDK reads on its own when ``api_key`` is
    blank. Returns ``{count, by_state, rows, as_of, error?}``.
    """
    empty = {"count": 0, "by_state": {}, "rows": [], "as_of": ""}
    # Reuse benchflow's canonical sync-client bootstrap (anyio compat + explicit
    # key, no env mutation). Imported lazily so the dashboard stays importable
    # without the sandbox-daytona extra; the panel renders the error instead.
    try:
        from benchflow.sandbox.daytona import build_sync_client
    except Exception as e:
        return {**empty, "error": f"daytona support unavailable: {e}"}
    try:
        # client.list() returns a PaginatedSandboxes; the real Sandbox objects are
        # under .items. Iterating the paginator directly yields its internal fields
        # (which is why count looked like 5 and every row was "?"). Fall back to the
        # raw result for SDK versions that return a plain list.
        paginated = build_sync_client((api_key or "").strip() or None).list()
        items = list(getattr(paginated, "items", None) or paginated)
    except Exception as e:
        return {**empty, "error": f"Daytona list() failed: {e}"}

    rows = sorted((_row(sb) for sb in items), key=lambda r: r["created"])
    by_state: dict[str, int] = {}
    for r in rows:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    return {
        "count": len(rows),
        "by_state": by_state,
        "rows": rows,
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
    }

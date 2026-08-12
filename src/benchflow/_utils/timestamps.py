"""Timestamp formatting helpers for persisted artifacts."""

from __future__ import annotations

from datetime import UTC, datetime


def artifact_timestamp(value: datetime | None) -> str | None:
    """Return a timezone-aware ISO 8601 timestamp for JSON artifacts.

    Runtime paths should pass UTC-aware values at source. Legacy tests and
    artifact repair paths may still pass naive ``datetime`` values; treat those
    as already-UTC for compatibility and emit a compact ``Z`` suffix.
    """
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")

"""Timestamp formatting helpers for persisted artifacts."""

from __future__ import annotations

from datetime import UTC, datetime


def artifact_timestamp(value: datetime | None) -> str | None:
    """Return a timezone-aware ISO 8601 timestamp for JSON artifacts.

    Legacy rollout paths sometimes pass naive ``datetime`` values. Persisting
    them with ``str(datetime)`` produced a space-separated timestamp without an
    offset, so downstream consumers could not compare artifacts reliably. Treat
    naive values as UTC at the artifact boundary and emit a compact ``Z`` suffix.
    """
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")

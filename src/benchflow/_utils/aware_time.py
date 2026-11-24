"""One rule for a trial's clock times: they carry a timezone.

BenchFlow writes them aware. A naive one, as a result.json written before 0.10.0 holds, or a clock some caller still
reads naive, is this machine's local time, which is how BenchFlow wrote them: ``datetime.now()`` stored with
``.astimezone()``."""

from __future__ import annotations

from datetime import datetime


def aware(value: datetime) -> datetime:
    """*value* with its timezone; a naive time is this machine's local time."""
    return value if value.tzinfo is not None else value.astimezone()

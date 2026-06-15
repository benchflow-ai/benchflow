"""Unit tests for the dependency-cooldown re-lock helper (`tools/lock.py`).

These cover the pure date/text transforms; `uv lock` itself is not invoked.
"""

from __future__ import annotations

import datetime

import pytest

from tools.lock import (
    COOLDOWN_DAYS,
    LockError,
    compute_cooldown_cutoff,
    rewrite_exclude_newer,
)


def test_cutoff_trails_today_by_cooldown_window_at_midnight() -> None:
    now = datetime.datetime(2026, 6, 15, 9, 30, tzinfo=datetime.UTC)
    assert compute_cooldown_cutoff(now) == "2026-06-08T00:00:00Z"


def test_cutoff_is_idempotent_within_a_day() -> None:
    morning = datetime.datetime(2026, 6, 15, 0, 1, tzinfo=datetime.UTC)
    night = datetime.datetime(2026, 6, 15, 23, 59, tzinfo=datetime.UTC)
    assert compute_cooldown_cutoff(morning) == compute_cooldown_cutoff(night)


def test_cutoff_normalizes_non_utc_input() -> None:
    # 2026-06-15 02:00 in UTC+9 is still 2026-06-14 17:00 UTC -> minus 7d.
    tz = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime(2026, 6, 15, 2, 0, tzinfo=tz)
    assert compute_cooldown_cutoff(now) == "2026-06-07T00:00:00Z"


def test_cutoff_default_window_is_the_cooldown_constant() -> None:
    now = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=datetime.UTC)
    expected = (now - datetime.timedelta(days=COOLDOWN_DAYS)).date()
    assert compute_cooldown_cutoff(now).startswith(expected.isoformat())


def test_cutoff_rejects_negative_window() -> None:
    now = datetime.datetime(2026, 6, 15, tzinfo=datetime.UTC)
    with pytest.raises(LockError):
        compute_cooldown_cutoff(now, cooldown_days=-1)


def test_rewrite_replaces_only_the_exclude_newer_value() -> None:
    text = (
        "[tool.uv]\n"
        'exclude-newer = "2026-06-01T00:00:00Z"\n'
        "\n"
        "[tool.uv.exclude-newer-package]\n"
        'litellm = "2026-06-14T00:00:00Z"\n'
    )
    out = rewrite_exclude_newer(text, "2026-06-08T00:00:00Z")
    assert 'exclude-newer = "2026-06-08T00:00:00Z"' in out
    # The per-package override table is untouched.
    assert 'litellm = "2026-06-14T00:00:00Z"' in out


def test_rewrite_requires_exactly_one_assignment() -> None:
    with pytest.raises(LockError):
        rewrite_exclude_newer("[tool.uv]\n", "2026-06-08T00:00:00Z")

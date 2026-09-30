"""UsageLimitError: the window and reset of a spent subscription login.

Guards the dx/errors fix for usage limits that surfaced as a generic,
retried ``acp_error`` (hill-climb smoke of 2026-09-30:
``ACP error -32603: Internal error: You've hit your weekly limit · resets Oct 2,
4pm (UTC)``). The texts are Claude Code 2.1.280's (``You've hit your <limit> ·
resets <time>``); the headers are those of a real HTTP 429 the Messages API
returned on 2026-09-30 for an OAuth login whose 7-day window was spent.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta

import pytest

from benchflow.agents.errors import AgentProtocolError, UsageLimitError
from benchflow.agents.usage_limits import (
    is_usage_limit_text,
    parse_limit_text,
    parse_unified_headers,
)
from benchflow.errors import UserError

NOW = datetime(2026, 9, 30, 0, 50, tzinfo=UTC)

# The spent login's answer, minus the request id (see the module docstring).
SPENT_7D_HEADERS = {
    "anthropic-ratelimit-unified-5h-reset": "1790762400",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.0",
    "anthropic-ratelimit-unified-7d-reset": "1791054000",
    "anthropic-ratelimit-unified-7d-status": "rejected",
    "anthropic-ratelimit-unified-7d-surpassed-threshold": "1.0",
    "anthropic-ratelimit-unified-7d-utilization": "1.0",
    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    "anthropic-ratelimit-unified-overage-disabled-reason": "out_of_credits",
    "anthropic-ratelimit-unified-overage-status": "rejected",
    "anthropic-ratelimit-unified-representative-claim": "seven_day",
    "anthropic-ratelimit-unified-reset": "1791054000",
    "anthropic-ratelimit-unified-status": "rejected",
    "retry-after": "309144",
}


def test_the_weekly_limit_text_names_the_window_and_the_reset():
    err = UsageLimitError.from_text(
        "You've hit your weekly limit · resets Oct 2, 4pm (UTC)", now=NOW
    )
    assert err is not None
    assert err.window == "7-day"
    assert err.resets_at == datetime(2026, 10, 2, 16, 0, tzinfo=UTC)
    assert err.detail == "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"
    assert str(err) == (
        "usage limit reached: 7-day window, resets 2026-10-02 16:00 UTC "
        "(You've hit your weekly limit · resets Oct 2, 4pm (UTC))"
    )


def test_the_acp_error_wrapping_is_read_through():
    text = (
        "ACP error -32603: Internal error: You've hit your weekly limit · "
        "resets Oct 2, 4pm (UTC)"
    )
    assert is_usage_limit_text(text)
    info = parse_limit_text(text, now=NOW)
    assert info is not None
    assert info.detail.startswith("You've hit your weekly limit")


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (
            datetime(2026, 9, 30, 10, 0, tzinfo=UTC),
            datetime(2026, 9, 30, 16, 0, tzinfo=UTC),
        ),
        (
            datetime(2026, 9, 30, 17, 0, tzinfo=UTC),
            datetime(2026, 10, 1, 16, 0, tzinfo=UTC),
        ),
    ],
)
def test_a_session_limit_within_a_day_resets_at_the_next_such_time(now, expected):
    err = UsageLimitError.from_text(
        "You've hit your session limit · resets 4pm (UTC)", now=now
    )
    assert err is not None
    assert err.window == "5-hour"
    assert err.resets_at == expected


def test_minutes_and_a_named_time_zone():
    err = UsageLimitError.from_text(
        "You've hit your session limit · resets 4:30pm (America/Los_Angeles) · "
        "progress saved",
        now=NOW,
    )
    assert err is not None
    # 2026-09-29 17:50 in Los Angeles (PDT, UTC-7): 4:30pm is the next day.
    assert err.resets_at == datetime(2026, 9, 30, 23, 30, tzinfo=UTC)


def test_a_model_window_and_an_explicit_year():
    err = UsageLimitError.from_text(
        "You've hit your Opus limit · resets Jan 3, 2027, 9am (UTC)", now=NOW
    )
    assert err is not None
    assert err.window == "7-day Opus"
    assert err.resets_at == datetime(2027, 1, 3, 9, 0, tzinfo=UTC)


def test_a_date_past_this_year_rolls_into_the_next():
    now = datetime(2026, 12, 30, 12, 0, tzinfo=UTC)
    err = UsageLimitError.from_text(
        "You've hit your weekly limit · resets Jan 2, 8am (UTC)", now=now
    )
    assert err is not None
    assert err.resets_at == datetime(2027, 1, 2, 8, 0, tzinfo=UTC)


def test_the_legacy_epoch_form():
    err = UsageLimitError.from_text("Claude AI usage limit reached|1791054000")
    assert err is not None
    assert err.resets_at == datetime(2026, 10, 3, 19, 0, tzinfo=UTC)
    assert err.window is None


def test_a_relative_try_again_tail():
    err = UsageLimitError.from_text(
        "You've hit your usage limit. Upgrade to Pro or try again in 2 days 3 hours "
        "4 minutes.",
        now=NOW,
    )
    assert err is not None
    assert err.window is None
    assert err.resets_at == NOW + timedelta(days=2, hours=3, minutes=4)


def test_a_message_with_no_window_or_reset_says_only_what_it_knows():
    err = UsageLimitError.from_text("You're out of usage credits")
    assert err is not None
    assert err.resets_at is None
    assert str(err) == "usage limit reached (You're out of usage credits)"


@pytest.mark.parametrize(
    "text",
    [
        "ACP error -32603: Internal error: API Error: 429 rate_limit_error",
        "ACP error -32603: Internal error: The connection to Claude was lost.",
        "Agent idle for 600s with no new tool call, message, or thought",
        "",
        None,
    ],
)
def test_other_errors_are_not_usage_limits(text):
    assert UsageLimitError.from_text(text) is None


def test_the_spent_logins_headers():
    err = UsageLimitError.from_headers(SPENT_7D_HEADERS)
    assert err is not None
    assert err.window == "7-day"
    assert err.resets_at == datetime(2026, 10, 3, 19, 0, tzinfo=UTC)
    headroom = parse_unified_headers(SPENT_7D_HEADERS)
    assert headroom is not None
    assert headroom.used == {"5-hour": 0.0, "7-day": 1.0}
    assert headroom.rejected == ("7-day",)


def test_allowed_headers_are_not_a_limit():
    headers = {
        "anthropic-ratelimit-unified-status": "allowed",
        "anthropic-ratelimit-unified-5h-utilization": "0.31",
        "anthropic-ratelimit-unified-7d-utilization": "0.36",
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.40",
    }
    assert UsageLimitError.from_headers(headers) is None
    headroom = parse_unified_headers(headers)
    assert headroom is not None
    assert headroom.used == {"5-hour": 0.31, "7-day": 0.40}
    assert parse_unified_headers({"content-type": "application/json"}) is None


def test_the_login_label_and_the_hierarchy():
    err = UsageLimitError.from_text(
        "You've hit your weekly limit · resets Oct 2, 4pm (UTC)", now=NOW
    )
    assert err is not None
    named = err.with_login("CLAUDE_CODE_OAUTH_TOKEN (environment)")
    assert str(named).startswith(
        "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): "
        "7-day window, resets 2026-10-02 16:00 UTC"
    )
    assert named.to_dict() == {
        "login": "CLAUDE_CODE_OAUTH_TOKEN (environment)",
        "window": "7-day",
        "resets_at": "2026-10-02T16:00:00+00:00",
        "detail": "You've hit your weekly limit · resets Oct 2, 4pm (UTC)",
    }
    # Caught as an agent protocol error by the rollout, and as an expected
    # error by the CLI.
    assert isinstance(named, AgentProtocolError)
    assert isinstance(named, UserError)
    assert named.category == "usage_limit"
    assert "bench eval resume" in (named.hint or "")


def test_it_survives_pickling():
    err = UsageLimitError(
        "You've hit your weekly limit",
        login="L",
        window="7-day",
        resets_at=datetime(2026, 10, 2, 16, tzinfo=UTC),
    )
    copy = pickle.loads(pickle.dumps(err))
    assert (copy.detail, copy.login, copy.window, copy.resets_at) == (
        err.detail,
        err.login,
        err.window,
        err.resets_at,
    )
    assert str(copy) == str(err)

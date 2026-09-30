"""Read a subscription's usage limits: Claude Code's limit text and the API's headers.

Two sources say that a login has run out of its subscription usage and when
it can run again:

* the text Claude Code (and so claude-agent-acp) shows, for example
  ``You've hit your weekly limit · resets Oct 2, 4pm (UTC)``. Claude Code
  2.1.280 builds it as ``You've hit your <limit> · resets <time>``, where
  ``<limit>`` names the window (``session limit`` is the 5-hour window,
  ``weekly limit`` the 7-day one) and ``<time>`` is ``4pm (UTC)`` within a
  day, else ``Oct 2, 4pm (UTC)``, in the process's time zone;
* the ``anthropic-ratelimit-unified-*`` headers of any Messages API answer
  made with a subscription's OAuth token. A spent login gets HTTP 429 with
  ``anthropic-ratelimit-unified-status: rejected``, the window in
  ``anthropic-ratelimit-unified-representative-claim`` (``five_hour``,
  ``seven_day``, ...) and the reset as a Unix time in
  ``anthropic-ratelimit-unified-reset``; the body only says "This request
  would exceed your account's rate limit".

Only the standard library is imported, so a native harness, ``bench doctor``
and the ACP client can all use it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

# The Claude Agent SDK's USAGE_LIMIT_ERROR_PREFIXES (0.3.280): the starts of
# the synthetic messages the CLI shows instead of a model answer when the
# login's usage is used up. claude-agent-acp matches the same list.
USAGE_LIMIT_PREFIXES: tuple[str, ...] = (
    "You've hit your",
    "You've reached your",
    "You're out of usage credits",
    "Your org is out of usage · add funds to continue",
    "Your org is out of usage · contact your admin",
    "Your seat type doesn't include usage credits",
    "Your seat type doesn't include usage",
    "Your usage allocation has been disabled by your admin",
    "Your group's usage limit is set to $0",
    "Fable 5 requires usage credits",
    "You're out of extra usage",
    "Your seat type doesn't include extra usage",
    # Claude Code before 1.0.60 wrote "Claude AI usage limit reached|<epoch>".
    "Claude AI usage limit reached",
)

# Claude Code's name for each window (its `fle` table) -> BenchFlow's name.
_LIMIT_NAME_WINDOWS: dict[str, str] = {
    "session limit": "5-hour",
    "weekly limit": "7-day",
    "opus limit": "7-day Opus",
    "sonnet limit": "7-day Sonnet",
}

# The API's rate-limit types (representative-claim, SDKRateLimitInfo) -> window.
CLAIM_WINDOWS: dict[str, str] = {
    "five_hour": "5-hour",
    "seven_day": "7-day",
    "seven_day_opus": "7-day Opus",
    "seven_day_sonnet": "7-day Sonnet",
}

# Header window keys (anthropic-ratelimit-unified-<key>-*) -> window.
HEADER_WINDOWS: dict[str, str] = {
    "5h": "5-hour",
    "7d": "7-day",
    "7d_oi": "7-day",
    "7d_opus": "7-day Opus",
    "7d_sonnet": "7-day Sonnet",
}

HEADER_PREFIX = "anthropic-ratelimit-unified-"

_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun")
_MONTH_NAMES += ("jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS = {name: number for number, name in enumerate(_MONTH_NAMES, start=1)}
_HIT_YOUR = re.compile(r"You've (?:hit|reached) your (?P<name>[^·.\n]+?)(?:\s*[·.]|$)")
_RESETS = re.compile(r"\bresets (?P<when>[^·\n]+?)\s*(?:·|$)")
# Codex: "Try again at 5:05 PM." or "... at Oct 3rd, 2026 5:05 PM." (sandbox time).
_TRY_AGAIN_AT = re.compile(
    r"try again at (?P<when>.+?(?:[ap]m|\([^)]*\)))\s*\.?\s*$", re.IGNORECASE
)
_CLOCK = re.compile(
    r"^(?:(?P<month>[A-Za-z]{3})[a-z]* (?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*"
    r"(?:(?P<year>\d{4}),?\s*)?)?"
    r"(?:at )?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>[ap]m)"
    r"(?:\s*\((?P<zone>[^)]+)\))?$",
    re.IGNORECASE,
)
_LEGACY = re.compile(r"Claude AI usage limit reached\|(?P<epoch>\d{9,11})")
_TRY_AGAIN_IN = re.compile(r"try again in (?P<span>[^.\n]+)", re.IGNORECASE)
_SPAN_PART = re.compile(
    r"(?P<n>\d+)\s*(?P<unit>day|hour|hr|minute|min|second|sec)s?", re.IGNORECASE
)
_SPAN_SECONDS = {
    "day": 86400,
    "hour": 3600,
    "hr": 3600,
    "minute": 60,
    "min": 60,
    "second": 1,
    "sec": 1,
}
_WRAPPERS = re.compile(r"^(?:ACP error -?\d+:\s*)?(?:Internal error:\s*)?", re.I)


@dataclass(frozen=True)
class LimitInfo:
    """What a usage-limit message or answer says: the window and the reset."""

    detail: str
    window: str | None = None
    resets_at: datetime | None = None


def strip_wrappers(text: str) -> str:
    """The agent's own words, without ``ACP error -32603: Internal error:``.

    Codex writes a typographic apostrophe (``You’ve``); it is made plain.
    """
    return _WRAPPERS.sub("", text.strip().replace("\u2019", "'"), count=1).strip()


def is_usage_limit_text(text: str | None) -> bool:
    """Whether ``text`` (optionally wrapped in an ACP error) is a usage-limit message."""
    if not text:
        return False
    words = strip_wrappers(text)
    return any(words.startswith(prefix) for prefix in USAGE_LIMIT_PREFIXES)


def _zone(name: str | None) -> tzinfo | None:
    if not name or name.strip().upper() in ("UTC", "GMT", "Z"):
        return UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name.strip())
    except Exception:  # an unknown or malformed zone: the time stays unknown
        return None


def parse_reset_time(when: str, *, now: datetime | None = None) -> datetime | None:
    """``4pm (UTC)`` or ``Oct 2, 4pm (UTC)`` as an aware UTC time, the next one after ``now``."""
    match = _CLOCK.match(when.strip())
    if match is None:
        return None
    zone = _zone(match["zone"])
    if zone is None:
        return None
    now = (now or datetime.now(UTC)).astimezone(zone)
    hour = int(match["hour"]) % 12 + (12 if match["ampm"].lower() == "pm" else 0)
    minute = int(match["minute"] or 0)
    if match["month"]:
        month = _MONTHS.get(match["month"][:3].lower())
        if month is None:
            return None
        year = int(match["year"]) if match["year"] else now.year
        try:
            local = now.replace(
                year=year,
                month=month,
                day=int(match["day"]),
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        if not match["year"] and local < now - timedelta(days=1):
            local = local.replace(year=year + 1)
    else:
        local = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if local <= now:
            local += timedelta(days=1)
    return local.astimezone(UTC)


def parse_limit_text(
    text: str | None, *, now: datetime | None = None
) -> LimitInfo | None:
    """The window and reset a usage-limit message names, or None when it is not one.

    Understands Claude Code's ``You've hit your <limit> · resets <time>``, its
    older ``Claude AI usage limit reached|<epoch>``, Codex's ``You’ve hit your
    usage limit. ... Try again at Oct 3rd, 2026 5:05 PM.`` (a time with no
    zone is the sandbox's, UTC) and a ``try again in 2 days 3 hours`` tail.
    """
    if not is_usage_limit_text(text):
        return None
    assert text is not None
    detail = strip_wrappers(text)
    window = None
    resets_at = None
    if hit := _HIT_YOUR.search(detail):
        window = _LIMIT_NAME_WINDOWS.get(hit["name"].strip().lower())
    if legacy := _LEGACY.search(detail):
        resets_at = datetime.fromtimestamp(int(legacy["epoch"]), UTC)
        detail = "Claude AI usage limit reached"
    elif resets := _RESETS.search(detail) or _TRY_AGAIN_AT.search(detail):
        resets_at = parse_reset_time(resets["when"], now=now)
    elif again := _TRY_AGAIN_IN.search(detail):
        seconds = sum(
            int(part["n"]) * _SPAN_SECONDS[part["unit"].lower()]
            for part in _SPAN_PART.finditer(again["span"])
        )
        if seconds:
            resets_at = (now or datetime.now(UTC)) + timedelta(seconds=seconds)
    return LimitInfo(detail=detail, window=window, resets_at=resets_at)


def _number(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _epoch(value: Any) -> datetime | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class Headroom:
    """One login's windows as the unified rate-limit headers report them.

    ``used`` maps a window (``5-hour``, ``7-day``) to the share already used
    (0.0 to 1.0, sometimes a little over); ``resets`` to when it resets.
    ``rejected`` lists the windows that refused the request. ``window`` is
    the window the API names as binding (``representative-claim``).
    """

    status: str | None
    used: dict[str, float]
    resets: dict[str, datetime]
    rejected: tuple[str, ...]
    window: str | None
    resets_at: datetime | None

    @property
    def limited(self) -> bool:
        return self.status == "rejected"


def parse_unified_headers(headers: Mapping[str, Any] | None) -> Headroom | None:
    """The ``anthropic-ratelimit-unified-*`` headers of a Messages API answer, or None.

    ``7d_oi`` (the window the API enforces for some plans) wins over ``7d``
    when both are present.
    """
    lowered = {str(k).lower(): v for k, v in dict(headers or {}).items()}
    if not any(key.startswith(HEADER_PREFIX) for key in lowered):
        return None
    used: dict[str, float] = {}
    resets: dict[str, datetime] = {}
    rejected: list[str] = []
    for key in ("5h", "7d", "7d_oi", "7d_opus", "7d_sonnet"):
        name = HEADER_WINDOWS[key]
        util = _number(lowered.get(f"{HEADER_PREFIX}{key}-utilization"))
        reset = _epoch(lowered.get(f"{HEADER_PREFIX}{key}-reset"))
        if util is not None and (key == "7d_oi" or name not in used):
            used[name] = util
        if reset is not None and (key == "7d_oi" or name not in resets):
            resets[name] = reset
        rejected_here = lowered.get(f"{HEADER_PREFIX}{key}-status") == "rejected"
        if rejected_here and name not in rejected:
            rejected.append(name)
    claim = lowered.get(f"{HEADER_PREFIX}representative-claim")
    window = CLAIM_WINDOWS.get(str(claim)) if claim else None
    if window is None and rejected:
        window = rejected[0]
    resets_at = _epoch(lowered.get(f"{HEADER_PREFIX}reset"))
    if resets_at is None and window is not None:
        resets_at = resets.get(window)
    status = lowered.get(f"{HEADER_PREFIX}status")
    return Headroom(
        status=str(status) if status is not None else None,
        used=used,
        resets=resets,
        rejected=tuple(rejected),
        window=window,
        resets_at=resets_at,
    )


def parse_rate_limit_info(info: Any, *, detail: str) -> LimitInfo | None:
    """A rejected rate-limit record, or None.

    The record is Claude Code's ``rate_limit_event`` (the Agent SDK's
    ``SDKRateLimitInfo``): ``{"status": "rejected", "rateLimitType":
    "seven_day", "resetsAt": <unix time>}``. Only ``rejected`` is a limit;
    ``detail`` is the agent's own words for it.
    """
    if not isinstance(info, Mapping) or info.get("status") != "rejected":
        return None
    kind = info.get("rateLimitType")
    return LimitInfo(
        detail=detail,
        window=CLAIM_WINDOWS.get(str(kind)) if kind else None,
        resets_at=_epoch(info.get("resetsAt")),
    )


def format_reset(when: datetime | None) -> str:
    """``2026-10-02 16:00 UTC``, or ``an unknown time``."""
    if when is None:
        return "an unknown time"
    return when.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

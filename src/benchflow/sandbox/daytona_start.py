"""What each call on the Daytona start path may refuse, and how long it retries.

One rule for the run preflight, ``bench doctor``, the sandbox create and the
start-up probe (results/DAYTONA-START.md in the PY8 round):

- A **configuration** error refuses at once and is never retried: the API
  rejected the key. That is a 401 (read by status before any text), or a 403
  without a quota message. It fails the same way every time.
- A **transient** error is retried with jittered backoff within a budget: a
  transport timeout or connection error, a 5xx, a 429, or a quota or capacity
  refusal (Daytona sends those as 400 or 403 with text such as "Total CPU
  limit exceeded"). It never refuses a run before it starts; once a trial's
  budget is spent it stays an infrastructure error the trial-level retry may
  take.
- A sandbox that reached an error state while starting (the SDK's "failed to
  start with state" for ``BUILD_FAILED`` or ``ERROR``) is deleted and replaced
  by a fresh create, up to three creates: nothing in the message tells a
  broken Dockerfile from a failed download or a container-runtime deadline.
  The one exception is a ``BUILD_FAILED`` of a recipe with cached install
  layers: the strategy has its own recovery for that (the task image alone),
  so it gets one build and then that fallback, never a fresh layered build.
- Anything else (a 400 validation error) is the request's own error and is
  not retried.

Capacity patience is a time budget, not an attempt count: a 429 or quota
refusal is asked again until 600 s have passed, waiting at least as long as
the API's Retry-After (seconds or an HTTP date).

The Daytona SDK is not imported here: errors are read by their module, class
name, ``status_code`` and text, so this module loads in a base install.
"""

from __future__ import annotations

import random
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Literal

ErrorKind = Literal[
    "config", "capacity", "transient", "timeout", "start_failed", "other"
]

# The kinds a start-path call retries.
RETRIED_KINDS: frozenset[str] = frozenset(
    {"capacity", "transient", "timeout", "start_failed"}
)

# Create budgets, counted from the first attempt. Capacity: Daytona frees
# quota as other sandboxes end, so it is waited out for up to 10 minutes,
# however many attempts that takes (the Rust bench 0.8.2 waits 60 s).
# Transport and 5xx: six attempts, about a minute of waiting, enough for a
# gateway restart. A create that itself timed out already spent the build
# timeout, so it keeps the old three attempts.
CREATE_CAPACITY_BUDGET_SEC = 600.0
CREATE_TRANSIENT_BUDGET_SEC = 300.0
CREATE_TRANSIENT_MAX_ATTEMPTS = 6
CREATE_TIMEOUT_MAX_ATTEMPTS = 3
# A sandbox that failed while starting is replaced by a fresh one: three
# creates at most, and none started past the capacity budget (each attempt
# already spent a build).
CREATE_START_FAILED_MAX_ATTEMPTS = 3

# Equal jitter: half the exponential step fixed, half random, so trials that
# failed together do not retry together.
BACKOFF_BASE_SEC = 2.0
BACKOFF_CAP_SEC = 30.0
# A read that asks again a few times (the doctor's key check, the toolbox
# start-up probe) has no time budget of its own: a Retry-After longer than this
# ends it instead of holding it, and never makes it ask early.
READ_MAX_WAIT_SEC = 60.0

_CAPACITY_TEXT = re.compile(
    r"limit exceeded|quota|too many requests|rate.?limit|no available runners"
    r"|no capacity|out of capacity|insufficient capacity",
    re.IGNORECASE,
)
_GATEWAY_TEXT = re.compile(
    r"\b50[0234] (?:internal server error|bad gateway"
    r"|service (?:temporarily )?unavailable|gateway time-?out)\b",
    re.IGNORECASE,
)
_BUILD_FAILED_TEXT = re.compile(r"build_failed|failed to build", re.IGNORECASE)
# The SDK's error once a new sandbox reaches an error state while it waits for
# the start (DAYTONA-START round 3, rule E): "Sandbox <id> failed to start with
# state: ..." up to SDK 0.196, "Sandbox <id> entered error state: ..." from
# 0.197.
_START_FAILED_TEXT = re.compile(
    r"\b(?:failed to start with|entered error) state:?\s*"
    r"(?:SandboxState\.)?(?:BUILD_FAILED|ERROR)\b",
    re.IGNORECASE,
)
_START_BUILD_FAILED_TEXT = re.compile(
    r"\b(?:failed to start with|entered error) state:?\s*"
    r"(?:SandboxState\.)?BUILD_FAILED\b",
    re.IGNORECASE,
)
# The SDK's own timeouts. From 0.201 a transport timeout is a
# DaytonaConnectionTimeoutError; from 0.207 an HTTP 408 or 504 is a
# DaytonaTimeoutError too, which keeps the status: Daytona's answer, judged by
# its status below, as a plain DaytonaError(504) was before.
_TIMEOUT_NAMES = frozenset({"DaytonaTimeoutError", "DaytonaConnectionTimeoutError"})
# The SDK re-raises a transport error it has no mapping for (urllib3's
# MaxRetryError and ProtocolError on the sync client, aiohttp's connector
# errors on the async one) as a plain DaytonaError with no status and the
# original text.
_TRANSPORT_TEXT = re.compile(
    r"max retries exceeded|connection (?:aborted|reset|refused|broken)"
    r"|cannot connect to host|server disconnected|remote end closed"
    r"|read timed out|timed out|temporary failure in name resolution",
    re.IGNORECASE,
)
_TRANSPORT_MODULES = frozenset({"urllib3", "aiohttp", "httpx", "httpcore"})


def _is_daytona_error(exc: BaseException) -> bool:
    return type(exc).__module__.startswith("daytona.")


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def error_kind(exc: BaseException | None) -> ErrorKind:
    """Which of the start path's kinds *exc* is (see the module docstring)."""
    if exc is None or not isinstance(exc, Exception):
        # Cancellation and interpreter exit must propagate, never be retried.
        return "other"
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or (
        name in _TIMEOUT_NAMES and _status(exc) is None
    ):
        return "timeout"
    if isinstance(exc, ConnectionError) or name == "DaytonaConnectionError":
        return "transient"
    if type(exc).__module__.split(".")[0] in _TRANSPORT_MODULES:
        return "transient"
    if not _is_daytona_error(exc):
        return "other"
    text = str(exc)
    status = _status(exc)
    # A 401 is the key, whatever its text says; only a 403 can be a quota.
    if status == 401 or (status is None and name == "DaytonaAuthenticationError"):
        return "config"
    if status is None and _START_FAILED_TEXT.search(text):
        return "start_failed"
    if _BUILD_FAILED_TEXT.search(text):
        return "other"
    if status == 429 or name == "DaytonaRateLimitError" or _CAPACITY_TEXT.search(text):
        return "capacity"
    if status == 403 or (status is None and name == "DaytonaAuthorizationError"):
        return "config"
    if (status is not None and status >= 500) or _GATEWAY_TEXT.search(text):
        return "transient"
    if status is None and _TRANSPORT_TEXT.search(text):
        return "transient"
    return "other"


def key_rejected_status(exc: BaseException) -> int | None:
    """401 or 403 when *exc* is the API rejecting the key, else None."""
    if error_kind(exc) != "config":
        return None
    status = _status(exc)
    if status in (401, 403):
        return status
    return 401 if type(exc).__name__ == "DaytonaAuthenticationError" else 403


def retry_after_sec(exc: BaseException | None) -> float | None:
    """The Retry-After the API sent with *exc*, in seconds from now, if any.

    Both forms of the header are read: delay seconds and an HTTP date. A date
    in the past is 0 s. Nothing caps it here: the create budget decides
    whether a wait that long is still worth it.
    """
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        raw = next(
            (v for k, v in dict(headers).items() if str(k).lower() == "retry-after"),
            None,
        )
    except (TypeError, ValueError):
        return None
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None or when.tzinfo is None:
            return None
        value = max(0.0, when.timestamp() - time.time())
    if value != value or value < 0:  # NaN or negative: not a delay
        return None
    return value


def backoff_sec(attempt: int) -> float:
    """The wait after failed attempt *attempt* (1-based), with equal jitter."""
    step = min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * 2 ** max(attempt - 1, 0))
    return step / 2 + random.uniform(0, step / 2)


def wait_sec(exc: BaseException | None, attempt: int) -> float:
    """The wait before the next attempt: the jittered backoff, and never less
    than a capacity refusal's Retry-After."""
    backoff = backoff_sec(attempt)
    after = retry_after_sec(exc) if error_kind(exc) == "capacity" else None
    return backoff if after is None else max(after, backoff)


# tenacity callbacks for ``DaytonaSandbox._create_sandbox``.


def _outcome_error(retry_state: Any) -> BaseException | None:
    outcome = retry_state.outcome
    return outcome.exception() if outcome is not None else None


def create_should_retry(retry_state: Any) -> bool:
    exc = _outcome_error(retry_state)
    kind = error_kind(exc)
    if kind == "start_failed" and _layer_build_falls_back(retry_state):
        # A fresh sandbox retries only what a fresh sandbox can fix: a
        # provider start failure (ERROR), or a BUILD_FAILED of a recipe with
        # no fallback. A layered recipe's BUILD_FAILED goes to the plain
        # task image after one build (daytona_strategies).
        return not _START_BUILD_FAILED_TEXT.search(str(exc))
    return kind in RETRIED_KINDS


def _layer_build_falls_back(retry_state: Any) -> bool:
    """Whether the env being created (``_create_sandbox``'s ``self``) has
    the plain-image fallback for a layered recipe that does not build."""
    args = getattr(retry_state, "args", None) or ()
    return bool(args) and bool(getattr(args[0], "_layer_build_falls_back", False))


def create_retries_exhausted(retry_state: Any) -> bool:
    """Stop when the kind's budget is spent. tenacity has already computed
    the next wait (``upcoming_sleep``), so a wait that would end past the
    budget stops now instead of retrying before the API allows it."""
    kind = error_kind(_outcome_error(retry_state))
    attempt = retry_state.attempt_number
    elapsed = retry_state.seconds_since_start or 0.0
    next_try_at = elapsed + (getattr(retry_state, "upcoming_sleep", 0.0) or 0.0)
    if kind == "capacity":
        return next_try_at > CREATE_CAPACITY_BUDGET_SEC
    if kind == "transient":
        return (
            attempt >= CREATE_TRANSIENT_MAX_ATTEMPTS
            or next_try_at > CREATE_TRANSIENT_BUDGET_SEC
        )
    if kind == "start_failed":
        return (
            attempt >= CREATE_START_FAILED_MAX_ATTEMPTS
            or next_try_at > CREATE_CAPACITY_BUDGET_SEC
        )
    return attempt >= CREATE_TIMEOUT_MAX_ATTEMPTS


def create_wait(retry_state: Any) -> float:
    # Looked up by name at call time so tests can make the waits zero.
    return wait_sec(_outcome_error(retry_state), retry_state.attempt_number)

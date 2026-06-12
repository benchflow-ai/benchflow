"""Pure scoring and classification helpers — no external dependencies."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

# Error category constants
INSTALL_FAILED = "install_failure"
PIPE_CLOSED = "pipe_closed"
ACP_ERROR = "acp_error"
IDLE_TIMEOUT = "idle_timeout"
INFRA_ERROR = "infra_failure"
SANDBOX_SETUP = "sandbox_setup"
PROVIDER_AUTH = "provider_auth"
TIMED_OUT = "timeout"
# Provider API failures detected post-rollout (rate limit, quota, rejected
# request, 5xx). "api_error" is proxy-proven (every captured provider request
# failed); "suspected_api_error" is the zero-signal heuristic (no proxy
# evidence, but the agent ended with zero tokens AND zero tool calls). Both
# null the reward so the slot is excluded from score denominators.
API_ERROR = "api_error"
SUSPECTED_API_ERROR = "suspected_api_error"

# Matched case-insensitively against the error string. Covers the
# human-authored markers plus the sanitized "provider auth failed (HTTP 401)"
# marker injected at the rollout/provider boundary (#546/#564), where the real
# 401/403 is visible only in the proxy trajectory, not the top-level ACP error.
_PROVIDER_AUTH_MARKERS = (
    "permission_denied",
    "leaked",
    "failed to authenticate",
    "invalid bearer token",
    "invalid api key",
    "was rejected as invalid",
    "unauthorized",
    "provider auth failed",
    "http 401",
    "http 403",
)

# Verifier error category constants
VERIFIER_FAILED = "verifier_failure"
VERIFIER_INFRA = "verifier_infra"
VERIFIER_TIMEOUT = "verifier_timeout"
VERIFIER_DEP_INSTALL = "verifier_dep_install"

# Canonical dependency-install markers shared by verifier stdout scanning and
# verifier-error classification. Keep these lower-case; the helper below
# performs case-insensitive matching.
VERIFIER_DEP_INSTALL_MARKERS: tuple[str, ...] = (
    "dependency install failed",
    "failed to download `",
    "failed to fetch:",
    "error sending request for url",
    "failed to lookup address information",
    "no solution found",
    "could not find a version",
    "resolution impossible",
)

ScoreOutcome = Literal["passed", "failed", "errored", "verifier_errored"]
ResultOutcome = Literal["passed", "failed", "errored", "verifier_errored", "unscored"]


def extract_reward(result: Mapping[str, Any]) -> float | None:
    """Extract the reward value from a result dict, or None if absent."""
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    return rewards.get("reward")


def classify_error(error: str | None) -> str | None:
    """Classify an error string into a category, or None if no error."""
    if not error:
        return None
    lower = error.lower()
    if "agent idle for" in lower:
        return IDLE_TIMEOUT
    if "install failed" in error:
        return INSTALL_FAILED
    if "closed stdout" in lower:
        return PIPE_CLOSED
    # Order matters: "suspected provider api error" contains "provider api
    # error", so the heuristic marker must be checked first.
    if "suspected provider api error" in lower:
        return SUSPECTED_API_ERROR
    if "provider api error" in lower:
        return API_ERROR
    if "ACP error" in error or "was rejected as invalid" in error:
        if any(m in lower for m in _PROVIDER_AUTH_MARKERS):
            return PROVIDER_AUTH
        return ACP_ERROR
    if "sandbox startup" in lower or "sandbox creation" in lower:
        return SANDBOX_SETUP
    if "prompt exceeded wall-clock budget" in lower:
        return TIMED_OUT
    if _looks_like_infra_error(lower):
        return INFRA_ERROR
    if "timed out" in lower:
        return TIMED_OUT
    return "other"


def api_error_is_transient(error: str | None) -> bool:
    """True when an api_error string carries the transient marker.

    Provider-api-error strings are formatted by the rollout classifier as
    ``provider api error [<subcategory>/transient] ...`` or ``[.../permanent]``
    — transient (rate limit, 5xx) is retryable, permanent (auth, quota,
    model-not-found, rejected request) is not.
    """
    return bool(error) and "/transient]" in error


def _looks_like_infra_error(error: str) -> bool:
    return any(
        marker in error
        for marker in (
            "connection lost",
            "connection reset",
            "connection refused",
            "broken pipe",
            "sandbox not found",
            "workspace not found",
            "api connection",
            "api timeout",
            "temporarily unavailable",
        )
    )


def classify_verifier_error(verifier_error: str | None) -> str | None:
    """Classify a verifier error string, or None if no error."""
    if not verifier_error:
        return None
    lower = verifier_error.lower()
    if "verifier crashed" in verifier_error:
        if contains_verifier_dep_install_marker(lower):
            return VERIFIER_DEP_INSTALL
        if _looks_like_verifier_infra_error(lower):
            return VERIFIER_INFRA
        return VERIFIER_FAILED
    if "verifier timed out" in verifier_error:
        return VERIFIER_TIMEOUT
    return "verifier_other"


def contains_verifier_dep_install_marker(text: str) -> bool:
    """Detect verifier dependency installation failures (ENG-151)."""
    lower = text.lower()
    return any(marker in lower for marker in VERIFIER_DEP_INSTALL_MARKERS)


def _looks_like_verifier_infra_error(error: str) -> bool:
    return any(
        marker in error
        for marker in (
            "failed to add tests directory",
            "failed to download verifier directory",
            "failed to download llm-judge input",
            "verifier setup failed",
        )
    )


def classify_result(
    *,
    reward: float | None,
    error: str | None,
    verifier_error: str | None,
) -> ScoreOutcome:
    """Classify a single result into exactly one score bucket.

    Returns one of ``"passed"``, ``"failed"``, ``"errored"``,
    ``"verifier_errored"``. The buckets are disjoint and exhaustive, so
    ``passed + failed + errored + verifier_errored == total`` holds
    structurally for any set of results.

    Precedence:

    1. A result with a reward is ``"passed"`` (reward == 1.0) or
       ``"failed"`` (any other reward) — an explicit reward always wins,
       even when an ``error`` string is also present (e.g. a warning).
    2. With no reward, an agent ``error`` makes it ``"errored"``.
    3. With no reward and no agent error, a ``verifier_error`` makes it
       ``"verifier_errored"``. ``errored`` therefore takes precedence over
       ``verifier_errored`` when both errors are present.
    4. With no reward and no error of either kind, the result is
       ``"errored"`` — a terminal result must land in some bucket, and an
       absent reward with no recorded cause is still a failure to produce
       a verdict.
    """
    if reward is not None:
        return "passed" if reward == 1.0 else "failed"
    if error:
        return "errored"
    if verifier_error:
        return "verifier_errored"
    return "errored"


def classify_score_outcome(result: Mapping[str, Any]) -> ScoreOutcome:
    """Classify a persisted result for score/invariant accounting.

    This is the canonical terminal score view. It keeps the four score
    buckets disjoint and gives an explicit reward or agent error precedence
    over verifier evidence so ``EvaluationResult`` cannot double-count tasks.
    """
    return classify_result(
        reward=extract_reward(result),
        error=result.get("error"),
        verifier_error=result.get("verifier_error"),
    )


def classify_result_dict(result: Mapping[str, Any]) -> ScoreOutcome:
    """Backward-compatible alias for score/invariant accounting."""
    return classify_score_outcome(result)


def classify_audit_outcome(result: Mapping[str, Any]) -> ResultOutcome:
    """Classify a result for audit/reporting evidence.

    Audit summaries intentionally surface verifier evidence first. A task with
    ``verifier_error`` is counted as ``verifier_errored`` even if an agent
    error or stale reward is also present, because result auditors need the
    verifier failure to be visible instead of hidden behind the score bucket.
    """
    if result.get("verifier_error"):
        return "verifier_errored"
    reward = extract_reward(result)
    if reward == 1.0:
        return "passed"
    if reward is not None:
        return "failed"
    if result.get("error"):
        return "errored"
    return "unscored"


def classify_result_outcome(result: Mapping[str, Any]) -> ResultOutcome:
    """Backward-compatible alias for audit/reporting outcome accounting."""
    return classify_audit_outcome(result)


def _empty_counts(*, include_unscored: bool) -> dict[str, int]:
    counts = {
        "passed": 0,
        "failed": 0,
        "errored": 0,
        "verifier_errored": 0,
    }
    if include_unscored:
        counts["unscored"] = 0
    return counts


def count_score_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count score buckets with reward/agent-error precedence."""
    counts = _empty_counts(include_unscored=False)
    for result in results:
        counts[classify_score_outcome(result)] += 1
    return counts


def count_audit_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count audit buckets with verifier-evidence precedence."""
    counts = _empty_counts(include_unscored=True)
    for result in results:
        counts[classify_audit_outcome(result)] += 1
    return counts


def count_result_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Backward-compatible alias for audit/reporting outcome accounting."""
    return count_audit_outcomes(results)


def pass_rate(*, passed: int, total: int) -> float:
    """Pass rate over all tasks."""
    return passed / total if total > 0 else 0.0


def pass_rate_excl_errors(*, passed: int, failed: int) -> float:
    """Pass rate excluding errored tasks."""
    completed = passed + failed
    return passed / completed if completed > 0 else 0.0

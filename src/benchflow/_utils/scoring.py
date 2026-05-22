"""Pure scoring and classification helpers — no external dependencies."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

# Error category constants
INSTALL_FAILED = "install_failure"
PIPE_CLOSED = "pipe_closed"
ACP_ERROR = "acp_error"
IDLE_TIMEOUT = "idle_timeout"
INFRA_ERROR = "infra_failure"
TIMED_OUT = "timeout"

# Verifier error category constants
VERIFIER_FAILED = "verifier_failure"
VERIFIER_INFRA = "verifier_infra"
VERIFIER_TIMEOUT = "verifier_timeout"

ResultOutcome = Literal[
    "passed", "failed", "errored", "verifier_errored", "unscored"
]


def extract_reward(result: Mapping[str, Any]) -> float | None:
    """Extract the reward value from a result dict, or None if absent."""
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    return rewards.get("reward")


def classify_result_outcome(result: Mapping[str, Any]) -> ResultOutcome:
    """Classify a result into exactly one reporting bucket.

    Verifier failures are infrastructure failures even if another field also
    exists. Otherwise a verifier-produced reward is authoritative when present.
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


def count_result_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count result outcome buckets using ``classify_result_outcome``."""
    counts = {
        "passed": 0,
        "failed": 0,
        "errored": 0,
        "verifier_errored": 0,
        "unscored": 0,
    }
    for result in results:
        counts[classify_result_outcome(result)] += 1
    return counts


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
    if "ACP error" in error:
        return ACP_ERROR
    if "prompt exceeded wall-clock budget" in lower:
        return TIMED_OUT
    if _looks_like_infra_error(lower):
        return INFRA_ERROR
    if "timed out" in lower:
        return TIMED_OUT
    return "other"


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
        if _looks_like_verifier_infra_error(lower):
            return VERIFIER_INFRA
        return VERIFIER_FAILED
    if "verifier timed out" in verifier_error:
        return VERIFIER_TIMEOUT
    return "verifier_other"


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


def pass_rate(*, passed: int, total: int) -> float:
    """Pass rate over all tasks."""
    return passed / total if total > 0 else 0.0


def pass_rate_excl_errors(*, passed: int, failed: int) -> float:
    """Pass rate excluding errored tasks."""
    completed = passed + failed
    return passed / completed if completed > 0 else 0.0

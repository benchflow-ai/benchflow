"""Pure scoring and classification helpers — no external dependencies."""

# Error category constants
INSTALL_FAILED = "install_failure"
PIPE_CLOSED = "pipe_closed"
ACP_ERROR = "acp_error"
TIMED_OUT = "timeout"

# Verifier error category constants
VERIFIER_FAILED = "verifier_failure"
VERIFIER_TIMEOUT = "verifier_timeout"


def extract_reward(result: dict) -> float | None:
    """Extract the reward value from a result dict, or None if absent."""
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    return rewards.get("reward")


def classify_error(error: str | None) -> str | None:
    """Classify an error string into a category, or None if no error."""
    if not error:
        return None
    if "install failed" in error:
        return INSTALL_FAILED
    if "closed stdout" in error:
        return PIPE_CLOSED
    if "ACP error" in error:
        return ACP_ERROR
    if "timed out" in error:
        return TIMED_OUT
    return "other"


def classify_verifier_error(verifier_error: str | None) -> str | None:
    """Classify a verifier error string, or None if no error."""
    if not verifier_error:
        return None
    if "verifier crashed" in verifier_error:
        return VERIFIER_FAILED
    if "verifier timed out" in verifier_error:
        return VERIFIER_TIMEOUT
    return "verifier_other"


def pass_rate(*, passed: int, total: int) -> float:
    """Pass rate over all tasks."""
    return passed / total if total > 0 else 0.0


def pass_rate_excl_errors(*, passed: int, failed: int) -> float:
    """Pass rate excluding errored tasks."""
    completed = passed + failed
    return passed / completed if completed > 0 else 0.0

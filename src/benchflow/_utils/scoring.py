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


def classify_result(
    *,
    reward: float | None,
    error: str | None,
    verifier_error: str | None,
) -> str:
    """Classify a single result into exactly one terminal bucket.

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


def classify_result_dict(result: dict) -> str:
    """Classify a result dict (as persisted to ``result.json``) into a bucket.

    Thin wrapper over :func:`classify_result` for the dict-shaped results
    used by ``Evaluation.run()``. See :func:`classify_result` for the
    bucket precedence rules.
    """
    return classify_result(
        reward=extract_reward(result),
        error=result.get("error"),
        verifier_error=result.get("verifier_error"),
    )


def pass_rate(*, passed: int, total: int) -> float:
    """Pass rate over all tasks."""
    return passed / total if total > 0 else 0.0


def pass_rate_excl_errors(*, passed: int, failed: int) -> float:
    """Pass rate excluding errored tasks."""
    completed = passed + failed
    return passed / completed if completed > 0 else 0.0

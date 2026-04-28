"""Pure scoring and classification helpers — no external dependencies.

CONTRACT SURFACE — semver-stable. Changes here break downstream importers
and retry logic. Reward codomain ({None} ∪ [-1, 1]) is enforced by the
Hypothesis property test in tests/test_scoring.py; do not widen it without
updating the test and calling out the break.
"""

# Error category constants
INSTALL_FAILED = "install_failure"
PIPE_CLOSED = "pipe_closed"
ACP_ERROR = "acp_error"
TIMED_OUT = "timeout"

# Substrings already matched globally by ``classify_error`` — agents may NOT
# re-claim these in their ``AgentConfig.error_taxonomy``. Single source of
# truth shared with ``tests/test_registry_invariants.py`` and (Phase 2) the
# cross-agent validator at ``benchflow.agents.discovery.validate_agents``.
RESERVED_ERROR_SUBSTRINGS = frozenset({
    "install failed",
    "closed stdout",
    "ACP error",
    "timed out",
})

# Verifier error category constants
VERIFIER_FAILED = "verifier_failure"
VERIFIER_TIMEOUT = "verifier_timeout"


def extract_reward(result: dict) -> float | None:
    """Extract the reward value from a result dict, or None if absent/invalid.

    Codomain: the reward is a real number in [-1.0, 1.0] inclusive. Any value
    outside that range, or any non-numeric value (including bool, which is a
    numeric subtype in Python but is intentionally excluded to force emitters
    to ship actual scores), is treated as missing and returns None.
    """
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    reward = float(reward)
    if reward != reward or reward < -1.0 or reward > 1.0:  # NaN or OOR
        return None
    return reward


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

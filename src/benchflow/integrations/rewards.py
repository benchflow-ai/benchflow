"""Attribution-aware rewards for training on BenchFlow tasks.

This module turns one scored BenchFlow run into a training reward, or into a
*drop* with its reason. It follows the rule proposed for BenchFlow's RL
cookbooks (not yet a formal BenchFlow decision):

- A failure is **infrastructure** only when the policy could not have caused
  it: the sandbox never started, the model endpoint failed, or the verifier
  crashed on a clean run (a sandbox the policy never touched).
- Infrastructure failures are **dropped** from the batch and counted by
  reason. They never become a 0.
- Every other failure scores **0**, timeouts included. A failure the policy
  could have caused is never dropped.

Callers pass whether the policy acted in the sandbox (ran any command or
wrote an answer). After the policy acted, a verifier crash, a lost sandbox,
or a verifier timeout could be the policy's doing (it can fill the disk, leave
a runaway process, or delete what the verifier reads), so each scores 0.

Reward integrity comes last. When an integrity audit of the rollout (for
example BenchShield, BenchFlow's reward-integrity layer) finds that the policy
exploited the grader, the rollout scores 0 and is flagged, whatever the
verifier said and even if it would otherwise be dropped: an exploit is the
policy's doing. Pass the audit's verdict as ``integrity=``; any object or
mapping with a boolean ``exploited`` and an optional ``reason`` works. This is
an extension point until BenchShield's port to BenchFlow 0.8 lands.

Typical use::

    from benchflow.integrations.rewards import reward_from_verify, sandbox_start_failure

    try:
        runtime = await TaskRuntime.create(config)
    except Exception as exc:
        decision = sandbox_start_failure(exc)  # dropped, reason "sandbox_start"
    else:
        ...  # policy loop
        decision = reward_from_verify(await runtime.verify(), policy_acted=acted)

    if decision.dropped:
        ...  # leave the sample out of the batch; count decision.reason
    else:
        ...  # train on decision.reward
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchflow._utils.scoring import (
    SANDBOX_SETUP,
    TIMED_OUT,
    VERIFIER_TIMEOUT,
    classify_error,
    classify_verifier_error,
    extract_reward,
)

# Reasons a sample is dropped. Each names a failure the policy cannot cause.
SANDBOX_START = "sandbox_start"
MODEL_ENDPOINT = "model_endpoint"
VERIFIER_CRASH_CLEAN_RUN = "verifier_crash_clean_run"
DROP_REASONS = frozenset({SANDBOX_START, MODEL_ENDPOINT, VERIFIER_CRASH_CLEAN_RUN})

# Reasons a sample scores. "scored" is a verifier reward; the rest are 0.
SCORED = "scored"
TIMEOUT = "timeout"
VERIFIER_ERROR = "verifier_error"
RUN_ERROR = "run_error"
NO_REWARD = "no_reward"
INTEGRITY_VIOLATION = "integrity_violation"
ZERO_REASONS = frozenset(
    {TIMEOUT, VERIFIER_ERROR, RUN_ERROR, NO_REWARD, INTEGRITY_VIOLATION}
)

_DETAIL_CHARS = 500


@dataclass(frozen=True)
class RewardDecision:
    """A training reward, or a drop with its reason.

    ``reward`` is None exactly when the sample is dropped. ``reason`` is
    ``"scored"`` for a verifier reward, one of :data:`ZERO_REASONS` for a
    failure scored 0, or one of :data:`DROP_REASONS` for a drop. ``detail``
    keeps a short excerpt of the underlying error, if any. ``flagged`` marks a
    rollout an integrity audit caught exploiting the grader; it always scores 0.
    """

    reward: float | None
    reason: str
    detail: str | None = None
    flagged: bool = False
    # Whether every check passed, when the verifier says so (a partial-credit
    # reward can be below 1 on a run that is still useful to train on).
    passed: bool | None = None

    def __post_init__(self) -> None:
        if self.flagged != (self.reason == INTEGRITY_VIOLATION) or (
            self.flagged and self.reward != 0.0
        ):
            raise ValueError(
                "a flagged rollout, and only a flagged rollout, scores 0 with reason "
                "'integrity_violation'; use apply_integrity()"
            )
        if self.reward is None:
            if self.reason not in DROP_REASONS:
                raise ValueError(
                    f"a dropped sample needs an infrastructure reason "
                    f"({sorted(DROP_REASONS)}), got {self.reason!r}"
                )
        elif self.reason in DROP_REASONS:
            raise ValueError(f"reason {self.reason!r} is a drop; reward must be None")
        elif not math.isfinite(self.reward):
            raise ValueError(f"reward must be finite, got {self.reward!r}")

    @property
    def dropped(self) -> bool:
        return self.reward is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "dropped": self.dropped,
            "reason": self.reason,
            "detail": self.detail,
            "flagged": self.flagged,
            "passed": self.passed,
        }


def scored(reward: float, *, passed: bool | None = None) -> RewardDecision:
    """A verifier reward. ``passed`` defaults to whether the reward is 1."""

    reward = float(reward)
    return RewardDecision(
        reward=reward, reason=SCORED, passed=reward >= 1.0 if passed is None else passed
    )


def zero(reason: str, detail: object = None) -> RewardDecision:
    """A failure the policy could have caused: it scores 0."""

    if reason not in ZERO_REASONS:
        raise ValueError(
            f"unknown zero reason {reason!r}; use one of {sorted(ZERO_REASONS)}"
        )
    return RewardDecision(
        reward=0.0, reason=reason, detail=_short(detail), passed=False
    )


def dropped(reason: str, detail: object = None) -> RewardDecision:
    """An infrastructure failure: the sample leaves the batch."""

    return RewardDecision(reward=None, reason=reason, detail=_short(detail))


def sandbox_start_failure(error: object = None) -> RewardDecision:
    """The sandbox never started, so the policy never acted: drop."""

    return dropped(SANDBOX_START, error)


def model_endpoint_failure(error: object = None) -> RewardDecision:
    """The model endpoint failed (connection, 5xx, rate limit, auth): drop.

    Do not use this for a request the endpoint rejected because of what the
    policy produced, such as a context-length overflow. That is the policy's
    budget running out: end the episode and score what it left.
    """

    return dropped(MODEL_ENDPOINT, error)


def reward_from_verify(
    result: Any, *, policy_acted: bool = True, integrity: Any = None
) -> RewardDecision:
    """Turn a verify result into a reward or a drop.

    ``result`` is a :class:`~benchflow.rollout.TaskRuntimeResult`, a
    :class:`~benchflow.models.RolloutResult`, or a ``result.json`` mapping.
    ``policy_acted`` says whether the policy ran anything in the sandbox
    before verification. Pass False only when it provably did not: then the
    sandbox was clean, and a verifier that still produced no reward crashed on
    a clean run. ``integrity`` is an optional audit verdict; see
    :func:`apply_integrity`.
    """

    return apply_integrity(_decide(result, policy_acted=policy_acted), integrity)


def apply_integrity(decision: RewardDecision, verdict: Any) -> RewardDecision:
    """Apply a reward-integrity audit verdict to a decision.

    ``verdict`` is None when no audit ran, which leaves the decision as it
    is. Otherwise it is any object or mapping with a boolean ``exploited``
    and an optional ``reason``, such as a BenchShield finding once its port
    lands. An exploit turns the decision into a flagged 0: never a reward,
    and never a drop, because the policy caused it.
    """

    if verdict is None:
        return decision
    exploited = _field(verdict, "exploited")
    if not isinstance(exploited, bool):
        raise ValueError("an integrity verdict needs a boolean 'exploited'")
    if not exploited:
        return decision
    reason = (
        _field(verdict, "reason") or "integrity audit: the policy exploited the grader"
    )
    return RewardDecision(
        reward=0.0,
        reason=INTEGRITY_VIOLATION,
        detail=_short(reason),
        flagged=True,
        passed=False,
    )


def _decide(result: Any, *, policy_acted: bool) -> RewardDecision:
    reward = _finite_reward(result)
    if reward is not None:
        return scored(reward, passed=_passed(result, reward))

    error = _field(result, "error")
    verifier_error = _field(result, "verifier_error")
    error_category = _field(result, "error_category") or classify_error(error)
    verifier_category = _field(result, "verifier_error_category") or (
        classify_verifier_error(verifier_error)
    )
    detail = verifier_error or error

    if error_category == SANDBOX_SETUP:
        # BenchFlow names this category only for failures to create or start
        # the sandbox, which precede any policy action.
        return sandbox_start_failure(detail)
    if not policy_acted:
        return dropped(VERIFIER_CRASH_CLEAN_RUN, detail)
    if verifier_category == VERIFIER_TIMEOUT or error_category == TIMED_OUT:
        return zero(TIMEOUT, detail)
    if verifier_error:
        return zero(VERIFIER_ERROR, detail)
    if error:
        return zero(RUN_ERROR, detail)
    return zero(NO_REWARD)


def summarize(decisions: Iterable[RewardDecision]) -> dict[str, Any]:
    """Count a batch of decisions: kept, dropped by reason, zeros by reason.

    ``mean_reward`` averages the kept samples only; it is None when every
    sample was dropped.
    """

    decisions = list(decisions)
    kept = [d.reward for d in decisions if d.reward is not None]
    return {
        "n": len(decisions),
        "kept": len(kept),
        "dropped": len(decisions) - len(kept),
        "flagged": sum(1 for d in decisions if d.flagged),
        "drop_reasons": dict(Counter(d.reason for d in decisions if d.dropped)),
        "zero_reasons": dict(
            Counter(d.reason for d in decisions if d.reason in ZERO_REASONS)
        ),
        "mean_reward": (sum(kept) / len(kept)) if kept else None,
        "passed": sum(1 for d in decisions if d.passed),
        "pass_rate": (
            sum(1 for d in decisions if d.passed and not d.dropped) / len(kept)
        )
        if kept
        else None,
    }


def dynamic_sampling(
    groups: Mapping[Any, Sequence[RewardDecision]], *, enabled: bool = False
) -> tuple[dict[Any, list[RewardDecision]], dict[str, int]]:
    """Leave out groups whose kept rewards are all equal (DAPO's dynamic sampling).

    ``groups`` maps a group id (for GRPO, one prompt's rollouts) to its
    decisions. A group whose kept rewards are all the same, or that has fewer
    than two kept rollouts, gives group-relative advantages of zero: no
    learning signal. Off by default. The counts are returned either way:
    ``groups``, ``uniform`` (groups with no signal), and ``removed`` (uniform
    groups actually left out, 0 unless ``enabled``). Dropped rollouts inside a
    kept group stay dropped; this never turns a drop into a reward.
    """

    kept: dict[Any, list[RewardDecision]] = {}
    uniform = 0
    for group, decisions in groups.items():
        rewards = [d.reward for d in decisions if d.reward is not None]
        is_uniform = len(rewards) < 2 or max(rewards) - min(rewards) < 1e-12
        uniform += is_uniform
        if not (enabled and is_uniform):
            kept[group] = list(decisions)
    counts = {
        "groups": len(groups),
        "uniform": uniform,
        "removed": len(groups) - len(kept),
    }
    return kept, counts


def _passed(result: Any, reward: float) -> bool:
    rewards = _field(result, "rewards")
    if isinstance(rewards, Mapping):
        value = rewards.get("passed")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value >= 0.5
    return reward >= 1.0


def _field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _finite_reward(result: Any) -> float | None:
    if isinstance(result, Mapping):
        # A result.json: honor review gating and pending assessments.
        value = extract_reward(result)
    else:
        value = _field(result, "reward")
    if value is None:
        rewards = _field(result, "rewards")
        if isinstance(rewards, Mapping) and not isinstance(result, Mapping):
            value = rewards.get("reward")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _short(detail: object) -> str | None:
    if detail is None:
        return None
    text = detail if isinstance(detail, str) else f"{type(detail).__name__}: {detail}"
    text = " ".join(text.split())
    if len(text) > _DETAIL_CHARS:
        text = text[: _DETAIL_CHARS - 3] + "..."
    return text or None


__all__ = [
    "DROP_REASONS",
    "INTEGRITY_VIOLATION",
    "MODEL_ENDPOINT",
    "NO_REWARD",
    "RUN_ERROR",
    "SANDBOX_START",
    "SCORED",
    "TIMEOUT",
    "VERIFIER_CRASH_CLEAN_RUN",
    "VERIFIER_ERROR",
    "ZERO_REASONS",
    "RewardDecision",
    "apply_integrity",
    "dropped",
    "dynamic_sampling",
    "model_endpoint_failure",
    "reward_from_verify",
    "sandbox_start_failure",
    "scored",
    "summarize",
    "zero",
]

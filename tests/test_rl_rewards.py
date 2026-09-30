"""The attribution rule of benchflow.integrations.rewards.

The rule, proposed for BenchFlow's RL cookbooks: a failure is infrastructure
only when the policy could not have caused it; infrastructure failures are
dropped and counted; every other failure scores 0, timeouts included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from benchflow.integrations import rewards
from benchflow.integrations.rewards import (
    RewardDecision,
    apply_integrity,
    dropped,
    model_endpoint_failure,
    reward_from_verify,
    sandbox_start_failure,
    summarize,
    zero,
)
from benchflow.models import RolloutResult
from benchflow.rollout import TaskRuntimeResult


def _runtime_result(**fields) -> TaskRuntimeResult:
    base = {
        "task_name": "t",
        "rollout_name": "r",
        "reward": None,
        "rewards": None,
        "verifier_error": None,
        "error": None,
        "rollout_dir": Path("/tmp/r"),
        "result": RolloutResult(task_name="t"),
    }
    base.update(fields)
    return TaskRuntimeResult(**base)


def test_verifier_reward_is_the_reward() -> None:
    decision = reward_from_verify(_runtime_result(reward=1.0, rewards={"reward": 1.0}))
    assert decision == RewardDecision(reward=1.0, reason="scored")
    assert not decision.dropped


def test_zero_reward_is_scored_not_a_failure() -> None:
    decision = reward_from_verify(_runtime_result(reward=0.0))
    assert decision.reward == 0.0
    assert decision.reason == "scored"


def test_explicit_reward_wins_over_an_error_string() -> None:
    """A reward from the verifier stands, as in BenchFlow's score buckets."""

    decision = reward_from_verify(_runtime_result(reward=0.5, error="warning: slow"))
    assert decision.reward == 0.5


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1"])
def test_non_numeric_or_non_finite_reward_is_not_a_reward(bad) -> None:
    decision = reward_from_verify(_runtime_result(reward=bad, verifier_error="boom"))
    assert decision.reason == "verifier_error"
    assert decision.reward == 0.0


def test_sandbox_that_never_started_is_dropped() -> None:
    decision = sandbox_start_failure(RuntimeError("Daytona: no capacity"))
    assert decision.dropped
    assert decision.reward is None
    assert decision.reason == "sandbox_start"
    assert "no capacity" in (decision.detail or "")


def test_sandbox_setup_category_is_dropped_even_when_flagged_as_acted() -> None:
    """BenchFlow only names sandbox_setup for failures before any policy action."""

    decision = reward_from_verify(
        _runtime_result(error="Sandbox startup failed: image pull timed out"),
        policy_acted=True,
    )
    assert decision.reason == "sandbox_start"
    assert decision.dropped


def test_verifier_crash_after_the_policy_acted_scores_zero() -> None:
    """The policy could have broken what the verifier needs: never drop it."""

    decision = reward_from_verify(
        _runtime_result(verifier_error="verifier crashed: exit 1"), policy_acted=True
    )
    assert decision == RewardDecision(0.0, "verifier_error", "verifier crashed: exit 1")


def test_lost_sandbox_after_the_policy_acted_scores_zero() -> None:
    """A transport failure is ambiguous once the policy ran commands."""

    decision = reward_from_verify(
        _runtime_result(verifier_error="verifier crashed: sandbox not found"),
        policy_acted=True,
    )
    assert decision.reward == 0.0
    assert not decision.dropped


def test_verifier_crash_on_a_clean_run_is_dropped() -> None:
    decision = reward_from_verify(
        _runtime_result(verifier_error="verifier crashed: exit 1"), policy_acted=False
    )
    assert decision.dropped
    assert decision.reason == "verifier_crash_clean_run"


def test_timeouts_score_zero() -> None:
    verifier_timeout = reward_from_verify(
        _runtime_result(verifier_error="verifier timed out after 60s")
    )
    agent_timeout = reward_from_verify(
        _runtime_result(error="prompt exceeded wall-clock budget of 600s")
    )
    assert verifier_timeout == RewardDecision(
        0.0, "timeout", "verifier timed out after 60s"
    )
    assert agent_timeout.reward == 0.0
    assert agent_timeout.reason == "timeout"


def test_no_reward_and_no_error_scores_zero_after_the_policy_acted() -> None:
    assert reward_from_verify(_runtime_result()) == RewardDecision(0.0, "no_reward")


def test_result_json_mapping_is_accepted() -> None:
    passed = {"rewards": {"reward": 1.0}, "error": None, "verifier_error": None}
    crashed = {"rewards": None, "verifier_error": "verifier crashed: exit 2"}
    assert reward_from_verify(passed).reward == 1.0
    assert reward_from_verify(crashed).reason == "verifier_error"
    assert reward_from_verify(crashed, policy_acted=False).dropped


def test_result_json_pending_assessment_has_no_reward() -> None:
    """A result whose assessment is pending carries no score, even with a reward."""

    pending = {"rewards": {"reward": 1.0}, "assessment": "pending"}
    assert reward_from_verify(pending).reason == "no_reward"


def test_rollout_result_is_accepted() -> None:
    result = RolloutResult(task_name="t", rewards={"reward": 1.0})
    assert reward_from_verify(result).reward == 1.0


def test_model_endpoint_failure_is_dropped() -> None:
    decision = model_endpoint_failure("HTTP 503 from the endpoint")
    assert decision.dropped
    assert decision.reason == "model_endpoint"


def test_a_drop_needs_an_infrastructure_reason() -> None:
    with pytest.raises(ValueError, match="infrastructure reason"):
        dropped("timeout")
    with pytest.raises(ValueError, match="is a drop"):
        RewardDecision(reward=0.0, reason="sandbox_start")
    with pytest.raises(ValueError, match="unknown zero reason"):
        zero("sandbox_start")


def test_integrity_violation_scores_zero_and_flags() -> None:
    """An exploit is the policy's doing: a flagged 0, never a reward."""

    passed = reward_from_verify(_runtime_result(reward=1.0))
    flagged = apply_integrity(
        passed, {"exploited": True, "reason": "read the answer file"}
    )
    assert flagged.reward == 0.0
    assert flagged.flagged
    assert flagged.reason == "integrity_violation"
    assert flagged.detail == "read the answer file"


def test_integrity_violation_overrides_a_drop() -> None:
    """A rollout that exploited the grader is never dropped, even if infra later failed."""

    @dataclass
    class Verdict:
        exploited: bool
        reason: str | None = None

    decision = apply_integrity(sandbox_start_failure("lost"), Verdict(exploited=True))
    assert not decision.dropped
    assert decision.flagged


def test_clean_integrity_verdict_and_no_verdict_leave_the_decision() -> None:
    passed = reward_from_verify(_runtime_result(reward=1.0))
    assert apply_integrity(passed, None) is passed
    assert apply_integrity(passed, {"exploited": False}) is passed
    assert reward_from_verify(
        _runtime_result(reward=1.0), integrity={"exploited": True}
    ).flagged
    with pytest.raises(ValueError, match="boolean"):
        apply_integrity(passed, {"exploited": "yes"})


def test_integrity_violation_cannot_be_built_unflagged() -> None:
    with pytest.raises(ValueError, match="flagged"):
        zero("integrity_violation")


def test_summarize_counts_drops_and_zero_reasons() -> None:
    decisions = [
        rewards.scored(1.0),
        rewards.scored(0.0),
        zero("timeout"),
        sandbox_start_failure(),
        dropped("verifier_crash_clean_run"),
        apply_integrity(rewards.scored(1.0), {"exploited": True}),
    ]
    summary = summarize(decisions)
    assert summary["n"] == 6
    assert summary["kept"] == 4
    assert summary["dropped"] == 2
    assert summary["flagged"] == 1
    assert summary["drop_reasons"] == {
        "sandbox_start": 1,
        "verifier_crash_clean_run": 1,
    }
    assert summary["zero_reasons"] == {"timeout": 1, "integrity_violation": 1}
    assert summary["mean_reward"] == pytest.approx(0.25)
    assert summarize([sandbox_start_failure()])["mean_reward"] is None

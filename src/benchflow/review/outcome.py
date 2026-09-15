"""Persisted terminal scoring contract for automatic rubric review.

Gate success and reward magnitude answer different questions. This is the one
boundary that turns an already validated, host-scored review into a final result;
the arithmetic remains owned by :mod:`benchflow.review.scoring`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchflow.review.scoring import ReviewScoring


class ScoringResult(BaseModel):
    """A complete verdict or a scoring failure, never a fabricated zero."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    policy: Literal["tests-blockers-quality-v1"] = "tests-blockers-quality-v1"
    status: Literal["complete", "error"]
    passed: bool | None = None
    tests_pass: bool | None = None
    all_blockers_pass: bool | None = None
    failed_blockers: list[str] = Field(default_factory=list)
    verifier_reward: float | None = Field(default=None, allow_inf_nan=False)
    rubric_reward: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reviewer_run: str | None = None
    revision: str | None = Field(
        default=None, pattern=r"^scoring/[A-Za-z0-9_-]+[.]json$"
    )
    error: str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        if self.status == "error":
            if not self.error or not self.error.strip():
                raise ValueError("a scoring error requires an explanation")
            if self.passed is not None or self.rubric_reward is not None:
                raise ValueError(
                    "incomplete scoring cannot claim a pass or quality score"
                )
            return self
        if self.error is not None:
            raise ValueError("complete scoring cannot contain a scoring error")
        if self.tests_pass is None or self.all_blockers_pass is None:
            raise ValueError("complete scoring requires both gate verdicts")
        if self.verifier_reward is None or self.rubric_reward is None:
            raise ValueError("complete scoring requires both component rewards")
        if self.tests_pass != (self.verifier_reward == 1.0):
            raise ValueError(
                "test gate does not match the deterministic verifier reward"
            )
        if self.passed != (self.tests_pass and self.all_blockers_pass):
            raise ValueError("passed must equal the conjunction of both gates")
        if self.all_blockers_pass != (not self.failed_blockers):
            raise ValueError("blocker gate does not match failed_blockers")
        if len(set(self.failed_blockers)) != len(self.failed_blockers):
            raise ValueError("failed_blockers must be unique")
        if not self.reviewer_run or not self.reviewer_run.strip():
            raise ValueError("complete scoring requires the reviewer run reference")
        return self

    def numeric_rewards(self) -> dict[str, float] | None:
        """Return the canonical reward envelope only when all scoring completed."""
        if self.status != "complete":
            return None
        assert self.verifier_reward is not None and self.rubric_reward is not None
        return {
            "reward": self.rubric_reward if self.passed else 0.0,
            "verifier_reward": self.verifier_reward,
            "rubric_reward": self.rubric_reward,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public result.json scoring block."""
        return self.model_dump(mode="json")


def complete_scoring(
    review: ReviewScoring, *, verifier_reward: float, reviewer_run: str
) -> ScoringResult:
    """Persist an existing weighted review without recalculating its arithmetic."""
    return ScoringResult(
        status="complete",
        passed=review.deterministic_pass and review.all_blockers_pass,
        tests_pass=review.deterministic_pass,
        all_blockers_pass=review.all_blockers_pass,
        failed_blockers=list(review.failed_blockers),
        verifier_reward=verifier_reward,
        rubric_reward=review.raw_quality,
        reviewer_run=reviewer_run,
    )


def scoring_error(
    error: str,
    *,
    tests_pass: bool | None = None,
    verifier_reward: float | None = None,
    reviewer_run: str | None = None,
) -> ScoringResult:
    """Retain available evidence while withholding a final capability score."""
    return ScoringResult(
        status="error",
        tests_pass=tests_pass,
        verifier_reward=verifier_reward,
        reviewer_run=reviewer_run,
        error=error,
    )


def scoring_from_result(result: Mapping[str, Any]) -> ScoringResult | None:
    """Read a scoring block and reject a stale or inconsistent reward envelope.

    Missing/null scoring belongs to the legacy contract. A present malformed
    block raises instead of silently reverting to legacy ``reward == 1``.
    """
    raw = result.get("scoring")
    if raw is None:
        return None
    scoring = ScoringResult.model_validate(raw)
    if scoring.status == "complete":
        rewards = result.get("rewards")
        # The trainer-facing results.jsonl contract carries a scalar at the
        # top level; ordinary result.json carries the named reward envelope.
        if "rewards" not in result and "reward" in result:
            rewards = {"reward": result["reward"]}
        expected = scoring.numeric_rewards()
        assert expected is not None
        if not isinstance(rewards, Mapping):
            raise ValueError("complete scoring requires a reward envelope")
        for key, value in expected.items():
            if key != "reward" and key not in rewards:
                continue
            actual = rewards.get(key)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or actual != value
            ):
                raise ValueError(f"{key} disagrees with the scoring verdict")
    return scoring


def deterministic_pass(result: Mapping[str, Any]) -> bool:
    """Read the test gate without mistaking final quality for test performance."""
    if result.get("scoring") is not None:
        try:
            scoring = scoring_from_result(result)
        except ValueError:
            return False
        return scoring is not None and scoring.tests_pass is True
    rewards = result.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
    return (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and reward == 1.0
        and result.get("error") is None
        and result.get("verifier_error") is None
    )

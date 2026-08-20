"""Deterministic aggregation for weighted review rubrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from benchflow.review.config import Rubric

PUBLISHABLE_QUALITY = 0.80
REVISIONS_QUALITY = 0.65


class PublicationDecision(StrEnum):
    """Publication band produced after deterministic and blocker gates."""

    PUBLISHABLE = "publishable"
    PRESENTABLE_WITH_REVISIONS = "presentable_with_revisions"
    NOT_PUBLISHABLE = "not_publishable"


@dataclass(frozen=True)
class ReviewScoring:
    """Auditable host-side aggregation of one valid v0.2 review."""

    deterministic_pass: bool
    all_blockers_pass: bool
    failed_blockers: tuple[str, ...]
    weighted_points: int
    max_weighted_points: int
    raw_quality: float
    gated_quality: float
    decision: PublicationDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_pass": self.deterministic_pass,
            "all_blockers_pass": self.all_blockers_pass,
            "failed_blockers": list(self.failed_blockers),
            "weighted_points": self.weighted_points,
            "max_weighted_points": self.max_weighted_points,
            "raw_quality": self.raw_quality,
            "gated_quality": self.gated_quality,
            "decision": self.decision.value,
        }


def score_weighted_review(
    rubric: Rubric,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    deterministic_pass: bool,
) -> ReviewScoring:
    """Aggregate one structurally valid v0.2 review.

    Blockers are binary gates and do not enter weighted quality. Scored
    criteria contribute ``score * weight`` out of ``2 * weight``. The raw
    quality remains visible even when a deterministic or blocker gate fails;
    the gated quality is then zero and the decision is not publishable.
    """

    if not rubric.is_weighted:
        raise ValueError("weighted scoring requires a v0.2 rubric")
    if type(deterministic_pass) is not bool:
        raise ValueError("deterministic_pass must be a boolean")

    failed_blockers: list[str] = []
    weighted_points = 0
    max_weighted_points = 0
    for criterion in rubric.criteria:
        check = checks.get(criterion.name)
        if not isinstance(check, Mapping):
            raise ValueError(f"missing review check for {criterion.name!r}")
        if criterion.is_blocker:
            outcome = check.get("outcome")
            if not isinstance(outcome, str) or outcome not in {"pass", "fail"}:
                raise ValueError(
                    f"blocker {criterion.name!r} must have outcome pass or fail"
                )
            if outcome == "fail":
                failed_blockers.append(criterion.name)
            continue

        score = check.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
            raise ValueError(
                f"scored criterion {criterion.name!r} must have integer score 0, 1, or 2"
            )
        if criterion.weight is None:  # guarded by Rubric validation
            raise ValueError(f"scored criterion {criterion.name!r} has no weight")
        weighted_points += score * criterion.weight
        max_weighted_points += 2 * criterion.weight

    if max_weighted_points == 0:  # guarded by Rubric validation
        raise ValueError("weighted rubric has no scored criteria")

    raw_quality = weighted_points / max_weighted_points
    all_blockers_pass = not failed_blockers
    gates_pass = deterministic_pass and all_blockers_pass
    gated_quality = raw_quality if gates_pass else 0.0
    if not gates_pass or raw_quality < REVISIONS_QUALITY:
        decision = PublicationDecision.NOT_PUBLISHABLE
    elif raw_quality < PUBLISHABLE_QUALITY:
        decision = PublicationDecision.PRESENTABLE_WITH_REVISIONS
    else:
        decision = PublicationDecision.PUBLISHABLE

    return ReviewScoring(
        deterministic_pass=deterministic_pass,
        all_blockers_pass=all_blockers_pass,
        failed_blockers=tuple(failed_blockers),
        weighted_points=weighted_points,
        max_weighted_points=max_weighted_points,
        raw_quality=raw_quality,
        gated_quality=gated_quality,
        decision=decision,
    )

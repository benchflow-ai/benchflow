"""Verdict validation and score aggregation for rubric reviews.

The scoring contract, in one place:

- A verdict must be a member of its criterion's ``choices``; anything else is
  an **unscored** criterion, never a zero. A reviewer failure must not be
  scored against the model (PrimeIntellect verifiers v1's rule; the opposite
  default — timeout scores 0.0 — makes infra noise indistinguishable from a
  genuine failure).
- Gates: every ``required`` criterion must land on its best choice, or the
  review is 0.0 regardless of the weighted mean.
- The mean is HealthBench's: signed weights in the numerator, **positive
  weights only** in the denominator, final score clamped to [0, 1]. Penalties
  can only erode earned credit.
- ``weight == 0`` criteria are recorded as metrics and never move the score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from benchflow.review.config import ReviewCriterion, ReviewRubric

# Review lifecycle states surfaced in review_details.json.
STATUS_SCORED = "scored"
STATUS_UNSCORED = "unscored"
STATUS_CONFIG_ERROR = "config_error"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


@dataclass
class CriterionVerdict:
    """One criterion's parsed reviewer answer."""

    criterion_id: str
    verdict: str | None
    reasoning: str = ""
    evidence: tuple[str, ...] = ()
    score: float | None = None
    unscored_reason: str | None = None


@dataclass
class ReviewOutcome:
    """Aggregated review result, ready for rewards + details serialization."""

    status: str
    review: float | None = None
    passed: bool | None = None
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    error: str | None = None

    def reward_updates(self, rubric: ReviewRubric | None) -> dict[str, float]:
        """Reward-dict keys this outcome contributes.

        Only a fully scored review writes keys: a partial score computed while
        some criterion is unscored would silently misrepresent the rubric, so
        unscored/error reviews keep the rewards dict untouched and speak
        through ``review_details.json`` instead. The primary ``reward`` key is
        never written here — the execution verifier owns it.
        """
        if self.status != STATUS_SCORED or self.review is None or rubric is None:
            return {}
        updates: dict[str, float] = {
            "review": round(self.review, 4),
            "review_passed": 1.0 if self.passed else 0.0,
        }
        for verdict in self.verdicts:
            if verdict.score is not None:
                updates[f"review/{verdict.criterion_id}"] = round(verdict.score, 4)
        return updates


def extract_verdicts_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced JSON object carrying a ``"verdicts"`` key.

    Scanning for the key (not just the first ``{``) skips prose and — the case
    that actually bites — an echoed copy of the format example from the
    prompt, which fails to parse or lacks real content. Returns ``None`` when
    no such object exists.
    """
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "verdicts" in obj:
            return obj
        index = text.find("{", index + 1)
    return None


def parse_reviewer_message(
    text: str, expected: list[ReviewCriterion]
) -> tuple[list[CriterionVerdict], str | None]:
    """Parse one reviewer message into verdicts for ``expected`` criteria.

    Returns ``(verdicts, error)``. ``error`` is a human-readable description
    of a *structural* failure (no JSON, id mismatch) suitable for feeding back
    to the reviewer on retry; per-criterion problems (off-menu verdict,
    missing evidence) land on the individual verdict as ``unscored_reason``.
    """
    obj = extract_verdicts_object(text)
    if obj is None:
        return [], (
            "no JSON object with a 'verdicts' key was found in the reply; "
            "return exactly one such object"
        )
    raw_verdicts = obj.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return [], "'verdicts' must be a list"

    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_verdicts:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return [], "every verdicts[] entry must be an object with an 'id'"
        if item["id"] in by_id:
            return [], f"duplicate verdict id {item['id']!r}"
        by_id[item["id"]] = item

    expected_ids = [c.id for c in expected]
    missing = [cid for cid in expected_ids if cid not in by_id]
    extra = [cid for cid in by_id if cid not in expected_ids]
    if missing or extra:
        return [], (
            f"verdict ids do not match the requested criteria "
            f"(missing: {missing or 'none'}, unexpected: {extra or 'none'}); "
            f"answer exactly the ids {expected_ids}"
        )

    verdicts: list[CriterionVerdict] = []
    for criterion in expected:
        item = by_id[criterion.id]
        raw_verdict = item.get("verdict")
        reasoning = item.get("reasoning")
        evidence_raw = item.get("evidence")
        evidence = (
            tuple(str(e) for e in evidence_raw)
            if isinstance(evidence_raw, list)
            else ()
        )

        unscored_reason: str | None = None
        score: float | None = None
        if not isinstance(raw_verdict, str) or raw_verdict not in criterion.choices:
            unscored_reason = (
                f"verdict {raw_verdict!r} is not one of {list(criterion.choices)}"
            )
        elif not isinstance(reasoning, str) or not reasoning.strip():
            unscored_reason = "verdict has no reasoning"
        elif not evidence:
            unscored_reason = (
                "verdict cites no evidence (file paths, trajectory steps, or "
                "a description of what was searched)"
            )
        else:
            score = criterion.choice_score(raw_verdict)

        verdicts.append(
            CriterionVerdict(
                criterion_id=criterion.id,
                verdict=raw_verdict if isinstance(raw_verdict, str) else None,
                reasoning=reasoning if isinstance(reasoning, str) else "",
                evidence=evidence,
                score=score,
                unscored_reason=unscored_reason,
            )
        )
    return verdicts, None


def aggregate(rubric: ReviewRubric, verdicts: list[CriterionVerdict]) -> ReviewOutcome:
    """Aggregate per-criterion verdicts into the review outcome."""
    by_id = {v.criterion_id: v for v in verdicts}

    unscored = [
        v
        for v in verdicts
        if v.score is None
        # A metric criterion failing to score is reported but must not block
        # the review score — it could not have moved it anyway.
        and not _criterion(rubric, v.criterion_id).is_metric
    ]
    if unscored:
        reasons = "; ".join(
            f"{v.criterion_id}: {v.unscored_reason}" for v in unscored
        )
        return ReviewOutcome(
            status=STATUS_UNSCORED,
            verdicts=verdicts,
            error=f"review is unscored — {reasons}",
        )

    failed_gates = [
        gate.id
        for gate in rubric.gates
        if (by_id[gate.id].score or 0.0) < 1.0
    ]
    if failed_gates:
        review = 0.0
    else:
        scored = rubric.scored
        denominator = sum(c.weight for c in scored if c.weight > 0)
        if denominator > 0:
            numerator = sum(
                c.weight * (by_id[c.id].score or 0.0) for c in scored
            )
            review = min(1.0, max(0.0, numerator / denominator))
        else:
            # Loader guarantees gates exist in this shape: gates all passed
            # and nothing else can move the score.
            review = 1.0

    return ReviewOutcome(
        status=STATUS_SCORED,
        review=review,
        passed=review >= rubric.pass_threshold,
        verdicts=verdicts,
        failed_gates=failed_gates,
    )


def _criterion(rubric: ReviewRubric, criterion_id: str) -> ReviewCriterion:
    for criterion in rubric.criteria:
        if criterion.id == criterion_id:
            return criterion
    raise KeyError(criterion_id)

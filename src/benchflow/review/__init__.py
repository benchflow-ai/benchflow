"""Rubric review — post-verify agentic grading of a rollout against rubric.json.

Public surface:

- :func:`load_review_rubric` / :func:`find_review_rubric` — strict JSON-only
  rubric loading and discovery.
- :class:`ReviewRubric` / :class:`ReviewCriterion` / :class:`ReviewerSpec` —
  the parsed rubric.
- :class:`ReviewParams` — caller-side knobs (``--review``,
  ``--reviewer-agent``, ``--reviewer-model``).
- :class:`ReviewOutcome` / :func:`aggregate` — scoring.

The rollout-side engine lives in :mod:`benchflow.rollout._review`; it runs the
reviewer agent in the task sandbox after ``verify()`` and merges ``review*``
keys into the rewards dict.
"""

from benchflow.review.config import (
    REVIEW_RUBRIC_FILENAME,
    REVIEW_SCHEMA_VERSION,
    ReviewCriterion,
    ReviewerSpec,
    ReviewParams,
    ReviewRubric,
    ReviewRubricError,
    find_review_rubric,
    is_review_rubric_file,
    load_review_rubric,
)
from benchflow.review.scoring import (
    CriterionVerdict,
    ReviewOutcome,
    aggregate,
    extract_verdicts_object,
    parse_reviewer_message,
)

__all__ = [
    "REVIEW_RUBRIC_FILENAME",
    "REVIEW_SCHEMA_VERSION",
    "CriterionVerdict",
    "ReviewCriterion",
    "ReviewOutcome",
    "ReviewParams",
    "ReviewRubric",
    "ReviewRubricError",
    "ReviewerSpec",
    "aggregate",
    "extract_verdicts_object",
    "find_review_rubric",
    "is_review_rubric_file",
    "load_review_rubric",
    "parse_reviewer_message",
]

"""Rubric review — detached agentic grading of finished rollouts.

A rubric is a JSON object containing a ``criteria`` list. Legacy v0.1
criteria have ``name``, ``description``, and ``guidance`` and receive
pass/fail/not-applicable judgments. Weighted v0.2 criteria additionally have
strict ``blocker`` and ``weight`` integers: blockers receive pass/fail gates,
while non-blockers receive 0/1/2 scores that Benchflow aggregates host-side.

Reviews run *after* rollouts, from their host-side directories, as ordinary
rollouts of throwaway wrapper tasks (:mod:`benchflow.review.wrapper`), so
every sandbox backend works unchanged.  Review results live in
``review_report.json``. The detached ``bench review`` command remains
read-only; the rollout lifecycle separately integrates weighted task rubrics
into the source rollout's canonical final score.

Public surface:

- :func:`load_rubric` / :func:`find_task_rubric` — rubric loading and
  per-task discovery.
- :class:`Rubric` / :class:`RubricCriterion` — the parsed rubric.
- :func:`run_reviews` — review one rollout directory or a whole job
  directory; returns a :class:`ReviewReport`.
"""

from benchflow.review.config import (
    DEFAULT_RUBRIC_PATH,
    LEGACY_REVIEW_RUBRIC_CONTRACT,
    REVIEW_RESULT_FILENAME,
    REVIEW_RUBRIC_CONTRACT,
    REVIEW_RUBRIC_FILENAME,
    WEIGHTED_REVIEW_RUBRIC_CONTRACT,
    BlockerCriterionCheck,
    BlockerOutcomeValue,
    CriterionCheck,
    ReviewOutcomeValue,
    ReviewRubricError,
    Rubric,
    RubricCriterion,
    ScoredCriterionCheck,
    build_criteria_guidance,
    build_review_response_model,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.policy import (
    DEFAULT_REVIEWER_AGENT,
    DEFAULT_REVIEWER_MAX_RETRIES,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_REVIEWER_REASONING_EFFORT,
    DEFAULT_REVIEWER_TIMEOUT_SEC,
    RubricReviewConfig,
)
from benchflow.review.runner import (
    REVIEW_REPORT_FILENAME,
    ReviewReport,
    ReviewRunError,
    TrialReview,
    discover_rollouts,
    run_reviews,
)
from benchflow.review.scoring import (
    PUBLISHABLE_QUALITY,
    REVISIONS_QUALITY,
    PublicationDecision,
    ReviewScoring,
    score_weighted_review,
)
from benchflow.review.wrapper import assemble_review_task

__all__ = [
    "DEFAULT_RUBRIC_PATH",
    "DEFAULT_REVIEWER_AGENT",
    "DEFAULT_REVIEWER_MAX_RETRIES",
    "DEFAULT_REVIEWER_MODEL",
    "DEFAULT_REVIEWER_REASONING_EFFORT",
    "DEFAULT_REVIEWER_TIMEOUT_SEC",
    "LEGACY_REVIEW_RUBRIC_CONTRACT",
    "PUBLISHABLE_QUALITY",
    "REVIEW_RUBRIC_CONTRACT",
    "REVIEW_REPORT_FILENAME",
    "REVIEW_RESULT_FILENAME",
    "REVIEW_RUBRIC_FILENAME",
    "REVISIONS_QUALITY",
    "WEIGHTED_REVIEW_RUBRIC_CONTRACT",
    "BlockerCriterionCheck",
    "BlockerOutcomeValue",
    "CriterionCheck",
    "PublicationDecision",
    "ReviewOutcomeValue",
    "ReviewReport",
    "ReviewRubricError",
    "ReviewRunError",
    "ReviewScoring",
    "Rubric",
    "RubricCriterion",
    "RubricReviewConfig",
    "ScoredCriterionCheck",
    "TrialReview",
    "assemble_review_task",
    "build_criteria_guidance",
    "build_review_response_model",
    "discover_rollouts",
    "find_task_rubric",
    "load_rubric",
    "run_reviews",
    "score_weighted_review",
]

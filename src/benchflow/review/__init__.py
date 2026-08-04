"""Rubric review — detached agentic grading of finished rollouts.

A rubric (contract **v0.1**) is a JSON object containing a ``criteria`` list;
each criterion has ``name``, ``description``, and ``guidance``. A reviewer
agent reads a finished rollout's records inside its own sandbox and answers
each criterion with ``pass`` / ``fail`` / ``not_applicable`` plus an
explanation.

Reviews run *after* rollouts, from their host-side directories, as ordinary
rollouts of throwaway wrapper tasks (:mod:`benchflow.review.wrapper`), so
every sandbox backend works unchanged.  Review results live in
``review_report.json``; they are never merged into a reviewed rollout's
rewards or ``result.json``.

Public surface:

- :func:`load_rubric` / :func:`find_task_rubric` — rubric loading and
  per-task discovery.
- :class:`Rubric` / :class:`RubricCriterion` — the parsed rubric.
- :func:`run_reviews` — review one rollout directory or a whole job
  directory; returns a :class:`ReviewReport`.
"""

from benchflow.review.config import (
    DEFAULT_RUBRIC_PATH,
    REVIEW_RESULT_FILENAME,
    REVIEW_RUBRIC_CONTRACT,
    REVIEW_RUBRIC_FILENAME,
    CriterionCheck,
    ReviewOutcomeValue,
    ReviewRubricError,
    Rubric,
    RubricCriterion,
    build_criteria_guidance,
    build_review_response_model,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.runner import (
    REVIEW_REPORT_FILENAME,
    ReviewReport,
    ReviewRunError,
    TrialReview,
    discover_rollouts,
    run_reviews,
)
from benchflow.review.wrapper import assemble_review_task

__all__ = [
    "DEFAULT_RUBRIC_PATH",
    "REVIEW_RUBRIC_CONTRACT",
    "REVIEW_REPORT_FILENAME",
    "REVIEW_RESULT_FILENAME",
    "REVIEW_RUBRIC_FILENAME",
    "CriterionCheck",
    "ReviewOutcomeValue",
    "ReviewReport",
    "ReviewRubricError",
    "ReviewRunError",
    "Rubric",
    "RubricCriterion",
    "TrialReview",
    "assemble_review_task",
    "build_criteria_guidance",
    "build_review_response_model",
    "discover_rollouts",
    "find_task_rubric",
    "load_rubric",
    "run_reviews",
]

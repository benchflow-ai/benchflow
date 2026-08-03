"""Rubric schema and loading for post-run rubric review.

A review rubric is a small JSON document listing the criteria a reviewer
agent grades a finished rollout against:

.. code-block:: json

    {
      "criteria": [
        {
          "name": "reward_hacking",
          "description": "Author-facing note; never shown to the reviewer.",
          "guidance": "What to inspect and when to answer pass/fail/..."
        }
      ]
    }

Each criterion carries exactly three strings:

- ``name`` — stable identifier for the criterion.  It becomes a field in the
  reviewer's structured-output schema, so it must be a valid Python
  identifier.
- ``description`` — documentation for humans reading the rubric.  It is never
  included in the reviewer prompt; grading behavior must not depend on it.
- ``guidance`` — the grading contract shown to the reviewer.  Put the full
  pass/fail conditions here.

The reviewer answers every criterion with an ``outcome`` of ``pass``,
``fail``, or ``not_applicable`` plus a free-text ``explanation``.  There is
no scoring layer on top: no weights, no thresholds, no aggregation into a
single number.  Consumers read per-criterion outcomes from the review
report.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
)

REVIEW_RUBRIC_FILENAME = "rubric.json"
REVIEW_RESULT_FILENAME = "review-result.json"

DEFAULT_RUBRIC_PATH = Path(__file__).parent / "default-rubric.json"


class ReviewRubricError(ValueError):
    """Raised when a rubric file cannot be loaded or is not a valid rubric."""


class RubricCriterion(BaseModel):
    """One criterion the reviewer grades."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    guidance: str

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError(
                f"criterion name {value!r} must be a valid Python identifier "
                "(it becomes a structured-output field name)"
            )
        return value


class Rubric(BaseModel):
    """A parsed review rubric."""

    model_config = ConfigDict(extra="forbid")

    criteria: list[RubricCriterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _names_are_unique(cls, value: list[RubricCriterion]) -> list[RubricCriterion]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for criterion in value:
            if criterion.name in seen:
                duplicates.add(criterion.name)
            seen.add(criterion.name)
        if duplicates:
            raise ValueError(
                f"criterion names must be unique; duplicated: {sorted(duplicates)} "
                "(duplicate names would silently collapse into one "
                "structured-output field)"
            )
        return value


class ReviewOutcomeValue(StrEnum):
    """Closed outcome vocabulary for one criterion."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CriterionCheck(BaseModel):
    """The reviewer's answer for one criterion."""

    explanation: str
    outcome: ReviewOutcomeValue


def load_rubric(path: Path | None = None) -> Rubric:
    """Load a rubric from a JSON file, or the built-in default rubric.

    Only JSON is accepted; the on-disk shape is
    ``{"criteria": [{"name", "description", "guidance"}, ...]}``.
    """

    rubric_path = path if path is not None else DEFAULT_RUBRIC_PATH
    if rubric_path.suffix.lower() != ".json":
        raise ReviewRubricError(
            f"unsupported rubric format {rubric_path.suffix!r}: rubrics are JSON files"
        )
    try:
        text = rubric_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewRubricError(f"cannot read {rubric_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewRubricError(f"{rubric_path} is not valid JSON: {exc}") from exc
    try:
        return Rubric.model_validate(data)
    except ValidationError as exc:
        raise ReviewRubricError(f"{rubric_path} is not a valid rubric: {exc}") from exc


def is_review_rubric_file(path: Path) -> bool:
    """Whether ``path`` is shaped like a review rubric.

    ``rubric.json`` is an overloaded filename: llm-judge verifier rubrics use
    entries shaped like ``{id, match_criteria}``. A file only counts as a
    review rubric when every criteria entry carries exactly this contract's
    three keys, so judge rubrics are never claimed or misvalidated.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    criteria = data.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return False
    return all(
        isinstance(entry, dict) and {"name", "description", "guidance"} <= set(entry)
        for entry in criteria
    )


def find_task_rubric(task_path: Path) -> Path | None:
    """Return the review rubric a task ships, if any.

    Looks for ``rubric.json`` next to the task's test files (``verifier/`` or
    ``tests/``). Files that exist but are not shaped like a review rubric
    (for example llm-judge rubrics) are left alone.
    """

    for tests_dir_name in ("verifier", "tests"):
        candidate = task_path / tests_dir_name / REVIEW_RUBRIC_FILENAME
        if candidate.is_file() and is_review_rubric_file(candidate):
            return candidate
    return None


def build_criteria_guidance(rubric: Rubric) -> str:
    """Render the criterion guidance lines included in the reviewer prompt."""

    return "\n".join(
        f"- {criterion.name}: {criterion.guidance}" for criterion in rubric.criteria
    )


def build_review_response_model(rubric: Rubric) -> type[BaseModel]:
    """Build the structured-output model for a rollout review.

    The reviewer must return ``{trial_name, summary, checks}`` where
    ``checks`` has one :class:`CriterionCheck` field per rubric criterion.
    """

    checks_fields: dict[str, Any] = {
        criterion.name: (CriterionCheck, ...) for criterion in rubric.criteria
    }
    checks_model = create_model("ReviewChecks", **checks_fields)
    return create_model(
        "ReviewResponse",
        trial_name=(str, ...),
        summary=(str, ...),
        checks=(checks_model, ...),
    )

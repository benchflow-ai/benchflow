"""Rubric schemas and loading for detached post-run review.

The current weighted contract (v0.2) extends each criterion with two strict
integer fields::

    {
      "criteria": [
        {
          "name": "method_soundness",
          "blocker": 0,
          "weight": 5,
          "description": "Author-facing note; never shown to the reviewer.",
          "guidance": "Score 2 when ...; 1 when ...; 0 when ..."
        }
      ]
    }

``blocker: 1`` criteria receive a binary pass/fail judgment. ``blocker: 0``
criteria receive an integer score from 0 to 2 and contribute to a weighted
research-quality score. ``weight`` is an integer from 1 to 10.

Versionless v0.1 rubrics remain supported for compatibility. They omit both
``blocker`` and ``weight`` and retain pass/fail/not-applicable judgments. A
single rubric may not mix contracts, and a criterion may not provide only one
of the v0.2 fields; both cases fail closed instead of guessing intent.
"""

from __future__ import annotations

import json
import keyword
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

LEGACY_REVIEW_RUBRIC_CONTRACT = "v0.1"
WEIGHTED_REVIEW_RUBRIC_CONTRACT = "v0.2"
# Backward-compatible public alias. New code that needs to distinguish
# contracts should use the explicit LEGACY_* and WEIGHTED_* constants.
REVIEW_RUBRIC_CONTRACT = LEGACY_REVIEW_RUBRIC_CONTRACT
REVIEW_RUBRIC_FILENAME = "rubric.json"
REVIEW_RESULT_FILENAME = "review-result.json"

DEFAULT_RUBRIC_PATH = Path(__file__).parent / "default-rubric.json"


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


NonBlankText = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_non_blank),
]


class ReviewRubricError(ValueError):
    """Raised when a rubric file cannot be loaded or is not a valid rubric."""


class RubricCriterion(BaseModel):
    """One criterion the reviewer grades."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    guidance: str
    blocker: int | None = Field(default=None, strict=True, ge=0, le=1)
    weight: int | None = Field(default=None, strict=True, ge=1, le=10)

    @field_validator("name")
    @classmethod
    def _name_is_usable_field(cls, value: str) -> str:
        # The name becomes a dynamically created model field, so anything the
        # schema library reserves either crashes model construction
        # (``model_config``), trips protected-namespace rules
        # (``model_dump``), or is silently dropped from the generated schema
        # (private/dunder names) -- yielding an impossible reviewer/verifier
        # contract instead of a loud failure.
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError(
                f"criterion name {value!r} must be a valid, non-keyword "
                "Python identifier (it becomes a structured-output field)"
            )
        if value.startswith("_"):
            raise ValueError(
                f"criterion name {value!r} must not start with '_': private "
                "and dunder names are dropped from the generated schema"
            )
        if value.startswith("model_"):
            raise ValueError(
                f"criterion name {value!r} must not start with 'model_': "
                "that namespace is reserved by the schema library"
            )
        if hasattr(BaseModel, value):
            raise ValueError(
                f"criterion name {value!r} collides with reserved schema "
                f"attribute {value!r}; choose another name"
            )
        return value

    @model_validator(mode="after")
    def _scoring_fields_are_paired(self) -> Self:
        blocker_supplied = "blocker" in self.model_fields_set
        weight_supplied = "weight" in self.model_fields_set
        if blocker_supplied != weight_supplied:
            raise ValueError(
                "criterion must provide both blocker and weight for the v0.2 "
                "contract, or omit both for legacy v0.1"
            )
        if blocker_supplied and (self.blocker is None or self.weight is None):
            raise ValueError(
                "blocker and weight must be strict integers for the v0.2 "
                "contract; null is not allowed"
            )
        return self

    @property
    def is_legacy(self) -> bool:
        return self.blocker is None

    @property
    def is_blocker(self) -> bool:
        return self.blocker == 1

    def metadata(self) -> dict[str, str | int | None]:
        """Return the non-prompt criterion contract recorded in artifacts."""

        return {
            "name": self.name,
            "blocker": self.blocker,
            "weight": self.weight,
        }


class Rubric(BaseModel):
    """A parsed review rubric."""

    model_config = ConfigDict(extra="forbid")

    criteria: list[RubricCriterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _criteria_are_coherent(
        cls, value: list[RubricCriterion]
    ) -> list[RubricCriterion]:
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
        contracts = {criterion.is_legacy for criterion in value}
        if len(contracts) != 1:
            raise ValueError(
                "all criteria must use one rubric contract; legacy v0.1 and "
                "weighted v0.2 criteria cannot be mixed"
            )
        if not value[0].is_legacy and not any(
            not criterion.is_blocker for criterion in value
        ):
            raise ValueError(
                "weighted v0.2 rubrics require at least one scored "
                "(blocker: 0) criterion"
            )
        return value

    @property
    def contract(self) -> str:
        return (
            LEGACY_REVIEW_RUBRIC_CONTRACT
            if self.criteria[0].is_legacy
            else WEIGHTED_REVIEW_RUBRIC_CONTRACT
        )

    @property
    def is_weighted(self) -> bool:
        return self.contract == WEIGHTED_REVIEW_RUBRIC_CONTRACT

    def metadata(self) -> dict[str, Any]:
        """Return the rubric contract carried into wrapper/report artifacts."""

        return {
            "contract": self.contract,
            "criteria": [criterion.metadata() for criterion in self.criteria],
        }


class ReviewOutcomeValue(StrEnum):
    """Closed outcome vocabulary for one legacy v0.1 criterion."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CriterionCheck(BaseModel):
    """The reviewer's answer for one legacy v0.1 criterion."""

    explanation: NonBlankText
    outcome: ReviewOutcomeValue


class BlockerOutcomeValue(StrEnum):
    """Closed outcome vocabulary for a v0.2 blocker."""

    PASS = "pass"
    FAIL = "fail"


class BlockerCriterionCheck(BaseModel):
    """The reviewer's binary answer for a v0.2 blocker."""

    model_config = ConfigDict(extra="forbid")

    explanation: NonBlankText
    outcome: BlockerOutcomeValue


class ScoredCriterionCheck(BaseModel):
    """The reviewer's 0-2 answer for a weighted v0.2 criterion."""

    model_config = ConfigDict(extra="forbid")

    explanation: NonBlankText
    score: int = Field(strict=True, ge=0, le=2)


def load_rubric(path: Path | None = None) -> Rubric:
    """Load a rubric from a JSON file, or the built-in default rubric.

    Only JSON is accepted. Every criterion must consistently use either the
    legacy v0.1 three-field shape or the weighted v0.2 five-field shape.
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
    """Whether ``path`` claims either detached-review rubric contract.

    ``rubric.json`` is an overloaded filename: llm-judge verifier rubrics
    use entries carrying the full ``{id, match_criteria}`` shape. Only that
    dialect is disclaimed; everything else in this slot is claimed and then
    validated loudly by :func:`load_rubric`, so malformed review rubrics can
    never silently fall back to the built-in default.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
        return True
    criteria = data.get("criteria")
    if not isinstance(criteria, list):
        return True

    def is_judge_entry(entry: object) -> bool:
        return isinstance(entry, dict) and {"id", "match_criteria"} <= set(entry)

    return not (criteria and all(is_judge_entry(entry) for entry in criteria))


def find_task_rubric(task_path: Path) -> Path | None:
    """Return the detached-review rubric a task ships, if any."""

    for tests_dir_name in ("verifier", "tests"):
        candidate = task_path / tests_dir_name / REVIEW_RUBRIC_FILENAME
        if candidate.is_file() and is_review_rubric_file(candidate):
            return candidate
    return None


def build_criteria_guidance(rubric: Rubric) -> str:
    """Render the criterion guidance lines included in the reviewer prompt."""

    if not rubric.is_weighted:
        return "\n".join(
            f"- {criterion.name}: {criterion.guidance}" for criterion in rubric.criteria
        )

    return "\n".join(
        (
            f"- {criterion.name} [BLOCKER; answer pass or fail; weight "
            f"{criterion.weight} does not enter weighted quality]: "
            f"{criterion.guidance}"
            if criterion.is_blocker
            else f"- {criterion.name} [SCORED; weight {criterion.weight}; "
            f"answer 0, 1, or 2]: {criterion.guidance}"
        )
        for criterion in rubric.criteria
    )


def build_review_response_model(rubric: Rubric) -> type[BaseModel]:
    """Build the structured-output model for one rollout review."""

    checks_fields: dict[str, Any] = {}
    for criterion in rubric.criteria:
        check_model: type[BaseModel]
        if criterion.is_legacy:
            check_model = CriterionCheck
        elif criterion.is_blocker:
            check_model = BlockerCriterionCheck
        else:
            check_model = ScoredCriterionCheck
        checks_fields[criterion.name] = (check_model, ...)
    extra_policy = "forbid" if rubric.is_weighted else "ignore"
    checks_model = create_model(
        "ReviewChecks",
        __config__=ConfigDict(extra="forbid"),
        **checks_fields,
    )
    return create_model(
        "ReviewResponse",
        __config__=ConfigDict(extra=extra_policy),
        trial_name=(str, ...),
        summary=(NonBlankText, ...),
        checks=(checks_model, ...),
    )

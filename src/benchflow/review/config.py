"""Strict ``verifier/rubric.json`` schema for agentic planning review.

This module implements the public v1.2 rubric contract.  Rubrics are flat,
binary, and self-contained: every criterion is either met or not met, signed
weights carry polarity, and gates are explicit.  Reviewer execution details
live in one optional ``reviewer`` block and every unknown key fails closed.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow._utils.config import normalize_reasoning_effort

REVIEW_RUBRIC_FILENAME = "rubric.json"
REVIEW_SCHEMA_VERSION = "1.2"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
_TOLERANCE_MARKERS = ("within", "tolerance", "accuracy", "percent", "%")

_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "reviewer", "pass_threshold", "criteria"}
)
_REVIEWER_KEYS = frozenset({"harness", "model", "timeout_sec", "mode"})
_CRITERION_KEYS = frozenset({"id", "criterion", "criterion_type", "weight", "gating"})

CriterionType = Literal[
    "physical-model",
    "approximation",
    "numerical-method",
    "uncertainty",
    "data-handling",
    "failure-check",
]
CRITERION_TYPES: tuple[CriterionType, ...] = (
    "physical-model",
    "approximation",
    "numerical-method",
    "uncertainty",
    "data-handling",
    "failure-check",
)
ReviewMode = Literal["per_criterion", "batched"]
REVIEW_MODES: tuple[ReviewMode, ...] = ("per_criterion", "batched")


class ReviewRubricError(ValueError):
    """Raised when ``rubric.json`` is not a valid review rubric."""


@dataclass(frozen=True)
class ReviewerSpec:
    """Task-level defaults for the independent reviewer runtime."""

    harness: str | None = None
    model: str | None = None
    timeout_sec: float = 1800.0
    mode: ReviewMode = "per_criterion"


@dataclass(frozen=True)
class ReviewCriterion:
    """One atomic binary claim about the solver's plan or trajectory."""

    id: str
    criterion: str
    criterion_type: CriterionType
    weight: float = 1.0
    gating: bool = False

    @property
    def is_metric(self) -> bool:
        return not self.gating and self.weight == 0.0


@dataclass(frozen=True)
class ReviewRubric:
    """Fully parsed and validated ``rubric.json``."""

    criteria: tuple[ReviewCriterion, ...]
    reviewer: ReviewerSpec = field(default_factory=ReviewerSpec)
    pass_threshold: float = 0.7
    source_path: Path | None = None

    @property
    def gates(self) -> tuple[ReviewCriterion, ...]:
        return tuple(criterion for criterion in self.criteria if criterion.gating)

    @property
    def scored(self) -> tuple[ReviewCriterion, ...]:
        return tuple(
            criterion
            for criterion in self.criteria
            if not criterion.gating and criterion.weight != 0.0
        )

    @property
    def metrics(self) -> tuple[ReviewCriterion, ...]:
        return tuple(criterion for criterion in self.criteria if criterion.is_metric)


@dataclass
class ReviewParams:
    """Run-level reviewer overrides from the CLI or evaluation config."""

    enabled: bool | None = None
    harness: str | None = None
    model: str | None = None
    timeout_sec: float | None = None
    mode: ReviewMode | None = None
    reasoning_effort: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "harness": self.harness,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "mode": self.mode,
            "reasoning_effort": self.reasoning_effort,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReviewParams:
        allowed = {
            "enabled",
            "harness",
            "model",
            "timeout_sec",
            "mode",
            "reasoning_effort",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"review has unknown key(s): {unknown}")
        mode = value.get("mode")
        if mode is not None and mode not in REVIEW_MODES:
            raise ValueError(
                f"review.mode must be one of {list(REVIEW_MODES)}, got {mode!r}"
            )
        timeout = value.get("timeout_sec")
        if timeout is not None:
            timeout = _require_finite_number(timeout, "review.timeout_sec")
            if timeout <= 0:
                raise ValueError("review.timeout_sec must be > 0")
        return cls(
            enabled=value.get("enabled"),
            harness=value.get("harness"),
            model=value.get("model"),
            timeout_sec=timeout,
            mode=mode,
            reasoning_effort=normalize_reasoning_effort(
                value.get("reasoning_effort")
            ),
        )


def is_review_rubric_file(path: Path) -> bool:
    """Return whether ``path`` claims the versioned review-rubric format."""

    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "schema_version" in data


def _reject_nonfinite(constant: str) -> float:
    raise ReviewRubricError(
        f"rubric.json contains the non-finite JSON constant {constant!r}; "
        "weights and thresholds must be finite numbers"
    )


def _require_known_keys(
    mapping: dict[str, Any], allowed: frozenset[str], context: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ReviewRubricError(
            f"{context} has unknown key(s) {unknown}; allowed keys are "
            f"{sorted(allowed)}. Unknown keys are rejected so a typo cannot "
            "silently change scoring."
        )


def _require_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewRubricError(f"{context} must be a non-empty string")
    return value.strip()


def _require_finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewRubricError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ReviewRubricError(f"{context} must be finite")
    return number


def _require_registered_harness(harness: str, context: str) -> None:
    from benchflow.agents.registry import resolve_agent

    try:
        resolve_agent(harness)
    except KeyError as exc:
        raise ReviewRubricError(f"{context}: {exc.args[0]}") from exc


def _parse_reviewer(raw: Any, *, context: str = "rubric.reviewer") -> ReviewerSpec:
    if raw is None:
        return ReviewerSpec()
    if not isinstance(raw, dict):
        raise ReviewRubricError(f"{context} must be an object")
    _require_known_keys(raw, _REVIEWER_KEYS, context)

    harness = raw.get("harness")
    if harness is not None:
        harness = _require_str(harness, f"{context}.harness")
        _require_registered_harness(harness, f"{context}.harness")
    model = raw.get("model")
    if model is not None:
        model = _require_str(model, f"{context}.model")

    timeout_sec = _require_finite_number(
        raw.get("timeout_sec", 1800.0), f"{context}.timeout_sec"
    )
    if timeout_sec <= 0:
        raise ReviewRubricError(f"{context}.timeout_sec must be > 0")

    mode = raw.get("mode", "per_criterion")
    if mode not in REVIEW_MODES:
        raise ReviewRubricError(
            f"{context}.mode must be one of {list(REVIEW_MODES)}, got {mode!r}"
        )
    return ReviewerSpec(
        harness=harness,
        model=model,
        timeout_sec=timeout_sec,
        mode=mode,
    )


def _parse_criterion(raw: Any, index: int) -> ReviewCriterion:
    context = f"rubric.criteria[{index}]"
    if not isinstance(raw, dict):
        raise ReviewRubricError(f"{context} must be an object")
    _require_known_keys(raw, _CRITERION_KEYS, context)

    criterion_id = _require_str(raw.get("id"), f"{context}.id")
    if not _ID_RE.fullmatch(criterion_id):
        raise ReviewRubricError(
            f"{context}.id {criterion_id!r} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$"
        )
    criterion = _require_str(raw.get("criterion"), f"{context}.criterion")
    if len(criterion) < 20:
        raise ReviewRubricError(
            f"{context}.criterion must contain at least 20 characters"
        )

    criterion_type = raw.get("criterion_type")
    if criterion_type not in CRITERION_TYPES:
        raise ReviewRubricError(
            f"{context}.criterion_type must be one of {list(CRITERION_TYPES)}, "
            f"got {criterion_type!r}"
        )

    gating = raw.get("gating", False)
    if not isinstance(gating, bool):
        raise ReviewRubricError(f"{context}.gating must be a boolean")
    if gating and "weight" in raw:
        raise ReviewRubricError(
            f"{context}: gating=true forbids weight because gates are excluded "
            "from the weighted mean"
        )
    weight = _require_finite_number(raw.get("weight", 1.0), f"{context}.weight")
    return ReviewCriterion(
        id=criterion_id,
        criterion=criterion,
        criterion_type=criterion_type,
        weight=0.0 if gating else weight,
        gating=gating,
    )


def _numeric_literals(text: str) -> set[str]:
    return {match.group(0) for match in _NUMBER_RE.finditer(text)}


def _criterion_answer_literals(text: str) -> set[str]:
    literals: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        context = text[max(0, match.start() - 32) : match.end() + 32].lower()
        if not any(marker in context for marker in _TOLERANCE_MARKERS):
            literals.add(match.group(0))
    return literals


def _lint_answer_leaks(path: Path, criteria: tuple[ReviewCriterion, ...]) -> None:
    test_outputs = path.parent / "test_outputs.py"
    if not test_outputs.is_file():
        return
    try:
        verifier_literals = _numeric_literals(test_outputs.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReviewRubricError(f"cannot inspect {test_outputs}: {exc}") from exc
    for criterion in criteria:
        leaked = sorted(
            _criterion_answer_literals(criterion.criterion) & verifier_literals
        )
        if leaked:
            raise ReviewRubricError(
                f"rubric.criteria[{criterion.id!r}].criterion contains numeric "
                f"literal(s) also present in test_outputs.py: {leaked}; rubric "
                "criteria must not expose expected answers"
            )


def load_review_rubric(path: Path) -> ReviewRubric:
    """Load and strictly validate a versioned review rubric."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewRubricError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ReviewRubricError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewRubricError(f"{path} must contain a JSON object")

    schema_version = data.get("schema_version")
    if schema_version != REVIEW_SCHEMA_VERSION:
        raise ReviewRubricError(
            f"{path} schema_version must be {REVIEW_SCHEMA_VERSION!r}, got "
            f"{schema_version!r}. A rubric.json without schema_version is an "
            "llm-judge rubric, not a review rubric."
        )
    _require_known_keys(data, _TOP_LEVEL_KEYS, "rubric.json")

    reviewer = _parse_reviewer(data.get("reviewer"))
    pass_threshold = _require_finite_number(
        data.get("pass_threshold", 0.7), "rubric.pass_threshold"
    )
    if not 0.0 <= pass_threshold <= 1.0:
        raise ReviewRubricError("rubric.pass_threshold must be within [0, 1]")

    criteria_raw = data.get("criteria")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise ReviewRubricError("rubric.criteria must be a non-empty list")
    criteria = tuple(
        _parse_criterion(raw, index) for index, raw in enumerate(criteria_raw)
    )
    ids = [criterion.id for criterion in criteria]
    duplicates = sorted(
        {criterion_id for criterion_id in ids if ids.count(criterion_id) > 1}
    )
    if duplicates:
        raise ReviewRubricError(f"rubric.criteria has duplicate id(s): {duplicates}")

    positive_weight = sum(
        criterion.weight
        for criterion in criteria
        if not criterion.gating and criterion.weight > 0
    )
    if positive_weight <= 0:
        raise ReviewRubricError(
            "rubric cannot produce a score: positive non-gating weights must sum to > 0"
        )
    _lint_answer_leaks(path, criteria)
    if not any(criterion.gating for criterion in criteria):
        warnings.warn(
            f"{path}: review rubric has no gating criterion",
            UserWarning,
            stacklevel=2,
        )
    return ReviewRubric(
        criteria=criteria,
        reviewer=reviewer,
        pass_threshold=pass_threshold,
        source_path=path,
    )


def find_review_rubric(verifier_dir: Path) -> Path | None:
    """Return the versioned review rubric under ``verifier_dir``, if present."""

    candidate = verifier_dir / REVIEW_RUBRIC_FILENAME
    return candidate if is_review_rubric_file(candidate) else None

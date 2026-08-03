"""Declarative ``rubric.json`` schema for the post-verify rubric review.

A rubric review grades a completed rollout with an *agentic* reviewer: a
registered ACP agent (any harness in :mod:`benchflow.agents.registry`) that
explores the agent's workspace and the captured trajectory files, then answers
each rubric criterion with a verdict from that criterion's ordered ``choices``.

The rubric is **JSON only** and lives next to the verifier
(``verifier/rubric.json`` in native packages, ``tests/rubric.json`` in legacy
packages). It is deliberately a different document from the ``llm-judge``
strategy's ``RubricConfig`` (:mod:`benchflow.rewards.rubric_config`): that one
grades deliverable *text* with a single chat call; this one drives a real agent
over the whole rollout. A review rubric is recognized by its
``"schema_version": "1.0"`` key so the two file families cannot be confused.

Design lineage (per-field provenance, so future edits keep the semantics):

- ``choices`` ordered worst→best with rank scoring — PrimeIntellect verifiers
  v1 ``RubricJudge``; one mechanism covers binary and graded criteria and an
  out-of-range score is unrepresentable.
- signed ``weight`` with a positive-only denominator and a final clamp —
  HealthBench; penalties can only erode earned credit. ``weight: 0`` records a
  criterion as a metric that cannot move the reward (verifiers v1).
- ``required`` — a criterion that must score 1.0 or the whole review is 0.0.
- ``guidance`` — extra reviewer instructions per criterion (Terminal-Bench
  Science's ``[[criteria]].guidance``).
- unknown keys are rejected everywhere — Harbor RewardKit silently ignores
  typo'd keys, which makes a misconfigured rubric look like a working one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REVIEW_RUBRIC_FILENAME = "rubric.json"
REVIEW_SCHEMA_VERSION = "1.0"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "reviewer", "pass_threshold", "criteria"}
)
_REVIEWER_KEYS = frozenset({"agent", "model", "timeout", "mode"})
_CRITERION_KEYS = frozenset(
    {"id", "criterion", "guidance", "choices", "weight", "required", "tags"}
)

ReviewMode = Literal["batched", "individual"]


class ReviewRubricError(ValueError):
    """Raised when ``rubric.json`` is not a valid review rubric."""


@dataclass(frozen=True)
class ReviewerSpec:
    """The ``reviewer`` object: which agent reviews, and how.

    ``agent``/``model`` are *defaults* — CLI / config parameters take
    precedence (see :class:`ReviewParams`). ``agent`` must be a registered
    benchflow agent name; ``model`` falls back to the agent's registry default.
    """

    agent: str | None = None
    model: str | None = None
    timeout: float = 900.0
    mode: ReviewMode = "batched"


@dataclass(frozen=True)
class ReviewCriterion:
    """One atomic, independently checkable claim about the rollout."""

    id: str
    criterion: str
    guidance: str | None = None
    # Ordered worst → best. The verdict's rank index normalizes to [0, 1]:
    # choices[0] -> 0.0, choices[-1] -> 1.0, the rest evenly spaced.
    choices: tuple[str, ...] = ("no", "yes")
    # Signed. Positive earns credit; negative subtracts (and never enters the
    # denominator); zero records the criterion as a metric only.
    weight: float = 1.0
    # A gate must score exactly 1.0 (the best choice) or the review is 0.0.
    # Gates are excluded from the weighted mean, so ``weight`` is forbidden.
    required: bool = False
    tags: tuple[str, ...] = ()

    @property
    def is_metric(self) -> bool:
        return not self.required and self.weight == 0.0

    def choice_score(self, verdict: str) -> float:
        """Rank-normalize ``verdict`` (a member of ``choices``) to [0, 1]."""
        return self.choices.index(verdict) / (len(self.choices) - 1)


@dataclass(frozen=True)
class ReviewRubric:
    """Fully parsed and validated ``rubric.json``."""

    criteria: tuple[ReviewCriterion, ...]
    reviewer: ReviewerSpec = field(default_factory=ReviewerSpec)
    pass_threshold: float = 0.7
    source_path: Path | None = None

    @property
    def gates(self) -> tuple[ReviewCriterion, ...]:
        return tuple(c for c in self.criteria if c.required)

    @property
    def scored(self) -> tuple[ReviewCriterion, ...]:
        """Non-required criteria that can move the review score."""
        return tuple(c for c in self.criteria if not c.required and c.weight != 0.0)

    @property
    def metrics(self) -> tuple[ReviewCriterion, ...]:
        return tuple(c for c in self.criteria if c.is_metric)


@dataclass
class ReviewParams:
    """Caller-side review parameters (CLI flags / ``EvaluationConfig``).

    ``enabled=None`` means *auto*: review runs iff the task ships a review
    rubric. ``True`` requires one (a missing/invalid rubric is a review config
    error); ``False`` skips review even when a rubric exists. ``agent`` /
    ``model`` override the rubric's ``reviewer`` defaults.
    """

    enabled: bool | None = None
    agent: str | None = None
    model: str | None = None


def is_review_rubric_file(path: Path) -> bool:
    """Cheap claim check: does ``path`` look like a *review* rubric?

    ``rubric.json`` is also a legal filename for the ``llm-judge`` strategy's
    Harvey-LAB-style rubric, which has no ``schema_version``. Auto-discovery
    must not swallow those files, so a review rubric is claimed only by the
    ``schema_version`` marker.
    """
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "schema_version" in data


def _reject_nonfinite(constant: str) -> float:
    # json.loads accepts NaN/Infinity as a nonstandard extension; a rubric
    # carrying them would corrupt the weighted mean, so fail loudly.
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
    return value


def _require_finite_number(value: Any, context: str) -> float:
    # bool is an int subclass; a bare true/false weight is a rubric bug.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewRubricError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ReviewRubricError(f"{context} must be finite")
    return number


def _parse_reviewer(raw: Any, *, context: str = "rubric.reviewer") -> ReviewerSpec:
    if raw is None:
        return ReviewerSpec()
    if not isinstance(raw, dict):
        raise ReviewRubricError(f"{context} must be an object")
    _require_known_keys(raw, _REVIEWER_KEYS, context)

    agent = raw.get("agent")
    if agent is not None:
        agent = _require_str(agent, f"{context}.agent")
        _require_registered_agent(agent, f"{context}.agent")
    model = raw.get("model")
    if model is not None:
        model = _require_str(model, f"{context}.model")

    timeout = raw.get("timeout", 900.0)
    timeout = _require_finite_number(timeout, f"{context}.timeout")
    if timeout <= 0:
        raise ReviewRubricError(f"{context}.timeout must be > 0")

    mode = raw.get("mode", "batched")
    if mode not in ("batched", "individual"):
        raise ReviewRubricError(
            f"{context}.mode must be 'batched' or 'individual', got {mode!r}"
        )

    return ReviewerSpec(
        agent=agent, model=model, timeout=timeout, mode=mode
    )


def _require_registered_agent(agent: str, context: str) -> None:
    from benchflow.agents.registry import resolve_agent

    try:
        resolve_agent(agent)
    except KeyError as e:
        # resolve_agent's KeyError already carries close-match suggestions.
        raise ReviewRubricError(f"{context}: {e.args[0]}") from e


def _parse_criterion(raw: Any, index: int) -> ReviewCriterion:
    context = f"rubric.criteria[{index}]"
    if not isinstance(raw, dict):
        raise ReviewRubricError(f"{context} must be an object")
    _require_known_keys(raw, _CRITERION_KEYS, context)

    criterion_id = _require_str(raw.get("id"), f"{context}.id")
    if not _ID_RE.match(criterion_id):
        raise ReviewRubricError(
            f"{context}.id {criterion_id!r} must match "
            "^[a-z0-9][a-z0-9_-]{0,63}$ (stable ids survive rewording and key "
            "the per-criterion reward metrics)"
        )
    criterion = _require_str(raw.get("criterion"), f"{context}.criterion")

    guidance = raw.get("guidance")
    if guidance is not None:
        guidance = _require_str(guidance, f"{context}.guidance")

    choices_raw = raw.get("choices", ["no", "yes"])
    if (
        not isinstance(choices_raw, list)
        or len(choices_raw) < 2
        or not all(isinstance(c, str) and c.strip() for c in choices_raw)
    ):
        raise ReviewRubricError(
            f"{context}.choices must be a list of at least two non-empty "
            "strings, ordered worst to best"
        )
    if len(set(choices_raw)) != len(choices_raw):
        raise ReviewRubricError(f"{context}.choices must not contain duplicates")

    required = raw.get("required", False)
    if not isinstance(required, bool):
        raise ReviewRubricError(f"{context}.required must be a boolean")

    if required and "weight" in raw:
        raise ReviewRubricError(
            f"{context}: a required criterion must not declare 'weight' — gates "
            "are excluded from the weighted mean, so the key would be dead "
            "config. Remove it (or drop 'required')."
        )
    weight = _require_finite_number(raw.get("weight", 1.0), f"{context}.weight")

    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or not all(
        isinstance(t, str) and t.strip() for t in tags_raw
    ):
        raise ReviewRubricError(f"{context}.tags must be a list of strings")

    return ReviewCriterion(
        id=criterion_id,
        criterion=criterion,
        guidance=guidance,
        choices=tuple(choices_raw),
        weight=weight if not required else 0.0,
        required=required,
        tags=tuple(tags_raw),
    )


def load_review_rubric(path: Path) -> ReviewRubric:
    """Load and strictly validate a review ``rubric.json``.

    Every failure raises :class:`ReviewRubricError` with an operator-facing
    message; nothing is silently defaulted or ignored.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReviewRubricError(f"cannot read {path}: {e}") from e
    try:
        data = json.loads(text, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as e:
        raise ReviewRubricError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ReviewRubricError(f"{path} must contain a JSON object")

    schema_version = data.get("schema_version")
    if schema_version != REVIEW_SCHEMA_VERSION:
        raise ReviewRubricError(
            f"{path} schema_version must be {REVIEW_SCHEMA_VERSION!r}, got "
            f"{schema_version!r}. (A rubric.json without schema_version is an "
            "llm-judge rubric, not a review rubric.)"
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

    seen: set[str] = set()
    for criterion in criteria:
        if criterion.id in seen:
            raise ReviewRubricError(
                f"rubric.criteria has duplicate id {criterion.id!r}"
            )
        seen.add(criterion.id)

    rubric = ReviewRubric(
        criteria=criteria,
        reviewer=reviewer,
        pass_threshold=pass_threshold,
        source_path=path,
    )

    positive_weight = sum(c.weight for c in rubric.scored if c.weight > 0)
    if positive_weight <= 0 and not rubric.gates:
        raise ReviewRubricError(
            "rubric cannot produce a score: it has no required criteria and no "
            "positively weighted criteria (metrics and penalties alone have "
            "nothing to erode)"
        )
    return rubric


def find_review_rubric(verifier_dir: Path) -> Path | None:
    """Return the review rubric path under ``verifier_dir`` when one exists.

    Only files claimed by :func:`is_review_rubric_file` are returned, so an
    llm-judge ``rubric.json`` in the same directory never triggers a review.
    """
    candidate = verifier_dir / REVIEW_RUBRIC_FILENAME
    if is_review_rubric_file(candidate):
        return candidate
    return None

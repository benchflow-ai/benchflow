"""Pydantic schema models for ``evals.json``.

These models validate ``evals.json`` at the input boundary BEFORE any
TOML / Python / Dockerfile generation runs. Without this, a malformed
field (e.g. a non-numeric ``timeout_sec`` or a ``judge_model`` value
with quotes/newlines) would silently flow into generated artifacts and
surface as a confusing downstream parse error (invalid TOML,
``SyntaxError`` in generated judge.py, etc.). Issue #424 documents that
production footgun.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Conservative model-id alphabet — every real provider/model identifier
# we ship fits this shape. Generated ``judge.py`` interpolates
# ``judge_model`` into a Python string literal; restricting the alphabet
# keeps a hostile value from breaking the generated source.
DEFAULT_SKILL_MOUNT_DIR = "/skills"
_SAFE_JUDGE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]*$")

# ``strict=True`` blocks pydantic's default permissive coercion, so a
# string like ``"120"`` is rejected for an int field rather than silently
# coerced — values that flow into ``task.toml`` must match their declared
# types exactly.
_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", strict=True)


class _EvalCaseModel(BaseModel):
    """Schema for a single case in ``evals.json``.

    Extra fields are rejected so typos like ``expecte_behavior`` fail fast
    instead of being silently dropped. Field shapes match ``EvalCase``.
    """

    model_config = _STRICT_MODEL_CONFIG

    id: str | None = None
    question: str
    ground_truth: str = ""
    expected_behavior: list[str] = Field(default_factory=list)
    expected_skill: str = ""
    expected_script: str = ""
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_explicit_null_id(self) -> _EvalCaseModel:
        # An explicit ``"id": null`` is almost certainly a mistake — the
        # implicit ``case-NNN`` fallback only applies when ``id`` is
        # omitted entirely.
        if self.id is None and "id" in self.model_fields_set:
            raise ValueError("id must be a string when provided")
        return self

    @field_validator("expected_behavior", mode="before")
    @classmethod
    def _reject_string_expected_behavior(cls, v: Any) -> Any:
        if isinstance(v, str):
            raise ValueError(
                "expected_behavior must be a list of strings, got a single string. "
                'Wrap it in a list: ["<rubric item>"].'
            )
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def _reject_non_string_env_values(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError("environment must be a mapping of str -> str")
        bad_keys = [k for k in v if not isinstance(k, str)]
        if bad_keys:
            raise ValueError("environment keys must be strings")
        bad_values = [k for k, val in v.items() if not isinstance(val, str)]
        if bad_values:
            raise ValueError(
                "environment values must be strings; "
                f"non-string entries: {sorted(bad_values)}"
            )
        return v


class _EvalDefaultsModel(BaseModel):
    """Schema for ``defaults`` block in ``evals.json``."""

    model_config = _STRICT_MODEL_CONFIG

    timeout_sec: int = 300
    judge_model: str = "gemini-3.1-flash-lite"
    skill_mount_dir: str = DEFAULT_SKILL_MOUNT_DIR

    @field_validator("timeout_sec")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"timeout_sec must be a positive int, got {v!r}")
        return v

    @field_validator("judge_model")
    @classmethod
    def _safe_judge_model(cls, v: str) -> str:
        if not _SAFE_JUDGE_MODEL_RE.fullmatch(v):
            raise ValueError(
                f"judge_model {v!r} contains unsafe characters; "
                "only alphanumerics, '.', '_', ':', '/', '-' are allowed"
            )
        return v


class _EvalsJsonModel(BaseModel):
    """Top-level schema for ``evals.json``."""

    model_config = _STRICT_MODEL_CONFIG

    version: str = "1"
    skill_name: str = ""
    defaults: _EvalDefaultsModel = Field(default_factory=_EvalDefaultsModel)
    cases: list[_EvalCaseModel]

    @field_validator("cases")
    @classmethod
    def _non_empty_cases(cls, v: list[_EvalCaseModel]) -> list[_EvalCaseModel]:
        if not v:
            raise ValueError("evals.json 'cases' array is empty")
        return v


def validate_evals_json(data: object) -> _EvalsJsonModel:
    """Validate raw ``evals.json`` data and return the parsed schema model.

    Wraps :class:`_EvalsJsonModel.model_validate` so callers get one
    consistent ``ValueError`` (including the historical
    ``'cases' array`` text) instead of either a bare ``KeyError`` for the
    missing top-level key or pydantic's raw ``ValidationError``.
    """
    if not isinstance(data, dict):
        raise ValueError("evals.json must contain an object")
    if "cases" not in data:
        raise ValueError("evals.json must contain a 'cases' array")
    try:
        return _EvalsJsonModel.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"evals.json failed schema validation: {e}") from e

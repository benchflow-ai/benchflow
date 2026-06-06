"""Typed validation for the ``benchflow:`` document namespace (P3 subset)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchflow.task.prompt_composition import (
    PromptCompositionSettings,
    prompt_composition_settings,
)

CompositionMode = Literal["append", "replace"]
PromptPart = Literal["base", "role", "scene", "turn"]
MetadataOnlyRuntime = Literal["metadata-only", "metadata_only"]


class BenchflowPromptSection(BaseModel):
    """``benchflow.prompt`` composition settings."""

    model_config = ConfigDict(extra="forbid")

    composition: CompositionMode | None = None
    order: list[PromptPart] | None = None


class BenchflowNudgesSection(BaseModel):
    """``benchflow.nudges`` simulated-user policy."""

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    branchable: bool | None = None
    nudge_budget: int | None = Field(default=None, ge=1)
    confirmation_policy: dict[str, Any] | None = None
    runtime: MetadataOnlyRuntime | str | None = None


class BenchflowCompatibilitySection(BaseModel):
    """``benchflow.compatibility`` export target metadata."""

    model_config = ConfigDict(extra="allow")

    target: str | None = None
    mode: str | None = None
    extra: list[str] | None = None


class BenchflowVerifierSection(BaseModel):
    """``benchflow.verifier`` native verifier package references."""

    model_config = ConfigDict(extra="allow")

    spec: str | None = None
    rubric: str | None = None
    structured_rubric: str | None = None
    entrypoint: str | None = None
    reward_kit: str | None = None


class BenchflowMetadata(BaseModel):
    """Typed subset of ``benchflow:`` frontmatter."""

    model_config = ConfigDict(extra="allow")

    document_version: str | None = None
    prompt: BenchflowPromptSection | None = None
    nudges: BenchflowNudgesSection | None = None
    compatibility: BenchflowCompatibilitySection | None = None
    verifier: BenchflowVerifierSection | None = None
    user_runtime: MetadataOnlyRuntime | str | None = None


def validate_benchflow_metadata(raw: Any) -> list[str]:
    """Validate a ``benchflow`` block and return human-readable issues."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["benchflow must be a mapping"]

    issues: list[str] = []
    try:
        BenchflowMetadata.model_validate(raw)
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            prefix = f"benchflow.{loc}" if loc else "benchflow"
            issues.append(f"{prefix}: {error.get('msg', 'invalid value')}")

    try:
        prompt_composition_settings(raw)
    except ValueError as exc:
        issues.append(str(exc))

    return issues


def parse_benchflow_metadata(raw: Any) -> BenchflowMetadata | None:
    """Parse ``benchflow`` frontmatter when validation succeeds."""
    if not isinstance(raw, dict):
        return None
    if validate_benchflow_metadata(raw):
        return None
    return BenchflowMetadata.model_validate(raw)


def prompt_settings_from_metadata(
    metadata: BenchflowMetadata | None,
    *,
    raw: dict[str, Any] | None = None,
) -> PromptCompositionSettings:
    """Resolve prompt composition from typed metadata or raw benchflow."""
    if raw is not None:
        return prompt_composition_settings(raw)
    if metadata is None or metadata.prompt is None:
        return PromptCompositionSettings()
    section = metadata.prompt
    return PromptCompositionSettings(
        composition=section.composition,
        order=tuple(section.order) if section.order is not None else PromptCompositionSettings().order,
    )


__all__ = [
    "BenchflowCompatibilitySection",
    "BenchflowMetadata",
    "BenchflowNudgesSection",
    "BenchflowPromptSection",
    "BenchflowVerifierSection",
    "parse_benchflow_metadata",
    "prompt_settings_from_metadata",
    "validate_benchflow_metadata",
]

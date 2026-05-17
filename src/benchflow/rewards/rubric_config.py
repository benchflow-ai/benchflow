"""Declarative rubric configuration parsed from ``rubric.toml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]  # ty: ignore[unresolved-import]


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class Criterion:
    """A single evaluation criterion."""

    description: str
    type: Literal["binary", "likert", "numeric"] = "binary"
    name: str | None = None
    points: int = 5
    min: float = 0.0
    max: float = 100.0
    weight: float = 1.0
    files: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.name or self.description[:40]

    def normalize(self, raw: float) -> float:
        """Normalize a raw score to [0, 1]."""
        if self.type == "binary":
            return 1.0 if raw >= 0.5 else 0.0
        if self.type == "likert":
            denom = self.points - 1
            return float((raw - 1) / denom) if denom > 0 else 0.0
        # numeric
        span = self.max - self.min
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (raw - self.min) / span))


@dataclass
class ScoringConfig:
    """How criteria scores are aggregated."""

    aggregation: Literal[
        "weighted_mean", "all_pass", "any_pass", "threshold"
    ] = "weighted_mean"
    threshold: float = 0.7


@dataclass
class JudgeConfig:
    """The ``[judge]`` section of a rubric."""

    model: str = "claude-sonnet-4-6"
    mode: Literal["batched", "individual"] = "individual"
    files: list[str] = field(default_factory=list)
    timeout: int = 120
    reference: str | None = None
    prompt_template: str | None = None


@dataclass
class RubricConfig:
    """Fully parsed ``rubric.toml``."""

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    criteria: list[Criterion] = field(default_factory=list)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def _parse_criterion(raw: dict) -> Criterion:
    return Criterion(
        description=raw.get("description", ""),
        type=raw.get("type", "binary"),
        name=raw.get("name") or raw.get("id"),
        points=raw.get("points", 5),
        min=raw.get("min", 0.0),
        max=raw.get("max", 100.0),
        weight=raw.get("weight", 1.0),
        files=raw.get("files", []),
    )


def _parse_judge(raw: dict) -> JudgeConfig:
    return JudgeConfig(
        model=raw.get("model", "claude-sonnet-4-6"),
        mode=raw.get("mode", "individual"),
        files=raw.get("files", []),
        timeout=raw.get("timeout", 120),
        reference=raw.get("reference"),
        prompt_template=raw.get("prompt_template"),
    )


def _parse_scoring(raw: dict) -> ScoringConfig:
    return ScoringConfig(
        aggregation=raw.get("aggregation", "weighted_mean"),
        threshold=raw.get("threshold", 0.7),
    )


def load_rubric_toml(path: Path) -> RubricConfig:
    """Load and parse a ``rubric.toml`` file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    judge = _parse_judge(data.get("judge", {}))
    criteria = [_parse_criterion(c) for c in data.get("criterion", [])]
    scoring = _parse_scoring(data.get("scoring", {}))

    return RubricConfig(judge=judge, criteria=criteria, scoring=scoring)

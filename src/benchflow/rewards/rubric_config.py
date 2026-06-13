"""Declarative rubric configuration parsed from ``rubric.toml`` / ``rubric.json``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from benchflow.rewards.events import Space

# The valid evaluation spaces a criterion may declare. Mirrors
# ``benchflow.rewards.events.Space`` so rubrics can tag a criterion as
# ``"action"``, ``"reasoning"``, ``"memory"``, or ``"latent"`` instead of
# letting it default to ``"output"`` — the architecture's outcome space.
_VALID_SPACES: frozenset[str] = frozenset(
    {"output", "action", "reasoning", "memory", "latent"}
)
_VALID_REWARD_KIT_AGGREGATIONS = {
    "weighted_mean",
    "weighted_sum",
    "all_pass",
    "any_pass",
    "threshold",
}


def _coerce_space(raw: object) -> Space:
    """Validate a raw ``space`` value from a rubric file.

    Falls back to ``"output"`` when the field is absent. A present-but-invalid
    value is rejected loudly — silently downgrading to ``"output"`` would
    hide a misconfigured rubric and re-introduce the very mistag this
    contract is meant to prevent.
    """
    if raw is None:
        return "output"
    if isinstance(raw, str) and raw in _VALID_SPACES:
        return cast(Space, raw)
    raise ValueError(
        f"Rubric criterion 'space' must be one of {sorted(_VALID_SPACES)}; got {raw!r}"
    )


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
    # Evaluation space this criterion scores — propagated to every dense
    # ``RewardEvent`` it emits. Defaults to ``"output"`` (the architecture's
    # outcome space) so existing rubrics keep their current behaviour; tag a
    # process-like criterion as ``"action"`` / ``"reasoning"`` / ``"memory"``
    # to keep dense events from being mistaken for terminal outcome rewards.
    space: Space = "output"

    @property
    def id(self) -> str:
        return self.name or self.description[:40]

    def normalize(self, raw: float) -> float:
        """Normalize a raw score to [0, 1]."""
        if self.type == "binary":
            return 1.0 if raw >= 0.5 else 0.0
        if self.type == "likert":
            denom = self.points - 1
            if denom <= 0:
                return 0.0
            return max(0.0, min(1.0, (raw - 1) / denom))
        # numeric
        span = self.max - self.min
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (raw - self.min) / span))


@dataclass
class ScoringConfig:
    """How criteria scores are aggregated."""

    aggregation: Literal["weighted_mean", "all_pass", "any_pass", "threshold"] = (
        "weighted_mean"
    )
    threshold: float = 0.7


@dataclass
class JudgeConfig:
    """The ``[judge]`` section of a rubric."""

    model: str = "claude-sonnet-4-6"
    mode: Literal["batched", "individual"] = "individual"
    files: list[str] = field(default_factory=list)
    timeout: int = 120


@dataclass
class RubricConfig:
    """Fully parsed ``rubric.toml``."""

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    criteria: list[Criterion] = field(default_factory=list)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def _parse_criterion(raw: dict) -> Criterion:
    return Criterion(
        description=raw.get("description") or raw.get("match_criteria", ""),
        type=raw.get("type", "binary"),
        name=raw.get("name") or raw.get("id") or raw.get("title"),
        points=raw.get("points", 5),
        min=raw.get("min", 0.0),
        max=raw.get("max", 100.0),
        weight=raw.get("weight", 1.0),
        files=raw.get("files", []),
        space=_coerce_space(raw.get("space")),
    )


def _parse_judge(raw: dict) -> JudgeConfig:
    return JudgeConfig(
        model=raw.get("model", "claude-sonnet-4-6"),
        mode=raw.get("mode", "individual"),
        files=raw.get("files", []),
        timeout=raw.get("timeout", 120),
    )


def _parse_scoring(raw: dict) -> ScoringConfig:
    aggregation = raw.get("aggregation", raw.get("method", "weighted_mean"))
    if aggregation == "mean":
        aggregation = "weighted_mean"
    return ScoringConfig(
        aggregation=aggregation,
        threshold=raw.get("threshold", 0.7),
    )


def load_rubric_toml(path: Path) -> RubricConfig:
    """Load and parse a ``rubric.toml`` file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    judge = _parse_judge(data.get("judge", {}))
    raw_criteria = data.get("criterion", data.get("criteria", []))
    criteria = [_parse_criterion(c) for c in raw_criteria]
    scoring = _parse_scoring(data.get("scoring", {}))

    return RubricConfig(judge=judge, criteria=criteria, scoring=scoring)


def load_rubric_json(path: Path) -> RubricConfig:
    """Load and parse a ``rubric.json`` file (Harvey LAB style).

    The JSON schema::

        {
          "title": "Task Title",
          "criteria": [
            {"id": "c-1", "title": "...", "match_criteria": "..."}
          ]
        }

    Optional ``[judge]`` and ``[scoring]`` objects are also honoured so a
    JSON rubric can be fully self-describing.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    judge = _parse_judge(data.get("judge", {}))
    scoring = _parse_scoring(data.get("scoring", {}))
    criteria: list[Criterion] = []
    for idx, raw in enumerate(data.get("criteria", [])):
        description = (
            raw.get("match_criteria") or raw.get("description") or raw.get("title")
        )
        if not description:
            raise ValueError(
                f"Rubric criterion #{idx} in {path} has no description: set one "
                f"of 'match_criteria', 'description', or 'title'."
            )
        criteria.append(
            Criterion(
                description=description,
                type=raw.get("type", "binary"),
                name=raw.get("id") or raw.get("name") or raw.get("title"),
                points=raw.get("points", 5),
                min=raw.get("min", 0.0),
                max=raw.get("max", 100.0),
                weight=raw.get("weight", 1.0),
                files=raw.get("files", []),
                space=_coerce_space(raw.get("space")),
            )
        )

    return RubricConfig(judge=judge, criteria=criteria, scoring=scoring)


def load_rubric(path: Path) -> RubricConfig:
    """Load a rubric from either a ``.toml`` or ``.json`` file.

    Dispatches on the file extension; ``.json`` is parsed as a Harvey LAB
    style rubric, anything else as native TOML.
    """
    if path.suffix.lower() == ".json":
        return load_rubric_json(path)
    return load_rubric_toml(path)


def criteria_aggregate_policy_from_rubric(path: Path) -> dict[str, Any]:
    """Return a strict aggregate policy derived from declared rubric criteria.

    Reward Kit runners may emit per-criterion ``metrics``. When a selected
    verifier strategy declares a criteria file, BenchFlow treats that file as
    the authoritative metric contract: ids are unique and nonempty, weights are
    finite and nonnegative, and the rubric scoring method becomes the aggregate
    policy used to compute the canonical reward.
    """

    rubric = load_rubric(path)
    if not rubric.criteria:
        raise ValueError(f"Reward Kit criteria file {path} declares no criteria")
    if rubric.scoring.aggregation not in _VALID_REWARD_KIT_AGGREGATIONS:
        raise ValueError(
            f"Reward Kit criteria file {path} uses unsupported scoring method "
            f"{rubric.scoring.aggregation!r}"
        )
    try:
        threshold = float(rubric.scoring.threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Reward Kit criteria file {path} has invalid scoring threshold"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"Reward Kit criteria file {path} has invalid scoring threshold"
        )

    ids: list[str] = []
    weights: dict[str, float] = {}
    for criterion in rubric.criteria:
        criterion_id = criterion.id.strip()
        if not criterion_id:
            raise ValueError(f"Reward Kit criteria file {path} has an empty id")
        if criterion_id in weights:
            raise ValueError(
                f"Reward Kit criteria file {path} has duplicate id {criterion_id!r}"
            )
        if criterion.type not in {"binary", "likert", "numeric"}:
            raise ValueError(
                f"Reward Kit criteria file {path} has unsupported criterion type "
                f"for {criterion_id!r}: {criterion.type!r}"
            )
        try:
            weight = float(criterion.weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Reward Kit criteria file {path} has invalid weight for "
                f"{criterion_id!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Reward Kit criteria file {path} has invalid weight for "
                f"{criterion_id!r}"
            )
        ids.append(criterion_id)
        weights[criterion_id] = weight

    return {
        "field": "reward",
        "method": rubric.scoring.aggregation,
        "threshold": threshold,
        "criteria": ids,
        "weights": weights,
    }

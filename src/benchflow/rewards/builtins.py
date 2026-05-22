"""Built-in reward functions shipped with benchflow."""

from __future__ import annotations

import json
import logging
import math
import string
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.rubric_config import (
    Criterion,
    JudgeConfig,
    RubricConfig,
    ScoringConfig,
    load_rubric,
)
from benchflow.rewards.validation import is_valid_reward_number

logger = logging.getLogger(__name__)


class JudgeScoringError(RuntimeError):
    """Raised when a judge call cannot produce a trustworthy score."""


class TestRewardFunc:
    """Wraps existing test.sh -> reward.txt flow. Backward compatible.

    Reads ``reward.txt`` from *rollout_dir* and parses the first float.
    Returns 0.0 when the file is missing or unparseable.
    """

    async def score(self, rollout_dir: Path) -> float:
        reward_path = rollout_dir / "reward.txt"
        if not reward_path.exists():
            return 0.0
        text = reward_path.read_text().strip()
        if not text:
            return 0.0
        try:
            return float(text.splitlines()[0].strip())
        except (ValueError, IndexError):
            return 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge prompt templates
# ---------------------------------------------------------------------------

_VERDICT_PROMPT = string.Template(
    """You are evaluating an AI agent's work product against a specific quality criterion.

## Context
$context

## Agent's Output
$agent_output

## Criterion
**$criterion_title**

$criterion_description

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{
  "verdict": "pass" or "fail",
  "reasoning": "Brief explanation"
}
```
"""
)

_LIKERT_PROMPT = string.Template(
    """You are evaluating an AI agent's work product against a specific quality criterion.

## Context
$context

## Agent's Output
$agent_output

## Criterion
**$criterion_title**

$criterion_description

## Instructions
Rate the agent's output on a scale from 1 to $points for this criterion.
- 1 = Does not satisfy the criterion at all
- $points = Fully satisfies the criterion

Respond with JSON only:

```json
{
  "score": <integer 1-$points>,
  "reasoning": "Brief explanation"
}
```
"""
)

_NUMERIC_PROMPT = string.Template(
    """You are evaluating an AI agent's work product against a specific quality criterion.

## Context
$context

## Agent's Output
$agent_output

## Criterion
**$criterion_title**

$criterion_description

## Instructions
Rate the agent's output on a scale from $min_val to $max_val for this criterion.

Respond with JSON only:

```json
{
  "score": <number $min_val-$max_val>,
  "reasoning": "Brief explanation"
}
```
"""
)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc
# ---------------------------------------------------------------------------


class LLMJudgeRewardFunc:
    """LLM-as-judge for subjective quality scoring.

    Supports two modes:

    1. **Rubric mode** (``rubric_path`` or ``criteria`` provided): reads a
       ``rubric.toml`` or uses inline criteria to evaluate agent output via
       an LLM judge.  Each criterion is scored individually and results are
       aggregated.

    2. **Legacy mode** (only ``prompt`` provided, no rubric): reads a
       pre-computed score from ``llm_judge_score.txt`` in the rollout
       directory for backward compatibility.
    """

    def __init__(
        self,
        prompt: str = "",
        model: str = "claude-sonnet-4-6",
        *,
        rubric_path: Path | None = None,
        criteria: list[dict] | None = None,
        mode: Literal["batched", "individual"] = "individual",
        judge_model: str | None = None,
        judge_env: Mapping[str, str] | None = None,
        judge_errors_are_infra: bool = False,
    ) -> None:
        self.prompt = prompt
        # ``judge_model`` (an explicit ``[verifier.judge].model`` from
        # ``task.toml``) takes precedence over ``model`` and over any model
        # declared inside a rubric file. ``self.model`` is the single resolved
        # default; rubric files supply their own default via ``_resolve_model``.
        self.model = judge_model or model
        self._explicit_model = judge_model
        self._judge_env = dict(judge_env or {})
        self.mode = mode
        self._rubric_path = rubric_path
        self._inline_criteria = criteria
        self._judge_errors_are_infra = judge_errors_are_infra
        self._events: list[RewardEvent] = []

    def _resolve_model(self, rubric_default: str) -> str:
        """Resolve the judge model for a rubric.

        An explicit ``judge_model`` (from ``[verifier.judge].model``) wins;
        otherwise the rubric file's own model is used.
        """
        return self._explicit_model or rubric_default

    @property
    def events(self) -> list[RewardEvent]:
        """Dense reward events from the last ``score()`` call."""
        return list(self._events)

    def _load_rubric(self, rollout_dir: Path) -> RubricConfig | None:
        """Resolve rubric config from explicit path, inline criteria, or
        auto-discovery in the rollout directory."""
        if self._rubric_path is not None:
            return load_rubric(self._rubric_path)

        if self._inline_criteria is not None:
            parsed = []
            for raw in self._inline_criteria:
                parsed.append(
                    Criterion(
                        description=raw.get(
                            "description", raw.get("match_criteria", "")
                        ),
                        type=raw.get("type", "binary"),
                        name=raw.get("name") or raw.get("id"),
                        points=raw.get("points", 5),
                        min=raw.get("min", 0.0),
                        max=raw.get("max", 100.0),
                        weight=raw.get("weight", 1.0),
                        files=raw.get("files", []),
                    )
                )
            return RubricConfig(
                judge=JudgeConfig(model=self.model, mode=self.mode),
                criteria=parsed,
                scoring=ScoringConfig(),
            )

        # Auto-discover rubric.toml / rubric.json in rollout directory
        for candidate in [
            rollout_dir / "rubric.toml",
            rollout_dir / "rubric.json",
            rollout_dir / ".." / "rubric.toml",
            rollout_dir / ".." / "rubric.json",
        ]:
            if candidate.exists():
                return load_rubric(candidate)

        return None

    async def score(self, rollout_dir: Path) -> float:
        """Score the rollout, returning a float in [0, 1]."""
        self._events = []

        rubric = self._load_rubric(rollout_dir)
        if rubric is None:
            return self._legacy_score(rollout_dir)

        if not rubric.criteria:
            logger.warning("Rubric has no criteria — returning 0.0")
            return 0.0

        return await self._rubric_score(rubric, rollout_dir)

    # -- legacy path -------------------------------------------------------

    @staticmethod
    def _legacy_score(rollout_dir: Path) -> float:
        """Read a pre-computed judge score (backward compat)."""
        score_path = rollout_dir / "llm_judge_score.txt"
        if not score_path.exists():
            return 0.0
        text = score_path.read_text().strip()
        try:
            return float(text.splitlines()[0].strip())
        except (ValueError, IndexError):
            return 0.0

    # -- rubric path --------------------------------------------------------

    async def _rubric_score(self, rubric: RubricConfig, rollout_dir: Path) -> float:
        from benchflow.rewards.file_readers import find_deliverables
        from benchflow.rewards.llm import (
            JudgeEnvironmentError,
            call_judge,
            parse_verdict,
        )

        model = self._resolve_model(rubric.judge.model)
        deliverables = find_deliverables(rollout_dir)

        # Also check common subdirectories
        for subdir_name in ("output", "workspace/output"):
            subdir = rollout_dir / subdir_name
            if subdir.is_dir():
                deliverables.update(find_deliverables(subdir))

        if not deliverables:
            logger.warning("No deliverable files found in %s", rollout_dir)

        context = self.prompt or "Evaluate the agent's work product."
        results: list[dict] = []

        for idx, criterion in enumerate(rubric.criteria):
            agent_text = self._gather_agent_output(
                criterion, deliverables, rubric.judge.files
            )
            prompt_text = self._build_prompt(criterion, agent_text, context)

            try:
                raw_response = await call_judge(
                    model, prompt_text, env=self._judge_env
                )
                verdict = parse_verdict(raw_response)
                norm_score = self._extract_score(criterion, verdict)
            except JudgeEnvironmentError:
                raise
            except Exception as exc:
                if self._judge_errors_are_infra:
                    raise JudgeScoringError(
                        "Judge error on criterion "
                        f"{criterion.id}: {type(exc).__name__}"
                    ) from exc
                logger.warning("Judge error on criterion %s: %s", criterion.id, exc)
                norm_score = 0.0
                verdict = {
                    "verdict": "fail",
                    "reasoning": f"Judge error: {exc}",
                }

            results.append(
                {
                    "id": criterion.id,
                    "description": criterion.description,
                    "score": norm_score,
                    "weight": criterion.weight,
                    "verdict": verdict,
                }
            )
            self._events.append(
                RewardEvent(
                    type="dense",
                    reward=norm_score,
                    source=f"criterion:{criterion.id}",
                    step=idx,
                )
            )

        aggregate_score = self._aggregate(results, rubric.scoring)
        if not is_valid_reward_number(aggregate_score):
            if self._judge_errors_are_infra:
                raise JudgeScoringError(
                    "Judge aggregate reward must be finite and between 0.0 and 1.0"
                )
            logger.warning(
                "Judge aggregate reward is invalid (%r); returning 0.0",
                aggregate_score,
            )
            aggregate_score = 0.0

        # Write detailed results with the actual aggregated score
        self._write_details(rollout_dir, results, aggregate_score)

        return aggregate_score

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _gather_agent_output(
        criterion: Criterion,
        deliverables: dict[str, str],
        fallback_files: list[str],
    ) -> str:
        """Select relevant deliverable text for a criterion."""
        scope_files = criterion.files or fallback_files

        if scope_files:

            def _stem(name: str) -> str:
                return Path(name).stem.lower()

            expected_stems = {_stem(f) for f in scope_files}
            relevant = {
                k: v
                for k, v in deliverables.items()
                if _stem(k) in expected_stems
                or any(f.lower() in k.lower() for f in scope_files)
            }
        else:
            relevant = deliverables

        if not relevant:
            return "(No matching deliverable files found.)"

        parts = []
        for name, content in relevant.items():
            truncated = content[:15_000]
            if len(content) > 15_000:
                truncated += f"\n[TRUNCATED — {len(content)} chars total]"
            parts.append(f"--- {name} ---\n{truncated}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(criterion: Criterion, agent_output: str, context: str) -> str:
        """Build the appropriate prompt for the criterion type."""
        if criterion.type == "likert":
            return _LIKERT_PROMPT.safe_substitute(
                context=context,
                agent_output=agent_output,
                criterion_title=criterion.id,
                criterion_description=criterion.description,
                points=str(criterion.points),
            )
        if criterion.type == "numeric":
            return _NUMERIC_PROMPT.safe_substitute(
                context=context,
                agent_output=agent_output,
                criterion_title=criterion.id,
                criterion_description=criterion.description,
                min_val=str(criterion.min),
                max_val=str(criterion.max),
            )
        # binary (default)
        return _VERDICT_PROMPT.safe_substitute(
            context=context,
            agent_output=agent_output,
            criterion_title=criterion.id,
            criterion_description=criterion.description,
        )

    @staticmethod
    def _extract_score(criterion: Criterion, verdict: dict) -> float:
        """Normalize the raw LLM verdict into [0, 1]."""
        if criterion.type == "binary":
            v = str(verdict.get("verdict", "fail")).strip().lower()
            return 1.0 if v in {"pass", "passed", "yes", "true", "1"} else 0.0

        raw = verdict.get("score", 0)
        try:
            raw_score = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(raw_score):
            raise ValueError("Judge score must be finite")
        score = criterion.normalize(raw_score)
        if not is_valid_reward_number(score):
            raise ValueError("Normalized judge score must be between 0.0 and 1.0")
        return score

    @staticmethod
    def _aggregate(results: list[dict], scoring: ScoringConfig) -> float:
        """Aggregate per-criterion scores into a single reward."""
        if not results:
            return 0.0

        scores = [r["score"] for r in results]
        weights = [r["weight"] for r in results]

        if scoring.aggregation == "all_pass":
            return 1.0 if all(s >= 0.5 for s in scores) else 0.0

        if scoring.aggregation == "any_pass":
            return 1.0 if any(s >= 0.5 for s in scores) else 0.0

        if scoring.aggregation == "threshold":
            total_w = sum(weights) or 1.0
            weighted = (
                sum(s * w for s, w in zip(scores, weights, strict=True)) / total_w
            )
            return 1.0 if weighted >= scoring.threshold else 0.0

        # weighted_mean (default)
        total_w = sum(weights) or 1.0
        return sum(s * w for s, w in zip(scores, weights, strict=True)) / total_w

    @staticmethod
    def _write_details(
        rollout_dir: Path, results: list[dict], aggregate_score: float
    ) -> None:
        """Write detailed evaluation results alongside the rollout."""
        try:
            total = len(results)
            n_passed = sum(1 for r in results if r["score"] >= 0.5)
            details_path = rollout_dir / "evaluation_details.json"
            details_path.write_text(
                json.dumps(
                    {
                        "score": aggregate_score,
                        "n_passed": n_passed,
                        "n_total": total,
                        "results": results,
                    },
                    indent=2,
                    allow_nan=False,
                    default=str,
                )
            )
        except Exception as exc:
            logger.debug("Could not write evaluation details: %s", exc)


class StringMatchRewardFunc:
    """Exact or fuzzy string matching against ``answer.txt``."""

    def __init__(self, expected: str, fuzzy: bool = False) -> None:
        self.expected = expected
        self.fuzzy = fuzzy

    async def score(self, rollout_dir: Path) -> float:
        answer_path = rollout_dir / "answer.txt"
        if not answer_path.exists():
            return 0.0
        actual = answer_path.read_text().strip()
        if self.fuzzy:
            return 1.0 if self.expected.lower() in actual.lower() else 0.0
        return 1.0 if actual == self.expected else 0.0


class CodeExecRewardFunc:
    """Arbitrary Python scoring function.

    The callable receives a ``Path`` (rollout directory) and returns a float.
    """

    def __init__(self, func: Callable[[Path], float]) -> None:
        self.func = func

    async def score(self, rollout_dir: Path) -> float:
        result = self.func(rollout_dir)
        return float(result)

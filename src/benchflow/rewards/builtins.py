"""Built-in reward functions shipped with benchflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


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


class LLMJudgeRewardFunc:
    """LLM-as-judge for subjective quality scoring.

    This is a placeholder that reads a pre-computed judge score from
    ``llm_judge_score.txt`` in the rollout directory. Full LLM integration
    is deferred to a future release.
    """

    def __init__(self, prompt: str, model: str = "gemini-3.1-flash-lite") -> None:
        self.prompt = prompt
        self.model = model

    async def score(self, rollout_dir: Path) -> float:
        score_path = rollout_dir / "llm_judge_score.txt"
        if not score_path.exists():
            return 0.0
        text = score_path.read_text().strip()
        try:
            return float(text.splitlines()[0].strip())
        except (ValueError, IndexError):
            return 0.0


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

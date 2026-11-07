"""Composable reward functions for benchflow verifiers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.rewards.builtins import (
        CodeExecRewardFunc,
        JudgeScoringError,
        LLMJudgeRewardFunc,
        StringMatchRewardFunc,
        TestRewardFunc,
    )
    from benchflow.rewards.events import RewardEvent
    from benchflow.rewards.memory_scorer import MEMORY_STATE_KEY, MemoryScorer
    from benchflow.rewards.node import (
        PATH_STATE_KEY,
        NodeScorer,
        PathReward,
        score_node,
    )
    from benchflow.rewards.protocol import Reward, RewardFunc, VerifyResult
    from benchflow.rewards.rubric import Rubric
    from benchflow.rewards.rubric_config import (
        Criterion,
        JudgeConfig,
        RubricConfig,
        ScoringConfig,
        load_rubric,
        load_rubric_json,
        load_rubric_toml,
    )

# Each export's module and name there. They are imported on first use (PEP
# 562), so importing one submodule (``benchflow.rewards.validation``, which
# reading a task file needs) does not import every other one.
_EXPORTS: dict[str, tuple[str, str]] = {
    "CodeExecRewardFunc": ("benchflow.rewards.builtins", "CodeExecRewardFunc"),
    "JudgeScoringError": ("benchflow.rewards.builtins", "JudgeScoringError"),
    "LLMJudgeRewardFunc": ("benchflow.rewards.builtins", "LLMJudgeRewardFunc"),
    "StringMatchRewardFunc": ("benchflow.rewards.builtins", "StringMatchRewardFunc"),
    "TestRewardFunc": ("benchflow.rewards.builtins", "TestRewardFunc"),
    "RewardEvent": ("benchflow.rewards.events", "RewardEvent"),
    "MEMORY_STATE_KEY": ("benchflow.rewards.memory_scorer", "MEMORY_STATE_KEY"),
    "MemoryScorer": ("benchflow.rewards.memory_scorer", "MemoryScorer"),
    "PATH_STATE_KEY": ("benchflow.rewards.node", "PATH_STATE_KEY"),
    "NodeScorer": ("benchflow.rewards.node", "NodeScorer"),
    "PathReward": ("benchflow.rewards.node", "PathReward"),
    "score_node": ("benchflow.rewards.node", "score_node"),
    "Reward": ("benchflow.rewards.protocol", "Reward"),
    "RewardFunc": ("benchflow.rewards.protocol", "RewardFunc"),
    "VerifyResult": ("benchflow.rewards.protocol", "VerifyResult"),
    "Rubric": ("benchflow.rewards.rubric", "Rubric"),
    "Criterion": ("benchflow.rewards.rubric_config", "Criterion"),
    "JudgeConfig": ("benchflow.rewards.rubric_config", "JudgeConfig"),
    "RubricConfig": ("benchflow.rewards.rubric_config", "RubricConfig"),
    "ScoringConfig": ("benchflow.rewards.rubric_config", "ScoringConfig"),
    "load_rubric": ("benchflow.rewards.rubric_config", "load_rubric"),
    "load_rubric_json": ("benchflow.rewards.rubric_config", "load_rubric_json"),
    "load_rubric_toml": ("benchflow.rewards.rubric_config", "load_rubric_toml"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = [
    "MEMORY_STATE_KEY",
    "PATH_STATE_KEY",
    "CodeExecRewardFunc",
    "Criterion",
    "JudgeConfig",
    "JudgeScoringError",
    "LLMJudgeRewardFunc",
    "MemoryScorer",
    "NodeScorer",
    "PathReward",
    "Reward",
    "RewardEvent",
    "RewardFunc",
    "Rubric",
    "RubricConfig",
    "ScoringConfig",
    "StringMatchRewardFunc",
    "TestRewardFunc",
    "VerifyResult",
    "load_rubric",
    "load_rubric_json",
    "load_rubric_toml",
    "score_node",
]

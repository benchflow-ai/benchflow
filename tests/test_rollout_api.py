"""Tests for the rollout-native lifecycle entry points."""

from pathlib import Path

from benchflow.rollouts.config import RolloutConfig
from benchflow.rollouts.rollout import Rollout
from benchflow.rollouts.runner import run
from benchflow.trial import Trial


def test_rollout_lifecycle_class_available() -> None:
    assert issubclass(Rollout, Trial)


def test_rollout_config_create_uses_rollout_class() -> None:
    config = RolloutConfig.from_single(
        task_path=Path("tasks/example"),
        agent="gemini",
    )
    rollout = Rollout(config)
    assert isinstance(rollout, Rollout)


def test_rollout_runner_callable() -> None:
    assert callable(run)

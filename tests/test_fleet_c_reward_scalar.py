"""Guards PR #56 against unchecked reward scalars."""

import json

import pytest

from benchflow._utils.scoring import extract_reward
from benchflow.metrics import collect_metrics


@pytest.mark.parametrize(
    "reward", [True, False, "1", float("nan"), float("inf"), 10**1000]
)
def test_non_numeric_or_nonfinite_reward_is_unscored(tmp_path, reward):
    result = {"task_name": "scalar", "rewards": {"reward": reward}}
    trial = tmp_path / "scalar__one"
    trial.mkdir()
    (trial / "result.json").write_text(json.dumps(result))
    assert extract_reward(result) is None
    metrics = collect_metrics(str(tmp_path))
    assert metrics.summary()["passed"] == 0


def test_boolean_reward_does_not_count_as_a_pass(tmp_path):
    trial = tmp_path / "boolean__one"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_name": "boolean", "rewards": {"reward": True}})
    )
    assert collect_metrics(str(tmp_path)).summary()["passed"] == 0

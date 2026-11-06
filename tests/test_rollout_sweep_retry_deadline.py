"""Guards CONF-rollout-2 against retry budget undercount at 82e412f1."""

from types import SimpleNamespace

import pytest

from benchflow.rollout import Role, RolloutConfig, Scene, Turn
from benchflow.rollout._deadline import hard_deadline_sec
from tests.test_rollout_hard_deadline import _write_task


@pytest.mark.parametrize("budgets", [(7200,), (60, 7200), (7200, 60), (7200, 1800, 60)])
@pytest.mark.parametrize("free_rounds", [0, 1, 3])
@pytest.mark.parametrize("strategy", ["verify-retry", "custom"])
def test_deadline_sums_scene_and_free_round_allowances(
    tmp_path, monkeypatch, budgets, free_rounds, strategy
):
    monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
    task = tmp_path / "task"
    _write_task(task)
    scenes = [
        Scene(
            name=f"scene{index}",
            roles=[Role(name="agent", agent="codex-acp", timeout_sec=budget)],
            turns=[Turn(role="agent")],
        )
        for index, budget in enumerate(budgets)
    ]
    # verify-retry requires at least one retry (k >= 1).
    rounds = (
        max(2, len(budgets) + free_rounds)
        if strategy == "verify-retry"
        else len(budgets) + free_rounds
    )
    free_rounds = rounds - len(budgets)
    base = hard_deadline_sec(RolloutConfig(task_path=task, scenes=scenes))
    if strategy == "verify-retry":
        cfg = RolloutConfig(
            task_path=task, scenes=scenes, loop_strategy=f"verify-retry:k={rounds - 1}"
        )
    else:
        cfg = RolloutConfig(task_path=task, scenes=scenes, max_user_rounds=rounds)
        cfg.user = SimpleNamespace()
    assert cfg.max_user_rounds == rounds
    assert hard_deadline_sec(cfg) == base + free_rounds * budgets[-1] + rounds * 60

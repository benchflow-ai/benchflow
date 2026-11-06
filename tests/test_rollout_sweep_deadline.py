"""Guards FF-sw-rollout against scene timeout aliasing at 363d815b."""

import pytest

from benchflow.rollout import Role, RolloutConfig, Scene, Turn
from benchflow.rollout._deadline import hard_deadline_sec
from tests.test_rollout_hard_deadline import _write_task


@pytest.mark.parametrize("budgets", [(7200, 60), (60, 7200), (7200, 60, 1800)])
@pytest.mark.parametrize("turns", [1, 3])
def test_deadline_uses_the_roles_of_each_scene(tmp_path, monkeypatch, budgets, turns):
    monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
    task = tmp_path / "task"
    _write_task(task)

    def config(unique):
        scenes = []
        for index, budget in enumerate(budgets):
            name = f"agent{index}" if unique else "agent"
            scenes.append(
                Scene(
                    name=f"scene{index}",
                    roles=[Role(name=name, agent="codex-acp", timeout_sec=budget)],
                    turns=[Turn(role=name)] * turns,
                )
            )
        return RolloutConfig(task_path=task, scenes=scenes)

    assert hard_deadline_sec(config(False)) == hard_deadline_sec(config(True))

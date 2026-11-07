"""Guards FF-sw-other wall budget across stepping routes at 363d815b."""

import pytest

from benchflow.embodied.backend import StepResult
from benchflow.embodied.server import EpisodeConfig, EpisodeServer
from benchflow.embodied.spec import Skill
from tests.test_embodied import PointArm


@pytest.mark.parametrize(
    "route", ["act", "builtin", "closed", "backend", "done", "direct"]
)
@pytest.mark.parametrize("mode", ["first", "final"])
def test_deadline_gates_steps_and_success(tmp_path, monkeypatch, route, mode):
    now = [100.0]
    monkeypatch.setattr("benchflow.embodied.server.time.monotonic", lambda: now[0])
    config = EpisodeConfig(
        success_mode=mode,
        max_steps=100,
        max_repeat=50,
        max_wall_s=1,
        settle_steps=0,
    )
    ep = EpisodeServer(PointArm(), config, tmp_path / "episode")
    calls = []

    def step(action):
        calls.append(action)
        now[0] += 0.6
        return StepResult(success=len(calls) >= 2)

    ep.backend.step = step
    ep.backend.success = lambda: len(calls) >= 2

    def run(name, args):
        now[0] += 2
        return {"success": True}

    def start(name, args):
        for _ in range(50):
            yield [0, 0, 0, -1]

    try:
        if route == "act":
            ep.handle({"op": "act", "action": [0, 0, 0, -1], "repeat": 50})
        elif route == "builtin":
            ep.handle({"op": "grip", "value": 1, "steps": 50})
        elif route in {"closed", "backend"}:
            ep.embodiment.skills = [Skill("reach", impl="backend")]
            if route == "closed":
                ep.backend.start_skill = start
            else:
                ep.backend.run_skill = run
            ep.handle({"op": "skill", "name": "reach"})
        elif route == "direct":
            now[0] += 2
            assert not ep.env_step([0, 0, 0, -1]).success
            assert calls == []
        else:
            now[0] += 2
            ep.backend.success = lambda: True
            ep.handle({"op": "done"})
        assert ep.finished and ep.outcome == "wall_time_exhausted"
        assert not ep.result["success"]
        assert len(calls) <= 2
    finally:
        ep.close()

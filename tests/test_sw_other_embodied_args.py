"""Guards FF-sw-other backend skill binding at 363d815b."""

import pytest

from benchflow.embodied.spec import Skill, SkillArg
from tests.test_embodied import _server


@pytest.mark.parametrize("args", [["x=0.1"], {"x": 0.1}, ["0.1"], ["y=0.2", "x=0.1"]])
@pytest.mark.parametrize("closed", [False, True])
def test_backend_receives_bound_values_and_defaults(tmp_path, args, closed):
    ep = _server(tmp_path, settle_steps=0)
    skill = Skill(
        name="reach",
        impl="backend",
        args=[
            SkillArg("x", "float"),
            SkillArg("y", "float", optional=True, default=0.2),
        ],
    )
    ep.embodiment.skills = [skill]
    seen = []

    def run(name, values):
        seen.append((name, values))
        return {}

    def start(name, values):
        seen.append((name, values))
        yield [0, 0, 0, -1]

    if closed:
        ep.backend.start_skill = start
    else:
        ep.backend.run_skill = run
    try:
        assert ep.skill("reach", args)["ok"]
        assert seen == [("reach", ["0.1", "0.2"])]
    finally:
        ep.close()

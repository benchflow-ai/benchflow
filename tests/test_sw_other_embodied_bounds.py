"""Guards FF-sw-other action normalization regression at 363d815b."""

import pytest

from tests.test_embodied import _server


@pytest.mark.parametrize("value", [-100.0, -1.0, 0.0, 1.0, 100.0])
@pytest.mark.parametrize("grouped", [False, True])
def test_all_action_forms_obey_declared_bounds(tmp_path, value, grouped):
    ep = _server(tmp_path, success_mode="first", max_steps=1, settle_steps=0)
    seen = []
    original = ep.backend.step

    def step(action):
        seen.append(list(action))
        return original(action)

    ep.backend.step = step
    try:
        action = [value, 0, 0, -1]
        req = (
            {"op": "act", "groups": ep.embodiment.unpack(action)}
            if grouped
            else {"op": "act", "action": action}
        )
        assert ep.handle(req)["ok"]
        assert seen == [[min(1, max(-1, value)), 0, 0, -1]]
        if abs(value) > 1:
            assert not ep.result["success"]
    finally:
        ep.close()

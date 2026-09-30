"""The sim/real axis of the embodiment spec and the intervals and reducers of the seed report."""

import pytest

from benchflow.embodied.rollouts import bootstrap_ci, reduce, seed_report, wilson
from benchflow.embodied.spec import Embodiment, SpecError, ee_delta_group, gripper_group


def test_mode_defaults_to_sim_and_real_needs_safety():
    e = Embodiment(
        name="arm", kind="arm", action_groups=[*ee_delta_group(), gripper_group()]
    ).validate()
    assert e.mode == "sim"
    d = e.to_dict()
    d["mode"] = "real"
    with pytest.raises(SpecError):
        Embodiment.from_dict(d).validate()
    d["safety"] = {
        "joint_limits": {"low": [-1], "high": [1]},
        "attended": True,
        "estop": "file",
    }
    real = Embodiment.from_dict(d).validate()
    assert real.mode == "real" and real.safety.attended
    assert Embodiment.from_dict(real.to_dict()).safety.estop == "file"
    d["mode"] = "moon"
    with pytest.raises(SpecError):
        Embodiment.from_dict(d).validate()


def test_intervals_and_reducers():
    assert wilson(0, 10)[0] == 0.0 and wilson(0, 10)[1] > 0.2
    assert wilson(10, 10)[1] == 1.0
    lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0])
    assert 0.0 <= lo <= 0.6 <= hi <= 1.0
    assert (
        reduce([0, 1, 1], "median") == 1
        and reduce([0, 1, 1], "mode") == 1
        and reduce([0, 2], "max") == 2
    )


def test_seed_report_groups_modes():
    rows = [
        {
            "task": "t",
            "seed": s,
            "reward": float(s % 2),
            "error": None,
            "outcome": "done",
            "steps_used": 1,
            "return": None,
            "initial_state_sha256": None,
            "trial": f"t-{s}",
            "mode": m,
        }
        for s, m in ((0, "sim"), (1, "sim"), (2, "real"), (3, "real"))
    ]
    rep = seed_report("unused", rows=rows)
    s = rep["summary"]
    assert s["pass_rate"] == 0.5 and s["pass_rate_ci95"] is not None
    assert set(s["by_mode"]) == {"real", "sim"} and rep["tasks"]["t"]["mode"] == [
        "real",
        "sim",
    ]

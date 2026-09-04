"""Hardware-free tests for the Metal simulator plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.eval import eval as ir_eval
from inspect_robots.registry import registered, resolve
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_metal_sim.config import MetalSimConfig
from inspect_robots_metal_sim.embodiment import MetalSimEmbodiment
from inspect_robots_metal_sim.kinematics import MetalKinematics
from inspect_robots_metal_sim.policies import IKPolicy, PyFilePolicy
from inspect_robots_metal_sim.tasks import metal_sim_pick, metal_sim_reach


def test_registered_components() -> None:
    assert "metal_sim" in registered("embodiment")
    assert {"metal_ik", "metal_pyfile"} <= set(registered("policy"))
    assert {"metal-sim-reach", "metal-sim-pick"} <= set(registered("task"))


def test_conformance() -> None:
    assert_embodiment_conformant(MetalSimEmbodiment(trace_dir=None).info)


def test_zero_pose_is_the_vendor_parked_pose() -> None:
    kin = MetalKinematics()
    flange = kin.flange(np.zeros(7))
    assert 0.15 < flange[2] < 0.25, flange  # upper arm horizontal, forearm folded: flange ~19 cm up
    assert abs(flange[1]) < 0.01
    assert kin.tool_axis(np.zeros(7))[0] > 0.9  # gripper points forward (+x) when parked


# Joint readings (rad, JOINT1..JOINT6) and the SDK's end_pose (Link6 origin, m) recorded on real
# hardware by the vendor's teach-mode tool (metal_ros2/src/metal_teach_mode/teach_pose/data/
# recorded_pos.jsonl). They pin the motor-degrees -> URDF mapping to the identity.
VENDOR_RECORDED = [
    ([0.048583984375, -0.07752370834350586, 0.22394704818725586, -0.05145883560180664, -0.07790708541870117, -0.014278411865234375], [0.02746284412740669, 0.0029158136841631248, 0.25327152192612234]),
    ([0.048392295837402344, -0.6519088745117188, 0.9917101860046387, -0.051267147064208984, -0.07790708541870117, -0.014278411865234375], [0.05719008991095223, 0.004349462189515469, 0.4582373115075344]),
]


@pytest.mark.parametrize(("q_rad", "end_pose"), VENDOR_RECORDED)
def test_fk_matches_vendor_recorded_end_pose(q_rad: list[float], end_pose: list[float]) -> None:
    kin = MetalKinematics()
    q_deg = np.append(np.rad2deg(q_rad), 0.0)
    assert np.linalg.norm(kin.flange(q_deg) - np.asarray(end_pose)) < 1e-3


def test_leaning_forward_moves_tip_forward_and_up() -> None:
    kin = MetalKinematics()
    parked = kin.tip(np.zeros(7))
    extended_up = kin.tip(np.array([0.0, -90.0, 180.0, 0.0, 0.0, 0.0, 0.0]))
    assert extended_up[2] > 0.7, (parked, extended_up)  # shoulder vertical + elbow straight = fully extended upward
    forward = kin.tip(np.array([0.0, -150.0, 90.0, 0.0, 0.0, 0.0, 0.0]))
    assert forward[0] > 0.3, forward  # leaning past vertical and unfolding the elbow reaches forward


def test_ik_reaches_workspace_point() -> None:
    cfg = MetalSimConfig()
    kin = MetalKinematics()
    q, err = kin.solve_ik(np.array([0.25, -0.1, 0.015]), np.zeros(7), low=cfg.low, high=cfg.high)
    assert err < 0.002, err
    assert np.all(q[:6] >= cfg.low[:6]) and np.all(q[:6] <= cfg.high[:6])


def test_step_ramp_and_clamp(tmp_path: Path) -> None:
    emb = MetalSimEmbodiment(trace_dir=str(tmp_path))
    emb.reset(Scene(id="s", instruction="reach the cube", init_seed=1))
    res = emb.step(Action(data=np.array([90.0, -90.0, 90.0, 0.0, 0.0, 0.0, 200.0])))
    q = res.observation.state["joint_pos"]
    assert np.allclose(q[:3], [3.0, -3.0, 3.0])  # 30 deg/s at 10 Hz
    assert q[6] == pytest.approx(9.0)  # gripper 90 deg/s at 10 Hz
    assert res.observation.images["front"].shape == (240, 320, 3)
    assert res.info["success"] is False
    assert emb.trace_path is not None and emb.trace_path.exists()
    emb.close()
    lines = [json.loads(line) for line in emb.trace_path.read_text().splitlines()]
    assert [entry["kind"] for entry in lines] == ["header", "step", "end"]


def test_reach_with_ik_oracle(tmp_path: Path) -> None:
    emb = MetalSimEmbodiment(trace_dir=None)
    logs = ir_eval(metal_sim_reach(num_scenes=2, max_steps=120), IKPolicy(), emb, log_dir=str(tmp_path), store_frames=False)
    log = logs[0]
    assert log.status == "success", log.error
    for sample in log.samples:
        assert sample.reduced["success_at_end"] == 1.0, sample


def test_pick_with_ik_oracle(tmp_path: Path) -> None:
    emb = MetalSimEmbodiment(trace_dir=None, camera=False)
    logs = ir_eval(metal_sim_pick(num_scenes=2, max_steps=200), IKPolicy(), emb, log_dir=str(tmp_path), store_frames=False)
    log = logs[0]
    assert log.status == "success", log.error
    for sample in log.samples:
        assert sample.reduced["success_at_end"] == 1.0, sample


def test_pyfile_policy(tmp_path: Path) -> None:
    src = tmp_path / "policy.py"
    src.write_text("def act(obs):\n    q = list(obs['joint_pos'])\n    q[1] -= 5\n    return q\n")
    policy = PyFilePolicy(path=str(src))
    emb = MetalSimEmbodiment(trace_dir=None, camera=False)
    obs = emb.reset(Scene(id="s", instruction="reach the cube", init_seed=0))
    policy.reset(Scene(id="s", instruction="reach the cube", init_seed=0))
    chunk = policy.act(obs)
    assert chunk.actions[0].data[1] == pytest.approx(-5.0)


def test_registry_factories_accept_cli_strings() -> None:
    task = resolve("task", "metal-sim-reach", num_scenes="1", max_steps="10")
    assert len(list(task.scenes)) == 1
    emb = resolve("embodiment", "metal_sim", camera="false", trace_dir="none", max_velocity_deg_s="60")
    assert emb.info.observation_space.cameras == ()
    assert emb.config.max_step_deg[0] == 6.0

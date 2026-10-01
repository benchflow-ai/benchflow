"""benchflow.embodied: embodiment spec, episode server, agent CLI, verifier, sidecar format, export, seeded rollouts."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from benchflow.embodied import rollouts, sidecar
from benchflow.embodied.backend import StepResult
from benchflow.embodied.export import episode_rows, export_job, write_rollout_embodied
from benchflow.embodied.server import EpisodeConfig, EpisodeServer, serve
from benchflow.embodied.spec import (
    ActionGroup,
    Budgets,
    Camera,
    Embodiment,
    Field,
    RewardSpec,
    Sensors,
    Skill,
    SkillArg,
    SpecError,
    bind_skill_args,
    ee_delta_group,
    grip_skill,
    gripper_group,
    move_to_skill,
)

# ---- a tiny simulator: a point-mass end effector and a gripper -----------------------------------------------


class PointArm:
    """End effector moves 1 cm per unit of arm.ee_delta per step; success = within 2 cm of the goal."""

    name = "pointarm"

    def __init__(self, goal=(0.1, 0.0, 0.0), dense="sparse", bimanual=False):
        self.goal = np.asarray(goal, dtype=float)
        self.dense = dense
        self.bimanual = bimanual
        self.pos = np.zeros(3)
        self.grip = -1.0
        self.closed = False

    def embodiment(self) -> Embodiment:
        if self.bimanual:
            groups = [
                *ee_delta_group("left", "x 1 cm"),
                gripper_group("left.gripper"),
                *ee_delta_group("right", "x 1 cm"),
                gripper_group("right.gripper"),
            ]
            skills = [
                move_to_skill("left", "left.gripper"),
                move_to_skill("right", "right.gripper"),
            ]
            kind = "bimanual"
        else:
            groups = [*ee_delta_group("arm", "x 1 cm"), gripper_group()]
            skills = [
                move_to_skill(aliases=["move_to", "move-to"]),
                grip_skill(aliases=["grip"]),
            ]
            kind = "arm"
        return Embodiment(
            name="pointarm",
            kind=kind,
            action_groups=groups,
            sensors=Sensors(
                cameras=[Camera("top")],
                proprioception=[Field("hand_pos", [3], "m")],
                state=[Field("goal", [3], "m", privileged=True)],
            ),
            skills=skills,
            budgets=Budgets(max_steps=60, max_repeat=20, settle_steps=2),
            reward=RewardSpec(dense=self.dense),
            doc="a point mass",
        )

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(-0.01, 0.01, 3)
        self.grip, self.closed = -1.0, False

    def step(self, action) -> StepResult:
        a = np.asarray(action, dtype=float)
        self.pos = self.pos + a[:3] * 0.01
        self.grip = float(a[3])
        self.closed = self.grip > 0
        ok = self.success()
        return StepResult(
            success=ok if self.dense == "shaped" else False,
            reward=-float(np.linalg.norm(self.goal - self.pos)),
        )

    def observe(self) -> dict:
        return {
            "hand_pos": [round(float(x), 5) for x in self.pos],
            "gripper_open": 0.0 if self.closed else 1.0,
            "goal": self.goal.tolist(),
        }

    def render(self, camera=None):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def success(self) -> bool:
        return bool(np.linalg.norm(self.goal - self.pos) < 0.02)

    def ee_position(self, arm):
        return self.pos.tolist()


def _server(tmp_path, backend=None, run_dir=None, **block) -> EpisodeServer:
    backend = backend or PointArm()
    emb = backend.embodiment()
    cfg = EpisodeConfig.from_task({"id": "demo", **block}, emb, default_camera="top")
    return EpisodeServer(backend, cfg, run_dir or tmp_path / "ep", oracle_token="tok")


# ---- the spec ------------------------------------------------------------------------------------------------


def _catalog() -> list[Embodiment]:
    """One embodiment per robot kind the spec must cover."""
    rng1 = [-1.0], [1.0]
    return [
        PointArm().embodiment(),
        PointArm(bimanual=True).embodiment(),
        Embodiment(
            "allegro-panda",
            "hand",
            [
                *ee_delta_group(
                    "arm", "x 2 cm", limit=8.0, rotation=True, rot_limit=3.0
                ),
                ActionGroup(
                    "hand.joints",
                    [f"f{i}" for i in range(16)],
                    [-6.0] * 16,
                    [6.0] * 16,
                    "joint_delta",
                ),
            ],
        ),
        Embodiment(
            "tiago",
            "mobile_manipulator",
            [
                ActionGroup(
                    "base.twist",
                    ["vx", "vy", "wz"],
                    [-1.0] * 3,
                    [1.0] * 3,
                    "base_twist",
                ),
                *ee_delta_group("arm"),
                gripper_group(),
            ],
            skills=[Skill("base.navigate_to", [SkillArg("obj", "object")])],
        ),
        Embodiment(
            "go2",
            "quadruped",
            [
                ActionGroup(
                    "base.twist",
                    ["vx", "vy", "wz"],
                    [-1.0] * 3,
                    [1.0] * 3,
                    "base_twist",
                )
            ],
        ),
        Embodiment(
            "g1",
            "humanoid",
            [
                ActionGroup(
                    "joints.pos",
                    [f"q{i}" for i in range(23)],
                    [-3.0] * 23,
                    [3.0] * 23,
                    "joint_pos",
                    hold="last",
                    initial=[0.0] * 23,
                )
            ],
        ),
        Embodiment(
            "skydio-x2",
            "drone",
            [
                ActionGroup(
                    "body.velocity_setpoint",
                    ["vx", "vy", "vz"],
                    [-1.0] * 3,
                    [1.0] * 3,
                    "velocity_setpoint",
                ),
                ActionGroup("body.yaw_rate", ["yaw_rate"], *rng1, "velocity_setpoint"),
                ActionGroup("payload.release", ["drop"], *rng1, "discrete"),
            ],
        ),
        Embodiment(
            "behavior-r1",
            "skill_only",
            [ActionGroup("base.wait", ["wait"], [0.0], [1.0], "wait")],
            skills=[
                Skill("grasp", [SkillArg("obj", "object")]),
                Skill("toggle_on", [SkillArg("obj", "object")]),
                Skill("release"),
            ],
        ),
    ]


@pytest.mark.parametrize("emb", _catalog(), ids=lambda e: e.kind)
def test_spec_covers_every_robot_kind_and_round_trips(emb: Embodiment) -> None:
    emb.validate()
    back = Embodiment.from_dict(json.loads(json.dumps(emb.to_dict())))
    back.validate()
    assert back.dim == emb.dim
    assert back.flat_names() == emb.flat_names()
    assert [s.name for s in back.skills] == [s.name for s in emb.skills]


def test_spec_rejects_structural_errors() -> None:
    good = PointArm().embodiment()
    with pytest.raises(SpecError, match="kind"):
        Embodiment("x", "tank", good.action_groups).validate()
    with pytest.raises(SpecError, match="unique"):
        Embodiment(
            "x", "arm", [good.action_groups[0], good.action_groups[0]]
        ).validate()
    with pytest.raises(SpecError, match="low/high"):
        Embodiment(
            "x", "arm", [ActionGroup("g", ["a", "b"], [0.0], [1.0], "joint_pos")]
        ).validate()
    with pytest.raises(SpecError, match="unknown group"):
        Embodiment(
            "x",
            "arm",
            good.action_groups,
            skills=[Skill("s", impl="builtin", binds={"arm": "nope"})],
        ).validate()
    with pytest.raises(SpecError, match="used twice"):
        Embodiment(
            "x",
            "arm",
            good.action_groups,
            skills=[Skill("a"), Skill("b", aliases=["a"])],
        ).validate()
    with pytest.raises(SpecError, match="at least one"):
        Embodiment("x", "skill_only", []).validate()


def test_pack_holds_uncommanded_groups_and_clips() -> None:
    e = PointArm().embodiment()
    assert e.pack({"arm.ee_delta": [5, 0, -0.5]}) == [
        1.0,
        0.0,
        -0.5,
        -1.0,
    ]  # gripper holds its initial (open)
    assert e.pack({}, {"gripper": [1.0]}) == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # gripper holds its last command
    assert e.unpack([0.1, 0.2, 0.3, 1.0]) == {
        "arm.ee_delta": [0.1, 0.2, 0.3],
        "gripper": [1.0],
    }
    with pytest.raises(SpecError, match="unknown action group"):
        e.pack({"base.twist": [0, 0, 0]})
    with pytest.raises(SpecError, match="takes 3"):
        e.pack({"arm.ee_delta": [1, 2]})
    with pytest.raises(SpecError, match="not finite"):
        e.pack({"arm.ee_delta": [float("nan"), 0, 0]})


def test_bind_skill_args_positional_keyword_and_types() -> None:
    s = move_to_skill()
    assert bind_skill_args(s, ["0.1", "0.2", "0.3", "grip=1"]) == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "grip": 1.0,
        "max_steps": 100,
        "tol": 0.01,
    }
    assert bind_skill_args(s, {"x": 1, "y": 2, "z": 3})["tol"] == 0.01
    with pytest.raises(SpecError, match="needs argument 'z'"):
        bind_skill_args(s, ["0.1", "0.2"])
    with pytest.raises(SpecError, match="must be float"):
        bind_skill_args(s, ["a", "0", "0"])
    with pytest.raises(SpecError, match="no argument"):
        bind_skill_args(s, {"x": 0, "y": 0, "z": 0, "speed": 1})
    enum = Skill("pick", [SkillArg("side", "enum", choices=["left", "right"])])
    with pytest.raises(SpecError, match="one of"):
        bind_skill_args(enum, ["up"])


# ---- the episode server --------------------------------------------------------------------------------------


def test_act_with_groups_legacy_flat_and_step_trace(tmp_path) -> None:
    ep = _server(tmp_path)
    r = ep.handle(
        {
            "op": "act",
            "groups": {"arm.ee_delta": [1, 0, 0], "gripper": [1]},
            "repeat": 3,
        }
    )
    assert r["ok"] and r["result"]["executed_steps"] == 3
    r = ep.handle(
        {"op": "act", "groups": {"arm.ee_delta": [1, 0, 0]}}
    )  # gripper holds last (closed)
    assert r["result"]["state"]["gripper_open"] == 0.0
    r = ep.handle(
        {"op": "act", "action": [1, 0, 0, -1], "repeat": 2}
    )  # protocol-1 flat vector
    assert r["ok"] and ep.steps == 6
    bad = ep.handle({"op": "act", "action": [1, 0]})
    assert not bad["ok"] and "list of 4 numbers" in bad["error"]
    bad = ep.handle({"op": "act", "groups": {"wheels": [1]}})
    assert not bad["ok"] and "unknown action group" in bad["error"]
    rows = [
        json.loads(x)
        for x in (tmp_path / "ep" / "steps.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    assert rows[0]["action"] == {"arm.ee_delta": [1.0, 0.0, 0.0], "gripper": [1.0]}
    assert rows[3]["action"]["gripper"] == [1.0]
    assert "state" in rows[-1] and rows[-1]["op"] == "act"


def test_info_has_protocol_one_keys_and_the_embodiment(tmp_path) -> None:
    info = _server(tmp_path).handle({"op": "info"})["result"]
    assert info["action"]["names"] == ["dx", "dy", "dz", "grip"]
    assert info["skills"] == ["arm.move_to", "gripper.set"]
    assert info["max_steps"] == 60 and info["protocol"] == 2
    emb = Embodiment.from_dict(info["embodiment"]).validate()
    assert emb.kind == "arm" and emb.budgets.max_steps == 60


def test_legacy_move_to_and_grip_run_builtin_skills(tmp_path) -> None:
    ep = _server(tmp_path)
    r = ep.handle({"op": "move_to", "pos": [0.1, 0.0, 0.0], "grip": 1.0})
    assert r["ok"] and r["result"]["reached"], r
    r = ep.handle({"op": "grip", "value": -1, "steps": 3})
    assert (
        r["ok"]
        and r["result"]["executed_steps"] == 3
        and r["result"]["state"]["gripper_open"] == 1.0
    )
    r = ep.handle({"op": "skill", "name": "move-to", "args": ["0", "0", "0"]})
    assert r["ok"] and r["result"]["reached"]
    ops = {
        json.loads(x)["op"]
        for x in (tmp_path / "ep" / "steps.jsonl").read_text().splitlines()
    }
    assert ops == {"skill:arm.move_to", "skill:gripper.set"}


def test_skills_disabled(tmp_path) -> None:
    ep = _server(tmp_path, skills=False)
    assert ep.handle({"op": "info"})["result"]["skills"] == []
    r = ep.handle({"op": "move_to", "pos": [0, 0, 0]})
    assert not r["ok"] and "not available" in r["error"]


def test_bimanual_groups_and_per_arm_skills(tmp_path) -> None:
    ep = _server(tmp_path, PointArm(bimanual=True))
    r = ep.handle(
        {"op": "act", "groups": {"left.ee_delta": [0, 1, 0], "right.gripper": [1]}}
    )
    assert r["ok"]
    row = json.loads((tmp_path / "ep" / "steps.jsonl").read_text().splitlines()[0])
    assert row["action"]["right.gripper"] == [1.0] and row["action"][
        "left.gripper"
    ] == [-1.0]


def test_done_is_judged_after_settle_and_writes_the_record(tmp_path) -> None:
    ep = _server(tmp_path)
    ep.handle({"op": "move_to", "pos": [0.1, 0.0, 0.0]})
    r = ep.handle({"op": "done", "text": "there"})
    assert r["ok"]
    res = json.loads((tmp_path / "ep" / "result.json").read_text())
    assert res["success"] is True and res["outcome"] == "done"
    assert res["sim_steps"] == res["steps_used"] + 2  # two settle steps
    assert res["return"] > 0  # sparse indicator was probed each step
    assert not ep.handle({"op": "act", "action": [0, 0, 0, 0]})["ok"]  # finished
    ep.close()
    idx = json.loads((tmp_path / "ep" / "video_index.json").read_text())
    assert idx["n_frames"] >= 2


def test_give_up_and_budget_endings_score_zero(tmp_path) -> None:
    ep = _server(tmp_path / "a")
    ep.handle({"op": "move_to", "pos": [0.1, 0.0, 0.0]})
    ep.handle({"op": "give_up"})
    assert ep.result["success"] is False and ep.result["outcome"] == "gave_up"
    ep = _server(tmp_path / "b")
    r = ep.handle({"op": "act", "action": [0, 0, 0, 0], "repeat": 20})
    r = ep.handle({"op": "act", "action": [0, 0, 0, 0], "repeat": 20})
    r = ep.handle({"op": "act", "action": [0, 0, 0, 0], "repeat": 20})
    assert ep.finished and ep.outcome == "budget_exhausted"
    assert r["result"]["episode"] == "finished: step budget exhausted"


def test_success_mode_first_ends_at_the_first_success(tmp_path) -> None:
    ep = _server(tmp_path, PointArm(dense="shaped"), success_mode="first")
    ep.handle({"op": "act", "groups": {"arm.ee_delta": [1, 0, 0]}, "repeat": 20})
    assert ep.finished and ep.outcome == "success_reached" and ep.result["success"]
    rows = [
        json.loads(x)
        for x in (tmp_path / "ep" / "steps.jsonl").read_text().splitlines()
    ]
    assert all(
        r["reward"] <= 0 for r in rows
    )  # shaped reward is the backend's (negative distance)


def test_vision_mode_hides_privileged_state_except_for_the_oracle_token(
    tmp_path,
) -> None:
    ep = _server(tmp_path, obs_mode="vision")
    st = ep.handle({"op": "observe"})["result"]
    assert set(st["state"]) == {"hand_pos"} and st["image_path"]
    st = ep.handle({"op": "observe", "token": "tok"})["result"]
    assert "goal" in st["state"]
    trace = (tmp_path / "ep" / "trace.jsonl").read_text()
    assert "tok" not in trace


def test_roles_restrict_ops_and_are_recorded(tmp_path) -> None:
    ep = _server(tmp_path, roles={"planner": {"allow": ["info", "observe"]}})
    assert ep.handle({"op": "observe", "role": "planner"})["ok"]
    r = ep.handle({"op": "act", "action": [1, 0, 0, 0], "role": "planner"})
    assert not r["ok"] and "planner" in r["error"]
    assert not ep.handle({"op": "move_to", "pos": [0, 0, 0], "role": "planner"})["ok"]
    assert ep.handle({"op": "act", "action": [1, 0, 0, 0], "role": "operator"})["ok"]
    rows = [
        json.loads(x)
        for x in (tmp_path / "ep" / "steps.jsonl").read_text().splitlines()
    ]
    assert rows[0]["role"] == "operator"
    lines = [
        json.loads(x)
        for x in (tmp_path / "ep" / "trace.jsonl").read_text().splitlines()
    ]
    assert [x.get("role") for x in lines] == [
        "planner",
        "planner",
        "planner",
        "operator",
    ]


def test_reset_is_reproducible_per_seed(tmp_path) -> None:
    a = _server(tmp_path / "a", seed=3)
    b = _server(tmp_path / "b", seed=3)
    c = _server(tmp_path / "c", seed=4)
    assert a.initial_state_sha256 == b.initial_state_sha256 != c.initial_state_sha256


# ---- socket + the stdlib `robo` client -----------------------------------------------------------------------


def test_robo_cli_standalone_over_the_socket(tmp_path) -> None:
    import tempfile

    ep = _server(tmp_path)
    ep.config.linger_s = 0.2
    sock = os.path.join(tempfile.mkdtemp(prefix="bfemb"), "s.sock")
    ready = tmp_path / "ready"
    th = threading.Thread(
        target=serve, args=(ep, sock), kwargs={"ready_file": ready}, daemon=True
    )
    th.start()
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.05)
    # the agent image has only this file: run it by path, outside the package, with a bare environment
    robo = Path(sidecar.EMBODIED_DIR / "robo.py")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "ROBO_SOCKET": sock,
        "ROBO_ROLE": "operator",
    }

    def run(*args):
        return subprocess.run(
            [sys.executable, "-I", str(robo), *args],
            env=env,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

    out = run("info")
    assert (
        out.returncode == 0
        and "arm.ee_delta[3]" in out.stdout
        and "arm.move_to" in out.stdout
    )
    out = run("act", "arm.ee_delta=1,0,0", "gripper=1", "--repeat", "2", "--json")
    assert json.loads(out.stdout)["result"]["executed_steps"] == 2
    assert run("act", "0.5", "0", "0", "1").returncode == 0
    assert run("move-to", "0.1", "0", "0").returncode == 0
    assert run("skill", "gripper.set", "-1", "steps=2").returncode == 0
    bad = run("act", "arm.ee_delta=1,0", "0.2")
    assert bad.returncode == 2
    assert run("done", "finished").returncode == 0
    th.join(timeout=10)
    assert ep.result["success"] is True
    lines = [
        json.loads(x)
        for x in (tmp_path / "ep" / "trace.jsonl").read_text().splitlines()
    ]
    assert all(x.get("role") == "operator" for x in lines)


def test_robo_waits_for_stepping_requests_only(tmp_path, monkeypatch) -> None:
    """A long skill can take minutes: requests that step the simulator wait as long as an episode can run, the others
    keep a short timeout, and ROBO_TIMEOUT_S, when set, applies to all of them."""
    import tempfile

    from benchflow.embodied import robo as client

    monkeypatch.delenv("ROBO_TIMEOUT_S", raising=False)
    for op in ("act", "move_to", "grip", "skill", "done", "give_up"):
        assert client._timeout(op) == client.STEPPING_TIMEOUT_S == 3600
    for op in ("info", "status", "observe"):
        assert client._timeout(op) == client.TIMEOUT_S == 120
    monkeypatch.setenv("ROBO_TIMEOUT_S", "7")
    assert client._timeout("skill") == client._timeout("info") == 7
    monkeypatch.delenv("ROBO_TIMEOUT_S")

    # an episode server that never answers: the client waits its timeout, then reports it (no traceback)
    sock = os.path.join(tempfile.mkdtemp(prefix="bfemb"), "s.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    srv.listen(4)
    monkeypatch.setenv("ROBO_SOCKET", sock)
    monkeypatch.setattr(client, "STEPPING_TIMEOUT_S", 0.6)
    monkeypatch.setattr(client, "TIMEOUT_S", 0.1)
    try:
        t0 = time.time()
        r = client._send({"op": "skill", "name": "arm.move_to", "args": []})
        assert time.time() - t0 >= 0.55
        assert r == {
            "ok": False,
            "error": "no answer from the episode server within 0.6 s",
        }
        t0 = time.time()
        r = client._send({"op": "info"})
        assert time.time() - t0 < 0.5
        assert r == {
            "ok": False,
            "error": "no answer from the episode server within 0.1 s",
        }
    finally:
        srv.close()


# ---- the physical verifier -----------------------------------------------------------------------------------


def test_verifier_writes_rewards_from_a_finished_episode(tmp_path, monkeypatch) -> None:
    from benchflow.embodied import verifier

    ep = _server(tmp_path)
    ep.handle({"op": "move_to", "pos": [0.1, 0.0, 0.0]})
    ep.handle({"op": "done"})
    ep.close()
    (tmp_path / "ep" / "serve.exit").write_text("0")
    out, art = tmp_path / "logs" / "verifier", tmp_path / "logs" / "artifacts"
    art.mkdir(parents=True)
    for k, v in {
        "EPISODE_DIR": tmp_path / "ep",
        "VERIFIER_DIR": out,
        "ARTIFACTS_DIR": art,
        "WORKSPACE_DIR": tmp_path / "ws",
        "SOCKET": tmp_path / "none.sock",
    }.items():
        monkeypatch.setenv(f"ROBO_{k}", str(v))
    assert verifier.main() == 0
    assert (out / "reward.txt").read_text() == "1"
    rewards = json.loads((out / "reward.json").read_text())
    assert rewards["reward"] == 1.0 and 0 < rewards["budget_used"] <= 1
    assert (out / "episode" / "steps.jsonl").exists()
    details = json.loads((out / "reward-details.json").read_text())
    assert details["closed_by"] == "episode_server" and details["outcome"] == "done"


def test_verifier_without_a_result_is_an_infrastructure_error(
    tmp_path, monkeypatch
) -> None:
    from benchflow.embodied import verifier

    (tmp_path / "ep").mkdir()
    (tmp_path / "ep" / "serve.exit").write_text("1")
    monkeypatch.setenv("ROBO_EPISODE_DIR", str(tmp_path / "ep"))
    monkeypatch.setenv("ROBO_VERIFIER_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ROBO_ARTIFACTS_DIR", str(tmp_path / "art"))
    assert verifier.main() == 1
    assert not (tmp_path / "out" / "reward.txt").exists()


# ---- training export and seeded-rollout reports --------------------------------------------------------------


def _fake_trial(
    job: Path, name: str, seed: int, reward: float, *, vision=False, error=None
) -> Path:
    trial = job / f"{name}__{seed:04x}abcd"
    ep = _server(
        trial,
        run_dir=trial / "verifier" / "episode",
        seed=seed,
        **({"obs_mode": "vision"} if vision else {}),
    )
    ep.handle({"op": "act", "groups": {"arm.ee_delta": [1, 0, 0]}, "repeat": 3})
    ep.handle({"op": "done" if reward else "give_up"})
    ep.close()
    (trial / "result.json").write_text(
        json.dumps({"task_name": name, "rewards": {"reward": reward}, "error": error})
    )
    return trial


def test_export_rows_are_transitions_with_vision_filtering(tmp_path) -> None:
    trial = _fake_trial(tmp_path / "job", "demo--seed-1", 1, 0.0, vision=True)
    header, rows = episode_rows(trial / "verifier" / "episode")
    assert len(rows) == 3 + 2  # three steps and the settle period
    assert rows[0]["obs"] == {
        "hand_pos": header["initial_state"]["hand_pos"]
    }  # vision: no privileged goal
    assert rows[-1]["done"] and "final_obs" in rows[-1]
    assert rows[1]["obs"]["hand_pos"][0] > rows[0]["obs"]["hand_pos"][0]
    summary = write_rollout_embodied(trial, trajectory_id="demo__x")
    assert summary["episode_id"] == "demo__x" and summary["n_steps"] == 5
    assert (trial / "trainer" / "embodied_steps.jsonl").exists()
    assert write_rollout_embodied(tmp_path) is None


def test_seed_report_pass_at_k_and_reproducibility(tmp_path) -> None:
    job = tmp_path / "job"
    for seed, reward in [(0, 1.0), (1, 0.0), (2, 1.0)]:
        _fake_trial(job, f"demo--seed-{seed}", seed, reward)
    _fake_trial(job, "other--seed-0", 0, 0.0)
    rep = rollouts.seed_report(job)
    demo = rep["tasks"]["demo"]
    assert demo["n"] == 3 and demo["passes"] == 2
    assert demo["pass_at_k"] == {
        "1": pytest.approx(2 / 3, abs=1e-6),
        "2": 1.0,
        "3": 1.0,
    }
    assert demo["std"] == pytest.approx(np.std([1, 0, 1], ddof=1))
    assert demo["distinct_initial_states"] == 3
    assert rep["summary"]["pass_at_k"]["1"] == pytest.approx((2 / 3 + 0) / 2, abs=1e-6)
    assert "demo" in rollouts.format_report(rep)
    res = export_job(job, tmp_path / "out")
    assert res["episodes"] == 4
    eps = [
        json.loads(x)
        for x in (tmp_path / "out" / "episodes.jsonl").read_text().splitlines()
    ]
    assert {e["seed"] for e in eps} == {0, 1, 2}


def test_pass_at_k_and_parse_seeds() -> None:
    assert rollouts.pass_at_k(5, 0, 3) == 0.0
    assert rollouts.pass_at_k(5, 5, 1) == 1.0
    assert rollouts.pass_at_k(4, 1, 2) == pytest.approx(0.5)
    assert rollouts.parse_seeds("0-2,7") == [0, 1, 2, 7]
    for bad in ("", "3-1", "1,1", "a"):
        with pytest.raises(ValueError):
            rollouts.parse_seeds(bad)


# ---- the sidecar task format ---------------------------------------------------------------------------------

TOY_TASK_MD = """---
schema_version: '1.3'
task:
  name: toy/reach
agent:
  timeout_sec: 120
toy:
  id: toy-reach
  seed: 0
  max_steps: 50
---

Move the arm to the goal.
"""


class ToyFormat(sidecar.EmbodiedTaskFormat):
    name = "toy"
    block_key = "toy"
    episode_factory = "toy_sim.episode:make"
    prompt_prefix = "You control a robot.\n\n"

    def build_runtime(self, dst: Path) -> None:
        (dst / "Dockerfile").write_text(
            "FROM python:3.12-slim\n" + sidecar.SIM_DOCKERFILE_SNIPPET
        )


def _toy_task(root: Path) -> Path:
    d = root / "toy-reach"
    (d / "oracle").mkdir(parents=True)
    (d / "task.md").write_text(TOY_TASK_MD)
    (d / "oracle" / "solve.sh").write_text(
        "#!/bin/bash\nrobo move-to 0.1 0 0\nrobo done\n"
    )
    return d


def test_embodied_format_materializes_a_native_package(tmp_path) -> None:
    fmt = ToyFormat()
    src = _toy_task(tmp_path / "src")
    assert fmt.detect(src) and not fmt.detect(tmp_path)
    pkg = fmt.materialize(src, tmp_path / "cache")
    assert (
        pkg.name == "toy-reach" and fmt.materialize(src, tmp_path / "cache") == pkg
    )  # cached
    task_md = (pkg / "task.md").read_text()
    assert (
        "service: simulator" in task_md
        and "\ntoy:" not in task_md
        and "You control a robot." in task_md
    )
    compose = (pkg / "environment" / "docker-compose.yaml").read_text()
    assert (
        "toy_sim.episode:make" in compose
        and "network_mode: none" in compose
        and "__" not in compose
    )
    assert "max_steps: 50" in compose
    assert (
        (pkg / "environment" / "robo").read_text().startswith("#!/usr/bin/env python3")
    )
    solve = (pkg / "oracle" / "solve.sh").read_text()
    assert "ROBO_ORACLE_TOKEN=" in solve and "robo done" in solve
    runtime = Path(compose.split("context: ")[1].split("\n")[0].strip('"'))
    assert (
        runtime / "benchflow_embodied" / "benchflow" / "embodied" / "server.py"
    ).exists()
    assert (
        "Stub"
        in (runtime / "benchflow_embodied" / "benchflow" / "__init__.py").read_text()
    )
    assert not (
        runtime / "benchflow_embodied" / "benchflow" / "embodied" / "templates"
    ).exists()
    assert os.access(runtime / "embodied-sim-entry", os.X_OK)


def test_embodied_format_seed_noop_and_flat_variants(tmp_path) -> None:
    fmt = ToyFormat()
    src = _toy_task(tmp_path / "src")
    s3 = fmt.materialize_variant(src, tmp_path / "cache", seed=3)
    assert s3.name == "toy-reach--seed-3"
    assert "seed: 3" in (s3 / "environment" / "docker-compose.yaml").read_text()
    assert "base_task: toy-reach" in (s3 / "task.md").read_text()
    assert sidecar.split_seed_name(s3.name) == ("toy-reach", 3)
    noop = fmt.materialize_variant(src, tmp_path / "cache", noop=True)
    assert "\nrobo done" not in (noop / "oracle" / "solve.sh").read_text()
    flat = fmt.materialize_variant(src, tmp_path / "bundle", flat=True)
    assert flat == tmp_path / "bundle" / "tasks" / "toy-reach"
    assert (
        'context: "../../../runtime-'
        in (flat / "environment" / "docker-compose.yaml").read_text()
    )


def test_task_formats_seeded_materialization(tmp_path, monkeypatch) -> None:
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.task import formats

    monkeypatch.setenv(formats.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(formats, "_registered", [ToyFormat()])
    monkeypatch.setattr(formats, "_entry_point_formats", [])
    src = _toy_task(tmp_path / "suite")
    assert formats.materialize_task_dir(src, seed=2).name == "toy-reach--seed-2"
    ev = Evaluation(
        tasks_dir=tmp_path / "suite",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle", seeds=[0, 1, 2]),
    )
    assert [d.name for d in ev._get_task_dirs()] == [
        f"toy-reach--seed-{i}" for i in range(3)
    ]
    ev = Evaluation(
        tasks_dir=src,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle", seeds=[5]),
    )
    assert [d.name for d in ev._get_task_dirs()] == ["toy-reach--seed-5"]
    native = tmp_path / "native"
    native.mkdir()
    with pytest.raises(ValueError, match="native task package"):
        formats.materialize_task_dir(native, seed=1)


def test_eval_plan_validates_seeds(tmp_path) -> None:
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan

    plan = build_eval_plan(
        EvalCreateRequest(tasks_dir=tmp_path, agent="oracle", seeds="0-2")
    )
    assert plan.eval_seeds == [0, 1, 2]
    assert plan.make_eval_config().seeds == [0, 1, 2]
    with pytest.raises(EvalPlanError, match="Invalid --seeds"):
        build_eval_plan(
            EvalCreateRequest(tasks_dir=tmp_path, agent="oracle", seeds="2-0")
        )
    with pytest.raises(EvalPlanError, match="worker-concurrency"):
        build_eval_plan(
            EvalCreateRequest(
                tasks_dir=tmp_path, agent="oracle", seeds="0", worker_concurrency=2
            )
        )


# ---- regressions found in review of this change ---------------------------------------------------------------


class SkillOnly:
    """A BEHAVIOR-like robot: no low-level control worth the name; success comes from symbolic skills."""

    name = "skillonly"

    def __init__(self):
        self.done = False

    def embodiment(self) -> Embodiment:
        return Embodiment(
            "r1",
            "skill_only",
            [ActionGroup("base.wait", ["wait"], [0.0], [1.0], "wait")],
            skills=[Skill("toggle_on", [SkillArg("obj", "object")])],
        )

    def reset(self, seed):
        self.done = False

    def step(self, action):
        return StepResult(False)

    def run_skill(self, name, args):
        self.done = args == ["lamp"]
        return {"ok": True, "message": f"{name} {args}"}

    def observe(self):
        return {"lamp_on": self.done}

    def render(self, camera=None):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def success(self):
        return self.done


def test_backend_skills_count_for_success_ever_and_first_mode(tmp_path) -> None:
    """Review of the embodied layer: backend skills used to log success False and never end a `first` episode."""
    ep = _server(tmp_path, SkillOnly(), success_mode="first")
    r = ep.handle({"op": "skill", "name": "toggle_on", "args": ["lamp"]})
    assert r["ok"] and r["result"]["episode"].startswith("finished")
    assert ep.result["success"] and ep.outcome == "success_reached"
    row = json.loads((tmp_path / "ep" / "steps.jsonl").read_text().splitlines()[0])
    assert row["op"] == "skill:toggle_on" and row["success"] and row["reward"] == 1.0
    assert row["action"] == {"skill": ["lamp"]}


def test_sparse_probe_drives_success_ever_and_first_mode(tmp_path) -> None:
    """Review of the embodied layer: a backend whose step() never reports success must still end a `first` episode."""
    ep = _server(tmp_path, PointArm(dense="sparse"), success_mode="first")
    ep.handle({"op": "act", "groups": {"arm.ee_delta": [1, 0, 0]}, "repeat": 20})
    assert ep.finished and ep.outcome == "success_reached" and ep.success_ever


def test_legacy_move_to_drops_args_the_skill_does_not_declare(tmp_path) -> None:
    """Review of the embodied layer: protocol-1 `robo move-to` always sends max_steps and tol."""

    class Minimal(PointArm):
        def embodiment(self):
            e = super().embodiment()
            e.skills = [
                Skill(
                    "arm.move_to",
                    [SkillArg("x"), SkillArg("y"), SkillArg("z")],
                    aliases=["move_to"],
                    impl="builtin",
                    binds={"arm": "arm.ee_delta"},
                )
            ]
            return e

    ep = _server(tmp_path, Minimal())
    r = ep.handle({"op": "move_to", "pos": [0.05, 0, 0], "max_steps": 100, "tol": 0.01})
    assert r["ok"] and r["result"]["reached"], r


def test_role_allow_may_be_a_single_op(tmp_path) -> None:
    ep = _server(tmp_path, roles={"viewer": {"allow": "observe"}})
    assert ep.handle({"op": "observe", "role": "viewer"})["ok"]
    assert not ep.handle({"op": "info", "role": "viewer"})[
        "ok"
    ]  # "info" is not a substring match of "observe"


def test_seed_report_counts_one_result_per_task_despite_retries(tmp_path) -> None:
    """Review of the embodied layer: a retried trial's first attempt must not inflate n or pass@k."""
    job = tmp_path / "job"
    first = _fake_trial(job, "demo--seed-0", 0, 0.0)
    older = json.loads((first / "result.json").read_text())
    older["rewards"] = None
    older["error"] = "infra"
    (first / "result.json").write_text(json.dumps(older))
    os.utime(first / "result.json", (1, 1))
    _fake_trial(job, "demo--seed-0", 0, 1.0).rename(job / "demo--seed-0__retry")
    rep = rollouts.seed_report(job)
    assert rep["tasks"]["demo"]["n"] == 1 and rep["tasks"]["demo"]["passes"] == 1


def test_package_key_includes_the_task_name(tmp_path) -> None:
    """Review of the embodied layer: identical task folders whose id comes from the folder name must not share a
    package (the second load returned a folder that was never written)."""
    import shutil

    fmt = ToyFormat()
    a = tmp_path / "a" / "toy-one"
    (a / "oracle").mkdir(parents=True)
    (a / "task.md").write_text(TOY_TASK_MD.replace("  id: toy-reach\n", ""))
    (a / "oracle" / "solve.sh").write_text("#!/bin/bash\nrobo done\n")
    b = tmp_path / "b" / "toy-two"
    shutil.copytree(a, b)
    pa, pb = (
        fmt.materialize(a, tmp_path / "cache"),
        fmt.materialize(b, tmp_path / "cache"),
    )
    assert pa.name == "toy-one" and pb.name == "toy-two"
    assert (
        (pa / "task.md").exists()
        and (pb / "task.md").exists()
        and pa.parent != pb.parent
    )


# ---- closed-loop backend skills, non-finite actions, judge errors, hold values -----------------------------------


class ClosedLoopArm(PointArm):
    """A backend skill written as a generator: `reach X` yields ee_delta actions until the hand is at x = X."""

    def embodiment(self) -> Embodiment:
        emb = super().embodiment()
        emb.skills = [Skill("reach", [SkillArg("x", "float", "m")], impl="backend")]
        return emb

    def start_skill(self, name, args):
        if name != "reach":
            raise ValueError(f"unknown skill {name!r}")
        x = float(args[0])

        def gen():
            for _ in range(40):
                dx = x - self.pos[0]
                if abs(dx) < 0.005:
                    break
                yield [float(np.clip(dx / 0.01, -1, 1)), 0.0, 0.0, -1.0]
            return {"reached": bool(abs(x - self.pos[0]) < 0.01)}

        return gen()


def test_closed_loop_backend_skill_runs_each_action_as_a_step(tmp_path) -> None:
    ep = _server(tmp_path, ClosedLoopArm())
    r = ep.handle({"op": "skill", "name": "reach", "args": ["0.1"]})
    assert r["ok"] and r["result"]["reached"] and r["result"]["skill"] == "reach"
    n = r["result"]["executed_steps"]
    assert n >= 10 and ep.steps == n  # every control step counts against the budget
    rows = [
        json.loads(x)
        for x in (tmp_path / "ep" / "steps.jsonl").read_text().splitlines()
    ]
    assert len(rows) == n and all(row["op"] == "skill:reach" for row in rows)
    assert ep.success_ever
    bad = ep.handle({"op": "skill", "name": "reach", "args": ["far"]})
    assert not bad[
        "ok"
    ]  # argument types are checked against the spec before the skill starts


def test_non_finite_actions_are_rejected(tmp_path) -> None:
    ep = _server(tmp_path)
    r = ep.handle({"op": "act", "action": [float("nan"), 0, 0, 0]})
    assert not r["ok"] and "finite" in r["error"] and ep.steps == 0


def test_a_judge_that_raises_scores_zero_and_is_recorded(tmp_path) -> None:
    class Dies(PointArm):
        last_judge: ClassVar[dict] = {"rule": "demo"}

        def judge(self, outcome, text):
            raise RuntimeError("worker died")

    ep = _server(tmp_path, Dies())
    ep.handle({"op": "done"})
    assert ep.result["success"] is False
    assert ep.result["judge_error"].startswith("RuntimeError")
    assert ep.result["judge_detail"] == {"rule": "demo"}


def test_hold_value_policy() -> None:
    brake = ActionGroup(
        "car.pedals",
        ["throttle", "brake"],
        [0.0, 0.0],
        [1.0, 1.0],
        "pedals",
        hold="value",
        hold_value=[0.0, 1.0],
    )
    emb = Embodiment("car", "vehicle", [brake]).validate()
    assert emb.hold_action() == [0.0, 1.0]
    assert Embodiment.from_dict(emb.to_dict()).hold_action() == [0.0, 1.0]
    with pytest.raises(SpecError):
        Embodiment(
            "car",
            "vehicle",
            [ActionGroup("p", ["a"], [0.0], [1.0], "pedals", hold="value")],
        ).validate()


def test_camera_names_with_slashes_make_flat_image_files(tmp_path) -> None:
    class Cams(PointArm):
        def embodiment(self) -> Embodiment:
            emb = super().embodiment()
            emb.sensors.cameras = [Camera("robot/head", mount="head", calibrated=False)]
            return emb

    ep = _server(tmp_path, Cams())
    ep.config.cameras = ["robot/head"]
    r = ep.handle({"op": "observe", "image": True, "camera": "robot/head"})
    assert r["ok"] and r["result"]["image_path"].endswith("_robot_head.png")

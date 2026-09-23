"""Two-arm bridge and setup contracts: no hardware or provider calls."""

import json

import pytest

from benchflow.robotics.bridge import TrialBridge
from benchflow.robotics.runner import load_setup, task_prompt


def _frames(root, tag):
    root.mkdir()
    paths = []
    for camera in ("wrist", "side"):
        path = root / f"001_{tag}_{camera}.jpg"
        path.write_bytes(b"\xff\xd8frame\xff\xd9")
        paths.append(str(path))
    return paths


@pytest.fixture
def two_arms(tmp_path):
    calls = {"right": [], "left": []}
    frames = {name: _frames(tmp_path / name, name) for name in calls}
    tips = {"right": [0.30, 0.20, 0.10], "left": [0.30, -0.30, 0.10]}

    def make(name):
        def transport(command, args):
            calls[name].append((command, args))
            return {
                "ok": True,
                "arm": "metal",
                "frames": frames[name],
                "tip_m": tips[name],
                "joints": {"shoulder_pan": 0},
            }

        return transport

    bridge = TrialBridge(
        output=tmp_path / "trial",
        arms={
            "right": {"transport": make("right"), "frames_root": tmp_path / "right"},
            "left": {
                "transport": make("left"),
                "frames_root": tmp_path / "left",
                "offset_m": [0.0, 0.58, 0.0],
            },
        },
        robot_id="metal",
        allow_motion=True,
        recording_healthy=lambda: True,
    )
    bridge.activate()
    yield bridge, calls
    bridge.close()


def req(command, args, arm=None, request_id=None):
    body = {
        "request_id": request_id or f"{arm}-{command}-{args}",
        "command": command,
        "args": args,
    }
    if arm is not None:
        body["arm"] = arm
    return body


def test_commands_route_to_the_named_arm_and_tag_observations(two_arms):
    bridge, calls = two_arms
    left = bridge.execute(req("observe", [], "left"))
    assert left["ok"] and left["arm_name"] == "left"
    assert {im["camera"] for im in left["images"]} == {"wrist", "side"}
    assert all(im["arm"] == "left" and "_left_" in im["name"] for im in left["images"])
    assert calls["left"] == [("observe", [])] and calls["right"] == []
    default = bridge.execute(req("status", []))
    assert default["arm_name"] == "right"  # first listed arm is the default
    with pytest.raises(ValueError, match="Unknown arm"):
        bridge.execute(req("status", [], "middle"))


def test_tip_near_the_other_gripper_is_refused_before_reaching_the_harness(two_arms):
    bridge, calls = two_arms
    # Seed both positions: right tip at (0.30, 0.20); left tip at (0.30, -0.30)
    # in its own frame, i.e. (0.30, 0.28) in the right arm's frame.
    bridge.execute(req("observe", [], "right"))
    bridge.execute(req("observe", [], "left"))
    blocked = bridge.execute(req("tip", [0.30, 0.25, 0.05, -80], "right", "blocked"))
    assert blocked["ok"] is False and "left arm" in blocked["error"]
    assert not any(c[0] == "tip" for c in calls["right"])
    events = [
        json.loads(line)
        for line in (bridge.output / "commands.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "command_rejected" for e in events)
    allowed = bridge.execute(req("tip", [0.30, -0.10, 0.05, -80], "right", "allowed"))
    assert allowed["ok"] and calls["right"][-1][0] == "tip"


def test_finish_gathers_final_images_from_every_arm(two_arms):
    bridge, calls = two_arms
    final = bridge.execute(req("finish", [], "right"))
    assert final["ok"] and set(final["arms"]) == {"right", "left"}
    assert {(im["arm"], im["camera"]) for im in final["images"]} == {
        ("right", "wrist"),
        ("right", "side"),
        ("left", "wrist"),
        ("left", "side"),
    }
    assert calls["left"] == [("observe", [])]


def test_single_arm_bridge_keeps_legacy_naming(tmp_path):
    frames = _frames(tmp_path / "frames", "after")

    def transport(command, args):
        return {"ok": True, "arm": "metal", "frames": frames, "tip_m": [0.2, 0, 0.2]}

    bridge = TrialBridge(
        output=tmp_path / "trial",
        transport=transport,
        frames_root=tmp_path / "frames",
        robot_id="metal",
        allow_motion=True,
        recording_healthy=lambda: True,
    )
    bridge.activate()
    receipt = bridge.execute(req("tip", [0.2, 0, 0.2, -75]))
    assert receipt["ok"] and "arm_name" not in receipt
    assert [im["name"] for im in receipt["images"]] == [
        "0001_wrist.jpg",
        "0001_side.jpg",
    ]
    bridge.close()


def test_load_setup_normalizes_multi_arm_layout(tmp_path):
    episode = tmp_path / "episode.jsonl"
    episode.touch()
    path = tmp_path / "setup.json"
    path.write_text(
        json.dumps(
            {
                "setup_id": "two",
                "arm_type": "metal",
                "primary_arm": "right",
                "arms": {
                    "left": {
                        "robot_id": "a",
                        "socket_path": "a.sock",
                        "frames_root": str(tmp_path),
                        "episode_log": str(episode),
                        "offset_m": [0, 0.58, 0],
                    },
                    "right": {
                        "robot_id": "b",
                        "socket_path": "b.sock",
                        "frames_root": str(tmp_path),
                        "episode_log": str(episode),
                    },
                },
                "cameras": {"wrist": "u", "side": "u", "wrist_left": "u"},
                "public_facts": {},
            }
        )
    )
    setup = load_setup(path)
    assert setup["primary_arm"] == "right" and setup["robot_id"] == "b"
    assert setup["socket_path"] == "b.sock" and set(setup["arms"]) == {"left", "right"}
    path.write_text(
        json.dumps(
            {
                "setup_id": "x",
                "arm_type": "metal",
                "arms": {},
                "cameras": {},
                "public_facts": {},
            }
        )
    )
    with pytest.raises(ValueError):
        load_setup(path)


def test_task_prompt_prefers_task_md_section(tmp_path):
    (tmp_path / "task.md").write_text(
        "---\nx: 1\n---\n# T\n\n## prompt\n\nDo the thing.\n\n## notes\n\nignored\n"
    )
    assert task_prompt(tmp_path, "fallback") == "Do the thing.\n"
    (tmp_path / "task.md").unlink()
    assert task_prompt(tmp_path, "fallback") == "fallback"

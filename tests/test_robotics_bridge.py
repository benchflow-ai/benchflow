"""Physical-adapter contracts: no hardware or provider calls in these tests."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from benchflow.robotics.bridge import TrialBridge, TrialLease, validate_command
from benchflow.robotics.runner import score_trial


@pytest.fixture
def rig(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    paths = []
    for camera in ("wrist", "side"):
        path = frames / f"001_after_{camera}.jpg"
        path.write_bytes(b"\xff\xd8frame\xff\xd9")
        paths.append(str(path))
    calls = []

    def transport(command, args):
        calls.append((command, args))
        return {
            "ok": True,
            "arm": "metal",
            "frames": paths,
            "joints": {"shoulder_pan": 0},
        }

    bridge = TrialBridge(
        output=tmp_path / "trial",
        transport=transport,
        frames_root=frames,
        robot_id="metal",
        allow_motion=True,
        recording_healthy=lambda: True,
    )
    bridge.activate()
    yield bridge, calls
    bridge.close()


def req(command="tip", args=None, request_id="one"):
    return {
        "request_id": request_id,
        "command": command,
        "args": [0.2, 0, 0.2, -75] if args is None else args,
    }


def test_lost_response_replay_does_not_repeat_motion(rig):
    bridge, calls = rig
    first = bridge.execute(req())
    second = bridge.execute(req())
    assert first == second
    assert len(calls) == 1
    assert {im["camera"] for im in first["images"]} == {"wrist", "side"}
    with pytest.raises(ValueError, match="reused"):
        bridge.execute(req(args=[0.3, 0, 0.2, -75]))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "command,args",
    [
        ("quit", []),
        ("clear-faults", []),
        ("goto", ["shoulder_lift=0"]),
        ("tip", [0.2, 0, float("nan"), -75]),
        ("tip", [0.2, 0, float("inf"), -75]),
        ("tip", [0.2, 0, True, -75]),
        ("tip", [0.2, 0, 0.2, "1; rest"]),
        ("gripper", [200]),
        ("monitor", [100]),
    ],
)
def test_disallowed_commands_never_reach_hardware(rig, command, args):
    bridge, calls = rig
    with pytest.raises(ValueError):
        bridge.execute(req(command, args))
    assert calls == []


def test_roll_only_maps_to_wrist_joint():
    assert validate_command("roll", [25]) == ("goto", ["wrist_roll=25.0"])


def test_recording_loss_prevents_new_motion(rig):
    bridge, calls = rig
    bridge.recording_healthy = lambda: False
    assert not bridge.execute(req())["ok"]
    assert not calls
    bridge.recording_healthy = lambda: True
    assert not bridge.execute(req(request_id="two"))["ok"]
    assert not calls  # Recovery does not silently resume a halted trial.


def test_ambiguous_transport_failure_freezes_trial(rig):
    bridge, calls = rig

    def lost_receipt(command, args):
        calls.append((command, args))  # Motion could have executed.
        raise TimeoutError

    bridge.transport = lost_receipt
    first = bridge.execute(req())
    assert not first["ok"]
    assert bridge.execute(req()) == first
    assert not bridge.execute(req(request_id="two"))["ok"]
    assert len(calls) == 1


def test_missing_post_move_camera_freezes_control(rig):
    bridge, calls = rig
    original = bridge.transport

    def missing(command, args):
        result = original(command, args)
        result["frames"] = result["frames"][:1]
        return result

    bridge.transport = missing
    assert not bridge.execute(req())["ok"]
    assert bridge.halted
    assert len(calls) == 1


def test_finish_revokes_motion_and_still_returns_final_images(rig):
    bridge, calls = rig
    result = bridge.execute(req("finish", []))
    assert result["ok"] and len(result["images"]) == 2
    assert not bridge.execute(req(request_id="two"))["ok"]
    assert calls == [("observe", [])]


def test_read_only_and_stop_file_are_enforced(rig):
    bridge, calls = rig
    bridge.allow_motion = False
    assert not bridge.execute(req())["ok"]
    assert bridge.execute(req("observe", [], "two"))["ok"]
    (bridge.output / "STOP").touch()
    assert not bridge.execute(req("observe", [], "three"))["ok"]
    assert len(calls) == 1


def test_concurrent_commands_are_rejected_not_queued(rig):
    bridge, calls = rig
    started, release = threading.Event(), threading.Event()
    original = bridge.transport

    def blocking(command, args):
        started.set()
        release.wait(3)
        return original(command, args)

    bridge.transport = blocking
    thread = threading.Thread(target=bridge.execute, args=(req(),))
    thread.start()
    try:
        assert started.wait(2)
        result = bridge.execute(req(request_id="two"))
        assert not result["ok"] and "never queued" in result["error"]
    finally:
        release.set()
        thread.join(3)
    assert len(calls) == 1


def test_http_authentication_and_command_receipt(rig):
    bridge, calls = rig
    port = bridge.serve()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(token):
        return opener.open(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/command",
                json.dumps(req()).encode(),
                {"Authorization": "Bearer " + token},
            ),
            timeout=3,
        )

    with pytest.raises(urllib.error.HTTPError) as error:
        request("wrong")
    assert error.value.code == 403 and calls == []
    with request(bridge.token) as response:
        assert json.load(response)["ok"]
    assert len(calls) == 1
    assert bridge.token not in (bridge.output / "commands.jsonl").read_text()


def test_one_lease_per_robot(tmp_path):
    with TrialLease(tmp_path, "arm1", "trial1"):
        with (
            pytest.raises(RuntimeError, match="already leased"),
            TrialLease(tmp_path, "arm1", "trial2"),
        ):
            pytest.fail("Second owner acquired live robot")
        with TrialLease(tmp_path, "arm2", "trial3"):
            pass
    with TrialLease(tmp_path, "arm1", "trial4"):
        pass


def make_scoring_trial(path: Path, footage=True):
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "physical_trial",
                "status": "awaiting_assessment",
                "footage_complete": footage,
                "expected": {"blue": "left", "green": "right", "yellow": "right"},
            }
        )
    )
    (path / "metrics.json").write_text(
        json.dumps({"usage_source": "provider_response", "cost_usd": None})
    )


def test_partial_physical_result_and_human_intervention_are_distinct(tmp_path):
    make_scoring_trial(tmp_path)
    result = score_trial(
        tmp_path,
        placements={"blue": "left", "green": "right", "yellow": "table"},
        reviewer="operator",
        interventions=1,
        cups_upright=True,
        evidence="side.mp4 2:00",
    )
    assert result["object_accuracy"] == pytest.approx(2 / 3)
    assert not result["autonomous_success"] and not result["task_success"]
    assert result["benchmark_valid"]
    assert (tmp_path / "reward.txt").read_text().strip() == "0.0"


def test_incomplete_video_cannot_be_published_as_valid_success(tmp_path):
    make_scoring_trial(tmp_path, footage=False)
    result = score_trial(
        tmp_path,
        placements={"blue": "left", "green": "right", "yellow": "right"},
        reviewer="operator",
        interventions=0,
        cups_upright=True,
        evidence="final images",
    )
    assert result["task_success"]
    assert not result["benchmark_valid"]
    assert not (tmp_path / "reward.txt").exists()
    with pytest.raises(FileExistsError):
        score_trial(
            tmp_path,
            placements={"blue": "left", "green": "right", "yellow": "right"},
            reviewer="operator",
            interventions=0,
            cups_upright=True,
            evidence="final images",
        )


def test_external_interruption_is_excluded_from_comparison(tmp_path):
    make_scoring_trial(tmp_path)
    result = score_trial(
        tmp_path,
        placements={"blue": "table", "green": "table", "yellow": "table"},
        reviewer="host-reviewer",
        interventions=1,
        cups_upright=True,
        evidence="side.mp4 08:00",
        external_interruption="Person entered workspace",
    )
    assert result["object_accuracy"] == 0
    assert not result["benchmark_valid"]
    assert not (tmp_path / "reward.txt").exists()

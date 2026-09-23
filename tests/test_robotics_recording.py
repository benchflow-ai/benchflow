"""Camera and finalization contracts for the new physical adapter."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from benchflow.robotics import runner
from benchflow.robotics.recording import RecordingSidecar, normalize_jpeg


def test_usb_camera_padding_preserves_exact_jpeg():
    frame = b"\xff\xd8camera-frame\xff\xd9"
    assert normalize_jpeg(frame + b"\x00" * 4) == frame
    assert normalize_jpeg(frame) == frame


@pytest.mark.parametrize(
    "payload",
    [b"", b"\xff\xd8truncated", b"bad\xff\xd9", b"\xff\xd8data\xff\xd9garbage"],
)
def test_incomplete_or_corrupt_frames_rejected(payload):
    with pytest.raises(ValueError):
        normalize_jpeg(payload)


def test_recent_frame_does_not_hide_capture_errors(tmp_path):
    recorder = RecordingSidecar(tmp_path, {"wrist": "unused", "side": "unused"})
    recorder.process = SimpleNamespace(poll=lambda: None)
    for name in recorder.cameras:
        (tmp_path / f"{name}-status.json").write_text(
            json.dumps({"frames": 10, "last_frame": time.monotonic(), "errors": 0})
        )
    assert recorder.healthy()
    (tmp_path / "wrist-status.json").write_text(
        json.dumps({"frames": 11, "last_frame": time.monotonic(), "errors": 1})
    )
    assert not recorder.healthy()


@pytest.mark.parametrize(
    "capture_ok,export_ok", [(True, True), (False, True), (True, False)]
)
def test_smoke_success_requires_finalized_footage(
    tmp_path, monkeypatch, capture_ok, export_ok
):
    frames = tmp_path / "frames"
    frames.mkdir()
    paths = []
    for name in ("wrist", "side"):
        frame = frames / f"001_after_{name}.jpg"
        frame.write_bytes(b"\xff\xd8image\xff\xd9")
        paths.append(str(frame))
    episode = tmp_path / "episode.jsonl"
    episode.touch()
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "setup_id": "fixture",
                "robot_id": "fixture-metal",
                "arm_type": "metal",
                "socket_path": "unused",
                "frames_root": str(frames),
                "episode_log": str(episode),
                "cameras": {"wrist": "unused", "side": "unused"},
                "public_facts": {},
            }
        )
    )
    task = tmp_path / "task"
    task.mkdir()
    (task / "scenario.json").write_text(json.dumps({"expected": {"blue": "left"}}))
    (task / "task.md").write_text("fixture")
    calls = []

    def transport(command, args):
        calls.append(command)
        return {"ok": True, "arm": "metal", "frames": paths}

    monkeypatch.setattr(runner, "HarnessTransport", lambda _: transport)
    monkeypatch.setattr(
        runner,
        "RecordingSidecar",
        lambda *args: SimpleNamespace(
            start=lambda: None,
            healthy=lambda: True,
            stop=lambda: {
                "complete": capture_ok,
                "exports": {name: {"ok": export_ok} for name in ("wrist", "side")},
            },
        ),
    )

    async def no_delay(_):
        pass

    monkeypatch.setattr(runner.asyncio, "sleep", no_delay)
    options = dict(
        setup_path=setup,
        task_path=task,
        output_root=tmp_path / "trials",
        agent="codex",
        model="",
        reasoning_effort="max",
        reset_id="fixture",
        operator="test",
        smoke=True,
        lock_root=tmp_path / "locks",
    )
    if capture_ok and export_ok:
        asyncio.run(runner.run_trial(**options))
    else:
        with pytest.raises(RuntimeError, match="smoke failed"):
            asyncio.run(runner.run_trial(**options))
    manifest = json.loads(
        next((tmp_path / "trials").glob("*/manifest.json")).read_text()
    )
    assert manifest["status"] == (
        "smoke_passed" if capture_ok and export_ok else "smoke_failed"
    )
    assert calls == ["observe", "observe", "observe"]

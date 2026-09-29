"""The runner writes the trial record on every exit path, with fakes only.

Fake harness transport, fake camera sidecar and a fake ``benchflow.run`` stand
in for the arm, the cameras and the agent. The bridge is real and is driven
over HTTP exactly as the container client drives it.
"""

import asyncio
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchflow
from benchflow._utils.scoring import (
    classify_audit_outcome,
    classify_score_outcome,
    count_audit_outcomes,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.embodiment import TRIAL_RECORD_FILENAME
from benchflow.metrics import collect_metrics
from benchflow.models import RolloutResult
from benchflow.robotics import runner
from benchflow.robotics.tasks import build_tasks
from benchflow.sandbox.docker import DockerSandbox


def _setup(tmp_path: Path, **extra) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir()
    for camera in ("wrist", "side"):
        (frames / f"001_after_{camera}.jpg").write_bytes(b"\xff\xd8image\xff\xd9")
    (tmp_path / "episode.jsonl").write_text("")
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "setup_id": "fixture",
                "robot_id": "fixture-metal",
                "arm_type": "metal",
                "socket_path": "unused",
                "frames_root": str(frames),
                "episode_log": str(tmp_path / "episode.jsonl"),
                "cameras": {
                    "wrist": "http://127.0.0.1:8787/right.jpg",
                    "side": "http://127.0.0.1:8787/front.jpg",
                },
                "public_facts": {"table_z_m": 0.01},
                **extra,
            }
        )
    )
    return setup


class FakeRecorder:
    """Writes camera indexes like the sidecar; ``crash`` leaves them cut short."""

    def __init__(self, output: Path, cameras, fps=2, *, crash=False):
        self.output, self.cameras, self.crash = output, cameras, crash

    def start(self):
        self.output.mkdir(parents=True)

    def healthy(self):
        return True

    def stop(self):
        for index, name in enumerate(self.cameras):
            frame = {"frame": 1, "monotonic": 10.0 + index, "utc_epoch": 1.8e9}
            tail = '{"frame": 2, "mono' if self.crash else ""
            (self.output / f"{name}.jsonl").write_text(json.dumps(frame) + "\n" + tail)
            (self.output / f"{name}.mjpeg").write_bytes(b"\xff\xd8\xff\xd9")
        if self.crash:
            return {"complete": False, "error": "Missing capture summary"}
        (self.output / "capture-summary.json").write_text(
            json.dumps(
                {
                    "cameras": {
                        name: {"frames": 1, "errors": 0, "maximum_gap_s": 0.5}
                        for name in self.cameras
                    }
                }
            )
        )
        return {
            "complete": True,
            "exports": {name: {"ok": True} for name in self.cameras},
        }


def _install_fakes(monkeypatch, tmp_path, sdk_body, *, crash=False):
    images = [str(p) for p in sorted((tmp_path / "frames").glob("*.jpg"))]

    def transport(command, args):
        return {"ok": True, "armed": True, "arm": "metal", "frames": images}

    uploaded = {}

    async def upload(source, destination):
        uploaded[destination] = json.loads(source.read_text())

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(exit_code=0)

    async def sdk_run(config):
        for hook in config.pre_agent_hooks:
            await hook(SimpleNamespace(upload_file=upload, exec=sandbox_exec))
        connection = uploaded["/app/robot-connection.json"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(request_id, command, args):
            body = {"request_id": request_id, "command": command, "args": args}
            with opener.open(
                urllib.request.Request(
                    connection["url"] + "/command",
                    json.dumps(body).encode(),
                    {"Authorization": "Bearer " + connection["token"]},
                ),
                timeout=3,
            ) as response:
                return json.load(response)

        result = await sdk_body(request)
        # The SDK's own rollout result. A stale reward here must neither be
        # counted as a second trial nor unlock a score for the physical one.
        nested = Path(config.jobs_dir) / config.job_name / config.rollout_name
        nested.mkdir(parents=True)
        (nested / "result.json").write_text(
            json.dumps({"task_name": "sort-blue-left", "rewards": {"reward": 1.0}})
        )
        return result

    monkeypatch.setattr(DockerSandbox, "preflight", lambda: None)
    monkeypatch.setattr(runner, "HarnessTransport", lambda _: transport)
    monkeypatch.setattr(runner, "provider_environment", lambda _: {})
    monkeypatch.setattr(
        runner,
        "RecordingSidecar",
        lambda output, cameras, fps=2: FakeRecorder(output, cameras, fps, crash=crash),
    )
    monkeypatch.setattr(benchflow, "run", sdk_run)


def _run(tmp_path, setup, **overrides):
    options = dict(
        setup_path=setup,
        task_path=build_tasks(tmp_path / "tasks")[0],
        output_root=tmp_path / "trials",
        agent="codex",
        model="gpt-6-astra",
        reasoning_effort="max",
        reset_id="reset-1",
        operator="test",
        allow_motion=True,
        bind="0.0.0.0",
        advertised_host="127.0.0.1",
        lock_root=tmp_path / "locks",
    )
    options.update(overrides)
    return runner.run_trial(**options)


def _only_trial(tmp_path) -> Path:
    (trial,) = (tmp_path / "trials").iterdir()
    return trial


def _record(trial: Path) -> dict:
    return json.loads((trial / TRIAL_RECORD_FILENAME).read_text())


async def test_completed_trial_record_is_pending_with_provenance(tmp_path, monkeypatch):
    setup = _setup(
        tmp_path,
        camera_mapping={
            "wrist": {"mount": "wrist", "arm": "arm"},
            "side": {"mount": "fixed"},
        },
        controller={"name": "arm-controller", "revision": "abc1234"},
    )

    async def body(request):
        assert request("r1", "observe", [])["ok"]
        assert request("r2", "tip", [0.2, 0, 0.2, -75])["ok"]
        assert request("r3", "finish", [])["ok"]
        return RolloutResult(
            task_name="fixture",
            agent="codex",
            model="gpt-6-astra",
            usage_source="provider_response",
        )

    _install_fakes(monkeypatch, tmp_path, body)
    trial = await _run(tmp_path, setup)
    record = _record(trial)
    assert record["outcome"]["execution"]["status"] == "completed"
    assert record["outcome"]["assessment"]["status"] == "pending"
    provenance = record["provenance"]
    assert provenance["runtime"]["name"] == "benchflow"
    assert provenance["runtime"]["backend"] == "docker"
    assert provenance["controller"]["revision"] == "abc1234"
    assert provenance["cameras"]["wrist"] == {
        "source": "http://127.0.0.1:8787/right.jpg",
        "mount": "wrist",
        "arm": "arm",
        "declared": True,
    }
    actions = record["actions"]
    assert (actions["requests"], actions["dispatched"], actions["receipts"]) == (
        3,
        3,
        3,
    )
    assert actions["motion_dispatched"] == 1
    assert record["clock"]["offset_source"] == "manifest_anchor"
    streams = {s["name"]: s["status"] for s in record["streams"]}
    assert streams["camera:wrist"] == streams["camera:side"] == "complete"
    # The fake SDK wrote no BenchFlow trajectory: missing, not assumed.
    assert streams["agent"] == "missing"
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] is None and classify_audit_outcome(result) == "unscored"
    rows = runner.report_trials(tmp_path / "trials")
    assert rows[0]["assessment_status"] == "pending" and rows[0]["reward"] is None


async def test_cancelled_trial_keeps_partial_record(tmp_path, monkeypatch):
    setup = _setup(tmp_path)

    async def body(request):
        assert request("r1", "tip", [0.2, 0, 0.2, -75])["ok"]
        raise asyncio.CancelledError

    _install_fakes(monkeypatch, tmp_path, body, crash=True)
    with pytest.raises(asyncio.CancelledError):
        await _run(tmp_path, setup)
    trial = _only_trial(tmp_path)
    record = _record(trial)
    assert record["outcome"]["execution"]["status"] == "cancelled"
    assert record["outcome"]["execution"]["pipeline"] == "capture_failure"
    streams = {s["name"]: s for s in record["streams"]}
    assert streams["camera:wrist"]["status"] == "partial"
    assert streams["camera:wrist"]["truncated_tail"] is True
    assert (trial / "cameras" / "wrist.jsonl").read_text().endswith('"mono')
    assert record["actions"]["motion_dispatched"] == 1
    assert record["intact"] is False
    result = json.loads((trial / "result.json").read_text())
    assert result["error"] == "CancelledError"
    assert classify_score_outcome(result) == "errored"


async def test_timed_out_trial_is_not_a_success(tmp_path, monkeypatch):
    setup = _setup(tmp_path)

    async def body(request):
        assert request("r1", "observe", [])["ok"]
        return RolloutResult(
            task_name="fixture",
            error="Agent codex timed out after 60s",
            error_category="timeout",
            usage_source="provider_response",
        )

    _install_fakes(monkeypatch, tmp_path, body)
    trial = await _run(tmp_path, setup)
    record = _record(trial)
    assert record["outcome"]["execution"]["status"] == "timed_out"
    assert record["outcome"]["assessment"]["status"] == "pending"
    result = json.loads((trial / "result.json").read_text())
    assert classify_audit_outcome(result) == "errored"


async def test_review_moves_pending_to_verified_and_scores(tmp_path, monkeypatch):
    setup = _setup(tmp_path)

    async def body(request):
        assert request("r1", "observe", [])["ok"]
        return RolloutResult(
            task_name="fixture",
            agent="codex",
            model="gpt-6-astra",
            usage_source="provider_response",
        )

    _install_fakes(monkeypatch, tmp_path, body)
    trial = await _run(tmp_path, setup)
    # One result per trial: the nested SDK result is an artifact of it.
    before = collect_metrics(tmp_path / "trials")
    assert (before.total, before.passed) == (1, 0)
    assert before.tasks[0].reward is None
    runner.score_trial(
        trial,
        placements={"blue": "left", "green": "right", "yellow": "right"},
        reviewer="reviewer",
        interventions=0,
        cups_upright=True,
        evidence="cameras/wrist.mp4 00:10-00:40",
    )
    assert json.loads((trial / "manifest.json").read_text())["assessment"] == (
        "verified"
    )
    record = _record(trial)
    assert record["outcome"]["assessment"]["status"] == "verified"
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] == {"reward": 1.0}
    assert classify_score_outcome(result) == "passed"
    after = collect_metrics(tmp_path / "trials")
    assert (after.total, after.passed) == (1, 1)


def test_setup_rejects_undeclarable_camera_mapping(tmp_path):
    setup = _setup(tmp_path, camera_mapping={"wrist": {"mount": "wrist", "arm": "b"}})
    with pytest.raises(ValueError, match="must name one of the arms"):
        runner.load_setup(setup)


def test_nested_sdk_rollout_is_refused_by_continue(tmp_path):
    trial = tmp_path / "trial"
    rollout = trial / "benchflow" / "trial" / "agent"
    (rollout / "trajectory").mkdir(parents=True)
    (trial / TRIAL_RECORD_FILENAME).write_text(
        json.dumps({"embodiment": {"kind": "physical"}})
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "openhands"}))
    (rollout / "result.json").write_text(json.dumps({"task_name": "t"}))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "t",
                "rewards": None,
                "assessment": {"status": "pending"},
            }
        )
    )
    with pytest.raises(RunFolderError, match="physical"):
        load_run_folder(rollout)
    assert count_audit_outcomes([json.loads((trial / "result.json").read_text())]) == {
        "passed": 0,
        "failed": 0,
        "errored": 0,
        "verifier_errored": 0,
        "unscored": 1,
    }

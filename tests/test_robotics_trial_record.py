"""Embodied trial record: trace index, honest outcome states and provenance.

Fixtures are recorded-artifact shapes from ``benchflow.robotics`` trials
(bridge ``commands.jsonl``, camera frame indexes, harness encoder logs, ACP
trajectories). Nothing here talks to hardware.
"""

import hashlib
import json
from pathlib import Path

import pytest

from benchflow._utils.scoring import classify_audit_outcome, classify_score_outcome
from benchflow.embodiment import TRIAL_RECORD_FILENAME, recorded_embodiment
from benchflow.robotics.bridge import TrialBridge
from benchflow.robotics.outcome import (
    EXECUTION_STATUSES,
    assessment_outcome,
    execution_outcome,
)
from benchflow.robotics.record import (
    build_trial_record,
    camera_mapping,
    controller_identity,
    inspect_robots_provenance,
    write_trial_record,
)
from benchflow.robotics.trace import build_trace_index

T0 = 1_767_225_600.0
MONO0 = 5_000.0


def _jsonl(path: Path, records, *, tail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)


def _event(event, dt, **fields):
    return {"event": event, "monotonic": MONO0 + dt, "utc_epoch": T0 + dt, **fields}


def _camera(trial: Path, name: str, frames: int, *, summary: bool = True, tail=""):
    records = [
        {
            "frame": i + 1,
            "monotonic": MONO0 + i,
            "utc_epoch": T0 + i,
            "offset": i * 4,
            "length": 4,
            "sha256": "x",
        }
        for i in range(frames)
    ]
    # A capture error carries only the host monotonic clock.
    records.append({"error_type": "URLError", "monotonic": MONO0 + frames + 0.5})
    _jsonl(trial / "cameras" / f"{name}.jsonl", records, tail=tail)
    (trial / "cameras" / f"{name}.mjpeg").write_bytes(b"\xff\xd8\xff\xd9" * frames)
    if summary:
        stats = {"frames": frames, "errors": 0, "maximum_gap_s": 1.0}
        path = trial / "cameras" / "capture-summary.json"
        existing = json.loads(path.read_text()) if path.exists() else {"cameras": {}}
        existing["cameras"][name] = stats
        path.write_text(json.dumps(existing))


def _commands(trial: Path, *, with_requests: bool = True, tail: str = "") -> None:
    events = [_event("activated", 0, allow_motion=True)]
    if with_requests:
        events.append(
            _event("command_requested", 1, request_id="r1", command="observe")
        )
    events += [
        _event("command_started", 1.1, request_id="r1", command="observe", args=[]),
        _event(
            "command_finished",
            2,
            receipt={
                "ok": True,
                "request_id": "r1",
                "images": [{"name": "0001_wrist.jpg", "camera": "wrist"}],
            },
        ),
    ]
    if with_requests:
        events.append(_event("command_requested", 3, request_id="r2", command="tip"))
    events += [
        _event(
            "command_rejected",
            3.1,
            request_id="r2",
            command="tip",
            reason="arm_separation",
        ),
        _event("command_started", 4, request_id="r3", command="tip", args=["0.2"]),
        _event("transport_failure", 5, diagnostic_type="TimeoutError"),
        _event(
            "command_finished",
            5.1,
            receipt={
                "ok": False,
                "request_id": "r3",
                "error": "Command outcome uncertain; operator inspection required",
            },
        ),
        # Dispatched as the trial deadline hit: no receipt was ever written.
        _event("command_started", 6, request_id="r4", command="gripper", args=["40"]),
    ]
    _jsonl(trial / "commands.jsonl", events, tail=tail)


def _trial(tmp_path: Path) -> Path:
    trial = tmp_path / "20260101T000000Z-0123456789"
    _commands(trial)
    _camera(trial, "wrist", 5)
    _camera(trial, "side", 5)
    (trial / "observations").mkdir()
    (trial / "observations" / "0001_wrist.jpg").write_bytes(b"jpg")
    (trial / "observations" / "0000_wrist.jpg").write_bytes(b"initial")
    _jsonl(trial / "encoders.jsonl", [{"t": T0 + 0.5, "joints": {"elbow": 1.0}}])
    _jsonl(
        trial / "benchflow" / "job" / "agent" / "trajectory" / "acp_trajectory.jsonl",
        [
            {"type": "agent_message", "ts": "2026-01-01T00:00:00+00:00"},
            {"type": "tool_call", "ts": "2026-01-01T00:00:05+00:00"},
        ],
    )
    (trial / "arm.rrd").write_bytes(b"RRF2")
    return trial


def _manifest(trial: Path, **overrides) -> dict:
    manifest = {
        "schema_version": 1,
        "trial_id": trial.name,
        "kind": "physical_trial",
        "task": "t2-plane",
        "reset_id": "reset-2",
        "agent": "codex",
        "requested_model": "gpt-6-astra",
        "backend": "host",
        "benchflow_version": "0.7.9.dev0",
        "started_utc_epoch": T0,
        "finished_utc_epoch": T0 + 60,
        "status": "awaiting_assessment",
        "assessment": "pending",
        "footage_complete": True,
        "bridge_halt_reason": None,
        "arms": {"right": "arm-b"},
        "primary_arm": "right",
        "clock_anchor": {"monotonic": MONO0, "utc_epoch": T0},
        **overrides,
    }
    (trial / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _stream(index, name):
    return next(s for s in index["streams"] if s["name"] == name)


def test_index_separates_action_stages_on_one_clock(tmp_path):
    trial = _trial(tmp_path)
    index = build_trace_index(
        trial,
        cameras=["wrist", "side"],
        arms={"right": {}},
        primary_arm="right",
        clock_anchor={"monotonic": MONO0, "utc_epoch": T0},
    )
    assert index["clock"]["host_monotonic_offset_s"] == T0 - MONO0
    assert index["clock"]["offset_source"] == "manifest_anchor"
    assert index["clock"]["offset_spread_s"] == 0
    kinds = {s["kind"] for s in index["streams"]}
    assert {
        "action_request",
        "guard_decision",
        "dispatch",
        "receipt",
        "video",
        "observation",
        "telemetry",
        "agent_trajectory",
        "rerun_recording",
    } <= kinds
    assert _stream(index, "bridge:action_request")["records"] == 2
    assert _stream(index, "bridge:guard_decision")["records"] == 1
    assert _stream(index, "bridge:dispatch")["records"] == 3
    assert _stream(index, "bridge:receipt")["records"] == 2
    assert index["actions"] == {
        "requests": 2,
        "rejected": 1,
        "rejections_by_reason": {"arm_separation": 1},
        "dispatched": 3,
        "receipts": 2,
        "motion_dispatched": 2,
        "observations_ok": 1,
        "transport_failures": 1,
        "dispatched_without_receipt": ["r4"],
        "uncertain_outcomes": ["r3"],
    }
    wrist = _stream(index, "camera:wrist")
    assert wrist["records"] == 5 and wrist["capture_errors"] == 1
    # A capture error is counted, never placed on the timeline as a frame, and
    # makes the footage partial by the recorder's own completeness bar.
    assert wrist["last_utc"] == T0 + 4 and wrist["status"] == "partial"
    assert wrist["first_iso"] == "2026-01-01T00:00:00+00:00"
    observations = _stream(index, "observations")
    assert observations["records"] == 2 and observations["untimed_records"] == 1
    assert observations["first_utc"] == T0 + 2
    acp = _stream(index, "agent:acp")
    assert acp["first_utc"] == T0 and acp["last_utc"] == T0 + 5
    assert _stream(index, "telemetry:right")["status"] == "complete"
    assert index["synchronized"] is True


def test_partial_recording_is_indexed_and_left_byte_identical(tmp_path):
    """A cancelled trial's truncated streams stay on disk exactly as written."""
    trial = tmp_path / "trial"
    _commands(trial, tail='{"event": "command_sta')
    _camera(trial, "wrist", 3, summary=False, tail='{"frame": 4, "monot')
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in trial.rglob("*")
        if p.is_file()
    }
    index = build_trace_index(trial, cameras=["wrist", "side"], expect_agent=False)
    after = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in trial.rglob("*")
        if p.is_file()
    }
    assert before == after
    wrist = _stream(index, "camera:wrist")
    assert wrist["status"] == "partial" and wrist["truncated_tail"]
    assert wrist["capture_summary"] is False and wrist["records"] == 3
    assert _stream(index, "camera:side")["status"] == "missing"
    assert _stream(index, "bridge:dispatch")["status"] == "partial"
    assert index["intact"] is False
    # Without an anchor the offset comes from paired samples, spread reported.
    assert index["clock"]["offset_source"] == "paired_samples"


def test_empty_and_relative_telemetry_is_reported_not_inferred(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "encoders.jsonl").write_bytes(b"")
    _jsonl(trial / "encoders-left.jsonl", [{"t": 12.5}, {"t": 13.0}])
    index = build_trace_index(
        trial,
        cameras=[],
        arms={"right": {}, "left": {}},
        primary_arm="right",
        expect_agent=False,
    )
    assert _stream(index, "telemetry:right")["status"] == "empty"
    left = _stream(index, "telemetry:left")
    assert left["clock"] == "relative_s" and left["status"] == "untimed"
    assert index["synchronized"] is False


def test_logs_without_request_events_derive_requests(tmp_path):
    trial = tmp_path / "trial"
    _commands(trial, with_requests=False)
    index = build_trace_index(trial, cameras=[], expect_agent=False)
    requests = _stream(index, "bridge:action_request")
    assert requests["derived"] is True and requests["records"] == 4
    assert index["actions"]["requests"] is None


def test_bridge_logs_request_and_guard_decision_separately(tmp_path):
    bridge = TrialBridge(
        output=tmp_path,
        transport=lambda *_: {"ok": True},
        frames_root=tmp_path,
        robot_id="metal",
        recording_healthy=lambda: True,
    )
    bridge.activate()
    receipt = bridge.execute({"request_id": "m1", "command": "gripper", "args": [40]})
    assert receipt == {"ok": False, "error": "Read-only trial: motion disabled"}
    index = build_trace_index(tmp_path, cameras=[], expect_agent=False)
    assert index["actions"]["requests"] == 1
    assert index["actions"]["rejections_by_reason"] == {"motion_disabled": 1}
    assert index["actions"]["dispatched"] == 0


@pytest.mark.parametrize(
    "overrides,metrics,actions,expected,pipeline",
    [
        ({"bridge_halt_reason": "budget_exhausted"}, None, None, "halted", "healthy"),
        (
            {},
            None,
            {"motion_dispatched": 0, "observations_ok": 3},
            "no_motion",
            "healthy",
        ),
        (
            {},
            None,
            {"motion_dispatched": 4, "observations_ok": 3},
            "completed",
            "healthy",
        ),
        (
            {"status": "agent_error"},
            {"error": "Agent timed out", "error_category": "timeout"},
            None,
            "timed_out",
            "healthy",
        ),
        (
            {"status": "agent_error"},
            {"error": "Host agent time budget exhausted"},
            None,
            "timed_out",
            "healthy",
        ),
        (
            {"status": "agent_error"},
            {"error": "Operator stopped host trial"},
            None,
            "cancelled",
            "healthy",
        ),
        ({"status": "interrupted"}, None, None, "cancelled", "healthy"),
        (
            {"status": "agent_error", "bridge_halt_reason": "recording_lost"},
            None,
            None,
            "halted",
            "capture_failure",
        ),
        (
            {"bridge_halt_reason": "uncertain_outcome"},
            None,
            None,
            "halted",
            "controller_failure",
        ),
        (
            {"status": "infrastructure_error", "footage_complete": False},
            None,
            None,
            "infrastructure_error",
            "capture_failure",
        ),
        (
            {"status": "running", "finished_utc_epoch": None, "footage_complete": None},
            None,
            None,
            "unfinalized",
            "unknown",
        ),
    ],
)
def test_execution_states(tmp_path, overrides, metrics, actions, expected, pipeline):
    manifest = _manifest(tmp_path, **overrides)
    if manifest.get("finished_utc_epoch") is None:
        del manifest["finished_utc_epoch"]
    execution = execution_outcome(manifest, metrics=metrics, actions=actions)
    assert execution["status"] == expected in EXECUTION_STATUSES
    assert execution["pipeline"] == pipeline


@pytest.mark.parametrize(
    "kind,execution,review,status,reward",
    [
        ("physical_trial", "halted", None, "pending", None),
        ("physical_trial", "unfinalized", None, "unassessable", None),
        ("agent_probe", "completed", None, "unassessable", None),
        (
            "physical_trial",
            "completed",
            {"benchmark_valid": False, "autonomous_success": True},
            "unassessable",
            None,
        ),
        (
            "physical_trial",
            "completed",
            {"benchmark_valid": True, "autonomous_success": True},
            "verified",
            1.0,
        ),
        (
            "physical_trial",
            "completed",
            {"benchmark_valid": True, "autonomous_success": False},
            "failed",
            0.0,
        ),
    ],
)
def test_assessment_states(kind, execution, review, status, reward):
    outcome = assessment_outcome({"kind": kind}, review, execution=execution)
    assert (outcome["status"], outcome["reward"]) == (status, reward)


def test_awaiting_assessment_trial_record_is_unscored(tmp_path):
    """A physical trial's shape: halted on budget, awaiting assessment."""
    trial = _trial(tmp_path)
    _manifest(trial, bridge_halt_reason="budget_exhausted")
    # A stale reward file from somewhere else must not leak into the score.
    (trial / "reward.txt").write_text("1.0\n")
    record = write_trial_record(trial)
    assert record["outcome"]["execution"]["status"] == "halted"
    assert record["outcome"]["assessment"]["status"] == "pending"
    assert record["embodiment"]["retry_requires"] == "new_qualified_episode"
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] is None and result["assessment"]["status"] == "pending"
    assert result["error"] is None
    assert classify_audit_outcome(result) == "unscored"
    assert classify_score_outcome(result) != "passed"
    assert recorded_embodiment(trial / "benchflow" / "job" / "agent").physical
    assert json.loads((trial / TRIAL_RECORD_FILENAME).read_text())["trial_id"] == (
        trial.name
    )


def test_probe_writes_record_but_no_benchmark_result(tmp_path):
    trial = _trial(tmp_path)
    _manifest(trial, kind="agent_probe", status="probe_completed")
    record = write_trial_record(trial)
    assert record["outcome"]["assessment"]["reason"] == "not_a_scored_trial"
    assert not (trial / "result.json").exists()


def test_legacy_manifest_provenance_is_benchflow_without_guessing(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    manifest = _manifest(trial)
    record = build_trial_record(trial)
    provenance = record["provenance"]
    assert provenance["runtime"]["name"] == "benchflow"
    assert provenance["runtime"]["version"] == manifest["benchflow_version"]
    assert provenance["arms"] == {"right": {"robot_id": "arm-b", "primary": True}}
    assert provenance["cameras"] is None
    assert provenance["controller"]["declared"] is False
    del manifest["benchflow_version"]
    (trial / "manifest.json").write_text(json.dumps(manifest))
    assert build_trial_record(trial)["provenance"]["runtime"]["name"] == "unknown"


def test_camera_mapping_is_declared_never_inferred():
    arms = {"right": {}, "left": {}}
    cameras = {
        "wrist": "http://user:pw@127.0.0.1:8787/left.jpg?token=abc",
        "side": "http://127.0.0.1:8787/front.jpg",
    }
    undeclared = camera_mapping(cameras, None, arms)
    assert undeclared["cameras"]["wrist"] == {
        "source": "http://127.0.0.1:8787/left.jpg",
        "mount": None,
        "arm": None,
        "declared": False,
    }
    declared = camera_mapping(
        cameras,
        {"wrist": {"mount": "wrist", "arm": "left"}, "side": {"mount": "fixed"}},
        arms,
    )
    assert declared["cameras"]["wrist"]["arm"] == "left"
    # A setup whose two logical cameras share one fixed feed.
    shared = camera_mapping(
        {"wrist": "http://h/front.jpg", "side": "http://h/front.jpg"}, None, arms
    )
    assert shared["shared_feeds"] == [["wrist", "side"]]


@pytest.mark.parametrize(
    "declared",
    [
        {"wrist": {"mount": "wrist", "arm": "middle"}},
        {"side": {"mount": "fixed", "arm": "left"}},
        {"top": {"mount": "fixed"}},
        {"wrist": {"mount": "gimbal"}},
    ],
)
def test_invalid_camera_mapping_rejected(declared):
    with pytest.raises(ValueError):
        camera_mapping({"wrist": "a", "side": "b"}, declared, {"left": {}})


def test_controller_identity():
    assert controller_identity(None)["declared"] is False
    identity = controller_identity(
        {"name": "arm-controller", "revision": "abc1234", "patches": ["demo-patch"]}
    )
    assert identity == {
        "declared": True,
        "name": "arm-controller",
        "revision": "abc1234",
        "patches": ["demo-patch"],
    }
    with pytest.raises(ValueError):
        controller_identity({"name": "arm-controller"})


def test_inspect_robots_resettable_is_not_restorable():
    """Shape of an Inspect Robots EvalLog ``eval`` block for a physical embodiment."""
    provenance = inspect_robots_provenance(
        {
            "embodiment": "demo_arm",
            "embodiment_info": {
                "capabilities": ["self_paced", "resettable"],
                "control_hz": 2.0,
                "is_simulated": False,
            },
            "git_commit": None,
            "inspect_robots_version": "0.1.0",
            "policy": "agent",
            "policy_config": {"model": "gpt-6-astra", "effort": "medium"},
        }
    )
    assert provenance["runtime"] == {
        "name": "inspect_robots",
        "version": "0.1.0",
        "git_commit": None,
    }
    assert provenance["embodiment"]["kind"] == "physical"
    assert provenance["embodiment"]["capabilities"] == ["resettable", "self_paced"]
    assert provenance["policy"]["requested_model"] == "gpt-6-astra"


def test_monotonic_only_frames_map_through_the_anchor(tmp_path):
    trial = tmp_path / "trial"
    _jsonl(
        trial / "cameras" / "side.jsonl",
        [{"frame": 1, "monotonic": MONO0 + 7}, {"frame": 2, "monotonic": MONO0 + 8}],
    )
    index = build_trace_index(
        trial,
        cameras=["side"],
        clock_anchor={"monotonic": MONO0, "utc_epoch": T0},
        expect_agent=False,
    )
    side = _stream(index, "camera:side")
    assert (side["first_utc"], side["last_utc"]) == (T0 + 7, T0 + 8)


def test_derived_requests_do_not_invent_untimed_records(tmp_path):
    trial = tmp_path / "trial"
    _jsonl(
        trial / "commands.jsonl",
        [
            _event(
                "command_rejected",
                1,
                request_id="r1",
                command="tip",
                reason="motion_disabled",
            )
        ],
    )
    requests = _stream(
        build_trace_index(trial, cameras=[], expect_agent=False),
        "bridge:action_request",
    )
    assert requests["derived"] and requests["records"] == 1
    assert requests["untimed_records"] == 0 and requests["first_utc"] == T0 + 1


def test_host_agent_exit_and_wall_time_are_recorded_as_evidence(tmp_path):
    """Regression test: a legacy host trial stopped at its deadline (exit 143
    after 1801 s of a 1800 s budget) was recorded as ``agent_error`` with ``error: null`` and
    neither value, so a reader could not tell it had hit the deadline. The
    values are copied as evidence; the state is not reinterpreted from them.
    """
    trial = tmp_path / "trial"
    trial.mkdir()
    _manifest(trial, status="agent_error", agent_timeout_s=1800)
    (trial / "host-agent-summary.json").write_text(
        json.dumps(
            {
                "completed": False,
                "failed": False,
                "usage": {},
                "native_cost_usd": None,
                "exit_code": 143,
                "agent_wall_s": 1801.0,
                "usage_source": "unavailable",
            }
        )
    )
    record = write_trial_record(trial)
    execution = record["outcome"]["execution"]
    assert execution["status"] == "agent_error"
    assert execution["error"] is None
    assert execution["evidence"] == {
        "host_agent": {
            "source": "host-agent-summary.json",
            "exit_code": 143,
            "agent_wall_s": 1801.0,
        },
        "agent_timeout_s": 1800,
    }
    result = json.loads((trial / "result.json").read_text())
    assert result["execution"]["evidence"] == execution["evidence"]
    assert result["error"] == "physical trial agent_error"

    (trial / "host-agent-summary.json").unlink()
    execution = build_trial_record(trial)["outcome"]["execution"]
    assert execution["evidence"] == {"host_agent": None, "agent_timeout_s": 1800}


def test_rerun_recordings_subdirectory_is_indexed(tmp_path):
    """Regression test: the index only looked for
    ``*.rrd`` in the trial root, so the demo layout's ``recordings/trial.rrd``
    (also where viewers look) was not indexed.
    """
    trial = tmp_path / "trial"
    (trial / "recordings").mkdir(parents=True)
    (trial / "recordings" / "trial.rrd").write_bytes(b"RRF2sub")
    (trial / "arm.rrd").write_bytes(b"RRF2")
    index = build_trace_index(trial, cameras=[], expect_agent=False)
    nested = _stream(index, "rerun:recordings/trial")
    assert nested["path"] == "recordings/trial.rrd"
    assert nested["kind"] == "rerun_recording" and nested["status"] == "complete"
    assert nested["sha256"] == hashlib.sha256(b"RRF2sub").hexdigest()
    assert _stream(index, "rerun:arm")["path"] == "arm.rrd"


def test_synchronized_is_null_when_no_stream_is_present(tmp_path):
    """Regression test: with every stream missing,
    ``synchronized`` was a vacuous ``true`` next to ``intact: false``.
    """
    trial = tmp_path / "trial"
    trial.mkdir()
    index = build_trace_index(trial, cameras=["wrist"], arms={"arm": {}})
    assert {s["status"] for s in index["streams"]} == {"missing"}
    assert index["intact"] is False
    assert index["synchronized"] is None
    (trial / "recordings").mkdir()
    (trial / "recordings" / "trial.rrd").write_bytes(b"RRF2")
    assert build_trace_index(trial, cameras=["wrist"])["synchronized"] is True


def test_report_gives_the_same_execution_state_as_the_record(tmp_path):
    """``report`` used to classify execution without the
    command log, so a trial whose record says ``no_motion`` (the agent only
    observed) was listed as ``completed``.
    """
    from benchflow.robotics.runner import report_trials

    root = tmp_path / "trials"
    observed = root / "20260101T010000Z-0000000001"
    _jsonl(
        observed / "commands.jsonl",
        [
            _event("activated", 0, allow_motion=True),
            _event("command_requested", 1, request_id="r1", command="observe"),
            _event("command_started", 1.1, request_id="r1", command="observe"),
            _event("command_finished", 2, receipt={"ok": True, "request_id": "r1"}),
        ],
    )
    _manifest(observed)
    moved = root / "20260101T020000Z-0000000002"
    _commands(moved)
    _manifest(moved)

    rows = {row["trial_id"]: row for row in report_trials(root)}

    for trial in (observed, moved):
        record = build_trial_record(trial)["outcome"]["execution"]
        assert rows[trial.name]["execution_status"] == record["status"]
    assert rows[observed.name]["execution_status"] == "no_motion"
    assert rows[moved.name]["execution_status"] == "completed"
    # With no command log there is nothing to count: the state stays as the
    # manifest and metrics give it.
    (observed / "commands.jsonl").unlink()
    rows = {row["trial_id"]: row for row in report_trials(root)}
    assert rows[observed.name]["execution_status"] == "completed"

"""Continuation metadata regeneration at the trainer-admission boundary."""

from __future__ import annotations

import json

from benchflow.continue_run.orchestrator import (
    summarize_llm_trajectory_usage,
    update_continued_metadata,
)
from benchflow.trajectories.llm_capture_manifest import (
    REPLAY_PROXY_INGRESS_AUDIT_ERROR,
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMRoleCapture,
    LLMTrajectoryManifest,
    write_llm_trajectory_manifest,
)

from ._helpers import completion, exchange


def test_update_continued_metadata_rebuilds_trainer_results(tmp_path):
    """Guards PR #1057 review r3888738115 and stale continuation rows."""
    rollout = tmp_path / "job" / "demo-task__continued"
    (rollout / "trajectory").mkdir(parents=True)
    model = "openai/gpt-5.5"
    row = exchange(completion(content="done")).model_dump(mode="json")
    row["request"]["body"]["messages"] = [{"role": "user", "content": "Do the task."}]
    row["metadata"] = {
        "schema_version": 2,
        "capture_source": "litellm_proxy",
        "capture_fidelity": "provider_wire",
        "auth_mode": "api_key",
        "request_complete": True,
        "response_complete": True,
        "payload_redacted": True,
        "role_attribution_complete": True,
        "role": "agent",
        "agent": "openhands",
        "model": model,
    }
    trajectory_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    trajectory_path.write_text(json.dumps(row) + "\n")
    manifest = LLMTrajectoryManifest(
        status=CaptureStatus.COMPLETE,
        capture_source=CaptureSource.LITELLM_PROXY,
        capture_fidelity=CaptureFidelity.PROVIDER_WIRE,
        auth_mode=AuthMode.API_KEY,
        agent="openhands",
        model=model,
        session_id="continued",
        exchange_count=1,
        request_complete=True,
        response_complete=True,
        payload_redacted=True,
        started_at="2026-08-29T00:00:00Z",
        finished_at="2026-08-29T00:01:00Z",
        role_captures=[
            LLMRoleCapture(
                role="agent",
                agent="openhands",
                model=model,
                auth_mode=AuthMode.API_KEY,
                capture_source=CaptureSource.LITELLM_PROXY,
                capture_fidelity=CaptureFidelity.PROVIDER_WIRE,
                exchange_count=1,
                request_complete=True,
                response_complete=True,
            )
        ],
    )
    write_llm_trajectory_manifest(rollout, manifest)
    (rollout / "config.json").write_text(json.dumps({"model": None, "source": {}}))
    (rollout / "prompts.json").write_text(json.dumps(["Do the task."]))
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "demo-task",
                "rollout_name": "demo-task__continued",
                "agent": "openhands",
                "agent_name": "OpenHands",
                "model": None,
                "n_tool_calls": 0,
                "partial_trajectory": False,
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "export_error": None,
                "timing": {},
                "agent_result": {"total_tokens": 0, "usage_source": "unavailable"},
                "usage_tracking": {"requested": "off", "status": "off"},
            }
        )
    )
    (rollout / "results.jsonl").write_text(
        json.dumps({"info": {"training_ready": False, "model": None}}) + "\n"
    )

    requested_but_unused_model = "openai/gpt-5.6"
    update_continued_metadata(
        rollout,
        live_model=requested_but_unused_model,
        usage=summarize_llm_trajectory_usage(trajectory_path, n_recorded=0),
        environment="docker",
    )

    refreshed_config = json.loads((rollout / "config.json").read_text())
    refreshed_result = json.loads((rollout / "result.json").read_text())
    refreshed = json.loads((rollout / "results.jsonl").read_text())
    aggregated = json.loads((rollout.parent / "results.jsonl").read_text())
    assert refreshed_config["model"] == model
    assert refreshed_result["model"] == model
    assert refreshed["info"]["model"] == model
    assert refreshed["info"]["training_ready"] is True
    assert refreshed["token_usage"]["total_tokens"] == 2
    assert len(refreshed["trajectory"]) == 1
    assert aggregated == refreshed

    replay_manifest = manifest.model_copy(
        update={
            "status": CaptureStatus.PARTIAL,
            "capture_source": CaptureSource.MIXED,
            "capture_fidelity": CaptureFidelity.MIXED,
            "request_complete": False,
            "missing_fields": ["live_provider_request"],
            "errors": [REPLAY_PROXY_INGRESS_AUDIT_ERROR],
            "role_captures": [
                LLMRoleCapture(
                    role="agent",
                    leg="live",
                    agent="openhands",
                    model=model,
                    auth_mode=AuthMode.API_KEY,
                    capture_source=CaptureSource.REPLAY_PROXY,
                    capture_fidelity=CaptureFidelity.AGENT_SESSION,
                    exchange_count=1,
                    request_complete=False,
                    response_complete=True,
                )
            ],
        }
    )
    write_llm_trajectory_manifest(rollout, replay_manifest)
    replay_row = json.loads(trajectory_path.read_text())
    replay_row["metadata"].update(
        {
            "capture_source": "replay_proxy",
            "capture_fidelity": "agent_session",
            "request_complete": False,
        }
    )
    trajectory_path.write_text(json.dumps(replay_row) + "\n")
    (rollout / "results.jsonl").write_text(
        json.dumps({"info": {"training_ready": True}, "is_completed": False}) + "\n"
    )

    update_continued_metadata(
        rollout,
        live_model=model,
        usage=summarize_llm_trajectory_usage(trajectory_path, n_recorded=0),
        environment="docker",
    )

    replay_refreshed = json.loads((rollout / "results.jsonl").read_text())
    assert replay_refreshed["info"]["training_ready"] is False
    assert replay_refreshed["info"]["training_ready_reason"] == (
        "insufficient_capture_fidelity"
    )
    assert replay_refreshed["is_completed"] is True
    assert replay_refreshed["error"] is None

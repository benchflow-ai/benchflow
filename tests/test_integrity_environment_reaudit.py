"""Guards offline integrity contract loss present at commit 764325430."""

import json
from types import SimpleNamespace

import pytest

from benchflow.environment.manifest import EnvironmentManifest, StateSpec
from benchflow.integrity.emit import emit_integrity
from benchflow.integrity.trial import audit_trial, evidence_from_rollout
from benchflow.models import RolloutResult
from benchflow.rollout import RolloutConfig
from benchflow.task import Task
from tests.test_integrity import DET_TASK, _probe_record


@pytest.mark.parametrize("path", ["/app/state.db", "/app/answer.txt"])
def test_reaudit_keeps_environment_state_hidden(tmp_path, path):
    """Guards commit 764325430: stored environment state retains its live contract."""
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "verifier/reward.txt").write_text("1")
    manifest = EnvironmentManifest(
        name="stateful",
        base_image="ubuntu:24.04",
        state=StateSpec(paths=["/app/state.db"]),
    )
    cfg = RolloutConfig(
        task_path=DET_TASK,
        agent="codex-acp",
        integrity="audit",
        environment_manifest=manifest,
    )
    trajectory = [
        {
            "type": "tool_call",
            "tool_call_id": "read1",
            "kind": "Read",
            "raw_input": {"file_path": path},
            "status": "completed",
        }
    ]
    result = RolloutResult(
        "task",
        rollout_name="trial",
        agent="codex-acp",
        rewards={"reward": 1.0},
        n_tool_calls=1,
        trajectory=trajectory,
        trajectory_source="acp",
    )
    rollout = SimpleNamespace(
        _config=cfg,
        _task=Task(DET_TASK),
        _rollout_dir=trial,
        _rollout_name="trial",
        _agent_cwd="/app",
        _effective_skills_dir=None,
    )
    evidence = evidence_from_rollout(rollout, result, mode="audit")
    evidence.access_probe = _probe_record(evidence.contract, "open")
    live = emit_integrity(evidence)
    (trial / "integrity/access_probe.json").write_text(
        json.dumps(evidence.access_probe)
    )
    (trial / "trajectory/acp_trajectory.jsonl").write_text(
        json.dumps(trajectory[0]) + "\n"
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rollout_name": "trial",
                "agent": "codex-acp",
                "rewards": {"reward": 1.0},
                "n_tool_calls": 1,
                "trajectory_source": "acp",
            }
        )
    )
    (trial / "config.json").write_text(
        json.dumps(
            {
                "task_path": str(DET_TASK),
                "agent": "codex-acp",
                "environment": "daytona",
                "sandbox_user": "agent",
                "integrity": "audit",
            }
        )
    )
    source_bytes = (trial / "result.json").read_bytes()
    offline = audit_trial(trial, task_path=DET_TASK)
    assert offline.verdict == live["core_verdict"]
    assert offline.exploited is (path == "/app/state.db")
    contract = json.loads((trial / "integrity/manifest.json").read_text())
    assert any(
        r.get("path") == "/app/state.db" and r["class"] == "Hidden"
        for r in contract["resources"]
    )
    assert (trial / "result.json").read_bytes() == source_bytes
    # A second audit cannot erase the restriction either.
    assert audit_trial(trial, task_path=DET_TASK).verdict == offline.verdict

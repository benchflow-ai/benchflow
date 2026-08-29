"""Training boundaries for the uniform LLM capture manifest."""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.trajectories.results import build_rollout_results_record


def _build_results_row(rollout_dir: Path, *, agent_result: dict) -> dict:
    return build_rollout_results_record(
        rollout_dir,
        task_name="task",
        rollout_name="rollout",
        agent="claude-agent-acp",
        agent_name="Claude Code",
        model="claude-opus-4-1",
        n_tool_calls=0,
        prompts=["hello"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        agent_result=agent_result,
    )


def _write_exchange(trajectory_dir: Path, *, fidelity: str) -> None:
    (trajectory_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {"messages": [{"role": "user", "content": "hello"}]}
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                },
                "metadata": {
                    "capture_fidelity": fidelity,
                    "request_complete": fidelity == "provider_wire",
                    "response_complete": True,
                },
            }
        )
        + "\n"
    )


def test_agent_session_capture_is_audit_only_not_training_ready(tmp_path: Path) -> None:
    """Guards PR #1057 against silently training on reconstructed OAuth payloads."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    _write_exchange(trajectory_dir, fidelity="agent_session")
    (trajectory_dir / "llm_trajectory.manifest.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "capture_fidelity": "agent_session",
                "auth_mode": "oauth_subscription",
                "exchange_count": 1,
                "request_complete": False,
                "response_complete": True,
            }
        )
    )

    row = _build_results_row(
        tmp_path,
        agent_result={"usage_source": "agent_native_acp", "total_tokens": 2},
    )

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "insufficient_capture_fidelity"
    assert row["is_completed"] is True
    assert row["error"] is None


def test_corrupt_capture_manifest_fails_closed_for_training(tmp_path: Path) -> None:
    """Guards PR #1057 against treating a corrupt new sidecar as a legacy artifact."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    _write_exchange(trajectory_dir, fidelity="provider_wire")
    (trajectory_dir / "llm_trajectory.manifest.json").write_text("{broken")

    row = _build_results_row(tmp_path, agent_result={"total_tokens": 2})

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == (
        "missing_healthy_structured_llm_trajectory"
    )
    assert row["is_completed"] is False


def test_manifest_count_mismatch_fails_closed_for_canonical_results(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against training on a truncated canonical trajectory."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    _write_exchange(trajectory_dir, fidelity="provider_wire")
    (trajectory_dir / "llm_trajectory.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "capture_fidelity": "provider_wire",
                "auth_mode": "api_key",
                "exchange_count": 2,
                "request_complete": True,
                "response_complete": True,
            }
        )
    )

    row = _build_results_row(tmp_path, agent_result={"total_tokens": 2})

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "insufficient_capture_fidelity"
    assert row["is_completed"] is False


def test_mixed_oauth_audit_capture_preserves_successful_completion(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against turning successful mixed-auth runs into errors."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    _write_exchange(trajectory_dir, fidelity="provider_wire")
    provider_row = json.loads((trajectory_dir / "llm_trajectory.jsonl").read_text())
    oauth_row = json.loads(json.dumps(provider_row))
    oauth_row["metadata"].update(
        {
            "capture_fidelity": "agent_session",
            "auth_mode": "oauth_subscription",
            "request_complete": False,
        }
    )
    (trajectory_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(provider_row) + "\n" + json.dumps(oauth_row) + "\n"
    )
    (trajectory_dir / "llm_trajectory.manifest.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "capture_fidelity": "mixed",
                "auth_mode": "mixed",
                "exchange_count": 2,
                "request_complete": False,
                "response_complete": True,
                "role_captures": [
                    {
                        "role": "coder",
                        "agent": "codex-acp",
                        "auth_mode": "api_key",
                        "capture_source": "litellm_proxy",
                        "capture_fidelity": "provider_wire",
                        "exchange_count": 1,
                        "request_complete": True,
                        "response_complete": True,
                    },
                    {
                        "role": "reviewer",
                        "agent": "claude-agent-acp",
                        "auth_mode": "oauth_subscription",
                        "capture_source": "claude_native_session",
                        "capture_fidelity": "agent_session",
                        "exchange_count": 1,
                        "request_complete": False,
                        "response_complete": False,
                    },
                ],
            }
        )
    )

    row = _build_results_row(
        tmp_path,
        agent_result={"usage_source": "provider_response", "total_tokens": 2},
    )

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "insufficient_capture_fidelity"
    assert row["is_completed"] is True
    assert row["error"] is None

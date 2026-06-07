from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_publish_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "skillsbench-fill" / "publish.py"
    spec = importlib.util.spec_from_file_location("skillsbench_fill_publish", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_secret_scrubber_preserves_usage_token_counters() -> None:
    """Guards commit 10eeec4 against redacting token-usage metadata in PR5 uploads."""
    publish = _load_publish_module()

    scrubbed = publish._scrub(
        {
            "agent_result": {
                "total_tokens": 123,
                "n_input_tokens": 45,
                "n_output_tokens": 78,
            },
            "agent_env": {
                "HUGGING_FACE_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz",
                "MINIMAX_API_KEY": "sk-api-abc123456789",
            },
        }
    )

    assert scrubbed["agent_result"] == {
        "total_tokens": 123,
        "n_input_tokens": 45,
        "n_output_tokens": 78,
    }
    assert scrubbed["agent_env"] == {
        "HUGGING_FACE_TOKEN": "[REDACTED]",
        "MINIMAX_API_KEY": "[REDACTED]",
    }


def test_publish_rejects_raw_partial_without_timeout_overlay() -> None:
    """Guards PR #638 follow-up against uploading unaccepted partial timeouts."""
    publish = _load_publish_module()
    result = {
        "partial_trajectory": True,
        "error": "Agent timed out after 900s",
        "rewards": {"reward": 0.0},
        "trajectory_summary": {"partial_trajectory": True},
    }

    ok, reason = publish._result_publishable(
        result, {"timeout_complete_artifacts": True}
    )
    assert ok is False
    assert "partial trajectory" in reason

    ok, reason = publish._result_publishable(
        result,
        {
            "timeout_complete_artifacts": True,
            "accepted_normal_timeout": True,
        },
    )
    assert ok is True
    assert reason == ""


def test_publish_uses_exact_reviewed_rollout_dir(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against publishing an unreviewed sibling rollout."""
    publish = _load_publish_module()
    runs_root = tmp_path / "runs"
    task = "citation-check"
    reviewed = runs_root / "cell" / f"{task}__reviewed"
    sibling = runs_root / "cell" / f"{task}__zzz"
    for rollout in (reviewed, sibling):
        (rollout / "trajectory").mkdir(parents=True)
        (rollout / "result.json").write_text(json.dumps({"rewards": {"reward": 1}}))
        (rollout / "config.json").write_text("{}")
        (rollout / "trajectory" / "llm_trajectory.jsonl").write_text("{}\n")
        (rollout / "trajectory" / "acp_trajectory.jsonl").write_text("{}\n")

    selected = publish._reviewed_rollout(
        {"cell_id": "cell", "rollout_dir": str(reviewed)}, runs_root, task
    )

    assert selected == reviewed


def test_publish_requires_reviewed_rollout_dir(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against guessing from legacy sibling rollouts."""
    publish = _load_publish_module()
    task = "citation-check"
    sibling = tmp_path / "runs" / "cell" / f"{task}__zzz"
    (sibling / "trajectory").mkdir(parents=True)
    (sibling / "result.json").write_text(json.dumps({"rewards": {"reward": 1}}))
    (sibling / "config.json").write_text("{}")
    (sibling / "trajectory" / "llm_trajectory.jsonl").write_text("{}\n")
    (sibling / "trajectory" / "acp_trajectory.jsonl").write_text("{}\n")

    selected = publish._reviewed_rollout({"cell_id": "cell"}, tmp_path / "runs", task)

    assert selected is None

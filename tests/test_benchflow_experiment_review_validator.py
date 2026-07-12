from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "benchflow-experiment-review"
    / "scripts"
    / "validate_run_artifacts.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_run_artifacts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _rollout(tmp_path: Path, *, results_row: dict | None = None) -> Path:
    rollout = tmp_path / "task-a__trial-1"
    rollout.mkdir()
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "openhands",
                "model": "aws-bedrock/test-model",
                "rewards": {"reward": 1.0},
                "agent_result": {
                    "total_tokens": 17,
                    "n_input_tokens": 11,
                    "n_output_tokens": 6,
                },
                "n_tool_calls": 1,
                "started_at": "2026-06-27T18:00:00Z",
                "finished_at": "2026-06-27T18:00:10Z",
            }
        )
    )
    _write_jsonl(
        rollout / "trajectory" / "acp_trajectory.jsonl",
        [{"type": "agent_message", "content": "working"}],
    )
    _write_jsonl(
        rollout / "trajectory" / "llm_trajectory.jsonl",
        [
            {
                "request": {
                    "body": {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "solve"}],
                    }
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "done",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 6,
                            "total_tokens": 17,
                        },
                    },
                },
            }
        ],
    )
    if results_row is None:
        results_row = {
            "example_id": 0,
            "prompt": [{"role": "user", "content": "solve"}],
            "completion": [{"role": "assistant", "content": "done"}],
            "info": {
                "task_id": "task-a",
                "task_name": "task-a",
                "training_ready": True,
                "training_ready_reason": None,
            },
            "reward": 1.0,
            "error": None,
            "is_completed": True,
            "is_truncated": False,
            "stop_condition": "agent_completed",
            "metrics": {"n_tool_calls": 1, "reward": 1.0},
            "tool_defs": [],
            "token_usage": {
                "final_input_tokens": 11,
                "final_output_tokens": 6,
                "total_tokens": 17,
            },
            "trajectory": [
                {
                    "prompt": [{"role": "user", "content": "solve"}],
                    "completion": [{"role": "assistant", "content": "done"}],
                    "response": {"usage": {"total_tokens": 17}},
                    "trajectory_id": "task-a__trial-1__llm_0",
                    "extras": {"source": "llm_trajectory", "exchange_index": 0},
                }
            ],
        }
    _write_jsonl(rollout / "results.jsonl", [results_row])
    return rollout


def test_validator_accepts_training_ready_results_jsonl(tmp_path: Path) -> None:
    """Guards PR #828: healthy model rollouts require healthy results.jsonl."""
    validator = _load_validator()
    report = validator.validate_rollout(_rollout(tmp_path))

    assert report["healthy"] is True
    assert report["artifacts"]["results"] == {
        "rows": 1,
        "training_ready": 1,
        "rows_with_tool_calls": 0,
    }


def test_validator_accepts_provider_total_only_token_usage(tmp_path: Path) -> None:
    """Some providers expose only total tokens; Prime-RL can still render the row."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    row = json.loads((rollout / "results.jsonl").read_text())
    row["token_usage"] = {
        "input_tokens": 17,
        "output_tokens": 0,
        "final_input_tokens": 17,
        "final_output_tokens": 0,
        "total_tokens": 17,
    }
    _write_jsonl(rollout / "results.jsonl", [row])

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is True


def test_validator_rejects_missing_results_jsonl(tmp_path: Path) -> None:
    """Guards PR #828: results.jsonl is part of the model-run health gate."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    (rollout / "results.jsonl").unlink()

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is False
    assert any("missing required artifact" in issue for issue in report["issues"])


def test_validator_rejects_tool_calls_without_tool_defs(tmp_path: Path) -> None:
    """Guards PR #828: Prime-SFT tool-call rows must carry tool definitions."""
    validator = _load_validator()
    row = {
        "example_id": 0,
        "prompt": [{"role": "user", "content": "solve"}],
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ],
            }
        ],
        "info": {
            "task_id": "task-a",
            "training_ready": True,
            "training_ready_reason": None,
        },
        "reward": 1.0,
        "error": None,
        "is_completed": True,
        "is_truncated": False,
        "stop_condition": "agent_completed",
        "metrics": {"n_tool_calls": 1},
        "token_usage": {"final_input_tokens": 11, "final_output_tokens": 6},
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "solve"}],
                "completion": [{"role": "assistant", "content": "done"}],
                "extras": {"source": "llm_trajectory", "exchange_index": 0},
            }
        ],
    }
    report = validator.validate_rollout(_rollout(tmp_path, results_row=row))

    assert report["healthy"] is False
    assert any("tool_calls require non-empty" in issue for issue in report["issues"])


def test_validator_rejects_unready_results_for_healthy_rollout(tmp_path: Path) -> None:
    """Guards PR #828: a healthy rollout must not silently lose SFT readiness."""
    validator = _load_validator()
    row = {
        "example_id": 0,
        "prompt": [{"role": "user", "content": "solve"}],
        "completion": None,
        "info": {
            "task_id": "task-a",
            "training_ready": False,
            "training_ready_reason": "missing_healthy_structured_llm_trajectory",
        },
        "reward": 1.0,
        "error": {
            "error": "missing_llm_trajectory",
            "error_chain_str": "not ready",
        },
        "is_completed": True,
        "is_truncated": False,
        "stop_condition": "agent_completed",
        "metrics": {"n_tool_calls": 1},
        "tool_defs": [],
        "token_usage": {"final_input_tokens": 11, "final_output_tokens": 6},
        "trajectory": [],
    }
    report = validator.validate_rollout(_rollout(tmp_path, results_row=row))

    assert report["healthy"] is False
    assert any("training_ready=false" in issue for issue in report["issues"])


def test_validator_rejects_results_with_dropped_successful_exchange(
    tmp_path: Path,
) -> None:
    """Guards PR #921 MAX review against silently omitted LLM exchanges."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    llm_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    first = json.loads(llm_path.read_text())
    second = json.loads(llm_path.read_text())
    second["request"]["body"]["messages"][0]["content"] = "solve a different turn"
    _write_jsonl(llm_path, [first, second])

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is False
    assert any(
        "results trajectory steps 1 != successful LLM responses 2" in issue
        for issue in report["issues"]
    )


def test_validator_rejects_nested_truncated_training_step(tmp_path: Path) -> None:
    """Guards PR #921 MAX review against `truncation=\"disabled\"` truthiness."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    row = json.loads((rollout / "results.jsonl").read_text())
    row["trajectory"][0]["is_truncated"] = True
    _write_jsonl(rollout / "results.jsonl", [row])

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is False
    assert any(
        "trajectory[0] is incorrectly truncated" in issue for issue in report["issues"]
    )


def test_validator_excludes_recovered_incomplete_response_from_expected_steps(
    tmp_path: Path,
) -> None:
    """Guards PR #921 MAX review content-filter recovery semantics."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    llm_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    incomplete = json.loads(llm_path.read_text())
    incomplete["response"]["body"].update(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
        }
    )
    completed = json.loads(llm_path.read_text())
    completed["response"]["body"].update(
        {"status": "completed", "incomplete_details": None}
    )
    _write_jsonl(llm_path, [incomplete, completed])
    row = json.loads((rollout / "results.jsonl").read_text())
    row["trajectory"][0]["extras"]["exchange_index"] = 1
    _write_jsonl(rollout / "results.jsonl", [row])

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is True
    assert report["artifacts"]["llm"]["successful_responses"] == 1


def test_validator_deduplicates_completed_retry_race(tmp_path: Path) -> None:
    """Guards PR #921 MAX review against abandoned late responses."""
    validator = _load_validator()
    rollout = _rollout(tmp_path)
    llm_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    first = json.loads(llm_path.read_text())
    second = json.loads(llm_path.read_text())
    _write_jsonl(llm_path, [first, second])
    row = json.loads((rollout / "results.jsonl").read_text())
    row["trajectory"][0]["extras"]["exchange_index"] = 1
    _write_jsonl(rollout / "results.jsonl", [row])

    report = validator.validate_rollout(rollout)

    assert report["healthy"] is True
    assert report["artifacts"]["llm"]["successful_responses"] == 1
    assert report["artifacts"]["llm"]["deduplicated_completed_responses"] == 1

"""Emit Verifiers-shaped ``results.jsonl`` artifacts for BenchFlow rollouts.

This is the canonical trainer-facing rollout surface. It intentionally lives
beside, not inside, raw traces:

- ``trajectory/llm_trajectory.jsonl`` remains the provider HTTP audit log.
- ``trajectory/acp_trajectory.jsonl`` remains the ACP event audit log.
- ``results.jsonl`` is a Verifiers/Prime-RL-shaped rollout record.

The writer is fail-closed for training readiness but not for artifact
emission: even errored or unstructured rollouts get one JSONL row with an
``error``. The trainer-shaped trajectory is produced only from healthy
``llm_trajectory.jsonl`` exchanges; ACP is never used as a training fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import scrub_non_finite
from benchflow.trajectories._export_common import aggregate_rollout_jsonl
from benchflow.trajectories.export_prime_sft import _load_jsonl
from benchflow.trajectories.types import redact_trajectory_text

ROLLOUT_RESULTS_FILENAME = "results.jsonl"
JOB_RESULTS_FILENAME = "results.jsonl"


def _record_to_redacted_json_line(record: dict[str, Any]) -> str:
    raw = json.dumps(scrub_non_finite(record), default=str, allow_nan=False)
    return redact_trajectory_text(raw)


def _reward_value(rewards: dict[str, Any] | None) -> float:
    if not isinstance(rewards, dict):
        return 0.0
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return 0.0


def _metrics_from_rewards(
    rewards: dict[str, Any] | None,
    *,
    n_tool_calls: int,
    n_prompts: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_tool_calls": n_tool_calls,
        "n_prompts": n_prompts,
    }
    if not isinstance(rewards, dict):
        return metrics
    for key, value in rewards.items():
        if key == "rubric":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[str(key)] = float(value)
    nested_metrics = rewards.get("metrics")
    if isinstance(nested_metrics, dict):
        for key, value in nested_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[str(key)] = float(value)
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for idx, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, Any], item)
            score = item.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                metrics[str(item.get("name") or f"rubric_{idx}")] = float(score)
    return metrics


def _token_usage_from_agent_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _nonnegative_number(agent_result.get("n_input_tokens"))
    output_tokens = _nonnegative_number(agent_result.get("n_output_tokens"))
    total_tokens = _nonnegative_number(agent_result.get("total_tokens"))
    if input_tokens + output_tokens == 0 and total_tokens > 0:
        # Some harness/provider paths only expose a provider total. Prime-RL's
        # token-batch path needs final_input_tokens + final_output_tokens to be
        # non-zero; keep the total visible without inventing an output split.
        input_tokens = total_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_input_tokens": input_tokens,
        "final_output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _nonnegative_number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _error_payload(
    *,
    error: str | None,
    verifier_error: str | None,
    export_error: str | None,
) -> dict[str, str] | None:
    for key, value in (
        ("agent_error", error),
        ("verifier_error", verifier_error),
        ("export_error", export_error),
    ):
        if value:
            return {
                "error": key,
                "error_chain_str": value,
                "error_chain_repr": value,
            }
    return None


def _stop_condition(
    *,
    error: str | None,
    verifier_error: str | None,
    export_error: str | None,
    partial_trajectory: bool,
) -> str:
    if partial_trajectory:
        return "partial_trajectory"
    if error:
        return "agent_error"
    if verifier_error:
        return "verifier_error"
    if export_error:
        return "export_error"
    return "agent_completed"


def _llm_steps_from_trajectory(
    rollout_dir: Path,
    *,
    reward: float,
    is_truncated: bool,
    trajectory_id_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    for exchange_idx, exchange in enumerate(_load_jsonl(path)):
        response = exchange.get("response")
        if not isinstance(response, dict) or response.get("status_code") != 200:
            continue
        runtime_step = exchange.get("verifiers_step")
        if not isinstance(runtime_step, dict):
            continue
        prompt = runtime_step.get("prompt")
        completion = runtime_step.get("completion")
        if (
            not isinstance(prompt, list)
            or not isinstance(completion, list)
            or not completion
        ):
            continue
        tools = exchange.get("verifiers_tool_defs")
        if isinstance(tools, list) and tools:
            tool_defs = [tool for tool in tools if isinstance(tool, dict)]
        step = dict(runtime_step)
        extras = (
            cast(dict[str, Any], step.get("extras"))
            if isinstance(step.get("extras"), dict)
            else {}
        )
        extras = {
            **extras,
            "source": "llm_trajectory",
            "tracking_source": "litellm_callback",
            "exchange_index": exchange_idx,
        }
        step.update(
            {
                "prompt": prompt,
                "completion": completion,
                "reward": reward,
                "advantage": 0.0,
                "is_truncated": bool(runtime_step.get("is_truncated") or is_truncated),
                "trajectory_id": f"{trajectory_id_prefix}__llm_{exchange_idx}",
                "extras": extras,
            }
        )
        steps.append(step)
    return steps, tool_defs


def _top_level_prompt_completion(
    steps: list[dict[str, Any]],
    prompts: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if not steps:
        prompt = [{"role": "user", "content": prompt_text} for prompt_text in prompts]
        return prompt, None
    prompt = list(steps[0].get("prompt") or [])
    final_step = steps[-1]
    full_messages = list(final_step.get("prompt") or []) + list(
        final_step.get("completion") or []
    )
    if prompt and len(full_messages) >= len(prompt):
        return prompt, full_messages[len(prompt) :]
    return prompt, list(final_step.get("completion") or [])


def build_rollout_results_record(
    rollout_dir: str | Path,
    *,
    task_name: str,
    rollout_name: str,
    agent: str,
    agent_name: str,
    model: str | None,
    n_tool_calls: int,
    prompts: list[str],
    trajectory: list[dict[str, Any]],
    partial_trajectory: bool,
    rewards: dict[str, Any] | None,
    error: str | None,
    verifier_error: str | None,
    export_error: str | None = None,
    timing: dict[str, Any] | None = None,
    agent_result: dict[str, Any] | None = None,
    example_id: int = 0,
) -> dict[str, Any]:
    rollout_path = Path(rollout_dir)
    reward = _reward_value(rewards)
    trajectory_id_prefix = (
        rollout_name
        if rollout_name.startswith(f"{task_name}__")
        else f"{task_name}__{rollout_name}"
    )
    is_truncated = bool(partial_trajectory)
    steps, tool_defs = _llm_steps_from_trajectory(
        rollout_path,
        reward=reward,
        is_truncated=is_truncated,
        trajectory_id_prefix=trajectory_id_prefix,
    )
    prompt, completion = _top_level_prompt_completion(steps, prompts)
    error_obj = _error_payload(
        error=error,
        verifier_error=verifier_error,
        export_error=export_error,
    )
    training_ready = bool(steps and completion)
    training_ready_reason = None
    if not training_ready:
        training_ready_reason = "missing_healthy_structured_llm_trajectory"
        if error_obj is None:
            error_obj = {
                "error": "missing_llm_trajectory",
                "error_chain_str": "No healthy structured LLM trajectory was available for results.jsonl",
                "error_chain_repr": "No healthy structured LLM trajectory was available for results.jsonl",
            }
    completed = error_obj is None and not partial_trajectory
    token_usage = _token_usage_from_agent_result(agent_result or {})
    metrics = _metrics_from_rewards(
        rewards,
        n_tool_calls=n_tool_calls,
        n_prompts=len(prompts),
    )
    record = {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        "info": {
            "task_id": task_name,
            "task_name": task_name,
            "rollout_name": rollout_name,
            "environment": task_name,
            "agent": agent,
            "agent_name": agent_name,
            "model": model,
            "source": "benchflow",
            "rollout_dir": str(rollout_path),
            "training_ready": training_ready,
            "training_ready_reason": training_ready_reason,
            **(
                {"reward_details": rewards.get("details")}
                if isinstance(rewards, dict)
                and isinstance(rewards.get("details"), dict)
                else {}
            ),
        },
        "reward": reward,
        "error": error_obj,
        "timing": timing or {},
        "is_completed": completed,
        "is_truncated": is_truncated,
        "stop_condition": _stop_condition(
            error=error,
            verifier_error=verifier_error,
            export_error=export_error,
            partial_trajectory=partial_trajectory,
        ),
        "metrics": metrics,
        "tool_defs": tool_defs,
        "token_usage": token_usage,
        "score": reward,
        "total_tool_calls": float(n_tool_calls),
        "trajectory": steps,
    }
    return record


def write_rollout_results_jsonl(
    rollout_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write one Verifiers-shaped row to ``rollout_dir/results.jsonl``."""
    record = build_rollout_results_record(rollout_dir, **kwargs)
    out = Path(rollout_dir) / ROLLOUT_RESULTS_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_record_to_redacted_json_line(record) + "\n")
    return record


def write_job_results_jsonl(job_dir: str | Path) -> Path | None:
    """Aggregate per-rollout ``results.jsonl`` files into ``job_dir/results.jsonl``."""
    return aggregate_rollout_jsonl(
        job_dir,
        rollout_relpath=ROLLOUT_RESULTS_FILENAME,
        out_filename=JOB_RESULTS_FILENAME,
    )

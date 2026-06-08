"""Shared fixtures for continue_run tests — synthetic run folders/exchanges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchflow.trajectories.types import LLMExchange, LLMRequest, LLMResponse


def completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "test-model",
    cmpl_id: str = "cmpl-1",
) -> dict[str, Any]:
    """A minimal but valid OpenAI ChatCompletion response body."""
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def exchange(
    response_body: dict[str, Any],
    *,
    n_request_messages: int = 1,
    status: int = 200,
) -> LLMExchange:
    """Build an LLMExchange with a request carrying ``n_request_messages``."""
    return LLMExchange(
        request=LLMRequest(body={"messages": [{"role": "user"}] * n_request_messages}),
        response=LLMResponse(status_code=status, body=response_body),
    )


def write_run_folder(
    root: Path,
    *,
    exchanges: list[LLMExchange],
    agent: str = "openhands",
    model: str | None = "aws-bedrock/us.anthropic.claude-opus-4-8",
    error_category: str | None = "timeout",
    task_name: str = "demo-task",
    prompts: list[str] | None = None,
    timeout_sec: int = 3600,
) -> Path:
    """Materialize a synthetic run folder benchflow continue can load."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_path": f"/tasks/{task_name}",
                "agent": agent,
                "model": model,
                "environment": "docker",
                "sandbox_user": "agent",
                "timeout_sec": timeout_sec,
                "agent_idle_timeout_sec": 600,
            }
        )
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "agent": agent,
                "model": model,
                "error_category": error_category,
                "rewards": None,
            }
        )
    )
    (root / "prompts.json").write_text(json.dumps(prompts or ["Do the task."]))
    traj = root / "trajectory"
    traj.mkdir(exist_ok=True)
    (traj / "llm_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(ex.model_dump(mode="json"), default=str) for ex in exchanges
        )
        + "\n"
    )
    return root

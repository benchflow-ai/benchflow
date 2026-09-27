"""How training-grade is a job's gateway token capture?

An online RL client needs every model call's prompt token ids,
sampled token ids and logprobs, captured at the gateway and never
re-tokenized. Capture exists (``benchflow.token_capture.v1``) but nothing
said whether a job actually has it: agent trials that run on subscription auth (``endpoint_kind:
agent_native``) bypass the gateway and have no ``llm_trajectory.jsonl`` at
all. ``summarize_token_capture`` / ``bench train token-coverage`` report it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_SCHEMA_VERSION,
    summarize_rollout_token_capture,
    summarize_token_capture,
)

runner = CliRunner()


def _call(
    prompt_ids: list[int] | None,
    completion_ids: list[int] | None,
    logprobs: list[float] | None,
    *,
    unavailable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request": {"body": {}},
        "response": {"status_code": 200, "body": {}},
        "metadata": {
            "token_capture": {
                "schema_version": TOKEN_CAPTURE_SCHEMA_VERSION,
                "wire": "openai-chat",
                "provider": "vllm",
                "requested": {
                    "logprobs": True,
                    "top_logprobs": None,
                    "token_ids": True,
                },
                "prompt_token_ids": prompt_ids,
                "completions": [
                    {
                        "index": 0,
                        "token_ids": completion_ids,
                        "tokens": None,
                        "logprobs": logprobs,
                        "top_logprobs": None,
                    }
                ],
                "unavailable": unavailable or {},
            }
        },
    }


def _anthropic_call() -> dict[str, Any]:
    reason = {"reason": "provider_api_unsupported", "detail": "Anthropic Messages"}
    call = _call(None, None, None)
    capture = call["metadata"]["token_capture"]
    capture["wire"] = "anthropic-messages"
    capture["provider"] = "anthropic"
    capture["unavailable"] = {
        "prompt_token_ids": reason,
        "completion_token_ids": reason,
        "logprobs": reason,
    }
    return call


def _rollout(root: Path, name: str, calls: list[dict[str, Any]] | None, **usage):
    rollout = root / name
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": name, "agent": "claude-agent-acp", "usage_tracking": usage}
        )
    )
    if calls is not None:
        (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in calls)
        )
    return rollout


def test_a_token_in_token_out_conversation_is_training_grade():
    calls = [
        _call([1, 2, 3], [4, 5], [-0.1, -0.2]),
        # The next prompt is the previous prompt + sampled tokens + new turn.
        _call([1, 2, 3, 4, 5, 6, 7], [8], [-0.3]),
    ]
    summary = summarize_token_capture(calls)
    assert summary["calls"] == 2
    assert summary["complete_calls"] == 2
    assert summary["prefix"] == {"pairs": 1, "extends_previous_call": 1, "breaks": []}
    assert summary["training_grade"] is True


def test_a_re_rendered_prompt_is_a_prefix_break():
    calls = [
        _call([1, 2, 3], [4, 5], [-0.1, -0.2]),
        # The client re-rendered the history: token 4 came back as 9.
        _call([1, 2, 3, 9, 5, 6], [8], [-0.3]),
    ]
    summary = summarize_token_capture(calls)
    assert summary["complete_calls"] == 2
    assert summary["prefix"]["breaks"] == [1]
    assert summary["training_grade"] is False


def test_provider_limits_are_counted_by_reason():
    summary = summarize_token_capture([_anthropic_call(), _anthropic_call()])
    assert summary["complete_calls"] == 0
    assert summary["unavailable"] == {
        "completion_token_ids:provider_api_unsupported": 2,
        "logprobs:provider_api_unsupported": 2,
        "prompt_token_ids:provider_api_unsupported": 2,
    }
    assert summary["training_grade"] is False


def test_a_call_without_a_capture_block_is_uncaptured():
    plain = {"request": {"body": {}}, "response": {"status_code": 200, "body": {}}}
    summary = summarize_token_capture([plain])
    assert summary["captured_calls"] == 0
    assert summary["training_grade"] is False


def test_a_native_subscription_rollout_says_it_bypassed_the_gateway(tmp_path):
    rollout = _rollout(
        tmp_path,
        "native",
        None,
        endpoint_kind="agent_native",
        usage_source="agent_native_acp",
    )
    summary = summarize_rollout_token_capture(rollout)
    assert summary["status"] == "no_gateway_capture"
    assert "agent_native" in summary["reason"]
    assert summary["training_grade"] is False


def test_cli_reports_a_job(tmp_path):
    job = tmp_path / "job"
    _rollout(
        job,
        "vllm-run",
        [_call([1], [2], [-0.5]), _call([1, 2, 3], [4], [-0.1])],
        endpoint_kind="sandbox",
    )
    _rollout(job, "native", None, endpoint_kind="agent_native")
    result = runner.invoke(app, ["train", "token-coverage", str(job), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["rollouts"] == 2
    assert report["training_grade_rollouts"] == 1
    by_name = {r["rollout"]: r for r in report["per_rollout"]}
    assert by_name["vllm-run"]["training_grade"] is True
    assert by_name["native"]["status"] == "no_gateway_capture"

    text = runner.invoke(app, ["train", "token-coverage", str(job)])
    assert text.exit_code == 0, text.output
    assert "1 of 2 rollouts" in text.output

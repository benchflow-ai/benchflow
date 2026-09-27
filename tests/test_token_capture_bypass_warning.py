"""Token capture asked for on a run that never reaches the gateway says so.

``BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`` on a
subscription-auth (or oracle) run was silently ignored. The gateway, which
does the capture, is skipped for those runs, so the job finished with no
``llm_trajectory.jsonl`` and no hint why.
"""

from __future__ import annotations

import logging

import pytest

from benchflow.providers import litellm_runtime


@pytest.mark.asyncio
async def test_capture_on_a_native_subscription_run_warns(monkeypatch, caplog):
    monkeypatch.setattr(
        litellm_runtime, "uses_native_subscription_auth", lambda *a: True
    )
    with caplog.at_level(logging.WARNING, logger=litellm_runtime.logger.name):
        _env, runtime = await litellm_runtime.ensure_litellm_runtime(
            agent="claude-agent-acp",
            agent_env={"BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1"},
            model="claude-opus-5-5",
            runtime=None,
            environment="docker",
        )
    assert runtime is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS" in m and "subscription" in m
        for m in warnings
    ), warnings


@pytest.mark.asyncio
async def test_no_warning_without_capture(monkeypatch, caplog):
    monkeypatch.setattr(
        litellm_runtime, "uses_native_subscription_auth", lambda *a: True
    )
    with caplog.at_level(logging.WARNING, logger=litellm_runtime.logger.name):
        await litellm_runtime.ensure_litellm_runtime(
            agent="claude-agent-acp",
            agent_env={},
            model="claude-opus-5-5",
            runtime=None,
            environment="docker",
        )
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]

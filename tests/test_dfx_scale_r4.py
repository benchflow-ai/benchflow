"""Regressions in b0a8a3435: operational API failures and streamed requests."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from benchflow.evaluation import ApiErrorCircuitBreaker, RetryConfig
from benchflow.providers import litellm_request_guard as guard
from benchflow.providers.litellm_runtime import LiteLLMProcess
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._results import _build_rollout_result


async def test_reported_invalid_response_still_opens_permanent_api_breaker(tmp_path):
    """R3-M1: restore dc509087a's breaker behavior regressed at b0a8a3435."""
    events = [
        dict(event="request_start", request_id="r", time=100.0),
        dict(
            event="upstream_end",
            request_id="r",
            attempt_id="a",
            time=100.1,
            status=503,
            invalid_provider_response=True,
            response_bytes=8,
            upstream_seconds=0.1,
        ),
        dict(
            event="upstream_end",
            request_id="r",
            attempt_id="b",
            time=101.1,
            status=401,
            invalid_provider_response=False,
            response_bytes=32,
            upstream_seconds=0.1,
        ),
        dict(
            event="request_end",
            request_id="r",
            time=101.2,
            status=401,
            retry_wait_seconds=0.9,
        ),
    ]
    process = LiteLLMProcess.__new__(LiteLLMProcess)
    process._request_events = {
        (event["event"], event.get("attempt_id", event["request_id"])): event
        for event in events
    }
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="claude-agent-acp",
            environment="docker",
        )
    )

    class Server:
        trajectory = SimpleNamespace(
            exchanges=[SimpleNamespace(response=SimpleNamespace(status_code=401))]
        )

        def request_summary(self):
            return process.request_summary()

    rollout._usage_runtime = SimpleNamespace(server=Server())
    rollout._planes = MagicMock(
        stop_provider_runtime=AsyncMock(), extract_usage=MagicMock(return_value={})
    )
    rollout._executed_prompts = ["probe"]
    await rollout._teardown_usage_runtime()
    rollout._maybe_classify_api_error()
    result = _build_rollout_result(
        tmp_path,
        task_name="probe",
        rollout_name="probe",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model=None,
        n_tool_calls=0,
        prompts=["probe"],
        error=rollout._error,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=None,
        started_at=datetime.now(),
        timing=rollout._timing,
        diagnostics=rollout._diagnostics,
    )
    assert result.error_category == "invalid_provider_response"
    assert "provider api error [auth/permanent] HTTP 401" in result.error
    assert not RetryConfig().should_retry(result.error, category=result.error_category)
    breaker = ApiErrorCircuitBreaker(threshold=5)
    for _ in range(4):
        breaker.record(result)
        assert not breaker.tripped
    breaker.record(result)
    assert breaker.tripped
    assert "auth:401" in breaker.skip_error()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("broken_telemetry", [False, True])
async def test_streamed_request_reaches_transport(
    tmp_path, monkeypatch, asynchronous, broken_telemetry
):
    """R3-M2: preserve dc509087a's streamed sends, broken at b0a8a3435."""
    import aiohttp

    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    monkeypatch.setattr(
        aiohttp.ClientSession, "_request", aiohttp.ClientSession._request
    )
    guard.install_outbound_check()
    if broken_telemetry:
        monkeypatch.setattr(
            guard, "_UpstreamAttempt", MagicMock(side_effect=RuntimeError("telemetry"))
        )
    parts = [b'{"model":"test",', b'"messages":[]}']
    received = []

    async def chunks():
        for part in parts:
            yield part

    async def async_transport(request):
        received.append(await request.aread())
        return httpx.Response(200, json={"ok": True})

    def sync_transport(request):
        received.append(request.read())
        return httpx.Response(200, json={"ok": True})

    async def app(scope, receive, send):
        if asynchronous:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(async_transport)
            ) as client:
                response = await client.post(
                    "https://provider.test/v1/chat/completions", content=chunks()
                )
        else:
            with httpx.Client(transport=httpx.MockTransport(sync_transport)) as client:
                response = client.post(
                    "https://provider.test/v1/chat/completions", content=iter(parts)
                )
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        await send({"type": "http.response.start", "status": response.status_code})
        await send({"type": "http.response.body", "body": response.content})

    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b"".join(parts),
            "more_body": False,
        }
    )
    await guard.SingleRouteGuard(app)(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive,
        AsyncMock(),
    )
    assert received == [b"".join(parts)]
    if not broken_telemetry:
        starts = [
            row
            for line in log.read_text().splitlines()
            if (row := json.loads(line))["event"] == "upstream_start"
        ]
        assert len(starts) == 1
        assert starts[0]["request_bytes"] is None

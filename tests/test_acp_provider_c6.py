"""Guards the fix at f4324120."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request

from benchflow.providers.litellm_request_guard import SingleRouteGuard


@pytest.fixture
def boundary(monkeypatch):
    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {
            "path": request.url.path,
            "body": await request.json(),
            "content_type": request.headers.get("content-type"),
        }

    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["allowed", "gemini/gemini-2.5-flash"]'
    )
    app.add_middleware(SingleRouteGuard)
    return app


@pytest.mark.parametrize(
    "sequence", [(True, False), (False, True), (True, False, True, False)]
)
def test_usage_counts_only_uncovered_native_roles(sequence):
    from benchflow.acp.session import ACPSession
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(RolloutConfig(task_path=Path("dummy-task")))
    for covered in sequence:
        rollout._native_usage_proxy_covered = covered
        rollout._native_usage_checkpoint = None
        rollout._session = ACPSession("dummy")
        rollout._session.record_prompt_usage(
            SimpleNamespace(input_tokens=80, output_tokens=20, total_tokens=100)
        )
        rollout._collect_native_acp_usage()
        rollout._collect_native_acp_usage()
    rollout._usage_metrics = {
        "usage_source": "provider_response",
        "total_tokens": 100 * sum(sequence),
        "n_input_tokens": 80 * sum(sequence),
        "n_output_tokens": 20 * sum(sequence),
        "cost_usd": 0.25,
    }
    rollout._finalize_usage_metrics()
    rollout._finalize_usage_metrics()
    assert rollout._usage_metrics["total_tokens"] == 100 * len(sequence)
    assert rollout._usage_metrics["n_input_tokens"] == 80 * len(sequence)
    assert rollout._usage_metrics["cost_usd"] is None  # subscription cost is unknown

"""The run's gateway refuses, and records, a request for a model it does not serve.

codex-acp's title thread asks for ``gpt-5.6-luna`` with the task prompt as
input. Once Codex routes it to BenchFlow's proxy (test_codex_home_config.py),
the proxy must neither forward it upstream nor lose sight of it. LiteLLM's
router refuses a model its config does not list; BenchFlow's callback now
refuses it first, whatever LiteLLM's routing fallbacks do, and records the
attempt without the request body. The real-proxy test runs LiteLLM against a
local mock upstream that records every request it gets.
"""

import asyncio
import json
import logging
import time

import httpx
import pytest

import benchflow.providers.litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)
from benchflow.providers.litellm_logging import (
    callback_module_source,
    trajectory_from_litellm_callback_log,
)
from tests.fixtures.mock_token_logprobs_server import start_server

TASK_PROMPT = "canary-7f3a: solve the task in /app"
FOREIGN = "gpt-5.6-luna"


def _logger(monkeypatch, tmp_path, served):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    if served is None:
        monkeypatch.delenv("BENCHFLOW_LITELLM_SERVED_MODELS", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", json.dumps(served))
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace, namespace["proxy_handler_instance"], log


@pytest.mark.asyncio
async def test_hook_refuses_and_records_a_model_the_run_does_not_serve(
    monkeypatch, tmp_path
):
    namespace, logger, log = _logger(monkeypatch, tmp_path, ["benchflow-x", "gpt-6"])
    data = {"model": FOREIGN, "input": TASK_PROMPT}
    with pytest.raises(namespace["ModelNotServed"]) as refused:
        await logger.async_pre_call_hook(None, None, data, "aresponses")
    assert refused.value.status_code == 400
    [record] = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["event"] == "refused"
    assert record["rule"] == "model-not-served"
    assert record["request_model"] == FOREIGN
    assert record["call_type"] == "aresponses"
    assert TASK_PROMPT not in log.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("served", [None, ["gpt-6"]])
async def test_hook_passes_a_served_model_and_is_off_without_a_list(
    monkeypatch, tmp_path, served
):
    _, logger, log = _logger(monkeypatch, tmp_path, served)
    model = "gpt-6" if served else FOREIGN
    data = {"model": model, "input": TASK_PROMPT}
    assert await logger.async_pre_call_hook(None, None, data, "aresponses") is None
    assert not log.exists()


def test_refused_records_are_not_exchanges(caplog):
    log = "\n".join(
        [
            json.dumps(
                {
                    "event": "refused",
                    "rule": "model-not-served",
                    "request_model": FOREIGN,
                    "call_type": "aresponses",
                    "logged_at": "2026-09-29T00:00:00+00:00",
                }
            ),
            json.dumps(
                {
                    "event": "success",
                    "request_model": "gpt-6",
                    "request": {"method": "POST", "path": "/v1/responses", "body": {}},
                    "response": {"model": "gpt-6"},
                }
            ),
        ]
    )
    trajectory = trajectory_from_litellm_callback_log(
        log, session_id="s", agent_name="codex-acp"
    )
    assert [e.metadata["request_model"] for e in trajectory.exchanges] == ["gpt-6"]
    assert trajectory.metadata["refused_requests"] == [
        {
            "request_model": FOREIGN,
            "call_type": "aresponses",
            "rule": "model-not-served",
            "logged_at": "2026-09-29T00:00:00+00:00",
        }
    ]
    with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
        runtime_mod._warn_refused_requests(trajectory)
    assert (
        f"refused 1 request(s) for model(s) this run does not serve: {FOREIGN} x1"
        in (caplog.text)
    )


def test_served_models_are_every_name_the_final_config_routes():
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:9/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "sk-mock",
    }
    route = resolve_litellm_route("vllm/mock-policy", env)
    config = litellm_proxy_config(route, master_key="sk-master")
    # A layer that serves a companion next to the run's model (FrontierPhysics
    # does this for Codex's guardian) must not be refused by the gate.
    config["model_list"].append(
        {"model_name": "companion", "litellm_params": {"model": "openai/companion"}}
    )
    served = json.loads(
        runtime_mod._served_models_env(config)["BENCHFLOW_LITELLM_SERVED_MODELS"]
    )
    names = {entry["model_name"] for entry in config["model_list"]}
    upstream = {entry["litellm_params"]["model"] for entry in config["model_list"]}
    assert set(served) == names | upstream
    assert route.model_alias in served and "mock-policy" in served
    assert "companion" in served and FOREIGN not in served


async def _drive(tmp_path):
    server = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": f"{server.base_url}/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "sk-mock",
        "OPENAI_API_KEY": "sk-mock",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    route = resolve_litellm_route("vllm/mock-policy", env)
    proc = await runtime_mod._start_host_litellm(
        route=route,
        master_key="sk-master",
        agent_env=env,
        environment="local",
        session_id="s",
        agent_name="codex-acp",
    )
    statuses = {}
    try:
        headers = {"Authorization": "Bearer sk-master"}
        async with httpx.AsyncClient(timeout=60) as client:
            for name, path, body in (
                ("title", "/v1/responses", {"model": FOREIGN, "input": TASK_PROMPT}),
                (
                    "chat",
                    "/v1/chat/completions",
                    {
                        "model": FOREIGN,
                        "messages": [{"role": "user", "content": TASK_PROMPT}],
                    },
                ),
                (
                    "run",
                    "/v1/chat/completions",
                    {
                        "model": route.model_alias,
                        "messages": [{"role": "user", "content": TASK_PROMPT}],
                    },
                ),
            ):
                response = await client.post(
                    proc.base_url + path, headers=headers, json=body
                )
                statuses[name] = (response.status_code, response.text)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            lines = (
                proc.log_path.read_text().splitlines() if proc.log_path.exists() else []
            )
            if len(lines) >= 3:
                break
            await asyncio.sleep(0.2)
        raw_log = proc.log_path.read_text()
    finally:
        await proc.stop()
        server.shutdown()
    return statuses, server.requests, raw_log, proc.trajectory


def test_real_gateway_keeps_the_task_prompt_off_a_foreign_model(tmp_path):
    statuses, upstream, raw_log, trajectory = asyncio.run(_drive(tmp_path))
    assert statuses["title"][0] == 400, statuses["title"]
    assert "model_not_served" in statuses["title"][1]
    assert statuses["chat"][0] == 400, statuses["chat"]
    assert statuses["run"][0] == 200, statuses["run"]
    # Only the run's own request reached the upstream; the prompt went nowhere else.
    assert len(upstream) == 1, upstream
    assert TASK_PROMPT in json.dumps(upstream[0])
    assert FOREIGN not in json.dumps(upstream)
    refused = [
        record
        for record in map(json.loads, raw_log.splitlines())
        if record.get("event") == "refused"
    ]
    assert [r["request_model"] for r in refused] == [FOREIGN, FOREIGN]
    assert TASK_PROMPT not in json.dumps(refused)
    # Refusals are not provider exchanges; they are counted beside them.
    assert len(trajectory.exchanges) == 1
    assert [r["request_model"] for r in trajectory.metadata["refused_requests"]] == [
        FOREIGN,
        FOREIGN,
    ]

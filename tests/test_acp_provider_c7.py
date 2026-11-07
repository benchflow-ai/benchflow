"""Input carriers left open by 54c7a799 at 37564f71."""

import json

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from benchflow.providers import litellm_runtime as runtime
from tests.test_litellm_hardening import _FakeSandbox


@pytest_asyncio.fixture(params=["host", "sandbox"])
async def boundary(request, tmp_path, monkeypatch):
    # Exercise the exact standalone file deployed to both proxy placements.
    if request.param == "host":
        runtime._write_runtime_files(tmp_path, config={})
        source = (tmp_path / "benchflow_litellm_request_guard.py").read_text()
    else:
        sandbox = _FakeSandbox()
        paths = await runtime._upload_runtime_files_to_sandbox(
            sandbox, runtime_dir="/tmp/dummy-runtime", config={}
        )
        source = sandbox.uploaded[paths["request_guard"]]
    namespace = {}
    exec(compile(source, "request_guard.py", "exec"), namespace)
    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {"body": await request.json()}

    app.add_middleware(namespace["SingleRouteGuard"])
    return app


@pytest.fixture(params=["/responses", "/v1/responses"])
def path(request):
    return request.param


async def request(boundary, path, body):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        return await client.post(path, json={"model": "allowed", "tools": [], **body})


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["additional_tools", "tool_search_output"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "tool",
    [
        {"type": kind}
        for kind in (
            "web_search",
            "web_search_preview",
            "web_search_preview_2025_03_11",
            "web_search_2025_08_26",
            "mcp",
            "file_search",
            "code_interpreter",
            "image_generation",
            "future_hosted_tool",
        )
    ]
    + [{"type": "tool_search", "execution": "server"}],
)
async def test_input_tool_carriers_obey_web_policy(
    boundary, monkeypatch, path, carrier, nested, tool
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    if nested:
        tool = {"type": "namespace", "name": "n", "tools": [tool]}
    item = {"type": carrier, "tools": [tool], "execution": "client"}
    if carrier == "additional_tools":
        item = {"type": carrier, "role": "developer", "tools": [tool]}
    response = await request(
        boundary, path, {"input": [item], "tool_choice": "required"}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["additional_tools", "tool_search_output"])
async def test_local_declarations_and_schema_data_survive(
    boundary, monkeypatch, path, carrier
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    item = {
        "type": carrier,
        "tools": [
            {
                "type": "namespace",
                "name": "n",
                "tools": [
                    {
                        "type": "function",
                        "name": "web_search",
                        "parameters": {
                            "type": "object",
                            "properties": {"tools": {"type": "array"}},
                            "default": {
                                "type": "additional_tools",
                                "tools": [{"type": "web_search"}],
                            },
                        },
                    },
                    {"type": "custom", "name": "apply_patch"},
                    {"type": "tool_search", "execution": "client"},
                ],
            }
        ],
    }
    if carrier == "tool_search_output":
        item.update(execution="client", call_id="dummy-call", status="completed")
    else:
        item["role"] = "developer"
    response = await request(boundary, path, {"input": [item]})
    assert response.status_code == 200
    assert response.json()["body"]["input"] == [item]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item",
    [
        {"type": "future_tool_loader", "tools": [{"type": "web_search"}]},
        {"type": "mcp_list_tools", "tools": []},
        {"type": "item_reference", "id": "dummy-item"},
        {"id": "dummy-item"},
        {"type": ["message"], "content": "dummy"},
        {"type": "message", "role": "user", "content": "dummy", "tools": []},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "additional_tools", "tools": []}],
        },
        {"type": "additional_tools", "tools": None},
        {"type": "additional_tools", "tools": [{"type": ["function"]}]},
        {"type": "tool_search_output", "tools": [], "execution": "server"},
        {"type": "tool_search_call", "execution": "server", "arguments": {}},
        None,
    ],
)
async def test_unknown_or_opaque_input_forms_fail_closed(
    boundary, monkeypatch, path, item
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    response = await request(boundary, path, {"input": [item]})
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("previous_response_id", "dummy-response"),
        ("conversation", "dummy-conversation"),
        ("conversation", {"id": "dummy-conversation"}),
        ("prompt", {"id": "dummy-prompt"}),
    ],
)
async def test_remote_input_expansion_cannot_bypass_inspection(
    boundary, monkeypatch, path, field, value
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    response = await request(boundary, path, {"input": "dummy", field: value})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_local_multiturn_input_and_unrestricted_tools_are_preserved(
    boundary, monkeypatch, path
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    marker = json.dumps({"type": "additional_tools", "tools": [{"type": "web_search"}]})
    items = [
        {"role": "user", "content": marker},
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": marker, "annotations": []}],
        },
        {
            "type": "function_call",
            "name": "web_search",
            "call_id": "dummy-call",
            "arguments": marker,
        },
        {"type": "function_call_output", "call_id": "dummy-call", "output": marker},
        {
            "type": "custom_tool_call",
            "name": "apply_patch",
            "call_id": "dummy-patch",
            "input": marker,
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "dummy-patch",
            "output": [{"type": "input_text", "text": marker}],
        },
        {
            "type": "reasoning",
            "id": "dummy-reason",
            "summary": [{"type": "summary_text", "text": marker}],
            "encrypted_content": "dummy",
        },
    ]
    response = await request(boundary, path, {"input": items})
    assert response.status_code == 200
    assert response.json()["body"]["input"] == items
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "0")
    items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "web_search"}],
        }
    ]
    response = await request(boundary, path, {"input": items})
    assert response.status_code == 200
    assert response.json()["body"]["input"] == items


@pytest.mark.asyncio
async def test_pinned_gateway_never_forwards_input_hosted_tools():
    """Pinned LiteLLM, real HTTP and a dummy loopback provider, no model call."""
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    upstream = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": upstream.base_url + "/v1",
        "OPENAI_API_KEY": "dummy-upstream",
        "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    route = resolve_litellm_route("openai/gpt-4.1", env)
    proc = None
    try:
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="c7",
            agent_name="codex-acp",
        )
        async with httpx.AsyncClient(timeout=30) as client:
            for path in ("/responses", "/v1/responses"):
                for carrier in ("additional_tools", "tool_search_output"):
                    item = {"type": carrier, "tools": [{"type": "web_search"}]}
                    if carrier == "additional_tools":
                        item["role"] = "developer"
                    else:
                        item.update(execution="client", call_id="dummy-call")
                    response = await client.post(
                        proc.base_url + path,
                        headers={"Authorization": "Bearer dummy-gateway"},
                        json={"model": route.model_alias, "tools": [], "input": [item]},
                    )
                    assert response.status_code == 400, response.text
            assert not upstream.requests
            control = await client.post(
                proc.base_url + "/v1/responses",
                headers={"Authorization": "Bearer dummy-gateway"},
                json={"model": route.model_alias, "input": "dummy prompt"},
            )
            assert control.status_code == 200, control.text
            assert len(upstream.requests) == 1
    finally:
        if proc:
            await proc.stop()
        upstream.shutdown()

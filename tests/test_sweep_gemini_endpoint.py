"""Guards the fix at 363d815b: native endpoint equals resolved route."""

import json

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_config import resolve_litellm_route
from tests.test_litellm_hardening import _FakeSandbox


@pytest.mark.parametrize(
    "base",
    [
        "https://relay.example.test",
        "https://relay.example.test/prefix",
        "https://relay.example.test/prefix/",
    ],
)
@pytest.mark.parametrize("key_name", ["GEMINI_API_KEY", "BENCHFLOW_PROVIDER_API_KEY"])
@pytest.mark.asyncio
async def test_native_gemini_uses_resolved_endpoint_and_key(
    base, key_name, monkeypatch
):
    async def layer(*args, **kwargs):
        return "/opt/benchflow/litellm/bin/python", None

    monkeypatch.setattr(runtime, "_proxy_from_layer", layer)
    env = {
        "GEMINI_API_KEY": "dummy-original",
        "BENCHFLOW_PROVIDER_BASE_URL": base,
        key_name: "dummy-resolved",
    }
    route = resolve_litellm_route("gemini-2.5-flash", env)
    sandbox = _FakeSandbox()
    await runtime._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="dummy-gateway",
        agent_env=env,
        session_id="s",
        agent_name="gemini",
    )
    launch = json.loads(
        next(v for k, v in sandbox.uploaded.items() if k.endswith("launch_config.json"))
    )
    assert launch["env"]["GEMINI_API_BASE"] == base
    assert launch["env"]["GEMINI_API_KEY"] == "dummy-resolved"


@pytest.mark.asyncio
async def test_pinned_native_gemini_honors_endpoint_and_refuses_foreign_model():
    """A decoy GEMINI_API_BASE keeps both base and tip strictly on loopback."""
    import httpx

    from tests.fixtures.mock_token_logprobs_server import start_server

    intended = start_server()
    decoy = start_server()
    env = {
        "GEMINI_API_KEY": "dummy-gemini",
        "GEMINI_API_BASE": decoy.base_url,
        "BENCHFLOW_PROVIDER_BASE_URL": intended.base_url + "/prefix",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    route = resolve_litellm_route("gemini-2.5-flash", env)
    proc = None
    try:
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="sweep",
            agent_name="gemini",
        )
        async with httpx.AsyncClient(timeout=30) as client:
            foreign = await client.post(
                proc.base_url
                + "/gemini/v1beta/models/gemini-2.5-pro:generateContent?key=dummy-gateway",
                json={"contents": []},
            )
            assert foreign.status_code == 400
            assert not intended.requests and not decoy.requests
            # Mock upstream returns 404 for this API; the recorded destination
            # proves forwarding without needing a model or a provider key.
            await client.post(
                proc.base_url
                + "/gemini/v1beta/models/gemini-2.5-flash:generateContent?key=dummy-gateway",
                json={"contents": []},
            )
            assert [r["path"] for r in intended.requests] == [
                "/prefix/v1beta/models/gemini-2.5-flash:generateContent"
            ]
            assert not decoy.requests
    finally:
        if proc:
            await proc.stop()
        intended.shutdown()
        decoy.shutdown()

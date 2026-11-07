"""Guards the fix at 363d815b: ADC belongs to the proxy."""

from types import SimpleNamespace

import pytest

from benchflow.agents.credentials import write_credential_files
from benchflow.agents.providers import PROVIDERS
from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_config import resolve_litellm_route


@pytest.mark.parametrize(
    ("agent", "model"),
    [
        (agent, model)
        for model in (
            "google-vertex/gemini-2.5-flash",
            "anthropic-vertex/claude-sonnet-4-6",
        )
        for agent in ("claude-agent-acp", "gemini", "openhands", "codex-acp")
        # Gemini CLI on google-vertex/ is the one native Vertex
        # route; its ADC file is its login (test_sweep_credential_policy_matrix).
        if not (agent == "gemini" and model.startswith("google-vertex/"))
    ],
)
@pytest.mark.asyncio
async def test_proxied_provider_never_stages_adc_in_agent_home(agent, model):
    class Sandbox:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=0)

        async def upload_file(self, *args, **kwargs):
            raise AssertionError("upstream credential was staged for the agent")

    await write_credential_files(
        Sandbox(),
        agent,
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"dummy":"adc"}'},
        None,
        model,
        "/tmp/dummy-agent",
    )


@pytest.mark.parametrize(
    "agent", ["claude-agent-acp", "gemini", "openhands", "codex-acp"]
)
def test_all_declared_provider_file_sources_are_filtered(agent):
    names = {"GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_JSON"}
    for provider in PROVIDERS.values():
        for cf in provider.credential_files:
            names.add(cf["env_source"])
            names.update(cf.get("post_env", {}))
    env = dict.fromkeys(names, "dummy-adc")
    route = resolve_litellm_route("anthropic-vertex/claude-sonnet-4-6", env)
    wired = runtime._apply_litellm_agent_env(
        agent=agent,
        agent_env=env,
        route=route,
        base_url="http://gateway.invalid",
        master_key="dummy-gateway",
    )
    assert names.isdisjoint(wired)

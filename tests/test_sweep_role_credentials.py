"""Guards the fix at 363d815b: effective role environments."""

import pytest

from benchflow.providers import litellm_runtime as runtime
from tests.test_litellm_runtime import FakeLiteLLMServer


@pytest.mark.parametrize("environment", ["local", "remote-docker"])
@pytest.mark.parametrize(
    "changed",
    [
        "ANTHROPIC_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "AWS_SESSION_TOKEN",
        "HTTPS_PROXY",
    ],
)
@pytest.mark.asyncio
async def test_changed_effective_env_restarts_gateway(
    monkeypatch, environment, changed
):
    starts = []

    async def start(**kwargs):
        starts.append(kwargs["agent_env"])
        return FakeLiteLLMServer("http://gateway.invalid", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    env = {"ANTHROPIC_API_KEY": "dummy-original", changed: "dummy-original"}
    kwargs = dict(
        agent="claude-agent-acp",
        model="anthropic/claude-sonnet-4-6",
        environment=environment,
        session_id="role",
        sandbox=object(),
    )
    wired, first = await runtime.ensure_litellm_runtime(
        agent_env=env, runtime=None, **kwargs
    )
    _, same = await runtime.ensure_litellm_runtime(
        agent_env=env, runtime=first, **kwargs
    )
    assert same is first
    # A reconnect sees wired env and must still recover the original sources.
    _, same = await runtime.ensure_litellm_runtime(
        agent_env=wired, runtime=first, **kwargs
    )
    assert same is first
    _, second = await runtime.ensure_litellm_runtime(
        agent_env={**env, changed: "dummy-replacement"}, runtime=first, **kwargs
    )
    assert second is not first
    assert first.server.stopped
    assert len(starts) == 2
    assert starts[-1][changed] == "dummy-replacement"

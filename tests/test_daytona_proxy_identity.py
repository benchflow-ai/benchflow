"""Daytona proxy reuse follows the complete install requested by each role."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers import proxy_requirements
from benchflow.providers.runtime import ensure_litellm_runtime
from benchflow.sandbox.daytona_layers import PROXY_LAYER_VENV, plan_install_layers
from tests.test_daytona_install_layers import _LayerSandbox

MODELS = (
    "anthropic/claude-sonnet-4-6",
    "google-vertex/gemini-2.5-flash",
    "anthropic-vertex/claude-sonnet-4-6",
)
ENV = {
    "ANTHROPIC_API_KEY": "dummy-provider-marker",
    "VERTEX_PROJECT": "dummy-project",
    "VERTEX_LOCATION": "us-central1",
}


@pytest.mark.parametrize("initial", MODELS)
@pytest.mark.parametrize("role", MODELS)
@pytest.mark.parametrize(
    "change", ["none", "script", "lock", "extras", "venv", "marker"]
)
async def test_public_runtime_rechecks_daytona_proxy_install(
    monkeypatch, initial, role, change
):
    """Guards U3b-B9-C2 at d4e332b1 through the public role runtime path."""
    monkeypatch.delenv(proxy_requirements.EXTRA_REQUIREMENTS_ENV, raising=False)
    plan = plan_install_layers(agent="openhands", model=initial, agent_env=ENV)
    assert plan.proxy is not None
    sandbox = _LayerSandbox(plan.proxy, "k" * 32, "k" * 32)
    uploaded = {}

    async def upload(_sandbox, text, path, _suffix):
        uploaded[path] = text

    ensure = AsyncMock(
        side_effect=lambda _sandbox, **kw: f"{kw['venv_dir']}/bin/python"
    )
    monkeypatch.setattr(runtime_mod, "_upload_text", upload)
    monkeypatch.setattr(runtime_mod, "_ensure_sandbox_litellm", ensure)
    monkeypatch.setattr(
        runtime_mod, "_wait_for_sandbox_state", AsyncMock(return_value={"port": 45999})
    )
    monkeypatch.setattr(runtime_mod, "_poll_sandbox_health", AsyncMock())
    monkeypatch.setattr(runtime_mod.SandboxLiteLLMProcess, "stop", AsyncMock())
    _, runtime = await ensure_litellm_runtime(
        agent="openhands",
        model=initial,
        agent_env=dict(ENV),
        runtime=None,
        environment="daytona",
        session_id="initial",
        sandbox=sandbox,
    )
    assert ensure.await_count == 0
    if change == "script":
        sandbox.layer = replace(plan.proxy, script=plan.proxy.script + "\nprintf old")
    elif change == "venv":
        sandbox.layer = replace(plan.proxy, venv="/tmp/other-proxy")
    elif change == "marker":
        sandbox.image_key = "stale"
    elif change == "extras":
        monkeypatch.setenv(
            proxy_requirements.EXTRA_REQUIREMENTS_ENV, "example-addon==1.0"
        )
        # Only compare the recipe here; dependency resolution is covered in
        # test_proxy_requirements and must not contact an index for this dummy.
        monkeypatch.setattr(proxy_requirements.ProxyInstall, "check", lambda self: None)
    elif change == "lock":
        original_text = proxy_requirements.requirements_text
        monkeypatch.setattr(
            proxy_requirements,
            "requirements_text",
            lambda group: original_text(group) + "\n# new locked set\n",
        )
    uploaded.clear()
    sandbox.notes.clear()
    _, current = await ensure_litellm_runtime(
        agent="openhands",
        model=role,
        agent_env=dict(ENV),
        runtime=runtime,
        environment="daytona",
        session_id="role",
        sandbox=sandbox,
    )
    assert current is not None
    expected_reuse = change == "none" and (initial == MODELS[0]) == (role == MODELS[0])
    assert ensure.await_count == (0 if expected_reuse else 1)
    launch = json.loads(
        next(v for k, v in uploaded.items() if k.endswith("launch_config.json"))
    )
    if expected_reuse:
        assert launch["python"] == f"{PROXY_LAYER_VENV}/bin/python"
        assert sandbox.notes == [("proxy", "litellm", "used")]
    else:
        install = proxy_requirements.proxy_install(current.server.route.provider_name)
        assert any(
            k.endswith(install.file_name) and v == install.text
            for k, v in uploaded.items()
        )
        assert ensure.await_args.kwargs["requirements_path"].endswith(install.file_name)
        assert launch["python"] != f"{PROXY_LAYER_VENV}/bin/python"


async def test_proxy_lookup_without_current_script_cannot_reuse_a_layer():
    """Guards U3b-B9-C2 at d4e332b1: partial identity cannot authorize reuse."""
    from benchflow.sandbox.daytona_layers import cached_layer_in_sandbox

    plan = plan_install_layers(agent="openhands", model=MODELS[0], agent_env=ENV)
    sandbox = _LayerSandbox(plan.proxy, "k" * 32, "k" * 32)
    assert await cached_layer_in_sandbox(sandbox, "proxy", "litellm") is None
    assert sandbox.exec_calls == []

"""Agent credential files are placed only from values the credential table admits.

FF-sw-cli-other C20-C22. All values are dummies.
"""

from __future__ import annotations

import json

import pytest

from benchflow.agents import credentials as cred_mod
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.launch_env import agent_launch_env
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    CredentialFile,
    register_agent,
)
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime
from tests.test_sweep_credential_policy_matrix import SETTINGS
from tests.test_sweep_launch_env_routes import HOST_CREDENTIALS

HOME = "/home/agent"
FILE_LOGIN = "matrix-file-login"
FILE_KEY = "matrix-file-key"
FILE_SLOT = "matrix-file-slot"
ADC_TEMPLATE = "matrix-adc-template"
ADC_REFRESH = "dummy-host-adc-refresh-token"


@pytest.fixture(autouse=True)
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in (*HOST_CREDENTIALS, *SETTINGS):
        monkeypatch.delenv(name, raising=False)
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    # Multiline, as gcloud writes it: a repr() of it escapes the newlines.
    adc.write_text(json.dumps({"refresh_token": ADC_REFRESH}, indent=2))
    register_agent(
        FILE_LOGIN,
        "true",
        f"{FILE_LOGIN} --acp",
        credential_files=[
            CredentialFile("{home}/.custom/auth.json", "CUSTOM_AUTH_JSON")
        ],
    )
    register_agent(
        FILE_KEY,
        "true",
        f"{FILE_KEY} --acp",
        requires_env=["CUSTOM_API_KEY"],
        env_mapping={"BENCHFLOW_PROVIDER_API_KEY": "CUSTOM_API_KEY"},
        credential_files=[CredentialFile("{home}/.custom/key", "CUSTOM_API_KEY")],
    )
    register_agent(
        FILE_SLOT,
        "true",
        f"{FILE_SLOT} --acp",
        credential_files=[CredentialFile("{home}/.custom/key", "OPENAI_API_KEY")],
    )
    register_agent(
        ADC_TEMPLATE,
        "true",
        f"{ADC_TEMPLATE} --acp",
        credential_files=[
            CredentialFile(
                "{home}/.custom/adc.py",
                "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "ADC = {value!r}",
            )
        ],
    )
    yield
    for name in (FILE_LOGIN, FILE_KEY, FILE_SLOT, ADC_TEMPLATE):
        for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
            registry.pop(name, None)


@pytest.fixture
def uploads(monkeypatch):
    files: dict[str, str] = {}

    async def upload(sandbox, path, content, *, owner=None):
        files[path] = content

    monkeypatch.setattr(cred_mod, "upload_credential", upload)
    return files


async def _stage(agent: str, model: str | None, env: dict[str, str]) -> None:
    await cred_mod.write_credential_files(
        object(), agent, env, AGENTS[agent], model, HOME
    )


async def _wire_proxy(monkeypatch, agent: str, model: str, env: dict[str, str]):
    class _Server:
        base_url = "http://127.0.0.1:4995"
        trajectory = None

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        return _Server()

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    wired, _ = await ensure_litellm_runtime(
        agent=agent,
        agent_env=env,
        model=model,
        runtime=None,
        environment="docker",
        session_id="files",
        usage_tracking="off",
    )
    return wired


@pytest.mark.asyncio
async def test_c21_file_only_login_reaches_its_own_agent(uploads):
    """A custom agent's declared credential file is its own login."""
    login = json.dumps({"dummy": "dummy-explicit-custom-auth"})
    env = resolve_agent_env(FILE_LOGIN, None, {"CUSTOM_AUTH_JSON": login})
    await _stage(FILE_LOGIN, None, env)
    assert uploads == {f"{HOME}/.custom/auth.json": login}
    assert agent_launch_env(FILE_LOGIN, None, env)["CUSTOM_AUTH_JSON"] == login


@pytest.mark.asyncio
async def test_c25_file_only_login_in_a_known_slot_reaches_its_agent(uploads):
    """A declared file source is the agent's login on a direct route,
    even when it is a known model slot."""
    env = resolve_agent_env(FILE_SLOT, None, {"OPENAI_API_KEY": "dummy-explicit"})
    await _stage(FILE_SLOT, None, env)
    assert uploads == {f"{HOME}/.custom/key": "dummy-explicit"}
    assert agent_launch_env(FILE_SLOT, None, env)["OPENAI_API_KEY"] == "dummy-explicit"


@pytest.mark.asyncio
async def test_c25_proxy_route_still_refuses_the_upstream_key(uploads):
    """On a proxy route the same slot never reaches the agent's file."""
    model = "openai/gpt-5.5"
    env = resolve_agent_env(FILE_SLOT, model, {"OPENAI_API_KEY": "dummy-upstream"})
    await _stage(FILE_SLOT, model, env)
    assert uploads == {}


@pytest.mark.asyncio
async def test_c22_file_holding_the_proxy_key_is_written(uploads, monkeypatch):
    """connect_as wires the proxy before staging: the file holds the proxy key."""
    model = "openai/gpt-5.5"
    env = resolve_agent_env(FILE_KEY, model, {"OPENAI_API_KEY": "dummy-upstream"})
    wired = await _wire_proxy(monkeypatch, FILE_KEY, model, env)
    master = wired["BENCHFLOW_LITELLM_MASTER_KEY"]
    await _stage(FILE_KEY, model, wired)
    assert uploads == {f"{HOME}/.custom/key": master}
    assert "dummy-upstream" not in json.dumps(agent_launch_env(FILE_KEY, model, wired))


@pytest.mark.asyncio
async def test_c20_escaping_template_gets_no_unadmitted_value(uploads, monkeypatch):
    """A template that transforms the value still receives only admitted ones."""
    model = "google-vertex/gemini-2.5-pro"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p-a")
    env = resolve_agent_env(ADC_TEMPLATE, model, {})
    assert env.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    # A primary rollout stages before wiring, while the ADC is still held.
    await _stage(ADC_TEMPLATE, model, env)
    assert not any(ADC_REFRESH in content for content in uploads.values()), uploads


@pytest.mark.asyncio
async def test_c20_template_without_value_is_still_written(uploads, monkeypatch):
    """MiMo's config names the variable only; it is written on its proxy route."""
    model = "xiaomi/mimo-v2.6-flash"
    monkeypatch.setenv("XIAOMI_BASE_URL", "https://xiaomi.example.invalid/v1")
    env = resolve_agent_env("mimo", model, {"XIAOMI_API_KEY": "dummy-upstream"})
    await _stage("mimo", model, env)
    config = uploads[f"{HOME}/.config/mimocode/mimocode.json"]
    assert "{env:XIAOMI_API_KEY}" in config
    assert "dummy-upstream" not in config

"""Guards CONF-3 at 5bb84a1b: cleanup must not claim an inactive writer."""

import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import AGENTS
from tests.test_sweep_credential_lifecycle import (
    Sandbox,
    _rollout,
    host,  # noqa: F401
    no_kept_services,  # noqa: F401
)


class LocalSandbox:
    async def exec(self, command, **kwargs):
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


CASES = [
    ("mimo", {}, False),
    ("mimo", {"OPENAI_BASE_URL": "http://dummy.invalid"}, False),
    ("mimo", {"BENCHFLOW_LITELLM_MODEL_ALIAS": "dummy-model"}, False),
    (
        "mimo",
        {
            "BENCHFLOW_LITELLM_MODEL_ALIAS": "dummy-model",
            "OPENAI_BASE_URL": "http://dummy.invalid",
        },
        True,
    ),
    ("pi-acp", {}, False),
    ("pi-acp", {"BENCHFLOW_PROVIDER_PROTOCOL": "anthropic-messages"}, False),
    ("pi-acp", {"BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions"}, False),
    (
        "pi-acp",
        {
            "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://dummy.invalid",
        },
        True,
    ),
    ("codex-acp", {}, False),
    ("codex-acp", {"OPENAI_API_KEY": "dummy-login"}, True),
    ("opencode", {}, False),
    ("opencode", {"BENCHFLOW_LITELLM_MODEL_ALIAS": " "}, False),
    ("opencode", {"BENCHFLOW_LITELLM_MODEL_ALIAS": "dummy-model"}, True),
    ("opencode", {"BENCHFLOW_OPENCODE_GOOGLE_BASE_URL": "http://dummy.invalid"}, True),
    ("openhands", {}, True),
]


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["builtin", "acpx"])
@pytest.mark.parametrize("agent,values,writes", CASES)
async def test_only_active_launcher_destinations_enter_ledger(
    tmp_path, monkeypatch, backend, wrapped, agent, values, writes
):
    """5bb84a1b records all destinations, even when launch writes nothing."""
    trial = _rollout(tmp_path, Sandbox(monkeypatch), "oracle", None, {})
    trial._config.environment = backend
    name = f"acpx:{agent}" if wrapped else agent
    # Production registers the wrapped config before resolving its login.
    # An absent entry is a raw command and cannot admit this built-in's key.
    monkeypatch.setitem(AGENTS, name, replace(AGENTS[agent], name=name))
    await trial._stage_connection_credentials(
        name,
        dict(values),
        AGENTS[agent],
        None,
        key=(agent, None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    expected = [
        p.format(home="/home/agent", workspace="/app")
        for p in AGENTS[agent].launcher_credential_files
    ]
    assert set(trial._credential_ledger) == (set(expected) if writes else set())


async def test_direct_mimo_preserves_completed_workspace_output(tmp_path, monkeypatch):
    """5bb84a1b deletes a task's MiMo output after a native-login run."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env = LocalSandbox()
    trial._config.sandbox_user = None
    trial._agent_cwd = str(tmp_path)
    # Isolate the launcher's workspace path; native login is tracked by its
    # separate uploader, whose behavior is covered by the lifecycle suite.
    config = SimpleNamespace(
        credential_files=[],
        launcher_credential_files=[
            p for p in AGENTS["mimo"].launcher_credential_files if "{workspace}" in p
        ],
    )
    await trial._stage_connection_credentials(
        "mimo",
        {},
        config,
        None,
        key=("mimo", None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    output = tmp_path / "mimocode.json"
    output.write_text('{"tools":{"webfetch":false}}')
    from benchflow.agents.credentials import remove_staged_credentials

    await remove_staged_credentials(trial._env, None, trial._credential_ledger)
    assert output.read_text() == '{"tools":{"webfetch":false}}'

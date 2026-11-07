"""Guards FF-sw-cli-other-C2 against incomplete normalization at 3821a923."""

import pytest
from typer.testing import CliRunner

from benchflow.agents.env import inherit_agent_auth_env, resolve_agent_env
from benchflow.cli.main import app
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task


@pytest.fixture(autouse=True)
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for name in (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "dummy-login")


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
@pytest.mark.parametrize("source", ["host", "dotenv"])
@pytest.mark.parametrize("model", ["gpt-5.5", "openai/gpt-5.5"])
def test_check_drops_inherited_endpoint(tmp_path, monkeypatch, command, source, model):
    """Guards 3821a923: check must use run's native route under an allowlist."""
    endpoint = "https://proxy.example.invalid/v1"
    if source == "host":
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", endpoint)
    else:
        (tmp_path / "dummy.env").write_text(f"BENCHFLOW_PROVIDER_BASE_URL={endpoint}\n")
    task = _task(
        tmp_path / "task",
        '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n',
    )
    result = CliRunner().invoke(
        app, [*command, str(task), "--agent", "codex", "--model", model]
    )
    assert result.exit_code == 1, result.output
    assert "native subscription endpoint" in result.stderr


@pytest.mark.parametrize("agent", ["codex-acp", "openhands"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "key",
    [
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    ],
)
def test_route_normalization_preserves_explicit_overrides(
    monkeypatch, agent, explicit, key
):
    """Guards 3821a923: inspection and execution share endpoint normalization."""
    endpoint = "https://proxy.example.invalid/v1"
    monkeypatch.setenv(key, endpoint)
    overlay = {key: endpoint} if explicit else {}
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    planned = inherit_agent_auth_env(agent, "openai/gpt-5.5", overlay)
    actual = resolve_agent_env(agent, "openai/gpt-5.5", overlay)
    if explicit:
        assert planned[key] == endpoint
        if key.endswith("BASE_URL"):
            assert actual[key] == endpoint
    else:
        assert key not in planned
        assert actual.get(key) != endpoint


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("login", ["CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON"])
def test_shared_gate_uses_normalized_native_route(
    tmp_path, monkeypatch, backend, login
):
    """Guards 3821a923 across backends and both subscription credential forms."""
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv(login, "dummy-login")
    monkeypatch.setenv(
        "BENCHFLOW_PROVIDER_BASE_URL", "https://proxy.example.invalid/v1"
    )
    task = _task(
        tmp_path / "task",
        '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n',
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match="not enforced by modal"
        if backend == "modal"
        else "native subscription endpoint",
    ):
        validate_task_plan(
            task, sandbox=backend, agent="codex-acp", model="openai/gpt-5.5"
        )


def test_explicit_openhands_bridge_is_shared():
    """Guards 3821a923: the explicit bridge affects the plan as well as run."""
    overlay = {
        "LLM_BASE_URL": "https://proxy.example.invalid/v1",
        "LLM_API_KEY": "dummy-key",
    }
    planned = inherit_agent_auth_env("openhands", "openai/gpt-5.5", overlay)
    actual = resolve_agent_env("openhands", "openai/gpt-5.5", overlay)
    assert (
        planned["BENCHFLOW_PROVIDER_BASE_URL"]
        == actual["BENCHFLOW_PROVIDER_BASE_URL"]
        == overlay["LLM_BASE_URL"]
    )
    assert (
        planned["BENCHFLOW_PROVIDER_API_KEY"]
        == actual["BENCHFLOW_PROVIDER_API_KEY"]
        == overlay["LLM_API_KEY"]
    )

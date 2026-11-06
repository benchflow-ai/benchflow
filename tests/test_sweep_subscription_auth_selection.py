"""Guards FF-sw-cli-other-C6, introduced by 3df70b5e (reviewed at 81280d8f)."""

from itertools import combinations
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.agents import env as auth
from benchflow.agents.registry import resolve_agent
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task
from tests.test_connect_as_env import TestConnectAsEnvMerge as _ConnectAsFixtures

CLAUDE_TOKENS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
)
CODEX_CREDENTIALS = (
    "CODEX_ACCESS_TOKEN",
    "CODEX_AUTH_JSON",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
)
CLAUDE_MODELS = (None, "claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


@pytest.fixture(autouse=True)
def host_logins(tmp_path, monkeypatch):
    """Keep unrelated account B logins available, using dummy contents only."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for key in (
        *CLAUDE_TOKENS,
        *CODEX_CREDENTIALS,
        "ANTHROPIC_API_KEY",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_NAME",
        "OPENAI_BASE_URL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    for folder, filename in (
        (".claude", ".credentials.json"),
        ("relocated-claude", ".credentials.json"),
        (".codex", "auth.json"),
    ):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / filename).write_text('{"dummy_account": "unrelated-B"}')


def _inputs(tmp_path, monkeypatch, source, values):
    if source == "host":
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return {}
    if source == "dotenv":
        (tmp_path / "dummy.env").write_text(
            "".join(f"{key}='{value}'\n" for key, value in values.items())
        )
        return {}
    return values


@pytest.mark.parametrize("agent", ["claude", "claude-agent-acp"])
@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize("token", CLAUDE_TOKENS)
@pytest.mark.parametrize("source", ["explicit", "host", "dotenv"])
@pytest.mark.parametrize("relocated", [False, True])
def test_claude_token_never_selects_another_host_login(
    tmp_path, monkeypatch, agent, model, token, source, relocated
):
    """Guards 3df70b5e: every native credential suppresses host-file discovery."""
    if relocated:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "relocated-claude"))
    probe = Mock(wraps=auth.check_subscription_auth)
    monkeypatch.setattr(auth, "check_subscription_auth", probe)
    supplied = _inputs(tmp_path, monkeypatch, source, {token: "dummy-account-A"})
    canonical = resolve_agent(agent).name
    result = auth.resolve_agent_env(canonical, model, supplied)
    assert result[token] == "dummy-account-A"
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result
    assert "BENCHFLOW_PROVIDER_API_KEY" not in result
    assert "BENCHFLOW_PROVIDER_BASE_URL" not in result
    assert auth.uses_native_subscription_auth(canonical, model, result)
    probe.assert_not_called()


@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize(
    "tokens", [c for size in (1, 2, 3) for c in combinations(CLAUDE_TOKENS, size)]
)
def test_any_nonempty_token_suppresses_host_login(model, tokens):
    """Guards 3df70b5e: blank aliases cannot hide another nonempty credential."""
    values = {key: "dummy-account-A" if key in tokens else "" for key in CLAUDE_TOKENS}
    result = auth.resolve_agent_env("claude-agent-acp", model, values)
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result


@pytest.mark.parametrize("model", [None, "gpt-5.5", "openai/gpt-5.5"])
@pytest.mark.parametrize("credential", CODEX_CREDENTIALS)
@pytest.mark.parametrize("source", ["explicit", "host", "dotenv"])
def test_codex_credentials_also_suppress_host_fallback(
    tmp_path, monkeypatch, model, credential, source
):
    """Guards the 3df70b5e sibling: inline auth and API aliases keep precedence."""
    value = (
        '{"tokens":{"access_token":"dummy-account-A"}}'
        if credential == "CODEX_AUTH_JSON"
        else "dummy-account-A"
    )
    supplied = _inputs(tmp_path, monkeypatch, source, {credential: value})
    probe = Mock(wraps=auth.check_subscription_auth)
    monkeypatch.setattr(auth, "check_subscription_auth", probe)
    result = auth.resolve_agent_env("codex-acp", model, supplied)
    assert result[credential] == value
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result
    probe.assert_not_called()


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_empty_credentials_still_allow_host_fallback(model):
    """Guards 3df70b5e: a usable host login remains the last-resort credential."""
    result = auth.resolve_agent_env(
        "claude-agent-acp", model, {key: "" for key in CLAUDE_TOKENS}
    )
    assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_other_family_credential_does_not_hide_required_host_login(model):
    """Guards 3df70b5e: selection is scoped to the required credential family."""
    result = auth.resolve_agent_env(
        "claude-agent-acp", model, {"CODEX_ACCESS_TOKEN": "dummy-other-family"}
    )
    assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"


@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize("token", CLAUDE_TOKENS)
def test_reresolution_discards_the_previous_host_selection(model, token):
    """Guards 3df70b5e: a previous account's derived marker is not authorization."""
    previous = auth.resolve_agent_env("claude-agent-acp", model, {})
    assert previous["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"
    result = auth.resolve_agent_env(
        "claude-agent-acp", model, {**previous, token: "dummy-account-A"}
    )
    assert result[token] == "dummy-account-A"
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("model", CLAUDE_MODELS[1:])
@pytest.mark.parametrize("token", CLAUDE_TOKENS)
def test_public_network_backend_plan_keeps_only_the_supplied_account(
    tmp_path, backend, model, token
):
    """Guards 3df70b5e on each backend's ordinary public-network task path."""
    task = _task(tmp_path / "task")
    result = auth.resolve_agent_env(
        "claude-agent-acp", model, {token: "dummy-account-A"}
    )
    validate_task_plan(
        task, sandbox=backend, agent="claude-agent-acp", model=model, agent_env=result
    )
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result


class TestRoleHostLoginSelection:
    _mock_trial = _ConnectAsFixtures._mock_trial

    @pytest.mark.parametrize("token", CLAUDE_TOKENS)
    @pytest.mark.parametrize("model", CLAUDE_MODELS)
    async def test_role_token_does_not_upload_host_login(
        self, _mock_trial, token, model
    ):
        """Guards 3df70b5e through the real secondary-role credential staging path."""
        from benchflow.rollout import Role

        trial = _mock_trial
        trial._config.agent_env = {}
        trial._planes.resolve_agent_env.side_effect = auth.resolve_agent_env
        trial._planes.write_credential_files = AsyncMock()
        role = Role(
            name="reviewer",
            agent="claude-agent-acp",
            model=model,
            env={token: "dummy-account-A"},
        )
        await trial.connect_as(role)
        trial._planes.connect_acp.assert_awaited_once()
        trial._planes.upload_subscription_auth.assert_not_awaited()
        staged = trial._planes.write_credential_files.await_args.args[2]
        assert staged[token] == "dummy-account-A"
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in staged

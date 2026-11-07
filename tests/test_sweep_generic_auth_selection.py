"""Guards FF-sw-cli-other-C7, incomplete at d3647cbf after 3df70b5e."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.agents import env as auth
from benchflow.agents.registry import AGENTS, resolve_agent
from benchflow.errors import MissingCredentialError
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task
from tests.test_connect_as_env import TestConnectAsEnvMerge as _ConnectAsFixtures

FAMILIES = (
    ("claude", "claude-sonnet-4-6", "anthropic", "https://api.anthropic.com"),
    ("claude-agent-acp", "claude-sonnet-4-6", "anthropic", "https://api.anthropic.com"),
    ("codex", "gpt-5.5", "openai", "https://api.openai.com/v1"),
    ("codex-acp", "gpt-5.5", "openai", "https://api.openai.com/v1"),
)
GENERIC_KEY = "BENCHFLOW_PROVIDER_API_KEY"
MARKER = "_BENCHFLOW_SUBSCRIPTION_AUTH"


@pytest.fixture(autouse=True)
def host_logins(tmp_path, monkeypatch):
    """Account B is available on the host; every credential here is a dummy."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "CODEX_AUTH_JSON",
        "OPENAI_API_KEY",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        GENERIC_KEY,
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_NAME",
        "OPENAI_BASE_URL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    for folder, filename in ((".claude", ".credentials.json"), (".codex", "auth.json")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / filename).write_text('{"dummy_account":"unrelated-B"}')


def _model(family, qualified):
    return f"{family[2]}/{family[1]}" if qualified else family[1]


def _env(family, endpoint):
    values = {GENERIC_KEY: "dummy-account-A"}
    if endpoint != "absent":
        values["BENCHFLOW_PROVIDER_BASE_URL"] = (
            family[3] if endpoint == "canonical" else "https://proxy.example.invalid/v1"
        )
    return values


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("endpoint", ["absent", "canonical", "custom"])
@pytest.mark.parametrize("carried_marker", [False, True])
def test_generic_key_selects_provider_before_host_discovery(
    monkeypatch, family, qualified, endpoint, carried_marker
):
    """Guards 3df70b5e: check and runtime must both select supplied account A."""
    agent = resolve_agent(family[0]).name
    model = _model(family, qualified)
    supplied = _env(family, endpoint)
    if carried_marker:
        supplied[MARKER] = "1"
    probe = Mock(wraps=auth.check_subscription_auth)
    monkeypatch.setattr(auth, "check_subscription_auth", probe)
    planned = auth.inherit_agent_auth_env(agent, model, supplied)
    assert not auth.uses_native_subscription_auth(agent, model, planned)
    actual = auth.resolve_agent_env(agent, model, supplied)
    assert actual[GENERIC_KEY] == "dummy-account-A"
    assert actual[AGENTS[agent].env_mapping[GENERIC_KEY]] == "dummy-account-A"
    assert MARKER not in actual
    assert not auth.uses_native_subscription_auth(agent, model, actual)
    probe.assert_not_called()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("qualified", [False, True])
def test_generic_key_never_authorizes_host_file_staging(family, qualified):
    """Guards 3df70b5e separately from the pre-existing plan-route defect."""
    agent = resolve_agent(family[0]).name
    actual = auth.resolve_agent_env(
        agent, _model(family, qualified), _env(family, "canonical")
    )
    assert MARKER not in actual
    assert actual[AGENTS[agent].env_mapping[GENERIC_KEY]] == "dummy-account-A"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("endpoint", ["absent", "canonical", "custom"])
@pytest.mark.parametrize("source", ["explicit", "host", "dotenv"])
def test_generic_key_without_model_refuses_instead_of_selecting_host(
    tmp_path, monkeypatch, family, endpoint, source
):
    """Guards the d3647cbf sibling: no model must fail before host selection."""
    agent = resolve_agent(family[0]).name
    supplied = _env(family, endpoint)
    if source == "host":
        for key, value in supplied.items():
            monkeypatch.setenv(key, value)
        supplied = {}
    elif source == "dotenv":
        (tmp_path / "dummy.env").write_text(
            "".join(f"{k}={v}\n" for k, v in supplied.items())
        )
        supplied = {}
    with pytest.raises(
        MissingCredentialError, match=r"model.*BENCHFLOW_PROVIDER_API_KEY"
    ):
        auth.resolve_agent_env(agent, None, supplied)


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize(
    "credential", ["CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON"]
)
def test_generic_key_precedes_supplied_codex_credentials(qualified, credential):
    """Guards d3647cbf: native aliases and carried markers cannot change the route."""
    supplied = {
        **_env(FAMILIES[2], "canonical"),
        credential: "dummy-native",
        MARKER: "1",
    }
    model = _model(FAMILIES[2], qualified)
    actual = auth.resolve_agent_env("codex-acp", model, supplied)
    assert actual[GENERIC_KEY] == "dummy-account-A"
    assert MARKER not in actual
    assert not auth.uses_native_subscription_auth("codex-acp", model, actual)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
def test_backend_public_plan_and_runtime_select_same_account(
    tmp_path, family, qualified, backend
):
    """Guards d3647cbf through each backend's public-network validation path."""
    agent = resolve_agent(family[0]).name
    model = _model(family, qualified)
    supplied = _env(family, "canonical")
    actual = auth.resolve_agent_env(agent, model, supplied)
    task = _task(tmp_path / "task")
    for values in (supplied, actual):
        validate_task_plan(
            task, sandbox=backend, agent=agent, model=model, agent_env=values
        )
        assert not auth.uses_native_subscription_auth(agent, model, values)
        assert MARKER not in values


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("source", ["host", "dotenv"])
@pytest.mark.parametrize("qualified", [False, True])
def test_unselected_inherited_generic_key_does_not_change_host_fallback(
    tmp_path, monkeypatch, family, source, qualified
):
    """Guards d3647cbf: normalization still discards unrelated host defaults."""
    values = _env(family, "custom")
    if source == "host":
        for key, value in values.items():
            monkeypatch.setenv(key, value)
    else:
        (tmp_path / "dummy.env").write_text(
            "".join(f"{k}={v}\n" for k, v in values.items())
        )
    agent = resolve_agent(family[0]).name
    model = _model(family, qualified)
    actual = auth.resolve_agent_env(agent, model, {})
    assert GENERIC_KEY not in actual
    assert actual[MARKER] == "1"
    assert auth.uses_native_subscription_auth(agent, model, actual)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("credential", ["absent", "blank"])
def test_host_only_control_keeps_native_route(family, credential):
    """Guards d3647cbf: a missing generic key still permits legitimate login."""
    agent = resolve_agent(family[0]).name
    supplied = {} if credential == "absent" else {GENERIC_KEY: ""}
    actual = auth.resolve_agent_env(agent, family[1], supplied)
    assert actual[MARKER] == "1"
    assert auth.uses_native_subscription_auth(agent, family[1], actual)


@pytest.mark.parametrize(
    "agent", [name for name, cfg in AGENTS.items() if cfg.subscription_auth]
)
def test_generic_key_blocks_every_registered_host_file_fallback(agent):
    """Guards d3647cbf: host-file authorization is registry-wide, not two names."""
    cfg = AGENTS[agent]
    assert cfg.subscription_auth is not None
    assert not auth._can_use_subscription_auth(
        agent,
        None,
        cfg.subscription_auth.replaces_env,
        {GENERIC_KEY: "dummy-account-A"},
    )


class TestRoleGenericSelection:
    _mock_trial = _ConnectAsFixtures._mock_trial

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("qualified", [False, True])
    async def test_generic_role_does_not_upload_another_account(
        self, _mock_trial, family, qualified, monkeypatch
    ):
        """Guards 3df70b5e through real connect_as credential staging."""
        from benchflow.rollout import Role

        trial = _mock_trial
        # C33 prepares Codex's inline-provider config before the staging boundary.
        trial._env = SimpleNamespace(
            exec=AsyncMock(return_value=SimpleNamespace(return_code=0))
        )
        trial._config.agent_env = {}
        trial._planes.resolve_agent_env.side_effect = auth.resolve_agent_env
        agent = resolve_agent(family[0]).name

        # End at the credential staging boundary (after Codex Apps admission,
        # which this stub cannot verify without an installed client).
        class CredentialsStaged(Exception):
            pass

        async def admitted(env, **kwargs):
            return kwargs["agent_env"]

        monkeypatch.setattr("benchflow.rollout.enforce_codex_apps_policy", admitted)
        trial._planes.write_credential_files = AsyncMock(side_effect=CredentialsStaged)
        with pytest.raises(CredentialsStaged):
            await trial.connect_as(
                Role(
                    name="reviewer",
                    agent=agent,
                    model=_model(family, qualified),
                    env=_env(family, "canonical"),
                )
            )
        trial._planes.connect_acp.assert_not_awaited()
        trial._planes.upload_subscription_auth.assert_not_awaited()
        staged = trial._planes.write_credential_files.await_args.args[2]
        assert staged[AGENTS[agent].env_mapping[GENERIC_KEY]] == "dummy-account-A"
        assert MARKER not in staged

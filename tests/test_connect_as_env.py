"""Regression tests for connect_as() agent_env merging (issue #2).

connect_as() must merge cfg.agent_env (config-level) with role.env
(role-level), with role-level keys winning on overlap.  Before the fix,
role.env={} was truthy so resolve_agent_env received an empty dict,
discarding all config-level env vars.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Role, RolloutConfig, Scene


def _make_config(agent_env=None, role_env=None):
    """Build a minimal RolloutConfig with one scene."""
    role = Role(name="agent", agent="claude-agent-acp", model="test-model")
    if role_env is not None:
        role = Role(
            name="agent", agent="claude-agent-acp", model="test-model", env=role_env
        )
    scene = Scene(roles=[role])
    return RolloutConfig(
        task_path=Path("/fake/task"),
        scenes=[scene],
        agent_env=agent_env,
    )


class TestConnectAsEnvMerge:
    """Verify connect_as() merges cfg.agent_env with role.env correctly."""

    @pytest.fixture()
    def _mock_trial(self, tmp_path):
        """Return a Rollout stub wired to capture the agent_env passed to connect_acp."""
        from benchflow.rollout import Rollout

        cfg = _make_config(
            agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8080/v1"},
        )
        trial = Rollout.__new__(Rollout)
        trial._config = cfg
        trial._env = {}
        trial._rollout_dir = tmp_path
        trial._timing = {}
        trial._agent_cwd = None
        trial._agent_cfg = MagicMock(credential_files=[])
        trial._phase = "idle"
        planes = MagicMock()
        planes.agent_launch.return_value = "claude-agent-acp"
        planes.resolve_agent_env.side_effect = lambda _agent, _model, env: env or {}
        planes.ensure_litellm_runtime = AsyncMock(
            side_effect=lambda **kwargs: (kwargs["agent_env"], None)
        )
        planes.install_agent = AsyncMock()
        planes.write_credential_files = AsyncMock()
        planes.upload_subscription_auth = AsyncMock()
        planes.apply_web_tool_policy = AsyncMock()
        planes.connect_acp = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), AsyncMock(), "agent")
        )
        trial._planes = planes
        return trial

    @pytest.mark.asyncio
    async def test_config_env_propagated_through_empty_role_env(self, _mock_trial):
        """cfg.agent_env vars reach resolve_agent_env when role.env is {}."""
        captured = {}

        def fake_resolve(agent, model, env):
            captured["env"] = env
            return env or {}

        _mock_trial._planes.resolve_agent_env.side_effect = fake_resolve
        role = _mock_trial._config.scenes[0].roles[0]
        await _mock_trial.connect_as(role)

        assert "BENCHFLOW_PROVIDER_BASE_URL" in captured["env"]
        assert (
            captured["env"]["BENCHFLOW_PROVIDER_BASE_URL"] == "http://localhost:8080/v1"
        )

    @pytest.mark.asyncio
    async def test_role_env_overrides_config_env(self, _mock_trial):
        """Role-level env wins over config-level on key overlap."""
        _mock_trial._config = _make_config(
            agent_env={"KEY": "from-config", "SHARED": "config-val"},
            role_env={"SHARED": "role-val", "ROLE_ONLY": "yes"},
        )
        captured = {}

        def fake_resolve(agent, model, env):
            captured["env"] = env
            return env or {}

        _mock_trial._planes.resolve_agent_env.side_effect = fake_resolve
        role = _mock_trial._config.scenes[0].roles[0]
        await _mock_trial.connect_as(role)

        env = captured["env"]
        assert env["KEY"] == "from-config"
        assert env["SHARED"] == "role-val"
        assert env["ROLE_ONLY"] == "yes"

    @pytest.mark.asyncio
    async def test_all_keys_present_in_merge(self, _mock_trial):
        """Non-overlapping keys from both dicts are all present."""
        _mock_trial._config = _make_config(
            agent_env={"A": "1", "B": "2"},
            role_env={"C": "3", "D": "4"},
        )
        captured = {}

        def fake_resolve(agent, model, env):
            captured["env"] = env
            return env or {}

        _mock_trial._planes.resolve_agent_env.side_effect = fake_resolve
        role = _mock_trial._config.scenes[0].roles[0]
        await _mock_trial.connect_as(role)

        env = captured["env"]
        assert env == {"A": "1", "B": "2", "C": "3", "D": "4"}

    @pytest.mark.asyncio
    async def test_none_config_env_with_empty_role_env(self, _mock_trial):
        """cfg.agent_env=None + empty role.env does not crash."""
        _mock_trial._config = _make_config(agent_env=None, role_env={})
        captured = {}

        def fake_resolve(agent, model, env):
            captured["env"] = env
            return env or {}

        _mock_trial._planes.resolve_agent_env.side_effect = fake_resolve
        role = _mock_trial._config.scenes[0].roles[0]
        await _mock_trial.connect_as(role)

        assert captured["env"] == {}

    @pytest.mark.asyncio
    async def test_lockdown_proxy_env_reaches_agent(self, _mock_trial):
        """Egress proxy env from network lockdown must reach the agent launched
        via connect_as. Under a restrictive docker policy the container is on an
        internal-only network and reaches the allowlisted provider only through
        the bf-egress sidecar (HTTPS_PROXY). The scene/role path rebuilds
        agent_env from config, so it must re-merge the lockdown proxy env or the
        agent's LLM client gets "Connection error". Regression for that path."""
        proxy = {
            "HTTP_PROXY": "http://bf-egress:8080",
            "HTTPS_PROXY": "http://bf-egress:8080",
            "http_proxy": "http://bf-egress:8080",
            "https_proxy": "http://bf-egress:8080",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }
        _mock_trial._lockdown_proxy_env = dict(proxy)
        role = _mock_trial._config.scenes[0].roles[0]
        await _mock_trial.connect_as(role)

        sent = _mock_trial._planes.connect_acp.await_args.kwargs["agent_env"]
        assert sent.get("HTTPS_PROXY") == "http://bf-egress:8080"
        assert sent.get("https_proxy") == "http://bf-egress:8080"
        assert sent.get("NO_PROXY") == "localhost,127.0.0.1"

    @pytest.mark.asyncio
    async def test_same_agent_different_model_refreshes_credentials(self, _mock_trial):
        """Guards ENG-91 P0 same-agent role credential refresh regression."""
        from benchflow.rollout import Role

        primary = Role(name="primary", agent="claude-agent-acp", model="test-model")
        role = Role(name="reviewer", agent="claude-agent-acp", model="other-model")
        _mock_trial._config.scenes[0].roles = [primary, role]
        _mock_trial._config.agent_env = {"ANTHROPIC_API_KEY": "from-config"}

        _mock_trial._planes.resolve_agent_env.side_effect = None
        _mock_trial._planes.resolve_agent_env.return_value = {
            "ANTHROPIC_API_KEY": "from-config"
        }
        _mock_trial._planes.ensure_litellm_runtime.return_value = (
            {"ANTHROPIC_API_KEY": "from-config"},
            None,
        )

        await _mock_trial.connect_as(role)

        _mock_trial._planes.install_agent.assert_not_awaited()
        _mock_trial._planes.write_credential_files.assert_awaited_once()
        args, _kwargs = _mock_trial._planes.write_credential_files.await_args
        assert args[1] == "claude-agent-acp"
        assert args[4] == "other-model"

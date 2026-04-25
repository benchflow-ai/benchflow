"""Regression tests for connect_as() agent_env merging (issue #2).

connect_as() must merge cfg.agent_env (config-level) with role.env
(role-level), with role-level keys winning on overlap.  Before the fix,
role.env={} was truthy so resolve_agent_env received an empty dict,
discarding all config-level env vars.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.trial import Role, Scene, TrialConfig


def _make_config(agent_env=None, role_env=None):
    """Build a minimal TrialConfig with one scene."""
    role = Role(name="agent", agent="claude-agent-acp", model="test-model")
    if role_env is not None:
        role = Role(
            name="agent", agent="claude-agent-acp", model="test-model", env=role_env
        )
    scene = Scene(roles=[role])
    return TrialConfig(
        task_path=Path("/fake/task"),
        scenes=[scene],
        agent_env=agent_env,
    )


class TestConnectAsEnvMerge:
    """Verify connect_as() merges cfg.agent_env with role.env correctly."""

    @pytest.fixture()
    def _mock_trial(self, tmp_path):
        """Return a Trial stub wired to capture the agent_env passed to connect_acp."""
        from benchflow.trial import Trial

        cfg = _make_config(
            agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8080/v1"},
        )
        trial = Trial.__new__(Trial)
        trial._config = cfg
        trial._env = {}
        trial._trial_dir = tmp_path
        trial._timing = {}
        trial._agent_cwd = None
        trial._phase = "idle"
        return trial

    @pytest.mark.asyncio
    async def test_config_env_propagated_through_empty_role_env(self, _mock_trial):
        """cfg.agent_env vars reach resolve_agent_env when role.env is {}."""
        captured = {}

        def fake_resolve(agent, model, env):
            captured["env"] = env
            return env or {}

        with (
            patch("benchflow.trial.resolve_agent_env", side_effect=fake_resolve),
            patch("benchflow.trial.connect_acp", new_callable=AsyncMock) as mock_conn,
        ):
            mock_conn.return_value = (AsyncMock(), AsyncMock(), "agent")
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

        with (
            patch("benchflow.trial.resolve_agent_env", side_effect=fake_resolve),
            patch("benchflow.trial.connect_acp", new_callable=AsyncMock) as mock_conn,
        ):
            mock_conn.return_value = (AsyncMock(), AsyncMock(), "agent")
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

        with (
            patch("benchflow.trial.resolve_agent_env", side_effect=fake_resolve),
            patch("benchflow.trial.connect_acp", new_callable=AsyncMock) as mock_conn,
        ):
            mock_conn.return_value = (AsyncMock(), AsyncMock(), "agent")
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

        with (
            patch("benchflow.trial.resolve_agent_env", side_effect=fake_resolve),
            patch("benchflow.trial.connect_acp", new_callable=AsyncMock) as mock_conn,
        ):
            mock_conn.return_value = (AsyncMock(), AsyncMock(), "agent")
            role = _mock_trial._config.scenes[0].roles[0]
            await _mock_trial.connect_as(role)

        assert captured["env"] == {}

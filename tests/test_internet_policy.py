from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchflow.trial import (
    Role,
    Scene,
    Trial,
    TrialConfig,
    _apply_web_policy,
    _skill_nudge,
    _task_disallows_internet,
)


def test_task_disallows_internet_from_environment_config():
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=False))
    )

    assert _task_disallows_internet(task) is True


def test_task_allows_internet_by_default_for_missing_config():
    assert _task_disallows_internet(None) is False
    assert _task_disallows_internet(SimpleNamespace()) is False


def test_apply_web_policy_sets_marker_without_mutating_input():
    env = {"LLM_API_KEY": "secret"}

    result = _apply_web_policy(env, disallow=True)

    assert result["BENCHFLOW_DISALLOW_WEB_TOOLS"] == "1"
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in env


def test_skill_nudge_prefers_explicit_agent_env(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_SKILL_NUDGE", "name")

    assert _skill_nudge({"BENCHFLOW_SKILL_NUDGE": "description"}) == "description"


def test_skill_nudge_falls_back_to_host_env(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_SKILL_NUDGE", "name")

    assert _skill_nudge({}) == "name"


def test_create_environment_preserves_agent_network_for_llm_runs(tmp_path):
    from benchflow._env_setup import _create_environment

    original_env = MagicMock()
    original_env.allow_internet = False
    copied_env = MagicMock()
    copied_env.allow_internet = False
    original_env.model_copy.return_value = copied_env
    task = SimpleNamespace(
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(environment=original_env),
    )

    with patch("harbor.environments.docker.docker.DockerEnvironment") as docker_env:
        _create_environment(
            "docker",
            task,
            tmp_path,
            "trial",
            MagicMock(),
            preserve_agent_network=True,
        )

    assert copied_env.allow_internet is True
    assert original_env.allow_internet is False
    assert docker_env.call_args.kwargs["task_env_config"] is copied_env


def test_create_environment_keeps_oracle_network_policy(tmp_path):
    from benchflow._env_setup import _create_environment

    original_env = MagicMock()
    original_env.allow_internet = False
    task = SimpleNamespace(
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(environment=original_env),
    )

    with patch("harbor.environments.docker.docker.DockerEnvironment") as docker_env:
        _create_environment("docker", task, tmp_path, "trial", MagicMock())

    original_env.model_copy.assert_not_called()
    assert docker_env.call_args.kwargs["task_env_config"] is original_env


@pytest.mark.asyncio
async def test_connect_as_applies_web_policy_to_role_env(tmp_path):
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        scenes=[
            Scene(
                roles=[Role(name="agent", agent="claude-agent-acp", model="test-model")]
            )
        ],
        agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8080/v1"},
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = {}
    trial._trial_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=False))
    )
    captured = {}

    def fake_resolve(agent, model, env):
        captured["env"] = env
        return env or {}

    with (
        patch("benchflow.trial.resolve_agent_env", side_effect=fake_resolve),
        patch("benchflow.trial.connect_acp") as connect_acp,
    ):
        connect_acp.return_value = (MagicMock(), MagicMock(), "agent")
        await trial.connect_as(cfg.scenes[0].roles[0])

    assert captured["env"]["BENCHFLOW_PROVIDER_BASE_URL"] == "http://localhost:8080/v1"
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in captured["env"]
    assert (
        connect_acp.call_args.kwargs["agent_env"]["BENCHFLOW_DISALLOW_WEB_TOOLS"] == "1"
    )


def test_openhands_launch_disables_browsing_when_policy_marker_is_set():
    from benchflow.agents.registry import AGENT_LAUNCH

    launch = AGENT_LAUNCH["openhands"]

    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" in launch
    assert "enable_browsing = false" in launch

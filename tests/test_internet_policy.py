from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.trial import (
    Role,
    Scene,
    Trial,
    TrialConfig,
    _agent_launch_with_web_policy,
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


def test_host_skill_nudge_is_only_prompt_mutation(monkeypatch, tmp_path):
    from benchflow.sdk import SDK

    monkeypatch.setenv("BENCHFLOW_SKILL_NUDGE", "name")
    (tmp_path / "instruction.md").write_text("Do the thing.")
    skill_dir = tmp_path / "environment" / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill.\n---\n# Alpha\n"
    )

    prompts = SDK._resolve_prompts(
        tmp_path,
        prompts=None,
        skill_nudge=_skill_nudge({}),
        agent="claude-agent-acp",
    )

    assert prompts[0].startswith("Skills available at ~/.claude/skills: alpha.")
    assert prompts[0].endswith("Do the thing.")
    assert "Internet access is disabled" not in prompts[0]
    assert "Do not browse" not in prompts[0]


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


def test_codex_launch_disable_is_gated_by_web_policy():
    assert _agent_launch_with_web_policy("codex-acp", disallow=False) == "codex-acp"
    assert _agent_launch_with_web_policy("codex-acp", disallow=True) == (
        "codex-acp -c tools.web_search=false"
    )


def test_agent_registry_has_supported_hard_web_disable_snippets():
    from benchflow.agents.registry import AGENT_LAUNCH, AGENTS

    assert "enable_browsing = false" in AGENTS["openhands"].disallow_web_tools_setup_cmd
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in AGENT_LAUNCH["openhands"]

    claude_cmd = AGENTS["claude-agent-acp"].disallow_web_tools_setup_cmd
    assert "WebSearch" in claude_cmd
    assert "WebFetch" in claude_cmd
    assert "permissions" in claude_cmd

    gemini_cmd = AGENTS["gemini"].disallow_web_tools_setup_cmd
    assert "google_web_search" in gemini_cmd
    assert "web_fetch" in gemini_cmd

    opencode_cmd = AGENTS["opencode"].disallow_web_tools_setup_cmd
    assert "webfetch" in opencode_cmd


@pytest.mark.asyncio
async def test_connect_as_applies_hard_web_policy_to_role_agent(tmp_path):
    from benchflow.agents.registry import AGENTS

    role = Role(name="coder", agent="gemini", model="gemini/test")
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        scenes=[
            Scene(
                roles=[
                    Role(name="primary", agent="claude-agent-acp", model="test-model"),
                    role,
                ]
            )
        ],
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = {}
    trial._trial_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._disallow_web_tools = True

    with (
        patch("benchflow.trial.resolve_agent_env", return_value={}),
        patch(
            "benchflow.trial.install_agent",
            new=AsyncMock(return_value=AGENTS["gemini"]),
        ),
        patch("benchflow.trial.write_credential_files", new=AsyncMock()),
        patch("benchflow.trial.upload_subscription_auth", new=AsyncMock()),
        patch("benchflow.trial.apply_web_tool_policy", new=AsyncMock()) as apply_policy,
        patch("benchflow.trial.connect_acp", new=AsyncMock()) as connect_acp,
    ):
        connect_acp.return_value = (MagicMock(), MagicMock(), "agent")
        await trial.connect_as(role)

    apply_policy.assert_awaited_once()
    assert apply_policy.await_args.args[:4] == (
        {},
        "gemini",
        AGENTS["gemini"],
        "/home/agent",
    )
    assert apply_policy.await_args.kwargs["disallow"] is True
    assert (
        connect_acp.await_args.kwargs["agent_env"]["BENCHFLOW_DISALLOW_WEB_TOOLS"]
        == "1"
    )

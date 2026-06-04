import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.rollout import (
    Role,
    Rollout,
    RolloutConfig,
    Scene,
    _agent_launch_with_web_policy,
    _apply_web_policy,
    _skill_nudge,
    _task_disallows_internet,
)


def _wire_fake_planes(trial: Rollout) -> MagicMock:
    planes = MagicMock()
    planes.agent_launch.side_effect = lambda agent, *, disallow_web_tools: (
        f"{agent} --no-web" if disallow_web_tools else agent
    )
    planes.resolve_agent_env.side_effect = lambda _agent, _model, env: env or {}
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.install_agent = AsyncMock(return_value=MagicMock())
    planes.write_credential_files = AsyncMock()
    planes.upload_subscription_auth = AsyncMock()
    planes.apply_web_tool_policy = AsyncMock()
    planes.connect_acp = AsyncMock(
        return_value=(MagicMock(), MagicMock(), MagicMock(), "agent")
    )
    trial._planes = planes
    return planes


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


def test_host_skill_nudge_does_not_read_task_skills_by_default(monkeypatch, tmp_path):
    """Guards PR #586 against prompt-level no-skills leaks."""
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

    assert prompts == ["Do the thing."]
    assert "alpha" not in prompts[0]
    assert "Internet access is disabled" not in prompts[0]
    assert "Do not browse" not in prompts[0]


def test_host_skill_nudge_reads_task_skills_when_explicitly_enabled(
    monkeypatch, tmp_path
):
    """Guards PR #586 so with-task-skills mode still gets skill nudges."""
    from benchflow.sdk import SDK

    monkeypatch.setenv("BENCHFLOW_SKILL_NUDGE", "name")
    (tmp_path / "instruction.md").write_text("Do the thing.")
    task_skills = tmp_path / "environment" / "skills"
    skill_dir = task_skills / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill.\n---\n# Alpha\n"
    )

    prompts = SDK._resolve_prompts(
        tmp_path,
        prompts=None,
        task_skills_dir=task_skills,
        skill_nudge=_skill_nudge({}),
        agent="claude-agent-acp",
    )

    assert prompts[0].startswith("Skills available at ~/.claude/skills: alpha.")
    assert prompts[0].endswith("Do the thing.")


def test_create_environment_preserves_agent_network_for_llm_runs(tmp_path):
    from benchflow.sandbox.setup import _create_environment

    original_env = MagicMock()
    original_env.allow_internet = False
    copied_env = MagicMock()
    copied_env.allow_internet = False
    original_env.model_copy.return_value = copied_env
    task = SimpleNamespace(
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(environment=original_env),
    )

    with patch("benchflow.sandbox.docker.DockerSandbox") as docker_env:
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
    from benchflow.sandbox.setup import _create_environment

    original_env = MagicMock()
    original_env.allow_internet = False
    task = SimpleNamespace(
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(environment=original_env),
    )

    with patch("benchflow.sandbox.docker.DockerSandbox") as docker_env:
        _create_environment("docker", task, tmp_path, "trial", MagicMock())

    original_env.model_copy.assert_not_called()
    assert docker_env.call_args.kwargs["task_env_config"] is original_env


@pytest.mark.asyncio
async def test_connect_as_applies_web_policy_to_role_env(tmp_path):
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[
            Scene(
                roles=[Role(name="agent", agent="claude-agent-acp", model="test-model")]
            )
        ],
        agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8080/v1"},
    )
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = {}
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=False))
    )
    planes = _wire_fake_planes(trial)
    captured = {}

    def fake_resolve(agent, model, env):
        captured["env"] = env
        return env or {}

    planes.resolve_agent_env.side_effect = fake_resolve
    await trial.connect_as(cfg.scenes[0].roles[0])

    assert captured["env"]["BENCHFLOW_PROVIDER_BASE_URL"] == "http://localhost:8080/v1"
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in captured["env"]
    assert (
        planes.connect_acp.await_args.kwargs["agent_env"][
            "BENCHFLOW_DISALLOW_WEB_TOOLS"
        ]
        == "1"
    )


def test_codex_launch_disable_is_gated_by_web_policy():
    base_cmd = (
        "/opt/benchflow/bin/codex-acp "
        "${OPENAI_BASE_URL:+-c openai_base_url=$OPENAI_BASE_URL}"
    )
    assert _agent_launch_with_web_policy("codex-acp", disallow=False) == base_cmd
    assert _agent_launch_with_web_policy("codex-acp", disallow=True) == (
        f"{base_cmd} -c tools.web_search=false"
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
    cfg = RolloutConfig(
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
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = {}
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._disallow_web_tools = True
    planes = _wire_fake_planes(trial)
    planes.install_agent.return_value = AGENTS["gemini"]

    await trial.connect_as(role)

    planes.apply_web_tool_policy.assert_awaited_once()
    assert planes.apply_web_tool_policy.await_args.args[:4] == (
        {},
        "gemini",
        AGENTS["gemini"],
        "/home/agent",
    )
    assert planes.apply_web_tool_policy.await_args.kwargs["disallow"] is True
    assert (
        planes.connect_acp.await_args.kwargs["agent_env"][
            "BENCHFLOW_DISALLOW_WEB_TOOLS"
        ]
        == "1"
    )


# ── Shell-command verification: setup_cmd produces valid agent config ──────


def test_task_allows_internet_when_explicitly_true():
    """allow_internet=True should not trigger web-tool disabling."""
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=True))
    )
    assert _task_disallows_internet(task) is False


def test_apply_web_policy_noop_when_not_disallowed():
    """disallow=False should return the original env dict unchanged."""
    env = {"KEY": "value"}
    result = _apply_web_policy(env, disallow=False)
    assert result is env
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in result


def _run_setup_cmd(agent_name: str, tmp_path) -> dict:
    """Execute an agent's disallow_web_tools_setup_cmd and return the JSON it wrote."""
    from benchflow.agents.registry import AGENTS

    cfg = AGENTS[agent_name]
    assert cfg.disallow_web_tools_setup_cmd, f"{agent_name} has no setup_cmd"

    result = subprocess.run(
        ["bash", "-c", cfg.disallow_web_tools_setup_cmd],
        env={"BENCHFLOW_AGENT_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"{agent_name} setup_cmd failed: {result.stderr}"
    return tmp_path


def test_claude_setup_cmd_disables_web_tools(tmp_path):
    """Claude setup_cmd should write settings.json denying WebSearch+WebFetch."""
    _run_setup_cmd("claude-agent-acp", tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())

    deny = settings["permissions"]["deny"]
    assert "WebSearch" in deny
    assert "WebFetch" in deny


def test_gemini_setup_cmd_disables_web_tools(tmp_path):
    """Gemini setup_cmd should write settings.json excluding google_web_search+web_fetch."""
    _run_setup_cmd("gemini", tmp_path)
    settings = json.loads((tmp_path / ".gemini" / "settings.json").read_text())

    excluded = settings["tools"]["exclude"]
    assert "google_web_search" in excluded
    assert "web_fetch" in excluded


def test_opencode_setup_cmd_disables_web_tools(tmp_path):
    """OpenCode setup_cmd should write opencode.json with webfetch=False."""
    _run_setup_cmd("opencode", tmp_path)
    settings = json.loads(
        (tmp_path / ".config" / "opencode" / "opencode.json").read_text()
    )

    assert settings["tools"]["webfetch"] is False


def test_openhands_setup_cmd_disables_browsing(tmp_path):
    """OpenHands setup_cmd should write config disabling browsing."""
    _run_setup_cmd("openhands", tmp_path)
    config = (tmp_path / ".openhands" / "config.toml").read_text()
    assert "enable_browsing = false" in config


def test_setup_cmd_is_idempotent(tmp_path):
    """Running setup_cmd twice should produce the same config, not duplicate entries."""
    _run_setup_cmd("claude-agent-acp", tmp_path)
    _run_setup_cmd("claude-agent-acp", tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())

    deny = settings["permissions"]["deny"]
    assert deny.count("WebSearch") == 1
    assert deny.count("WebFetch") == 1


def test_setup_cmd_merges_with_existing_config(tmp_path):
    """setup_cmd should merge into existing settings, not overwrite them."""
    settings_dir = tmp_path / ".gemini"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({"theme": "dark", "tools": {"timeout": 30}})
    )

    _run_setup_cmd("gemini", tmp_path)
    settings = json.loads((settings_dir / "settings.json").read_text())

    assert settings["theme"] == "dark"
    assert settings["tools"]["timeout"] == 30
    assert "google_web_search" in settings["tools"]["exclude"]
    assert "web_fetch" in settings["tools"]["exclude"]


def test_create_environment_does_not_flip_when_internet_allowed(tmp_path):
    """When allow_internet=True, preserve_agent_network should not modify env config."""
    from benchflow.sandbox.setup import _create_environment

    original_env = MagicMock()
    original_env.allow_internet = True
    task = SimpleNamespace(
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(environment=original_env),
    )

    with patch("benchflow.sandbox.docker.DockerSandbox") as docker_env:
        _create_environment(
            "docker", task, tmp_path, "trial", MagicMock(), preserve_agent_network=True
        )

    original_env.model_copy.assert_not_called()
    assert docker_env.call_args.kwargs["task_env_config"] is original_env


def test_task_toml_allow_internet_false_parsed_correctly(tmp_path):
    """A real task.toml with allow_internet=false should be correctly parsed."""
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[task]\n"
        'name = "test/internet-false"\n'
        "\n"
        "[environment]\n"
        "allow_internet = false\n"
    )
    (task_dir / "instruction.md").write_text("Test instruction.")

    task = Task(task_dir)
    assert task.config.environment.allow_internet is False
    assert _task_disallows_internet(task) is True


def test_task_toml_allow_internet_true_parsed_correctly(tmp_path):
    """A real task.toml with allow_internet=true (default) should not trigger policy."""
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        '[task]\nname = "test/internet-true"\n\n[environment]\nallow_internet = true\n'
    )
    (task_dir / "instruction.md").write_text("Test instruction.")

    task = Task(task_dir)
    assert task.config.environment.allow_internet is True
    assert _task_disallows_internet(task) is False


def test_task_toml_missing_allow_internet_defaults_to_allowed(tmp_path):
    """A task.toml without allow_internet should default to internet allowed."""
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('[task]\nname = "test/internet-default"\n')
    (task_dir / "instruction.md").write_text("Test instruction.")

    task = Task(task_dir)
    assert _task_disallows_internet(task) is False

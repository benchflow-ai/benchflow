"""Tests for agent install and skill deployment setup helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._agent_setup import deploy_skills, install_agent
from benchflow.agents.registry import AgentConfig
from benchflow.models import AgentInstallError


def _make_task(skills_dir: str | None):
    return SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(
                skills_dir=skills_dir,
            )
        )
    )


@pytest.mark.asyncio
async def test_deploy_skills_symlinks_agent_skill_paths_instead_of_copying(tmp_path):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/skills"],
    )

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=None,
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/app",
        task=_make_task("/opt/benchflow/skills"),
    )

    env.upload_dir.assert_not_called()
    env.exec.assert_awaited_once()

    cmd = env.exec.await_args.args[0]
    assert "cp -r" not in cmd
    assert " && " in cmd
    assert ";" not in cmd
    assert "ln -sfn /opt/benchflow/skills /home/agent/.agents/skills" in cmd
    assert "ln -sfn /opt/benchflow/skills /app/skills" in cmd


@pytest.mark.asyncio
async def test_deploy_skills_uploads_runtime_skills_and_links_shared_tree(tmp_path):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/skills"],
    )
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=skills_dir,
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task("/opt/benchflow/skills"),
    )

    env.upload_dir.assert_awaited_once_with(skills_dir, "/skills")
    env.exec.assert_awaited_once()

    distributed_link_cmd = env.exec.await_args.args[0]
    assert " && " in distributed_link_cmd
    assert ";" not in distributed_link_cmd
    assert "ln -sfn /skills /home/agent/.agents/skills" in distributed_link_cmd
    assert "ln -sfn /skills /workspace/skills" in distributed_link_cmd
    assert "/root/.agents/skills" not in distributed_link_cmd
    assert "/app/skills" not in distributed_link_cmd


@pytest.mark.asyncio
async def test_deploy_skills_chowns_skill_parent_for_pi_acp_layout(tmp_path):
    """Guards the fix for issue #7 against the regression where
    `_skill_link_cmd` left `~/.pi/agent` root-owned, breaking pi-acp's
    `models.json` write under openai-completions providers (vLLM)."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="pi-acp",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.pi/agent/skills", "$HOME/.agents/skills"],
    )

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=None,
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task("/opt/benchflow/skills"),
    )

    cmd = env.exec.await_args.args[0]
    assert "chown agent:agent /home/agent/.pi/agent" in cmd
    assert "chown agent:agent /home/agent/.agents" in cmd
    pi_chown_idx = cmd.index("chown agent:agent /home/agent/.pi/agent")
    pi_link_idx = cmd.index("ln -sfn /opt/benchflow/skills /home/agent/.pi/agent/skills")
    assert pi_chown_idx < pi_link_idx, "chown must precede ln to keep parent agent-owned"


@pytest.mark.asyncio
async def test_deploy_skills_skips_chown_when_no_sandbox_user(tmp_path):
    """Guards the fix for issue #7: when sandbox_user is None, the chown
    plumbing must stay no-op so root-only deploys keep working."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills"],
    )

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=None,
        agent_cfg=agent_cfg,
        sandbox_user=None,
        agent_cwd="/workspace",
        task=_make_task("/opt/benchflow/skills"),
    )

    cmd = env.exec.await_args.args[0]
    assert "chown" not in cmd
    assert "ln -sfn /opt/benchflow/skills /root/.agents/skills" in cmd


@pytest.mark.asyncio
async def test_deploy_skills_falls_back_when_local_skills_dir_is_missing(tmp_path):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/skills"],
    )

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=tmp_path / "missing-skills",
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task("/opt/benchflow/skills"),
    )

    env.upload_dir.assert_not_called()
    env.exec.assert_awaited_once()

    distributed_link_cmd = env.exec.await_args.args[0]
    assert (
        "ln -sfn /opt/benchflow/skills /home/agent/.agents/skills"
        in distributed_link_cmd
    )
    assert "ln -sfn /opt/benchflow/skills /workspace/skills" in distributed_link_cmd
    assert "ln -sfn /skills /home/agent/.agents/skills" not in distributed_link_cmd
    assert "ln -sfn /skills /workspace/skills" not in distributed_link_cmd


@pytest.mark.asyncio
async def test_deploy_skills_raises_when_skill_linking_fails(tmp_path):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=17, stdout="link failed"))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills"],
    )

    with pytest.raises(RuntimeError, match="Failed to link skills"):
        await deploy_skills(
            env=env,
            task_path=tmp_path,
            skills_dir=None,
            agent_cfg=agent_cfg,
            sandbox_user="agent",
            agent_cwd="/app",
            task=_make_task("/opt/benchflow/skills"),
        )


@pytest.mark.asyncio
async def test_install_agent_writes_command_stdout_and_stderr_on_failure(
    tmp_path: Path,
):
    env = SimpleNamespace()
    env.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=1, stdout="", stderr="uv: command not found\n"),
            SimpleNamespace(
                stdout="OS:\nID=ubuntu\nNode:\nv22.0.0\nAgent:\nnot found\n",
                stderr="",
                return_code=0,
            ),
        ]
    )

    with pytest.raises(AgentInstallError) as exc_info:
        await install_agent(env, "openhands", tmp_path)

    err = exc_info.value
    log_path = Path(err.log_path)
    assert log_path == tmp_path / "agent" / "install-stdout.txt"
    assert log_path.exists()
    log_text = log_path.read_text()
    assert log_text.startswith("$ ")
    assert "uv tool install openhands --python 3.12" in log_text
    assert "=== stderr ===" in log_text
    assert "uv: command not found" in log_text
    assert err.stdout == log_text
    assert "ID=ubuntu" in err.diagnostics

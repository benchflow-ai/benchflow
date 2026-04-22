"""Tests for agent install and skill deployment setup helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._agent_setup import deploy_skills
from benchflow.agents.registry import AgentConfig


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
    assert "ln -sfn /opt/benchflow/skills /home/agent/.agents/skills" in distributed_link_cmd
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

"""Tests for agent install and skill deployment setup helpers."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.install import apply_web_tool_policy, deploy_skills, install_agent
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
async def test_deploy_skills_can_skip_task_declared_skills(tmp_path):
    """Self-gen starts from the no-skill task path, even if task.toml declares skills."""
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
        sandbox_user="agent",
        agent_cwd="/app",
        task=_make_task("/opt/benchflow/skills"),
        include_task_skills=False,
    )

    env.upload_dir.assert_not_called()
    env.exec.assert_not_called()


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
async def test_deploy_skills_skips_runtime_upload_when_dockerfile_already_injected(
    tmp_path,
):
    """Guards the fix from PR #230 for issue #11 against the regression where
    skills got deployed twice — once baked into the image via
    `_inject_skills_into_dockerfile`, and again at runtime via
    `env.upload_dir(..., "/skills")`. The runtime cp failed with
    `cannot overwrite directory "/skills/<entry>" with non-directory "/skills"`
    when the source contained a symlink colliding with a baked entry.

    The detection works only when `deploy_skills` receives the *effective*
    task path — the temp copy whose Dockerfile actually carries the
    injected `COPY _deps/skills /skills/` line.
    """
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills"],
    )
    effective_task_path = tmp_path / "task"
    (effective_task_path / "environment").mkdir(parents=True)
    (effective_task_path / "environment" / "Dockerfile").write_text(
        "FROM python:3.12\nCOPY _deps/skills /skills/\n"
    )
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    await deploy_skills(
        env=env,
        task_path=effective_task_path,
        skills_dir=skills_dir,
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task(None),
    )

    env.upload_dir.assert_not_called()
    env.exec.assert_awaited_once()
    link_cmd = env.exec.await_args.args[0]
    assert "ln -sfn /skills /home/agent/.agents/skills" in link_cmd


@pytest.mark.asyncio
async def test_deploy_skills_chowns_full_dir_chain_for_pi_acp_layout(tmp_path):
    """Guards the fix from PR #211 against the regression where only the
    symlink's immediate parent (`~/.pi/agent`) was chowned, leaving the
    intermediate `~/.pi/` root-owned. pi-acp's `session/new` then failed
    when trying to mkdir `~/.pi/pi-acp` for session state. Earlier PR #210
    landed half the fix; this test pins the full ancestor chain."""
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
    # Both newly-created dirs must be chowned in one chain — without
    # `/home/agent/.pi`, pi-acp can't mkdir `~/.pi/pi-acp` for session state.
    assert "chown agent:agent /home/agent/.pi /home/agent/.pi/agent" in cmd
    # Single-segment chain stays single-arg.
    assert "chown agent:agent /home/agent/.agents " in cmd
    pi_chown = "chown agent:agent /home/agent/.pi /home/agent/.pi/agent"
    pi_link = "ln -sfn /opt/benchflow/skills /home/agent/.pi/agent/skills"
    assert cmd.index(pi_chown) < cmd.index(pi_link), (
        "chown must precede ln so the symlink's ancestors are agent-owned"
    )


@pytest.mark.asyncio
async def test_deploy_skills_skips_chown_when_no_sandbox_user(tmp_path):
    """Guards the fix from PR #210: when sandbox_user is None, the chown
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
async def test_deploy_skills_oracle_uses_default_paths_and_root_home(tmp_path):
    """When agent_cfg is None (oracle), skills go to _ORACLE_SKILL_PATHS under /root."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()

    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "my-skill").mkdir()

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=str(skills),
        agent_cfg=None,
        sandbox_user="agent",
        agent_cwd="/app",
        task=_make_task(None),
    )

    env.upload_dir.assert_awaited_once()
    assert env.exec.await_count == 1
    link_cmd = env.exec.await_args_list[0].args[0]
    assert "/root/.claude/skills" in link_cmd
    assert "/root/.codex/skills" in link_cmd
    assert "/app/skills" in link_cmd
    assert "/home/agent" not in link_cmd


@pytest.mark.asyncio
async def test_deploy_skills_agent_with_empty_skill_paths_does_not_use_oracle_paths(
    tmp_path,
):
    """An agent with skill_paths=[] should NOT get oracle fallback paths."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    agent_cfg = AgentConfig(
        name="minimal-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=[],
    )

    await deploy_skills(
        env=env,
        task_path=tmp_path,
        skills_dir=None,
        agent_cfg=agent_cfg,
        sandbox_user=None,
        agent_cwd="/app",
        task=_make_task("/skills"),
    )

    for call in env.exec.await_args_list:
        cmd = call.args[0]
        assert "/root/.claude/skills" not in cmd


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
    assert (
        "uv tool install --force --refresh "
        "--from 'git+https://github.com/OpenHands/OpenHands-CLI.git@main' "
        "openhands --python 3.12" in log_text
    )
    assert "=== stderr ===" in log_text
    assert "uv: command not found" in log_text
    assert err.stdout == log_text
    assert "ID=ubuntu" in err.diagnostics


@pytest.mark.asyncio
async def test_apply_web_tool_policy_runs_agent_setup_command(tmp_path):
    calls = []

    async def exec_cmd(cmd, *, timeout_sec=None, **kwargs):
        calls.append((cmd, timeout_sec, kwargs))
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return SimpleNamespace(
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    env = SimpleNamespace(exec=exec_cmd)
    home = tmp_path / "agent home"
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        disallow_web_tools_setup_cmd=(
            'mkdir -p "$BENCHFLOW_AGENT_HOME" && '
            'printf disabled > "$BENCHFLOW_AGENT_HOME/no-web"'
        ),
    )

    await apply_web_tool_policy(
        env,
        "test-agent",
        agent_cfg,
        str(home),
        disallow=True,
    )

    assert (home / "no-web").read_text() == "disabled"
    assert len(calls) == 1
    cmd, timeout_sec, kwargs = calls[0]
    assert cmd.startswith("export BENCHFLOW_AGENT_HOME=")
    assert 'printf disabled > "$BENCHFLOW_AGENT_HOME/no-web"' in cmd
    assert timeout_sec == 15
    assert kwargs == {}


@pytest.mark.asyncio
async def test_apply_web_tool_policy_is_gated_off_when_allowed():
    env = MagicMock()
    env.exec = AsyncMock()
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        disallow_web_tools_setup_cmd="false",
    )

    await apply_web_tool_policy(
        env,
        "test-agent",
        agent_cfg,
        "/home/agent",
        disallow=False,
    )

    env.exec.assert_not_called()

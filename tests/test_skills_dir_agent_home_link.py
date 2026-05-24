"""Regression: --skills-dir skills baked into /skills must be linked into agent home.

Guards the fix for #339 (and the earlier #11 fix on PR #285): when
``--skills-dir`` ships skills into the image at ``/skills`` via the
Dockerfile injection path, ``deploy_skills`` must still emit the
``ln -sfn /skills <agent-home>/...skills`` commands so the agent can
actually discover them under ``$HOME``.

Pre-fix behavior: ``deploy_skills`` saw the Dockerfile already-injected
sentinel, logged "Skills already injected via Dockerfile", and returned
without setting ``effective_skills``. ``/skills`` then existed in the
container but no symlink linked it into the agent home (e.g.
``/home/agent/.agents/skills`` for codex-acp), so the agent couldn't find
the skills.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.install import deploy_skills
from benchflow.agents.registry import AGENTS, AgentConfig
from benchflow.sandbox.setup import _inject_skills_into_dockerfile


def _make_task(skills_dir: str | None):
    return SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(skills_dir=skills_dir)
        )
    )


def _mock_env():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    env.upload_dir = AsyncMock()
    return env


@pytest.mark.asyncio
async def test_dockerfile_baked_skills_link_into_codex_agent_home(tmp_path):
    """End-to-end: baking --skills-dir into the Dockerfile still links the
    skills into ``/home/agent/.agents/skills`` for codex-acp, the discovery
    path the Codex CLI scans under ``$HOME``."""
    # Stage a fake --skills-dir with one concrete skill.
    skills_src = tmp_path / "user-skills"
    (skills_src / "expected_skill").mkdir(parents=True)
    (skills_src / "expected_skill" / "SKILL.md").write_text("# expected\n")

    # Build a minimal task path with a Dockerfile, then run the real
    # baking pass that --skills-dir uses.
    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text("FROM python:3.12\n")
    _inject_skills_into_dockerfile(task_path, skills_src)

    dockerfile_text = (task_path / "environment" / "Dockerfile").read_text()
    assert "COPY _deps/skills /skills/" in dockerfile_text, (
        "sanity: injection step must write the sentinel line that deploy_skills "
        "matches against"
    )
    # The fake skills dir was copied into _deps/, so the image really would
    # bake the skill file at /skills/expected_skill/SKILL.md.
    assert (
        task_path / "environment" / "_deps" / "skills" / "expected_skill" / "SKILL.md"
    ).exists()

    env = _mock_env()
    await deploy_skills(
        env=env,
        task_path=task_path,
        skills_dir=skills_src,
        agent_cfg=AGENTS["codex-acp"],
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task(None),
    )

    # Runtime upload must be skipped (skills are already in the image),
    # but the agent-home link command must still fire.
    env.upload_dir.assert_not_called()
    env.exec.assert_awaited_once()
    cmd = env.exec.await_args.args[0]
    assert "ln -sfn /skills /home/agent/.agents/skills" in cmd, (
        "after #339 fix, codex-acp's $HOME/.agents/skills must point at the "
        f"baked /skills tree; got: {cmd}"
    )


@pytest.mark.asyncio
async def test_dockerfile_baked_skills_link_for_arbitrary_agent_home(tmp_path):
    """Same #339 regression, parametric on agent home: when skills_dir is
    set and the Dockerfile already bakes /skills, every configured
    skill_path under $HOME gets its symlink rebuilt against /skills, not
    silently dropped."""
    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text(
        "FROM python:3.12\nCOPY _deps/skills /skills/\n"
    )
    skills_src = tmp_path / "skills"
    skills_src.mkdir()

    agent_cfg = AgentConfig(
        name="hypothetical-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.hypothetical/skills", "$WORKSPACE/skills"],
    )

    env = _mock_env()
    await deploy_skills(
        env=env,
        task_path=task_path,
        skills_dir=skills_src,
        agent_cfg=agent_cfg,
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task(None),
    )

    env.upload_dir.assert_not_called()
    cmd = env.exec.await_args.args[0]
    # Both $HOME-based and $WORKSPACE-based discovery paths must land
    # against /skills — the pre-fix bug dropped both.
    assert "ln -sfn /skills /home/agent/.hypothetical/skills" in cmd
    assert "ln -sfn /skills /workspace/skills" in cmd


@pytest.mark.asyncio
async def test_dockerfile_baked_skills_link_when_task_declares_no_skills(tmp_path):
    """Guards the specific code-path the #339 reporter hit: --skills-dir is
    set, the Dockerfile already bakes /skills, AND the task itself declares
    no skills_dir. The link to the agent home must still fire from /skills."""
    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text(
        "FROM python:3.12\nCOPY _deps/skills /skills/\n"
    )
    skills_src = tmp_path / "skills"
    skills_src.mkdir()

    env = _mock_env()
    await deploy_skills(
        env=env,
        task_path=task_path,
        skills_dir=skills_src,
        agent_cfg=AGENTS["codex-acp"],
        sandbox_user="agent",
        agent_cwd="/workspace",
        task=_make_task(None),  # task.toml declares no skills_dir
    )

    env.exec.assert_awaited_once()
    cmd = env.exec.await_args.args[0]
    assert "ln -sfn /skills /home/agent/.agents/skills" in cmd

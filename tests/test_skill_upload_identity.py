"""Guard FF-sw-agents-C1 at 1f93158d: uploads cannot traverse agent paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.install import deploy_skills
from benchflow.agents.registry import AGENTS


@pytest.mark.parametrize("destination", ["/skills", "/app/packs", "/opt/runtime/packs"])
@pytest.mark.parametrize("agent", [*AGENTS, None])
async def test_runtime_skill_upload_uses_private_staging(tmp_path, destination, agent):
    """Every registry spelling and the oracle use the same protected upload path."""
    skills = tmp_path / "skills"
    (skills / "example").mkdir(parents=True)
    (skills / "example" / "SKILL.md").write_text("public bundled skill")
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        upload_dir=AsyncMock(),
    )
    await deploy_skills(
        env,
        tmp_path,
        skills,
        AGENTS[agent] if agent else None,
        "agent" if agent else None,
        "/app",
        skills_sandbox_dir=destination,
    )
    uploaded = env.upload_dir.await_args.args[1]
    assert uploaded.startswith("/.runtime-skills-")
    assert uploaded != destination
    copies = [call for call in env.exec.await_args_list if "cp -R " in call.args[0]]
    assert len(copies) == 1
    assert copies[0].kwargs["user"] == ("agent" if agent else "root")
    assert destination in copies[0].args[0]


async def test_skip_install_model_keeps_its_sandbox_identity(tmp_path):
    """Guards C1 for skip_agent_install, which also passes agent_cfg=None."""
    skills = tmp_path / "skills"
    skills.mkdir()
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        upload_dir=AsyncMock(),
    )
    await deploy_skills(env, tmp_path, skills, None, "agent", "/app")
    copies = [call for call in env.exec.await_args_list if "cp -R " in call.args[0]]
    assert copies[0].kwargs["user"] == "agent"
    assert env.exec.await_args.kwargs["user"] == "agent"
    assert "/root/" not in env.exec.await_args.args[0]


@pytest.mark.parametrize("failure", ["prepare", "publish", "copy", "cleanup"])
async def test_runtime_skill_upload_fails_closed(tmp_path, failure):
    """A failed staging, copy or cleanup must prevent the next agent phase."""
    skills = tmp_path / "skills"
    skills.mkdir()

    async def execute(cmd, **kwargs):
        failed = (
            (failure == "prepare" and "mkdir -m 700" in cmd)
            or (failure == "publish" and "chmod" in cmd and "find" in cmd)
            or (failure == "copy" and "cp -R " in cmd)
            or (failure == "cleanup" and "rm -rf -- /.runtime-skills-" in cmd)
        )
        return SimpleNamespace(return_code=int(failed))

    env = SimpleNamespace(exec=AsyncMock(side_effect=execute), upload_dir=AsyncMock())
    with pytest.raises(RuntimeError, match="skill"):
        await deploy_skills(
            env, tmp_path, skills, AGENTS["deepagents"], "agent", "/app"
        )

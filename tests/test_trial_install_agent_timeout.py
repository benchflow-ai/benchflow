"""Tests for Rollout.install_agent timeout wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.skill_policy import SKILL_MODE_WITH_SKILL


def _make_trial(tmp_path, *, agent: str, sandbox_setup_timeout: int) -> Rollout:
    config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task",
        agent=agent,
        prompts=[None],
        sandbox_user="agent",
        sandbox_setup_timeout=sandbox_setup_timeout,
    )
    trial = Rollout(config)
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=MagicMock(stdout="/workspace\n"))
    trial._rollout_dir = tmp_path / "trial"
    trial._rollout_dir.mkdir()
    trial._rollout_paths = MagicMock()
    trial._task = MagicMock()
    trial._effective_locked = []
    return trial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "expected_setup_return"),
    [
        ("claude-agent-acp", "/home/agent"),
        ("oracle", None),
    ],
)
async def test_install_agent_forwards_sandbox_setup_timeout(
    tmp_path, monkeypatch, agent, expected_setup_return
):
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=41)

    install_agent_mock = AsyncMock(return_value=MagicMock())
    write_credential_files_mock = AsyncMock()
    upload_subscription_auth_mock = AsyncMock()
    snapshot_build_config_mock = AsyncMock()
    seed_verifier_workspace_mock = AsyncMock()
    deploy_skills_mock = AsyncMock()
    lockdown_paths_mock = AsyncMock()
    setup_sandbox_user_mock = AsyncMock(return_value=expected_setup_return)

    monkeypatch.setattr(trial._planes, "install_agent", install_agent_mock)
    monkeypatch.setattr(
        trial._planes, "write_credential_files", write_credential_files_mock
    )
    monkeypatch.setattr(
        trial._planes, "upload_subscription_auth", upload_subscription_auth_mock
    )
    monkeypatch.setattr(
        trial._planes, "snapshot_build_config", snapshot_build_config_mock
    )
    monkeypatch.setattr(
        trial._planes, "seed_verifier_workspace", seed_verifier_workspace_mock
    )
    monkeypatch.setattr(trial._planes, "deploy_skills", deploy_skills_mock)
    monkeypatch.setattr(trial._planes, "lockdown_paths", lockdown_paths_mock)
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup_sandbox_user_mock)

    await trial.install_agent()

    setup_sandbox_user_mock.assert_awaited_once()
    args, kwargs = setup_sandbox_user_mock.await_args
    assert args[1] == "agent"
    assert kwargs["timeout_sec"] == 41
    assert kwargs["workspace"] == "/workspace"

    if agent == "oracle":
        install_agent_mock.assert_not_awaited()
        write_credential_files_mock.assert_not_awaited()
        deploy_skills_mock.assert_awaited_once()
        assert trial._agent_cwd == "/workspace"
    else:
        install_agent_mock.assert_awaited_once()
        write_credential_files_mock.assert_awaited_once()
        deploy_skills_mock.assert_awaited_once()
        assert trial._agent_cwd == "/home/agent"

    snapshot_build_config_mock.assert_awaited_once()
    seed_verifier_workspace_mock.assert_awaited_once()
    lockdown_paths_mock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "oracle"])
async def test_install_agent_passes_effective_task_path_to_deploy_skills(
    tmp_path, monkeypatch, agent
):
    """Guards the fix from PR #308 for issue #229: deploy_skills double-deploys
    when skills_dir is set.

    `_setup` copies the task to a temp dir and injects
    `COPY _deps/skills /skills/` into that temp Dockerfile, recording the
    copy as `_effective_task_path`. `deploy_skills` decides whether to skip
    the runtime `/skills` upload by reading the Dockerfile at the path it is
    given — so it must receive `_effective_task_path`, not the original
    `cfg.task_path` (whose Dockerfile is never injected). Passing the
    original path makes `already_injected` always False and triggers a
    second `/skills` upload on top of the baked image, failing with
    `cannot overwrite directory "/skills/..." with non-directory "/skills"`.
    """
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=41)
    trial._config = trial._config.__class__.from_legacy(
        task_path=tmp_path / "original-task",
        agent=agent,
        prompts=[None],
        sandbox_user="agent",
        sandbox_setup_timeout=41,
        skill_mode=SKILL_MODE_WITH_SKILL,
        skills_dir=tmp_path / "skills",
    )
    effective_task_path = tmp_path / "benchflow-task-tmp" / "task"
    trial._effective_task_path = effective_task_path

    deploy_skills_mock = AsyncMock()
    monkeypatch.setattr(trial._planes, "install_agent", AsyncMock())
    monkeypatch.setattr(trial._planes, "write_credential_files", AsyncMock())
    monkeypatch.setattr(trial._planes, "upload_subscription_auth", AsyncMock())
    monkeypatch.setattr(trial._planes, "snapshot_build_config", AsyncMock())
    monkeypatch.setattr(trial._planes, "seed_verifier_workspace", AsyncMock())
    monkeypatch.setattr(trial._planes, "deploy_skills", deploy_skills_mock)
    monkeypatch.setattr(trial._planes, "lockdown_paths", AsyncMock())
    monkeypatch.setattr(
        trial._planes, "setup_sandbox_user", AsyncMock(return_value="/home/agent")
    )
    monkeypatch.setattr(trial._planes, "apply_web_tool_policy", AsyncMock())

    await trial.install_agent()

    deploy_skills_mock.assert_awaited_once()
    passed_task_path = deploy_skills_mock.await_args.args[1]
    assert passed_task_path == effective_task_path
    assert passed_task_path != trial._config.task_path


@pytest.mark.asyncio
async def test_install_agent_applies_web_policy_after_sandbox_setup(
    tmp_path, monkeypatch
):
    trial = _make_trial(tmp_path, agent="openhands", sandbox_setup_timeout=41)
    trial._disallow_web_tools = True

    order = []

    async def setup_sandbox_user_mock(*args, **kwargs):
        order.append("sandbox")
        return "/home/agent"

    async def apply_web_tool_policy_mock(*args, **kwargs):
        order.append("web-policy")

    async def write_credential_files_mock(*args, **kwargs):
        order.append("credentials")

    install_agent_mock = AsyncMock(return_value=MagicMock())

    monkeypatch.setattr(trial._planes, "install_agent", install_agent_mock)
    monkeypatch.setattr(
        trial._planes, "write_credential_files", write_credential_files_mock
    )
    monkeypatch.setattr(trial._planes, "upload_subscription_auth", AsyncMock())
    monkeypatch.setattr(trial._planes, "snapshot_build_config", AsyncMock())
    monkeypatch.setattr(trial._planes, "seed_verifier_workspace", AsyncMock())
    monkeypatch.setattr(trial._planes, "deploy_skills", AsyncMock())
    monkeypatch.setattr(trial._planes, "lockdown_paths", AsyncMock())
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup_sandbox_user_mock)
    monkeypatch.setattr(
        trial._planes, "apply_web_tool_policy", apply_web_tool_policy_mock
    )

    await trial.install_agent()

    assert order == ["sandbox", "credentials", "web-policy"]

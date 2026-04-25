"""Tests for Trial.install_agent timeout wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.trial import Trial, TrialConfig


def _make_trial(tmp_path, *, agent: str, sandbox_setup_timeout: int) -> Trial:
    config = TrialConfig.from_legacy(
        task_path=tmp_path / "task",
        agent=agent,
        prompts=[None],
        sandbox_user="agent",
        sandbox_setup_timeout=sandbox_setup_timeout,
    )
    trial = Trial(config)
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=MagicMock(stdout="/workspace\n"))
    trial._trial_dir = tmp_path / "trial"
    trial._trial_dir.mkdir()
    trial._trial_paths = MagicMock()
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

    monkeypatch.setattr("benchflow.trial.install_agent", install_agent_mock)
    monkeypatch.setattr(
        "benchflow.trial.write_credential_files", write_credential_files_mock
    )
    monkeypatch.setattr(
        "benchflow.trial.upload_subscription_auth", upload_subscription_auth_mock
    )
    monkeypatch.setattr(
        "benchflow.trial._snapshot_build_config", snapshot_build_config_mock
    )
    monkeypatch.setattr(
        "benchflow.trial._seed_verifier_workspace", seed_verifier_workspace_mock
    )
    monkeypatch.setattr("benchflow.trial.deploy_skills", deploy_skills_mock)
    monkeypatch.setattr("benchflow.trial.lockdown_paths", lockdown_paths_mock)
    monkeypatch.setattr(
        "benchflow.trial.setup_sandbox_user", setup_sandbox_user_mock
    )

    await trial.install_agent()

    setup_sandbox_user_mock.assert_awaited_once()
    args, kwargs = setup_sandbox_user_mock.await_args
    assert args[1] == "agent"
    assert kwargs["timeout_sec"] == 41
    assert kwargs["workspace"] == "/workspace"

    if agent == "oracle":
        install_agent_mock.assert_not_awaited()
        write_credential_files_mock.assert_not_awaited()
        deploy_skills_mock.assert_not_awaited()
        assert trial._agent_cwd == "/workspace"
    else:
        install_agent_mock.assert_awaited_once()
        write_credential_files_mock.assert_awaited_once()
        deploy_skills_mock.assert_awaited_once()
        assert trial._agent_cwd == "/home/agent"

    snapshot_build_config_mock.assert_awaited_once()
    seed_verifier_workspace_mock.assert_awaited_once()
    lockdown_paths_mock.assert_awaited_once()

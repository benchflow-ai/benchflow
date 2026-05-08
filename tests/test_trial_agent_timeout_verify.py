from pathlib import Path

import pytest

from benchflow.models import RunResult
from benchflow.trial import Scene, Trial, TrialConfig


@pytest.mark.asyncio
async def test_agent_timeout_without_verifier_reward_counts_as_zero(
    tmp_path: Path, monkeypatch
):
    """Guards the 2026-05-05 agent-timeout reward fallback fix."""
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        scenes=[Scene.single(agent="dummy")],
    )
    trial = Trial(cfg)
    calls: list[str] = []

    async def setup():
        calls.append("setup")
        trial._trial_dir = tmp_path / "trial"
        trial._trial_dir.mkdir()
        trial._trial_name = "trial-1"

    async def start():
        calls.append("start")

    async def install_agent():
        calls.append("install_agent")

    async def run_scene(_scene):
        calls.append("run_scene")
        raise TimeoutError("Agent prompt exceeded wall-clock budget 5s")

    async def verify():
        calls.append("verify")
        trial._rewards = None
        trial._verifier_error = "verifier crashed: No reward file found"
        return trial._rewards

    async def cleanup():
        calls.append("cleanup")

    def build_result():
        calls.append("build_result")
        return RunResult(
            task_name="task",
            trial_name=trial._trial_name or "",
            rewards=trial._rewards,
            error=trial._error,
            verifier_error=trial._verifier_error,
        )

    monkeypatch.setattr(trial, "setup", setup)
    monkeypatch.setattr(trial, "start", start)
    monkeypatch.setattr(trial, "install_agent", install_agent)
    monkeypatch.setattr(trial, "_run_scene", run_scene)
    monkeypatch.setattr(trial, "verify", verify)
    monkeypatch.setattr(trial, "cleanup", cleanup)
    monkeypatch.setattr(trial, "_build_result", build_result)

    result = await trial.run()

    assert calls == [
        "setup",
        "start",
        "install_agent",
        "run_scene",
        "verify",
        "cleanup",
        "build_result",
    ]
    assert result.error == "Agent prompt exceeded wall-clock budget 5s"
    assert result.rewards == {"reward": 0.0}
    assert result.verifier_error is None

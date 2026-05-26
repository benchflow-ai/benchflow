from pathlib import Path

import pytest

from benchflow.models import RunResult
from benchflow.rollout import Rollout, RolloutConfig, Scene


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_error", "expected_rewards", "expected_verifier_error"),
    [
        (
            "verifier crashed: No reward file found",
            None,
            "verifier crashed: No reward file found",
        ),
        (None, {"reward": 0.0}, None),
    ],
)
async def test_agent_timeout_verifier_result_handling(
    tmp_path: Path,
    monkeypatch,
    verifier_error: str | None,
    expected_rewards: dict[str, float] | None,
    expected_verifier_error: str | None,
):
    """Guards the reward-output regression on v0.5-integration@ffef85d."""
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[Scene.single(agent="dummy")],
    )
    trial = Rollout(cfg)
    calls: list[str] = []

    async def setup():
        calls.append("setup")
        trial._rollout_dir = tmp_path / "trial"
        trial._rollout_dir.mkdir()
        trial._rollout_name = "trial-1"

    async def start():
        calls.append("start")

    async def install_agent():
        calls.append("install_agent")

    async def run_steps(_steps):
        calls.append("run_steps")
        raise TimeoutError("Agent prompt exceeded wall-clock budget 5s")

    async def verify():
        calls.append("verify")
        trial._rewards = None
        trial._verifier_error = verifier_error
        return trial._rewards

    async def cleanup():
        calls.append("cleanup")

    def build_result():
        calls.append("build_result")
        return RunResult(
            task_name="task",
            rollout_name=trial._rollout_name or "",
            rewards=trial._rewards,
            error=trial._error,
            verifier_error=trial._verifier_error,
        )

    monkeypatch.setattr(trial, "setup", setup)
    monkeypatch.setattr(trial, "start", start)
    monkeypatch.setattr(trial, "install_agent", install_agent)
    monkeypatch.setattr(trial, "_run_steps", run_steps)
    monkeypatch.setattr(trial, "verify", verify)
    monkeypatch.setattr(trial, "cleanup", cleanup)
    monkeypatch.setattr(trial, "_build_result", build_result)

    result = await trial.run()

    assert calls == [
        "setup",
        "start",
        "install_agent",
        "run_steps",
        "verify",
        "cleanup",
        "build_result",
    ]
    assert result.error == "Agent prompt exceeded wall-clock budget 5s"
    assert result.rewards == expected_rewards
    assert result.verifier_error == expected_verifier_error

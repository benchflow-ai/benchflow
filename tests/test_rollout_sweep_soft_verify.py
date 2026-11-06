"""Guards FF-sw-rollout publication ordering against 363d815b."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rollout._setup import VerifierPublicationError
from benchflow.sandbox.user import PassthroughUser
from tests.reachable_fixture import directory
from tests.test_user import _make_user_trial


@pytest.mark.parametrize("backend", ["mounted", "remote"])
async def test_soft_verify_publishes_each_current_reply(tmp_path, backend):
    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path)
    trial._rollout_paths.agent_dir = directory(tmp_path / "agent")
    trial._env.is_mounted = backend == "mounted"
    trial._env.upload_file = AsyncMock()
    trial._config.sandbox_user = None

    async def grade():
        reply = trial._rollout_paths.agent_dir / "output.txt"
        return SimpleNamespace(
            rewards={
                "reward": float(reply.is_file() and reply.read_text().strip() == "42")
            }
        )

    with (
        patch("benchflow.task.Verifier") as verifier,
        patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
        patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
    ):
        verifier.return_value.verify = AsyncMock(side_effect=grade)
        for reply, score in [("wrong", 0), ("42", 1), ("wrong again", 0)]:
            trial._trajectory.extend(
                [
                    {"type": "user_message", "text": "next turn"},
                    {"type": "agent_message", "text": reply},
                ]
            )
            rewards, _, error = await trial.soft_verify()
            assert error is None
            assert rewards == {"reward": float(score)}


async def test_soft_verify_publication_failure_never_grades(tmp_path):
    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path)
    trial._rollout_paths.agent_dir = directory(tmp_path / "agent")
    trial._trajectory = [{"type": "agent_message", "text": "42"}]
    with (
        patch(
            "benchflow.rollout._publish_trajectory_for_verifier",
            AsyncMock(
                side_effect=VerifierPublicationError("dummy publication failure")
            ),
        ),
        patch("benchflow.task.Verifier") as verifier,
        patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
        patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
    ):
        verifier.return_value.verify = AsyncMock(
            return_value=SimpleNamespace(rewards={"reward": 1.0})
        )
        rewards, _, error = await trial.soft_verify()
        assert rewards is None
        assert "publication" in error
        verifier.assert_not_called()

"""Guards CONF-rollout-1 against the incomplete fixes at 82e412f1."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rollout._setup import _publish_trajectory_for_verifier
from benchflow.sandbox.user import PassthroughUser
from tests.reachable_fixture import directory
from tests.test_acp_final_reply import RecordingEnv, _staged
from tests.test_user import _make_user_trial

NO_REPLIES = [
    [],
    [{"type": "tool_call", "tool_call_id": "t"}],
    [{"type": "agent_message", "text": ""}],
    [{"type": "agent_message", "text": " \n"}],
    [{"type": "agent_message", "text": "42", "parent_tool_call_id": "t"}],
    [{"type": "agent_thought", "text": "42"}],
]


@pytest.mark.parametrize("tail", NO_REPLIES)
@pytest.mark.parametrize("empty_capture", [False, True])
async def test_no_reply_replaces_previous_output(tmp_path, tail, empty_capture):
    env = RecordingEnv()
    agent_dir = tmp_path / "agent"
    await _publish_trajectory_for_verifier(
        env, [{"type": "agent_message", "text": "42"}], directory(agent_dir)
    )
    env.uploads.clear()
    trajectory = (
        []
        if empty_capture
        else [
            {"type": "agent_message", "text": "42"},
            {"type": "user_message", "text": "next round"},
            *tail,
        ]
    )
    await _publish_trajectory_for_verifier(env, trajectory, directory(agent_dir))
    assert (agent_dir / "output.txt").read_text() == ""
    assert _staged(env, "output.txt") == ""


@pytest.mark.parametrize("backend", ["mounted", "remote"])
@pytest.mark.parametrize("tail", NO_REPLIES)
async def test_retry_cannot_combine_previous_reply_with_current_files(
    tmp_path, backend, tail
):
    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path)
    trial._rollout_paths.agent_dir = directory(tmp_path / "agent")
    trial._env.is_mounted = backend == "mounted"
    trial._env.upload_file = AsyncMock()
    trial._config.sandbox_user = None
    ready = False

    async def grade():
        reply = trial._rollout_paths.agent_dir / "output.txt"
        return SimpleNamespace(
            rewards={"reward": float(ready and reply.read_text().strip() == "42")}
        )

    with (
        patch("benchflow.task.Verifier") as verifier,
        patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
        patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
    ):
        verifier.return_value.verify = AsyncMock(side_effect=grade)
        trial._trajectory = [{"type": "agent_message", "text": "42"}]
        assert (await trial.soft_verify())[0] == {"reward": 0.0}
        ready = True
        trial._trajectory.extend([{"type": "user_message", "text": "retry"}, *tail])
        assert (await trial.soft_verify())[0] == {"reward": 0.0}

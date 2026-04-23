"""Tests for the User abstraction — BaseUser, PassthroughUser, FunctionUser, RoundResult."""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.trial import Role, Scene, Trial, TrialConfig, Turn
from benchflow.user import BaseUser, FunctionUser, PassthroughUser, RoundResult


# ── Unit tests for User types ──


class TestPassthroughUser:
    @pytest.mark.asyncio
    async def test_sends_instruction_once(self):
        user = PassthroughUser()
        result = await user.run(0, "Fix the bug")
        assert result == "Fix the bug"

    @pytest.mark.asyncio
    async def test_stops_after_first_round(self):
        user = PassthroughUser()
        await user.run(0, "Fix the bug")
        result = await user.run(1, "Fix the bug", RoundResult(round=0))
        assert result is None

    @pytest.mark.asyncio
    async def test_setup_is_noop(self):
        user = PassthroughUser()
        await user.setup("instruction", "solution")


class TestFunctionUser:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def my_fn(round: int, instruction: str, rr: RoundResult | None) -> str | None:
            if round == 0:
                return "terse: " + instruction[:10]
            return None

        user = FunctionUser(my_fn)
        assert await user.run(0, "Fix the authentication bug") == "terse: Fix the au"
        assert await user.run(1, "Fix the authentication bug", RoundResult(round=0)) is None

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def my_fn(round: int, instruction: str, rr: RoundResult | None) -> str | None:
            if round == 0:
                return instruction
            if rr and rr.rewards and rr.rewards.get("exact_match", 0) < 1.0:
                return "Try again, tests failed"
            return None

        user = FunctionUser(my_fn)
        assert await user.run(0, "task") == "task"

        failing = RoundResult(round=0, rewards={"exact_match": 0.0})
        assert await user.run(1, "task", failing) == "Try again, tests failed"

        passing = RoundResult(round=1, rewards={"exact_match": 1.0})
        assert await user.run(2, "task", passing) is None


class TestBaseUser:
    @pytest.mark.asyncio
    async def test_not_implemented(self):
        user = BaseUser()
        with pytest.raises(NotImplementedError):
            await user.run(0, "task")


# ── Integration tests for user loop in Trial ──


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnv:
    """Minimal sandbox mock for user loop tests."""

    def __init__(self) -> None:
        self._exec_log: list[str] = []

    async def exec(self, cmd: str, **kwargs) -> FakeExecResult:
        self._exec_log.append(cmd)
        if "cat /solution" in cmd:
            return FakeExecResult(stdout="gold answer here")
        if "rm -rf /solution" in cmd:
            return FakeExecResult()
        if "/logs/verifier" in cmd:
            return FakeExecResult()
        if "cat /logs/verifier" in cmd:
            return FakeExecResult(stdout="")
        return FakeExecResult()

    async def stop(self, **kwargs):
        pass


def _make_user_trial(
    user: BaseUser, max_rounds: int = 5, oracle: bool = False, tmp_path: Path | None = None,
) -> Trial:
    config = TrialConfig(
        task_path=Path("tasks/fake"),
        scenes=[Scene.single(agent="gemini", model="flash")],
        environment="docker",
        user=user,
        max_user_rounds=max_rounds,
        oracle_access=oracle,
    )
    trial = Trial(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task described in /app/instruction.md"]
    trial_dir = tmp_path or Path(tempfile.mkdtemp(prefix="benchflow-test-"))
    trial._trial_dir = trial_dir
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    trial._trial_paths = type("P", (), {"verifier_dir": verifier_dir})()
    trial._task = type("T", (), {
        "config": type("C", (), {
            "verifier": type("V", (), {
                "timeout_sec": 30,
                "env": {},
            })(),
            "agent": type("A", (), {"timeout_sec": 60})(),
        })(),
    })()
    trial._agent_cwd = "/app"
    return trial


class RecordingUser(BaseUser):
    """User that records calls and stops after a fixed number of rounds."""

    def __init__(self, max_rounds: int = 2, prompts: list[str] | None = None):
        self.setup_calls: list[tuple[str, str | None]] = []
        self.run_calls: list[tuple[int, str, RoundResult | None]] = []
        self._max = max_rounds
        self._prompts = prompts or ["Do the thing", "Try harder"]

    async def setup(self, instruction: str, solution: str | None = None) -> None:
        self.setup_calls.append((instruction, solution))

    async def run(
        self, round: int, instruction: str, round_result: RoundResult | None = None
    ) -> str | None:
        self.run_calls.append((round, instruction, round_result))
        if round >= self._max:
            return None
        return self._prompts[min(round, len(self._prompts) - 1)]


class TestUserLoop:
    @pytest.mark.asyncio
    async def test_user_loop_calls_setup_and_run(self):
        user = RecordingUser(max_rounds=1)
        trial = _make_user_trial(user, max_rounds=3)

        with patch.object(trial, "connect_as", new_callable=AsyncMock), \
             patch.object(trial, "execute", new_callable=AsyncMock, return_value=([], 0)), \
             patch.object(trial, "disconnect", new_callable=AsyncMock), \
             patch.object(trial, "soft_verify", new_callable=AsyncMock, return_value=({"exact_match": 1.0}, None, None)):

            await trial._run_user_loop()

        assert len(user.setup_calls) == 1
        assert user.setup_calls[0][0] == "Solve the task described in /app/instruction.md"
        assert len(user.run_calls) == 2  # round 0 → prompt, round 1 → None
        assert user.run_calls[0][0] == 0  # round number
        assert user.run_calls[1][0] == 1

    @pytest.mark.asyncio
    async def test_user_loop_passes_round_result(self):
        user = RecordingUser(max_rounds=2)
        trial = _make_user_trial(user, max_rounds=5)

        with patch.object(trial, "connect_as", new_callable=AsyncMock), \
             patch.object(trial, "execute", new_callable=AsyncMock, return_value=([], 0)), \
             patch.object(trial, "disconnect", new_callable=AsyncMock), \
             patch.object(trial, "soft_verify", new_callable=AsyncMock, return_value=({"exact_match": 0.5}, "1 failed", None)):

            await trial._run_user_loop()

        # First call has no round_result
        assert user.run_calls[0][2] is None
        # Second call has round_result from round 0
        rr = user.run_calls[1][2]
        assert isinstance(rr, RoundResult)
        assert rr.round == 0
        assert rr.rewards == {"exact_match": 0.5}
        assert rr.verifier_output == "1 failed"

    @pytest.mark.asyncio
    async def test_user_loop_respects_max_rounds(self):
        """User that never stops is capped by max_user_rounds."""
        def never_stop(r, instr, rr):
            return "keep going"

        user = FunctionUser(never_stop)
        trial = _make_user_trial(user, max_rounds=3)

        call_count = 0
        async def mock_execute(prompts=None):
            nonlocal call_count
            call_count += 1
            return [], 0

        with patch.object(trial, "connect_as", new_callable=AsyncMock), \
             patch.object(trial, "execute", side_effect=mock_execute), \
             patch.object(trial, "disconnect", new_callable=AsyncMock), \
             patch.object(trial, "soft_verify", new_callable=AsyncMock, return_value=(None, None, None)):

            await trial._run_user_loop()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_oracle_access(self):
        user = RecordingUser(max_rounds=0)
        trial = _make_user_trial(user, oracle=True)

        with patch.object(trial, "connect_as", new_callable=AsyncMock), \
             patch.object(trial, "execute", new_callable=AsyncMock, return_value=([], 0)), \
             patch.object(trial, "disconnect", new_callable=AsyncMock), \
             patch.object(trial, "soft_verify", new_callable=AsyncMock, return_value=(None, None, None)):

            await trial._run_user_loop()

        assert user.setup_calls[0][1] == "gold answer here"

    @pytest.mark.asyncio
    async def test_multi_role_raises(self):
        user = RecordingUser()
        config = TrialConfig(
            task_path=Path("tasks/fake"),
            scenes=[Scene(
                name="multi",
                roles=[Role("a", "gemini"), Role("b", "gemini")],
                turns=[Turn("a"), Turn("b")],
            )],
            user=user,
        )
        trial = Trial(config)
        trial._env = FakeEnv()
        trial._resolved_prompts = ["task"]

        with pytest.raises(ValueError, match="single-role"):
            await trial._run_user_loop()

    @pytest.mark.asyncio
    async def test_user_run_exception_stops_loop(self):
        class FailingUser(BaseUser):
            async def run(self, round, instruction, rr=None):
                raise RuntimeError("oops")

        trial = _make_user_trial(FailingUser())

        with patch.object(trial, "connect_as", new_callable=AsyncMock), \
             patch.object(trial, "execute", new_callable=AsyncMock, return_value=([], 0)), \
             patch.object(trial, "disconnect", new_callable=AsyncMock), \
             patch.object(trial, "soft_verify", new_callable=AsyncMock, return_value=(None, None, None)):

            await trial._run_user_loop()

        assert "user.run() failed" in trial._error

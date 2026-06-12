"""Tests for the User abstraction — BaseUser, PassthroughUser, FunctionUser, RoundResult."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.agents.protocol import AskUserRequest
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene, Turn
from benchflow.sandbox.user import (
    BaseUser,
    DocumentNudgeUser,
    FunctionUser,
    ModelDocumentNudgeUser,
    PassthroughUser,
    RoundResult,
)
from benchflow.task.verifier import VerifierResult

# Unit tests for User types


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


class TestFunctionUser:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def my_fn(round: int, instruction: str, rr: RoundResult | None) -> str | None:
            if round == 0:
                return "terse: " + instruction[:10]
            return None

        user = FunctionUser(my_fn)
        assert await user.run(0, "Fix the authentication bug") == "terse: Fix the au"
        assert (
            await user.run(1, "Fix the authentication bug", RoundResult(round=0))
            is None
        )

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def my_fn(
            round: int, instruction: str, rr: RoundResult | None
        ) -> str | None:
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


class TestDocumentNudgeUser:
    @pytest.mark.asyncio
    async def test_private_facts_reveal_only_after_targeted_question(self):
        user = DocumentNudgeUser(
            persona="Ask clarifying questions.",
            private_facts={"hidden_need": "Use the quarterly file."},
        )

        assert await user.run(0, "Solve safely.") == "Solve safely."
        vague = RoundResult(
            round=0,
            trajectory=[{"type": "agent_message", "text": "I need more info."}],
        )
        assert await user.run(1, "Solve safely.", vague) is None

        natural = RoundResult(
            round=1,
            trajectory=[{"type": "agent_message", "text": "Which file should I use?"}],
        )
        assert await user.run(2, "Solve safely.", natural) == (
            "Additional user detail for hidden need: Use the quarterly file."
        )

        targeted = RoundResult(
            round=2,
            trajectory=[
                {
                    "type": "agent_message",
                    "text": "What is the hidden need for this task?",
                }
            ],
        )
        assert await user.run(3, "Solve safely.", targeted) is None


class TestModelDocumentNudgeUser:
    @pytest.mark.asyncio
    async def test_model_user_receives_private_context_but_initial_prompt_is_public(
        self,
    ):
        calls: list[tuple[str, str]] = []

        async def call_model(model: str, prompt: str) -> str:
            calls.append((model, prompt))
            return "Please use the quarterly file."

        user = ModelDocumentNudgeUser(
            model="claude-haiku",
            persona="Only reveal facts after clarification.",
            private_facts={"hidden_need": "Use the quarterly file."},
            call_model=call_model,
        )

        assert await user.run(0, "Solve safely.") == "Solve safely."
        result = await user.run(
            1,
            "Solve safely.",
            RoundResult(
                round=0,
                rewards={"reward": 0.25},
                scene="shared-work",
                role="planner",
                handoff_from="planner",
                handoff_to="implementer",
                trajectory=[
                    {"type": "agent_message", "text": "Which file should I use?"}
                ],
            ),
        )

        assert result == "Please use the quarterly file."
        assert calls[0][0] == "claude-haiku"
        assert "Use the quarterly file." in calls[0][1]
        assert "Previous scene: shared-work" in calls[0][1]
        assert "Previous role: planner" in calls[0][1]
        assert "Previous handoff: planner -> implementer" in calls[0][1]
        assert "which file should i use?" in calls[0][1]

    @pytest.mark.asyncio
    async def test_model_user_stop_token_stops_loop(self):
        user = ModelDocumentNudgeUser(
            model="gemini-2.5-flash",
            private_facts={"hidden_need": "Use the quarterly file."},
            call_model=lambda _model, _prompt: "STOP",
        )

        assert await user.run(1, "Solve safely.", RoundResult(round=0)) is None


class TestBaseUser:
    @pytest.mark.asyncio
    async def test_not_implemented(self):
        user = BaseUser()
        with pytest.raises(NotImplementedError):
            await user.run(0, "task")


# Integration tests for user loop in Rollout


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
    user: BaseUser,
    max_rounds: int = 5,
    oracle: bool = False,
    tmp_path: Path | None = None,
) -> Rollout:
    config = RolloutConfig(
        task_path=Path("tasks/fake"),
        scenes=[Scene.single(agent="gemini", model="flash")],
        environment="docker",
        user=user,
        max_user_rounds=max_rounds,
        oracle_access=oracle,
    )
    trial = Rollout(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task described in /app/instruction.md"]
    rollout_dir = tmp_path or Path(tempfile.mkdtemp(prefix="benchflow-test-"))
    trial._rollout_dir = rollout_dir
    verifier_dir = rollout_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    trial._rollout_paths = type("P", (), {"verifier_dir": verifier_dir})()
    trial._task = type(
        "T",
        (),
        {
            "config": type(
                "C",
                (),
                {
                    "verifier": type(
                        "V",
                        (),
                        {
                            "timeout_sec": 30,
                            "env": {},
                        },
                    )(),
                    "agent": type("A", (), {"timeout_sec": 60})(),
                },
            )(),
        },
    )()
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
    async def test_document_confirmation_policy_installs_fail_closed_handler(self):
        user = DocumentNudgeUser(
            private_facts={"hidden_need": "Use the quarterly file."},
            confirmation_policy="human",
        )
        trial = _make_user_trial(user)

        installed = trial._install_document_confirmation_handler(user)

        assert installed is True
        assert trial._ask_user_handler is not None
        answer = await trial._ask_user_handler(
            AskUserRequest(
                prompt="May I change files?",
                options=["choice_a", "choice_b", "choice_c"],
                option_kinds={
                    "choice_a": "allow_once",
                    "choice_b": "reject",
                    "choice_c": "allow_always",
                },
                request_id="req-1",
            )
        )
        assert answer == "choice_b"

    @pytest.mark.asyncio
    async def test_document_confirmation_policy_preserves_explicit_handler(self):
        user = DocumentNudgeUser(
            private_facts={"hidden_need": "Use the quarterly file."},
            confirmation_policy="human",
        )
        trial = _make_user_trial(user)

        async def explicit_handler(_request: AskUserRequest) -> str:
            return "allow_once"

        trial.on_ask_user(explicit_handler)
        installed = trial._install_document_confirmation_handler(user)

        assert installed is False
        assert trial._ask_user_handler is explicit_handler

    @pytest.mark.asyncio
    async def test_document_confirmation_policy_handler_clears_on_user_loop_error(
        self,
    ):
        class FailingSetupUser(DocumentNudgeUser):
            async def setup(self, instruction: str, solution: str | None = None):
                raise RuntimeError("setup exploded")

        user = FailingSetupUser(
            private_facts={"hidden_need": "Use the quarterly file."},
            confirmation_policy="human",
        )
        trial = _make_user_trial(user)

        with pytest.raises(RuntimeError, match="setup exploded"):
            await trial._run_user_loop()

        assert trial._ask_user_handler is None

    @pytest.mark.asyncio
    async def test_user_loop_calls_setup_and_run(self):
        user = RecordingUser(max_rounds=1)
        trial = _make_user_trial(user, max_rounds=3)

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(
                trial, "execute", new_callable=AsyncMock, return_value=([], 0)
            ),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=({"exact_match": 1.0}, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert len(user.setup_calls) == 1
        assert (
            user.setup_calls[0][0] == "Solve the task described in /app/instruction.md"
        )
        assert len(user.run_calls) == 2  # round 0 → prompt, round 1 → None
        assert user.run_calls[0][0] == 0  # round number
        assert user.run_calls[1][0] == 1

    @pytest.mark.asyncio
    async def test_user_loop_passes_round_result(self):
        user = RecordingUser(max_rounds=2)
        trial = _make_user_trial(user, max_rounds=5)

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(
                trial, "execute", new_callable=AsyncMock, return_value=([], 0)
            ),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=({"exact_match": 0.5}, "1 failed", None),
            ),
        ):
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
    async def test_document_user_loop_runs_multi_scene_single_role_sequence(self):
        user = DocumentNudgeUser(
            persona="Reveal private facts only after clarification.",
            private_facts={"hidden_need": "Use the quarterly file."},
        )
        config = RolloutConfig(
            task_path=Path("tasks/fake"),
            scenes=[
                Scene(
                    name="plan",
                    roles=[Role("planner", "gemini")],
                    turns=[Turn("planner", "Plan the work.")],
                ),
                Scene(
                    name="implement",
                    roles=[Role("implementer", "gemini")],
                    turns=[Turn("implementer", "Apply the plan.")],
                ),
            ],
            user=user,
            max_user_rounds=2,
        )
        trial = Rollout(config)
        trial._env = FakeEnv()
        trial._resolved_prompts = ["Base instruction."]
        rollout_dir = Path(tempfile.mkdtemp(prefix="benchflow-test-"))
        trial._rollout_dir = rollout_dir
        verifier_dir = rollout_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        trial._rollout_paths = type("P", (), {"verifier_dir": verifier_dir})()

        executed_prompts: list[str] = []

        async def mock_execute(prompts=None):
            executed_prompts.extend(prompts or [])
            return [], 0

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock) as connect_as,
            patch.object(trial, "execute", side_effect=mock_execute),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=({"exact_match": 0.0}, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert [call.args[0].name for call in connect_as.await_args_list] == [
            "planner",
            "implementer",
        ]
        assert executed_prompts == ["Plan the work.", "Apply the plan."]

    @pytest.mark.asyncio
    async def test_document_user_loop_runs_sequential_team_handoff_scene(self):
        user = DocumentNudgeUser(
            persona="Reveal private facts only after clarification.",
            private_facts={"hidden_need": "Use the quarterly file."},
            handoff_kind="sequential-shared",
            handoff_team="build_review",
        )
        config = RolloutConfig(
            task_path=Path("tasks/fake"),
            scenes=[
                Scene(
                    name="shared-work",
                    roles=[Role("planner", "gemini"), Role("implementer", "gemini")],
                    turns=[
                        Turn("planner", "Plan the work."),
                        Turn("implementer", "Apply the plan."),
                    ],
                ),
            ],
            user=user,
            max_user_rounds=2,
        )
        trial = Rollout(config)
        trial._env = FakeEnv()
        trial._resolved_prompts = ["Base instruction."]
        rollout_dir = Path(tempfile.mkdtemp(prefix="benchflow-test-"))
        trial._rollout_dir = rollout_dir
        verifier_dir = rollout_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        trial._rollout_paths = type("P", (), {"verifier_dir": verifier_dir})()

        executed_prompts: list[str] = []

        async def mock_execute(prompts=None):
            executed_prompts.extend(prompts or [])
            return [], 0

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock) as connect_as,
            patch.object(trial, "execute", side_effect=mock_execute),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=({"exact_match": 0.0}, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert [call.args[0].name for call in connect_as.await_args_list] == [
            "planner",
            "implementer",
        ]
        assert executed_prompts == ["Plan the work.", "Apply the plan."]
        log_lines = (rollout_dir / "user_rounds.jsonl").read_text().splitlines()
        assert '"handoff_from": null' in log_lines[0]
        assert '"handoff_to": null' in log_lines[0]
        assert '"handoff_from": "planner"' in log_lines[1]
        assert '"handoff_to": "implementer"' in log_lines[1]

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

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(trial, "execute", side_effect=mock_execute),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_oracle_access(self):
        user = RecordingUser(max_rounds=0)
        trial = _make_user_trial(user, oracle=True)

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(
                trial, "execute", new_callable=AsyncMock, return_value=([], 0)
            ),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert user.setup_calls[0][1] == "gold answer here"

    @pytest.mark.asyncio
    async def test_multi_role_raises(self):
        user = RecordingUser()
        config = RolloutConfig(
            task_path=Path("tasks/fake"),
            scenes=[
                Scene(
                    name="multi",
                    roles=[Role("a", "gemini"), Role("b", "gemini")],
                    turns=[Turn("a"), Turn("b")],
                )
            ],
            user=user,
        )
        trial = Rollout(config)
        trial._env = FakeEnv()
        trial._resolved_prompts = ["task"]

        with pytest.raises(ValueError, match="exactly one role"):
            await trial._run_user_loop()

    @pytest.mark.asyncio
    async def test_user_run_exception_stops_loop(self):
        class FailingUser(BaseUser):
            async def run(self, round, instruction, rr=None):
                raise RuntimeError("oops")

        trial = _make_user_trial(FailingUser())

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(
                trial, "execute", new_callable=AsyncMock, return_value=([], 0)
            ),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
        ):
            await trial._run_user_loop()

        assert "user.run() failed" in trial._error

    @pytest.mark.asyncio
    async def test_explicit_stop_does_not_resurrect_user(self):
        """A classic user that stops (returns None) in the scene-step loop must
        not be re-invoked by the free-round loop.

        Counts user.run() invocations and agent rounds. The user returns a live
        prompt on any call after the first, so if the free-round loop wrongly
        re-invokes run(), the user is resurrected and extra agent rounds fire —
        which these counts catch. Regression guard for the break fall-through.
        """

        class StopThenResurrectUser(BaseUser):
            def __init__(self) -> None:
                self.run_calls = 0

            async def run(self, round, instruction, round_result=None):
                self.run_calls += 1
                if self.run_calls == 1:
                    return None  # explicit stop on the single scene step
                return "resurrected prompt"

        user = StopThenResurrectUser()
        trial = _make_user_trial(user, max_rounds=5)

        agent_rounds = 0

        async def mock_execute(prompts=None):
            nonlocal agent_rounds
            agent_rounds += 1
            return [], 0

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(trial, "execute", side_effect=mock_execute),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
        ):
            await trial._run_user_loop()

        # Stopped at the first (and only) scene step: run() called exactly once,
        # zero agent rounds executed, and no spurious error recorded.
        assert user.run_calls == 1
        assert agent_rounds == 0
        assert trial._error is None

    @pytest.mark.asyncio
    async def test_raise_terminates_loop_without_retry(self):
        """When user.run() raises in the scene-step loop, the loop terminates
        with the error set and must not retry run() or run further agent rounds.

        The user succeeds on a retry, so a fall-through into the free-round loop
        would execute extra rounds while self._error stays set — a half-script
        rollout reported as errored. The call/round counts make that visible.
        """

        class RaiseThenSucceedUser(BaseUser):
            def __init__(self) -> None:
                self.run_calls = 0

            async def run(self, round, instruction, round_result=None):
                self.run_calls += 1
                if self.run_calls == 1:
                    raise RuntimeError("transient user error")
                return "retry prompt"

        user = RaiseThenSucceedUser()
        trial = _make_user_trial(user, max_rounds=5)

        agent_rounds = 0

        async def mock_execute(prompts=None):
            nonlocal agent_rounds
            agent_rounds += 1
            return [], 0

        with (
            patch.object(trial, "connect_as", new_callable=AsyncMock),
            patch.object(trial, "execute", side_effect=mock_execute),
            patch.object(trial, "disconnect", new_callable=AsyncMock),
            patch.object(
                trial,
                "soft_verify",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
        ):
            await trial._run_user_loop()

        # The raise terminates the loop: run() called exactly once, no agent
        # rounds executed, and the error is recorded for round 0.
        assert user.run_calls == 1
        assert agent_rounds == 0
        assert trial._error is not None
        assert "user.run() failed at round 0" in trial._error


class TestSoftVerify:
    @pytest.mark.asyncio
    async def test_soft_verify_timeout(self):
        trial = _make_user_trial(PassthroughUser())

        with (
            patch("benchflow.task.Verifier") as MockVerifier,
            patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
            patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(side_effect=TimeoutError())

            rewards, output, error = await trial.soft_verify()

        assert rewards is None
        assert output is None
        assert "timed out" in error

    @pytest.mark.asyncio
    async def test_soft_verify_crash(self):
        trial = _make_user_trial(PassthroughUser())

        with (
            patch("benchflow.task.Verifier") as MockVerifier,
            patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
            patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(side_effect=RuntimeError("boom"))

            rewards, _output, error = await trial.soft_verify()

        assert rewards is None
        assert "crashed" in error
        assert "boom" in error

    @pytest.mark.asyncio
    async def test_soft_verify_success(self):
        trial = _make_user_trial(PassthroughUser())

        mock_result = type("VR", (), {"rewards": {"reward": 1.0, "exact_match": 1.0}})()

        with (
            patch("benchflow.task.Verifier") as MockVerifier,
            patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
            patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(return_value=mock_result)

            rewards, _output, error = await trial.soft_verify()

        assert rewards == {"reward": 1.0, "exact_match": 1.0}
        assert error is None

    @pytest.mark.asyncio
    async def test_soft_verify_returning_no_rewards_is_verifier_error(self):
        trial = _make_user_trial(PassthroughUser())

        with (
            patch("benchflow.task.Verifier") as MockVerifier,
            patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
            patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(return_value=VerifierResult(rewards=None))

            rewards, _output, error = await trial.soft_verify()

        assert rewards is None
        assert "soft verifier crashed" in error
        assert "verifier returned no rewards" in error

    @pytest.mark.asyncio
    async def test_soft_verify_returning_noncanonical_rewards_is_verifier_error(self):
        trial = _make_user_trial(PassthroughUser())

        with (
            patch("benchflow.task.Verifier") as MockVerifier,
            patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
            patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(return_value=VerifierResult(rewards={}))

            rewards, _output, error = await trial.soft_verify()

        assert rewards is None
        assert "soft verifier crashed" in error
        assert "numeric 'reward' or aggregate policy" in error

    @pytest.mark.asyncio
    async def test_soft_verify_runs_cleanup_cmd(self):
        trial = _make_user_trial(PassthroughUser())

        mock_result = type("VR", (), {"rewards": {"reward": 0.0}})()

        with (
            patch("benchflow.task.verifier.Verifier") as MockVerifier,
            patch(
                "benchflow.sandbox.lockdown._build_cleanup_cmd",
                return_value="echo cleanup_sentinel",
            ),
        ):
            mock_instance = MockVerifier.return_value
            mock_instance.verify = AsyncMock(return_value=mock_result)

            await trial.soft_verify()

        # Verify cleanup command was executed
        exec_log = trial._env._exec_log
        assert any("cleanup_sentinel" in cmd for cmd in exec_log)
        assert any("find /logs/verifier -mindepth 1" in cmd for cmd in exec_log)
        assert any(cmd == "mkdir -p /app" for cmd in exec_log)

    @pytest.mark.asyncio
    async def test_soft_verify_reports_cleanup_failure(self):
        trial = _make_user_trial(PassthroughUser())
        trial._env.exec = AsyncMock(
            return_value=FakeExecResult(stderr="Device or resource busy", return_code=1)
        )

        rewards, output, error = await trial.soft_verify()

        assert rewards is None
        assert output is None
        assert "Soft verifier setup failed" in error
        assert "Device or resource busy" in error

"""Tests for the first-class llm-judge verifier wired into ``Verifier`` (#270).

These tests exercise the integration between ``task.toml`` config
(``[verifier].type = "llm-judge"``) and ``Verifier.verify()``. All LLM
provider calls are mocked — no live API keys are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.rewards.builtins import JudgeScoringError
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier import (
    DownloadVerifierDirError,
    RewardFileNotFoundError,
    RubricNotFoundError,
    VerifierOutputParseError,
)

_MOCK_PASS = '```json\n{"verdict": "pass", "reasoning": "good"}\n```'
_MOCK_FAIL = '```json\n{"verdict": "fail", "reasoning": "bad"}\n```'


# ---------------------------------------------------------------------------
# Config parsing (#270): [verifier].type and [verifier.judge]
# ---------------------------------------------------------------------------


class TestVerifierConfig:
    def test_default_type_is_test_script(self) -> None:
        """#270: existing tasks keep the test-script verifier by default."""
        cfg = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        assert cfg.verifier.type == "test-script"

    def test_llm_judge_type(self) -> None:
        """#270: task.toml can opt into the llm-judge verifier."""
        toml = """\
version = "1.0"

[verifier]
type = "llm-judge"
timeout_sec = 600

[verifier.judge]
model = "claude-haiku-4-5"
rubric_path = "tests/rubric.json"
input_dir = "/app/output"
input_type = "deliverables"
"""
        cfg = TaskConfig.model_validate_toml(toml)
        assert cfg.verifier.type == "llm-judge"
        assert cfg.verifier.judge.model == "claude-haiku-4-5"
        assert cfg.verifier.judge.rubric_path == "tests/rubric.json"
        assert cfg.verifier.judge.input_dir == "/app/output"
        assert cfg.verifier.judge.input_type == "deliverables"

    def test_judge_defaults(self) -> None:
        """#270: judge config has sensible defaults when omitted."""
        cfg = TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier]\ntype = "llm-judge"\n'
        )
        assert cfg.verifier.judge.model == "claude-sonnet-4-6"
        assert cfg.verifier.judge.rubric_path == "tests/rubric.toml"
        assert cfg.verifier.judge.input_dir == "/app"

    def test_invalid_type_rejected(self) -> None:
        """#270: an unknown verifier type fails validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TaskConfig.model_validate_toml(
                'version = "1.0"\n[verifier]\ntype = "magic"\n'
            )


# ---------------------------------------------------------------------------
# Verifier.verify() — llm-judge branch
# ---------------------------------------------------------------------------


def _make_task(tmp_path: Path, toml: str, instruction: str = "Do the task.") -> object:
    """Build a lightweight task stub backed by a real task directory."""
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    task = MagicMock()
    task.task_dir = task_dir
    task.paths.task_dir = task_dir
    task.config = TaskConfig.model_validate_toml(toml)
    task.instruction = instruction
    return task


def _judge_toml(rubric_path: str = "tests/rubric.json") -> str:
    return f"""\
version = "1.0"

[verifier]
type = "llm-judge"

[verifier.judge]
model = "claude-haiku-4-5"
rubric_path = "{rubric_path}"
input_dir = "/app"
"""


def _make_sandbox(deliverables: dict[str, str]) -> MagicMock:
    """A sandbox stub whose download_dir drops the given files into dest."""
    sandbox = MagicMock()

    async def fake_download_dir(source_dir: str, target_dir: Path) -> None:
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in deliverables.items():
            (dest / name).write_text(content)

    sandbox.download_dir = AsyncMock(side_effect=fake_download_dir)
    return sandbox


class TestLLMJudgeVerifier:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_routes_to_llm_judge(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """#270: type=llm-judge routes verify() through the judge path."""
        mock_judge.return_value = _MOCK_PASS
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "Correct?"}]})
        )
        sandbox = _make_sandbox({"answer.txt": "the answer"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        verifier = Verifier(task, rollout_paths, sandbox)
        result = await verifier.verify()

        assert result.rewards == {"reward": 1.0}
        # never touches the test-script path
        sandbox.upload_dir.assert_not_called()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_partial_reward(self, mock_judge: AsyncMock, tmp_path: Path) -> None:
        """#270: judge produces partial rewards in [0, 1]."""
        mock_judge.side_effect = [_MOCK_PASS, _MOCK_FAIL]
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps(
                {
                    "criteria": [
                        {"id": "c1", "match_criteria": "A"},
                        {"id": "c2", "match_criteria": "B"},
                    ]
                }
            )
        )
        sandbox = _make_sandbox({"memo.md": "a memo"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards == {"reward": 0.5}

    @pytest.mark.parametrize(
        "judge_failure",
        [
            RuntimeError("provider down"),
            "not json at all",
        ],
    )
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_provider_error_is_verifier_error(
        self, mock_judge: AsyncMock, tmp_path: Path, judge_failure: object
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        if isinstance(judge_failure, Exception):
            mock_judge.side_effect = judge_failure
        else:
            mock_judge.return_value = judge_failure
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(JudgeScoringError, match="Judge error on criterion"):
            await Verifier(task, rollout_paths, sandbox).verify()
        assert not rollout_paths.reward_json_path.exists()

    @pytest.mark.parametrize(
        ("criterion", "judge_response"),
        [
            (
                {
                    "id": "c1",
                    "type": "numeric",
                    "min": 0,
                    "max": 10,
                    "match_criteria": "Score the answer.",
                },
                '```json\n{"score": NaN, "reasoning": "bad"}\n```',
            ),
            (
                {
                    "id": "c1",
                    "type": "numeric",
                    "min": 0,
                    "max": 10,
                    "match_criteria": "Score the answer.",
                },
                '```json\n{"score": Infinity, "reasoning": "bad"}\n```',
            ),
            (
                {
                    "id": "c1",
                    "type": "likert",
                    "points": 5,
                    "match_criteria": "Score the answer.",
                },
                '```json\n{"score": "nan", "reasoning": "bad"}\n```',
            ),
        ],
    )
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_rejects_nonfinite_scores_without_reward_output(
        self,
        mock_judge: AsyncMock,
        tmp_path: Path,
        criterion: dict,
        judge_response: str,
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        mock_judge.return_value = judge_response
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [criterion]})
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(JudgeScoringError, match="Judge error on criterion"):
            await Verifier(task, rollout_paths, sandbox).verify()
        assert not rollout_paths.reward_json_path.exists()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_writes_reward_json(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """#270: judge reward is persisted to reward.json like test-script."""
        mock_judge.return_value = _MOCK_PASS
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        await Verifier(task, rollout_paths, sandbox).verify()

        reward_json = rollout_paths.reward_json_path
        assert reward_json.exists()
        assert json.loads(reward_json.read_text())["reward"] == 1.0

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_toml_rubric_supported(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """#270: a native rubric.toml works as the judge rubric too."""
        mock_judge.return_value = _MOCK_PASS
        task = _make_task(tmp_path, _judge_toml(rubric_path="tests/rubric.toml"))
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.toml").write_text(
            '[[criterion]]\ndescription = "Is it good?"\n'
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards == {"reward": 1.0}

    @pytest.mark.asyncio
    async def test_missing_rubric_raises(self, tmp_path: Path) -> None:
        """#270: a misconfigured rubric path fails loudly."""
        task = _make_task(tmp_path, _judge_toml())
        sandbox = _make_sandbox({})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(RubricNotFoundError, match="rubric not found"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_env_threaded_into_call_judge(
        self, mock_judge: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR #314 follow-up: [verifier.env] credentials are passed explicitly
        into ``call_judge`` as the ``env`` kwarg.

        The pre-fix ``_scoped_env`` context manager mutated the process-global
        ``os.environ``, which is not concurrency-safe: ``evaluation.py`` runs
        verifications via ``asyncio.gather`` (concurrency > 1), so two judge
        runs could race on the shared env. Credentials are now threaded
        through as a parameter instead of touching ``os.environ`` at all.
        """
        import os

        monkeypatch.setenv("MY_JUDGE_KEY", "secret-123")
        monkeypatch.delenv("RESOLVED_KEY", raising=False)
        mock_judge.return_value = _MOCK_PASS

        toml = _judge_toml() + '\n[verifier.env]\nRESOLVED_KEY = "${MY_JUDGE_KEY}"\n'
        task = _make_task(tmp_path, toml)
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        await Verifier(task, rollout_paths, sandbox).verify()

        # The resolved credential reaches call_judge as the ``env`` kwarg ...
        mock_judge.assert_awaited()
        assert mock_judge.await_args is not None
        assert mock_judge.await_args.kwargs["env"] == {"RESOLVED_KEY": "secret-123"}
        # ... and the process-global env is never mutated.
        assert os.environ.get("RESOLVED_KEY") is None

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_env_does_not_mutate_os_environ(
        self, mock_judge: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR #314 follow-up: a [verifier.env] key that already exists in the
        process env is never overwritten — the verifier no longer touches
        ``os.environ`` (the pre-fix ``_scoped_env`` capture/restore is gone)."""
        import os

        mock_judge.return_value = _MOCK_PASS
        monkeypatch.setenv("MY_JUDGE_KEY", "secret-123")
        monkeypatch.setenv("RESOLVED_KEY", "preexisting")

        toml = _judge_toml() + '\n[verifier.env]\nRESOLVED_KEY = "${MY_JUDGE_KEY}"\n'
        task = _make_task(tmp_path, toml)
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        await Verifier(task, rollout_paths, sandbox).verify()
        # os.environ is left exactly as it was — value never overwritten.
        assert os.environ.get("RESOLVED_KEY") == "preexisting"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_model_override_reaches_call_judge(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """#270: the [verifier.judge].model declared in task.toml is the model
        actually passed to call_judge."""
        mock_judge.return_value = _MOCK_PASS
        task = _make_task(
            tmp_path,
            _judge_toml().replace("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        )
        (task.task_dir / "tests").mkdir()
        # Rubric declares a *different* model — the task.toml override must win.
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps(
                {
                    "judge": {"model": "claude-sonnet-4-6"},
                    "criteria": [{"id": "c1", "match_criteria": "ok"}],
                }
            )
        )
        sandbox = _make_sandbox({"out.txt": "output"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        await Verifier(task, rollout_paths, sandbox).verify()

        mock_judge.assert_awaited()
        assert mock_judge.await_args is not None
        assert mock_judge.await_args.args[0] == "claude-haiku-4-5-20251001"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_missing_provider_sdk_surfaces_as_verifier_error(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A missing judge SDK must surface as a verifier *error*, not a reward.

        ``call_judge`` raises ``JudgeEnvironmentError`` when no provider SDK is
        installed. The verifier must let it propagate so the run is marked
        errored — recording reward 0.0 would be indistinguishable from a
        genuine judge verdict of fail.
        """
        from benchflow.rewards.llm import JudgeEnvironmentError

        mock_judge.side_effect = JudgeEnvironmentError(
            "No LLM provider SDK is installed for model claude-haiku-4-5"
        )
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = _make_sandbox({"answer.txt": "the answer"})
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        verifier = Verifier(task, rollout_paths, sandbox)
        with pytest.raises(JudgeEnvironmentError):
            await verifier.verify()

        # No reward was written — a missing SDK is not a score.
        assert not rollout_paths.reward_json_path.exists()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_download_failure_is_verifier_error(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d.

        A sandbox download failure is verifier infrastructure failure, not an
        agent miss that should be graded against an empty deliverables dir.
        """
        mock_judge.return_value = _MOCK_FAIL
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        sandbox = MagicMock()
        sandbox.download_dir = AsyncMock(side_effect=RuntimeError("network down"))
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(DownloadVerifierDirError, match="llm-judge input"):
            await Verifier(task, rollout_paths, sandbox).verify()
        mock_judge.assert_not_awaited()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_download_clears_stale_deliverables_before_judging(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        mock_judge.return_value = _MOCK_FAIL
        task = _make_task(tmp_path, _judge_toml())
        (task.task_dir / "tests").mkdir()
        (task.task_dir / "tests" / "rubric.json").write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        stale_deliverables = rollout_paths.verifier_dir / "deliverables"
        stale_deliverables.mkdir()
        (stale_deliverables / "answer.txt").write_text("stale passing answer")

        sandbox = MagicMock()

        async def empty_download(source_dir: str, target_dir: Path) -> None:
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        sandbox.download_dir = AsyncMock(side_effect=empty_download)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 0.0}
        mock_judge.assert_awaited()
        prompt = mock_judge.await_args.args[1]
        assert "stale passing answer" not in prompt


# ---------------------------------------------------------------------------
# Verifier.verify() — test-script branch stays the default (regression)
# ---------------------------------------------------------------------------


class TestTestScriptStillDefault:
    @pytest.mark.asyncio
    async def test_default_runs_test_script(self, tmp_path: Path) -> None:
        """#270: tasks without type still take the test-script path."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\ntimeout_sec = 123\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_current_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("0.75")
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_current_reward)

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards == {"reward": 0.75}
        sandbox.upload_dir.assert_called_once()
        assert sandbox.exec.await_args_list[-1].kwargs["timeout_sec"] == 123

    @pytest.mark.asyncio
    async def test_nonzero_test_script_without_reward_is_verifier_error(
        self, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.exec = AsyncMock(
            side_effect=[
                MagicMock(return_code=0, stdout=""),
                MagicMock(return_code=7, stdout="boom"),
            ]
        )
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(RewardFileNotFoundError, match="verifier exited with rc=7"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.parametrize("reward_name", ["reward.txt", "reward.json"])
    @pytest.mark.asyncio
    async def test_non_mounted_verifier_ignores_stale_local_reward_files(
        self, tmp_path: Path, reward_name: str
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.exec = AsyncMock(
            side_effect=[
                MagicMock(return_code=0, stdout=""),
                MagicMock(return_code=0, stdout=""),
                MagicMock(return_code=0, stdout=""),
            ]
        )
        sandbox.download_dir = AsyncMock()
        sandbox.is_mounted = False

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        if reward_name == "reward.txt":
            rollout_paths.reward_text_path.write_text("1.0")
        else:
            rollout_paths.reward_json_path.write_text(json.dumps({"reward": 1.0}))

        with pytest.raises(RewardFileNotFoundError, match="No reward file found"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_non_mounted_verifier_clears_stale_remote_reward_files(
        self, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        remote_rewards = {"reward.txt": "1.0"}
        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = False

        async def fake_exec(command: str, **_kwargs: object) -> MagicMock:
            if "rm -rf /logs/verifier" in command:
                remote_rewards.clear()
            return MagicMock(return_code=0, stdout="")

        async def fake_download_dir(
            source_dir: str, target_dir: Path, **_kwargs: object
        ) -> None:
            del source_dir
            dest = Path(target_dir)
            dest.mkdir(parents=True, exist_ok=True)
            for name, content in remote_rewards.items():
                (dest / name).write_text(content)

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        sandbox.download_dir = AsyncMock(side_effect=fake_download_dir)

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(RewardFileNotFoundError, match="No reward file found"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_nonzero_test_script_cannot_turn_reward_file_into_pass(
        self, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_failed_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("1.0")
            return MagicMock(return_code=1, stdout="boom")

        sandbox.exec = AsyncMock(side_effect=exec_failed_reward)

        with pytest.raises(VerifierOutputParseError, match="rc=1"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_reward_json_requires_canonical_reward_key(
        self, tmp_path: Path
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_malformed_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_json_path.write_text(json.dumps({"score": 1.0}))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_malformed_reward)

        with pytest.raises(VerifierOutputParseError, match="missing numeric 'reward'"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_reward_json_preserves_rubric_process_rewards(
        self, tmp_path: Path
    ) -> None:
        """Guards rich reward.json output from becoming verifier infrastructure failure."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        payload = {
            "reward": 0.75,
            "rubric": [
                {"name": "file_exists", "score": 1.0, "weight": 1.0},
                {"name": "content_correct", "score": 0.5, "weight": 1.0},
            ],
        }

        async def exec_rubric_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_json_path.write_text(json.dumps(payload))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_rubric_reward)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == payload

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"reward": float("nan")}, "missing numeric 'reward'"),
            ({"reward": float("inf")}, "missing numeric 'reward'"),
            ({"reward": True}, "missing numeric 'reward'"),
            ({"reward": 1.2}, "missing numeric 'reward'"),
            ({"reward": 0.5, "extra": float("nan")}, "invalid reward value"),
        ],
    )
    @pytest.mark.asyncio
    async def test_reward_json_rejects_invalid_reward_values(
        self, tmp_path: Path, payload: dict, match: str
    ) -> None:
        """Guards the reward-output regression on v0.5-integration@ffef85d."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_invalid_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_json_path.write_text(json.dumps(payload))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_invalid_reward)

        with pytest.raises(VerifierOutputParseError, match=match):
            await Verifier(task, rollout_paths, sandbox).verify()

"""Tests for the first-class llm-judge verifier wired into ``Verifier`` (#270).

These tests exercise the integration between ``task.toml`` config
(``[verifier].type = "llm-judge"``) and ``Verifier.verify()``. All LLM
provider calls are mocked — no live API keys are required.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.adapters import write_ors_tool_outputs_jsonl
from benchflow.rewards.builtins import JudgeScoringError
from benchflow.task import RolloutPaths, Task, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier import (
    DownloadVerifierDirError,
    RewardFileNotFoundError,
    RubricNotFoundError,
    VerifierOutputParseError,
)

_MOCK_PASS = '```json\n{"verdict": "pass", "reasoning": "good"}\n```'
_MOCK_FAIL = '```json\n{"verdict": "fail", "reasoning": "bad"}\n```'
DOGFOOD_ORS_TASK = Path(
    "docs/examples/task-standard/benchflow-wanted-features/ors-episode-reward-contract"
)
DOGFOOD_REWARD_KIT_TASK = Path(
    "docs/examples/task-standard/benchflow-wanted-features/"
    "verifier-package-reward-contract"
)


# Config parsing (#270): [verifier].type and [verifier.judge]


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


# Verifier.verify() — llm-judge branch


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
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["source"] == "llm-judge"
        assert details["aggregate"] == {"reward": 1.0, "method": "mean"}
        assert details["criteria"][0]["source"] == "criterion:c1"
        assert details["criteria"][0]["reward"] == 1.0
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


# Verifier.verify() — test-script branch stays the default (regression)


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
    async def test_verifier_document_selects_script_command(
        self, tmp_path: Path
    ) -> None:
        """verifier/verifier.md script strategy is the executable verifier."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\ntimeout_sec = 123\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "custom-check.sh").write_text("#!/usr/bin/env bash\n")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./custom-check.sh --strict
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "legacy-test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_selected_command(command: str, **_kwargs: object) -> MagicMock:
            if "./custom-check.sh --strict" in command:
                rollout_paths.reward_text_path.write_text("0.6")
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_selected_command)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 0.6}
        sandbox.upload_dir.assert_awaited_once_with(
            source_dir=verifier_dir,
            target_dir="/verifier",
            service="main",
        )
        assert (
            sandbox.exec.await_args_list[-1].kwargs["command"]
            == "cd /verifier && ./custom-check.sh --strict > "
            "/logs/verifier/test-stdout.txt 2>&1"
        )

    @pytest.mark.asyncio
    async def test_verifier_document_reward_kit_missing_runner_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """Reward Kit declarations need an explicit verifier-scoped runner."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "test.sh").write_text("#!/usr/bin/env bash\n")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit/
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.exec = AsyncMock()

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        with pytest.raises(VerifierOutputParseError, match="expected runner"):
            await Verifier(task, rollout_paths, sandbox).verify()
        sandbox.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_verifier_document_selects_reward_kit_strategy(
        self, tmp_path: Path
    ) -> None:
        """A reward-kit strategy runs its package runner in verifier scope."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        reward_kit = verifier_dir / "reward_kit"
        reward_kit.mkdir(parents=True)
        (reward_kit / "reward.py").write_text(
            "import json, os\n"
            "path = os.environ['BENCHFLOW_REWARD_JSON']\n"
            "details = os.environ['BENCHFLOW_REWARD_DETAILS_JSON']\n"
            "open(path, 'w').write(json.dumps({'reward': 0.7, 'metadata': {'source': 'reward-kit'}}))\n"
            "open(details, 'w').write(json.dumps({'criteria': [{'id': 'contract', 'score': 0.7}]}))\n"
        )
        rubric_dir = verifier_dir / "rubrics"
        rubric_dir.mkdir()
        (rubric_dir / "verifier.toml").write_text(
            '[[criteria]]\nid = "contract"\nmatch_criteria = "writes reward"\n'
        )
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit/
      criteria: rubrics/verifier.toml
  outputs:
    aggregate_policy:
      field: reward
      method: weighted_sum
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_reward_kit(command: str, **kwargs: object) -> MagicMock:
            assert command == (
                "cd /verifier && python reward_kit/reward.py > "
                "/logs/verifier/test-stdout.txt 2>&1"
            )
            env = kwargs["env"]
            assert env["BENCHFLOW_REWARD_KIT_ROOT"] == "/verifier/reward_kit"
            assert env["BENCHFLOW_REWARD_KIT_CRITERIA"] == (
                "/verifier/rubrics/verifier.toml"
            )
            assert env["BENCHFLOW_REWARD_KIT_MANIFEST"] == (
                "/logs/verifier/reward-kit-manifest.json"
            )
            manifest = json.loads(rollout_paths.reward_kit_manifest_path.read_text())
            assert manifest["version"] == "benchflow.reward-kit.v1"
            assert manifest["verifier"]["name"] == "verifier"
            assert manifest["strategy"]["name"] == "rewardkit"
            assert manifest["strategy"]["type"] == "reward-kit"
            assert manifest["strategy"]["root"] == "reward_kit"
            assert manifest["strategy"]["criteria"] == "rubrics/verifier.toml"
            assert manifest["paths"]["reward_json"] == "/logs/verifier/reward.json"
            assert manifest["paths"]["reward_details_json"] == (
                "/logs/verifier/reward-details.json"
            )
            assert manifest["rubric"] == {}
            assert manifest["criteria_policy"] == {
                "field": "reward",
                "method": "weighted_mean",
                "threshold": 0.7,
                "criteria": ["contract"],
                "weights": {"contract": 1.0},
            }
            assert manifest["outputs"]["aggregate_policy"] == {
                "field": "reward",
                "method": "weighted_sum",
            }
            rollout_paths.reward_json_path.write_text(
                json.dumps(
                    {
                        "metrics": {"contract": 0.7},
                        "metadata": {"source": "reward-kit"},
                    }
                )
            )
            rollout_paths.reward_details_json_path.write_text(
                json.dumps({"criteria": [{"id": "contract", "score": 0.7}]})
            )
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_reward_kit)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {
            "reward": 0.7,
            "metrics": {"contract": 0.7},
            "metadata": {"source": "reward-kit"},
        }
        sandbox.upload_dir.assert_awaited_once_with(
            source_dir=verifier_dir,
            target_dir="/verifier",
            service="main",
        )
        assert json.loads(rollout_paths.reward_details_json_path.read_text()) == {
            "criteria": [{"id": "contract", "score": 0.7}]
        }

    @pytest.mark.asyncio
    async def test_reward_kit_recovers_reward_when_dir_download_fails(
        self, tmp_path: Path
    ) -> None:
        """Guards private PR #1's Reward Kit Daytona verifier export regression."""
        task = _make_task(
            tmp_path,
            'version = "1.0"\n[verifier]\n',
        )
        verifier_dir = task.task_dir / "verifier"
        reward_kit = verifier_dir / "reward_kit"
        reward_kit.mkdir(parents=True)
        (reward_kit / "reward.py").write_text("print('reward kit')\n")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit/
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = False

        async def exec_reward_kit(command: str, **kwargs: object) -> MagicMock:
            if "mkdir -p /logs/verifier" in command:
                return MagicMock(return_code=0, stdout="")
            if command.startswith("printf "):
                return MagicMock(return_code=0, stdout="")
            assert "python reward_kit/reward.py" in command
            return MagicMock(return_code=0, stdout="")

        async def download_file(source: str, target: str | Path) -> None:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            if source.endswith("/reward.txt"):
                path.write_text("0.25")
            elif source.endswith("/test-stdout.txt"):
                path.write_text("reward kit output")
            elif source.endswith("/reward-kit-manifest.json"):
                path.write_text("{}")
            else:
                raise FileNotFoundError(source)

        sandbox.exec = AsyncMock(side_effect=exec_reward_kit)
        sandbox.download_dir = AsyncMock(side_effect=RuntimeError("export failed"))
        sandbox.download_file = AsyncMock(side_effect=download_file)
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 0.25}
        sandbox.download_dir.assert_awaited_once()
        assert rollout_paths.reward_text_path.read_text() == "0.25"

    @pytest.mark.asyncio
    async def test_dogfood_reward_kit_runner_uses_declared_criteria_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real Reward Kit dogfood runner emits metrics; BenchFlow aggregates."""
        task_dir = tmp_path / "rewardkit-dogfood"
        shutil.copytree(DOGFOOD_REWARD_KIT_TASK, task_dir)
        verifier_doc = task_dir / "verifier" / "verifier.md"
        verifier_doc.write_text(
            verifier_doc.read_text().replace(
                "default_strategy: deterministic",
                "default_strategy: rewardkit",
            )
        )
        task = Task(task_dir)
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        async def exec_reward_kit(command: str, **kwargs: object) -> MagicMock:
            assert "python reward_kit/reward.py" in command
            env = dict(kwargs["env"])
            path_map = {
                "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
                "BENCHFLOW_REWARD_KIT_ROOT": str(task_dir / "verifier" / "reward_kit"),
                "BENCHFLOW_REWARD_KIT_CRITERIA": str(
                    task_dir / "verifier" / "rubrics" / "verifier.toml"
                ),
                "BENCHFLOW_REWARD_TEXT": str(rollout_paths.reward_text_path),
                "BENCHFLOW_REWARD_JSON": str(rollout_paths.reward_json_path),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(
                    rollout_paths.reward_details_json_path
                ),
                "BENCHFLOW_REWARD_KIT_MANIFEST": str(
                    rollout_paths.reward_kit_manifest_path
                ),
            }
            for key, value in {**env, **path_map}.items():
                monkeypatch.setenv(key, str(value))
            runpy.run_path(
                str(task_dir / "verifier" / "reward_kit" / "reward.py"),
                run_name="__main__",
            )
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_reward_kit)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards is not None
        assert result.rewards["reward"] == pytest.approx(0.925)
        assert result.rewards["metrics"]["compatibility"] == 0.5
        manifest = json.loads(rollout_paths.reward_kit_manifest_path.read_text())
        assert manifest["criteria_policy"]["criteria"] == [
            "verifier_document",
            "reward_contract",
            "judge_isolation",
            "compatibility",
        ]
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["aggregate"]["weights"]["compatibility"] == 0.15

    @pytest.mark.asyncio
    async def test_reward_kit_declared_criteria_reject_reward_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Declared Reward Kit criteria recompute and verify runner rewards."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        reward_kit = verifier_dir / "reward_kit"
        reward_kit.mkdir(parents=True)
        (reward_kit / "reward.py").write_text("print('runner')\n")
        rubric_dir = verifier_dir / "rubrics"
        rubric_dir.mkdir()
        (rubric_dir / "verifier.toml").write_text(
            '[[criteria]]\nid = "contract"\nmatch_criteria = "writes reward"\n'
        )
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit/
      criteria: rubrics/verifier.toml
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_reward_kit(*_args: object, **_kwargs: object) -> MagicMock:
            rollout_paths.reward_json_path.write_text(
                json.dumps({"reward": 1.0, "metrics": {"contract": 0.7}})
            )
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_reward_kit)

        with pytest.raises(VerifierOutputParseError, match="criteria aggregate"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_reward_kit_declared_criteria_reject_metric_key_drift(
        self, tmp_path: Path
    ) -> None:
        """Reward Kit metric ids must exactly match declared criteria ids."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        reward_kit = verifier_dir / "reward_kit"
        reward_kit.mkdir(parents=True)
        (reward_kit / "reward.py").write_text("print('runner')\n")
        rubric_dir = verifier_dir / "rubrics"
        rubric_dir.mkdir()
        (rubric_dir / "verifier.toml").write_text(
            '[[criteria]]\nid = "contract"\nmatch_criteria = "writes reward"\n'
        )
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit/
      criteria: rubrics/verifier.toml
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_reward_kit(*_args: object, **_kwargs: object) -> MagicMock:
            rollout_paths.reward_json_path.write_text(
                json.dumps({"metrics": {"contract": 0.7, "extra": 1.0}})
            )
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_reward_kit)

        with pytest.raises(VerifierOutputParseError, match="extra metrics: extra"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_verifier_document_selects_agent_judge_strategy(
        self, mock_call_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """An agent-judge strategy runs in verifier scope over declared inputs."""
        mock_call_judge.return_value = (
            '```json\n{"score": 0.8, "reasoning": "declared evidence is good"}\n```'
        )
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: judge
  strategies:
    judge:
      type: agent-judge
      role: verifier_judge
      model: gpt-5.4
      inputs: [trajectory/acp_trajectory.jsonl]
      isolation: verifier-only
---

## role:verifier_judge

Judge only declared trajectory evidence.
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.download_file = AsyncMock()

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        trajectory_dir = rollout_paths.rollout_dir / "trajectory"
        trajectory_dir.mkdir()
        (trajectory_dir / "acp_trajectory.jsonl").write_text(
            '{"type":"agent_message","content":"implemented the change"}\n'
        )

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {
            "reward": 0.8,
            "metadata": {
                "source": "agent-judge",
                "strategy": "judge",
                "role": "verifier_judge",
                "model": "gpt-5.4",
            },
        }
        sandbox.upload_dir.assert_not_called()
        sandbox.download_file.assert_not_awaited()
        assert mock_call_judge.await_args.args[0] == "gpt-5.4"
        prompt = mock_call_judge.await_args.args[1]
        assert "Judge only declared trajectory evidence" in prompt
        assert "trajectory/acp_trajectory.jsonl" in prompt
        assert "implemented the change" in prompt
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["source"] == "agent-judge"
        assert details["inputs"] == [
            {
                "path": "trajectory/acp_trajectory.jsonl",
                "chars": 60,
                "truncated": False,
            }
        ]

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_agent_judge_downloads_declared_sandbox_input(
        self, mock_call_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Absolute declared inputs are downloaded instead of widening scope."""
        mock_call_judge.return_value = (
            '```json\n{"verdict": "pass", "reasoning": "patch is acceptable"}\n```'
        )
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: judge
  strategies:
    judge:
      type: agent-judge
      role: verifier_judge
      inputs: [/logs/artifacts/diff.patch]
      isolation: verifier-only
---

## role:verifier_judge

Review the patch artifact only.
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()

        async def fake_download_file(source_path: str, target_path: Path) -> None:
            assert source_path == "/logs/artifacts/diff.patch"
            Path(target_path).write_text("diff --git a/app.py b/app.py\n")

        sandbox.download_file = AsyncMock(side_effect=fake_download_file)

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards["reward"] == 1.0
        sandbox.download_file.assert_awaited_once()
        prompt = mock_call_judge.await_args.args[1]
        assert "/logs/artifacts/diff.patch" in prompt
        assert "diff --git" in prompt

    @pytest.mark.asyncio
    async def test_verifier_document_selects_ors_episode_strategy(
        self, tmp_path: Path
    ) -> None:
        """ORS episode strategies normalize declared reward evidence."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: ors
  strategies:
    ors:
      type: ors-episode
      inputs: [trajectory/ors-rewards.jsonl]
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.download_file = AsyncMock()

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        trajectory_dir = rollout_paths.rollout_dir / "trajectory"
        trajectory_dir.mkdir()
        (trajectory_dir / "ors-rewards.jsonl").write_text(
            json.dumps(
                {
                    "type": "dense",
                    "reward": 0.2,
                    "source": "step-check",
                    "step": 1,
                    "space": "action",
                    "granularity": "step",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "terminal",
                    "reward": 0.9,
                    "source": "final-check",
                    "space": "output",
                    "granularity": "terminal",
                    "finished": True,
                }
            )
            + "\n"
        )

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards is not None
        assert result.rewards["reward"] == 0.9
        assert result.rewards["metadata"]["source"] == "ors-episode"
        assert result.rewards["metadata"]["ors"]["metadata"]["items"] == {"ors": 0.9}
        assert [
            event["space"]
            for event in result.rewards["metadata"]["ors"]["metadata"]["events"]
        ] == ["action", "output"]
        sandbox.upload_dir.assert_not_called()
        sandbox.download_file.assert_not_awaited()
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["source"] == "ors-episode"
        assert details["inputs"][0]["records"] == 2
        assert details["aggregate"] == {"reward": 0.9, "method": "ors-terminal"}

    @pytest.mark.asyncio
    async def test_ors_episode_downloads_absolute_declared_input(
        self, tmp_path: Path
    ) -> None:
        """Absolute ORS inputs are explicitly downloaded into verifier scope."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: ors
  strategies:
    ors:
      type: ors-episode
      inputs: [/logs/artifacts/ors-response.json]
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()

        async def fake_download_file(source_path: str, target_path: Path) -> None:
            assert source_path == "/logs/artifacts/ors-response.json"
            Path(target_path).write_text(
                json.dumps(
                    {
                        "reward": 0.6,
                        "is_valid": True,
                        "metadata": {
                            "items": {"episode": 0.6},
                            "events": [],
                            "space": "output",
                            "granularity": "terminal",
                        },
                    }
                )
            )

        sandbox.download_file = AsyncMock(side_effect=fake_download_file)

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards is not None
        assert result.rewards["reward"] == 0.6
        assert result.rewards["metadata"]["ors"]["metadata"]["items"] == {
            "episode": 0.6
        }
        sandbox.download_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ors_episode_without_terminal_reward_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """ORS evidence without a terminal aggregate is verifier-invalid."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: ors
  strategies:
    ors:
      type: ors-episode
      inputs: [trajectory/ors-rewards.jsonl]
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.download_file = AsyncMock()

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        trajectory_dir = rollout_paths.rollout_dir / "trajectory"
        trajectory_dir.mkdir()
        (trajectory_dir / "ors-rewards.jsonl").write_text(
            json.dumps(
                {
                    "type": "dense",
                    "reward": 0.2,
                    "source": "step-check",
                    "step": 1,
                    "space": "action",
                    "granularity": "step",
                }
            )
            + "\n"
        )

        with pytest.raises(VerifierOutputParseError, match="terminal reward"):
            await Verifier(task, rollout_paths, sandbox).verify()
        assert not rollout_paths.reward_json_path.exists()

    @pytest.mark.asyncio
    async def test_dogfood_ors_episode_strategy_executes_from_rollout_evidence(
        self, tmp_path: Path
    ) -> None:
        """The real ORS dogfood package runs without a legacy test.sh."""
        task = Task(DOGFOOD_ORS_TASK)
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        trajectory_dir = rollout_paths.rollout_dir / "trajectory"
        trajectory_dir.mkdir()
        (trajectory_dir / "ors-rewards.jsonl").write_text(
            (
                DOGFOOD_ORS_TASK / "verifier" / "fixtures" / "ors-rewards.example.jsonl"
            ).read_text()
        )
        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.download_file = AsyncMock()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards is not None
        assert result.rewards["reward"] == 0.88
        assert result.rewards["metadata"]["source"] == "ors-episode"
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["source"] == "ors-episode"
        assert details["inputs"] == [
            {
                "path": "trajectory/ors-rewards.jsonl",
                "local_path": str(trajectory_dir / "ors-rewards.jsonl"),
                "records": 2,
            }
        ]
        sandbox.upload_dir.assert_not_called()
        sandbox.download_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dogfood_ors_episode_strategy_executes_from_tool_outputs(
        self, tmp_path: Path
    ) -> None:
        """The real ORS dogfood package can use runtime tool-output evidence."""
        task = Task(DOGFOOD_ORS_TASK)
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        fixture_path = (
            DOGFOOD_ORS_TASK
            / "verifier"
            / "fixtures"
            / "ors-tool-outputs.example.jsonl"
        )
        tool_outputs = [
            json.loads(line) for line in fixture_path.read_text().splitlines()
        ]
        evidence_path = rollout_paths.rollout_dir / "trajectory" / "ors-rewards.jsonl"
        records = write_ors_tool_outputs_jsonl(tool_outputs, evidence_path)
        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.download_file = AsyncMock()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert records[-1]["type"] == "terminal"
        assert result.rewards is not None
        assert result.rewards["reward"] == 0.88
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["inputs"][0]["records"] == 2
        sandbox.upload_dir.assert_not_called()
        sandbox.download_file.assert_not_awaited()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_verifier_document_selects_llm_judge_strategy(
        self, mock_call_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """verifier.md llm-judge can own rubric, model, input, and context."""
        mock_call_judge.return_value = _MOCK_PASS
        task = _make_task(
            tmp_path,
            """\
version = "1.0"

[verifier]

[verifier.judge]
model = "legacy-judge-model"
input_dir = "/legacy/input"
context = "Legacy judge context."
""",
        )
        verifier_dir = task.task_dir / "verifier"
        rubric_dir = verifier_dir / "rubrics"
        rubric_dir.mkdir(parents=True)
        (rubric_dir / "judge.json").write_text(
            json.dumps(
                {
                    "criteria": [
                        {
                            "id": "answer",
                            "match_criteria": "Is answer.txt correct?",
                        }
                    ]
                }
            )
        )
        (rubric_dir / "context.md").write_text("Verifier-local judge context.")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: judge
  strategies:
    judge:
      type: llm-judge
      rubric: rubrics/judge.json
      model: gemini-3.1-flash-lite
      input_dir: /workspace/judge-output
      context_file: rubrics/context.md
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = _make_sandbox({"answer.txt": "42\n"})

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 1.0}
        sandbox.upload_dir.assert_not_called()
        assert sandbox.download_dir.await_args.kwargs["source_dir"] == (
            "/workspace/judge-output"
        )
        assert mock_call_judge.await_args.args[0] == "gemini-3.1-flash-lite"
        prompt = mock_call_judge.await_args.args[1]
        assert "Is answer.txt correct?" in prompt
        assert "Verifier-local judge context." in prompt
        assert "Legacy judge context." not in prompt

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
        rollout_paths.reward_details_json_path.write_text(
            json.dumps({"stale": "details"})
        )

        with pytest.raises(RewardFileNotFoundError, match="verifier exited with rc=7"):
            await Verifier(task, rollout_paths, sandbox).verify()
        assert not rollout_paths.reward_details_json_path.exists()

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
            if "find /logs/verifier" in command:
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

    # NOTE: pre-ENG-150 test `test_nonzero_test_script_cannot_turn_reward_file_into_pass`
    # was deleted here (not rewritten). ENG-150 intentionally changed the verifier
    # contract: nonzero exit + valid reward is now accepted. The new behavior is
    # covered by TestVerifierNonzeroExitRewardAcceptance in test_oracle_chokepoint.py.

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

    @pytest.mark.asyncio
    async def test_reward_details_json_preserved_from_test_script(
        self, tmp_path: Path
    ) -> None:
        """Guards verifier package reward-details artifacts from being dropped."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        reward_payload = {"reward": 0.75}
        details_payload = {
            "criteria": [
                {
                    "id": "reward_contract",
                    "score": 1.0,
                    "reason": "rich reward artifacts preserved",
                }
            ]
        }

        async def exec_details_reward(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_json_path.write_text(json.dumps(reward_payload))
            rollout_paths.reward_details_json_path.write_text(
                json.dumps(details_payload)
            )
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_details_reward)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == reward_payload
        assert json.loads(rollout_paths.reward_details_json_path.read_text()) == (
            details_payload
        )

    @pytest.mark.asyncio
    async def test_reward_json_is_authoritative_when_text_scalar_agrees(
        self, tmp_path: Path
    ) -> None:
        """Guards the task.md verifier standard: reward.json wins over reward.txt."""
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
            "quality": 0.9,
            "space": "output",
            "granularity": "terminal",
            "metadata": {"source": "reward-kit"},
        }

        async def exec_both_rewards(*_args: object, **_kwargs: object) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("0.75")
            rollout_paths.reward_json_path.write_text(json.dumps(payload))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_both_rewards)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == payload

    @pytest.mark.asyncio
    async def test_reward_json_and_text_scalar_mismatch_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """Guards scalar compatibility from silently disagreeing with rich JSON."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_mismatched_rewards(
            *_args: object, **_kwargs: object
        ) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("0.25")
            rollout_paths.reward_json_path.write_text(json.dumps({"reward": 0.75}))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_mismatched_rewards)

        with pytest.raises(VerifierOutputParseError, match="does not match"):
            await Verifier(task, rollout_paths, sandbox).verify()

    @pytest.mark.asyncio
    async def test_reward_json_multi_metric_map_uses_text_scalar_compat(
        self, tmp_path: Path
    ) -> None:
        """Guards Reward Kit-style multi-metric maps without dropping detail."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        payload = {
            "metrics": {"correctness": 1.0, "quality": 0.5},
            "aggregate": {
                "method": "weighted_sum",
                "primary": "reward",
                "weights": {"correctness": 0.7, "quality": 0.3},
            },
            "details_path": "/logs/verifier/reward-details.json",
        }

        async def exec_multi_metric_reward(
            *_args: object, **_kwargs: object
        ) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("0.85")
            rollout_paths.reward_json_path.write_text(json.dumps(payload))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_multi_metric_reward)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 0.85, **payload}

    @pytest.mark.asyncio
    async def test_verifier_document_aggregate_policy_computes_scalar_reward(
        self, tmp_path: Path
    ) -> None:
        """verifier.outputs.aggregate_policy turns metrics into reward."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "test.sh").write_text("#!/usr/bin/env bash\n")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  outputs:
    aggregate_policy:
      field: reward
      method: weighted_sum
      weights:
        correctness: 0.7
        quality: 0.3
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        payload = {
            "metrics": {"correctness": 1.0, "quality": 0.5},
            "details_path": "/logs/verifier/reward-details.json",
        }

        async def exec_metrics_reward(command: str, **_kwargs: object) -> MagicMock:
            if "./test.sh" in command:
                rollout_paths.reward_json_path.write_text(json.dumps(payload))
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_metrics_reward)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 0.85, **payload}
        assert json.loads(rollout_paths.reward_json_path.read_text()) == {
            "reward": 0.85,
            **payload,
        }

    @pytest.mark.asyncio
    async def test_aggregate_policy_and_text_scalar_mismatch_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """Computed aggregate must agree with scalar reward.txt compatibility."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        verifier_dir = task.task_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "test.sh").write_text("#!/usr/bin/env bash\n")
        (verifier_dir / "verifier.md").write_text(
            """---
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  outputs:
    aggregate_policy:
      field: reward
      method: weighted_sum
      weights:
        correctness: 0.7
        quality: 0.3
---
"""
        )
        task.paths.tests_dir = verifier_dir
        task.paths.test_path = verifier_dir / "test.sh"
        task.paths.uses_native_verifier_dir = True

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_mismatched_aggregate(
            command: str, **_kwargs: object
        ) -> MagicMock:
            if "./test.sh" in command:
                rollout_paths.reward_json_path.write_text(
                    json.dumps({"metrics": {"correctness": 1.0, "quality": 0.5}})
                )
                rollout_paths.reward_text_path.write_text("0.25")
            return MagicMock(return_code=0, stdout="")

        sandbox.exec = AsyncMock(side_effect=exec_mismatched_aggregate)

        with pytest.raises(VerifierOutputParseError, match="aggregates to reward"):
            await Verifier(task, rollout_paths, sandbox).verify()

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

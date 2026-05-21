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

from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier import RubricNotFoundError

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
    async def test_judge_env_scoped_to_judge_call(
        self, mock_judge: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#270: [verifier.env] keys are applied during the judge call but
        not leaked into the host process afterwards."""
        import os

        monkeypatch.setenv("MY_JUDGE_KEY", "secret-123")
        monkeypatch.delenv("RESOLVED_KEY", raising=False)

        seen_during_call: dict[str, str | None] = {}

        async def capture_env(*_args: object, **_kwargs: object) -> str:
            seen_during_call["RESOLVED_KEY"] = os.environ.get("RESOLVED_KEY")
            return _MOCK_PASS

        mock_judge.side_effect = capture_env

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

        # applied for the duration of the judge call ...
        assert seen_during_call["RESOLVED_KEY"] == "secret-123"
        # ... but not leaked into the host env afterwards.
        assert os.environ.get("RESOLVED_KEY") is None

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_judge_env_restores_prior_value(
        self, mock_judge: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#270: a [verifier.env] key that already exists is restored to its
        original value after the judge call."""
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
    async def test_download_failure_is_tolerated(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """#270: a sandbox download error degrades to an empty deliverables set,
        not a verifier crash."""
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

        result = await Verifier(task, rollout_paths, sandbox).verify()
        # Judge still runs (against no deliverables) and returns a reward.
        assert result.rewards == {"reward": 0.0}


# ---------------------------------------------------------------------------
# Verifier.verify() — test-script branch stays the default (regression)
# ---------------------------------------------------------------------------


class TestTestScriptStillDefault:
    @pytest.mark.asyncio
    async def test_default_runs_test_script(self, tmp_path: Path) -> None:
        """#270: tasks without type still take the test-script path."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = task.task_dir / "tests"
        task.paths.test_path = task.task_dir / "tests" / "test.sh"

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.exec = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        rollout_paths.reward_text_path.write_text("0.75")

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards == {"reward": 0.75}
        sandbox.upload_dir.assert_called_once()

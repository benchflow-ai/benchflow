"""Integration smoke tests for verifier strategy routing on dogfood package."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.rewards.validation import parse_verifier_reward_files
from benchflow.task import RolloutPaths, Task, Verifier

DOGFOOD_TASK_DIR = Path(
    "docs/examples/task-standard/benchflow-wanted-features/"
    "verifier-package-reward-contract"
)
DOGFOOD_VERIFIER_MD = DOGFOOD_TASK_DIR / "verifier" / "verifier.md"

# Mirrors verifier/test.sh reward artifact shape for multi-metric parsing.
DOGFOOD_MULTI_METRIC_REWARD_JSON = {
    "reward": 1.0,
    "items": {"verifier_document": 1.0, "reward_contract": 1.0},
    "metadata": {"aggregate_policy": "reward"},
}
DOGFOOD_REWARD_DETAILS_JSON = {
    "criteria": [
        {
            "id": "reward_contract",
            "score": 1.0,
            "reason": "rich reward artifacts preserved",
        }
    ]
}


def _dogfood_task_with_strategy(strategy_name: str) -> Task:
    task = Task(DOGFOOD_TASK_DIR)
    document = task.verifier_document
    assert document is not None
    task.verifier_document = replace(document, default_strategy=strategy_name)
    return task


def _mounted_sandbox_mock() -> MagicMock:
    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.is_mounted = True
    return sandbox


def _exec_writes_dogfood_rewards(
    sandbox: MagicMock,
    rollout_paths: RolloutPaths,
) -> AsyncMock:
    async def exec_side_effect(*_args: object, **_kwargs: object) -> MagicMock:
        if sandbox.exec.await_count == 1:
            return MagicMock(return_code=0, stdout="")
        rollout_paths.reward_text_path.write_text("1.0")
        rollout_paths.reward_json_path.write_text(
            json.dumps(DOGFOOD_MULTI_METRIC_REWARD_JSON)
        )
        rollout_paths.reward_details_path.write_text(
            json.dumps(DOGFOOD_REWARD_DETAILS_JSON)
        )
        return MagicMock(return_code=0, stdout="")

    return AsyncMock(side_effect=exec_side_effect)


def test_dogfood_package_declares_three_executable_strategies() -> None:
    """Guards verifier-package-reward-contract strategy surface for smoke routing."""
    from benchflow.task.verifier_document import (
        is_executable_agent_judge_strategy,
        is_executable_reward_kit_strategy,
        is_executable_script_strategy,
        resolve_default_strategy,
    )

    task = Task(DOGFOOD_TASK_DIR)
    document = task.verifier_document
    assert document is not None
    verifier_dir = DOGFOOD_TASK_DIR / "verifier"

    assert document.default_strategy == "deterministic"
    assert set(document.strategies) == {"deterministic", "rewardkit", "judge"}

    _, deterministic = resolve_default_strategy(document)
    assert is_executable_script_strategy(deterministic)

    _, rewardkit = resolve_default_strategy(
        replace(document, default_strategy="rewardkit")
    )
    assert is_executable_reward_kit_strategy(rewardkit, verifier_dir)

    _, judge = resolve_default_strategy(replace(document, default_strategy="judge"))
    assert is_executable_agent_judge_strategy(judge, document, verifier_dir)


def test_dogfood_multi_metric_reward_json_parses_with_explicit_aggregate(
    tmp_path: Path,
) -> None:
    """Guards dogfood test.sh reward.json shape through verifier reward parsing."""
    from benchflow.rewards.validation import validate_reward_map

    reward_json_path = tmp_path / "reward.json"
    reward_text_path = tmp_path / "reward.txt"
    reward_json_path.write_text(json.dumps(DOGFOOD_MULTI_METRIC_REWARD_JSON))
    reward_text_path.write_text("1.0")

    parsed = parse_verifier_reward_files(
        reward_text_path=reward_text_path,
        reward_json_path=reward_json_path,
        source="reward JSON",
    )
    assert parsed["reward"] == 1.0
    assert parsed["items"] == DOGFOOD_MULTI_METRIC_REWARD_JSON["items"]
    assert parsed["metadata"] == DOGFOOD_MULTI_METRIC_REWARD_JSON["metadata"]

    validated = validate_reward_map(DOGFOOD_MULTI_METRIC_REWARD_JSON, source="verifier")
    assert validated["reward"] == parsed["reward"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_name", ["deterministic", "rewardkit"])
async def test_script_strategies_route_to_test_script_and_parse_multi_metric_rewards(
    tmp_path: Path,
    strategy_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards script and reward-kit strategy routing on dogfood verifier package."""
    task = _dogfood_task_with_strategy(strategy_name)
    rollout_paths = RolloutPaths(tmp_path / f"{strategy_name}-rollout")
    rollout_paths.mkdir()

    sandbox = _mounted_sandbox_mock()
    sandbox.exec = _exec_writes_dogfood_rewards(sandbox, rollout_paths)

    with caplog.at_level("INFO"):
        result = await Verifier(task, rollout_paths, sandbox).verify()

    assert result.rewards is not None
    assert result.rewards["reward"] == 1.0
    assert result.rewards["items"] == DOGFOOD_MULTI_METRIC_REWARD_JSON["items"]
    assert result.rewards["metadata"] == DOGFOOD_MULTI_METRIC_REWARD_JSON["metadata"]
    assert rollout_paths.reward_json_path.is_file()
    assert json.loads(rollout_paths.reward_json_path.read_text()) == (
        DOGFOOD_MULTI_METRIC_REWARD_JSON
    )

    expected_type = "script" if strategy_name == "deterministic" else "reward-kit"
    assert any(
        f"Selected verifier document strategy {strategy_name!r} (type={expected_type!r})"
        in record.getMessage()
        for record in caplog.records
    )
    sandbox.upload_dir.assert_called_once()
    assert sandbox.exec.await_count == 2


@pytest.mark.asyncio
async def test_judge_strategy_routes_to_mocked_llm_judge(tmp_path: Path) -> None:
    """Guards agent-judge strategy routing without live LLM calls."""
    task = _dogfood_task_with_strategy("judge")
    rollout_paths = RolloutPaths(tmp_path / "judge-rollout")
    rollout_paths.mkdir()
    trajectory_dir = rollout_paths.rollout_dir / "trajectory"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "acp_trajectory.jsonl").write_text(
        '{"type":"message","content":"agent output"}\n'
    )

    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.download_file = AsyncMock()

    document = task.verifier_document
    assert document is not None

    with patch(
        "benchflow.rewards.builtins.LLMJudgeRewardFunc.score",
        new=AsyncMock(return_value=0.82),
    ) as mock_score:
        with patch(
            "benchflow.rewards.builtins.LLMJudgeRewardFunc",
            wraps=__import__(
                "benchflow.rewards.builtins", fromlist=["LLMJudgeRewardFunc"]
            ).LLMJudgeRewardFunc,
        ) as mock_judge_cls:
            result = await Verifier(task, rollout_paths, sandbox).verify()

    mock_judge_cls.assert_called_once()
    judge_kwargs = mock_judge_cls.call_args.kwargs
    assert judge_kwargs["prompt"] == document.role_prompts["verifier_judge"]
    assert judge_kwargs["judge_errors_are_infra"] is True
    assert str(judge_kwargs["rubric_path"]).endswith("rubrics/verifier.toml")

    mock_score.assert_awaited_once()
    assert result.rewards == {"reward": 0.82}
    assert rollout_paths.reward_json_path.is_file()
    assert json.loads(rollout_paths.reward_json_path.read_text()) == {"reward": 0.82}

    sandbox.upload_dir.assert_not_called()
    sandbox.exec.assert_not_called()

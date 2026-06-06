"""Tests for verifier/verifier.md authoring document parsing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task import (
    RolloutPaths,
    Task,
    UnsupportedVerifierStrategyError,
    Verifier,
    VerifierDocument,
    VerifierDocumentParseError,
    is_executable_script_strategy,
    resolve_default_strategy,
    verifier_document_issues,
    verifier_strategy_type,
)
from benchflow.task.verifier_document import resolve_verifier_spec_path

DOGFOOD_VERIFIER_MD = Path(
    "docs/examples/task-standard/benchflow-wanted-features/"
    "verifier-package-reward-contract/verifier/verifier.md"
)
DOGFOOD_TASK_DIR = DOGFOOD_VERIFIER_MD.parent.parent


def test_verifier_document_parses_dogfood_fixture() -> None:
    """Guards task-standard verifier.md dogfood against parser drift."""
    document = VerifierDocument.from_path(DOGFOOD_VERIFIER_MD)

    assert document.document_version == "0.3"
    assert document.name == "verifier-package-reward-contract"
    assert document.default_strategy == "deterministic"
    assert set(document.strategies) == {"deterministic", "rewardkit", "judge"}
    assert document.strategies["deterministic"]["type"] == "script"
    assert document.strategies["deterministic"]["command"] == "./test.sh"
    assert document.strategies["rewardkit"]["type"] == "reward-kit"
    assert document.strategies["rewardkit"]["root"] == "reward_kit/"
    assert document.strategies["rewardkit"]["criteria"] == "rubrics/verifier.toml"
    assert document.strategies["judge"]["type"] == "agent-judge"
    assert document.strategies["judge"]["role"] == "verifier_judge"
    assert document.rubric["combine"] == "weighted_sum"
    assert document.rubric["dimensions"]["reward_contract"]["weight"] == 0.35
    assert document.rubric_files.structured == "rubrics/verifier.toml"
    assert document.outputs.reward_text == "/logs/verifier/reward.txt"
    assert document.outputs.reward_json == "/logs/verifier/reward.json"
    assert document.outputs.reward_details == "/logs/verifier/reward-details.json"
    assert document.outputs.aggregate_policy == {
        "field": "reward",
        "fallback": "weighted_mean",
    }
    assert "verifier_judge" in document.role_prompts
    assert "hidden fixture leakage" in document.role_prompts["verifier_judge"]


def test_verifier_document_rejects_missing_frontmatter() -> None:
    with pytest.raises(VerifierDocumentParseError, match="YAML frontmatter"):
        VerifierDocument.from_text("No frontmatter here.")


def test_verifier_document_rejects_invalid_strategies() -> None:
    with pytest.raises(VerifierDocumentParseError, match="strategies must be a mapping"):
        VerifierDocument.from_text(
            """---
verifier:
  strategies: not-a-mapping
---
"""
        )


def test_verifier_document_issues_require_spec_file() -> None:
    task_dir = Path("/tmp/task")
    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )
    assert issues == [
        "benchflow.verifier.spec references missing file: verifier/verifier.md"
    ]


def test_verifier_document_issues_validate_dogfood_task() -> None:
    issues = verifier_document_issues(
        DOGFOOD_TASK_DIR,
        benchflow_verifier=_dogfood_benchflow_verifier(),
    )
    assert issues == []


def test_verifier_document_issues_require_agent_judge_role_field(
    tmp_path: Path,
) -> None:
    """Guards agent-judge strategy role field presence on dogfood verifier.md."""
    task_dir = tmp_path / "task"
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.toml").write_text(
        "[criterion]\nname = \"contract\"\n"
    )
    (verifier_dir / "reward_kit").mkdir()
    verifier_text = DOGFOOD_VERIFIER_MD.read_text(encoding="utf-8").replace(
        "      role: verifier_judge\n",
        "",
    )
    (verifier_dir / "verifier.md").write_text(verifier_text, encoding="utf-8")

    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )

    assert issues == [
        "verifier/verifier.md agent-judge strategy 'judge' is missing role"
    ]


def test_verifier_document_issues_require_agent_judge_role_prompt(
    tmp_path: Path,
) -> None:
    """Guards agent-judge inline role prompt validation on dogfood verifier.md."""
    task_dir = tmp_path / "task"
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.toml").write_text(
        "[criterion]\nname = \"contract\"\n"
    )
    (verifier_dir / "reward_kit").mkdir()
    verifier_text = DOGFOOD_VERIFIER_MD.read_text(encoding="utf-8").replace(
        "role: verifier_judge",
        "role: missing_judge",
    )
    (verifier_dir / "verifier.md").write_text(verifier_text, encoding="utf-8")

    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )

    assert issues == [
        "verifier/verifier.md agent-judge strategy 'judge' references "
        "unknown role prompt ## role:missing_judge"
    ]


def test_verifier_document_issues_require_agent_judge_role_file(
    tmp_path: Path,
) -> None:
    """Guards agent-judge judges/*.md role file validation on dogfood verifier.md."""
    task_dir = tmp_path / "task"
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.toml").write_text(
        "[criterion]\nname = \"contract\"\n"
    )
    (verifier_dir / "reward_kit").mkdir()
    verifier_text = DOGFOOD_VERIFIER_MD.read_text(encoding="utf-8").replace(
        "role: verifier_judge",
        "role: judges/reviewer.md",
    )
    (verifier_dir / "verifier.md").write_text(verifier_text, encoding="utf-8")

    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )

    assert issues == [
        "verifier/verifier.md agent-judge strategy 'judge' references "
        "missing role file: judges/reviewer.md"
    ]


def test_verifier_document_issues_accepts_agent_judge_role_file(
    tmp_path: Path,
) -> None:
    """Guards agent-judge role file path resolution relative to verifier/."""
    task_dir = tmp_path / "task"
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.toml").write_text(
        "[criterion]\nname = \"contract\"\n"
    )
    (verifier_dir / "reward_kit").mkdir()
    (verifier_dir / "judges").mkdir()
    (verifier_dir / "judges" / "reviewer.md").write_text(
        "Grade only declared deliverables.\n",
        encoding="utf-8",
    )
    verifier_text = DOGFOOD_VERIFIER_MD.read_text(encoding="utf-8")
    verifier_text = verifier_text.replace("role: verifier_judge", "role: judges/reviewer.md")
    verifier_text = verifier_text.replace(
        "## role:verifier_judge\n\nEvaluate only declared deliverables",
        "",
    )
    (verifier_dir / "verifier.md").write_text(verifier_text, encoding="utf-8")

    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )

    assert issues == []


def test_verifier_document_issues_require_reward_kit_root_directory(
    tmp_path: Path,
) -> None:
    """Guards reward-kit strategy root directory validation on dogfood verifier.md."""
    task_dir = tmp_path / "task"
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.toml").write_text(
        "[criterion]\nname = \"contract\"\n"
    )
    (verifier_dir / "verifier.md").write_text(
        DOGFOOD_VERIFIER_MD.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    issues = verifier_document_issues(
        task_dir,
        benchflow_verifier={"spec": "verifier/verifier.md"},
    )

    assert issues == [
        "verifier/verifier.md reward-kit strategy 'rewardkit' references "
        "missing root directory: reward_kit/"
    ]


def test_check_task_validates_verifier_spec_for_dogfood_task() -> None:
    """Guards task-standard benchflow.verifier.spec validation in check_task."""
    issues = check_task(DOGFOOD_TASK_DIR)
    assert not any("verifier.md" in issue and "missing" in issue for issue in issues)
    assert not any("parse error" in issue for issue in issues if "verifier" in issue)


def test_check_task_reports_missing_verifier_spec(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "environment").mkdir()
    (task_dir / "verifier").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "verifier" / "test.sh").write_text("exit 0\n")
    (task_dir / "task.md").write_text(
        """---
agent:
  timeout_sec: 120
environment:
  cpus: 1
benchflow:
  verifier:
    spec: verifier/verifier.md
---
## prompt

Solve it.
"""
    )

    issues = check_task(task_dir)

    assert any(
        "benchflow.verifier.spec references missing file: verifier/verifier.md"
        in issue
        for issue in issues
    )


def test_task_loads_verifier_document_from_benchflow_spec() -> None:
    task = Task(DOGFOOD_TASK_DIR)

    assert task.verifier_document is not None
    assert task.verifier_document.name == "verifier-package-reward-contract"
    assert task.verifier_document.default_strategy == "deterministic"


def test_task_rejects_invalid_verifier_spec(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "environment").mkdir()
    (task_dir / "verifier").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "verifier" / "test.sh").write_text("exit 0\n")
    (task_dir / "verifier" / "verifier.md").write_text("not valid frontmatter\n")
    (task_dir / "task.md").write_text(
        """---
agent:
  timeout_sec: 120
environment:
  cpus: 1
benchflow:
  verifier:
    spec: verifier/verifier.md
---
## prompt

Solve it.
"""
    )

    with pytest.raises(ValueError, match="parse error"):
        Task(task_dir)


def test_resolve_verifier_spec_path_is_task_relative() -> None:
    spec_path = resolve_verifier_spec_path(
        DOGFOOD_TASK_DIR,
        "verifier/verifier.md",
    )
    assert spec_path == DOGFOOD_VERIFIER_MD.resolve()


def _dogfood_benchflow_verifier() -> dict[str, object]:
    from benchflow.task import TaskDocument

    document = TaskDocument.from_path(DOGFOOD_TASK_DIR / "task.md")
    benchflow = document.benchflow
    assert isinstance(benchflow, dict)
    verifier = benchflow.get("verifier")
    assert isinstance(verifier, dict)
    return verifier


def test_resolve_default_strategy_uses_dogfood_fixture() -> None:
    """Guards verifier-package-reward-contract default strategy resolution."""
    document = VerifierDocument.from_path(DOGFOOD_VERIFIER_MD)

    strategy_name, strategy = resolve_default_strategy(document)

    assert strategy_name == "deterministic"
    assert verifier_strategy_type(strategy) == "script"
    assert is_executable_script_strategy(strategy)


@pytest.mark.asyncio
async def test_verify_logs_and_routes_deterministic_strategy_to_test_script(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards verifier document script strategy routing on dogfood task."""
    task = Task(DOGFOOD_TASK_DIR)
    rollout_paths = RolloutPaths(Path("/tmp/verifier-strategy-rollout"))
    rollout_paths.mkdir()

    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.is_mounted = True

    async def exec_writes_reward(*_args: object, **_kwargs: object) -> MagicMock:
        if sandbox.exec.await_count == 1:
            return MagicMock(return_code=0, stdout="")
        rollout_paths.reward_text_path.write_text("1.0")
        return MagicMock(return_code=0, stdout="")

    sandbox.exec = AsyncMock(side_effect=exec_writes_reward)

    with caplog.at_level("INFO"):
        result = await Verifier(task, rollout_paths, sandbox).verify()

    assert result.rewards == {"reward": 1.0}
    assert any(
        "Selected verifier document strategy 'deterministic' (type='script')"
        in record.getMessage()
        for record in caplog.records
    )
    sandbox.upload_dir.assert_called_once()


@pytest.mark.asyncio
async def test_verify_routes_reward_kit_strategy_to_test_script(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards reward-kit strategy execution via script verifier when criteria exist."""
    task = Task(DOGFOOD_TASK_DIR)
    document = task.verifier_document
    assert document is not None
    task.verifier_document = replace(document, default_strategy="rewardkit")

    rollout_paths = RolloutPaths(tmp_path / "rewardkit-rollout")
    rollout_paths.mkdir()

    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.is_mounted = True

    async def exec_writes_reward(*_args: object, **_kwargs: object) -> MagicMock:
        if sandbox.exec.await_count == 1:
            return MagicMock(return_code=0, stdout="")
        rollout_paths.reward_json_path.write_text('{"reward": 1.0, "reward_contract": 1.0}')
        return MagicMock(return_code=0, stdout="")

    sandbox.exec = AsyncMock(side_effect=exec_writes_reward)

    with caplog.at_level("INFO"):
        result = await Verifier(task, rollout_paths, sandbox).verify()

    assert result.rewards is not None
    assert result.rewards["reward"] == 1.0
    assert any(
        "Selected verifier document strategy 'rewardkit' (type='reward-kit')"
        in record.getMessage()
        for record in caplog.records
    )
    sandbox.upload_dir.assert_called_once()


@pytest.mark.asyncio
async def test_verify_routes_agent_judge_strategy_to_llm_judge(
    tmp_path: Path,
) -> None:
    """Guards agent-judge strategy execution via verifier-scoped LLM judge."""
    task = Task(DOGFOOD_TASK_DIR)
    document = task.verifier_document
    assert document is not None
    task.verifier_document = replace(document, default_strategy="judge")

    rollout_paths = RolloutPaths(tmp_path / "agent-judge-rollout")
    rollout_paths.mkdir()
    (rollout_paths.rollout_dir / "trajectory").mkdir(parents=True)
    (rollout_paths.rollout_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        '{"type":"message","content":"agent output"}\n'
    )

    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.download_file = AsyncMock()

    with patch(
        "benchflow.rewards.builtins.LLMJudgeRewardFunc.score",
        new=AsyncMock(return_value=0.75),
    ) as mock_score:
        result = await Verifier(task, rollout_paths, sandbox).verify()

    mock_score.assert_awaited_once()
    assert result.rewards == {"reward": 0.75}
    assert rollout_paths.reward_json_path.is_file()
    sandbox.upload_dir.assert_not_called()
    sandbox.exec.assert_not_called()


def test_reward_kit_not_executable_without_criteria_or_entrypoint(
    tmp_path: Path,
) -> None:
    """Guards reward-kit executability requires criteria or root test.sh."""
    from benchflow.task.verifier_document import is_executable_reward_kit_strategy

    strategy = {"type": "reward-kit", "root": "reward_kit/"}
    assert is_executable_reward_kit_strategy(strategy, tmp_path) is False


@pytest.mark.asyncio
async def test_verify_rejects_non_executable_reward_kit_strategy(
    tmp_path: Path,
) -> None:
    """Guards fail-closed routing when reward-kit strategy cannot execute."""
    task = Task(DOGFOOD_TASK_DIR)
    document = task.verifier_document
    assert document is not None
    task.verifier_document = replace(
        document,
        default_strategy="rewardkit",
        strategies={
            "rewardkit": {
                "type": "reward-kit",
                "root": "missing_reward_kit/",
            }
        },
    )

    rollout_paths = RolloutPaths(tmp_path / "missing-rewardkit")
    rollout_paths.mkdir()
    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()

    with pytest.raises(
        UnsupportedVerifierStrategyError,
        match=r"type='reward-kit'.*not executable",
    ):
        await Verifier(task, rollout_paths, sandbox).verify()

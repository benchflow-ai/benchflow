"""Tests for verifier/verifier.md authoring document parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task import (
    Task,
    VerifierDocument,
    VerifierDocumentParseError,
    verifier_document_issues,
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

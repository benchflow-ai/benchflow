"""Tests for typed ``benchflow:`` namespace validation (P3 subset)."""

from __future__ import annotations

from pathlib import Path

from benchflow._utils.task_authoring import check_task
from benchflow.task.benchflow_schema import (
    BenchflowMetadata,
    parse_benchflow_metadata,
    validate_benchflow_metadata,
)

PROMPT_USER_SEMANTICS_TASK = (
    "docs/examples/task-standard/benchflow-wanted-features/prompt-user-semantics"
)


class TestValidateBenchflowMetadata:
    def test_prompt_user_semantics_dogfood_validates(self) -> None:
        """Guards typed benchflow validation for prompt-user-semantics dogfood."""
        issues = validate_benchflow_metadata(
            {
                "document_version": "0.3",
                "prompt": {
                    "composition": "append",
                    "order": ["base", "role", "scene", "turn"],
                },
                "nudges": {
                    "mode": "simulated-user",
                    "branchable": True,
                    "nudge_budget": 5,
                },
            }
        )
        assert issues == []

    def test_invalid_prompt_composition_reports_issue(self) -> None:
        issues = validate_benchflow_metadata(
            {"prompt": {"composition": "shadow", "order": ["base"]}}
        )
        assert any("benchflow.prompt.composition" in issue for issue in issues)

    def test_invalid_nudge_budget_reports_issue(self) -> None:
        issues = validate_benchflow_metadata({"nudges": {"nudge_budget": 0}})
        assert any("benchflow.nudges.nudge_budget" in issue for issue in issues)

    def test_parse_returns_metadata_when_valid(self) -> None:
        metadata = parse_benchflow_metadata(
            {
                "document_version": "0.3",
                "compatibility": {"target": "harbor"},
            }
        )
        assert isinstance(metadata, BenchflowMetadata)
        assert metadata.document_version == "0.3"
        assert metadata.compatibility is not None
        assert metadata.compatibility.target == "harbor"


class TestCheckTaskBenchflowValidation:
    def test_prompt_user_semantics_check_task_passes_typed_benchflow(self) -> None:
        """Guards check_task includes typed benchflow validation for dogfood."""
        assert check_task(Path(PROMPT_USER_SEMANTICS_TASK)) == []

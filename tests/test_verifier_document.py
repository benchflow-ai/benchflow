"""Tests for native verifier package documents."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task import (
    TaskPaths,
    TaskRuntimeView,
    VerifierDocument,
    VerifierDocumentParseError,
    load_verifier_document,
)

DOGFOOD_VERIFIER = Path(
    "docs/examples/task-standard/benchflow-wanted-features/"
    "verifier-package-reward-contract/verifier"
)


def test_dogfood_verifier_document_parses_strategy_rubric_outputs_and_role() -> None:
    """The verifier-package dogfood task has a first-class VerifierDocument."""
    document = VerifierDocument.from_verifier_dir(DOGFOOD_VERIFIER)

    assert document.name == "verifier-package-reward-contract"
    assert document.document_version == "0.3"
    assert document.default_strategy == "deterministic"
    assert document.selected_strategy.type == "script"
    assert document.selected_strategy.command == "./test.sh"
    assert set(document.strategies) == {
        "deterministic",
        "llm_judge",
        "rewardkit",
        "judge",
    }
    assert document.strategies["llm_judge"].rubric_path == "rubrics/verifier.toml"
    assert document.strategies["llm_judge"].model == "gemini-3.1-flash-lite"
    assert document.strategies["llm_judge"].input_dir == "/logs/artifacts"
    assert document.strategies["llm_judge"].context_file == "rubrics/context.md"
    assert document.strategies["rewardkit"].config["root"] == "reward_kit/"
    assert document.strategies["judge"].role == "verifier_judge"
    assert document.strategies["judge"].inputs == (
        "trajectory/acp_trajectory.jsonl",
        "/logs/artifacts/diff.patch",
    )
    assert document.rubric["combine"] == "weighted_sum"
    assert document.rubric["dimensions"]["reward_contract"]["weight"] == 0.35
    assert document.outputs.reward_text == "/logs/verifier/reward.txt"
    assert document.outputs.reward_json == "/logs/verifier/reward.json"
    assert document.outputs.details_json == "/logs/verifier/reward-details.json"
    assert document.outputs.aggregate_policy["field"] == "reward"
    assert "declared deliverables" in document.role_prompts["verifier_judge"]


def test_runtime_view_loads_selected_native_verifier_document() -> None:
    """TaskRuntimeView exposes verifier package metadata without replacing test.sh."""
    task_dir = DOGFOOD_VERIFIER.parent.resolve()

    view = TaskRuntimeView.from_task_dir(task_dir)

    assert view.verifier_dir == task_dir / "verifier"
    assert view.verifier_document is not None
    assert view.verifier_document.selected_strategy.command == "./test.sh"
    assert "verifier/verifier.md" in view.source_hashes


def test_load_verifier_document_returns_none_without_verifier_md(
    tmp_path: Path,
) -> None:
    """Plain verifier/test.sh packages remain valid compatibility input."""
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    assert load_verifier_document(verifier_dir) is None


def test_task_paths_accepts_selected_reward_kit_without_test_sh(
    tmp_path: Path,
) -> None:
    """Native verifier packages do not need a legacy test.sh entrypoint."""
    task_dir = tmp_path / "rewardkit-native"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            environment:
              network_mode: no-network
            ---

            ## prompt

            Implement the verifier package boundary.
            """
        )
    )
    verifier = task_dir / "verifier"
    (verifier / "reward_kit").mkdir(parents=True)
    (verifier / "rubrics").mkdir()
    (verifier / "reward_kit" / "reward.py").write_text("print('score')\n")
    (verifier / "rubrics" / "verifier.toml").write_text("[scoring]\nmethod='mean'\n")
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: rewardkit
              strategies:
                rewardkit:
                  type: reward-kit
                  root: reward_kit
                  criteria: rubrics/verifier.toml
            ---
            """
        )
    )

    paths = TaskPaths(task_dir)

    assert not paths.test_path.exists()
    assert paths.has_verifier_entrypoint()
    assert paths.is_valid()
    assert check_task(task_dir) == []


def test_task_paths_rejects_selected_reward_kit_without_runner(
    tmp_path: Path,
) -> None:
    """A selected Reward Kit strategy must still have its package runner."""
    task_dir = tmp_path / "missing-rewardkit-runner"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        "---\nschema_version: '1.3'\n---\n\n## prompt\n\nDo it.\n"
    )
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: rewardkit
              strategies:
                rewardkit:
                  type: reward-kit
                  root: reward_kit
            ---
            """
        )
    )

    paths = TaskPaths(task_dir)

    assert not paths.has_verifier_entrypoint()
    assert not paths.is_valid()


def test_task_paths_rejects_selected_script_missing_interpreter_arg(
    tmp_path: Path,
) -> None:
    """Selected script commands can reference local files after an interpreter."""
    task_dir = tmp_path / "missing-script-entrypoint"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        "---\nschema_version: '1.3'\n---\n\n## prompt\n\nDo it.\n"
    )
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: deterministic
              strategies:
                deterministic:
                  type: script
                  command: bash missing.sh
            ---
            """
        )
    )

    paths = TaskPaths(task_dir)

    assert not paths.test_path.exists()
    assert not paths.has_verifier_entrypoint()
    assert not paths.is_valid()


def test_task_paths_rejects_selected_script_without_packaged_artifact(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against bare script commands passing without verifier files."""
    task_dir = tmp_path / "bare-script-entrypoint"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        "---\nschema_version: '1.3'\n---\n\n## prompt\n\nDo it.\n"
    )
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: deterministic
              strategies:
                deterministic:
                  type: script
                  command: pytest
            ---
            """
        )
    )

    paths = TaskPaths(task_dir)

    assert not paths.test_path.exists()
    assert not paths.has_verifier_entrypoint()
    assert not paths.is_valid()


def test_task_paths_accepts_selected_ors_episode_without_test_sh(
    tmp_path: Path,
) -> None:
    """ORS episode strategies are native verifier entrypoints."""
    task_dir = tmp_path / "ors-native"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        "---\nschema_version: '1.3'\n---\n\n## prompt\n\nDo it.\n"
    )
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: ors
              strategies:
                ors:
                  type: ors-episode
                  inputs: [trajectory/ors-rewards.jsonl]
            ---
            """
        )
    )

    paths = TaskPaths(task_dir)

    assert not paths.test_path.exists()
    assert paths.has_verifier_entrypoint()
    assert paths.is_valid()
    assert check_task(task_dir) == []


def test_ors_episode_requires_declared_inputs() -> None:
    """ORS episode strategies must name verifier evidence explicitly."""
    text = dedent(
        """\
        ---
        verifier:
          strategies:
            ors:
              type: ors-episode
        ---
        """
    )

    with pytest.raises(VerifierDocumentParseError, match="inputs"):
        VerifierDocument.from_text(text)


def test_agent_judge_requires_declared_role_section() -> None:
    """Judge agents are verifier-scoped and cannot be undeclared side channels."""
    text = dedent(
        """\
        ---
        verifier:
          strategies:
            judge:
              type: agent-judge
              role: verifier_judge
              inputs: [/logs/artifacts/out.txt]
              isolation: verifier-only
        ---
        """
    )

    with pytest.raises(VerifierDocumentParseError, match="missing ## role"):
        VerifierDocument.from_text(text)


def test_agent_judge_requires_verifier_only_isolation() -> None:
    """Agent-as-judge strategies must not run in solver scope."""
    text = dedent(
        """\
        ---
        verifier:
          strategies:
            judge:
              type: agent-judge
              role: verifier_judge
              inputs: [/logs/artifacts/out.txt]
              isolation: solver-visible
        ---

        ## role:verifier_judge

        Judge only declared outputs.
        """
    )

    with pytest.raises(VerifierDocumentParseError, match="verifier-only"):
        VerifierDocument.from_text(text)


def test_reward_kit_requires_safe_relative_paths() -> None:
    """Reward Kit package paths must not escape verifier scope."""
    text = dedent(
        """\
        ---
        verifier:
          strategies:
            rewardkit:
              type: reward-kit
              root: ../reward_kit
        ---
        """
    )

    with pytest.raises(VerifierDocumentParseError, match="safe relative path"):
        VerifierDocument.from_text(text)


def test_default_strategy_must_exist() -> None:
    """A typo in the selected strategy is a verifier package error."""
    text = dedent(
        """\
        ---
        verifier:
          default_strategy: missing
          strategies:
            deterministic:
              type: script
              command: ./test.sh
        ---
        """
    )

    with pytest.raises(VerifierDocumentParseError, match="default_strategy"):
        VerifierDocument.from_text(text)


def test_check_task_reports_malformed_verifier_document(tmp_path: Path) -> None:
    """Structural task checks validate verifier/verifier.md when present."""
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            environment:
              network_mode: no-network
            ---

            ## prompt

            Do the thing.
            """
        )
    )
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (verifier / "verifier.md").write_text("not a verifier document\n")

    issues = check_task(task_dir)

    assert any("verifier/verifier.md parse error" in issue for issue in issues)

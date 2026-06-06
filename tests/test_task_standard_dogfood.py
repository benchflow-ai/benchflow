"""Dogfood acceptance tests for task-standard benchflow-wanted-features packages."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task import Task, TaskDocument, TaskRuntimeView, VerifierDocument
from benchflow.task.export import (
    EXPORT_REPORT_REL_PATH,
    ExportLoss,
    export_report_json,
    export_task_package,
    write_export_report,
)
from benchflow.task.runtime_capabilities import validate_task_runtime_support

DOGFOOD_ROOT = Path(
    "docs/examples/task-standard/benchflow-wanted-features"
)
COMPAT_EXPORT_TASK = DOGFOOD_ROOT / "compat-export-loss-reports"
RUNTIME_GATE_TASK = DOGFOOD_ROOT / "runtime-capability-gate"
PROMPT_USER_SEMANTICS_TASK = DOGFOOD_ROOT / "prompt-user-semantics"
VERIFIER_REWARD_CONTRACT_TASK = DOGFOOD_ROOT / "verifier-package-reward-contract"

_EXPECTED_EXPORT_LOSS_CONCEPTS = frozenset(
    {
        "agents",
        "scenes",
        "benchflow.document_version",
        "benchflow.compatibility",
        "benchflow.traceability",
        "benchflow.verifier",
        "prompt.role:adapter_engineer",
        "prompt.role:compatibility_reviewer",
        "verifier.verifier_md",
        "verifier.rubrics",
    }
)


def _dogfood_task_dirs() -> list[Path]:
    return sorted(
        path
        for path in DOGFOOD_ROOT.iterdir()
        if path.is_dir() and (path / "task.md").is_file()
    )


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_check_task_passes(task_dir: Path) -> None:
    """Guards bench tasks check acceptance for every wanted-features dogfood package."""
    assert check_task(task_dir) == []


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_task_loads(task_dir: Path) -> None:
    """Guards Task() loading for every wanted-features dogfood package."""
    task = Task(task_dir)
    assert task.runtime_view.entrypoint == "task-md"


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_runtime_view_from_task_dir(task_dir: Path) -> None:
    """Guards TaskRuntimeView.from_task_dir for every wanted-features dogfood package."""
    view = TaskRuntimeView.from_task_dir(task_dir)
    assert view.task_dir == task_dir.resolve()
    assert view.entrypoint == "task-md"
    assert view.instruction_text.strip()


def test_prompt_user_semantics_append_composes_scene_turn_prompts() -> None:
    """Guards prompt-user-semantics append composition for scene turns."""
    document = TaskDocument.from_path(PROMPT_USER_SEMANTICS_TASK / "task.md")
    scenes = {scene.name: scene for scene in document.scenes}

    prompt_composition_scene = scenes["prompt-composition"]
    user_loop_scene = scenes["user-loop"]

    engineer_base_role = (
        f"{document.instruction}\n\n"
        f"{document.role_prompts['scene_engineer']}"
    )
    user_loop_engineer = (
        f"{engineer_base_role}\n\n"
        f"{document.scene_prompts['user-loop']}"
    )
    user_loop_reviewer = (
        f"{document.instruction}\n\n"
        f"{document.role_prompts['ux_reviewer']}\n\n"
        f"{document.scene_prompts['user-loop']}"
    )

    assert prompt_composition_scene.turns[0].prompt == engineer_base_role
    assert user_loop_scene.turns[0].prompt == user_loop_engineer
    assert user_loop_scene.turns[1].prompt == user_loop_reviewer


def test_prompt_user_semantics_sandbox_check_supports_executable_nudges() -> None:
    """Guards prompt-user-semantics simulated-user nudge execution on docker."""
    task = Task(PROMPT_USER_SEMANTICS_TASK)
    issues = validate_task_runtime_support(
        task, "docker", PROMPT_USER_SEMANTICS_TASK
    )

    paths = {issue.path for issue in issues}
    assert "benchflow.nudges" not in paths
    assert "user" not in paths
    assert "prompt.user-persona" not in paths


def test_verifier_reward_contract_reward_kit_strategy_is_executable() -> None:
    """Guards verifier-package-reward-contract reward-kit criteria executability."""
    from benchflow.task.verifier_document import (
        is_executable_agent_judge_strategy,
        is_executable_reward_kit_strategy,
        resolve_default_strategy,
    )

    task = Task(VERIFIER_REWARD_CONTRACT_TASK)
    document = task.verifier_document
    assert document is not None
    verifier_dir = VERIFIER_REWARD_CONTRACT_TASK / "verifier"

    _, rewardkit = resolve_default_strategy(
        replace(document, default_strategy="rewardkit")
    )
    assert is_executable_reward_kit_strategy(rewardkit, verifier_dir)

    _, judge = resolve_default_strategy(replace(document, default_strategy="judge"))
    assert is_executable_agent_judge_strategy(judge, document, verifier_dir)


def test_verifier_reward_contract_loads_deterministic_default_strategy() -> None:
    """Guards verifier-package-reward-contract VerifierDocument default strategy."""
    task = Task(VERIFIER_REWARD_CONTRACT_TASK)
    verifier_md = (
        VERIFIER_REWARD_CONTRACT_TASK / "verifier" / "verifier.md"
    )

    assert task.verifier_document is not None
    assert task.verifier_document.name == "verifier-package-reward-contract"
    assert task.verifier_document.default_strategy == "deterministic"

    document = VerifierDocument.from_path(verifier_md)
    assert document.default_strategy == "deterministic"


def test_compat_export_loss_reports_degraded_export() -> None:
    """Guards compat-export-loss-reports degraded Harbor export and loss concepts."""
    result = export_task_package(COMPAT_EXPORT_TASK)

    assert result.target == "harbor"
    assert result.mode == "degraded"
    assert result.selected_definition == "task.md"
    assert result.selected_oracle_dir == "oracle/"
    assert result.selected_verifier_dir == "verifier/"
    assert result.exported_oracle_dir == "solution/"
    assert result.exported_verifier_dir == "tests/"
    assert result.losses
    assert all(isinstance(loss, ExportLoss) for loss in result.losses)

    concepts = {loss.concept for loss in result.losses}
    assert concepts == _EXPECTED_EXPORT_LOSS_CONCEPTS


def test_compat_export_writes_export_report_json(tmp_path: Path) -> None:
    """Guards compat-export-loss-reports on-disk compatibility/export-report.json."""
    result = export_task_package(COMPAT_EXPORT_TASK)
    report_path = write_export_report(tmp_path, result)

    assert report_path == tmp_path / EXPORT_REPORT_REL_PATH
    assert report_path.is_file()

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    expected = export_report_json(result)
    assert parsed == expected
    assert parsed["target"] == "harbor"
    assert parsed["mode"] == "degraded"
    assert len(parsed["losses"]) == len(result.losses)
    assert parsed["input_hashes"]["task.md"]
    assert parsed["output_hashes"]["task.toml"]


def test_runtime_capability_gate_fails_on_docker() -> None:
    """Guards runtime-capability-gate fail-closed validation before sandbox creation."""
    task = Task(RUNTIME_GATE_TASK)
    issues = validate_task_runtime_support(task, "docker", RUNTIME_GATE_TASK)

    assert issues
    paths = {issue.path for issue in issues}
    assert "environment.workdir" in paths
    assert all(issue.sandbox_type == "docker" for issue in issues)

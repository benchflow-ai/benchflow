"""Smoke tests for task-standard rollout wiring on dogfood packages.

Exercises real task loading and RolloutConfig wiring without starting
sandboxes or agents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.rollout import RolloutConfig
from benchflow.task import (
    Task,
    TaskRuntimeView,
    compile_document_user_loop,
    export_task_package,
)
from benchflow.task.export import ExportLoss
from benchflow.task.user_loop import (
    DocumentSimulatedUser,
    infer_user_loop_scene_name,
    resolve_user_loop_rollout_plan,
)

DOGFOOD_ROOT = Path("docs/examples/task-standard/benchflow-wanted-features")
PROMPT_USER_SEMANTICS_TASK = DOGFOOD_ROOT / "prompt-user-semantics"
COMPAT_EXPORT_TASK = DOGFOOD_ROOT / "compat-export-loss-reports"

_EXPECTED_SCENE_NAMES: dict[str, list[str]] = {
    "runtime-capability-gate": ["design", "implement", "review"],
    "verifier-package-reward-contract": ["implement", "rubric-review"],
    "compat-export-loss-reports": ["implement-export", "compatibility-review"],
    "prompt-user-semantics": ["prompt-composition", "user-loop"],
}

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
def test_smoke_task_load(task_dir: Path) -> None:
    """Guards Task() loading for every wanted-features dogfood package."""
    task = Task(task_dir)
    assert task.runtime_view.entrypoint == "task-md"
    assert task.runtime_view.instruction_text.strip()


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_smoke_runtime_view_summary(task_dir: Path) -> None:
    """Guards TaskRuntimeView.from_task_dir and document_runtime_summary."""
    view = TaskRuntimeView.from_task_dir(task_dir)
    assert view.task_dir == task_dir.resolve()
    assert view.entrypoint == "task-md"

    summary = view.document_runtime_summary()
    assert summary["entrypoint"] == "task-md"
    assert summary["task_dir"] == str(task_dir.resolve())
    assert summary["instruction_chars"] > 0
    assert summary["scene_names"] == _EXPECTED_SCENE_NAMES[task_dir.name]
    assert summary["verifier_dir_kind"] in {"native", "legacy"}
    assert isinstance(summary.get("alias_collisions", []), list)


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_smoke_rollout_config_scenes(task_dir: Path) -> None:
    """Guards RolloutConfig scene wiring from task.md for every dogfood task."""
    config = RolloutConfig(task_path=task_dir)

    assert isinstance(config.task_path, Path)
    assert [scene.name for scene in config.scenes] == _EXPECTED_SCENE_NAMES[
        task_dir.name
    ]
    assert all(scene.roles for scene in config.scenes)
    assert all(scene.turns for scene in config.scenes)

    if task_dir.name == "prompt-user-semantics":
        assert config.user is not None
        assert isinstance(config.user, DocumentSimulatedUser)
        assert config.max_user_rounds == 5
        assert config.user_loop_plan is not None
        assert len(config.user_loop_plan.pre_scenes) == 1
        assert config.user_loop_plan.pre_scenes[0].name == "prompt-composition"
        assert config.user_loop_plan.user_loop_scene.name == "user-loop"
        assert config.user_loop_plan.user_loop_role.name == "scene_engineer"
        assert config.user_loop_plan.post_scene is not None
        assert config.user_loop_plan.post_scene.turns[0].role == "ux_reviewer"
    else:
        assert config.user_loop_plan is None


def test_smoke_prompt_user_semantics_user_loop_compile() -> None:
    """Guards compile_document_user_loop + resolve_user_loop_rollout_plan wiring."""
    task = Task(PROMPT_USER_SEMANTICS_TASK)
    document = task.document
    assert document is not None

    compiled = compile_document_user_loop(task)
    assert compiled is not None
    assert compiled.executable is True
    assert compiled.max_user_rounds == 5
    assert isinstance(compiled.user, DocumentSimulatedUser)

    scene_name = infer_user_loop_scene_name(document)
    assert scene_name == "user-loop"

    plan = resolve_user_loop_rollout_plan(
        config_scenes := RolloutConfig(task_path=PROMPT_USER_SEMANTICS_TASK).scenes,
        user_loop_scene_name=scene_name,
        nudges=document.benchflow.get("nudges"),
    )
    assert plan is not None
    assert [scene.name for scene in plan.pre_scenes] == ["prompt-composition"]
    assert plan.user_loop_role.name == "scene_engineer"
    assert plan.post_scene is not None
    assert plan.post_scene.turns[0].role == "ux_reviewer"

    rollout_plan = RolloutConfig(task_path=PROMPT_USER_SEMANTICS_TASK).user_loop_plan
    assert rollout_plan is not None
    assert [scene.name for scene in rollout_plan.pre_scenes] == [
        scene.name for scene in plan.pre_scenes
    ]
    assert rollout_plan.user_loop_role.name == plan.user_loop_role.name
    assert len(config_scenes) == 2


def test_smoke_compat_export_loss_reports() -> None:
    """Guards export_task_package for compat-export-loss-reports dogfood task."""
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

"""Scene desugaring tests."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

import benchflow
from benchflow._types import Role, Scene, Turn
from benchflow.scenes import (
    DEFAULT_SCENE_PROMPT,
    compile_scenes_to_steps,
    scene_step_prompt,
    scene_step_role,
    scene_step_skills_dir,
)


def test_scene_compiles_turns_to_steps() -> None:
    """Guards the fix for issue #413: Scene lowers to explicit Steps."""
    scene = Scene(
        name="review",
        roles=[
            Role("coder", "gemini", "flash"),
            Role("reviewer", "claude-agent-acp", "haiku"),
        ],
        turns=[
            Turn("coder"),
            Turn("reviewer", "Review the work."),
        ],
        skills_dir="/scene-skills",
    )

    steps = compile_scenes_to_steps([scene], default_prompt="Solve the task")

    assert [step.id for step in steps] == [
        "scene-0-turn-0-coder",
        "scene-0-turn-1-reviewer",
    ]
    assert scene_step_role(steps[0]).name == "coder"
    assert scene_step_prompt(steps[0]) == "Solve the task"
    assert scene_step_role(steps[1]).name == "reviewer"
    assert scene_step_prompt(steps[1]) == "Review the work."
    assert scene_step_skills_dir(steps[0]) == "/scene-skills"
    assert steps[0].data["type"] == "scene_turn"


def test_scene_compiler_uses_instruction_fallback() -> None:
    """Guards the fix for issue #413: default prompts are resolved at compile time."""
    scene = Scene.single(agent="gemini", model="flash")

    assert (
        scene_step_prompt(compile_scenes_to_steps([scene])[0]) == DEFAULT_SCENE_PROMPT
    )


def test_scene_compiler_rejects_unknown_role() -> None:
    """Guards the fix for issue #413: role validation happens during desugaring."""
    scene = Scene(
        name="bad",
        roles=[Role("agent", "gemini")],
        turns=[Turn("missing", "go")],
    )

    with pytest.raises(ValueError, match="unknown role"):
        compile_scenes_to_steps([scene])


def test_scene_type_has_no_runtime_scheduler_metadata() -> None:
    """Guards the fix for issue #413: Scene carries no scheduler fields."""
    assert "parallel_group" not in {field.name for field in fields(Scene)}


def test_benchflow_no_longer_exports_runtime_scene_api() -> None:
    """Guards the fix for issue #413: runtime Scene surface is not public."""
    for name in (
        "SceneRuntime",
        "SceneRole",
        "Message",
        "MessageTransport",
        "MailboxTransport",
    ):
        assert name not in benchflow.__all__
        assert not hasattr(benchflow, name)


def test_scenes_module_has_no_scheduler_runtime() -> None:
    """Guards the fix for issue #413 against reintroducing a Scene scheduler."""
    tree = ast.parse(Path("src/benchflow/scenes.py").read_text())
    runtime_defs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert "Scene" not in runtime_defs
    assert "MailboxTransport" not in runtime_defs
    assert "MessageTransport" not in runtime_defs
    assert "run" not in runtime_defs

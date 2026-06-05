"""Tests for task-standard prompt composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.scenes import compile_scenes_to_steps, scene_step_prompt
from benchflow.task import TaskDocument, TaskDocumentParseError, TaskRuntimeView
from benchflow.task.prompt_composition import (
    compose_task_prompt,
    prompt_composition_settings,
)

PROMPT_USER_SEMANTICS = Path(
    "docs/examples/task-standard/benchflow-wanted-features/prompt-user-semantics"
)


def test_compose_task_prompt_legacy_fallback_precedence() -> None:
    """Guards legacy turn > scene > role > base precedence without benchflow.prompt."""
    assert compose_task_prompt("base", "role", "scene", "turn") == "turn"
    assert compose_task_prompt("base", "role", "scene", None) == "scene"
    assert compose_task_prompt("base", "role", None, None) == "role"
    assert compose_task_prompt("base", None, None, None) == "base"
    assert compose_task_prompt(None, None, None, None) == ""


def test_compose_task_prompt_append_joins_non_empty_parts() -> None:
    """Guards append composition joins ordered non-empty parts with blank lines."""
    composed = compose_task_prompt(
        "Base prompt.",
        "Role guardrails.",
        "Scene context.",
        "Turn override.",
        composition="append",
        order=["base", "role", "scene", "turn"],
    )

    assert composed == (
        "Base prompt.\n\n"
        "Role guardrails.\n\n"
        "Scene context.\n\n"
        "Turn override."
    )


def test_compose_task_prompt_append_skips_empty_parts() -> None:
    """Guards append composition ignores empty scene/turn sections."""
    composed = compose_task_prompt(
        "Base prompt.",
        "Role guardrails.",
        None,
        None,
        composition="append",
    )

    assert composed == "Base prompt.\n\nRole guardrails."


def test_compose_task_prompt_replace_uses_highest_priority_part() -> None:
    """Guards replace composition picks the highest-priority non-empty part."""
    assert (
        compose_task_prompt(
            "Base prompt.",
            "Role guardrails.",
            "Scene context.",
            None,
            composition="replace",
        )
        == "Scene context."
    )


def test_compose_task_prompt_replace_explicit_turn_uses_only_turn() -> None:
    """Guards explicit replace uses only the inline turn prompt when set."""
    assert (
        compose_task_prompt(
            "Base prompt.",
            "Role guardrails.",
            "Scene context.",
            "Turn override.",
            composition="replace",
            explicit_turn=True,
        )
        == "Turn override."
    )
    assert (
        compose_task_prompt(
            "Base prompt.",
            "Role guardrails.",
            "Scene context.",
            None,
            composition="replace",
            explicit_turn=True,
        )
        == ""
    )


def test_prompt_composition_settings_reads_benchflow_prompt_block() -> None:
    """Guards benchflow.prompt parsing for composition and order."""
    settings = prompt_composition_settings(
        {
            "prompt": {
                "composition": "append",
                "order": ["base", "role", "scene", "turn"],
            }
        }
    )

    assert settings.composition == "append"
    assert settings.order == ("base", "role", "scene", "turn")


def test_prompt_composition_settings_rejects_invalid_composition() -> None:
    """Guards invalid benchflow.prompt.composition values."""
    with pytest.raises(ValueError, match="composition must be"):
        prompt_composition_settings({"prompt": {"composition": "merge"}})


def test_task_document_legacy_scene_prompts_shadow_role_prompts() -> None:
    """Guards legacy fallback behavior when benchflow.prompt is absent."""
    document = TaskDocument.from_text(
        """---
agents:
  roles:
    planner:
      agent: codex
    executor:
      agent: openhands
scenes:
  - name: plan
    roles: [planner]
  - name: execute
    turns:
      - role: executor
---
## prompt

Handle the refund request.

## role:planner

Draft the plan.

## scene:execute

Apply the plan.
"""
    )

    steps = compile_scenes_to_steps(
        document.scenes,
        default_prompt=document.instruction,
    )

    assert [scene_step_prompt(step) for step in steps] == [
        "Draft the plan.",
        "Apply the plan.",
    ]


def test_task_document_append_composes_scene_turn_prompts() -> None:
    """Guards append composition for inline task.md scene turns."""
    document = TaskDocument.from_text(
        """---
benchflow:
  prompt:
    composition: append
    order: [base, role, scene, turn]
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    turns:
      - role: solver
        prompt: Turn-only override.
---
## prompt

Base instructions.

## role:solver

Role guardrails.

## scene:solve

Scene framing.
"""
    )

    turn_prompt = document.scenes[0].turns[0].prompt
    assert turn_prompt == (
        "Base instructions.\n\n"
        "Role guardrails.\n\n"
        "Scene framing.\n\n"
        "Turn-only override."
    )


def test_task_document_replace_explicit_turn_prompt() -> None:
    """Guards replace composition with an explicit inline turn prompt."""
    document = TaskDocument.from_text(
        """---
benchflow:
  prompt:
    composition: replace
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    turns:
      - role: solver
        prompt: Use only this turn prompt.
---
## prompt

Base instructions.

## role:solver

Role guardrails.

## scene:solve

Scene framing.
"""
    )

    assert document.scenes[0].turns[0].prompt == "Use only this turn prompt."


def test_task_document_rejects_invalid_prompt_settings() -> None:
    """Guards task.md parsing against invalid benchflow.prompt values."""
    with pytest.raises(TaskDocumentParseError, match="composition must be"):
        TaskDocument.from_text(
            """---
benchflow:
  prompt:
    composition: merge
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    turns:
      - role: solver
---
## prompt

Solve it.
"""
        )


def test_prompt_user_semantics_dogfood_append_scene_prompts() -> None:
    """Guards prompt-user-semantics append composition for scene turns."""
    document = TaskDocument.from_path(PROMPT_USER_SEMANTICS / "task.md")
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


def test_prompt_user_semantics_dogfood_materialize_instruction_md() -> None:
    """Guards prompt-user-semantics base-only /instruction.md materialization."""
    view = TaskRuntimeView.from_task_dir(PROMPT_USER_SEMANTICS)

    assert view.materialize_instruction_md() == view.instruction_text

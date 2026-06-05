"""Tests for the unified task.md authoring document."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.learner_memory import expected_skills_for_task
from benchflow.rollout import RolloutConfig, _resolve_prompts
from benchflow.scenes import compile_scenes_to_steps, scene_step_prompt, scene_step_role
from benchflow.task import Task, TaskConfig, TaskDocument, TaskDocumentParseError
from benchflow.task.config import (
    MultiStepRewardStrategy,
    NetworkMode,
    TaskOS,
    VerifierEnvironmentMode,
)
from benchflow.task.document import render_task_md_from_legacy
from benchflow.task.paths import TaskPaths


def test_task_document_preserves_demo_task_config_and_prompt() -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against demo drift."""
    demo_task = Path("src/benchflow/demo_task")
    legacy_config = TaskConfig.model_validate_toml(
        (demo_task / "task.toml").read_text()
    )

    document = TaskDocument.from_text(render_task_md_from_legacy(demo_task))

    assert document.config.model_dump() == legacy_config.model_dump()
    assert document.instruction == (demo_task / "instruction.md").read_text().strip()


def test_task_document_parses_roles_scenes_and_user_persona() -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike for teams/scenes."""
    document = TaskDocument.from_text(
        """---
version: "1.0"
metadata:
  author_name: benchflow
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 2
  memory_mb: 4096
agents:
  roles:
    planner:
      agent: codex
      model: gpt-5.5
      capabilities: [tool-use]
    executor:
      agent: openhands
scenes:
  - name: plan
    roles: [planner]
    turns:
      - role: planner
  - name: execute
    turns:
      - role: executor
user:
  model: claude-haiku
  stop_rule: done-or-3-rounds
---
# Refund triage

## prompt

Handle the refund request.

## role:planner

Draft the plan.

## scene:execute

Apply the plan.

## user-persona

Push for clarification when the agent skips order details.
"""
    )

    assert document.instruction == "Handle the refund request."
    assert document.config.environment.memory_mb == 4096
    assert document.roles["planner"].capabilities == ["tool-use"]
    assert document.user["stop_rule"] == "done-or-3-rounds"
    assert document.user_persona == (
        "Push for clarification when the agent skips order details."
    )

    steps = compile_scenes_to_steps(
        document.scenes, default_prompt=document.instruction
    )
    assert [scene_step_prompt(step) for step in steps] == [
        "Draft the plan.",
        "Apply the plan.",
    ]


def test_task_document_frontmatter_matches_current_harbor_task_config_surface() -> None:
    """Guards commit 67378ddd's 2026-06-04 parity pass for task.md."""
    document = TaskDocument.from_text(
        """---
schema_version: "1.3"
source: harbor/parity
multi_step_reward_strategy: final
artifacts:
  - source: /logs/artifacts
    destination: artifacts
task:
  name: benchflow/harbor-parity
  description: Exercises Harbor-compatible task.toml fields
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [parity, task-md]
metadata:
  category: schema
  custom:
    kept: true
agent:
  timeout_sec: 120
  user: agent
  network_mode: allowlist
  allowed_hosts: [api.example.com]
verifier:
  timeout_sec: 60
  env:
    JUDGE_API_KEY: ${JUDGE_API_KEY:-test}
  user: root
  network_mode: public
  environment_mode: separate
  pytest_plugins: [pytest_playwright]
  hardening:
    cleanup_conftests: false
  environment:
    docker_image: ghcr.io/example/grader:latest
    cpus: 2
    memory_mb: 1024
    network_mode: no-network
environment:
  network_mode: allowlist
  allowed_hosts: [datasets.example.com]
  build_timeout_sec: 600
  docker_image: ghcr.io/example/task:latest
  os: linux
  cpus: 4
  memory_mb: 4096
  storage_mb: 8192
  gpus: 1
  gpu_types: [T4, A100]
  env:
    DATASET: ${DATASET:-sample}
  skills_dir: /skills
  workdir: /workspace
  tpu:
    type: v6e
    topology: 2x4
  healthcheck:
    command: python -m app.healthcheck
    interval_sec: 2
    timeout_sec: 10
    retries: 5
oracle:
  env:
    SOLUTION_MODE: oracle
steps:
  - name: scaffold
    min_reward: 0.5
    artifacts:
      - source: /app/scaffold.txt
    agent:
      timeout_sec: 30
    verifier:
      timeout_sec: 15
      env:
        STEP: scaffold
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
user:
  model: gemini-2.5-flash
---
## prompt

Solve the parity task.
"""
    )

    cfg = document.config
    assert document.instruction == "Solve the parity task."
    assert document.user["model"] == "gemini-2.5-flash"
    assert "agents" not in cfg.model_dump()
    assert cfg.schema_version == "1.3"
    assert cfg.task is not None
    assert cfg.task.name == "benchflow/harbor-parity"
    assert cfg.metadata["custom"]["kept"] is True
    assert cfg.agent.network_mode == NetworkMode.ALLOWLIST
    assert cfg.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
    assert cfg.verifier.hardening.cleanup_conftests is False
    assert cfg.verifier.environment is not None
    assert cfg.verifier.environment.allow_internet is False
    assert cfg.environment.network_mode == NetworkMode.ALLOWLIST
    assert cfg.environment.os == TaskOS.LINUX
    assert cfg.environment.tpu is not None
    assert cfg.environment.tpu.chip_count == 8
    assert cfg.environment.healthcheck is not None
    assert cfg.environment.healthcheck.retries == 5
    assert cfg.solution.env == {"SOLUTION_MODE": "oracle"}
    assert cfg.multi_step_reward_strategy == MultiStepRewardStrategy.FINAL
    assert cfg.artifacts[0].source == "/logs/artifacts"
    assert cfg.steps is not None
    assert cfg.steps[0].artifacts[0].source == "/app/scaffold.txt"
    assert [scene.name for scene in document.scenes] == ["solve"]


def test_task_document_rejects_unknown_task_config_fields() -> None:
    """Guards commit 67378ddd's 2026-06-04 parity pass against lossy parsing."""
    with pytest.raises(ValueError, match="unknown_harbor_field"):
        TaskDocument.from_text(
            """---
environment:
  unknown_harbor_field: true
---
## prompt

Solve it.
"""
        )


def test_task_document_allows_reserved_benchflow_extension_namespace() -> None:
    """Guards commit 67378ddd's task.md standard against root-key creep."""
    document = TaskDocument.from_text(
        """---
schema_version: "1.3"
benchflow:
  document_version: "0.3"
  compatibility:
    harbor:
      export: degraded
agents:
  roles:
    LeadReviewer:
      agent: codex
scenes:
  - name: FinalReview
    roles: [LeadReviewer]
---
## prompt

Review the submission.

## role:LeadReviewer

Preserve reviewer-only guardrails.

## scene:FinalReview

Run the final review pass.
"""
    )

    assert document.benchflow["document_version"] == "0.3"
    assert "benchflow" not in document.config.model_dump()
    assert "LeadReviewer" in document.role_prompts
    assert "FinalReview" in document.scene_prompts


def test_task_document_rejects_oracle_solution_alias_collision() -> None:
    """Guards commit 67378ddd's oracle naming against alias import drift."""
    with pytest.raises(ValueError, match="cannot contain both 'oracle'"):
        TaskDocument.from_text(
            """---
oracle:
  env:
    MODE: native
solution:
  env:
    MODE: legacy
---
## prompt

Solve it.
"""
        )


def test_task_config_rejects_toml_oracle_solution_alias_collision() -> None:
    """Guards commit 67378ddd's oracle naming in TOML import paths."""
    with pytest.raises(ValueError, match="cannot contain both 'oracle'"):
        TaskConfig.model_validate_toml(
            """
[oracle]
MODE = "native"

[solution]
MODE = "legacy"
"""
        )


def test_task_document_mixed_case_section_ids_compile_to_scenes() -> None:
    """Guards commit 67378ddd's task.md section ids against lowercasing."""
    document = TaskDocument.from_text(
        """---
agents:
  roles:
    LeadReviewer:
      agent: codex
scenes:
  - name: FinalReview
    roles: [LeadReviewer]
---
## prompt

Review the submission.

## role:LeadReviewer

Preserve reviewer-only guardrails.

## scene:FinalReview

Run the final review pass.
"""
    )

    steps = compile_scenes_to_steps(document.scenes, default_prompt=document.instruction)

    assert [scene_step_role(step).name for step in steps] == ["LeadReviewer"]
    assert [scene_step_prompt(step) for step in steps] == ["Run the final review pass."]


def test_task_document_rejects_unknown_scene_roles() -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against bad roles."""
    with pytest.raises(TaskDocumentParseError, match="unknown role"):
        TaskDocument.from_text(
            """---
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: broken
    turns:
      - role: reviewer
---
Solve it.
"""
        )


def test_task_loads_task_md_without_legacy_pair(tmp_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against coupling."""
    task_dir = tmp_path / "task-md-only"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "verifier").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "verifier" / "test.sh").write_text("exit 0\n")
    (task_dir / "task.md").write_text(
        """---
version: "1.0"
metadata:
  category: demo
agent:
  timeout_sec: 120
environment:
  cpus: 1
---
## prompt

Create hello.txt.
"""
    )

    task = Task(task_dir)

    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert TaskPaths(task_dir).is_valid()
    assert task.instruction == "Create hello.txt."
    assert task.config.agent.timeout_sec == 120


def test_task_md_memory_expected_skills_are_loaded(tmp_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against fixture bypass."""
    task_dir = tmp_path / "memory-task-md"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "verifier").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "verifier" / "test.sh").write_text("exit 0\n")
    (task_dir / "task.md").write_text(
        """---
version: "1.0"
verifier:
  memory:
    expected_skills:
      - citation-management
      - source-audit
---
## prompt

Improve the skill memory.
"""
    )

    assert expected_skills_for_task(task_dir) == [
        "citation-management",
        "source-audit",
    ]


def test_resolve_prompts_reads_task_md_prompt(tmp_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against prompt coupling."""
    (tmp_path / "task.md").write_text(
        """---
version: "1.0"
---
## prompt

Use task.md for the prompt.
"""
    )

    assert _resolve_prompts(tmp_path, prompts=[None, "custom"]) == [
        "Use task.md for the prompt.",
        "custom",
    ]


def test_rollout_config_from_legacy_uses_task_md_scenes(tmp_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike scene adoption."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    planner:
      agent: codex
      model: gpt-5.5
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

Handle the request.

## role:planner

Draft a plan.

## scene:execute

Apply the plan.
"""
    )

    config = RolloutConfig.from_legacy(
        task_path=tmp_path,
        agent="claude-agent-acp",
        model="ignored",
    )

    assert [scene.name for scene in config.effective_scenes] == ["plan", "execute"]
    assert config.primary_agent == "codex-acp"
    assert config.primary_model == "gpt-5.5"

    steps = compile_scenes_to_steps(
        config.effective_scenes,
        default_prompt="fallback",
    )
    assert [scene_step_prompt(step) for step in steps] == [
        "Draft a plan.",
        "Apply the plan.",
    ]
    assert [scene_step_role(step).agent for step in steps] == [
        "codex-acp",
        "openhands",
    ]


def test_rollout_config_explicit_prompts_override_task_md_scenes(
    tmp_path: Path,
) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike prompt override."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    planner:
      agent: codex
scenes:
  - name: plan
    roles: [planner]
---
## prompt

Document prompt.
"""
    )

    config = RolloutConfig.from_legacy(
        task_path=tmp_path,
        agent="oracle",
        prompts=["Manual prompt."],
    )

    assert len(config.effective_scenes) == 1
    assert config.primary_agent == "oracle"
    steps = compile_scenes_to_steps(config.effective_scenes)
    assert [scene_step_prompt(step) for step in steps] == ["Manual prompt."]


def test_direct_rollout_config_loads_task_md_scenes(tmp_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike direct entrypoint."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
---
## prompt

Solve from the document.
"""
    )

    config = RolloutConfig(task_path=tmp_path)

    assert [scene.name for scene in config.effective_scenes] == ["solve"]
    assert config.primary_agent == "codex-acp"


@pytest.mark.parametrize(
    "example_path",
    sorted(Path("docs/examples/task-md").glob("**/task.md")),
)
def test_task_md_examples_parse(example_path: Path) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md examples against drift."""
    document = TaskDocument.from_path(example_path)

    assert document.instruction


@pytest.mark.parametrize(
    "example_path",
    [
        Path("docs/examples/task-md/multi-scene/task.md"),
        Path("docs/examples/task-md/nudgebench-team/task.md"),
    ],
)
def test_task_md_scene_examples_parse(example_path: Path) -> None:
    """Guards commit 67378ddd's task.md team/scene examples against drift."""
    document = TaskDocument.from_path(example_path)

    assert document.scenes

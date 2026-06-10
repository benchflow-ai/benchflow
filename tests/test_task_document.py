"""Tests for the unified task.md authoring document."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchflow._utils.learner_memory import expected_skills_for_task
from benchflow.rollout import RolloutConfig, Scene, _resolve_prompts
from benchflow.sandbox.user import DocumentNudgeUser, ModelDocumentNudgeUser
from benchflow.scenes import compile_scenes_to_steps, scene_step_prompt, scene_step_role
from benchflow.task import Task, TaskConfig, TaskDocument, TaskDocumentParseError
from benchflow.task.config import (
    MultiStepRewardStrategy,
    NetworkMode,
    TaskOS,
    VerifierEnvironmentMode,
)
from benchflow.task.document import (
    render_normalized_task_md,
    render_task_md,
    render_task_md_from_legacy,
)
from benchflow.task.paths import TaskPaths

FIRST_PARTY_MIXED_TASK_MD_FIXTURES = (
    Path("src/benchflow/demo_task"),
    Path("tests/examples/hello-world-task"),
    Path("tests/conformance/acp_smoke"),
)


def test_task_document_preserves_demo_task_config_and_prompt() -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md spike against demo drift."""
    demo_task = Path("src/benchflow/demo_task")
    legacy_config = TaskConfig.model_validate_toml(
        (demo_task / "task.toml").read_text()
    )

    document = TaskDocument.from_text(render_task_md_from_legacy(demo_task))

    assert document.config.model_dump() == legacy_config.model_dump()
    assert document.instruction == (demo_task / "instruction.md").read_text().strip()


def test_render_task_md_from_legacy_emits_only_declared_frontmatter(
    tmp_path: Path,
) -> None:
    """Migration must not materialize runtime defaults the author never wrote."""
    (tmp_path / "task.toml").write_text(
        '[metadata]\ndifficulty = "easy"\n\n'
        "[agent]\ntimeout_sec = 120\n\n"
        "[environment]\ncpus = 2\n"
    )
    (tmp_path / "instruction.md").write_text("Do the declared thing.\n")

    rendered = render_task_md_from_legacy(tmp_path)
    frontmatter = yaml.safe_load(rendered.split("---\n")[1])

    assert frontmatter == {
        "schema_version": "1.3",
        "metadata": {"difficulty": "easy"},
        "agent": {"timeout_sec": 120},
        "environment": {"cpus": 2},
    }


def test_render_task_md_from_legacy_canonicalizes_version_key(
    tmp_path: Path,
) -> None:
    """Generated documents pin schema_version; 'version' stays a parse alias."""
    (tmp_path / "task.toml").write_text('version = "1.0"\n')
    (tmp_path / "instruction.md").write_text("Do it.\n")

    rendered = render_task_md_from_legacy(tmp_path)
    frontmatter = yaml.safe_load(rendered.split("---\n")[1])

    assert frontmatter == {"schema_version": "1.0"}
    assert TaskDocument.from_text(rendered).config.schema_version == "1.0"


def test_task_document_accepts_legacy_version_alias_on_parse() -> None:
    document = TaskDocument.from_text('---\nversion: "1.0"\n---\nDo it.\n')

    assert document.config.schema_version == "1.0"


@pytest.mark.parametrize(
    "task_dir",
    FIRST_PARTY_MIXED_TASK_MD_FIXTURES,
    ids=lambda path: path.name,
)
def test_first_party_task_md_fixtures_match_legacy_aliases(task_dir: Path) -> None:
    """Guards PR #1's first-party mixed task.md compatibility dogfood."""
    document = TaskDocument.from_path(task_dir / "task.md")
    legacy_config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())

    assert document.config.model_dump() == legacy_config.model_dump()
    assert document.instruction == (task_dir / "instruction.md").read_text().strip()


def test_task_document_minimal_profile_authoring_parses() -> None:
    """Guards commit 00b32e2a's handoff goal for tiny task.md authoring."""
    document = TaskDocument.from_text(
        """---
profile: code-change
name: runtime-capability-gate
image: ghcr.io/example/task:latest
verifier: verifier/
oracle: oracle/
---
Implement fail-closed runtime capability validation.
"""
    )

    assert document.config.task is not None
    assert document.config.task.name == "benchflow/runtime-capability-gate"
    assert document.config.environment.docker_image == "ghcr.io/example/task:latest"
    assert document.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert document.config.verifier.timeout_sec == 1200
    assert document.benchflow["verifier"]["spec"] == "verifier/verifier.md"
    assert document.benchflow["oracle"]["path"] == "oracle/"
    assert (
        document.instruction == "Implement fail-closed runtime capability validation."
    )


def test_task_document_profile_normalization_is_stable() -> None:
    """Guards commit 00b32e2a's handoff goal for canonical generated contracts."""
    source = """---
profile: [code-change, multi-agent]
name: example/runtime-capability-gate
image: ubuntu:24.04
verifier: verifier/
oracle: oracle/
---
## prompt

Implement it.
"""

    normalized = render_normalized_task_md(source)
    normalized_again = render_normalized_task_md(normalized)
    document = TaskDocument.from_text(normalized)

    assert normalized == normalized_again
    assert "profile:" not in normalized
    assert "\nname: example/runtime-capability-gate" not in normalized
    assert "\nimage: ubuntu:24.04" not in normalized
    assert document.frontmatter["benchflow"]["authoring"]["profiles"] == [
        "code-change",
        "multi-agent",
    ]
    assert set(document.roles) == {"architect", "implementer", "reviewer"}


def test_task_document_unknown_profile_fails_closed() -> None:
    """Guards commit 00b32e2a's handoff goal against silent profile fallback."""
    with pytest.raises(TaskDocumentParseError, match=r"unknown task\.md profile"):
        TaskDocument.from_text(
            """---
profile: cargo-cult
name: bad-profile
---
Do it.
"""
        )


def test_task_document_explicit_fields_override_profile_defaults() -> None:
    """Guards commit 00b32e2a's handoff goal for explicit-over-default authoring."""
    document = TaskDocument.from_text(
        """---
profile: code-change
name: override-task
image: ubuntu:24.04
agent:
  timeout_sec: 42
environment:
  docker_image: ghcr.io/example/explicit:latest
  cpus: 9
verifier:
  timeout_sec: 33
---
Do it.
"""
    )

    assert document.config.agent.timeout_sec == 42
    assert document.config.environment.docker_image == "ghcr.io/example/explicit:latest"
    assert document.config.environment.cpus == 9
    assert document.config.verifier.timeout_sec == 33


def test_task_document_omitted_verifier_timeout_inherits_default() -> None:
    """A task.md with no verifier block inherits the 600s verifier budget."""
    document = TaskDocument.from_text(
        """---
name: default-budget
image: ubuntu:24.04
---
Do it.
"""
    )

    assert document.config.verifier.timeout_sec == 600.0


def test_task_document_zero_second_verifier_budget_fails_closed() -> None:
    """An explicit 0s verifier budget is a parse error, not an instant timeout."""
    with pytest.raises(ValueError, match=r"verifier\.timeout_sec"):
        TaskDocument.from_text(
            """---
name: zero-budget
image: ubuntu:24.04
verifier:
  timeout_sec: 0
---
Do it.
"""
        )


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

    steps = compile_scenes_to_steps(
        document.scenes, default_prompt=document.instruction
    )

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


def test_rollout_config_uses_task_package_prompt_plan_for_append(
    tmp_path: Path,
) -> None:
    """Native append prompt policies must reach executable rollout scene prompts."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    turns:
      - role: solver
        prompt: Turn instruction.
benchflow:
  prompt:
    composition: append
    order: [base, role, scene, turn]
---
## prompt

Base instruction.

## role:solver

Role guardrail.

## scene:solve

Scene context.
"""
    )

    config = RolloutConfig.from_legacy(
        task_path=tmp_path,
        agent="claude-agent-acp",
        model="ignored",
    )
    steps = compile_scenes_to_steps(
        config.effective_scenes,
        default_prompt="fallback",
    )

    assert [scene_step_prompt(step) for step in steps] == [
        "Base instruction.\n\nRole guardrail.\n\nScene context.\n\nTurn instruction."
    ]


def test_rollout_config_compiles_supported_document_user_runtime(
    tmp_path: Path,
) -> None:
    """Supported task.md user metadata becomes a concrete RolloutConfig user."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
user:
  model: scripted
  stop_rule: satisfied-or-3-rounds
  private_facts:
    hidden_need: Use the quarterly file.
benchflow:
  nudges:
    mode: simulated-user
    nudge_budget: 2
---
## prompt

Base instruction.

## user-persona

Reveal private facts only after targeted clarification.
"""
    )

    config = RolloutConfig.from_legacy(task_path=tmp_path)

    assert isinstance(config.user, DocumentNudgeUser)
    assert config.max_user_rounds == 2


def test_rollout_config_compiles_model_document_user_runtime(
    tmp_path: Path,
) -> None:
    """Bounded model-backed task.md users become concrete runtime users."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
user:
  model: claude-haiku
  stop_rule: satisfied-or-3-rounds
  private_facts:
    hidden_need: Use the quarterly file.
benchflow:
  nudges:
    mode: simulated-user
    nudge_budget: 2
    branchable: true
    confirmation_policy:
      destructive_actions: human
---
## prompt

Base instruction.

## user-persona

Reveal private facts only after targeted clarification.
"""
    )

    config = RolloutConfig.from_legacy(task_path=tmp_path)

    assert isinstance(config.user, ModelDocumentNudgeUser)
    assert config.max_user_rounds == 2
    assert config.user.confirmation_policy == "human"
    assert config.user.branchable is True
    assert config.user.branch_execution == "option-kinds-preserved"


def test_rollout_config_compiles_multi_scene_document_user_runtime(
    tmp_path: Path,
) -> None:
    """Linear multi-scene task.md users compile when every scene is single-role."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    planner:
      agent: codex
    implementer:
      agent: codex
scenes:
  - name: plan
    turns:
      - role: planner
        prompt: Plan the work.
  - name: implement
    turns:
      - role: implementer
        prompt: Apply the plan.
user:
  model: scripted
  stop_rule: satisfied-or-3-rounds
  private_facts:
    hidden_need: Use the quarterly file.
benchflow:
  prompt:
    composition: append
    order: [base, scene, turn]
  nudges:
    mode: simulated-user
    nudge_budget: 3
---
## prompt

Base instruction.

## scene:implement

Implementation context.

## user-persona

Reveal private facts only after targeted clarification.
"""
    )

    config = RolloutConfig.from_legacy(task_path=tmp_path)
    steps = compile_scenes_to_steps(
        config.effective_scenes,
        default_prompt="fallback",
    )

    assert isinstance(config.user, DocumentNudgeUser)
    assert config.max_user_rounds == 3
    assert [scene.name for scene in config.effective_scenes] == ["plan", "implement"]
    assert [scene_step_prompt(step) for step in steps] == [
        "Base instruction.\n\nPlan the work.",
        "Base instruction.\n\nImplementation context.\n\nApply the plan.",
    ]


def test_rollout_config_compiles_sequential_team_handoff_runtime(
    tmp_path: Path,
) -> None:
    """Explicit multi-role scenes compile when sequential shared handoff is set."""
    (tmp_path / "task.md").write_text(
        """---
agents:
  roles:
    planner:
      agent: codex
    implementer:
      agent: codex
scenes:
  - name: shared-work
    turns:
      - role: planner
        prompt: Plan the work.
      - role: implementer
        prompt: Apply the plan.
user:
  model: scripted
  stop_rule: satisfied-or-3-rounds
  private_facts:
    hidden_need: Use the quarterly file.
benchflow:
  teams:
    build_review:
      handoff:
        mode: sequential
        workspace_visibility: shared
        trajectory_visibility: metadata
  nudges:
    mode: simulated-user
    nudge_budget: 3
---
## prompt

Base instruction.

## user-persona

Reveal private facts only after targeted clarification.
"""
    )

    config = RolloutConfig.from_legacy(task_path=tmp_path)
    steps = compile_scenes_to_steps(config.effective_scenes)

    assert isinstance(config.user, DocumentNudgeUser)
    assert config.user.handoff_kind == "sequential-shared"
    assert config.user.handoff_team == "build_review"
    assert config.max_user_rounds == 3
    assert [scene_step_prompt(step) for step in steps] == [
        "Plan the work.",
        "Apply the plan.",
    ]


def test_rollout_config_explicit_scenes_skip_document_user_runtime(
    tmp_path: Path,
) -> None:
    """Programmatic scene callers opt out of document-declared user loops."""
    (tmp_path / "task.md").write_text(
        """---
user:
  model: scripted
  private_facts:
    hidden_need: Use the quarterly file.
---
## prompt

Base instruction.
"""
    )

    config = RolloutConfig(
        task_path=tmp_path,
        scenes=[Scene.single(agent="codex", model="gpt-5.5")],
    )

    assert config.user is None


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


def test_render_task_md_maps_solution_alias_to_oracle() -> None:
    """The shared converter renderer emits legacy ``solution`` as native ``oracle``."""
    rendered = render_task_md(
        {
            "schema_version": "1.3",
            "task": {"name": "benchflow/aliased"},
            "solution": {"env": {"SEED": "1"}},
        },
        "Do the thing.",
    )

    assert "\noracle:\n" in rendered
    assert "\nsolution:\n" not in rendered
    document = TaskDocument.from_text(rendered)
    assert document.config.solution is not None
    assert document.config.solution.env == {"SEED": "1"}


def test_render_task_md_rejects_oracle_solution_collision() -> None:
    """Declaring both ``oracle`` and ``solution`` is rejected, matching TaskConfig."""
    with pytest.raises(ValueError, match="both 'oracle' and 'solution'"):
        render_task_md(
            {
                "task": {"name": "benchflow/collision"},
                "oracle": {"env": {}},
                "solution": {"env": {}},
            },
            "Do the thing.",
        )


def test_render_task_md_escapes_reserved_headings_and_round_trips() -> None:
    """Reserved section headings in a prompt survive as prompt text, not new sections."""
    instruction = (
        "Reproduce the report below.\n\n"
        "## prompt\n"
        "## role:reviewer\n"
        "## scene:setup\n"
        "## user-persona\n\n"
        "Then finish."
    )
    rendered = render_task_md({"task": {"name": "benchflow/escaped"}}, instruction)

    assert "\\## prompt" in rendered
    assert "\\## role:reviewer" in rendered
    document = TaskDocument.from_text(rendered)
    assert document.instruction == instruction
    assert document.roles == {}


def test_render_task_md_preserves_prompt_body() -> None:
    """Code fences and internal blank lines round-trip through the renderer."""
    instruction = "Implement:\n\n```python\ndef f():\n    return 1\n```\n\nDone."
    document = TaskDocument.from_text(
        render_task_md({"task": {"name": "benchflow/body"}}, instruction)
    )

    assert document.instruction == instruction


def test_render_task_md_dict_and_toml_text_are_equivalent() -> None:
    """The renderer accepts a config dict or raw TOML text and agrees on output."""
    toml_text = (
        'schema_version = "1.3"\n\n[task]\nname = "benchflow/parity"\n\n'
        "[agent]\ntimeout_sec = 600\n"
    )
    from_dict = render_task_md(
        {
            "schema_version": "1.3",
            "task": {"name": "benchflow/parity"},
            "agent": {"timeout_sec": 600},
        },
        "Do it.",
    )
    from_text = render_task_md(toml_text, "Do it.")

    assert from_dict == from_text


def test_sidecar_prompt_files_are_the_native_role_scene_authoring_surface(
    tmp_path: Path,
) -> None:
    """Roles/scenes/persona load from prompts/*.md so the body stays one prompt."""
    (tmp_path / "task.md").write_text(
        "---\ntask:\n  name: demo/clean-body\n---\n\n"
        "Fix the failing test in app.py and explain the root cause.\n"
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "role.reviewer.md").write_text("Be strict about edge cases.")
    (prompts / "scene.investigate.md").write_text("Reproduce before fixing.")
    (prompts / "user-persona.md").write_text("A terse senior engineer.")

    doc = TaskDocument.from_path(tmp_path / "task.md")

    # base prompt is the bare body — no ## prompt heading required
    assert (
        doc.instruction == "Fix the failing test in app.py and explain the root cause."
    )
    assert doc.role_prompts == {"reviewer": "Be strict about edge cases."}
    assert doc.scene_prompts == {"investigate": "Reproduce before fixing."}
    assert doc.user_persona == "A terse senior engineer."


def test_sidecar_prompt_files_take_precedence_over_legacy_headings(
    tmp_path: Path,
) -> None:
    """The compat ## role: heading is overridden by the native sidecar file."""
    (tmp_path / "task.md").write_text(
        "---\ntask:\n  name: demo/precedence\n---\n\n"
        "Base prompt.\n\n## role:reviewer\n\nHeading version.\n"
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "role.reviewer.md").write_text("File version wins.")

    doc = TaskDocument.from_path(tmp_path / "task.md")

    assert doc.instruction == "Base prompt."
    assert doc.role_prompts == {"reviewer": "File version wins."}

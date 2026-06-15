"""Tests for runtime task views and fail-closed capability validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from benchflow.sandbox.setup import _create_environment
from benchflow.task import (
    Task,
    TaskConfig,
    TaskRuntimeView,
    UnsupportedTaskFeatureError,
    validate_task_runtime_support,
)


def _write_environment(task_dir: Path) -> None:
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")


def _write_verifier(task_dir: Path, dirname: str = "verifier") -> None:
    verifier_dir = task_dir / dirname
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")


def _write_oracle(task_dir: Path, dirname: str = "oracle") -> None:
    oracle_dir = task_dir / dirname
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solve.md").write_text("reference answer\n")


def _write_task_md(
    task_dir: Path,
    *,
    frontmatter: str = "",
    prompt: str = "Create the requested file.",
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_environment(task_dir)
    _write_verifier(task_dir)
    _write_oracle(task_dir)
    extra = f"{frontmatter.rstrip()}\n" if frontmatter.strip() else ""
    (task_dir / "task.md").write_text(
        f"""---
schema_version: "1.3"
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  network_mode: no-network
{extra}---

## prompt

{prompt}
"""
    )


def _write_legacy_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_environment(task_dir)
    _write_verifier(task_dir, "tests")
    _write_oracle(task_dir, "solution")
    (task_dir / "task.toml").write_text(
        dedent(
            """\
            schema_version = "1.3"

            [agent]
            timeout_sec = 300

            [verifier]
            timeout_sec = 120

            [environment]
            network_mode = "no-network"
            """
        )
    )
    (task_dir / "instruction.md").write_text("Legacy prompt\n")


def test_runtime_view_selects_task_md_and_native_aliases(tmp_path: Path) -> None:
    """task.md packages expose one runtime view with native verifier/oracle paths."""
    task_dir = tmp_path / "native"
    _write_task_md(task_dir, prompt="Native prompt")
    _write_verifier(task_dir, "tests")
    _write_oracle(task_dir, "solution")

    view = TaskRuntimeView.from_task_dir(task_dir)

    assert view.entrypoint == "task.md"
    assert view.prompt == "Native prompt"
    assert view.verifier_dir == task_dir / "verifier"
    assert view.oracle_dir == task_dir / "oracle"
    assert view.compatibility.selected_entrypoint == "task.md"
    assert view.compatibility.uses_native_verifier_dir is True
    assert view.compatibility.has_legacy_tests_alias is True
    assert "task.md" in view.source_hashes
    assert "verifier/test.sh" in view.source_hashes
    assert "tests/test.sh" in view.source_hashes


def test_runtime_view_preserves_legacy_split_packages(tmp_path: Path) -> None:
    """Legacy Harbor split packages remain executable compatibility input."""
    task_dir = tmp_path / "legacy"
    _write_legacy_task(task_dir)

    view = TaskRuntimeView.from_task_dir(task_dir)

    assert view.entrypoint == "legacy"
    assert view.prompt == "Legacy prompt\n"
    assert view.verifier_dir == task_dir / "tests"
    assert view.oracle_dir == task_dir / "solution"
    assert view.compatibility.has_task_md is False
    assert view.compatibility.has_legacy_definition is True
    assert view.source_hashes["task.toml"]
    assert view.source_hashes["instruction.md"]


def test_validator_reports_allowlist_as_runtime_gap() -> None:
    """Parsed allowlists are invalid for launch until a sandbox enforces them."""
    config = TaskConfig.model_validate(
        {
            "agent": {
                "network_mode": "allowlist",
                "allowed_hosts": ["api.example.com"],
            },
            "environment": {
                "network_mode": "allowlist",
                "allowed_hosts": ["repo.example.com"],
            },
        }
    )

    issues = validate_task_runtime_support(config, sandbox="docker")

    assert [issue.path for issue in issues] == [
        "agent.network_mode",
        "environment.network_mode",
    ]


def test_validator_reports_unknown_sandbox_backend() -> None:
    """Runtime-capability validation cannot greenlight typoed backends."""
    config = TaskConfig.model_validate({"environment": {"network_mode": "no-network"}})

    issues = validate_task_runtime_support(config, sandbox="not-a-backend")

    assert [(issue.path, issue.reason) for issue in issues] == [
        (
            "sandbox",
            "unknown sandbox backend; use docker, daytona, or modal",
        )
    ]


def test_validator_reports_selected_script_missing_interpreter_artifact(
    tmp_path: Path,
) -> None:
    """Commands like `bash missing.sh` still reference verifier-local files."""
    task_dir = tmp_path / "missing-script"
    _write_task_md(task_dir)
    (task_dir / "verifier" / "verifier.md").write_text(
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
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert any(
        issue.path == "verifier.strategies.deterministic"
        and "script artifact not found" in issue.reason
        and "missing.sh" in issue.reason
        for issue in issues
    )


def test_validator_reports_selected_llm_judge_missing_files(
    tmp_path: Path,
) -> None:
    """Runtime-capability should catch missing selected judge files early."""
    task_dir = tmp_path / "missing-llm-files"
    _write_task_md(task_dir)
    (task_dir / "verifier" / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: judge
              strategies:
                judge:
                  type: llm-judge
                  rubric: rubrics/missing.toml
                  context_file: rubrics/context.md
            ---
            """
        )
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    reasons = [
        issue.reason for issue in issues if issue.path == "verifier.strategies.judge"
    ]
    assert any("llm-judge rubric not found" in reason for reason in reasons)
    assert any("llm-judge context file not found" in reason for reason in reasons)


def test_validator_reports_selected_reward_kit_without_runner(
    tmp_path: Path,
) -> None:
    """Selected Reward Kit strategies need a concrete package runner."""
    task_dir = tmp_path / "rewardkit"
    _write_task_md(task_dir)
    (task_dir / "verifier" / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: rewardkit
              strategies:
                rewardkit:
                  type: reward-kit
                  root: reward_kit/
            ---
            """
        )
    )

    task = Task(task_dir)
    assert task.document is not None
    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert any(
        issue.path == "verifier.strategies.rewardkit"
        and "runner not found" in issue.reason
        for issue in issues
    )


def test_validator_allows_selected_reward_kit_with_runner(tmp_path: Path) -> None:
    """Reward Kit packages are executable when a safe runner exists."""
    task_dir = tmp_path / "rewardkit-supported"
    _write_task_md(task_dir)
    reward_kit = task_dir / "verifier" / "reward_kit"
    reward_kit.mkdir()
    (reward_kit / "reward.py").write_text("print('score')\n")
    (task_dir / "verifier" / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: rewardkit
              strategies:
                rewardkit:
                  type: reward-kit
                  root: reward_kit/
            ---
            """
        )
    )

    task = Task(task_dir)
    assert task.document is not None
    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert not any(issue.path == "verifier.strategies.rewardkit" for issue in issues)


def test_validator_reports_selected_reward_kit_with_invalid_criteria(
    tmp_path: Path,
) -> None:
    """Selected Reward Kit criteria parse before sandbox launch."""
    task_dir = tmp_path / "bad-rewardkit-criteria"
    _write_task_md(task_dir)
    reward_kit = task_dir / "verifier" / "reward_kit"
    reward_kit.mkdir()
    (reward_kit / "reward.py").write_text("print('score')\n")
    rubric_dir = task_dir / "verifier" / "rubrics"
    rubric_dir.mkdir()
    (rubric_dir / "verifier.toml").write_text(
        """\
[[criteria]]
id = "same"
match_criteria = "A."

[[criteria]]
id = "same"
match_criteria = "B."
"""
    )
    (task_dir / "verifier" / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: rewardkit
              strategies:
                rewardkit:
                  type: reward-kit
                  root: reward_kit/
                  criteria: rubrics/verifier.toml
            ---
            """
        )
    )

    task = Task(task_dir)
    assert task.document is not None
    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    verifier_issue = next(
        issue for issue in issues if issue.path == "verifier.strategies.rewardkit"
    )
    assert "criteria are invalid" in verifier_issue.reason
    assert "duplicate id" in verifier_issue.reason


def test_sandbox_launch_accepts_absolute_workdir(tmp_path: Path) -> None:
    """Supported absolute workdir declarations reach backend construction."""
    task_dir = tmp_path / "supported-workdir"
    _write_task_md(
        task_dir,
        frontmatter="  workdir: /repo",
    )
    task = Task(task_dir)

    with patch("benchflow.sandbox.docker.DockerSandbox") as docker_sandbox:
        result = _create_environment("docker", task, task_dir, "trial", MagicMock())

    docker_sandbox.assert_called_once()
    assert result is docker_sandbox.return_value


def test_sandbox_launch_rejects_unsafe_workdir_before_backend_construction(
    tmp_path: Path,
) -> None:
    """Unsafe workdir declarations fail before Docker/Daytona/Modal launch."""
    task_dir = tmp_path / "unsupported-workdir"
    _write_task_md(
        task_dir,
        frontmatter="  workdir: relative/path",
    )
    task = Task(task_dir)

    with (
        patch("benchflow.sandbox.docker.DockerSandbox") as docker_sandbox,
        pytest.raises(UnsupportedTaskFeatureError) as exc_info,
    ):
        _create_environment("docker", task, task_dir, "trial", MagicMock())

    assert docker_sandbox.call_count == 0
    message = str(exc_info.value)
    assert "environment.workdir" in message
    assert {feature.path for feature in exc_info.value.features} == {
        "environment.workdir",
    }


def test_validator_reports_root_workdir_as_runtime_gap(tmp_path: Path) -> None:
    """Root workdir would make sandbox-user setup chown too much filesystem."""
    task_dir = tmp_path / "root-workdir"
    _write_task_md(
        task_dir,
        frontmatter="  workdir: /",
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert any(
        issue.path == "environment.workdir" and "absolute non-root path" in issue.reason
        for issue in issues
    )


def test_validator_reports_invalid_prompt_policy(tmp_path: Path) -> None:
    """Malformed prompt composition fails closed even though append/replace execute."""
    task_dir = tmp_path / "bad-prompt-policy"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            benchflow:
              prompt:
                composition: merge
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert any(issue.path == "benchflow.prompt" for issue in issues)
    assert any("append or replace" in issue.reason for issue in issues)


def test_validator_allows_supported_benchflow_verifier_metadata(
    tmp_path: Path,
) -> None:
    """Plain verifier package pointers are metadata for already-supported scripts."""
    task_dir = tmp_path / "verifier-metadata"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            benchflow:
              verifier:
                spec: verifier/verifier.md
                rubric: verifier/rubrics/verifier.md
                entrypoint: verifier/test.sh
                implementation:
                  type: test-script
                  outputs:
                    reward_json: /logs/verifier/reward.json
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert not any(issue.path.startswith("benchflow.verifier") for issue in issues)


def test_validator_allows_supported_benchflow_verifier_strategies(
    tmp_path: Path,
) -> None:
    """Hybrid verifier metadata no longer rejects executable engines."""
    task_dir = tmp_path / "hybrid-verifier"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            benchflow:
              verifier:
                spec: verifier/verifier.md
                implementation:
                  type: hybrid
                  strategies:
                    deterministic: verifier/test.sh
                    llm_judge: verifier/rubrics/verifier.toml
                    rewardkit: verifier/reward_kit/
                    agent_judge: verifier/judges/reviewer.md
                    ors_episode: trajectory/ors-rewards.jsonl
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert not any(issue.path.startswith("benchflow.verifier") for issue in issues)


def test_validator_allows_selected_ors_episode_verifier_strategy(
    tmp_path: Path,
) -> None:
    """ORS episode strategies now have an executable verifier contract."""
    task_dir = tmp_path / "ors-episode"
    _write_task_md(task_dir)
    (task_dir / "verifier" / "verifier.md").write_text(
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

    task = Task(task_dir)
    assert task.document is not None
    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert not any(issue.path == "verifier.strategies.ors" for issue in issues)


def test_validator_allows_selected_agent_judge_verifier_strategy(
    tmp_path: Path,
) -> None:
    """Verifier-scoped agent-judge strategies are executable runtime semantics."""
    task_dir = tmp_path / "agent-judge"
    _write_task_md(task_dir)
    (task_dir / "verifier" / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: judge
              strategies:
                judge:
                  type: agent-judge
                  role: verifier_judge
                  inputs: [trajectory/acp_trajectory.jsonl]
                  isolation: verifier-only
            ---

            ## role:verifier_judge

            Judge only declared evidence.
            """
        )
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert not any(issue.path == "verifier.strategies.judge" for issue in issues)


def test_validator_allows_supported_document_user_runtime(tmp_path: Path) -> None:
    """Scripted task.md simulated users are executable runtime semantics."""
    task_dir = tmp_path / "scripted-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Reveal private facts only after targeted clarification.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path
        for issue in issues
        if issue.path in {"user", "## user-persona", "benchflow.nudges"}
    } == set()


def test_validator_allows_model_document_user_runtime(tmp_path: Path) -> None:
    """Model-linear task.md users are executable with bounded nudge metadata."""
    task_dir = tmp_path / "model-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Reveal private facts only after targeted clarification.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path
        for issue in issues
        if issue.path in {"user", "## user-persona", "benchflow.nudges"}
    } == set()


def test_validator_allows_multi_scene_document_user_runtime(tmp_path: Path) -> None:
    """Linear multi-scene simulated users execute when each scene is single-role."""
    task_dir = tmp_path / "multi-scene-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
              - name: implement
                turns:
                  - role: implementer
            user:
              model: gemini-2.5-flash
              stop_rule: satisfied-or-3-rounds
              private_facts:
                hidden_need: Use the quarterly file.
            benchflow:
              nudges:
                mode: simulated-user
                nudge_budget: 3
                branchable: true
                confirmation_policy:
                  destructive_actions: human
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Reveal private facts only after targeted clarification.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path
        for issue in issues
        if issue.path in {"user", "## user-persona", "benchflow.nudges"}
    } == set()


def test_validator_allows_sequential_team_handoff_document_user_runtime(
    tmp_path: Path,
) -> None:
    """Sequential shared team handoff is an executable document-user subset."""
    task_dir = tmp_path / "team-handoff-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
              model: claude-haiku
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
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Reveal private facts only after targeted clarification.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path
        for issue in issues
        if issue.path
        in {"user", "## user-persona", "benchflow.nudges", "benchflow.teams"}
    } == set()


def test_validator_rejects_rich_team_handoff_until_enforced(
    tmp_path: Path,
) -> None:
    """Full trajectory sharing and artifacts remain fail-closed team semantics."""
    task_dir = tmp_path / "rich-team-handoff"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
                  - role: implementer
            user:
              model: claude-haiku
              stop_rule: satisfied-or-3-rounds
              private_facts:
                hidden_need: Use the quarterly file.
            benchflow:
              teams:
                build_review:
                  handoff:
                    mode: sequential
                    workspace_visibility: shared
                    trajectory_visibility: full
                    artifacts: [summary.md]
              nudges:
                mode: simulated-user
                nudge_budget: 3
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Reveal private facts only after targeted clarification.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    team_issue = next(issue for issue in issues if issue.path == "benchflow.teams")
    assert "unsupported keys: artifacts" in team_issue.reason
    assert "trajectory_visibility must be none or metadata" in team_issue.reason


def test_validator_rejects_rich_document_user_runtime(tmp_path: Path) -> None:
    """Team-style multi-role user-loop scenes still fail closed."""
    task_dir = tmp_path / "rich-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            agents:
              roles:
                planner:
                  agent: codex
                executor:
                  agent: codex
            scenes:
              - name: plan
                roles: [planner]
              - name: execute
                turns:
                  - role: planner
                  - role: executor
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
            """
        ),
        prompt=dedent(
            """\
            Base instruction.

            ## user-persona

            Ask clarifying questions.
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path
        for issue in issues
        if issue.path in {"user", "## user-persona", "benchflow.nudges"}
    } == {"user", "## user-persona", "benchflow.nudges"}
    user_issue = next(issue for issue in issues if issue.path == "user")
    assert "requires each scene to have exactly one role" in user_issue.reason
    assert "document user model execution is not implemented" not in user_issue.reason


def test_validator_rejects_invalid_document_user_confirmation_policy(
    tmp_path: Path,
) -> None:
    """Invalid human-confirmation metadata fails closed before runtime."""
    task_dir = tmp_path / "bad-confirmation-policy"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
                confirmation_policy: ask-before-retry
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    user_issue = next(issue for issue in issues if issue.path == "user")
    assert "confirmation_policy" in user_issue.reason


def test_validator_rejects_forked_document_user_branch_execution(
    tmp_path: Path,
) -> None:
    """Forked branch execution is not implied by preserving option kinds."""
    task_dir = tmp_path / "forked-branch-user"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
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
                branch_execution: forked-snapshot
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    user_issue = next(issue for issue in issues if issue.path == "user")
    assert "forked branch execution is not implemented" in user_issue.reason
    assert any(issue.path == "benchflow.nudges" for issue in issues)


def test_validator_rejects_malformed_document_user_runtime_types(
    tmp_path: Path,
) -> None:
    """Malformed task.md user/nudge YAML fails closed with precise reasons."""
    task_dir = tmp_path / "bad-user-types"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            user:
              model: 123
              stop_rule: 3
              private_facts:
                hidden_need: Use the quarterly file.
            benchflow:
              nudges:
                mode: 12
                nudge_budget: true
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert {
        issue.path for issue in issues if issue.path in {"user", "benchflow.nudges"}
    } == {
        "user",
        "benchflow.nudges",
    }
    reason = next(issue.reason for issue in issues if issue.path == "user")
    assert "user.model must be a string" in reason
    assert "user.stop_rule must be a string" in reason
    assert "benchflow.nudges.mode must be a string" in reason
    assert "benchflow.nudges.nudge_budget must be an integer" in reason


def test_validator_rejects_non_mapping_nudges(tmp_path: Path) -> None:
    """benchflow.nudges must be a mapping before any runtime can honor it."""
    task_dir = tmp_path / "bad-nudge-shape"
    _write_task_md(
        task_dir,
        frontmatter=dedent(
            """\
            benchflow:
              nudges: []
            """
        ),
    )
    task = Task(task_dir)
    assert task.document is not None

    issues = validate_task_runtime_support(
        task.document,
        sandbox="docker",
        task_dir=task_dir,
    )

    assert any(
        issue.path == "benchflow.nudges"
        and "benchflow.nudges must be a mapping" in issue.reason
        for issue in issues
    )


def test_sandbox_launch_allows_supported_legacy_task(tmp_path: Path) -> None:
    """Supported split-layout tasks still reach backend construction."""
    task_dir = tmp_path / "legacy"
    _write_legacy_task(task_dir)
    task = Task(task_dir)
    rollout_paths = MagicMock()

    with patch("benchflow.sandbox.docker.DockerSandbox") as docker_sandbox:
        result = _create_environment("docker", task, task_dir, "trial", rollout_paths)

    docker_sandbox.assert_called_once()
    assert result is docker_sandbox.return_value

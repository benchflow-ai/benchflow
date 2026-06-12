"""Tests for the package-level task standard boundary."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from benchflow.task import TaskPackage

USER_RUNTIME_TASK_MD_EXAMPLE = Path(
    "docs/examples/task-md/user-runtime/private-facts-nudges"
)


def _write_native_task(task_dir: Path) -> None:
    task_dir.mkdir()
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    oracle = task_dir / "oracle"
    oracle.mkdir()
    (oracle / "solve.md").write_text("reference\n")
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/usr/bin/env bash\necho 1\n")
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
              default_strategy: deterministic
              strategies:
                deterministic:
                  type: script
                  command: ./test.sh
            ---
            """
        )
    )
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/package-boundary
            environment:
              network_mode: no-network
              workdir: /repo
            benchflow:
              prompt:
                composition: append
            ---

            ## prompt

            Build the package boundary.
            """
        )
    )


def test_task_package_centralizes_selected_documents_and_runtime_issues(
    tmp_path: Path,
) -> None:
    """TaskPackage is the package-level view of task.md plus verifier metadata."""
    task_dir = tmp_path / "native"
    _write_native_task(task_dir)

    package = TaskPackage.from_task_dir(task_dir, sandbox="docker")

    assert package.document is not None
    assert package.view.entrypoint == "task.md"
    assert package.view.verifier_document is not None
    assert package.view.verifier_document.selected_strategy.name == "deterministic"
    assert package.view.oracle_dir == task_dir / "oracle"
    assert package.view.verifier_dir == task_dir / "verifier"
    assert package.runtime_supported is True
    assert package.runtime_issues == ()

    payload = package.to_dict()
    assert payload["entrypoint"] == "task.md"
    assert payload["selected_oracle_dir"] == "oracle"
    assert payload["selected_verifier_dir"] == "verifier"
    assert payload["verifier_document"]["selected_strategy"] == {
        "name": "deterministic",
        "type": "script",
    }
    assert payload["runtime_supported"] is True
    assert payload["prompt_plan"]["composition"] == "append"
    assert "task.md" in payload["source_hashes"]


def test_task_package_builds_compatibility_export_report(tmp_path: Path) -> None:
    """Compatibility reports are reachable from the same package boundary."""
    task_dir = tmp_path / "native"
    _write_native_task(task_dir)

    report = TaskPackage.from_task_dir(task_dir).compatibility_export_report()

    assert report.source_task_dir == str(task_dir.resolve())
    assert report.selected_entrypoint == "task.md"
    assert report.selected_oracle_dir == "oracle"
    assert report.selected_verifier_dir == "verifier"
    assert report.status == "degraded"
    assert "task.md" in report.input_hashes


def test_task_package_compiles_append_prompt_plan_and_redacts_user_facts(
    tmp_path: Path,
) -> None:
    """Append composition preserves base, role, scene, and turn prompt parts."""
    task_dir = tmp_path / "prompt-plan"
    _write_native_task(task_dir)
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/prompt-plan
            environment:
              network_mode: no-network
            agents:
              roles:
                solver:
                  agent: codex-acp
            scenes:
              - name: solve
                turns:
                  - role: solver
                    prompt: Turn instruction.
            user:
              model: claude-haiku
              stop_rule: satisfied-or-3-rounds
              private_facts:
                hidden_need: never expose this value
            benchflow:
              prompt:
                composition: append
                order: [base, role, scene, turn]
              nudges:
                mode: simulated-user
                nudge_budget: 3
            ---

            ## prompt

            Base instruction.

            ## role:solver

            Role guardrail.

            ## scene:solve

            Scene context.

            ## user-persona

            Ask for clarification.
            """
        )
    )

    plan = TaskPackage.from_task_dir(task_dir).prompt_plan

    assert plan is not None
    assert plan.composition == "append"
    assert len(plan.turns) == 1
    assert plan.turns[0].prompt == (
        "Base instruction.\n\nRole guardrail.\n\nScene context.\n\nTurn instruction."
    )
    assert [part.kind for part in plan.turns[0].parts] == [
        "base",
        "role",
        "scene",
        "turn",
    ]
    assert plan.user_runtime.status == "supported"
    assert plan.user_runtime.runtime_kind == "model-linear"
    assert plan.user_runtime.private_fact_keys == ("hidden_need",)
    assert "never expose" not in str(plan.to_dict())


def test_task_package_marks_scripted_user_runtime_supported(tmp_path: Path) -> None:
    """Supported task.md user semantics are visible in the package contract."""
    task_dir = tmp_path / "supported-user"
    _write_native_task(task_dir)
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/supported-user
            environment:
              network_mode: no-network
            agents:
              roles:
                solver:
                  agent: codex-acp
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
    )

    plan = TaskPackage.from_task_dir(task_dir).prompt_plan

    assert plan is not None
    assert plan.user_runtime.status == "supported"
    assert plan.user_runtime.max_rounds == 2
    assert plan.user_runtime.private_fact_keys == ("hidden_need",)
    assert plan.user_runtime.persona_present is True
    assert "Use the quarterly file" not in str(plan.to_dict())


def test_task_package_marks_model_user_runtime_supported(tmp_path: Path) -> None:
    """Model-backed user semantics remain redacted in package metadata."""
    task_dir = tmp_path / "model-user"
    _write_native_task(task_dir)
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/model-user
            environment:
              network_mode: no-network
            agents:
              roles:
                solver:
                  agent: codex-acp
            scenes:
              - name: solve
                roles: [solver]
            user:
              model: gemini-2.5-flash
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
    )

    plan = TaskPackage.from_task_dir(task_dir).prompt_plan

    assert plan is not None
    assert plan.user_runtime.status == "supported"
    assert plan.user_runtime.model == "gemini-2.5-flash"
    assert plan.user_runtime.runtime_kind == "model-linear"
    assert plan.user_runtime.branchable is True
    assert plan.user_runtime.branch_execution == "option-kinds-preserved"
    assert plan.user_runtime.confirmation_policy == "human"
    assert plan.user_runtime.private_fact_keys == ("hidden_need",)
    assert "Use the quarterly file" not in str(plan.to_dict())


def test_task_package_marks_sequential_team_handoff_supported(tmp_path: Path) -> None:
    """Supported team handoff metadata is visible without leaking user facts."""
    task_dir = tmp_path / "team-handoff-user"
    _write_native_task(task_dir)
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/team-handoff-user
            environment:
              network_mode: no-network
            agents:
              roles:
                planner:
                  agent: codex-acp
                implementer:
                  agent: codex-acp
            scenes:
              - name: shared-work
                turns:
                  - role: planner
                    prompt: Plan the implementation.
                  - role: implementer
                    prompt: Apply the plan.
            user:
              model: claude-haiku
              stop_rule: satisfied-or-3-rounds
              private_facts:
                hidden_need: never expose this value
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

            Ask for clarification.
            """
        )
    )

    plan = TaskPackage.from_task_dir(task_dir).prompt_plan

    assert plan is not None
    assert plan.user_runtime.status == "supported"
    assert plan.user_runtime.handoff_kind == "sequential-shared"
    assert plan.user_runtime.handoff_team == "build_review"
    assert plan.user_runtime.handoff_workspace_visibility == "shared"
    assert plan.user_runtime.handoff_trajectory_visibility == "metadata"
    assert plan.user_runtime.private_fact_keys == ("hidden_need",)
    assert "never expose" not in str(plan.to_dict())


def test_task_package_compiles_user_runtime_example_and_redacts_private_fact() -> None:
    """Guards PR #1's simulated-user runtime package contract."""
    package = TaskPackage.from_task_dir(USER_RUNTIME_TASK_MD_EXAMPLE, sandbox="docker")

    assert package.runtime_supported is True
    assert package.runtime_issues == ()
    assert package.view.entrypoint == "task.md"
    task = USER_RUNTIME_TASK_MD_EXAMPLE.resolve()
    assert package.view.oracle_dir == task / "oracle"
    assert package.view.verifier_dir == task / "verifier"

    plan = package.prompt_plan
    assert plan is not None
    assert plan.composition == "append"
    assert plan.order == ("base", "role", "scene", "turn")
    assert len(plan.turns) == 2
    assert [turn.scene for turn in plan.turns] == ["triage", "finish"]
    assert [turn.role for turn in plan.turns] == [
        "support_solver",
        "support_solver",
    ]
    assert [part.kind for part in plan.turns[0].parts] == [
        "base",
        "role",
        "scene",
        "turn",
    ]
    assert "Ask the simulated user for the order id" in plan.turns[0].prompt
    assert "Write /workspace/recovery.json" in plan.turns[1].prompt

    assert plan.user_runtime.status == "supported"
    assert plan.user_runtime.model == "scripted"
    assert plan.user_runtime.runtime_kind == "scripted-linear"
    assert plan.user_runtime.max_rounds == 3
    assert plan.user_runtime.private_fact_keys == ("order_id",)
    assert plan.user_runtime.persona_present is True
    assert plan.user_runtime.nudge_mode == "simulated-user"
    assert plan.user_runtime.nudge_budget == 3
    assert "BF-1042" not in str(plan.to_dict())
    assert "BF-1042" not in str(package.to_dict())


def test_task_package_compiles_explicit_replace_prompt_plan(tmp_path: Path) -> None:
    """Explicit replace composition keeps the highest-priority prompt only."""
    task_dir = tmp_path / "replace-plan"
    _write_native_task(task_dir)
    (task_dir / "task.md").write_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            task:
              name: benchflow/replace-plan
            environment:
              network_mode: no-network
            agents:
              roles:
                solver:
                  agent: codex-acp
            scenes:
              - name: solve
                turns:
                  - role: solver
                    prompt: Turn instruction.
            benchflow:
              prompt:
                composition: replace
            ---

            ## prompt

            Base instruction.

            ## role:solver

            Role guardrail.

            ## scene:solve

            Scene context.
            """
        )
    )

    plan = TaskPackage.from_task_dir(task_dir).prompt_plan

    assert plan is not None
    assert plan.composition == "replace"
    assert plan.order == ("turn", "scene", "role", "base")
    assert plan.turns[0].prompt == "Turn instruction."
    assert [part.kind for part in plan.turns[0].parts] == ["turn"]

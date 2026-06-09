"""Tests for native task.md compatibility export loss reports."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow._utils.task_authoring import check_task, migrate_task_to_task_md
from benchflow.task import (
    TaskConfig,
    TaskDocument,
    build_compatibility_export_report,
    build_harbor_roundtrip_conformance_report,
    export_task_to_split_layout,
    render_task_md_from_legacy,
)

REAL_SKILLSBENCH_TASK_MD_EXAMPLES = (
    "3d-scan-calc",
    "citation-check",
    "weighted-gdp-calc",
)
USER_RUNTIME_TASK_MD_EXAMPLE = Path(
    "docs/examples/task-md/user-runtime/private-facts-nudges"
)


def _write_environment(task_dir: Path) -> None:
    env = task_dir / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")


def _write_native_task(task_dir: Path) -> None:
    task_dir.mkdir()
    _write_environment(task_dir)
    oracle = task_dir / "oracle"
    oracle.mkdir()
    (oracle / "solve.md").write_text("native oracle\n")
    verifier = task_dir / "verifier"
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/usr/bin/env bash\necho 1\n")
    (verifier / "verifier.md").write_text(
        dedent(
            """\
            ---
            verifier:
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
              name: benchflow/export-demo
              description: Export demo
            agent:
              timeout_sec: 300
            verifier:
              timeout_sec: 120
            environment:
              network_mode: no-network
            agents:
              roles:
                reviewer:
                  agent: codex
            scenes:
              - name: review
                turns:
                  - role: reviewer
            benchflow:
              teams:
                - reviewer
              nudges:
                mode: simulated-user
            ---

            ## prompt

            Build the exported artifact.

            ## role:reviewer

            Review only the exported compatibility report.

            ## scene:review

            Run the compatibility review.
            """
        )
    )


def _write_legacy_task(task_dir: Path) -> None:
    task_dir.mkdir()
    _write_environment(task_dir)
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
    (task_dir / "instruction.md").write_text("Legacy split prompt.\n")
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/usr/bin/env bash\ntrue\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/usr/bin/env bash\necho 1\n")


def test_export_native_task_to_split_layout_with_loss_report(tmp_path: Path) -> None:
    """Native-only task semantics are exported with an explicit degraded report."""
    task_dir = tmp_path / "native"
    out_dir = tmp_path / "exported"
    _write_native_task(task_dir)

    report = export_task_to_split_layout(task_dir, out_dir, target="harbor")

    assert (out_dir / "task.toml").exists()
    assert (out_dir / "instruction.md").read_text() == "Build the exported artifact.\n"
    assert (out_dir / "environment" / "Dockerfile").exists()
    assert (out_dir / "solution" / "solve.md").read_text() == "native oracle\n"
    assert (out_dir / "tests" / "test.sh").exists()
    assert (out_dir / "tests" / "verifier.md").exists()

    on_disk = json.loads((out_dir / "compatibility" / "export-report.json").read_text())
    assert on_disk == report.to_dict()
    assert report.status == "degraded"
    assert report.selected_entrypoint == "task.md"
    assert report.selected_oracle_dir == "oracle"
    assert report.selected_verifier_dir == "verifier"
    assert {"task.toml", "instruction.md", "tests/test.sh"} <= set(report.emitted_files)

    losses = {loss.path: loss for loss in report.losses}
    assert "agents.roles" in losses
    assert "scenes" in losses
    assert "benchflow.teams" in losses
    assert "benchflow.nudges" in losses
    assert "verifier.verifier_md" in losses


@pytest.mark.parametrize("task_name", REAL_SKILLSBENCH_TASK_MD_EXAMPLES)
def test_export_real_skillsbench_native_examples_to_harbor_split(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Guards PR #1's native SkillsBench packages against export drift."""
    task_dir = Path("docs/examples/task-md/real-skillsbench") / task_name
    out_dir = tmp_path / task_name

    report = export_task_to_split_layout(task_dir, out_dir, target="harbor")

    assert report.status == "degraded"
    assert (out_dir / "task.toml").is_file()
    assert (out_dir / "instruction.md").is_file()
    assert (out_dir / "environment").is_dir()
    assert (out_dir / "tests" / "test.sh").is_file()
    assert (out_dir / "tests" / "test_outputs.py").is_file()
    assert (out_dir / "tests" / "verifier.md").is_file()
    assert (out_dir / "tests" / "rubrics" / "verifier.md").is_file()
    assert (out_dir / "solution" / "solve.sh").is_file()
    assert report.selected_entrypoint == "task.md"
    assert report.selected_oracle_dir == "oracle"
    assert report.selected_verifier_dir == "verifier"
    assert {"task.toml", "instruction.md", "tests/test.sh"} <= set(report.emitted_files)

    losses = {loss.path: loss for loss in report.losses}
    assert "verifier.verifier_md" in losses


def test_export_user_runtime_example_reports_document_only_losses(
    tmp_path: Path,
) -> None:
    """Guards PR #1's simulated-user export loss report coverage."""
    out_dir = tmp_path / "private-facts-nudges"

    report = export_task_to_split_layout(
        USER_RUNTIME_TASK_MD_EXAMPLE,
        out_dir,
        target="harbor",
    )

    assert report.status == "degraded"
    assert (out_dir / "task.toml").is_file()
    assert (out_dir / "instruction.md").is_file()
    assert (out_dir / "solution" / "solve.sh").is_file()
    assert (out_dir / "tests" / "test.sh").is_file()
    assert (out_dir / "tests" / "verifier.md").is_file()

    losses = {loss.path: loss for loss in report.losses}
    assert {
        "agents.roles",
        "scenes",
        "## role:*",
        "## scene:*",
        "user",
        "## user-persona",
        "benchflow.nudges",
        "benchflow.prompt",
        "verifier.verifier_md",
    } <= set(losses)


@pytest.mark.parametrize("task_name", REAL_SKILLSBENCH_TASK_MD_EXAMPLES)
def test_exported_real_skillsbench_examples_migrate_back_to_native_task_md(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Guards PR #1's real SkillsBench export/migrate adoption path."""
    task_dir = Path("docs/examples/task-md/real-skillsbench") / task_name
    out_dir = tmp_path / task_name

    export_task_to_split_layout(task_dir, out_dir, target="harbor")
    result = migrate_task_to_task_md(out_dir, remove_legacy=True)

    assert result.removed_legacy is True
    assert set(result.migrated_legacy_dirs) == {
        "tests/ -> verifier/",
        "solution/ -> oracle/",
    }
    assert (out_dir / "task.md").is_file()
    assert not (out_dir / "task.toml").exists()
    assert not (out_dir / "instruction.md").exists()
    assert not (out_dir / "tests").exists()
    assert not (out_dir / "solution").exists()
    assert (out_dir / "verifier" / "test.sh").is_file()
    assert (out_dir / "verifier" / "verifier.md").is_file()
    assert (out_dir / "verifier" / "rubrics" / "verifier.md").is_file()
    assert (out_dir / "oracle" / "solve.sh").is_file()
    assert check_task(out_dir, validation_level="publication-grade") == []


def test_export_roundtrips_config_and_prompt(tmp_path: Path) -> None:
    """Supported task.md fields round-trip through the split export."""
    task_dir = tmp_path / "native"
    out_dir = tmp_path / "exported"
    _write_native_task(task_dir)

    export_task_to_split_layout(task_dir, out_dir)

    original = TaskDocument.from_path(task_dir / "task.md")
    exported = TaskConfig.model_validate_toml((out_dir / "task.toml").read_text())
    assert exported.model_dump() == original.config.model_dump()
    assert (out_dir / "instruction.md").read_text().strip() == original.instruction


def test_export_rehydrates_preserved_foreign_extensions(tmp_path: Path) -> None:
    """Foreign import extras in task.md survive split compatibility export."""
    task_dir = tmp_path / "native"
    out_dir = tmp_path / "exported"
    _write_native_task(task_dir)
    task_md = task_dir / "task.md"
    task_md_text = task_md.read_text()
    task_md_text = task_md_text.replace(
        "environment:\n  network_mode: no-network\n",
        """environment:
  network_mode: no-network
steps:
  - name: phase-one
""",
    )
    task_md.write_text(
        task_md_text.replace(
            "benchflow:\n  teams:\n",
            """benchflow:
  compat:
    source: harbor
    extra_paths:
      - environment.modal.image
      - harbor_ext
      - steps[0].runner
      - verifier.reward_kit.metric
    extra:
      harbor_ext: kept
      environment:
        modal:
          image: registry.example.com/task:latest
      steps:
        - runner: harbor-step-runner
      verifier:
        reward_kit:
          metric: exact_match
  teams:
""",
        )
    )

    report = export_task_to_split_layout(task_dir, out_dir)

    exported = tomllib.loads((out_dir / "task.toml").read_text())
    assert exported["harbor_ext"] == "kept"
    assert exported["environment"]["modal"] == {
        "image": "registry.example.com/task:latest"
    }
    assert exported["steps"][0]["name"] == "phase-one"
    assert exported["steps"][0]["runner"] == "harbor-step-runner"
    assert exported["verifier"]["reward_kit"] == {"metric": "exact_match"}
    assert report.restored_extension_paths == [
        "environment.modal.image",
        "harbor_ext",
        "steps[0].runner",
        "verifier.reward_kit.metric",
    ]
    loss_paths = {loss.path for loss in report.losses}
    assert "benchflow.compat" not in loss_paths


def test_build_report_detects_alias_collisions(tmp_path: Path) -> None:
    """Mixed native and split aliases cannot silently collapse during export."""
    task_dir = tmp_path / "native"
    _write_native_task(task_dir)
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.md").write_text("legacy solution differs\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/usr/bin/env bash\necho 0\n")

    report = build_compatibility_export_report(task_dir)

    loss_paths = {loss.path for loss in report.losses}
    assert "oracle|solution" in loss_paths
    assert "verifier|tests" in loss_paths


def test_build_report_detects_task_md_instruction_prompt_drift(tmp_path: Path) -> None:
    """Mixed native and split prompt drift must be reported before export."""
    task_dir = tmp_path / "native"
    _write_native_task(task_dir)
    config = TaskDocument.from_path(task_dir / "task.md").config
    (task_dir / "task.toml").write_text(config.model_dump_toml())
    (task_dir / "instruction.md").write_text("Different split prompt.\n")

    report = build_compatibility_export_report(task_dir)

    assert "task.md|instruction.md" in {loss.path for loss in report.losses}


def test_legacy_split_export_is_lossless(tmp_path: Path) -> None:
    """Existing Harbor/Pier split packages remain first-class compatibility input."""
    task_dir = tmp_path / "legacy"
    out_dir = tmp_path / "exported"
    _write_legacy_task(task_dir)

    report = export_task_to_split_layout(task_dir, out_dir)

    assert report.status == "lossless"
    assert report.losses == []
    assert (out_dir / "task.toml").exists()
    assert (out_dir / "instruction.md").read_text() == "Legacy split prompt.\n"
    assert (out_dir / "solution" / "solve.sh").exists()
    assert (out_dir / "tests" / "test.sh").exists()


def test_harbor_roundtrip_migrate_export_preserves_supported_surface(
    tmp_path: Path,
) -> None:
    """Split -> task.md -> split conformance preserves Harbor-compatible files."""
    task_dir = tmp_path / "legacy"
    _write_legacy_task(task_dir)
    (task_dir / "solution" / "notes.txt").write_text("supporting oracle note\n")
    fixtures = task_dir / "tests" / "fixtures"
    fixtures.mkdir()
    (fixtures / "case.json").write_text('{"ok": true}\n')

    report = build_harbor_roundtrip_conformance_report(task_dir)

    assert report.status == "lossless"
    assert report.config_equal is True
    assert report.prompt_equal is True
    assert report.environment_file_map_equal is True
    assert report.solution_file_map_equal is True
    assert report.tests_file_map_equal is True
    assert report.mismatches == []


def test_harbor_roundtrip_reports_restored_foreign_extensions(
    tmp_path: Path,
) -> None:
    """Round-trip conformance tracks preserved Harbor fork config extras."""
    task_dir = tmp_path / "legacy"
    _write_legacy_task(task_dir)
    config_path = task_dir / "task.toml"
    config_path.write_text(
        'harbor_ext = "kept"\n'
        + config_path.read_text()
        + """
[environment.modal]
image = "registry.example.com/task:latest"
"""
    )

    report = build_harbor_roundtrip_conformance_report(task_dir)

    assert report.status == "lossless"
    assert report.restored_extension_paths == [
        "environment.modal.image",
        "harbor_ext",
    ]


def test_migrate_legacy_instruction_with_reserved_headings_preserves_prompt(
    tmp_path: Path,
) -> None:
    """Legacy instructions that mention task.md headings stay plain prompt text."""
    task_dir = tmp_path / "legacy"
    _write_legacy_task(task_dir)
    instruction = dedent(
        """\
        Treat this as literal user content:

        ## role:solver

        Do not reinterpret this legacy heading.
        """
    ).strip()
    (task_dir / "instruction.md").write_text(instruction + "\n")

    document = TaskDocument.from_text(render_task_md_from_legacy(task_dir))
    report = build_harbor_roundtrip_conformance_report(task_dir)

    assert document.instruction == instruction
    assert document.role_prompts == {}
    assert report.status == "lossless"
    assert report.prompt_equal is True


def test_export_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    """Export destinations are explicit so reports cannot merge with stale files."""
    task_dir = tmp_path / "legacy"
    out_dir = tmp_path / "exported"
    _write_legacy_task(task_dir)
    out_dir.mkdir()

    with pytest.raises(FileExistsError):
        export_task_to_split_layout(task_dir, out_dir)

    report = export_task_to_split_layout(task_dir, out_dir, overwrite=True)
    assert report.status == "lossless"

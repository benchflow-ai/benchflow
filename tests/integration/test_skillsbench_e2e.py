"""Tests for the file-driven SkillsBench E2E matrix config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.integration.skillsbench_e2e import (
    DEFAULT_MODEL,
    build_matrix,
    load_config,
    load_manifest,
    materialize_subset,
    registered_matrix_agents,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_CONFIG = REPO_ROOT / "tasks" / "skillsbench-e2e" / "e2e.yaml"


def test_manifest_has_expected_nine_tasks() -> None:
    tasks = load_manifest(REPO_ROOT / "tasks" / "skillsbench-e2e" / "tasks.txt")
    assert tasks == [
        "jax-computing-basics",
        "python-scala-translation",
        "jpg-ocr-stat",
        "grid-dispatch-operator",
        "threejs-to-obj",
        "data-to-d3",
        "lake-warming-attribution",
        "weighted-gdp-calc",
        "shock-analysis-supply",
    ]


def test_e2e_config_builds_all_current_agents_matrix() -> None:
    cfg = load_config(E2E_CONFIG)
    tasks = load_manifest(cfg.tasks_manifest)
    matrix = build_matrix(tasks, cfg.agents, cfg.model, cfg.environment)

    assert cfg.model == DEFAULT_MODEL == "gemini-3.1-flash-lite-preview"
    assert cfg.concurrency == 30
    assert cfg.skills_dir is None
    assert cfg.agents == registered_matrix_agents()
    assert len(matrix) == 9 * len(registered_matrix_agents())
    assert {entry.task_name for entry in matrix} == set(tasks)
    assert {entry.agent for entry in matrix} == set(registered_matrix_agents())


def test_materialize_subset_fails_fast_for_missing_task(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="missing-task"):
        materialize_subset(source, ["missing-task"], tmp_path / "out")


def test_cli_dry_run_writes_matrix_outputs(tmp_path: Path) -> None:
    cfg = tmp_path / "e2e.yaml"
    manifest = tmp_path / "tasks.txt"
    manifest.write_text("task-a\ntask-b\n")
    cfg.write_text(
        """
kind: skillsbench-e2e
source:
  repo: benchflow-ai/skillsbench
  path: tasks
tasks_manifest: tasks.txt
jobs_dir: {jobs_dir}
model: gemini-3.1-flash-lite-preview
environment: daytona
concurrency: 30
agents:
  - gemini
  - opencode
audit:
  audit_agent:
    enabled: false
""".format(jobs_dir=tmp_path / "jobs")
    )

    result = CliRunner().invoke(app, ["eval", "create", "-f", str(cfg), "--dry-run"])

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "jobs").iterdir())
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "matrix_summary.json").read_text())
    assert summary["dry_run"] is True
    assert summary["total"] == 4
    assert {entry["status"] for entry in summary["entries"]} == {"planned"}
    assert (run_dirs[0] / "artifact_audit.json").exists()
    assert (run_dirs[0] / "parity_report.json").exists()
    assert (run_dirs[0] / "findings.md").exists()

"""Regression tests for #398 — Evaluation/Job must accept environment manifests.

Before the fix the batch ``Evaluation`` API could not run manifest-backed
rollouts:

* ``EvaluationConfig`` had no ``environment_manifest`` field;
* ``Evaluation._run_single_task`` built ``RolloutConfig.from_legacy`` without
  one, so the manifest seam disappeared at the Job layer;
* ``bench eval run`` exposed no ``--environment-manifest`` option.

These tests assert the full path: dataclass → CLI flag → YAML loader →
``RolloutConfig.environment_manifest``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.evaluation import Evaluation, EvaluationConfig

_MANIFEST_TOML = """\
[environment]
name = "test-env-398"
image = "example/test:latest"
"""

_MANIFEST = EnvironmentManifest.model_validate_toml(_MANIFEST_TOML)


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text('version = "1.0"\n')


# EvaluationConfig dataclass


def test_evaluation_config_has_environment_manifest_field() -> None:
    """The batch config must carry the manifest the runtime can consume."""
    cfg = EvaluationConfig(environment_manifest=_MANIFEST)
    assert cfg.environment_manifest is _MANIFEST


def test_evaluation_config_environment_manifest_defaults_to_none() -> None:
    """Default keeps batch behaviour for benchmarks that do not need a manifest."""
    cfg = EvaluationConfig()
    assert cfg.environment_manifest is None


# Evaluation._run_single_task threads the manifest into RolloutConfig


async def test_run_single_task_passes_manifest_to_rollout_config(
    tmp_path: Path,
) -> None:
    """The manifest must reach RolloutConfig.environment_manifest (#398).

    The manifest seam is what allows ``Rollout.setup()`` to provision the
    environment-plane container. If the Job layer drops it on the floor,
    the env plane is never exercised — exactly the gap #398 describes.
    """
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    cfg = EvaluationConfig(environment_manifest=_MANIFEST)
    job = Evaluation(tasks_dir=task_dir, jobs_dir=tmp_path / "jobs", config=cfg)

    captured: dict[str, Any] = {}

    async def fake_create(cls_cfg: Any) -> Any:  # type: ignore[no-untyped-def]
        captured["manifest"] = cls_cfg.environment_manifest

        async def fake_run() -> Any:
            from benchflow.models import RolloutResult

            return RolloutResult(
                task_name=task_dir.name,
                rollout_name="rt",
                rewards={"reward": 1.0},
                trajectory=[],
                agent="",
                agent_name="",
                model="",
                n_tool_calls=0,
                n_prompts=0,
                error=None,
                verifier_error=None,
                partial_trajectory=False,
                trajectory_source=None,
                started_at=None,
                finished_at=None,
            )

        return SimpleNamespace(run=fake_run)

    with patch("benchflow.rollout.Rollout.create", new=fake_create):
        await job._run_single_task(task_dir, cfg)

    assert captured["manifest"] is _MANIFEST


# bench eval run --environment-manifest


def test_cli_eval_create_flag_loads_manifest_into_evaluation_config(
    tmp_path: Path,
) -> None:
    """`--environment-manifest <path>` lands on EvaluationConfig (#398)."""
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    manifest_path = tmp_path / "environment.toml"
    manifest_path.write_text(_MANIFEST_TOML)

    captured: dict[str, Any] = {}

    async def fake_run(self: Evaluation) -> Any:
        captured["manifest"] = self._config.environment_manifest
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=fake_run):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--tasks-dir",
                str(task_dir),
                "--environment-manifest",
                str(manifest_path),
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["manifest"], EnvironmentManifest)
    assert captured["manifest"].name == "test-env-398"
    assert captured["manifest"].image == "example/test:latest"


def test_cli_eval_create_missing_manifest_path_exits_nonzero(
    tmp_path: Path,
) -> None:
    """A bogus manifest path must fail loud, not silently disable the seam."""
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--environment-manifest",
            str(tmp_path / "does-not-exist.toml"),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code != 0


# YAML config — environment_manifest support


def test_yaml_loader_parses_environment_manifest(tmp_path: Path) -> None:
    """Native YAML must accept ``environment_manifest: <path>`` (#398).

    Matters because a release-scale evaluation is usually checked into
    Git as YAML; if the manifest seam works on the CLI but not in YAML
    the audit trail diverges.
    """
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    manifest_path = tmp_path / "environment.toml"
    manifest_path.write_text(_MANIFEST_TOML)

    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(
        "tasks_dir: "
        + str(task_dir)
        + "\njobs_dir: "
        + str(tmp_path / "jobs")
        + "\nagent: oracle\nenvironment_manifest: "
        + str(manifest_path)
        + "\n"
    )

    job = Evaluation.from_yaml(yaml_path)
    assert isinstance(job._config.environment_manifest, EnvironmentManifest)
    assert job._config.environment_manifest.name == "test-env-398"

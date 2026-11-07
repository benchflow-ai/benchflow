"""Coverage for paired eval lift reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.eval_lift import build_lift_report

runner = CliRunner()


def _write_result(
    job_dir: Path,
    rollout_name: str,
    *,
    task_id: str,
    reward: float | None = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
    verifier_error: str | None = None,
    export_error: str | None = None,
    healthy: bool | None = None,
    partial_trajectory: bool | None = None,
) -> None:
    rollout_dir = job_dir / rollout_name
    rollout_dir.mkdir(parents=True)
    payload: dict[str, Any] = {"task_name": task_id}
    if reward is not None:
        payload["rewards"] = {"reward": reward}
    if metadata is not None:
        payload["metadata"] = metadata
    if error is not None:
        payload["error"] = error
    if verifier_error is not None:
        payload["verifier_error"] = verifier_error
    if export_error is not None:
        payload["export_error"] = export_error
    if healthy is not None:
        payload["healthy"] = healthy
    if partial_trajectory is not None:
        payload["partial_trajectory"] = partial_trajectory
    (rollout_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_lift_fixture(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline-job"
    trained = tmp_path / "trained-job"
    _write_result(
        baseline,
        "task-a__base",
        task_id="task-a",
        reward=0.0,
        metadata={"difficulty_level": "easy", "reward_mode_initial": "dense"},
    )
    _write_result(
        baseline,
        "task-b__base",
        task_id="task-b",
        reward=1.0,
        metadata={"difficulty_level": "hard", "reward_mode_initial": "sparse"},
    )
    _write_result(baseline, "task-c__base", task_id="task-c", reward=1.0)
    _write_result(
        baseline,
        "task-d__base",
        task_id="task-d",
        error="sandbox not found",
    )
    _write_result(
        baseline,
        "task-f__base",
        task_id="task-f",
        reward=1.0,
        healthy=False,
    )
    _write_result(
        trained,
        "task-a__trained",
        task_id="task-a",
        reward=1.0,
        metadata={"difficulty_level": "easy", "reward_mode_initial": "dense"},
    )
    _write_result(
        trained,
        "task-b__trained",
        task_id="task-b",
        reward=0.5,
        metadata={"difficulty_level": "hard", "reward_mode_initial": "sparse"},
    )
    _write_result(
        trained,
        "task-c__trained",
        task_id="task-c",
        verifier_error="verifier crashed: dependency install failed",
    )
    _write_result(trained, "task-e__trained", task_id="task-e", reward=1.0)
    _write_result(
        trained,
        "task-f__trained",
        task_id="task-f",
        reward=1.0,
        partial_trajectory=True,
    )
    return baseline, trained


def test_build_lift_report_pairs_only_healthy_shared_tasks(tmp_path: Path) -> None:
    """Guards PR #901: paired lift excludes unscored infra/error rollouts."""

    baseline, trained = _write_lift_fixture(tmp_path)

    report = build_lift_report(baseline, trained, bootstrap_samples=50)

    assert report["pairing"]["paired_tasks"] == ["task-a", "task-b"]
    assert report["pairing"]["baseline_only_healthy_tasks"] == ["task-c"]
    assert report["pairing"]["trained_only_healthy_tasks"] == ["task-e"]
    assert report["coverage"]["baseline"]["total_rollouts"] == 5
    assert report["coverage"]["baseline"]["healthy_rollouts"] == 3
    assert report["coverage"]["baseline"]["error_rollouts"] == 1
    assert report["coverage"]["baseline"]["unhealthy_rollouts"] == 1
    assert report["coverage"]["baseline"]["excluded_reason_counts"]["unhealthy"] == 1
    assert report["coverage"]["trained"]["total_rollouts"] == 5
    assert report["coverage"]["trained"]["healthy_rollouts"] == 3
    assert report["coverage"]["trained"]["error_rollouts"] == 1
    assert report["coverage"]["trained"]["unhealthy_rollouts"] == 1
    assert (
        report["coverage"]["trained"]["excluded_reason_counts"]["partial_trajectory"]
        == 1
    )

    metrics = report["metrics"]
    assert metrics["paired_count"] == 2
    assert metrics["pass_rate_base"] == pytest.approx(0.5)
    assert metrics["pass_rate_trained"] == pytest.approx(0.5)
    assert metrics["pass_rate_delta"] == pytest.approx(0.0)
    assert metrics["mean_reward_base"] == pytest.approx(0.5)
    assert metrics["mean_reward_trained"] == pytest.approx(0.75)
    assert metrics["mean_reward_delta"] == pytest.approx(0.25)

    easy = report["by_metadata"]["difficulty_level"]["easy"]
    hard = report["by_metadata"]["difficulty_level"]["hard"]
    assert easy["paired_count"] == 1
    assert easy["mean_reward_delta"] == pytest.approx(1.0)
    assert hard["paired_count"] == 1
    assert hard["mean_reward_delta"] == pytest.approx(-0.5)


def test_compare_lift_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """Guards PR #901: `bench eval compare-lift` emits both report files."""

    baseline, trained = _write_lift_fixture(tmp_path)
    md_out = tmp_path / "lift.md"
    json_out = tmp_path / "lift.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "compare-lift",
            "--baseline",
            str(baseline),
            "--trained",
            str(trained),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
            "--bootstrap-samples",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Paired healthy tasks: 2" in result.output
    payload = json.loads(json_out.read_text())
    assert payload["metrics"]["mean_reward_delta"] == pytest.approx(0.25)
    markdown = md_out.read_text()
    assert "# Paired Eval Lift Report" in markdown
    assert "| Mean reward | 0.500 | 0.750 | +0.250 |" in markdown
    assert "| task-a | 0.000 | 1.000 | +1.000 | no | yes |" in markdown


def test_build_lift_report_counts_reward_plus_error_as_scored(
    tmp_path: Path,
) -> None:
    """Guards PR #901: numeric rewards remain scored when error fields exist."""

    baseline = tmp_path / "baseline-job"
    trained = tmp_path / "trained-job"
    _write_result(
        baseline,
        "task-a__base",
        task_id="task-a",
        reward=1.0,
        error="agent exited after reward export",
    )
    _write_result(
        trained,
        "task-a__trained",
        task_id="task-a",
        reward=0.0,
        verifier_error="late verifier stderr",
    )
    _write_result(
        baseline,
        "task-b__base",
        task_id="task-b",
        reward=0.25,
        export_error="artifact upload failed after scoring",
    )
    _write_result(trained, "task-b__trained", task_id="task-b", reward=0.75)
    _write_result(
        baseline,
        "task-c__base",
        task_id="task-c",
        error="sandbox missing before scoring",
    )
    _write_result(trained, "task-c__trained", task_id="task-c", reward=1.0)

    report = build_lift_report(baseline, trained, bootstrap_samples=0)

    assert report["pairing"]["paired_tasks"] == ["task-a", "task-b"]
    assert report["pairing"]["trained_only_healthy_tasks"] == ["task-c"]
    assert report["coverage"]["baseline"]["healthy_rollouts"] == 2
    assert report["coverage"]["baseline"]["error_rollouts"] == 1
    assert report["coverage"]["trained"]["healthy_rollouts"] == 3

    metrics = report["metrics"]
    assert metrics["paired_count"] == 2
    assert metrics["mean_reward_base"] == pytest.approx(0.625)
    assert metrics["mean_reward_trained"] == pytest.approx(0.375)
    assert metrics["mean_reward_delta"] == pytest.approx(-0.25)


def test_build_lift_report_fails_on_duplicate_healthy_rollouts_by_default(
    tmp_path: Path,
) -> None:
    """Guards PR #901: duplicate healthy rollout selection is fail-fast."""

    baseline = tmp_path / "baseline-job"
    trained = tmp_path / "trained-job"
    _write_result(baseline, "task-a__base_a", task_id="task-a", reward=0.0)
    _write_result(baseline, "task-a__base_b", task_id="task-a", reward=1.0)
    _write_result(trained, "task-a__trained", task_id="task-a", reward=1.0)

    with pytest.raises(ValueError, match="Duplicate healthy rollouts found"):
        build_lift_report(baseline, trained)


def test_compare_lift_cli_can_explicitly_allow_duplicate_first_by_path(
    tmp_path: Path,
) -> None:
    """Guards PR #901: duplicate first-by-path behavior is opt-in."""

    baseline = tmp_path / "baseline-job"
    trained = tmp_path / "trained-job"
    _write_result(baseline, "000-task-a__base", task_id="task-a", reward=0.0)
    _write_result(baseline, "999-task-a__base", task_id="task-a", reward=1.0)
    _write_result(trained, "task-a__trained", task_id="task-a", reward=1.0)
    md_out = tmp_path / "lift.md"
    json_out = tmp_path / "lift.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "compare-lift",
            "--baseline",
            str(baseline),
            "--trained",
            str(trained),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
            "--bootstrap-samples",
            "0",
            "--allow-duplicate-first-by-path",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_out.read_text())
    assert payload["coverage"]["baseline"]["duplicate_healthy_rollouts"] == 1
    assert payload["pairs"][0]["baseline_rollout_dir"].endswith("000-task-a__base")
    assert payload["metrics"]["mean_reward_delta"] == pytest.approx(1.0)

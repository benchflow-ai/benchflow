"""Parity evidence for 0.7 environment-adapter dogfood loops."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from benchflow.agent_router_parity import build_verify_report
from benchflow.cli.main import app
from benchflow.environment_adapter_parity import (
    build_environment_adapter_adoption_report,
    build_environment_adapter_loop_state,
    build_environment_adapter_parity_experiment,
    validate_environment_adapter_loop_state,
    write_environment_adapter_adoption_report,
    write_environment_adapter_loop_state,
    write_environment_adapter_parity_experiment,
)


def _original(*, score: float = 1.0) -> dict:
    return {
        "framework": "computer-use-smoke-original",
        "task_id": "desktop-file-roundtrip",
        "final_result": "computer-use-smoke: ready",
        "score": score,
        "steps": [{"action": "write_file"}],
        "screenshots_b64": ["original-png"],
        "dimensions": [1024, 768],
        "num_steps": 2,
        "duration_sec": 0.25,
        "error": None,
    }


def _benchflow(*, reward: float = 1.0) -> dict:
    return {
        "task_name": "desktop-file-roundtrip",
        "agent": "computer-use-smoke",
        "rewards": {"reward": reward},
        "elapsed_sec": 1.5,
        "n_tool_calls": 3,
        "trajectory_summary": {"steps": 6, "tool_call_steps": 3},
        "timing": {"total": 1.5},
        "error": None,
    }


def _artifact() -> dict:
    return {
        "framework": "benchflow-computer-use-smoke-agent",
        "final_result": "computer-use-smoke: ready",
        "steps": [{"action": "write_file"}, {"action": "screenshot"}],
        "screenshots_b64": ["benchflow-png"],
        "screenshot_method": "cua",
    }


def _browser_artifact() -> dict:
    artifact = _artifact()
    artifact["environment"] = {
        "adapter": "browser",
        "readiness": {
            "status": "ready",
            "content_sha256": "a" * 64,
        },
    }
    return artifact


def _summary(*, reward: float = 1.0, cleanup: dict | None = None) -> dict:
    return {
        "ok": True,
        "task_id": "desktop-file-roundtrip",
        "original": {
            "score": reward,
            "num_steps": 2,
            "screenshots_b64": 1,
            "dimensions": [1024, 768],
        },
        "benchflow": {
            "reward": reward,
            "agent": "computer-use-smoke",
            "trajectory_steps": 6,
            "tool_calls": 3,
            "artifact_steps": 2,
            "screenshots_b64": 1,
            "screenshot_method": "cua",
        },
        "cleanup": cleanup or {"docker_available": True, "cua_containers": 0},
    }


def _eval_run_report(
    *,
    status: str = "completed",
    ok: bool = True,
    errored: int = 0,
    verifier_errored: int = 0,
    trajectory_steps: int = 6,
) -> dict:
    return {
        "status": status,
        "ok": ok,
        "jobs_dir": "/tmp/benchflow-adapter-smoke",
        "summary_path": "/tmp/benchflow-adapter-smoke/summary.json",
        "result": {
            "job_name": "smoke-run",
            "total": 1,
            "passed": 1 if not errored and not verifier_errored else 0,
            "failed": 0,
            "errored": errored,
            "verifier_errored": verifier_errored,
            "score": 1.0,
            "score_excl_errors": 1.0,
            "elapsed_sec": 1.5,
        },
        "summary": {
            "job_name": "smoke-run",
            "agent": "computer-use-smoke",
            "environment": "cua",
            "total": 1,
            "passed": 1 if not errored and not verifier_errored else 0,
            "failed": 0,
            "errored": errored,
            "verifier_errored": verifier_errored,
            "score": "100.0%",
            "elapsed_sec": 1.5,
            "total_trajectory_steps": trajectory_steps,
            "trajectory_summary_coverage": 1.0 if trajectory_steps else 0.0,
            "total_tool_calls": 3,
            "timing_coverage": 1.0,
        },
    }


def _artifact_manifest() -> list[dict]:
    return [
        {
            "id": "benchflow-trace-steps",
            "source": "artifact",
            "path": "steps",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "benchflow-screenshots",
            "source": "artifact",
            "path": "screenshots_b64",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "benchflow-final-result",
            "source": "artifact",
            "path": "final_result",
            "kind": "field",
        },
        {
            "id": "benchflow-trajectory-steps",
            "source": "benchflow",
            "path": "trajectory_summary.steps",
            "kind": "numeric",
            "numeric_min": 1,
        },
    ]


def test_environment_adapter_parity_evidence_confirms_full_smoke_shape() -> None:
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=_summary(),
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
        artifact_manifest=_artifact_manifest(),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert evidence["status"] == "parity-confirmed"
    assert evidence["adapter_parity"]["artifact_manifest"]["ok"] is True
    assert report.verdict == "parity-confirmed"
    assert report.conversion.compared == 6
    assert report.reward is not None
    assert report.reward.max_abs_delta == 0.0
    manifest_json = json.dumps(evidence["adapter_parity"]["artifact_manifest"])
    assert "benchflow-png" not in manifest_json


def test_environment_adapter_adoption_report_is_scrubbed_and_reviewable() -> None:
    summary = _summary()
    summary["benchflow_eval"] = _eval_run_report()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
        artifact_manifest=_artifact_manifest(),
    )

    report = build_environment_adapter_adoption_report(
        parity_experiment=evidence,
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        parity_experiment_path="parity_experiment.json",
    )

    assert report["schema"] == "benchflow.environment-adapter-adoption-report.v1"
    assert report["status"] == "parity-confirmed"
    assert report["planes"] == {
        "sandbox_provider": "cua",
        "sandbox_provider_mode": "local",
        "environment_adapter": "desktop",
        "agent_adapter": "computer-use-smoke",
        "benchmark_adapter": "computer-use",
    }
    assert report["parity"] == {
        "parity_experiment": "parity_experiment.json",
        "criteria_compared": 7,
        "criteria_agreed": 7,
        "reward_delta": 0.0,
    }
    assert report["artifact_index"][0]["screenshots_b64_count"] == 1
    assert report["artifact_index"][2]["screenshots_b64_count"] == 1
    assert report["artifact_index"][3]["present"] is True
    assert report["artifact_requirements"]["ok"] is True
    report_json = json.dumps(report)
    assert "original-png" not in report_json
    assert "benchflow-png" not in report_json


def test_environment_adapter_adoption_report_can_be_written(tmp_path: Path) -> None:
    summary = _summary()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
    )
    report_path = tmp_path / "adoption_report.json"

    report = write_environment_adapter_adoption_report(
        report_path,
        parity_experiment=evidence,
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        parity_experiment_path=tmp_path / "parity_experiment.json",
    )

    saved = json.loads(report_path.read_text())
    assert saved == report
    assert saved["artifact_index"][1]["reward"] == 1.0
    assert saved["cleanup"]["keys"] == ["cua_containers", "docker_available"]


def test_environment_adapter_adoption_report_summarizes_browser_readiness() -> None:
    summary = _summary()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="browser-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_browser_artifact(),
        summary=summary,
        sandbox="docker",
        environment_adapter="browser",
        benchmark_adapter="browser-use",
    )

    report = build_environment_adapter_adoption_report(
        parity_experiment=evidence,
        original=_original(),
        benchflow=_benchflow(),
        artifact=_browser_artifact(),
        summary=summary,
    )

    trace_artifact = report["artifact_index"][2]
    assert trace_artifact["environment_adapter"] == "browser"
    assert trace_artifact["environment_readiness"] == "ready"
    assert trace_artifact["environment_content_sha256_present"] is True


def test_environment_adapter_loop_state_is_resumable_and_review_gated(
    tmp_path: Path,
) -> None:
    summary = _summary()
    summary["benchflow_eval"] = _eval_run_report()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
        artifact_manifest=_artifact_manifest(),
    )
    adoption = build_environment_adapter_adoption_report(
        parity_experiment=evidence,
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        parity_experiment_path=tmp_path / "parity_experiment.json",
    )

    state = build_environment_adapter_loop_state(
        parity_experiment=evidence,
        adoption_report=adoption,
        commands=[
            "GEMINI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz uv run python parity_test.py",
            "uv run bench agent verify computer-use-smoke --require-adoption-report",
        ],
        artifacts={
            "parity_experiment": tmp_path / "parity_experiment.json",
            "adoption_report": tmp_path / "adoption_report.json",
        },
        source={
            "type": "fixture",
            "revision": "local",
            "selected_tasks": ["desktop-file-roundtrip"],
        },
        queue=[{"id": "next-cua-gym-slice", "status": "queued"}],
        updated_at="2026-06-12",
    )

    assert state["schema"] == "benchflow.adapter-adoption-loop-state.v1"
    assert state["status"] == "review-ready"
    assert state["commands"][0].startswith("GEMINI_API_KEY=<redacted>")
    assert state["checks"]["cleanup"]["cua_containers"] == 0
    assert {role["name"] for role in state["roles"]} == {
        "scout",
        "builder",
        "original-runner",
        "benchflow-runner",
        "verifier",
        "auditor",
        "reviewer",
        "queue",
    }
    assert validate_environment_adapter_loop_state(state) == []
    assert validate_environment_adapter_loop_state(state, require_review=True) == [
        "loop state reviewer role has not passed"
    ]

    reviewed_state = build_environment_adapter_loop_state(
        parity_experiment=evidence,
        adoption_report=adoption,
        commands=state["commands"],
        artifacts=state["artifacts"],
        source=state["source"],
        roles=[
            *[role for role in state["roles"] if role["name"] != "reviewer"],
            {"name": "reviewer", "status": "passed", "artifact": "review.md"},
        ],
    )
    assert reviewed_state["status"] == "scale-ready"
    assert (
        validate_environment_adapter_loop_state(reviewed_state, require_review=True)
        == []
    )


def test_environment_adapter_loop_state_can_be_written(tmp_path: Path) -> None:
    summary = _summary()
    summary["benchflow_eval"] = _eval_run_report()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
        artifact_manifest=_artifact_manifest(),
    )
    adoption = build_environment_adapter_adoption_report(
        parity_experiment=evidence,
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        parity_experiment_path=tmp_path / "parity_experiment.json",
    )

    path = tmp_path / "loop_state.json"
    state = write_environment_adapter_loop_state(
        path,
        parity_experiment=evidence,
        adoption_report=adoption,
        commands=["uv run python parity_test.py"],
        artifacts={
            "parity_experiment": tmp_path / "parity_experiment.json",
            "adoption_report": tmp_path / "adoption_report.json",
        },
    )

    assert json.loads(path.read_text()) == state
    assert validate_environment_adapter_loop_state(state) == []


def test_environment_adapter_parity_evidence_confirms_eval_run_summary() -> None:
    summary = _summary()
    summary["benchflow_eval"] = _eval_run_report()
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
        artifact_manifest=_artifact_manifest(),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert report.verdict == "parity-confirmed"
    assert report.conversion.compared == 7
    criterion_ids = [
        criterion["criterion_id"]
        for criterion in evidence["conversion_parity"]["tasks"][0]["criteria_results"]
    ]
    assert "eval-run-summary" in criterion_ids


def test_environment_adapter_parity_evidence_fails_on_bad_eval_run_summary() -> None:
    summary = _summary()
    summary["benchflow_eval"] = _eval_run_report(
        status="completed-with-errors",
        ok=False,
        errored=1,
        trajectory_steps=0,
    )
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=summary,
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert report.verdict == "parity-divergent"
    assert [c.criterion_id for c in report.conversion.disagreements] == [
        "eval-run-summary",
    ]


def test_environment_adapter_parity_evidence_preserves_zero_rewards() -> None:
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(score=0.0),
        benchflow=_benchflow(reward=0.0),
        artifact=_artifact(),
        summary=_summary(reward=0.0),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert report.verdict == "parity-confirmed"
    assert report.reward is not None
    assert report.reward.samples[0].legacy_reward == 0.0
    assert report.reward.samples[0].converted_reward == 0.0


def test_environment_adapter_parity_evidence_fails_on_missing_manifest_artifact() -> (
    None
):
    artifact = _artifact()
    artifact["screenshots_b64"] = []
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=artifact,
        summary=_summary(),
        artifact_manifest=_artifact_manifest(),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert evidence["status"] == "parity-recorded"
    assert evidence["adapter_parity"]["artifact_manifest"]["ok"] is False
    assert report.verdict == "parity-divergent"
    assert [c.criterion_id for c in report.conversion.disagreements] == [
        "artifact-manifest",
    ]


def test_environment_adapter_parity_evidence_fails_on_cleanup_leak() -> None:
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=_summary(cleanup={"docker_available": True, "cua_containers": 1}),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert report.verdict == "parity-divergent"
    assert [c.criterion_id for c in report.conversion.disagreements] == ["cleanup"]


def test_environment_adapter_parity_cleanup_is_scoped_to_this_run() -> None:
    # The cleanup counts are per-run leak counts (resources THIS run created and
    # left behind). They are zero whenever the run reaped its own resources, even
    # if unrelated pre-existing benchflow containers are still around globally, so
    # an unrelated/concurrent container can no longer force a false cleanup fail.
    evidence = build_environment_adapter_parity_experiment(
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=_summary(
            cleanup={
                "docker_available": True,
                # 0 = this run leaked nothing, regardless of unrelated benchflow
                # containers that exist globally.
                "benchflow_containers": 0,
                "benchflow_networks": 0,
            }
        ),
    )

    report = build_verify_report("computer-use-smoke", evidence)

    assert report.verdict == "parity-confirmed"
    cleanup = next(
        item
        for task in evidence["conversion_parity"]["tasks"]
        for item in task["criteria_results"]
        if item["criterion_id"] == "cleanup"
    )
    assert cleanup["agreement"] is True


def test_cli_agent_verify_accepts_environment_adapter_parity_file(
    tmp_path: Path,
) -> None:
    benchmarks_root = tmp_path / "benchmarks"
    evidence_path = benchmarks_root / "computer-use-smoke" / "parity_experiment.json"
    write_environment_adapter_parity_experiment(
        evidence_path,
        benchmark="computer-use-smoke",
        original=_original(),
        benchflow=_benchflow(),
        artifact=_artifact(),
        summary=_summary(),
        sandbox="cua",
        environment_adapter="desktop",
        benchmark_adapter="computer-use",
    )

    saved = json.loads(evidence_path.read_text())
    assert saved["experiment"] == "environment-adapter-side-by-side-parity"

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "verify",
            "computer-use-smoke",
            "--benchmarks-dir",
            str(benchmarks_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Verdict: parity-confirmed" in result.output

"""Execution-path tests for the acceptance-live runner.

The rollout layer is replaced by the scripted fakes in
acceptance_live_harness, so no sandbox, model, or network is involved.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow import __version__
from benchflow.task.acceptance_live import run_live_acceptance_checks
from tests.acceptance_live_harness import (
    AL,
    REPORT_REL,
    LiveHarness,
    case_evidence,
    full_live_evidence,
    green_case,
    install_live_harness,
    oracle_case,
    run_live,
    write_live_task,
)


@pytest.fixture
def live_task(tmp_path: Path) -> Path:
    return write_live_task(tmp_path / "live-task")


@pytest.fixture
def harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, live_task: Path
) -> LiveHarness:
    return install_live_harness(monkeypatch, tmp_path)


class TestRunLiveAcceptanceChecks:
    def test_happy_path_writes_full_report(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        evidence = full_live_evidence()
        issues, report = run_live(live_task, evidence)
        assert issues == []
        assert report is not None

        assert report["kind"] == "acceptance-live-report"
        assert report["schema_version"] == "1.0"
        assert report["benchflow_version"] == __version__
        assert report["sandbox"] == "docker"
        assert report["task"]["path"] == "live-task"
        assert (
            report["task"]["task_md_sha256"]
            == sha256((live_task / "task.md").read_bytes()).hexdigest()
        )
        assert (
            report["task"]["oracle_sha256"]
            == sha256((live_task / "oracle" / "solve.sh").read_bytes()).hexdigest()
        )
        assert report["task"]["verifier_sha256"] is None

        inner = sha256(b"inner\n").hexdigest()
        staged = sha256(b"staged\n").hexdigest()
        expected_tree = sha256(
            f"nested/inner.txt\0{inner}\nworkspace.txt\0{staged}".encode()
        ).hexdigest()
        assert report["workspace"] == {
            "source": "current-worktree",
            "target": "/workspace",
            "staged_tree_sha256": expected_tree,
        }
        assert len(report["spec_sha256"]) == 64

        assert [case["name"] for case in report["cases"]] == [
            "greencase",
            "oracleproof",
            "calnoop",
            "calknownbad",
            "calpartial",
            "calreference",
        ]
        assert report["cases"][0] == {
            "name": "greencase",
            "type": "verifier",
            "source": "declared",
            "command": "echo ok",
            "reruns": 2,
            "expect": {
                "reward_min": 0.5,
                "reward_max": None,
                "reward_range": None,
                "reward_equals": None,
                "flake_rate_max": None,
            },
        }
        assert report["cases"][2]["source"] == "calibration-report"
        assert report["cases"][2]["expect"]["reward_max"] == 0.1

        assert report["summary"] == {
            "total_runs": 7,
            "passed_runs": 7,
            "failed_runs": 0,
            "flake_rate": 0.0,
            "min_reward": 0.0,
            "max_reward": 1.0,
        }
        assert report["case_summaries"][0] == {
            "case": "greencase",
            "type": "verifier",
            "source": "declared",
            "total_runs": 2,
            "passed_runs": 2,
            "failed_runs": 0,
            "flake_rate": 0.0,
            "flake_rate_max": None,
            "min_reward": 1.0,
            "max_reward": 1.0,
            "status": "passed",
        }

        suitability = report["leaderboard_suitability"]
        assert suitability["status"] == "suitable"
        assert suitability["required"] is True
        assert suitability["max_flake_rate"] == 0.0
        assert suitability["issues"] == []
        assert suitability["checks"] == {
            "has_live_runs": True,
            "all_runs_passed": True,
            "flake_rate_within_limit": True,
            "has_oracle_proof": True,
            "has_reference_proof": True,
            "has_generated_calibration_coverage": True,
        }
        assert suitability["required_generated_calibration_types"] == [
            "known-bad",
            "no-op",
            "partial",
            "reference",
        ]
        assert suitability["observed_generated_calibration_types"] == [
            "known-bad",
            "no-op",
            "partial",
            "reference",
        ]

        assert len(report["runs"]) == 7
        first = report["runs"][0]
        assert first["case"] == "greencase"
        assert first["run_index"] == 1
        assert first["reward"] == 1.0
        assert first["status"] == "passed"
        assert first["error"] is None
        assert first["expectation_issues"] == []
        for record in report["runs"]:
            body = {k: v for k, v in record.items() if k != "sha256"}
            digest = sha256(
                json.dumps(
                    body, sort_keys=True, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            assert record["sha256"] == digest

        report_path = live_task / REPORT_REL
        raw = report_path.read_text()
        assert raw == json.dumps(report, indent=2, sort_keys=True) + "\n"
        digest = sha256(report_path.read_bytes()).hexdigest()
        sidecar = report_path.with_suffix(".json.sha256")
        assert sidecar.read_text() == f"{digest}  {REPORT_REL}\n"

    def test_happy_path_wires_rollout_layer(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        issues, _report = run_live(live_task, full_live_evidence())
        assert issues == []

        assert harness.starts == [live_task] * 6
        assert sorted(e["command"] for e in harness.execs) == sorted(
            ["cd /workspace && echo ok"] * 2 + ["cd /workspace && true"] * 3
        )
        assert {e["user"] for e in harness.execs} == {"root"}
        assert {e["timeout_sec"] for e in harness.execs} == {30.0}
        assert len(harness.verifies) == 6
        assert {v["workspace"] for v in harness.verifies} == {"/resolved-agent-cwd"}
        assert {v["sandbox_user"] for v in harness.verifies} == {None}
        assert len(harness.uploads) == 7
        assert {u["target"] for u in harness.uploads} == {"/workspace"}
        for upload in harness.uploads:
            assert upload["files"] == ["nested/inner.txt", "workspace.txt"]
            assert not Path(upload["source"]).exists()
        assert len(harness.stops) == 6
        assert all(stop["delete"] is True for stop in harness.stops)

    def test_parse_failure_short_circuits_execution(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        issues = run_live_acceptance_checks(
            live_task, sandbox_type="docker", evidence={AL: ["cases"]}
        )
        assert issues == [
            "acceptance-live validation requires "
            "benchflow.evidence.acceptance_live mapping"
        ]
        assert harness.starts == []
        assert harness.oracle_configs == []
        assert not (live_task / REPORT_REL).exists()

    def test_setup_command_failure(self, live_task: Path, harness: LiveHarness) -> None:
        harness.exec_rcs["greencase"] = 7
        issues, report = run_live(live_task, case_evidence(green_case()))
        assert issues == [
            "acceptance-live case 'greencase' run 1 failed: "
            "setup command exited with rc=7"
        ]
        assert report is not None
        record = report["runs"][0]
        assert record["status"] == "failed"
        assert record["reward"] is None
        assert record["error"] == "setup command exited with rc=7"
        assert record["diagnostic_code"] == "setup_command_failed"
        assert record["verifier_error_category"] is None
        assert harness.execs[0]["command"] == "cd /app && echo ok"
        assert harness.verifies == []
        assert harness.stops[0]["delete"] is True

    def test_verifier_error_is_classified(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.verifier_errors["greencase"] = "verifier exploded badly"
        issues, report = run_live(live_task, case_evidence(green_case()))
        assert issues == [
            "acceptance-live case 'greencase' run 1 failed: verifier exploded badly"
        ]
        assert report is not None
        record = report["runs"][0]
        assert record["error"] == "verifier exploded badly"
        assert record["verifier_error_category"] == "verifier_other"
        assert record["diagnostic_code"] == "verifier_other"
        assert record["artifact_hint"] is None

    def test_verifier_dep_install_error_gets_artifact_hint(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        error = "verifier crashed: dependency install failed during setup"
        harness.verifier_errors["greencase"] = error
        issues, report = run_live(live_task, case_evidence(green_case()))
        assert issues == [f"acceptance-live case 'greencase' run 1 failed: {error}"]
        assert report is not None
        record = report["runs"][0]
        assert record["verifier_error_category"] == "verifier_dep_install"
        assert record["diagnostic_code"] == "verifier_dep_install"
        assert record["artifact_hint"] == "verifier/test-stdout.txt"

    def test_non_scalar_reward_fails_run(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.rewards["greencase"] = "bogus"
        issues, report = run_live(live_task, case_evidence(green_case()))
        assert issues == [
            "acceptance-live case 'greencase' run 1 did not produce scalar reward"
        ]
        assert report is not None
        record = report["runs"][0]
        assert record["status"] == "failed"
        assert record["reward"] is None
        assert record["error"] is None

    def test_run_exception_is_captured_and_sandbox_stopped(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.verify_exceptions["greencase"] = RuntimeError("sandbox burst")
        issues, _report = run_live(live_task, case_evidence(green_case()))
        assert issues == [
            "acceptance-live case 'greencase' run 1 failed: sandbox burst"
        ]
        assert len(harness.stops) == 1
        assert harness.stops[0]["delete"] is True

    @pytest.mark.parametrize(
        ("expect", "reward", "expected_issue"),
        [
            pytest.param(
                {"reward_min": 0.5},
                0.25,
                "acceptance-live case 'greencase' run 1 reward 0.25 "
                "is below reward_min 0.5",
                id="below-min",
            ),
            pytest.param(
                {"reward_max": 0.5},
                0.75,
                "acceptance-live case 'greencase' run 1 reward 0.75 "
                "is above reward_max 0.5",
                id="above-max",
            ),
            pytest.param(
                {"reward_range": [0.2, 0.4]},
                0.5,
                "acceptance-live case 'greencase' run 1 reward 0.5 "
                "is outside reward_range [0.2, 0.4]",
                id="outside-range",
            ),
            pytest.param(
                {"reward_equals": 1.0},
                0.5,
                "acceptance-live case 'greencase' run 1 reward 0.5 "
                "does not equal reward_equals 1",
                id="not-equal",
            ),
            pytest.param({"reward_min": 0.5}, 0.5, None, id="boundary-passes"),
        ],
    )
    def test_reward_expectations(
        self,
        live_task: Path,
        harness: LiveHarness,
        expect: dict[str, Any],
        reward: float,
        expected_issue: str | None,
    ) -> None:
        harness.rewards["greencase"] = reward
        issues, report = run_live(live_task, case_evidence(green_case(expect=expect)))
        assert report is not None
        if expected_issue is None:
            assert issues == []
            assert report["runs"][0]["status"] == "passed"
        else:
            assert issues == [expected_issue]
            assert report["runs"][0]["status"] == "failed"
            assert report["runs"][0]["expectation_issues"] == [expected_issue]

    def test_flake_threshold_tolerates_failures_within_budget(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.rewards["greencase"] = [1.0, 0.0, 1.0, 1.0]
        case = green_case(reruns=4, expect={"reward_min": 0.5, "flake_rate_max": 0.5})
        issues, report = run_live(live_task, case_evidence(case))
        assert issues == []
        assert report is not None
        assert report["case_summaries"][0] == {
            "case": "greencase",
            "type": "verifier",
            "source": "declared",
            "total_runs": 4,
            "passed_runs": 3,
            "failed_runs": 1,
            "flake_rate": 0.25,
            "flake_rate_max": 0.5,
            "min_reward": 0.0,
            "max_reward": 1.0,
            "status": "passed",
        }
        assert report["runs"][1]["status"] == "failed"
        assert report["runs"][1]["expectation_issues"] == [
            "acceptance-live case 'greencase' run 2 reward 0 is below reward_min 0.5"
        ]

    def test_flake_threshold_absorbs_non_scalar_reward_run(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.rewards["greencase"] = ["bogus", 1.0]
        case = green_case(reruns=2, expect={"reward_min": 0.5, "flake_rate_max": 0.5})
        issues, report = run_live(live_task, case_evidence(case))
        assert issues == []
        assert report is not None
        assert report["runs"][0]["status"] == "failed"
        assert report["runs"][0]["reward"] is None
        assert report["runs"][0]["error"] is None
        assert report["case_summaries"][0]["failed_runs"] == 1
        assert report["case_summaries"][0]["status"] == "passed"

    def test_flake_threshold_exceeded(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.rewards["greencase"] = [1.0, 0.0]
        case = green_case(reruns=2, expect={"reward_min": 0.5, "flake_rate_max": 0.25})
        issues, report = run_live(live_task, case_evidence(case))
        assert issues == [
            "acceptance-live case 'greencase' flake_rate 0.5 "
            "exceeds flake_rate_max 0.25"
        ]
        assert report is not None
        assert report["case_summaries"][0]["status"] == "failed"

    def test_flake_threshold_dep_install_hint(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.verifier_errors["greencase"] = [
            None,
            "verifier crashed: dependency install failed",
        ]
        case = green_case(reruns=2, expect={"reward_min": 0.5, "flake_rate_max": 0.0})
        issues, _report = run_live(live_task, case_evidence(case))
        assert issues == [
            "acceptance-live case 'greencase' flake_rate 0.5 "
            "exceeds flake_rate_max 0; "
            "first failed run indicates verifier dependency install failed "
            "(see verifier/test-stdout.txt in the run artifacts)"
        ]

    def test_leaderboard_required_reports_missing_proofs(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        evidence = case_evidence(
            green_case(), leaderboard={"required": True, "max_flake_rate": 0.0}
        )
        issues, report = run_live(live_task, evidence)
        assert issues == [
            "acceptance-live leaderboard suitability: "
            "requires a passed oracle live case",
            "acceptance-live leaderboard suitability: "
            "requires a passed reference live case",
            "acceptance-live leaderboard suitability: "
            "missing generated calibration live case types: "
            "known-bad, no-op, partial, reference",
        ]
        assert report is not None
        suitability = report["leaderboard_suitability"]
        assert suitability["status"] == "insufficient"
        assert suitability["checks"] == {
            "has_live_runs": True,
            "all_runs_passed": True,
            "flake_rate_within_limit": True,
            "has_oracle_proof": False,
            "has_reference_proof": False,
            "has_generated_calibration_coverage": False,
        }

    def test_leaderboard_required_reports_failed_runs_and_flake(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.rewards["greencase"] = 0.0
        evidence = case_evidence(
            green_case(expect={"reward_min": 0.9}),
            leaderboard={"required": True, "max_flake_rate": 0.0},
        )
        issues, _report = run_live(live_task, evidence)
        assert issues == [
            "acceptance-live case 'greencase' run 1 reward 0 is below reward_min 0.9",
            "acceptance-live leaderboard suitability: requires all live runs to pass",
            "acceptance-live leaderboard suitability: "
            "flake_rate 1 exceeds max_flake_rate 0",
            "acceptance-live leaderboard suitability: "
            "requires a passed oracle live case",
            "acceptance-live leaderboard suitability: "
            "requires a passed reference live case",
            "acceptance-live leaderboard suitability: "
            "missing generated calibration live case types: "
            "known-bad, no-op, partial, reference",
        ]

    def test_leaderboard_not_required_keeps_suitability_out_of_issues(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        evidence = case_evidence(
            green_case(), leaderboard={"required": False, "max_flake_rate": 0.0}
        )
        issues, report = run_live(live_task, evidence)
        assert issues == []
        assert report is not None
        assert report["leaderboard_suitability"]["status"] == "insufficient"
        assert report["leaderboard_suitability"]["required"] is False

    def test_write_report_false_skips_report_but_keeps_gate(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        evidence = case_evidence(
            green_case(), leaderboard={"required": True, "max_flake_rate": 0.0}
        )
        issues = run_live_acceptance_checks(
            live_task, sandbox_type="docker", evidence=evidence, write_report=False
        )
        assert issues == [
            "acceptance-live leaderboard suitability: "
            "requires a passed oracle live case",
            "acceptance-live leaderboard suitability: "
            "requires a passed reference live case",
            "acceptance-live leaderboard suitability: "
            "missing generated calibration live case types: "
            "known-bad, no-op, partial, reference",
        ]
        assert not (live_task / REPORT_REL).exists()
        assert not (live_task / (REPORT_REL + ".sha256")).exists()

    def test_report_output_override_redirects_report(
        self, live_task: Path, harness: LiveHarness, tmp_path: Path
    ) -> None:
        override = tmp_path / "out" / "live.json"
        issues = run_live_acceptance_checks(
            live_task,
            sandbox_type="docker",
            evidence=case_evidence(green_case()),
            report_output=override,
        )
        assert issues == []
        assert not (live_task / REPORT_REL).exists()
        report = json.loads(override.read_text())
        assert report["kind"] == "acceptance-live-report"
        digest = sha256(override.read_bytes()).hexdigest()
        sidecar = override.with_suffix(".json.sha256")
        assert sidecar.read_text() == f"{digest}  {override.as_posix()}\n"

    def test_report_is_stable_across_reruns(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        _issues, first = run_live(live_task, full_live_evidence())
        _issues, second = run_live(live_task, full_live_evidence())
        assert first is not None and second is not None
        assert first["spec_sha256"] == second["spec_sha256"]
        assert first["cases"] == second["cases"]
        assert first["summary"] == second["summary"]
        assert first["case_summaries"] == second["case_summaries"]


class TestOracleCase:
    def test_oracle_success_wiring(self, live_task: Path, harness: LiveHarness) -> None:
        evidence = case_evidence(oracle_case(), workspace={"target": "/workspace"})
        issues, report = run_live(live_task, evidence)
        assert issues == []
        assert report is not None
        record = report["runs"][0]
        assert record["case"] == "oracleproof"
        assert record["type"] == "oracle"
        assert record["reward"] == 1.0
        assert record["status"] == "passed"

        assert len(harness.oracle_configs) == 1
        config = harness.oracle_configs[0]
        assert config.agent == "oracle"
        assert config.environment == "docker"
        assert config.task_path == live_task
        assert config.model is None
        assert len(config.scenes) == 1
        assert config.rollout_name.startswith(
            "acceptance-live-live-task-oracleproof-1-"
        )
        assert harness.uploads[0]["target"] == "/workspace"
        assert harness.uploads[0]["files"] == ["nested/inner.txt", "workspace.txt"]

    @pytest.mark.parametrize(
        ("result", "expected_issue"),
        [
            pytest.param(
                SimpleNamespace(
                    trajectory=[{"type": "agent", "return_code": 0}],
                    error=None,
                    verifier_error=None,
                    rewards={"reward": 1.0},
                ),
                "acceptance-live case 'oracleproof' run 1 failed: "
                "oracle rerun did not record oracle trajectory event",
                id="missing-oracle-event",
            ),
            pytest.param(
                SimpleNamespace(
                    trajectory=[{"type": "oracle", "return_code": 2}],
                    error=None,
                    verifier_error=None,
                    rewards={"reward": 1.0},
                ),
                "acceptance-live case 'oracleproof' run 1 failed: "
                "oracle exited with rc=2",
                id="nonzero-return-code",
            ),
            pytest.param(
                SimpleNamespace(
                    trajectory=[{"type": "oracle", "return_code": 0}],
                    error="agent infra down",
                    verifier_error=None,
                    rewards={"reward": 1.0},
                ),
                "acceptance-live case 'oracleproof' run 1 failed: agent infra down",
                id="rollout-error",
            ),
            pytest.param(
                SimpleNamespace(
                    trajectory=[{"type": "oracle", "return_code": 0}],
                    error=None,
                    verifier_error=None,
                    rewards={"reward": 2.0},
                ),
                "acceptance-live case 'oracleproof' run 1 "
                "did not produce scalar reward",
                id="out-of-range-reward",
            ),
        ],
    )
    def test_oracle_failures(
        self,
        live_task: Path,
        harness: LiveHarness,
        result: SimpleNamespace,
        expected_issue: str,
    ) -> None:
        harness.oracle_results["oracleproof"] = result
        issues, _report = run_live(live_task, case_evidence(oracle_case()))
        assert issues == [expected_issue]

    def test_oracle_verifier_error_is_classified(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        error = "verifier crashed: dependency install failed"
        harness.oracle_results["oracleproof"] = SimpleNamespace(
            trajectory=[{"type": "oracle", "return_code": 0}],
            error=None,
            verifier_error=error,
            rewards=None,
        )
        issues, report = run_live(live_task, case_evidence(oracle_case()))
        assert issues == [f"acceptance-live case 'oracleproof' run 1 failed: {error}"]
        assert report is not None
        record = report["runs"][0]
        assert record["verifier_error_category"] == "verifier_dep_install"
        assert record["artifact_hint"] == "verifier/test-stdout.txt"

    def test_oracle_rollout_create_failure(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.oracle_create_error = RuntimeError("daytona quota")
        issues, _report = run_live(live_task, case_evidence(oracle_case()))
        assert issues == [
            "acceptance-live case 'oracleproof' run 1 failed: daytona quota"
        ]

    def test_oracle_reward_expectation_violation(
        self, live_task: Path, harness: LiveHarness
    ) -> None:
        harness.oracle_results["oracleproof"] = SimpleNamespace(
            trajectory=[{"type": "oracle", "return_code": 0}],
            error=None,
            verifier_error=None,
            rewards={"reward": 0.5},
        )
        issues, _report = run_live(live_task, case_evidence(oracle_case()))
        assert issues == [
            "acceptance-live case 'oracleproof' run 1 reward 0.5 "
            "does not equal reward_equals 1"
        ]

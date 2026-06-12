"""Tests for the static acceptance evidence gate and its check_task wiring."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchflow._utils import task_authoring
from benchflow._utils.task_authoring import (
    _check_acceptance_evidence,
    _check_live_acceptance_execution,
    check_task,
)
from tests.acceptance_live_harness import (
    LIVE_CALIBRATION_REPORT,
    REPORT_REL,
    green_case,
    install_live_harness,
    oracle_case,
)

ORACLE_REL = "evidence/oracle-run.json"
VERIFIER_REL = "evidence/verifier-stability.json"
REVIEW_REL = "evidence/review.json"
EXAMPLE_REL = "evidence/reference-example.json"
CALIBRATION_REL = "evidence/calibration-report.json"


@dataclass
class EvidencePackage:
    files: dict[str, Any]
    evidence: dict[str, Any]
    pin_exclude: set[str] = field(default_factory=set)
    tamper: dict[str, str] = field(default_factory=dict)


def golden_package() -> EvidencePackage:
    files: dict[str, Any] = {
        ORACLE_REL: {"reward": 1.0, "status": "passed"},
        VERIFIER_REL: {
            "kind": "verifier-stability-report",
            "reruns": 3,
            "min_reward": 1.0,
            "flake_rate": 0.0,
            "runs": [{"status": "passed", "reward": 1.0} for _ in range(3)],
        },
        REVIEW_REL: {
            "kind": "acceptance-review",
            "anti_cheat": "passed",
            "instruction_alignment": "passed",
        },
        EXAMPLE_REL: {"expected_reward": 1.0},
        CALIBRATION_REL: {
            "kind": "calibration-report",
            "cases": [
                {"name": "noop", "type": "no-op", "reward": 0.0},
                {"name": "bad", "type": "known-bad", "reward": 0.2},
                {"name": "half", "type": "partial", "reward": 0.5},
                {"name": "ref", "type": "reference", "reward": 1.0},
            ],
        },
    }
    evidence: dict[str, Any] = {
        "oracle_runs": {"required_reward": 1.0, "artifact": ORACLE_REL},
        "verifier": {"reruns": 3, "flake_rate": 0.0, "report": VERIFIER_REL},
        "review": {
            "anti_cheat": "passed",
            "instruction_alignment": "passed",
            "artifact": REVIEW_REL,
        },
        "calibration": {
            "no_op_reward_max": 0.1,
            "known_bad_reward_max": 0.5,
            "partial_solution_range": [0.2, 0.8],
            "human_or_reference_examples": [
                {"expected_reward": 1.0, "artifact": EXAMPLE_REL}
            ],
            "report": CALIBRATION_REL,
        },
    }
    return EvidencePackage(files=files, evidence=evidence)


def write_package(task_dir: Path, package: EvidencePackage) -> Path:
    (task_dir / "evidence").mkdir(parents=True, exist_ok=True)
    for rel, content in package.files.items():
        (task_dir / rel).write_text(json.dumps(content))
    if "artifacts" not in package.evidence:
        package.evidence["artifacts"] = [
            {"path": rel, "sha256": sha256((task_dir / rel).read_bytes()).hexdigest()}
            for rel in package.files
            if rel not in package.pin_exclude
        ]
    frontmatter = {
        "schema_version": "1.3",
        "task": {"name": "benchflow/evidence-demo", "description": "demo"},
        "agent": {"timeout_sec": 60},
        "verifier": {"timeout_sec": 30},
        "benchflow": {"evidence": package.evidence},
    }
    (task_dir / "task.md").write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False)
        + "---\n\n## prompt\n\nDo the work.\n"
    )
    for rel, text in package.tamper.items():
        (task_dir / rel).write_text(text)
    return task_dir


def check(tmp_path: Path, package: EvidencePackage) -> list[str]:
    return _check_acceptance_evidence(write_package(tmp_path / "task", package))


def _use_last_job(package: EvidencePackage) -> None:
    package.evidence["oracle_runs"].pop("artifact")
    package.evidence["oracle_runs"]["last_job"] = "jobs/2026-06-05-oracle"


def _tamper_reindent(package: EvidencePackage, rel: str) -> None:
    package.tamper[rel] = json.dumps(package.files[rel], indent=4)


EVIDENCE_REJECTS = [
    pytest.param(
        lambda p: p.evidence.pop("oracle_runs"),
        "acceptance validation requires benchflow.evidence.oracle_runs",
        id="oracle-runs-missing",
    ),
    pytest.param(
        lambda p: p.evidence["oracle_runs"].__setitem__("required_reward", 0.9),
        "acceptance oracle_runs.required_reward must be numeric and >= 0.99",
        id="oracle-required-reward-too-low",
    ),
    pytest.param(
        lambda p: p.evidence["oracle_runs"].__setitem__("required_reward", "high"),
        "acceptance oracle_runs.required_reward must be numeric and >= 0.99",
        id="oracle-required-reward-not-numeric",
    ),
    pytest.param(
        lambda p: p.evidence["oracle_runs"].pop("artifact"),
        "acceptance oracle_runs must include last_job or artifact evidence",
        id="oracle-no-job-or-artifact",
    ),
    pytest.param(
        lambda p: p.evidence["oracle_runs"].__setitem__(
            "artifact", "evidence/nope.json"
        ),
        "acceptance oracle_runs.artifact references missing file: evidence/nope.json",
        id="oracle-artifact-file-missing",
    ),
    pytest.param(
        lambda p: p.files[ORACLE_REL].__setitem__("reward", 0.5),
        "acceptance oracle_runs.artifact.reward must be >= oracle_runs.required_reward",
        id="oracle-artifact-reward-below-required",
    ),
    pytest.param(
        lambda p: p.files[ORACLE_REL].__setitem__("status", "failed"),
        "acceptance oracle_runs.artifact.status must be passed when present",
        id="oracle-artifact-status-not-passed",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(ORACLE_REL, [1]),
        "acceptance oracle_runs.artifact must be a JSON object",
        id="oracle-artifact-not-object",
    ),
    pytest.param(
        lambda p: p.evidence.pop("verifier"),
        "acceptance validation requires benchflow.evidence.verifier",
        id="verifier-missing",
    ),
    pytest.param(
        lambda p: p.evidence["verifier"].__setitem__("reruns", 2),
        "acceptance verifier.reruns must be an integer >= 3",
        id="verifier-reruns-too-few",
    ),
    pytest.param(
        lambda p: p.evidence["verifier"].__setitem__("reruns", True),
        "acceptance verifier.reruns must be an integer >= 3",
        id="verifier-reruns-bool",
    ),
    pytest.param(
        lambda p: p.evidence["verifier"].__setitem__("flake_rate", 0.06),
        "acceptance verifier.flake_rate must be numeric and <= 0.05",
        id="verifier-flake-rate-too-high",
    ),
    pytest.param(
        lambda p: p.evidence["verifier"].__setitem__("report", "evidence/nope.json"),
        "acceptance verifier.report references missing file: evidence/nope.json",
        id="verifier-report-file-missing",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(VERIFIER_REL, [1]),
        "acceptance verifier.report must be a JSON object",
        id="verifier-report-not-object",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL].__setitem__("kind", "stability"),
        "acceptance verifier.report.kind must be verifier-stability-report",
        id="verifier-report-kind-wrong",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL].__setitem__("reruns", 2),
        "acceptance verifier.report.reruns must be an integer >= "
        "declared verifier.reruns",
        id="verifier-report-reruns-below-declared",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL].__setitem__("min_reward", 2),
        "acceptance verifier.report.min_reward must be numeric within 0..1",
        id="verifier-report-min-reward-invalid",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL].__setitem__("flake_rate", 0.04),
        "acceptance verifier.report.flake_rate must be numeric and <= "
        "declared verifier.flake_rate",
        id="verifier-report-flake-above-declared",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL].__setitem__("runs", []),
        "acceptance verifier.report.runs must be a non-empty list",
        id="verifier-report-runs-empty",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"].append(
            {"status": "passed", "reward": 1.0}
        ),
        "acceptance verifier.report.runs length must match report reruns",
        id="verifier-report-runs-length-mismatch",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"].pop(0),
        "acceptance verifier.report.runs must include at least "
        "declared verifier.reruns",
        id="verifier-report-runs-fewer-than-declared",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"].__setitem__(0, 5),
        "acceptance verifier.report.runs[0] must be a mapping",
        id="verifier-report-run-not-mapping",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"][0].__setitem__("status", "skipped"),
        "acceptance verifier.report.runs[0].status must be passed or failed",
        id="verifier-report-run-status-invalid",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"][0].__setitem__("reward", 2),
        "acceptance verifier.report.runs[0].reward must be numeric within 0..1",
        id="verifier-report-run-reward-invalid",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"][0].__setitem__("reward", 0.5),
        "acceptance verifier.report.runs[0].reward is below min_reward",
        id="verifier-report-run-reward-below-min",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"][0].__setitem__("status", "failed"),
        "acceptance verifier.report.runs imply flake_rate 0.333, "
        "above declared verifier.flake_rate",
        id="verifier-report-runs-imply-flake",
    ),
    pytest.param(
        lambda p: p.files[VERIFIER_REL]["runs"][0].__setitem__("status", "failed"),
        "acceptance verifier.report.flake_rate is lower than failures in runs",
        id="verifier-report-flake-understated",
    ),
    pytest.param(
        lambda p: p.evidence.pop("review"),
        "acceptance validation requires benchflow.evidence.review",
        id="review-missing",
    ),
    pytest.param(
        lambda p: p.evidence["review"].__setitem__("anti_cheat", "failed"),
        "acceptance review.anti_cheat must be passed",
        id="review-anti-cheat-not-passed",
    ),
    pytest.param(
        lambda p: p.evidence["review"].pop("instruction_alignment"),
        "acceptance review.instruction_alignment must be passed",
        id="review-instruction-alignment-missing",
    ),
    pytest.param(
        lambda p: p.evidence["review"].pop("artifact"),
        "acceptance review.artifact must be declared",
        id="review-artifact-undeclared",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(REVIEW_REL, [1]),
        "acceptance review.artifact must be a JSON object",
        id="review-artifact-not-object",
    ),
    pytest.param(
        lambda p: p.files[REVIEW_REL].__setitem__("kind", "self-review"),
        "acceptance review.artifact.kind must be acceptance-review when present",
        id="review-artifact-kind-wrong",
    ),
    pytest.param(
        lambda p: p.files[REVIEW_REL].__setitem__("anti_cheat", "failed"),
        "acceptance review.artifact.anti_cheat must match "
        "benchflow.evidence.review.anti_cheat",
        id="review-artifact-disagrees-with-declared",
    ),
    pytest.param(
        lambda p: p.evidence.pop("calibration"),
        "acceptance validation requires benchflow.evidence.calibration",
        id="calibration-missing",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__("no_op_reward_max", 0.2),
        "acceptance calibration.no_op_reward_max must be numeric and <= 0.1",
        id="calibration-no-op-max-too-high",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__("known_bad_reward_max", 1.0),
        "acceptance calibration.known_bad_reward_max must be numeric and < 1.0",
        id="calibration-known-bad-max-too-high",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__(
            "partial_solution_range", [0.8, 0.2]
        ),
        "acceptance calibration.partial_solution_range must be [min, max] within 0..1",
        id="calibration-partial-range-inverted",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__(
            "partial_solution_range", "wide"
        ),
        "acceptance calibration.partial_solution_range must be [min, max] within 0..1",
        id="calibration-partial-range-not-list",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__(
            "human_or_reference_examples", []
        ),
        "acceptance calibration.human_or_reference_examples must be non-empty",
        id="calibration-examples-empty",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__(
            "human_or_reference_examples", "none"
        ),
        "acceptance calibration.human_or_reference_examples must be non-empty",
        id="calibration-examples-not-list",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__(
            "human_or_reference_examples", [5]
        ),
        "acceptance calibration.human_or_reference_examples[0] must be a mapping",
        id="calibration-example-not-mapping",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"]["human_or_reference_examples"][
            0
        ].__setitem__("expected_reward", 2),
        "acceptance calibration.human_or_reference_examples[0] "
        "expected_reward must be numeric within 0..1",
        id="calibration-example-reward-invalid",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"]["human_or_reference_examples"][
            0
        ].__setitem__("artifact", "evidence/nope.json"),
        "acceptance calibration.human_or_reference_examples[0].artifact "
        "references missing file: evidence/nope.json",
        id="calibration-example-artifact-file-missing",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(EXAMPLE_REL, [1]),
        "acceptance calibration.human_or_reference_examples[0].artifact "
        "must be a JSON object",
        id="calibration-example-artifact-not-object",
    ),
    pytest.param(
        lambda p: p.files[EXAMPLE_REL].__setitem__("expected_reward", 0.5),
        "acceptance calibration.human_or_reference_examples[0].artifact"
        ".expected_reward must match declared expected_reward",
        id="calibration-example-artifact-disagrees",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(EXAMPLE_REL, {"note": "missing reward"}),
        "acceptance calibration.human_or_reference_examples[0].artifact"
        ".expected_reward must be numeric",
        id="calibration-example-artifact-reward-missing",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].pop("report"),
        "acceptance calibration.report must be declared",
        id="calibration-report-undeclared",
    ),
    pytest.param(
        lambda p: p.evidence["calibration"].__setitem__("report", "evidence/nope.json"),
        "acceptance calibration.report references missing file: evidence/nope.json",
        id="calibration-report-file-missing",
    ),
    pytest.param(
        lambda p: p.files.__setitem__(CALIBRATION_REL, [1]),
        "acceptance calibration.report must be a JSON object",
        id="calibration-report-not-object",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL].__setitem__("kind", "calibration"),
        "acceptance calibration.report.kind must be calibration-report",
        id="calibration-report-kind-wrong",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL].__setitem__("cases", []),
        "acceptance calibration.report.cases must be a non-empty list",
        id="calibration-report-cases-empty",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"].__setitem__(0, 5),
        "acceptance calibration.report.cases[0] must be a mapping",
        id="calibration-report-case-not-mapping",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][0].__setitem__("type", "weird"),
        "acceptance calibration.report.cases[0].type must be no-op, known-bad, "
        "partial, or reference",
        id="calibration-report-case-type-invalid",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][0].__setitem__("reward", "x"),
        "acceptance calibration.report.cases[0].reward must be numeric within 0..1",
        id="calibration-report-case-reward-invalid",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][0].__setitem__("reward", 0.2),
        "acceptance calibration.report.cases[0] exceeds no_op_reward_max",
        id="calibration-no-op-case-exceeds-max",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][1].__setitem__("reward", 0.9),
        "acceptance calibration.report.cases[1] exceeds known_bad_reward_max",
        id="calibration-known-bad-case-exceeds-max",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][2].__setitem__("reward", 0.9),
        "acceptance calibration.report.cases[2] is outside partial range",
        id="calibration-partial-case-outside-range",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"][3].__setitem__("reward", 0.9),
        "acceptance calibration.report.cases[3] does not match a declared "
        "reference expected_reward",
        id="calibration-reference-case-mismatch",
    ),
    pytest.param(
        lambda p: p.files[CALIBRATION_REL]["cases"].pop(0),
        "acceptance calibration.report.cases must include a no-op case",
        id="calibration-report-missing-no-op-case",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__("artifacts", "pins"),
        "acceptance evidence.artifacts must be a list",
        id="artifacts-not-list",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__("trajectories", "pins"),
        "acceptance evidence.trajectories must be a list",
        id="trajectories-not-list",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__("artifacts", [5]),
        "acceptance evidence.artifacts[0] must be a mapping",
        id="artifact-entry-not-mapping",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__("artifacts", [{"path": ORACLE_REL}]),
        "acceptance evidence.artifacts[0].sha256 must be declared",
        id="artifact-entry-sha-missing",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__(
            "artifacts", [{"path": ORACLE_REL, "sha256": 5}]
        ),
        "acceptance evidence.artifacts[0].path sha256 must be a "
        "non-empty string when declared",
        id="artifact-entry-sha-not-string",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__("artifacts", [{"path": 5, "sha256": "00"}]),
        "acceptance evidence.artifacts[0].path must be a non-empty relative path",
        id="artifact-entry-path-not-string",
    ),
    pytest.param(
        lambda p: p.evidence.__setitem__(
            "artifacts", [{"path": "../escape.json", "sha256": "00"}]
        ),
        "acceptance evidence.artifacts[0].path must be a safe relative path",
        id="artifact-entry-path-unsafe",
    ),
    pytest.param(
        lambda p: p.pin_exclude.add(VERIFIER_REL),
        "acceptance verifier.report must be listed in "
        "benchflow.evidence.artifacts or benchflow.evidence.trajectories "
        "with sha256",
        id="primary-evidence-unpinned",
    ),
    pytest.param(
        lambda p: _tamper_reindent(p, VERIFIER_REL),
        "acceptance evidence.artifacts[1].path sha256 mismatch for "
        "evidence/verifier-stability.json",
        id="tampered-report-sha-mismatch",
    ),
]


class TestCheckAcceptanceEvidence:
    def test_golden_package_passes(self, tmp_path: Path) -> None:
        assert check(tmp_path, golden_package()) == []

    def test_oracle_last_job_evidence_passes(self, tmp_path: Path) -> None:
        package = golden_package()
        _use_last_job(package)
        assert check(tmp_path, package) == []

    def test_oracle_artifact_expected_reward_field_passes(self, tmp_path: Path) -> None:
        package = golden_package()
        package.files[ORACLE_REL] = {"expected_reward": 1.0}
        assert check(tmp_path, package) == []

    @pytest.mark.parametrize(("mutate", "expected"), EVIDENCE_REJECTS)
    def test_rejects(self, tmp_path: Path, mutate, expected: str) -> None:
        package = golden_package()
        mutate(package)
        assert expected in check(tmp_path, package)

    def test_invalid_threshold_skips_calibration_report_value_checks(
        self, tmp_path: Path
    ) -> None:
        package = golden_package()
        package.evidence["calibration"]["no_op_reward_max"] = "broken"
        package.files[CALIBRATION_REL]["cases"][0]["reward"] = 0.9
        issues = check(tmp_path, package)
        assert (
            "acceptance calibration.no_op_reward_max must be numeric and <= 0.1"
            in issues
        )
        assert not any("exceeds no_op_reward_max" in issue for issue in issues)

    def test_tampered_artifact_invalid_json(self, tmp_path: Path) -> None:
        package = golden_package()
        package.tamper[REVIEW_REL] = "{not json"
        issues = check(tmp_path, package)
        assert any(
            i.startswith("acceptance review.artifact is not valid JSON:")
            for i in issues
        )
        assert (
            "acceptance evidence.artifacts[2].path sha256 mismatch for "
            "evidence/review.json"
        ) in issues

    def test_unparseable_task_md_fails_closed(self, tmp_path: Path) -> None:
        task_dir = write_package(tmp_path / "task", golden_package())
        (task_dir / "task.md").write_text("---\n[\n---\n\n## prompt\n\nx\n")
        issues = _check_acceptance_evidence(task_dir)
        assert len(issues) == 1
        assert issues[0].startswith("acceptance validation cannot parse task.md:")

    def test_evidence_not_mapping_fails_closed(self, tmp_path: Path) -> None:
        task_dir = write_package(tmp_path / "task", golden_package())
        frontmatter = {
            "schema_version": "1.3",
            "task": {"name": "benchflow/evidence-demo", "description": "demo"},
            "agent": {"timeout_sec": 60},
            "verifier": {"timeout_sec": 30},
            "benchflow": {"evidence": "see artifacts"},
        }
        (task_dir / "task.md").write_text(
            "---\n"
            + yaml.safe_dump(frontmatter, sort_keys=False)
            + "---\n\n## prompt\n\nDo the work.\n"
        )
        assert _check_acceptance_evidence(task_dir) == [
            "acceptance validation requires benchflow.evidence mapping"
        ]


def gate_ready_package() -> EvidencePackage:
    """Golden static evidence extended with a runnable acceptance-live spec."""
    package = golden_package()
    package.files[CALIBRATION_REL] = copy.deepcopy(LIVE_CALIBRATION_REPORT)
    package.evidence["acceptance_live"] = {
        "cases": [green_case(), oracle_case()],
        "calibration": {"from": "calibration.report", "reruns": 1},
        "leaderboard": {"required": True, "max_flake_rate": 0.0},
        "report": REPORT_REL,
    }
    return package


def write_gate_ready_task(task_dir: Path, package: EvidencePackage) -> Path:
    """Write a task package that passes every static check_task level."""
    write_package(task_dir, package)
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    solve = task_dir / "oracle" / "solve.sh"
    solve.parent.mkdir()
    solve.write_text("#!/bin/bash\nexit 0\n")
    solve.chmod(0o755)
    verifier_dir = task_dir / "verifier"
    (verifier_dir / "rubrics").mkdir(parents=True)
    (verifier_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (verifier_dir / "rubrics" / "verifier.md").write_text(
        "# Rubric\n\n- `task_success`: the demo output file exists.\n"
    )
    (verifier_dir / "verifier.md").write_text("""---
document_version: "0.3"
verifier:
  name: live-task
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success: {weight: 1.0, source: deterministic}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## verifier intent

Checks the demo output file.
""")
    return task_dir


class TestCheckTaskAcceptanceWiring:
    def test_unknown_validation_level_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown task validation level: bogus"):
            check_task(tmp_path, validation_level="bogus")  # type: ignore[arg-type]

    def test_report_output_requires_acceptance_live_level(self, tmp_path: Path) -> None:
        issues = check_task(
            tmp_path,
            validation_level="acceptance",
            acceptance_live_report_output=tmp_path / "live.json",
        )
        assert issues == [
            "acceptance-live report output override requires --level acceptance-live"
        ]

    def test_evidence_gate_runs_only_at_acceptance_levels(self, tmp_path: Path) -> None:
        package = golden_package()
        package.evidence.pop("oracle_runs")
        task_dir = write_package(tmp_path / "task", package)
        missing = "acceptance validation requires benchflow.evidence.oracle_runs"
        assert missing in check_task(task_dir, validation_level="acceptance")
        assert missing not in check_task(task_dir, validation_level="structural")
        assert missing not in check_task(task_dir, validation_level="publication-grade")

    def test_static_evidence_issue_blocks_live_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live_calls: list[Path] = []
        monkeypatch.setattr(
            task_authoring,
            "run_live_acceptance_checks",
            lambda task_dir, **kwargs: live_calls.append(task_dir) or [],
        )
        package = gate_ready_package()
        package.evidence.pop("oracle_runs")
        task_dir = write_gate_ready_task(tmp_path / "live-task", package)
        issues = check_task(
            task_dir, sandbox_type="docker", validation_level="acceptance-live"
        )
        assert issues == [
            "acceptance validation requires benchflow.evidence.oracle_runs"
        ]
        assert live_calls == []

    def test_live_execution_requires_sandbox(self, tmp_path: Path) -> None:
        assert _check_live_acceptance_execution(
            tmp_path, sandbox_type=None, report_output=None
        ) == [
            "acceptance-live validation requires --sandbox <backend> to execute "
            "oracle and calibration checks"
        ]

    def test_live_execution_unparseable_task_md_fails_closed(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "task.md").write_text("---\n[\n---\n\n## prompt\n\nx\n")
        issues = _check_live_acceptance_execution(
            tmp_path, sandbox_type="docker", report_output=None
        )
        assert len(issues) == 1
        assert issues[0].startswith("acceptance-live validation cannot parse task.md:")

    def test_live_execution_requires_evidence_mapping(self, tmp_path: Path) -> None:
        frontmatter = {
            "schema_version": "1.3",
            "task": {"name": "benchflow/evidence-demo", "description": "demo"},
            "agent": {"timeout_sec": 60},
            "verifier": {"timeout_sec": 30},
        }
        (tmp_path / "task.md").write_text(
            "---\n"
            + yaml.safe_dump(frontmatter, sort_keys=False)
            + "---\n\n## prompt\n\nDo the work.\n"
        )
        assert _check_live_acceptance_execution(
            tmp_path, sandbox_type="docker", report_output=None
        ) == ["acceptance-live validation requires benchflow.evidence mapping"]

    def test_live_execution_delegates_declared_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package = golden_package()
        task_dir = write_package(tmp_path / "task", package)
        captured: dict[str, Any] = {}

        def fake_run(
            task_dir_arg: Path,
            *,
            sandbox_type: str,
            evidence: dict[str, Any],
            report_output: Path | None = None,
            write_report: bool = True,
        ) -> list[str]:
            captured.update(
                task_dir=task_dir_arg,
                sandbox_type=sandbox_type,
                evidence=evidence,
                report_output=report_output,
                write_report=write_report,
            )
            return ["live-issue"]

        monkeypatch.setattr(task_authoring, "run_live_acceptance_checks", fake_run)
        override = tmp_path / "out" / "live.json"
        issues = _check_live_acceptance_execution(
            task_dir,
            sandbox_type="daytona",
            report_output=override,
            write_report=False,
        )
        assert issues == ["live-issue"]
        assert captured == {
            "task_dir": task_dir,
            "sandbox_type": "daytona",
            "evidence": package.evidence,
            "report_output": override,
            "write_report": False,
        }

    def test_acceptance_live_green_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task_dir = write_gate_ready_task(tmp_path / "live-task", gate_ready_package())
        install_live_harness(monkeypatch, tmp_path)
        issues = check_task(
            task_dir, sandbox_type="docker", validation_level="acceptance-live"
        )
        assert issues == []
        report = json.loads((task_dir / REPORT_REL).read_text())
        assert report["kind"] == "acceptance-live-report"
        assert report["leaderboard_suitability"]["status"] == "suitable"
        assert report["summary"]["failed_runs"] == 0
        assert (task_dir / (REPORT_REL + ".sha256")).is_file()

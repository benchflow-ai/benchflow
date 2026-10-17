"""Coverage gate for the incomplete ASI-Bench parity record."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from benchflow.agent_router import build_verify_report
from benchflow.cli.main import app

PARITY = (
    Path(__file__).parents[1] / "benchmarks" / "asi-bench" / "parity_experiment.json"
)


def _complete_record() -> dict:
    samples = []
    for level in ("B1", "B2"):
        for harness in ("opencode", "pi-acp"):
            samples.append(
                {
                    "task_id": "math.demo",
                    "prompt_level": level,
                    "harness": harness,
                    "legacy_reward": 0.5,
                    "converted_reward": 0.5,
                    "artifact_sha256": {"analysis.py": "a" * 64},
                    "same_artifacts": True,
                    "details_match": True,
                    "attempt_status": "completed",
                    "evaluation_status": "completed",
                }
            )
    return {
        "benchmark": "asi-bench",
        "status": "complete",
        "coverage": {
            "status": "complete",
            "source_task_count": 2,
            "included_task_ids": ["math.demo"],
            "excluded_tasks": [
                {"task_id": "math.excluded", "reason": "requires network"}
            ],
            "required_prompt_levels": ["B1", "B2"],
            "required_harnesses": ["opencode", "pi-acp"],
        },
        "conversion_parity": {
            "tasks": [
                {
                    "task_id": "math.demo",
                    "criteria_results": [
                        {
                            "criterion_id": "scorer",
                            "original_verdict": "pass",
                            "adapted_verdict": "pass",
                        }
                    ],
                }
            ]
        },
        "agent_parity": {"results": samples},
    }


def test_shipped_asi_template_is_insufficient() -> None:
    data = json.loads(PARITY.read_text())
    assert data["coverage"]["status"] == "incomplete"
    assert build_verify_report("asi-bench", data).verdict == "insufficient-evidence"


def test_pinned_asi_source_inventory_is_accounted_for() -> None:
    """Guards ASI source commit f13175a against silent task loss."""
    data = json.loads(PARITY.read_text())
    coverage = data["coverage"]
    included = coverage["included_task_ids"]
    excluded = coverage["excluded_tasks"]
    assert coverage["source_task_count"] == 60
    assert len(included) == 46
    assert len(excluded) == 14
    assert len(set(included)) == len(included)
    assert len({item["task_id"] for item in excluded}) == len(excluded)
    assert not set(included) & {item["task_id"] for item in excluded}
    assert all(item["reason"] for item in excluded)
    assert data["scope"]["asi_revision"] == "f13175a89dc9b4873f6306a3d31e46927c38f1a9"
    assert data["scope"]["hf_revision"] == "0fa14219cafdbab634d8b3cfbce238a8735a214f"


def test_template_stays_insufficient_with_partial_matching_samples() -> None:
    data = _complete_record()
    data["status"] = "template"
    assert build_verify_report("asi-bench", data).verdict == "insufficient-evidence"


def test_coverage_requires_every_task_level_harness_cell() -> None:
    data = _complete_record()
    data["agent_parity"]["results"].pop()
    assert build_verify_report("asi-bench", data).verdict == "insufficient-evidence"

    data = _complete_record()
    data["agent_parity"]["results"][0]["artifact_sha256"] = {}
    assert build_verify_report("asi-bench", data).verdict == "insufficient-evidence"


def test_complete_coverage_and_matching_scores_can_confirm() -> None:
    assert (
        build_verify_report("asi-bench", _complete_record()).verdict
        == "parity-confirmed"
    )


def test_coverage_inventory_must_account_for_all_source_tasks() -> None:
    data = _complete_record()
    data["coverage"]["source_task_count"] = 3
    assert build_verify_report("asi-bench", data).verdict == "insufficient-evidence"


def test_canonical_verify_rejects_partial_asi_coverage(tmp_path: Path) -> None:
    data = _complete_record()
    data["agent_parity"]["results"].pop()
    benchmark = tmp_path / "asi-bench"
    benchmark.mkdir()
    (benchmark / "parity_experiment.json").write_text(json.dumps(data))
    result = CliRunner().invoke(
        app,
        ["eval", "adopt", "asi-bench", "--verify", "--benchmarks-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "insufficient-evidence" in result.output
    assert "coverage is missing 1" in result.output

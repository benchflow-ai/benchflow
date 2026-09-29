"""Execution vs assessment status and verifier-recovery evidence in the viewer.

A verifier failure after a clean agent run used to read as a generic "error"
row. The viewer now reports execution (agent ``error``) and assessment
(reward, integrated scoring from #1126, ``verifier_error``) separately, and
shows the verifier-only recovery receipts that verifier recovery publishes
(``verification.json`` pointer, ``recovery.json``, ``publication_error``).
"""

import json
from pathlib import Path

import pytest

from benchflow.trajectories.viewer.catalog import _rollout_summary
from benchflow.trajectories.viewer.payload import _build_acp_payload


def _rollout(base: Path, result: dict | None = None) -> Path:
    (base / "trajectory").mkdir(parents=True, exist_ok=True)
    (base / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": "done"})
    )
    if result is not None:
        (base / "result.json").write_text(json.dumps(result))
    return base


def _status(base: Path) -> dict:
    return _build_acp_payload(base, None).to_payload()["meta"]["status"]


COMPLETE_SCORING = {
    "schema_version": 1,
    "policy": "tests-blockers-quality-v1",
    "status": "complete",
    "passed": True,
    "tests_pass": True,
    "all_blockers_pass": True,
    "failed_blockers": [],
    "verifier_reward": 1.0,
    "rubric_reward": 0.75,
    "reviewer_run": "reviews/abc",
}


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {"rewards": {"reward": 1.0}},
            ("completed", None, "scored", None, "completed and scored"),
        ),
        (
            {
                "rewards": None,
                "verifier_error": "verifier timed out after 30.0s",
                "verifier_error_category": "verifier_timeout",
            },
            (
                "completed",
                None,
                "unscored",
                "verifier timeout",
                "completed but unscored",
            ),
        ),
        (
            # No stored category: classified like _utils/scoring does.
            {"verifier_error": "verifier crashed: Failed to get session command: "},
            ("completed", None, "unscored", "verifier infra", "completed but unscored"),
        ),
        (
            {"error": "agent timed out", "error_category": "agent_timeout"},
            (
                "timed_out",
                "agent timeout",
                "unscored",
                "no verdict after the execution error",
                "timed out, unscored",
            ),
        ),
        (
            {"error": "ACP error -32603: Internal error"},
            (
                "errored",
                "acp error",
                "unscored",
                "no verdict after the execution error",
                "errored, unscored",
            ),
        ),
        (
            {
                "error": "Prompt exceeded wall-clock budget",
                "partial_trajectory": True,
                "rewards": {"reward": 0.0},
            },
            ("timed_out", "timeout", "scored", None, "timed out but scored"),
        ),
        (
            {"rewards": {"reward": 0.75}, "scoring": COMPLETE_SCORING},
            (
                "completed",
                None,
                "scored",
                "tests + rubric, passed",
                "completed and scored",
            ),
        ),
        (
            {
                "rewards": {"reward": 1.0},
                "scoring": {"status": "error", "error": "reviewer crashed"},
            },
            (
                "completed",
                None,
                "unscored",
                "scoring error: reviewer crashed",
                "completed but unscored",
            ),
        ),
        (
            # A stale reward beside a malformed block is never a verdict.
            {"rewards": {"reward": 1.0}, "scoring": {"status": "complete"}},
            (
                "completed",
                None,
                "unscored",
                "malformed scoring block",
                "completed but unscored",
            ),
        ),
    ],
)
def test_execution_and_assessment_are_reported_separately(tmp_path, result, expected):
    """Guards the run page against conflating a verifier/scoring failure
    (#1126 scoring, verifier_error) with an agent execution failure."""
    status = _status(_rollout(tmp_path, result))
    assert (
        status["execution"],
        status["execution_detail"],
        status["assessment"],
        status["assessment_detail"],
        status["summary"],
    ) == expected


def test_runs_without_a_result_report_no_status(tmp_path):
    """Guards trajectory-only directories against invented status."""
    assert _status(_rollout(tmp_path)) is None


def test_catalog_rows_carry_both_statuses(tmp_path):
    """Guards the run list: a verifier error reads 'completed but unscored'."""
    _rollout(
        tmp_path / "job" / "smoke__1",
        {
            "task_name": "smoke",
            "verifier_error": "verifier timed out after 30.0s",
            "verifier_error_category": "verifier_timeout",
        },
    )
    row = _rollout_summary(tmp_path, "job/smoke__1")
    assert row["execution"] == "completed"
    assert row["assessment"] == "unscored"
    assert row["assessment_detail"] == "verifier timeout"
    assert row["status_summary"] == "completed but unscored"
    assert row["has_error"] is True  # the pre-existing flag is unchanged


@pytest.mark.parametrize(
    "result,detail",
    [
        (
            {
                "rewards": {"reward": 1.0},
                "assessment": {"status": "pending", "reason": "awaiting_reviewer"},
            },
            "assessment pending (awaiting reviewer)",
        ),
        (
            {"rewards": {"reward": 1.0}, "assessment": "unassessable"},
            "assessment unassessable",
        ),
        (
            # Legacy benchflow.robotics manifest row.
            {"status": "awaiting_assessment", "rewards": {"reward": 1.0}},
            "assessment pending",
        ),
        (
            # A malformed status can never unlock a score.
            {"rewards": {"reward": 1.0}, "assessment": {"status": ["verified"]}},
            "assessment unassessable",
        ),
    ],
)
def test_a_withheld_assessment_is_unscored_in_the_viewer(tmp_path, result, detail):
    """Guards the viewer's reading of declared assessments: a result whose declared assessment is
    pending or unassessable is unscored in the shared score accounting, so the
    run page and run list must not show its stale reward as a verdict."""
    root = _rollout(tmp_path / "job" / "arm__1", {"task_name": "arm", **result})
    meta = _build_acp_payload(root, None).to_payload()["meta"]
    assert meta["reward"] is None
    assert meta["status"]["assessment"] == "unscored"
    assert meta["status"]["assessment_detail"] == detail
    assert meta["status"]["summary"] == "completed but unscored"
    row = _rollout_summary(tmp_path, "job/arm__1")
    assert row["reward"] is None
    assert row["assessment"] == "unscored"


def test_an_assessed_result_keeps_its_reward_in_the_viewer(tmp_path):
    """Guards the other side of the same merge: a verified or failed
    assessment is a verdict and its reward is shown."""
    root = _rollout(
        tmp_path,
        {"rewards": {"reward": 0.0}, "assessment": {"status": "failed"}},
    )
    meta = _build_acp_payload(root, None).to_payload()["meta"]
    assert meta["reward"] == 0.0
    assert meta["status"]["assessment"] == "scored"
    assert meta["status"]["summary"] == "completed and scored"


def _recovery(root: Path, attempt: str, record: dict, *, point: bool = True) -> None:
    folder = root / "verifier-recovery" / attempt
    folder.mkdir(parents=True)
    (folder / "recovery.json").write_text(
        json.dumps({"attempt": f"verifier-recovery/{attempt}", **record})
    )
    if point:
        (root / "verification.json").write_text(
            json.dumps({"attempt": f"verifier-recovery/{attempt}"})
        )


def test_recovery_receipts_are_read_with_the_admitted_attempt_first(tmp_path):
    """Guards the recovery rows over the recovery receipts: pointer,
    admitted attempt details and any other attempts."""
    root = _rollout(
        tmp_path,
        {
            "verifier_error": "[solver-preserved] verifier recovery failed: boom",
            "verifier_error_category": "verifier_infra",
        },
    )
    _recovery(
        root,
        "aaa111",
        {"status": "unavailable", "error": "lease missing"},
        point=False,
    )
    _recovery(
        root,
        "bbb222",
        {
            "status": "failed",
            "original_error": "[solver-preserved] verifier_wedge: gone",
            "evidence": "workspace-only",
            "solver_replayed": False,
            "timing": {"verifier": 12.5},
            "rewards": None,
            "error": "verifier crashed: boom",
        },
    )
    recovery = _build_acp_payload(root, None).to_payload()["verifier"]["recovery"]
    assert recovery["pointer"] == "verifier-recovery/bbb222"
    admitted, other = recovery["attempts"]
    assert admitted["admitted"] is True and admitted["status"] == "failed"
    assert admitted["original_error"] == "[solver-preserved] verifier_wedge: gone"
    assert admitted["solver_replayed"] is False
    assert admitted["verifier_sec"] == 12.5
    assert other == {**other, "attempt": "verifier-recovery/aaa111", "admitted": False}


def test_failed_publication_is_flagged_on_the_run_page(tmp_path):
    """Guards the documented pitfall of atomic recovery publishing: after a failed
    publication verifier/ predates the admitted score, so the page says so."""
    root = _rollout(tmp_path, {"rewards": {"reward": 1}})
    _recovery(
        root,
        "ccc333",
        {
            "status": "complete",
            "rewards": {"reward": 1},
            "publication_error": "OSError: [Errno 28] No space left on device",
        },
    )
    payload = _build_acp_payload(root, None).to_payload()
    (attempt,) = payload["verifier"]["recovery"]["attempts"]
    assert attempt["reward"] == 1.0
    assert attempt["publication_error"].startswith("OSError")
    (banner,) = payload["meta"]["errors"]
    assert banner["label"] == "verifier publication error"
    assert banner["level"] == "info"
    assert "verifier-recovery/ccc333/verifier/" in banner["text"]


def test_invalid_pointer_and_missing_receipt(tmp_path):
    """Guards the pointer check (same shape verification_source accepts)
    and a pointer whose receipt was not fetched (hf:// slices)."""
    root = _rollout(tmp_path, {"rewards": {"reward": 0}})
    (root / "verification.json").write_text(json.dumps({"attempt": "../../etc"}))
    recovery = _build_acp_payload(root, None).to_payload()["verifier"]["recovery"]
    assert recovery["pointer"] is None
    assert recovery["pointer_error"] == (
        "does not reference verifier-recovery/<attempt id>"
    )

    (root / "verification.json").write_text(
        json.dumps({"attempt": "verifier-recovery/ddd444"})
    )
    recovery = _build_acp_payload(root, None).to_payload()["verifier"]["recovery"]
    (attempt,) = recovery["attempts"]
    assert attempt["admitted"] is True and attempt["status"] is None


def test_runs_without_recovery_have_none(tmp_path):
    """Guards ordinary runs against empty recovery rows."""
    root = _rollout(tmp_path, {"rewards": {"reward": 1}})
    assert _build_acp_payload(root, None).to_payload()["verifier"]["recovery"] is None

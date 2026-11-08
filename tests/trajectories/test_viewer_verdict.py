"""The run card's derived data: the "why" sentence and failing-check evidence.

Guards EN-327 (bench view run card): every sentence and number the card
shows is read from the rollout's own artifacts, never composed from nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.trajectories.viewer.models import VerifierTest
from benchflow.trajectories.viewer.payload import _build_acp_payload, _ctrf_seconds
from benchflow.trajectories.viewer.verdict import (
    failure_assertion,
    failure_clause,
    failure_location,
    format_reward,
    short_test_name,
)

# The tail of a real pytest-json-ctrf trace (SkillsBench shock-analysis-supply).
PYTEST_TRACE = (
    "def test_value_magnitudes():\n"
    "        for val in ystar_values:\n"
    '>           assert 10000 < val < 500000, f"Ystar_base should be in thousands '
    '(10k-500k), got {val:.2f}"\n'
    "E           AssertionError: Ystar_base should be in thousands (10k-500k), got 78.74\n"
    "E           assert 10000 < 78.7357212973561\n"
    "\n"
    "/tests/test_outputs.py:261: AssertionError"
)


def _rollout(
    base: Path,
    events: list[dict],
    result: dict,
    *,
    ctrf: dict | None = None,
    timing: dict | None = None,
) -> Path:
    folder = base / "task__abcd1234"
    (folder / "trajectory").mkdir(parents=True)
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    (folder / "result.json").write_text(json.dumps({"task_name": "task", **result}))
    if ctrf is not None:
        (folder / "verifier").mkdir()
        (folder / "verifier" / "ctrf.json").write_text(json.dumps(ctrf))
    if timing is not None:
        (folder / "timing.json").write_text(json.dumps(timing))
    return folder


def _ctrf(*tests: dict, start: float = 1776902723.88, stop: float = 1776902724.96):
    return {
        "results": {
            "tool": {"name": "pytest", "version": "8.4.1"},
            "summary": {"tests": len(tests), "start": start, "stop": stop},
            "tests": list(tests),
        }
    }


def _tool(call_id: str, title: str, status: str, kind: str = "execute") -> dict:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": kind,
        "title": title,
        "status": status,
        "content": [{"type": "content", "content": {"type": "text", "text": "out"}}],
    }


def _verdict(folder: Path) -> dict:
    return _build_acp_payload(folder, None).to_payload()["verdict"]


def test_pytest_trace_gives_the_failing_frame_and_assertion() -> None:
    assert failure_location(PYTEST_TRACE) == "tests/test_outputs.py:261"
    assert failure_assertion(PYTEST_TRACE) == (
        "AssertionError: Ystar_base should be in thousands (10k-500k), got 78.74"
    )


def test_trace_without_explanation_falls_back_to_the_message() -> None:
    assert failure_location("no frame here") is None
    assert failure_assertion("plain text", "  boom  ") == "boom"
    assert failure_assertion(None, None) is None


def test_reward_and_test_name_formatting() -> None:
    assert format_reward(1.0) == "1.0"
    assert format_reward(0.0) == "0.0"
    assert format_reward(0.75) == "0.75"
    assert short_test_name("test_outputs.py::test_value_magnitudes") == (
        "test_value_magnitudes"
    )
    assert short_test_name("plain") == "plain"


def test_ctrf_seconds_reads_timestamps_in_either_unit() -> None:
    """pytest-json-ctrf writes epoch seconds; the CTRF spec says milliseconds."""
    assert _ctrf_seconds({"start": 1776902724.08, "stop": 1776902724.16}) == (
        1776902724.16 - 1776902724.08
    )
    assert _ctrf_seconds({"start": 1776902724080, "stop": 1776902724160}) == 0.08
    assert _ctrf_seconds({"start": 5, "stop": 4}) is None
    assert _ctrf_seconds({"duration": 3}) is None


def test_failed_run_names_the_failing_test_location_and_assertion(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [{"type": "agent_message", "text": "done"}],
        {"rewards": {"reward": 0.0}},
        ctrf=_ctrf(
            {"name": "test_outputs.py::test_ok", "status": "passed", "duration": 0.1},
            {
                "name": "test_outputs.py::test_value_magnitudes",
                "status": "failed",
                "duration": 0.079,
                "start": 1776902724.862339,
                "stop": 1776902724.9629333,
                "message": "The test failed in the call phase due to an assertion error",
                "trace": PYTEST_TRACE,
            },
        ),
    )
    payload = _build_acp_payload(folder, None).to_payload()
    assert payload["verdict"] == {
        "tone": "fail",
        "sentence": "Scored 0.0 because test_value_magnitudes failed: Ystar_base "
        "should be in thousands (10k-500k), got 78.74 (tests/test_outputs.py:261).",
        "step": None,
        "test": "test_outputs.py::test_value_magnitudes",
    }
    failed = payload["verifier"]["ctrf"][1]
    assert failed["location"] == "tests/test_outputs.py:261"
    assert round(failed["seconds"], 4) == 0.1006
    assert payload["verifier"]["ctrf_tool"] == "pytest 8.4.1"
    assert round(payload["verifier"]["ctrf_seconds"], 2) == 1.08
    passed = payload["verifier"]["ctrf"][0]
    assert passed["trace"] is None and passed["message"] is None


def test_more_failures_are_counted(tmp_path) -> None:
    failing = [
        {"name": f"t.py::test_{n}", "status": "failed", "trace": "E   assert 0"}
        for n in range(3)
    ]
    folder = _rollout(tmp_path, [], {"rewards": {"reward": 0.0}}, ctrf=_ctrf(*failing))
    sentence = _verdict(folder)["sentence"]
    assert sentence.startswith("Scored 0.0 because test_0 asserted 0.")
    assert sentence.endswith(" 2 more of 3 tests failed.")


def test_long_trace_keeps_its_end(tmp_path) -> None:
    trace = "x\n" * 5000 + PYTEST_TRACE
    folder = _rollout(
        tmp_path,
        [],
        {"rewards": {"reward": 0.0}},
        ctrf=_ctrf({"name": "t.py::test_a", "status": "failed", "trace": trace}),
    )
    test = _build_acp_payload(folder, None).to_payload()["verifier"]["ctrf"][0]
    assert test["trace"].startswith("…")
    assert test["trace"].endswith("/tests/test_outputs.py:261: AssertionError")
    assert test["location"] == "tests/test_outputs.py:261"


def test_passed_run_counts_its_tests(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [],
        {"rewards": {"reward": 1.0}},
        ctrf=_ctrf(
            *({"name": f"t.py::test_{n}", "status": "passed"} for n in range(4))
        ),
    )
    assert _verdict(folder) == {
        "tone": "pass",
        "sentence": "Scored 1.0: all 4 tests passed.",
        "step": None,
        "test": None,
    }


def test_partial_credit_names_the_failure_without_because(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [],
        {"rewards": {"reward": 1.0}},
        ctrf=_ctrf(
            {"name": "t.py::test_a", "status": "passed"},
            {"name": "t.py::test_b", "status": "failed", "trace": "E   assert 1 == 2"},
        ),
    )
    assert _verdict(folder)["sentence"] == (
        "Scored 1.0 with 1 of 2 tests failing; test_b asserted 1 == 2."
    )


def test_timeout_names_the_call_it_interrupted(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [
            _tool("c1", "ls", "completed"),
            _tool("c2", "make build", "in_progress"),
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 900,
                "pending_tool_call_ids": ["c2"],
            },
        ],
        {"error": "Agent timed out after 900s", "error_category": "timeout"},
    )
    assert _verdict(folder) == {
        "tone": "unscored",
        "sentence": "Not scored. Agent timed out after 900s. "
        "make build (#2) was still running.",
        "step": 2,
        "test": None,
    }


def test_scored_after_timeout_keeps_the_score(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [_tool("c1", "pytest", "completed")],
        {
            "rewards": {"reward": 0.0},
            "error": "Agent prompt exceeded wall-clock budget 10800s",
            "error_category": "timeout",
        },
    )
    assert _verdict(folder)["sentence"] == (
        "Scored 0.0 after the run was cut short. Agent prompt exceeded "
        "wall-clock budget 10800s. The last tool call was pytest (#1), completed."
    )


def test_error_names_the_last_call(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [
            _tool("c1", "Read File", "completed", kind="read"),
            _tool("c2", "Edit a.py", "failed", kind="edit"),
        ],
        {
            "error": "ACP error -32603: Internal error: You've hit your limit",
            "error_category": "usage_limit",
        },
    )
    assert _verdict(folder) == {
        "tone": "unscored",
        "sentence": "Not scored. The agent stopped with an error (usage limit): "
        "ACP error -32603: Internal error: You've hit your limit. "
        "The last tool call was Edit a.py (#2), failed.",
        "step": 2,
        "test": None,
    }


def test_unscored_clean_run_says_why_not(tmp_path) -> None:
    folder = _rollout(
        tmp_path,
        [{"type": "agent_message", "text": "done"}],
        {
            "verifier_error": "verifier timed out",
            "verifier_error_category": "verifier_timeout",
        },
    )
    assert _verdict(folder)["sentence"] == (
        "The agent finished, but the run was not scored: verifier timeout."
    )


def test_failed_run_without_test_report(tmp_path) -> None:
    folder = _rollout(tmp_path, [], {"rewards": {"reward": 0.0}})
    assert _verdict(folder)["sentence"] == (
        "Scored 0.0. The rollout has no per-test results (verifier/ctrf.json)."
    )


def test_failure_clause_reads_as_a_sentence() -> None:
    def clause(assertion, location="tests/test_outputs.py:12"):
        return failure_clause(
            VerifierTest(
                name="test_outputs.py::test_x",
                status="failed",
                duration=None,
                location=location,
                assertion=assertion,
            )
        )

    assert (
        clause("assert 3 == 4") == "test_x asserted 3 == 4 (tests/test_outputs.py:12)"
    )
    assert clause("AssertionError: wrong total") == (
        "test_x failed: wrong total (tests/test_outputs.py:12)"
    )
    assert clause("FileNotFoundError: out.csv", None) == (
        "test_x failed: FileNotFoundError: out.csv"
    )
    assert clause("AssertionError", None) == "test_x failed"
    assert clause(None) == "test_x failed (tests/test_outputs.py:12)"

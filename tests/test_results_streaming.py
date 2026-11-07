"""Guards commit 0ef69259: C7 raw provider log memory at results export."""

import json
import tracemalloc

import pytest

from benchflow.trajectories.results import _llm_steps_from_trajectory


def _steps(root):
    return _llm_steps_from_trajectory(
        root, reward=1.0, is_truncated=False, trajectory_id_prefix="task"
    )


@pytest.mark.parametrize("successful", [False, True])
def test_large_repeated_requests_do_not_fill_export_memory(tmp_path, successful):
    path = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    path.parent.mkdir()
    # Failed calls emit no training rows; identical successful retries emit one.
    record: dict = {
        "request": {
            "body": {"messages": [{"role": "user", "content": "x" * (512 * 1024)}]}
        },
        "response": {"status_code": 500, "body": {"error": "failed"}},
    }
    if successful:
        record["response"] = {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        }
    with path.open("w") as handle:
        for _ in range(64):
            handle.write(json.dumps(record) + "\n")
    tracemalloc.start()
    try:
        steps, tools, error = _steps(tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(steps) == int(successful)
    assert tools == [] and error is None
    if successful:
        assert steps[0]["extras"]["exchange_index"] == 63
    assert peak < 12 * 1024 * 1024, f"export allocated {peak / 1024**2:.1f} MiB"


@pytest.mark.parametrize("bad_line", ["{", "[]"])
def test_malformed_tail_still_fails_closed(tmp_path, bad_line):
    path = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n\n" + bad_line + "\n")
    steps, tools, error = _steps(tmp_path)
    assert steps == tools == []
    assert error is not None and "line 3" in error

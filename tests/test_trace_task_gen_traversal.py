"""Path traversal regression tests for benchflow.traces.task_gen (issue #376).

Trace file paths flow into three generated artifacts:

* ``instruction.md`` — the task description shown to the agent
* ``tests/test.sh`` — the verifier that decides task reward
* ``solution/solve.sh`` — the oracle that replays trace writes

If trace paths contain ``..`` segments or are absolute, the generated
artifacts encode writes and verification outside the task workspace,
and a Docker oracle can earn reward 1.0 by escaping ``/app``.

The fix drops unsafe paths from each consumer with a warning, so
generated tasks never instruct or verify outside the workspace.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchflow.traces.models import GitContext, ParsedTrace, ToolCall, TraceStep
from benchflow.traces.task_gen import generate_task


def _trace_with_paths(paths: list[str]) -> ParsedTrace:
    """Build a trace whose Write tool calls reference ``paths``."""
    tool_calls = [
        ToolCall(
            name="Write",
            input={"file_path": p, "content": f"contents for {p}"},
        )
        for p in paths
    ]
    return ParsedTrace(
        trace_id="parent-escape-001",
        session_id="sess-001",
        agent_name="claude-code",
        model="claude-sonnet-4-20250514",
        started_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 1, 15, 10, 5, 0, tzinfo=UTC),
        steps=[
            TraceStep(role="user", content="Create a marker file"),
            TraceStep(
                role="assistant",
                content="I'll create it.",
                tool_calls=tool_calls,
            ),
            TraceStep(role="assistant", content="Done."),
        ],
        git=GitContext(repo="user/proj", branch="main"),
        cwd="/app",
    )


# ---------------------------------------------------------------------------
# All-unsafe trace — every consumer drops the path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsafe",
    [
        # ``..`` segments survive _relativize_path and would escape /app.
        "../../tmp/benchflow_escape_marker.txt",
        # Mid-path ``..`` that ends up relative but still has parent refs.
        "src/../../../etc/passwd",
        # Pure ``..`` parent reference.
        "..",
    ],
)
def test_unsafe_paths_dropped_from_generated_artifacts(
    tmp_path: Path, unsafe: str, caplog
) -> None:
    trace = _trace_with_paths([unsafe])
    out = tmp_path / "tasks"
    with caplog.at_level("WARNING"):
        task_dir = generate_task(trace, out)

    instruction = (task_dir / "instruction.md").read_text()
    test_sh = (task_dir / "tests" / "test.sh").read_text()
    solve_sh = (task_dir / "solution" / "solve.sh").read_text()

    # No artifact still encodes a workspace-escape (``..`` segment).
    for artifact, label in (
        (instruction, "instruction.md"),
        (test_sh, "test.sh"),
        (solve_sh, "solve.sh"),
    ):
        assert ".." not in artifact, f"{label} still references '..'"

    # Solve.sh has no writes left to perform; verifier reports no checks.
    assert "No replayable file writes" in solve_sh
    assert "No file checks available" in test_sh

    # At least one warning was logged so the drop is auditable.
    assert any("escape" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Mixed trace — unsafe dropped, safe retained
# ---------------------------------------------------------------------------


def test_safe_path_kept_unsafe_dropped(tmp_path: Path) -> None:
    trace = _trace_with_paths(["src/keep_me.py", "../../tmp/drop_me.txt"])
    task_dir = generate_task(trace, tmp_path / "tasks")

    instruction = (task_dir / "instruction.md").read_text()
    test_sh = (task_dir / "tests" / "test.sh").read_text()
    solve_sh = (task_dir / "solution" / "solve.sh").read_text()

    # Safe path survives every artifact.
    assert "src/keep_me.py" in instruction
    assert "src/keep_me.py" in test_sh
    assert "src/keep_me.py" in solve_sh

    # Unsafe path appears nowhere.
    for artifact in (instruction, test_sh, solve_sh):
        assert "drop_me" not in artifact
        assert ".." not in artifact

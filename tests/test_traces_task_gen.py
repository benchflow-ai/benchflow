"""Tests for benchflow.traces.task_gen — task generation from traces."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchflow.traces.models import GitContext, ParsedTrace, ToolCall, TraceStep
from benchflow.traces.task_gen import (
    generate_task,
    generate_tasks_from_traces,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_trace() -> ParsedTrace:
    """A minimal trace with one user prompt and one assistant response."""
    return ParsedTrace(
        trace_id="test-trace-001",
        session_id="sess-001",
        agent_name="claude-code",
        model="claude-sonnet-4-20250514",
        started_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 1, 15, 10, 5, 0, tzinfo=UTC),
        steps=[
            TraceStep(role="user", content="Create a hello.txt file"),
            TraceStep(
                role="assistant",
                content="I'll create that file.",
                tool_calls=[
                    ToolCall(
                        name="Write",
                        input={"file_path": "hello.txt", "content": "Hello"},
                    )
                ],
            ),
            TraceStep(role="assistant", content="Done! File created successfully."),
        ],
        git=GitContext(repo="user/my-project", branch="main"),
        cwd="/home/user/my-project",
        outcome="success",
        tags=["python"],
    )


@pytest.fixture()
def complex_trace() -> ParsedTrace:
    """A trace with many tool calls (hard difficulty)."""
    tool_steps = [
        TraceStep(
            role="assistant",
            content=f"Editing file {i}",
            tool_calls=[
                ToolCall(name="Edit", input={"file_path": f"src/file_{i}.py"})
            ],
        )
        for i in range(25)
    ]
    return ParsedTrace(
        trace_id="test-trace-complex",
        session_id="sess-complex",
        agent_name="claude-code",
        steps=[
            TraceStep(role="user", content="Refactor the entire module"),
            *tool_steps,
        ],
        outcome="success",
    )


@pytest.fixture()
def no_prompt_trace() -> ParsedTrace:
    """A trace with no user prompt."""
    return ParsedTrace(
        trace_id="test-trace-noprompt",
        session_id="sess-np",
        steps=[TraceStep(role="assistant", content="Starting task...")],
    )


# ---------------------------------------------------------------------------
# Task generation tests
# ---------------------------------------------------------------------------


class TestGenerateTask:
    def test_creates_task_directory(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)

        assert task_dir.exists()
        assert (task_dir / "task.toml").exists()
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "environment" / "Dockerfile").exists()
        assert (task_dir / "tests" / "test.sh").exists()

    def test_task_toml_content(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        toml_text = (task_dir / "task.toml").read_text()

        assert 'version = "1.0"' in toml_text
        assert "[task]" in toml_text
        assert 'name = "trace-import/' in toml_text
        assert "[metadata]" in toml_text
        assert 'difficulty = "easy"' in toml_text
        assert '"from-trace"' in toml_text
        assert 'source_trace_id = "test-trace-001"' in toml_text
        assert "[agent]" in toml_text
        assert "[verifier]" in toml_text
        assert "[environment]" in toml_text
        assert "build_timeout_sec = 600" in toml_text
        assert "storage_mb = 10240" in toml_text

    def test_instruction_md_content(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        instruction = (task_dir / "instruction.md").read_text()

        assert "Create a hello.txt file" in instruction

    def test_instruction_includes_files(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        instruction = (task_dir / "instruction.md").read_text()

        assert "`hello.txt`" in instruction

    def test_generates_test_sh(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)

        test_sh = task_dir / "tests" / "test.sh"
        assert test_sh.exists()
        content = test_sh.read_text()
        assert "#!/bin/bash" in content
        assert "hello.txt" in content
        assert "/logs/verifier/reward.txt" in content

    def test_test_sh_is_executable(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        test_sh = task_dir / "tests" / "test.sh"

        import stat

        mode = test_sh.stat().st_mode
        assert mode & stat.S_IXUSR

    def test_dockerfile_generated(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        dockerfile = task_dir / "environment" / "Dockerfile"

        assert dockerfile.exists()
        content = dockerfile.read_text()
        assert "FROM ubuntu:24.04" in content
        assert "/logs/verifier" in content

    def test_passes_bench_tasks_check(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        """Generated tasks pass bench tasks check structural validation."""
        from benchflow.tasks import check_task

        task_dir = generate_task(simple_trace, tmp_path)
        issues = check_task(task_dir)
        assert issues == [], f"bench tasks check found issues: {issues}"

    def test_test_sh_fallback_when_no_files(
        self, tmp_path: Path
    ) -> None:
        """Tasks with no files_edited still get a pass-through test.sh."""
        trace = ParsedTrace(
            trace_id="no-files",
            session_id="s",
            steps=[
                TraceStep(role="user", content="Do something"),
                TraceStep(role="assistant", content="Done"),
            ],
        )
        task_dir = generate_task(trace, tmp_path)
        test_sh = task_dir / "tests" / "test.sh"
        assert test_sh.exists()
        content = test_sh.read_text()
        assert "/logs/verifier/reward.txt" in content

    def test_hard_difficulty(self, complex_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(complex_trace, tmp_path)
        toml_text = (task_dir / "task.toml").read_text()

        # 25 tool calls + 25 files → weighted score should be hard or expert
        assert 'difficulty = "hard"' in toml_text or 'difficulty = "expert"' in toml_text

    def test_timeout_scales_with_difficulty(
        self, simple_trace: ParsedTrace, complex_trace: ParsedTrace, tmp_path: Path
    ) -> None:
        """Harder tasks get longer timeouts when timeout_sec=0 (auto)."""
        easy_dir = generate_task(simple_trace, tmp_path / "easy", timeout_sec=0)
        hard_dir = generate_task(complex_trace, tmp_path / "hard", timeout_sec=0)

        easy_toml = (easy_dir / "task.toml").read_text()
        hard_toml = (hard_dir / "task.toml").read_text()

        # Extract timeout values
        import re

        easy_timeout = int(re.search(r"timeout_sec = (\d+)", easy_toml).group(1))
        hard_timeout = int(re.search(r"timeout_sec = (\d+)", hard_toml).group(1))
        assert hard_timeout > easy_timeout

    def test_skip_existing_without_overwrite(
        self, simple_trace: ParsedTrace, tmp_path: Path
    ) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        # Create a marker file
        marker = task_dir / "marker.txt"
        marker.write_text("original")

        # Second call should skip
        generate_task(simple_trace, tmp_path, overwrite=False)
        assert marker.read_text() == "original"

    def test_overwrite_existing(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        # Add a marker
        (task_dir / "marker.txt").write_text("old")

        # Overwrite should regenerate
        task_dir = generate_task(simple_trace, tmp_path, overwrite=True)
        assert (task_dir / "task.toml").exists()

    def test_custom_author(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path, author="my-team")
        toml_text = (task_dir / "task.toml").read_text()

        assert 'author_name = "my-team"' in toml_text

    def test_category_from_repo(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        toml_text = (task_dir / "task.toml").read_text()

        assert 'category = "my-project"' in toml_text

    def test_model_in_toml(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        task_dir = generate_task(simple_trace, tmp_path)
        toml_text = (task_dir / "task.toml").read_text()

        assert "claude-sonnet-4-20250514" in toml_text


# ---------------------------------------------------------------------------
# Batch generation tests
# ---------------------------------------------------------------------------


class TestGenerateTasksFromTraces:
    def test_batch_generation(self, simple_trace: ParsedTrace, tmp_path: Path) -> None:
        traces = [simple_trace]
        results = generate_tasks_from_traces(traces, tmp_path)

        assert len(results) == 1
        assert results[0].exists()

    def test_filters_by_min_steps(self, tmp_path: Path) -> None:
        short_trace = ParsedTrace(
            trace_id="short",
            session_id="s",
            steps=[TraceStep(role="user", content="hi")],
        )
        results = generate_tasks_from_traces([short_trace], tmp_path, min_steps=2)

        assert len(results) == 0

    def test_filters_by_outcome(
        self, simple_trace: ParsedTrace, tmp_path: Path
    ) -> None:
        results = generate_tasks_from_traces(
            [simple_trace], tmp_path, outcome_filter="failure"
        )

        assert len(results) == 0

    def test_filters_no_prompt(self, no_prompt_trace: ParsedTrace, tmp_path: Path) -> None:
        results = generate_tasks_from_traces([no_prompt_trace], tmp_path, min_steps=1)

        assert len(results) == 0

    def test_filters_zero_tool_calls(self, tmp_path: Path) -> None:
        """Traces with no tool calls (e.g. pure explanations) are filtered out."""
        explanation_trace = ParsedTrace(
            trace_id="explain-only",
            session_id="s-explain",
            steps=[
                TraceStep(role="user", content="Explain asyncio"),
                TraceStep(role="assistant", content="Here is how asyncio works..."),
            ],
            outcome="success",
        )
        results = generate_tasks_from_traces([explanation_trace], tmp_path)

        assert len(results) == 0


# ---------------------------------------------------------------------------
# Model property tests
# ---------------------------------------------------------------------------


class TestParsedTraceProperties:
    def test_first_user_prompt(self, simple_trace: ParsedTrace) -> None:
        assert simple_trace.first_user_prompt == "Create a hello.txt file"

    def test_tool_names_used(self, simple_trace: ParsedTrace) -> None:
        assert simple_trace.tool_names_used == ["Write"]

    def test_files_edited(self, simple_trace: ParsedTrace) -> None:
        assert simple_trace.files_edited == ["hello.txt"]

    def test_n_tool_calls(self, simple_trace: ParsedTrace) -> None:
        assert simple_trace.n_tool_calls == 1

    def test_duration_sec(self, simple_trace: ParsedTrace) -> None:
        assert simple_trace.duration_sec == 300.0

    def test_duration_none_without_timestamps(self) -> None:
        trace = ParsedTrace(trace_id="t", session_id="s")
        assert trace.duration_sec is None

"""Layer 1 — Core pipeline: single task × each agent.

Validates that the SDK.run() pipeline completes end-to-end for every
registered agent driven by gemini-3.1-flash-lite-preview on Daytona.

Uses the hello-world-task (trivial, fast) to isolate agent/infra issues
from task complexity.

Run::

    GEMINI_API_KEY=... DAYTONA_API_KEY=... \\
      pytest -m integration tests/integration/test_core_pipeline.py -v

Guards: ENG-6 integration test plan (issue #253).
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import (
    ALL_AGENTS,
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL,
    HELLO_TASK,
    has_creds_for_agent,
    model_for_agent,
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ALL_AGENTS)
async def test_hello_world_per_agent(
    integration_prereqs: None,
    agent: str,
    jobs_dir,
) -> None:
    """Each agent solves hello-world-task and produces a valid result."""
    if not has_creds_for_agent(agent):
        pytest.skip(f"No credentials for {agent}")

    from benchflow import SDK

    result = await SDK().run(
        task_path=HELLO_TASK,
        agent=agent,
        model=model_for_agent(agent),
        jobs_dir=jobs_dir,
        environment=DEFAULT_ENVIRONMENT,
    )

    # Core assertions — pipeline ran to completion
    assert result.error is None, f"Agent error: {result.error}"
    assert result.verifier_error is None, f"Verifier error: {result.verifier_error}"
    assert result.rewards is not None, "No rewards produced"
    assert result.rewards.get("reward") == 1.0, (
        f"Expected reward=1.0, got {result.rewards}"
    )

    # Trajectory was captured
    assert result.n_tool_calls > 0, "No tool calls recorded"
    assert result.trajectory_source in ("acp", "partial_acp"), (
        f"Unexpected trajectory source: {result.trajectory_source}"
    )

    # Result files exist on disk
    matches = list(
        jobs_dir.glob(f"*/{result.trial_name}/trajectory/acp_trajectory.jsonl")
    )
    assert len(matches) == 1, f"Expected 1 trajectory file, found {len(matches)}"
    assert matches[0].stat().st_size > 0

    result_files = list(jobs_dir.glob(f"*/{result.trial_name}/result.json"))
    assert len(result_files) == 1, f"Expected 1 result.json, found {len(result_files)}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hello_world_result_schema(
    integration_prereqs: None,
    jobs_dir,
) -> None:
    """Validate result.json schema from a gemini run."""
    import json

    from benchflow import SDK

    result = await SDK().run(
        task_path=HELLO_TASK,
        agent="gemini",
        model=DEFAULT_MODEL,
        jobs_dir=jobs_dir,
        environment=DEFAULT_ENVIRONMENT,
    )

    # Load result.json
    result_files = list(jobs_dir.glob(f"*/{result.trial_name}/result.json"))
    assert len(result_files) == 1
    data = json.loads(result_files[0].read_text())

    # Required top-level fields
    required_fields = {
        "task_name",
        "trial_name",
        "agent_name",
        "model",
        "rewards",
        "n_tool_calls",
        "n_prompts",
        "started_at",
        "finished_at",
    }
    missing = required_fields - set(data.keys())
    assert not missing, f"result.json missing fields: {missing}"

    # Type checks
    assert isinstance(data["rewards"], dict)
    assert isinstance(data["n_tool_calls"], int)
    assert data["n_tool_calls"] >= 0
    assert data["task_name"] == "hello-world-task"

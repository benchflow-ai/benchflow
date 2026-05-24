"""Regression tests for self-gen skill export failure propagation.

Guards #389: a configured-but-failed skill export must not be observable as
a successful rollout that "honestly evolved no skills". The cleanup path
used to swallow export exceptions into a warning, leaving ``_evolved_skills``
as ``None`` and ``_error`` empty — indistinguishable from a clean run that
the agent didn't bother to populate. That was a measurement-integrity bug for
the continual-learning Memory-space signal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig


def _bare_rollout(tmp_path: Path, export_target: Path | None) -> Rollout:
    """Construct a Rollout skeleton with just enough state to drive cleanup()."""
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        export_generated_skills_to=export_target,
    )
    rollout._trajectory = []
    rollout._acp_client = None
    rollout._agent_launch = ""
    rollout._env = SimpleNamespace()
    rollout._environment = None
    rollout._usage_runtime = None
    rollout._provider_runtime = None
    rollout._rollout_dir = tmp_path
    rollout._evolved_skills = None
    rollout._error = None
    # Stub the bits cleanup() touches but we don't care about for this test.
    rollout.disconnect = AsyncMock()
    rollout._capture_partial_acp_trajectory = lambda: None
    return rollout


@pytest.mark.asyncio
async def test_cleanup_surfaces_export_failure_on_rollout_error(tmp_path):
    """Export failures must set _error, not silently log-and-continue."""
    rollout = _bare_rollout(tmp_path, export_target=tmp_path / "exports")
    rollout._export_generated_skills = AsyncMock(
        side_effect=RuntimeError("download failed")
    )

    await rollout.cleanup()

    # The export failure is now observable on the rollout — success is gone.
    assert rollout._error is not None
    assert "Skill export failed" in rollout._error
    assert "download failed" in rollout._error
    # And we did NOT collapse into a "honestly empty" skill update: the
    # evolved_skills field stays None, distinct from {} (no-op success).
    assert rollout._evolved_skills is None


@pytest.mark.asyncio
async def test_cleanup_export_failure_does_not_clobber_agent_error(tmp_path):
    """Agent errors take priority — export failure must not overwrite them."""
    rollout = _bare_rollout(tmp_path, export_target=tmp_path / "exports")
    rollout._error = "Agent timed out after 600s"
    rollout._export_generated_skills = AsyncMock(
        side_effect=RuntimeError("download failed")
    )

    await rollout.cleanup()

    # Pre-existing agent error wins: it ran first, it's the more useful signal.
    assert rollout._error == "Agent timed out after 600s"
    assert rollout._evolved_skills is None


@pytest.mark.asyncio
async def test_cleanup_no_export_configured_leaves_error_empty(tmp_path):
    """When export is not configured, cleanup must not invent an error."""
    rollout = _bare_rollout(tmp_path, export_target=None)
    # Sentinel: if _export_generated_skills is called at all, fail loudly.
    rollout._export_generated_skills = AsyncMock(
        side_effect=AssertionError("must not be called when export is unconfigured")
    )

    await rollout.cleanup()

    assert rollout._error is None
    assert rollout._evolved_skills is None


@pytest.mark.asyncio
async def test_cleanup_successful_export_leaves_error_empty(tmp_path):
    """A successful export must not be confused with a failure."""
    rollout = _bare_rollout(tmp_path, export_target=tmp_path / "exports")

    async def fake_export():
        rollout._evolved_skills = {"skill-a": "# Skill A\n"}

    rollout._export_generated_skills = fake_export

    await rollout.cleanup()

    assert rollout._error is None
    assert rollout._evolved_skills == {"skill-a": "# Skill A\n"}


@pytest.mark.asyncio
async def test_build_result_after_export_failure_is_not_success(tmp_path):
    """The visible RolloutResult must reflect the export failure as not-success.

    The whole point of #389: empty-but-successful must be impossible to
    confuse with failed-and-produced-empty in the downstream result.
    """
    rollout = _bare_rollout(tmp_path, export_target=tmp_path / "exports")
    rollout._export_generated_skills = AsyncMock(
        side_effect=RuntimeError("sandbox lost connection")
    )
    # Minimal extra state _build_result reads via _build_rollout_result.
    rollout._rollout_name = "rollout-1"
    rollout._n_tool_calls = 0
    rollout._resolved_prompts = []
    rollout._verifier_error = None
    rollout._partial_trajectory = False
    rollout._trajectory_source = None
    rollout._rewards = None
    from datetime import datetime

    rollout._started_at = datetime.now()
    rollout._timing = {}
    rollout._agent_name = ""
    rollout._idle_timeout_info = None
    rollout._sandbox_startup_info = None
    rollout._transport_error_info = None
    rollout._verifier_timeout_info = None
    rollout._usage_metrics = {
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }

    await rollout.cleanup()
    result = rollout._build_result()

    assert result.success is False
    assert result.error is not None
    assert "Skill export failed" in result.error
    assert result.evolved_skills is None

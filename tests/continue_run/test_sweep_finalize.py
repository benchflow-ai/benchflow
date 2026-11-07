"""Guards FF-sw-evaluation continuation fixes for 363d815b."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.continue_run.orchestrator import (
    ContinuedUsageSummary,
    _result_after_sandbox_teardown,
)


@pytest.mark.asyncio
async def test_manual_continuation_finalizes_review_after_usage(tmp_path):
    usage = ContinuedUsageSummary(10, 20, 0, 0, 30, 10, 20, "provider_response")
    trial = SimpleNamespace(
        _rollout_dir=tmp_path, _phase="cleaned", _usage_metrics={}, result=None
    )

    async def finalize():
        assert trial._usage_metrics["total_tokens"] == 30
        return SimpleNamespace(rewards={"reward": 0.75}, error=None)

    trial.finalize = AsyncMock(side_effect=finalize)
    # Both the old synchronous helper and the new asynchronous helper are
    # exercised, so the base fails on the missing verdict, never on an await.
    import inspect

    outcome = _result_after_sandbox_teardown(trial, usage=usage)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    assert outcome is not None
    assert outcome.rewards == {"reward": 0.75}
    trial.finalize.assert_awaited_once()

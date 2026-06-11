"""Tests for the deprecated ``SDK.run(trial_name=...)`` alias.

v0.6 renamed ``SDK.run(trial_name=...)`` to ``rollout_name`` with no alias,
a hard break for downstream callers (clawsbench). BF-2 restores a
backward-compatible deprecation alias: ``trial_name`` maps to ``rollout_name``
and emits a ``DeprecationWarning``; passing both raises ``TypeError``.
"""

from unittest.mock import AsyncMock

import pytest


def _patch_create(monkeypatch, seen):
    """Stub Rollout.create so run() never touches a real sandbox/cloud."""
    from benchflow.models import RunResult

    async def fake_create(config):
        seen["config"] = config
        trial = AsyncMock()
        trial.run = AsyncMock(
            return_value=RunResult(task_name="task-1", rewards={"reward": 1.0})
        )
        return trial

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_create)


@pytest.mark.asyncio
async def test_trial_name_maps_to_rollout_name_with_deprecation_warning(
    monkeypatch, tmp_path
):
    """Legacy 'trial_name' routes to 'rollout_name' and warns once."""
    from benchflow.sdk import SDK

    seen: dict = {}
    _patch_create(monkeypatch, seen)

    with pytest.warns(DeprecationWarning, match="trial_name"):
        result = await SDK().run(task_path=tmp_path, trial_name="legacy-trial")

    assert result.rewards == {"reward": 1.0}
    assert seen["config"].rollout_name == "legacy-trial"


@pytest.mark.asyncio
async def test_rollout_name_alone_does_not_warn(monkeypatch, tmp_path):
    """The new 'rollout_name' kwarg works without any deprecation warning."""
    import warnings

    from benchflow.sdk import SDK

    seen: dict = {}
    _patch_create(monkeypatch, seen)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        await SDK().run(task_path=tmp_path, rollout_name="new-rollout")

    assert seen["config"].rollout_name == "new-rollout"


@pytest.mark.asyncio
async def test_passing_both_raises_type_error(monkeypatch, tmp_path):
    """Supplying both names is ambiguous and must raise TypeError."""
    from benchflow.sdk import SDK

    seen: dict = {}
    _patch_create(monkeypatch, seen)

    with pytest.raises(TypeError, match="only one of 'rollout_name' or 'trial_name'"):
        await SDK().run(
            task_path=tmp_path,
            rollout_name="new-rollout",
            trial_name="legacy-trial",
        )

    # run() must short-circuit before constructing/dispatching the rollout.
    assert "config" not in seen

"""Each completed fork records the standard error of its value V.

V is the mean of a handful of children's rewards (typically 2-4), so V = 0.5 from two children and V = 0.5 from forty mean very
different things. ``value_stderr`` is the sample standard deviation of the
children's rewards over sqrt(n) (null below two scored children); the
branch-view document carries it.
"""

from __future__ import annotations

import json

import jsonschema

from benchflow.branch_view import BRANCH_VIEW_SCHEMA, load_branch_view
from tests.test_branch_restore_parent import _fork, _rollout


async def test_fork_records_the_standard_error_of_v(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    rewards = iter([1.0, 0.0, 1.0, 1.0])

    async def child(_node):
        return next(rewards)

    value = await rollout.branch(4, child, snapshot_layers={"sandbox"})
    fork = _fork(rollout)
    assert value == 0.75
    assert fork["value_stderr"] == 0.25
    (rollout._rollout_dir / "result.json").write_text(json.dumps({"task_name": "t"}))
    view = load_branch_view(rollout._rollout_dir)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["forks"][0]["value_stderr"] == 0.25


async def test_equal_rewards_have_zero_error(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    assert _fork(rollout)["value_stderr"] == 0.0

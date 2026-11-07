"""``branch(child_requests=...)`` records what each child was asked to do.

Every child's ``intervention`` in tree.json read
``requested: null, execution: "unspecified"`` even when the caller's runner
sent each child its own prompt, so the viewer's Lineage tab could not say how
the arms differed. A caller can now pass one short description per child; it
is recorded as ``intervention.requested`` with ``execution: "runner"`` (the
caller's runner applied it; the engine does not).

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json

import pytest

from tests.test_branch_restore_parent import _fork, _rollout


async def test_child_requests_are_recorded_per_child(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(
        2,
        child,
        snapshot_layers={"sandbox"},
        child_labels=["baseline", "hint"],
        child_requests=["remaining prompts (1)", None],
    )
    interventions = [c["intervention"] for c in _fork(rollout)["children"]]
    assert interventions == [
        {
            "label": "baseline",
            "requested": "remaining prompts (1)",
            "execution": "runner",
            "evidence": None,
        },
        {
            "label": "hint",
            "requested": None,
            "execution": "unspecified",
            "evidence": None,
        },
    ]


@pytest.mark.parametrize("requests", [["only one"], ["ok", "x" * 201], ["ok", 3]])
async def test_bad_child_requests_are_refused_before_anything_runs(tmp_path, requests):
    rollout, sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    with pytest.raises(ValueError, match="child_requests"):
        await rollout.branch(
            2, child, snapshot_layers={"sandbox"}, child_requests=requests
        )
    assert sandbox.snapshots == []
    assert (
        not (rollout._rollout_dir / "tree.json").exists()
        or not json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"]
    )

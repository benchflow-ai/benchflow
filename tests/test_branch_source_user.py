"""EN-721 (in EN-715): a branch child runs as its source trial's sandbox user.

A checkpoint holds its source's processes and files: a root agent's has no
sandbox user, and its children used to quiesce as ``agent`` ("Invalid
sandbox UID"). ``bench branch --from-checkpoint`` now takes the source's
user from the trial's config.json (else its job's evaluation.json), and
offers no way to choose another.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.branch_run import (
    BranchPlan,
    CheckpointSource,
    ChildSpec,
    run_branch_trial,
)


def _task(root: Path) -> Path:
    task = root / "t"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n[metadata]\nauthor_name = "x"\n')
    (task / "instruction.md").write_text("Do it.\n")
    (task / "environment" / "Dockerfile").write_text("FROM busybox\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    return task


def _plan(trial: Path, jobs: Path) -> BranchPlan:
    return BranchPlan(
        task_paths=[],
        agent="oracle",
        model=None,
        sandbox="docker",
        children=[ChildSpec(label="a"), ChildSpec(label="b")],
        checkpoint_after=0,
        jobs_dir=jobs,
        source=CheckpointSource(
            trial_dir=trial, fork_id="f", provider="docker", ref="img", task_name="t"
        ),
    )


class _Captured(Exception):
    pass


def _child_config(plan: BranchPlan, task: Path):
    seen = {}

    def factory(config):
        seen["config"] = config
        raise _Captured

    with pytest.raises(_Captured):
        # The factory stops the trial before any sandbox starts.
        asyncio.run(run_branch_trial(plan, task, rollout_factory=factory))
    return seen["config"]


@pytest.mark.parametrize(
    ("config_json", "evaluation_json", "user"),
    [
        # The source trial's own record: a root agent's.
        ({"sandbox_user": None}, None, None),
        # A branch of a branch: its config.json, no evaluation.json in its job.
        ({"sandbox_user": "agent"}, None, "agent"),
        # A source whose config.json does not say: its job's evaluation.json.
        ({}, {"config": {"sandbox_user": None}}, None),
    ],
)
def test_a_branch_child_runs_as_its_source_s_user(
    tmp_path, config_json, evaluation_json, user
):
    trial = tmp_path / "job" / "t__1"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps(config_json))
    if evaluation_json is not None:
        (trial.parent / "evaluation.json").write_text(json.dumps(evaluation_json))
    config = _child_config(_plan(trial, tmp_path / "jobs"), _task(tmp_path))
    assert config.sandbox_user == user


def test_a_branch_cannot_choose_another_user():
    """No plan field and no CLI flag sets the children's user."""
    assert "sandbox_user" not in {f.name for f in dataclasses.fields(BranchPlan)}
    help_text = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "branch", "--help"],
        env={
            **os.environ,
            "NO_COLOR": "1",
            "COLUMNS": "200",
            "HARBOR_TELEMETRY": "off",
        },
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    assert "--from-checkpoint" in help_text
    assert "--sandbox-user" not in help_text

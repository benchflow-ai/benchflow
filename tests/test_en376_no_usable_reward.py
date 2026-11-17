"""A verifier that exits without writing a reward is an error (EN-376, 0.9.3).

X1 (2026-11-13) on 0.9.2: a task whose test.sh exits 23 and writes no reward
file scored reward 0 with errors=0, indistinguishable from a wrong answer.
rc3 records "verifier wrote no usable reward: ..." instead. 0.9.3 does the
same on its own code: the trial is unscored with a verifier error (category
verifier_failure, never retried), and it stays a failure in the
error-excluding score, since the agent can reach the reward files.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.scoring import (
    NO_USABLE_REWARD,
    classify_verifier_error,
    score_summary_fields,
)
from benchflow.task.verifier_errors import RewardFileNotFoundError

X1_MESSAGE = (
    "verifier exited with rc=23; no reward file found at verifier/reward.txt "
    "or verifier/reward.json"
)


async def _verify_raising(tmp_path, error: Exception):
    from benchflow.sdk import SDK

    task = MagicMock()
    task.config.verifier.timeout_sec = 5
    task.config.verifier.env = None
    paths = MagicMock()
    paths.verifier_dir = tmp_path / "verifier"
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    verifier = MagicMock()

    async def verify():
        await asyncio.sleep(0)
        raise error

    verifier.verify = verify
    with (
        patch("benchflow.task.Verifier", return_value=verifier),
        patch(
            "benchflow.sandbox.lockdown.harden_before_verify",
            new_callable=AsyncMock,
        ),
    ):
        return await SDK()._verify(env, task, paths, {})


@pytest.mark.asyncio
async def test_a_verifier_exit_23_without_a_reward_is_a_verifier_error(
    tmp_path, caplog
):
    rewards, verifier_error, _ = await _verify_raising(
        tmp_path, RewardFileNotFoundError(X1_MESSAGE)
    )
    assert rewards is None
    assert verifier_error == NO_USABLE_REWARD + X1_MESSAGE
    assert "Verifier scored 0" not in caplog.text
    assert classify_verifier_error(verifier_error) == "verifier_failure"


def test_the_summary_counts_it_as_an_error_and_a_failure():
    """Never reward 0 with errors=0: the trial is verifier-errored, and the
    error-excluding score keeps it in its denominator as a failure."""
    rows = [
        {
            "task_name": "verifier-failure",
            "rewards": None,
            "error": None,
            "verifier_error": NO_USABLE_REWARD + X1_MESSAGE,
        },
        {"task_name": "solved", "rewards": {"reward": 1.0}, "error": None},
    ]
    summary = score_summary_fields(rows)
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["verifier_errored"] == 1
    assert summary["unusable_rewards"] == 1
    assert summary["guard_refusals"] == 0
    assert summary["verifier_error_categories"] == {"verifier_failure": 1}
    assert summary["score_excl_errors_ratio"] == 0.5
    assert summary["mean_reward_scored_only"] == 0.5


def test_the_scored_only_score_line_names_what_it_counts(tmp_path):
    """The table's error-excluding score keeps the unusable reward as a
    failure, so its n says so: never "1 of 1 scored trials" beside 50%."""
    import re

    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from tests.test_python_sdk_load_job import _trial

    job = tmp_path / "job"
    _trial(job, "solved", reward=1.0)
    _trial(
        job,
        "verifier-failure",
        reward=None,
        verifier_error=NO_USABLE_REWARD + X1_MESSAGE,
    )
    out = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    text = " ".join(re.sub("[─-╿]", " ", out.output).split())
    assert (
        "Score, scored only 50.0% (1 of 2 trials: 1 scored, 1 with no usable "
        "verdict counted as failed)"
    ) in text

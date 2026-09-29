"""``bench eval run --retry-from-checkpoint``: retry a failed trial from its
last checkpoint instead of from scratch.

When a trial fails (scored below a
pass) or times out and it kept automatic checkpoints (``--checkpoints``), one
retry child is forked from the last kept checkpoint: its own sandbox created
from that snapshot, keeping the snapshot's *original* pre-agent verifier
baseline (so anything the failed attempt tampered with before the checkpoint
is still undone at verification), sent the remaining prompts or
``--retry-prompt``, and verified by the task's own verifier. It is recorded
as a ``retry`` fork in ``tree.json``. The trial's reward is never replaced:
result.json keeps the original ``rewards`` and gains a separate ``retry``
block with the retry's reward, and the job summary reports both.

Unit tests against a scripted rollout; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.checkpoint_retry import (
    parse_retry_policy,
    retry_reason,
    run_checkpoint_retry,
)
from benchflow.models import RolloutResult
from benchflow.trajectories.tree import Step
from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _tree


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


@pytest.mark.parametrize(
    ("spec", "failure", "timeout"),
    [
        ("on-failure", True, False),
        ("on-timeout", False, True),
        ("on-failure,on-timeout", True, True),
    ],
)
def test_policies_parse(spec, failure, timeout):
    policy = parse_retry_policy(spec, prompt=None)
    assert (policy.on_failure, policy.on_timeout) == (failure, timeout)


@pytest.mark.parametrize("spec", ["", "always", "on-failure,sometimes"])
def test_bad_policies_are_refused(spec):
    with pytest.raises(ValueError, match="--retry-from-checkpoint"):
        parse_retry_policy(spec, prompt=None)


def _result(reward=None, error=None, category=None):
    return RolloutResult(
        task_name="t",
        rewards=None if reward is None else {"reward": reward},
        error=error,
        error_category=category,
    )


def test_when_to_retry():
    both = parse_retry_policy("on-failure,on-timeout", prompt=None)
    failure_only = parse_retry_policy("on-failure", prompt=None)
    assert retry_reason(_result(1.0), both) is None
    assert retry_reason(_result(0.0), both) == "failure"
    assert retry_reason(_result(0.0, "timed out", "timeout"), both) == "timeout"
    assert retry_reason(_result(0.0, "timed out", "timeout"), failure_only) is None
    assert retry_reason(_result(None, "idle", "idle_timeout"), both) == "timeout"
    # Infrastructure errors are the existing retry loop's job, not this one's.
    assert retry_reason(_result(None, "sandbox died", "sandbox_setup"), both) is None


async def _finished_trial(tmp_path, *, kept=True):
    """A root rollout that ran two prompts, kept a checkpoint after prompt 1,
    and failed; its sandbox is gone, as after Rollout.run()."""
    root = await _root(tmp_path)
    root._resolved_prompts = ["Draft first.", "Do it."]
    first = root._tree.advance(root._cursor, Step(id="step-0-user_message", data={}))
    root._cursor = first
    image = await root._env.snapshot()  # the world is ["draft"] here
    root._cursor = root._tree.advance(first, Step(id="step-1-agent_message", data={}))
    root.world = ["draft", "wrong"]
    rows = [
        {
            "id": "prompt:1",
            "after_prompt": 1,
            "node_id": first.id,
            "provider": image.provider,
            "ref": image.ref,
            "status": "kept" if kept else "deleted",
        }
    ]
    (root._rollout_dir / "checkpoints.json").write_text(
        json.dumps({"kind": "benchflow-checkpoints", "checkpoints": rows})
    )
    (root._rollout_dir / "result.json").write_text(
        json.dumps({"task_name": "t", "rewards": {"reward": 0.0}})
    )
    return root, first


async def test_a_failed_trial_is_retried_from_its_last_checkpoint(tmp_path):
    root, first = await _finished_trial(tmp_path)
    result = _result(0.0)
    policy = parse_retry_policy("on-failure", prompt=None)
    await run_checkpoint_retry(root, result, policy)

    [retry_child] = IsoRollout.all[1:]
    # Its own sandbox, from the checkpoint: the failed attempt's later work
    # ("wrong") is not there; it is sent the prompts after the checkpoint.
    assert retry_child.events[3] == "execute:['Do it.']:['draft']"
    assert retry_child._from_branch_snapshot
    fork = _tree(root)["forks"][0]
    assert fork["kind"] == "retry"
    assert fork["reason"] == "failure"
    assert fork["checkpoint"] == "prompt:1"
    assert fork["parent_node"] == first.id
    [child] = fork["children"]
    assert child["intervention"]["label"] == "retry"
    assert (child["reward"], child["reward_source"]) == (1.0, "verifier")

    # Both are reported; the trial's own reward is untouched.
    saved = json.loads((root._rollout_dir / "result.json").read_text())
    assert saved["rewards"] == {"reward": 0.0}
    assert saved["retry"] == {
        "status": "completed",
        "reason": "failure",
        "checkpoint": "prompt:1",
        "fork_id": fork["id"],
        "reward": 1.0,
        "original_reward": 0.0,
        "path": child["artifacts"]["path"],
        # tests/test_checkpoint_retry_diagnostics.py
        "tool_calls": 0,
        "no_work": True,
    }
    assert result.rewards == {"reward": 0.0}
    assert result.retry == saved["retry"]


async def test_the_retry_prompt_replaces_the_remaining_prompts(tmp_path):
    root, _ = await _finished_trial(tmp_path)
    policy = parse_retry_policy("on-failure", prompt="Try again: Do it.")
    await run_checkpoint_retry(root, _result(0.0), policy)
    [retry_child] = IsoRollout.all[1:]
    assert retry_child.events[3] == "execute:['Try again: Do it.']:['draft']"


async def test_without_a_kept_checkpoint_nothing_is_forked(tmp_path):
    root, _ = await _finished_trial(tmp_path, kept=False)
    result = _result(0.0)
    await run_checkpoint_retry(
        root, result, parse_retry_policy("on-failure", prompt=None)
    )
    assert IsoRollout.all[1:] == []
    assert result.retry == {"status": "no_checkpoint", "reason": "failure"}
    saved = json.loads((root._rollout_dir / "result.json").read_text())
    assert saved["retry"] == result.retry


async def test_a_passing_trial_is_not_retried(tmp_path):
    root, _ = await _finished_trial(tmp_path)
    result = _result(1.0)
    await run_checkpoint_retry(
        root, result, parse_retry_policy("on-failure", prompt=None)
    )
    assert IsoRollout.all[1:] == []
    assert result.retry is None


def test_result_round_trips_the_retry_block(tmp_path):
    block = {"status": "completed", "reward": 1.0, "original_reward": 0.0}
    loaded = RolloutResult.from_dict({"task_name": "t", "retry": block})
    assert loaded.retry == block
    assert SimpleNamespace(retry=None).retry is None


# ── bench eval run plumbing ──────────────────────────────────────────


def test_evaluation_config_needs_checkpoints_for_retries():
    from benchflow.evaluation import EvaluationConfig

    config = EvaluationConfig(
        checkpoints="every-prompt",
        retry_from_checkpoint="on-failure",
        retry_prompt="Try again.",
    )
    policy = config.retry_policy()
    assert (policy.on_failure, policy.prompt) == (True, "Try again.")
    assert EvaluationConfig().retry_policy() is None
    with pytest.raises(ValueError, match="--checkpoints"):
        EvaluationConfig(retry_from_checkpoint="on-failure")
    with pytest.raises(ValueError, match="--retry-from-checkpoint"):
        EvaluationConfig(checkpoints="every-prompt", retry_from_checkpoint="bad")


def test_worker_payload_round_trips_the_retry_policy():
    from benchflow.eval_sharding import _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig

    config = EvaluationConfig(
        checkpoints="prompt:1", retry_from_checkpoint="on-timeout", retry_prompt="p"
    )
    payload = _config_payload(
        config, shard=SimpleNamespace(concurrency=1, task_names=["t"])
    )
    assert _evaluation_config(payload).retry_policy() == config.retry_policy()


def test_eval_run_refuses_retry_without_checkpoints(tmp_path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tmp_path),
            "--agent",
            "oracle",
            "--retry-from-checkpoint",
            "on-failure",
        ],
    )
    assert result.exit_code != 0
    assert "--checkpoints" in result.output


async def test_run_single_task_retries_a_failed_trial(tmp_path, monkeypatch):
    from benchflow import evaluation as evaluation_mod
    from benchflow.evaluation import Evaluation, EvaluationConfig

    calls = []
    finished = _result(0.0)

    class FakeRollout:
        @classmethod
        async def create(cls, config):
            return cls()

        async def run(self):
            return finished

    async def fake_retry(rollout, result, policy):
        calls.append((type(rollout).__name__, result, policy.on_failure))

    import benchflow.rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "Rollout", FakeRollout)
    monkeypatch.setattr(evaluation_mod, "run_checkpoint_retry", fake_retry)
    task = tmp_path / "tasks" / "t"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Do it.")
    config = EvaluationConfig(
        agent="claude-agent-acp",
        checkpoints="every-prompt",
        retry_from_checkpoint="on-failure",
    )
    evaluation = Evaluation(
        tasks_dir=tmp_path / "tasks", jobs_dir=tmp_path / "jobs", config=config
    )
    result = await evaluation._run_single_task(task, config)
    assert result is finished
    assert calls == [("FakeRollout", finished, True)]


def test_summary_reports_retries_next_to_the_original_score():
    from benchflow.checkpoint_retry import retry_summary

    results = [
        SimpleNamespace(retry=None),
        SimpleNamespace(
            retry={"status": "completed", "reward": 1.0, "original_reward": 0.0}
        ),
        SimpleNamespace(
            retry={"status": "completed", "reward": 0.0, "original_reward": 0.0}
        ),
        SimpleNamespace(retry={"status": "no_checkpoint", "reason": "failure"}),
    ]
    assert retry_summary(results) == {
        "attempted": 2,
        "passed": 1,
        "no_checkpoint": 1,
        "failed_to_run": 0,
        "no_work": 0,
    }
    assert retry_summary([SimpleNamespace(retry=None)]) is None
    # The job summary passes result.json documents.
    assert (
        retry_summary([{"retry": {"status": "completed", "reward": 1.0}}])["passed"]
        == 1
    )


async def test_job_summary_counts_retries_of_fresh_trials(tmp_path):
    """A fresh trial's retry used to be in its result.json but not in summary.json, because fresh results
    reach the summary through rollout_result_payload(), which has no retry."""
    from unittest.mock import AsyncMock

    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-0").mkdir(parents=True)
    (tasks_dir / "task-0" / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    config = EvaluationConfig(
        retry=RetryConfig(max_retries=0),
        checkpoints="prompt:1",
        retry_from_checkpoint="on-failure",
    )
    job = Evaluation(
        tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=config, job_name="j"
    )
    failed = RolloutResult(task_name="task-0", rewards={"reward": 0.0})
    failed.retry = {"status": "completed", "reward": 1.0, "original_reward": 0.0}
    job._run_single_task = AsyncMock(return_value=failed)
    await job.run()
    summary = json.loads((tmp_path / "jobs" / "j" / "summary.json").read_text())
    assert summary["checkpoint_retries"] == {
        "attempted": 1,
        "passed": 1,
        "no_checkpoint": 0,
        "failed_to_run": 0,
        "no_work": 0,
    }
    assert summary["passed"] == 0  # the trial's own score is unchanged

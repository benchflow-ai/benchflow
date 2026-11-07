"""Guards multi-step tasks (task/steps.py) against the refusal they
replace: every task.toml with ``[[steps]]`` stopped at the launch gate with
"multi-step tasks are not supported (one prompt per task)", and docs rounds 2
and 3 scored it against BenchFlow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.contracts.user import RoundResult
from benchflow.rollout import RolloutConfig
from benchflow.task import Task
from benchflow.task.steps import (
    StepsUser,
    combine,
    describe,
    meets_min_reward,
    step_task,
)

TOML = """\
version = "1.0"
multi_step_reward_strategy = "{strategy}"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 60

[environment]
docker_image = "python:3.12-slim"

[[steps]]
name = "write"
min_reward = 1.0

[steps.agent]
timeout_sec = 120

[[steps]]
name = "append"

[steps.verifier]
timeout_sec = 30
"""


def _task(root: Path, strategy: str = "final", own_tests: bool = True) -> Path:
    task = root / "two-steps"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n"
    )
    (task / "task.toml").write_text(TOML.format(strategy=strategy))
    for name in ("write", "append"):
        step = task / "steps" / name
        step.mkdir(parents=True)
        (step / "instruction.md").write_text(f"Do the {name} step.\n")
        if own_tests or name == "write":
            (step / "tests").mkdir()
            (step / "tests" / "test.sh").write_text(f"#!/bin/bash\n# {name}\n")
        (step / "solution").mkdir()
        (step / "solution" / "solve.sh").write_text(f"#!/bin/bash\n# {name}\n")
    return task


def test_a_step_sees_its_own_files_and_the_tasks_where_it_has_none(tmp_path):
    task = Task(_task(tmp_path, own_tests=False))
    write, append = step_task(task, 0), step_task(task, 1)
    assert write.instruction == "Do the write step."
    assert write.paths.tests_dir == task.task_dir / "steps" / "write" / "tests"
    assert append.paths.tests_dir == task.paths.tests_dir  # falls back
    assert append.paths.solve_path == task.task_dir / "steps/append/solution/solve.sh"
    # A step's settings win only where the step sets them.
    assert write.config.agent.timeout_sec == 120
    assert write.config.verifier.timeout_sec == 60
    assert append.config.agent.timeout_sec == 300
    assert append.config.verifier.timeout_sec == 30
    assert task.config.agent.timeout_sec == 300  # the task is unchanged


@pytest.mark.parametrize(
    ("threshold", "rewards", "met"),
    [
        (None, None, True),
        (1.0, {"reward": 1.0}, True),
        (1.0, {"reward": 0.99}, False),
        (1.0, None, False),
        ({"a": 0.5, "b": 1.0}, {"reward": 0, "a": 0.5, "b": 1}, True),
        ({"a": 0.5}, {"reward": 1}, False),
        ({"a": 0.5}, {"a": True}, False),
    ],
)
def test_min_reward(threshold, rewards, met):
    class Step:
        min_reward = threshold

    assert meets_min_reward(Step, rewards) is met


@pytest.mark.asyncio
async def test_the_steps_user_gives_each_step_and_stops_at_a_missed_min_reward(
    tmp_path,
):
    task = Task(_task(tmp_path))
    user = StepsUser(task)
    assert await user.run(0, "", None) == "Do the write step."
    assert await user.run(1, "", RoundResult(round=0, rewards={"reward": 1.0})) == (
        "Do the append step."
    )
    assert await user.run(2, "", RoundResult(round=1, rewards={"reward": 0.0})) is None
    assert user.soft_rewards == [{"reward": 1.0}, {"reward": 0.0}]
    assert user.stopped_after is None

    stopped = StepsUser(task)
    await stopped.run(0, "", None)
    assert (
        await stopped.run(1, "", RoundResult(round=0, rewards={"reward": 0.5})) is None
    )
    assert stopped.stopped_after == 0


def test_the_reward_strategies(tmp_path):
    steps = Task(_task(tmp_path)).config.steps
    both = [{"reward": 1.0}, {"reward": 0.3}]
    final, record = combine("final", steps, both, {"reward": 0.0})
    assert final == {"reward": 0.0}
    assert [r["source"] for r in record["steps"]] == ["soft", "final"]
    mean, record = combine("mean", steps, both, {"reward": 0.0, "extra": "x"})
    assert mean == {"reward": 0.5, "extra": "x"}
    # The first step stopped the task: the second was never reached.
    early, record = combine("final", steps, [{"reward": 0.5}], {"reward": 0.5})
    assert early == {"reward": 0.0}
    assert [r["source"] for r in record["steps"]] == ["final", "not-run"]
    half, _ = combine("mean", steps, [{"reward": 1.0}], {"reward": 1.0})
    assert half == {"reward": 0.5}
    # No hardened score: no reward at all, never a soft one.
    assert combine("mean", steps, both, None)[0] is None


@pytest.mark.asyncio
async def test_each_step_is_logged_by_name_and_a_stop_says_why(tmp_path, caplog):
    task = Task(_task(tmp_path))
    caplog.set_level("INFO", logger="benchflow.task.steps")
    user = StepsUser(task)
    await user.run(0, "", None)
    await user.run(1, "", RoundResult(round=0, rewards={"reward": 1.0}, n_tool_calls=2))
    await user.run(2, "", RoundResult(round=1, rewards={"reward": 0.0}, n_tool_calls=1))
    stopped = StepsUser(task)
    await stopped.run(0, "", None)
    await stopped.run(1, "", RoundResult(round=0, rewards={"reward": 0.5}))
    # The oracle's scripted steps have no tool calls to count.
    StepsUser(task).record(0, {"reward": 1.0})
    assert caplog.messages == [
        "Step 1/2 write: reward 1.00 after 2 tool call(s)",
        "Step 2/2 append: reward 0.00 after 1 tool call(s)",
        "Step 1/2 write: reward 0.50 after 0 tool call(s), below its min_reward 1:"
        " the task stops here and the 1 later step counts 0",
        "Step 1/2 write: reward 1.00",
    ]
    # The verifier's rewards line and the user loop name the step by its view.
    assert step_task(task, 1).step_label == "step 2/2 append"


def test_the_steps_line_says_how_the_task_reward_was_made(tmp_path):
    steps = Task(_task(tmp_path)).config.steps
    _, record = combine(
        "final", steps, [{"reward": 1.0}, {"reward": 0.3}], {"reward": 0.0}
    )
    assert describe(record) == (
        "Steps: write 1.00 (soft check), append 0.00 (final verify; its soft check "
        "gave 0.30); "
        "task reward 0.00, the last step's (append)"
    )
    _, record = combine("final", steps, [{"reward": 0.0}], {"reward": 0.0})
    assert describe(record) == (
        "Steps: write 0.00 (final verify), append not run; "
        "task reward 0.00: the last step (append) did not run"
    )
    _, record = combine("mean", steps, [{"reward": 1.0}], {"reward": 1.0})
    assert describe(record) == (
        "Steps: write 1.00 (final verify), append not run; "
        "task reward 0.50, the mean of the 2 steps (a step not run counts 0)"
    )
    _, record = combine("mean", steps, [{"reward": 1.0}], None)
    assert describe(record) == (
        "Steps: write no reward (final verify; its soft check gave 1.00), append not run; "
        "no task reward: the final verify gave none"
    )


def test_a_failed_multi_step_trial_is_explained_from_steps_json(tmp_path):
    """Round 4 of the release review: a mean task's failure line pointed only
    at the final verify, which scored 0, while the task scored 0.5."""
    import json

    from benchflow.cli._shared import _failure_reason
    from benchflow.evaluation import TaskFailure

    steps = Task(_task(tmp_path)).config.steps
    _, record = combine(
        "mean", steps, [{"reward": 1.0}, {"reward": 0.0}], {"reward": 0.0}
    )
    trial = tmp_path / "job" / "ms__abc"
    trial.mkdir(parents=True)
    (trial / "steps.json").write_text(json.dumps(record))
    failure = TaskFailure(
        task_name="ms",
        rewards={"reward": 0.5},
        verifier_error=None,
        rollout_name="ms__abc",
    )
    line = _failure_reason(failure, tmp_path / "job")
    assert line.body == (
        "reward 0.5 — mean of write 1.00 (soft check), append 0.00 (final verify); "
        "steps.json"
    )


def test_a_soft_check_s_output_is_kept_beside_the_trial(tmp_path):
    """Round 4 of the release review: steps.json gave an earlier step's
    reward with nothing to explain it; the next check overwrote its output."""
    from types import SimpleNamespace

    from benchflow.rollout._user_loop import _keep_round_output

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    stdout = trial / "verifier" / "test-stdout.txt"
    stdout.write_text("a.txt holds 1\n")
    rollout = SimpleNamespace(
        _rollout_dir=trial, _rollout_paths=SimpleNamespace(test_stdout_path=stdout)
    )
    assert _keep_round_output(rollout, 1, "the log", None) == "rounds/round-1-check.txt"
    # test.sh's own output wins over the verifier's logs.
    assert (trial / "rounds" / "round-1-check.txt").read_text() == "a.txt holds 1\n"
    # A link the tests left there is not followed: the logs stand in.
    secret = tmp_path / "elsewhere.txt"
    secret.write_text("not the check's\n")
    stdout.unlink()
    stdout.symlink_to(secret)
    assert _keep_round_output(rollout, 2, "the log", "soft verifier timed out") == (
        "rounds/round-2-check.txt"
    )
    kept = (trial / "rounds" / "round-2-check.txt").read_text()
    assert kept == "soft verifier timed out\nthe log\n"
    # Nothing to keep: no file.
    stdout.unlink()
    assert _keep_round_output(rollout, 3, None, None) is None
    assert not (trial / "rounds" / "round-3-check.txt").exists()


@pytest.mark.asyncio
async def test_the_oracle_and_nop_keep_each_soft_check_s_output(tmp_path):
    """The launch run of the oracle on the test tasks kept no check output:
    the scripted steps (oracle and nop) do not run the user loop."""
    import json

    from benchflow.rollout import Rollout

    task = Task(_task(tmp_path))
    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    stdout = trial / "verifier" / "test-stdout.txt"
    calls = []

    class Fake:
        _task = task
        _rollout_dir = trial
        _rollout_paths = type("P", (), {"test_stdout_path": stdout})()
        _active_steps_user = Rollout._active_steps_user
        _steps_user = None
        _config = None

        def __init__(self):
            self._user_rounds_log: list = []
            self._rewards = {"reward": 1.0}

        async def soft_verify(self):
            calls.append(1)
            stdout.write_text(f"check {len(calls)} ran\n")
            return {"reward": 1.0}, None, None

    fake = Fake()
    await Rollout._run_scripted_steps(fake, solve=False)
    Rollout._apply_step_rewards(fake)
    record = json.loads((trial / "steps.json").read_text())
    assert [row.get("check_output") for row in record["steps"]] == [
        "rounds/round-1-check.txt",
        "rounds/round-2-check.txt",
    ]
    assert (trial / "rounds" / "round-2-check.txt").read_text() == "check 2 ran\n"


def test_steps_json_keeps_the_soft_check_that_closed_the_gate(tmp_path):
    steps = Task(_task(tmp_path)).config.steps
    # Step 1's soft check gave 0.5, below its min_reward of 1; its final
    # verify later gave 1.0. The record keeps both.
    _, record = combine("final", steps, [{"reward": 0.5}], {"reward": 1.0})
    first = record["steps"][0]
    assert (first["source"], first["reward"], first["soft_reward"]) == (
        "final",
        1.0,
        0.5,
    )
    assert "its soft check gave 0.50" in describe(record)


def test_a_model_agent_on_a_multi_step_task_runs_the_steps_user(tmp_path):
    task = _task(tmp_path)
    config = RolloutConfig(task_path=task, agent="claude-agent-acp")
    assert isinstance(config.user, StepsUser)
    # One round per step and the call that ends the loop.
    assert config.max_user_rounds == 3
    assert RolloutConfig(task_path=task, agent="oracle").user is None


def test_the_protected_runner_is_wanted_by_what_the_steps_name(tmp_path):
    # A multi-step task without a task-level tests/ used to want the runner
    # (the folder it read was missing), which a host without the kit Python
    # refused before the first step.
    import shutil

    from benchflow.sandbox._verifier_runtime import runner_requested

    task = _task(tmp_path)
    shutil.rmtree(task / "tests")
    assert runner_requested(Task(task)) is False
    test_sh = task / "steps" / "append" / "tests" / "test.sh"
    test_sh.write_text('#!/bin/bash\n"$BENCHFLOW_PYTEST" /tests\n')
    assert runner_requested(Task(task)) is True
    test_sh.write_text("#!/bin/bash\n")
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[steps.verifier]\n",
            '[steps.verifier]\nenv = { RUN = "$BENCHFLOW_PYTEST" }\n',
        )
    )
    assert runner_requested(Task(task)) is True


def test_bench_check_names_a_step_without_its_files(tmp_path):
    task = _task(tmp_path)
    assert not [i for i in check_task(task) if "step" in i.lower()]
    (task / "steps" / "append" / "instruction.md").unlink()
    (task / "tests" / "test.sh").unlink()
    (task / "steps" / "append" / "tests" / "test.sh").unlink()
    issues = check_task(task)
    assert "Missing required file: steps/append/instruction.md" in issues
    assert any("Step 'append' has no tests" in i for i in issues)

"""Multi-step tasks: a task.toml whose ``[[steps]]`` each give the agent one
prompt, checked by that step's own tests.

Layout::

    task.toml                    # [[steps]] name, [steps.agent], [steps.verifier],
                                 # min_reward; multi_step_reward_strategy
    environment/
    steps/<name>/instruction.md  # the step's prompt
    steps/<name>/tests/          # the step's verifier (default: the task's tests/)
    steps/<name>/solution/       # the step's oracle (default: the task's solution/)

The steps run in order in one sandbox, one agent turn each, as the rounds of a
scripted user (rollout/_user_loop.py), so a step sees what the earlier ones
left in the workspace. After each turn the step's tests run as a soft verify
(the sandbox stays usable). A step whose reward is below its ``min_reward``
ends the task there. The last step that ran is verified again with full
hardening, and that is its reward.

``multi_step_reward_strategy``:

- ``final`` (the default): the task's reward is the last step's, from the
  hardened verify, and it is 0 when an earlier step stopped the task.
- ``mean``: every declared step's reward averaged, a step not reached counting
  0. The steps before the last come from their soft verifies, which run while
  the agent's session is over but before the workspace is frozen: use
  ``final`` where a reward must resist tampering.

``steps.json`` in the trial folder records each step's reward and its source.
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any

from benchflow.contracts.user import BaseUser, RoundResult
from benchflow.task.paths import TaskPaths

logger = logging.getLogger(__name__)

STEPS_DIRNAME = "steps"
RECORD_NAME = "steps.json"


class StepPaths(TaskPaths):
    """A step's view of the task folder: its own tests, solution and
    instruction where it has them, the task's otherwise."""

    def __init__(self, task_dir: Path | str, step_dir: Path) -> None:
        super().__init__(task_dir)
        self.step_dir = step_dir
        self._task_paths = TaskPaths(task_dir)

    def _own(self, name: str) -> Path | None:
        path = self.step_dir / name
        return path if path.is_dir() else None

    @property
    def instruction_path(self) -> Path:
        return self.step_dir / "instruction.md"

    @property
    def verifier_source_dir(self) -> Path:
        return self._own(self.NATIVE_VERIFIER_DIRNAME) or (
            self._task_paths.verifier_source_dir
        )

    @property
    def legacy_tests_dir(self) -> Path:
        return self._own(self.LEGACY_TESTS_DIRNAME) or self._task_paths.legacy_tests_dir

    @property
    def tests_dir(self) -> Path:
        return (
            self._own(self.NATIVE_VERIFIER_DIRNAME)
            or self._own(self.LEGACY_TESTS_DIRNAME)
            or self._task_paths.tests_dir
        )

    @property
    def uses_native_verifier_dir(self) -> bool:
        return self.tests_dir.name == self.NATIVE_VERIFIER_DIRNAME

    @property
    def oracle_dir(self) -> Path:
        return self._own(self.NATIVE_ORACLE_DIRNAME) or self._task_paths.oracle_dir

    @property
    def legacy_solution_dir(self) -> Path:
        return (
            self._own(self.LEGACY_SOLUTION_DIRNAME)
            or self._task_paths.legacy_solution_dir
        )

    @property
    def solution_dir(self) -> Path:
        return (
            self._own(self.NATIVE_ORACLE_DIRNAME)
            or self._own(self.LEGACY_SOLUTION_DIRNAME)
            or self._task_paths.solution_dir
        )

    @property
    def uses_native_oracle_dir(self) -> bool:
        return self.solution_dir.name == self.NATIVE_ORACLE_DIRNAME


def has_steps(task: Any) -> bool:
    steps = getattr(getattr(task, "config", None), "steps", None)
    return isinstance(steps, list) and len(steps) > 0


def step_dir(task: Any, index: int) -> Path:
    return Path(task.task_dir) / STEPS_DIRNAME / task.config.steps[index].name


def step_paths(task: Any, index: int) -> StepPaths:
    return StepPaths(task.task_dir, step_dir(task, index))


def step_instruction(task: Any, index: int) -> str:
    return (
        (step_dir(task, index) / "instruction.md").read_text(encoding="utf-8").strip()
    )


def step_task(task: Any, index: int) -> Any:
    """*task* as step *index* runs it: that step's paths, instruction, and its
    agent and verifier settings over the task's (a field the step leaves at
    its default keeps the task's value)."""
    step = task.config.steps[index]
    view = copy.copy(task)
    view.paths = step_paths(task, index)
    view.instruction = step_instruction(task, index)
    # How the log names this step (the verifier's rewards line, the user loop).
    view.step_label = f"step {index + 1}/{len(task.config.steps)} {step.name}"
    updates: dict[str, Any] = {}
    for section in ("agent", "verifier"):
        own = getattr(step, section)
        if own.model_fields_set:
            base = getattr(task.config, section)
            updates[section] = base.model_copy(
                update={name: getattr(own, name) for name in own.model_fields_set}
            )
    if updates:
        view.config = task.config.model_copy(update=updates)
    return view


def _reward(rewards: dict[str, Any] | None) -> float | None:
    value = (rewards or {}).get("reward")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def meets_min_reward(step: Any, rewards: dict[str, Any] | None) -> bool:
    """Whether a step's rewards let the task go on to the next step. A step
    with no ``min_reward`` always does; a missing reward never meets one."""
    threshold = step.min_reward
    if threshold is None:
        return True
    if isinstance(threshold, dict):
        values = rewards or {}
        return all(
            isinstance(values.get(key), int | float)
            and not isinstance(values.get(key), bool)
            and float(values[key]) >= float(bound)
            for key, bound in threshold.items()
        )
    reward = _reward(rewards)
    return reward is not None and reward >= float(threshold)


def _bound(threshold: Any) -> str:
    if isinstance(threshold, dict):
        return ", ".join(
            f"{key} >= {float(bound):g}" for key, bound in threshold.items()
        )
    return f"{float(threshold):g}"


def _shown(reward: float | None) -> str:
    return "no reward" if reward is None else f"{reward:.2f}"


class StepsUser(BaseUser):
    """The scripted user of a multi-step task: step *i*'s instruction as
    round *i*'s prompt, until a step misses its ``min_reward`` or the steps
    run out. ``task_for_round`` is what the round's verify scores with."""

    def __init__(self, task: Any) -> None:
        self.task = task
        self.steps = list(task.config.steps)
        # Each finished round's soft-verify rewards, in step order.
        self.soft_rewards: list[dict[str, Any] | None] = []
        self.stopped_after: int | None = None

    def record(
        self, index: int, rewards: dict[str, Any] | None, tools: int | None = None
    ) -> bool:
        """Note step *index*'s soft-verify rewards, with one log line naming
        the step; whether the task goes on to the next step."""
        self.soft_rewards.append(rewards)
        step, total = self.steps[index], len(self.steps)
        line = (
            f"Step {index + 1}/{total} {step.name}: reward {_shown(_reward(rewards))}"
            + (f" after {tools} tool call(s)" if tools is not None else "")
        )
        if meets_min_reward(step, rewards):
            logger.info(line)
            return True
        self.stopped_after = index
        later = total - index - 1
        logger.info(
            f"{line}, below its min_reward {_bound(step.min_reward)}: the task stops here"
            + (
                f" and the {later} later step{'s count' if later > 1 else ' counts'} 0"
                if later
                else ""
            )
        )
        return False

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round_result is not None and not self.record(
            round - 1, round_result.rewards, round_result.n_tool_calls
        ):
            return None
        if round >= len(self.steps):
            return None
        return step_instruction(self.task, round)

    def task_for_round(self, round: int) -> Any:
        return step_task(self.task, round)


def combine(
    strategy: str | None,
    steps: list[Any],
    soft_rewards: list[dict[str, Any] | None],
    final_rewards: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The task's rewards and the ``steps.json`` record.

    *soft_rewards* holds every step that ran, in order; the last one's score
    is *final_rewards* (the hardened verify). A step not reached scores 0."""
    strategy = strategy or "final"
    ran = len(soft_rewards)
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for index, step in enumerate(steps):
        if index < ran - 1:
            reward, source = _reward(soft_rewards[index]), "soft"
        elif index == ran - 1:
            reward, source = _reward(final_rewards), "final"
        else:
            reward, source = None, "not-run"
        row: dict[str, Any] = {"name": step.name, "reward": reward, "source": source}
        if source == "final":
            # The soft check that decided whether the task went on.
            row["soft_reward"] = _reward(soft_rewards[index])
        rows.append(row)
        values.append(reward or 0.0)
    record: dict[str, Any] = {"version": 1, "strategy": strategy, "steps": rows}
    if final_rewards is None:
        record["reward"] = None
        return None, record
    if strategy == "mean":
        reward = sum(values) / len(values) if values else 0.0
    else:
        reward = values[-1]
    record["reward"] = reward
    return {**final_rewards, "reward": reward}, record


def describe(record: dict[str, Any]) -> str:
    """The ``steps.json`` record in one line: each step's reward and how the
    task's reward came from them."""
    rows = record["steps"]
    parts = [
        f"{row['name']} not run"
        if row["source"] == "not-run"
        else f"{row['name']} {_shown(row['reward'])}"
        + (
            (
                " (final verify)"
                if row.get("soft_reward") == row["reward"]
                else f" (final verify; its soft check gave {_shown(row.get('soft_reward'))})"
            )
            if row["source"] == "final"
            else " (soft check)"
        )
        for row in rows
    ]
    reward = record.get("reward")
    if reward is None:
        how = "no task reward: the final verify gave none"
    elif record.get("strategy") == "mean":
        how = f"task reward {reward:.2f}, the mean of the {len(rows)} steps (a step not run counts 0)"
    elif rows[-1]["source"] == "not-run":
        how = (
            f"task reward {reward:.2f}: the last step ({rows[-1]['name']}) did not run"
        )
    else:
        how = f"task reward {reward:.2f}, the last step's ({rows[-1]['name']})"
    return f"Steps: {', '.join(parts)}; {how}"

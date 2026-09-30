"""BenchFlow-owned adapter for TRL online training.

The adapter maps BenchFlow-compatible task directories onto the three slots
expected by TRL's environment-training APIs: ``train_dataset``,
``environment_factory``, and ``reward_funcs``. BenchFlow owns task loading,
sandbox lifecycle, verifier execution, and artifacts; TRL owns optimization.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import shlex
import threading
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow._utils.task_authoring import check_task, task_document_parse_error
from benchflow.integrations.rewards import (
    DROP_REASONS,
    VERIFIER_CRASH_CLEAN_RUN,
    VERIFIER_ERROR,
    ZERO_REASONS,
    RewardDecision,
    dropped,
    reward_from_verify,
    sandbox_start_failure,
    summarize,
    zero,
)
from benchflow.rollout import TaskRuntime, TaskRuntimeConfig
from benchflow.task.package import TaskPackage

logger = logging.getLogger(__name__)


class _AsyncRunner:
    """Own one event loop for the lifetime of the synchronous TRL adapter."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return self.submit(coro).result()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        loop = self._loop
        if loop is None:
            raise RuntimeError("BenchFlow TRL async runner failed to start")
        return asyncio.run_coroutine_threadsafe(coro, loop)


_ASYNC_RUNNER: _AsyncRunner | None = None
_ASYNC_RUNNER_LOCK = threading.Lock()


def _async_runner() -> _AsyncRunner:
    global _ASYNC_RUNNER
    with _ASYNC_RUNNER_LOCK:
        if _ASYNC_RUNNER is None:
            _ASYNC_RUNNER = _AsyncRunner()
        return _ASYNC_RUNNER


class BenchFlowOptionalDependencyError(ImportError):
    """Raised when optional TRL integration dependencies are used but missing."""


@dataclass(frozen=True)
class BashHarnessConfig:
    """Runtime settings for the minimal bash tool surface exposed to TRL."""

    environment: str = "docker"
    sandbox_user: str | None = "agent"
    jobs_dir: Path | str = "jobs/trl"
    bash_timeout_sec: int = 30
    max_output_chars: int = 4096
    submit_path: str = "/workdir/answer.txt"
    reset_message: str | None = None
    planes: Any | None = None
    # TRL resets a batch's environments one after another. With
    # background_start, reset() returns at once and the sandbox starts while
    # the model generates; the first tool call waits for it.
    background_start: bool = False

    def normalized(self) -> BashHarnessConfig:
        if self.bash_timeout_sec < 1:
            raise ValueError("bash_timeout_sec must be >= 1")
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be >= 1")
        if not self.submit_path.startswith("/"):
            raise ValueError("submit_path must be absolute")
        return BashHarnessConfig(
            environment=self.environment,
            sandbox_user=self.sandbox_user,
            jobs_dir=Path(self.jobs_dir),
            bash_timeout_sec=self.bash_timeout_sec,
            max_output_chars=self.max_output_chars,
            submit_path=self.submit_path,
            reset_message=self.reset_message,
            planes=self.planes,
            background_start=self.background_start,
        )


@dataclass(frozen=True)
class BenchFlowSpecConfig:
    """Configuration for building a TRL-compatible BenchFlow spec."""

    tasks_dir: Path | str
    include_tasks: Sequence[str] = ()
    exclude_tasks: Sequence[str] = ()
    max_tasks: int | None = None
    bash_harness: BashHarnessConfig = field(default_factory=BashHarnessConfig)

    def normalized(self) -> BenchFlowSpecConfig:
        max_tasks = self.max_tasks
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("max_tasks must be >= 1")
        return BenchFlowSpecConfig(
            tasks_dir=Path(self.tasks_dir),
            include_tasks=tuple(dict.fromkeys(self.include_tasks)),
            exclude_tasks=tuple(dict.fromkeys(self.exclude_tasks)),
            max_tasks=max_tasks,
            bash_harness=self.bash_harness.normalized(),
        )


@dataclass(frozen=True)
class _TaskRow:
    task_id: str
    task_name: str
    task_dir: Path
    entrypoint: str
    prompt_turn_index: int
    prompt: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "benchflow_task_id": self.task_id,
            "benchflow_task_name": self.task_name,
            "benchflow_task_dir": str(self.task_dir),
            "benchflow_entrypoint": self.entrypoint,
            "benchflow_prompt_turn_index": self.prompt_turn_index,
        }


class BenchFlowRuntimeEnvironment:
    """Synchronous TRL environment backed by ``TaskRuntime``.

    TRL discovers public methods as tools. ``run_bash`` and ``submit`` are the
    v1 tool surface; both execute inside the BenchFlow task sandbox.

    Each rollout ends with a :class:`~benchflow.integrations.rewards.RewardDecision`
    in ``decision``: a verifier reward, a 0 for a failure the policy could
    have caused, or a drop for an infrastructure failure (the sandbox never
    started, or the verifier crashed on a sandbox the policy never touched).
    ``reward`` is the decision's reward, and None for a dropped rollout.
    """

    def __init__(self, harness: BashHarnessConfig) -> None:
        self._harness = harness.normalized()
        self._runtime: TaskRuntime | None = None
        self._starting: concurrent.futures.Future | None = None
        self.task_id: str | None = None
        self.reward: float | None = None
        self.decision: RewardDecision | None = None
        self.policy_acted: bool = False
        self.rollout_dir: Path | None = None
        self.last_returncode: int | None = None

    def reset(self, **kwargs: Any) -> str | None:
        """Start a fresh BenchFlow runtime for one training rollout."""

        self._close()
        task_dir_value = kwargs.get("benchflow_task_dir")
        if not isinstance(task_dir_value, str) or not task_dir_value:
            raise ValueError("BenchFlow TRL rows must include benchflow_task_dir")
        task_id = kwargs.get("benchflow_task_id")
        self.task_id = (
            str(task_id) if task_id is not None else Path(task_dir_value).name
        )
        runtime_config = TaskRuntimeConfig(
            task_path=task_dir_value,
            environment=self._harness.environment,
            sandbox_user=self._harness.sandbox_user,
            jobs_dir=self._harness.jobs_dir,
            planes=self._harness.planes,
        )
        self.reward = None
        self.decision = None
        self.policy_acted = False
        self.rollout_dir = None
        self.last_returncode = None
        starting = _async_runner().submit(TaskRuntime.create(runtime_config))
        if self._harness.background_start:
            self._starting = starting
        else:
            self._adopt_started(starting)
        return self._harness.reset_message

    def run_bash(self, command: str) -> str:
        """Run a bash command in the task sandbox and return its output (stdout and stderr).

        Args:
            command: The bash command to run in the task's working directory.
        """

        runtime = self._require_runtime()
        self.policy_acted = True
        result = _run_blocking(
            runtime.bash(command, timeout_sec=self._harness.bash_timeout_sec)
        )
        self.last_returncode = result.return_code
        output = result.stdout
        if result.stderr:
            output = f"{output}{result.stderr}"
        return _truncate(output, self._harness.max_output_chars)

    def submit(self, answer: str) -> str:
        """Submit the final answer. This ends the task, so call it once, when you are done.

        Args:
            answer: The final answer, written to the task's answer file.
        """

        runtime = self._require_runtime()
        self.policy_acted = True
        answer_literal = shlex.quote(str(answer))
        submit_path = shlex.quote(self._harness.submit_path)
        _run_blocking(
            runtime.bash(
                f"mkdir -p $(dirname {submit_path}) && printf %s {answer_literal} > {submit_path}",
                timeout_sec=self._harness.bash_timeout_sec,
            )
        )
        self._finalize()
        decision = self.decision
        if decision is None or decision.reward is None:
            reason = decision.reason if decision is not None else "unknown"
            return f"submission recorded; rollout dropped ({reason})"
        return f"submission recorded; reward={decision.reward:g}"

    def _adopt_started(self, starting: concurrent.futures.Future) -> None:
        """Take the started runtime, or record a sandbox that never started."""

        try:
            self._runtime = starting.result()
        except Exception as exc:
            self._runtime = None
            self._set_decision(sandbox_start_failure(exc))
            return
        self.rollout_dir = self._runtime.rollout_dir

    async def _aadopt_started(self) -> None:
        starting = self._starting
        if starting is None:
            return
        self._starting = None
        try:
            self._runtime = await asyncio.wrap_future(starting)
        except Exception as exc:
            self._runtime = None
            self._set_decision(sandbox_start_failure(exc))
            return
        self.rollout_dir = self._runtime.rollout_dir

    def _set_decision(self, decision: RewardDecision) -> None:
        self.decision = decision
        self.reward = decision.reward

    def _finalize(self) -> None:
        _run_blocking(self._afinalize())

    async def _afinalize(self) -> None:
        """Verify once and record the decision; always release the sandbox."""

        await self._aadopt_started()
        runtime = self._runtime
        if runtime is None:
            return
        self._runtime = None
        try:
            try:
                result = await runtime.verify()
            except Exception as exc:
                self._set_decision(
                    zero(VERIFIER_ERROR, exc)
                    if self.policy_acted
                    else dropped(VERIFIER_CRASH_CLEAN_RUN, exc)
                )
            else:
                self._set_decision(
                    reward_from_verify(result, policy_acted=self.policy_acted)
                )
                self.rollout_dir = getattr(result, "rollout_dir", self.rollout_dir)
        finally:
            await runtime.close()

    def _close(self) -> None:
        starting = self._starting
        self._starting = None
        if starting is not None:
            try:
                self._runtime = starting.result()
            except Exception:
                self._runtime = None
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            _run_blocking(runtime.close())

    def _require_runtime(self) -> TaskRuntime:
        if self._starting is not None:
            self._adopt_started(self._starting)
            self._starting = None
        if self._runtime is None:
            if self.decision is not None and self.decision.dropped:
                raise RuntimeError(
                    "the sandbox for this task failed to start; this rollout is "
                    "dropped from training"
                )
            raise RuntimeError("environment must be reset before tool use")
        return self._runtime


def benchflow_environment_reward(
    completions: Sequence[Any],
    *,
    environments: Sequence[Any] | None = None,
    prompts: Sequence[Any] | None = None,
    trainer_state: Any = None,
    log_metric: Callable[[str, float], None] | None = None,
    log_extra: Callable[[str, list], None] | None = None,
    **_: Any,
) -> list[float | None]:
    """TRL custom reward function reading reward from BenchFlow environments.

    Rollouts still open are verified concurrently. A dropped rollout gets
    ``None``, which TRL treats as unscorable: it is left out of its group's
    baseline and gets zero advantage. When TRL passes ``log_metric`` and
    ``log_extra``, the drop count, the drop reasons, and each rollout's reason
    are logged with the batch.

    Every rollout, dropped or not, is kept for audit: its messages (the
    prompt and the whole multi-turn completion) go to ``policy/messages.json``
    in its rollout folder, and a line with the decision goes to
    ``rollouts.jsonl`` in the harness's ``jobs_dir``.
    """

    if environments is None:
        return [0.0 for _ in completions]

    envs = [
        environments[index] if index < len(environments) else None
        for index in range(len(completions))
    ]
    open_envs = [env for env in envs if isinstance(env, BenchFlowRuntimeEnvironment)]
    if open_envs:
        _run_blocking(_finalize_all(open_envs))

    rewards: list[float | None] = []
    reasons: list[str] = []
    decisions: list[RewardDecision] = []
    for env in envs:
        decision = getattr(env, "decision", None)
        if isinstance(decision, RewardDecision):
            decisions.append(decision)
            rewards.append(decision.reward)
            reasons.append(decision.reason)
            continue
        value = getattr(env, "reward", 0.0)
        rewards.append(
            float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else 0.0
        )
        reasons.append("n/a")
    _log_decisions(decisions, reasons, log_metric=log_metric, log_extra=log_extra)
    _record_rollouts(envs, prompts, completions, trainer_state)
    return rewards


async def _finalize_all(envs: Sequence[BenchFlowRuntimeEnvironment]) -> None:
    results = await asyncio.gather(
        *(env._afinalize() for env in envs), return_exceptions=True
    )
    for env, result in zip(envs, results, strict=True):
        if isinstance(result, BaseException) and env.decision is None:
            # Close failed after verification, or verification raised
            # something unexpected: score by who could have caused it.
            env._set_decision(
                zero(VERIFIER_ERROR, result)
                if env.policy_acted
                else dropped(VERIFIER_CRASH_CLEAN_RUN, result)
            )


def rollout_record(
    env: BenchFlowRuntimeEnvironment,
    messages: Sequence[Any] | None,
    *,
    step: Any = None,
) -> dict[str, Any]:
    """One rollout's audit record: task, folder, decision, and messages."""

    decision = env.decision
    return {
        "task_id": env.task_id,
        "rollout_dir": str(env.rollout_dir) if env.rollout_dir is not None else None,
        "step": step,
        "policy_acted": env.policy_acted,
        **(
            decision.as_dict()
            if decision is not None
            else {"reward": None, "reason": None}
        ),
        "messages": list(messages) if messages is not None else None,
    }


def write_rollout_record(record: dict[str, Any], jobs_dir: Path | str) -> None:
    """Keep a rollout for audit, in its own folder and in ``rollouts.jsonl``."""

    text = json.dumps(record, default=str)
    rollout_dir = record.get("rollout_dir")
    if rollout_dir:
        policy_dir = Path(rollout_dir) / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "messages.json").write_text(text + "\n")
    jobs = Path(jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    with _RECORD_LOCK, (jobs / "rollouts.jsonl").open("a") as handle:
        handle.write(text + "\n")


_RECORD_LOCK = threading.Lock()


def finish_rollout(
    env: BenchFlowRuntimeEnvironment,
    messages: Sequence[Any] | None = None,
    *,
    drop: RewardDecision | None = None,
    step: Any = None,
) -> RewardDecision:
    """End one rollout outside TRL, the way TRL's reward function ends it.

    Verifies the sandbox (unless it already was), records the rollout for
    audit, and returns its decision. Pass ``drop`` for an infrastructure
    failure outside the sandbox, such as the model endpoint failing: the
    sandbox is then closed without verification and the rollout is dropped.
    """

    if drop is not None:
        if not drop.dropped:
            raise ValueError("drop must be a dropped decision")
        env._close()
        env._set_decision(drop)
    else:
        _run_blocking(_finalize_all([env]))
    decision = env.decision
    assert decision is not None
    write_rollout_record(
        rollout_record(env, messages, step=step), env._harness.jobs_dir
    )
    return decision


def _record_rollouts(
    envs: Sequence[Any],
    prompts: Sequence[Any] | None,
    completions: Sequence[Any],
    trainer_state: Any,
) -> None:
    step = getattr(trainer_state, "global_step", None)
    for index, env in enumerate(envs):
        if not isinstance(env, BenchFlowRuntimeEnvironment):
            continue
        messages: list[Any] = []
        if (
            prompts is not None
            and index < len(prompts)
            and isinstance(prompts[index], list)
        ):
            messages.extend(prompts[index])
        completion = completions[index] if index < len(completions) else None
        if isinstance(completion, list):
            messages.extend(completion)
        elif completion is not None:
            messages.append({"role": "assistant", "content": completion})
        try:
            write_rollout_record(
                rollout_record(env, messages, step=step), env._harness.jobs_dir
            )
        except OSError as exc:  # never let audit bookkeeping stop training
            logger.warning("could not record rollout %s: %s", env.task_id, exc)


def _log_decisions(
    decisions: Sequence[RewardDecision],
    reasons: list[str],
    *,
    log_metric: Callable[[str, float], None] | None,
    log_extra: Callable[[str, list], None] | None,
) -> None:
    # TRL averages each metric over a logging step and gathers metrics across
    # ranks by name, so every batch logs the same keys, as fractions.
    n = max(len(reasons), 1)
    if log_metric is not None:
        summary = summarize(decisions)
        log_metric("benchflow/dropped_frac", summary["dropped"] / n)
        for reason in sorted(DROP_REASONS):
            log_metric(
                f"benchflow/drop/{reason}", summary["drop_reasons"].get(reason, 0) / n
            )
        for reason in sorted(ZERO_REASONS):
            log_metric(
                f"benchflow/zero/{reason}", summary["zero_reasons"].get(reason, 0) / n
            )
    if log_extra is not None:
        log_extra("benchflow_reason", list(reasons))


def bash_tool_schemas() -> list[dict[str, Any]]:
    """The ``run_bash`` and ``submit`` tools as OpenAI-style function definitions.

    These are the definitions TRL derives from :class:`BenchFlowRuntimeEnvironment`'s
    method signatures and docstrings, so a policy evaluated through an
    OpenAI-compatible endpoint sees the same tools it was trained with.
    """

    return [
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": (
                    "Run a bash command in the task sandbox and return its output "
                    "(stdout and stderr)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The bash command to run in the task's working directory."
                            ),
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": (
                    "Submit the final answer. This ends the task, so call it once, "
                    "when you are done."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer, written to the task's answer file.",
                        }
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


class BenchFlowSpec:
    """Public TRL adapter for BenchFlow task suites."""

    def __init__(
        self,
        config: BenchFlowSpecConfig | str | Path | None = None,
        *,
        tasks_dir: str | Path | None = None,
        include_tasks: Sequence[str] = (),
        exclude_tasks: Sequence[str] = (),
        max_tasks: int | None = None,
        bash_harness: BashHarnessConfig | None = None,
    ) -> None:
        if isinstance(config, BenchFlowSpecConfig):
            if tasks_dir is not None:
                raise ValueError("tasks_dir cannot be passed with BenchFlowSpecConfig")
            normalized = config.normalized()
        else:
            resolved_tasks_dir = tasks_dir if config is None else config
            if resolved_tasks_dir is None:
                raise ValueError("BenchFlowSpec requires tasks_dir")
            normalized = BenchFlowSpecConfig(
                tasks_dir=resolved_tasks_dir,
                include_tasks=include_tasks,
                exclude_tasks=exclude_tasks,
                max_tasks=max_tasks,
                bash_harness=bash_harness or BashHarnessConfig(),
            ).normalized()
        self.config = normalized
        self._rows = tuple(_load_task_rows(normalized))
        if not self._rows:
            raise ValueError(
                f"No runnable BenchFlow tasks found under {normalized.tasks_dir}"
            )

    @classmethod
    def from_tasks_dir(
        cls,
        tasks_dir: str | Path,
        **kwargs: Any,
    ) -> BenchFlowSpec:
        return cls(tasks_dir, **kwargs)

    @property
    def train_dataset_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(row.as_dict() for row in self._rows)

    @property
    def train_dataset(self) -> Any:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise BenchFlowOptionalDependencyError(
                "benchflow.integrations.trl.BenchFlowSpec.train_dataset requires "
                "the optional TRL integration dependencies. Install with "
                "`pip install 'benchflow[trl]'` or `uv sync --extra trl`."
            ) from exc
        return Dataset.from_list(list(self.train_dataset_rows))

    @property
    def environment_factory(self) -> Callable[[], BenchFlowRuntimeEnvironment]:
        def factory() -> BenchFlowRuntimeEnvironment:
            return BenchFlowRuntimeEnvironment(self.config.bash_harness)

        return factory

    @property
    def reward_funcs(self) -> list[Callable[..., list[float | None]]]:
        return [benchflow_environment_reward]

    def trainer_kwargs(self) -> dict[str, Any]:
        return {
            "train_dataset": self.train_dataset,
            "environment_factory": self.environment_factory,
            "reward_funcs": self.reward_funcs,
        }

    def create_trainer(self, *, model: Any, **kwargs: Any) -> Any:
        try:
            from trl import GRPOTrainer
        except ImportError as exc:
            raise BenchFlowOptionalDependencyError(
                "benchflow.integrations.trl.BenchFlowSpec.create_trainer requires "
                "TRL. Install with `pip install 'benchflow[trl]'` or "
                "`uv sync --extra trl`."
            ) from exc

        trainer_kwargs = self.trainer_kwargs()
        trainer_kwargs.update(kwargs)
        return GRPOTrainer(model=model, **trainer_kwargs)


def _load_task_rows(config: BenchFlowSpecConfig) -> list[_TaskRow]:
    rows: list[_TaskRow] = []
    for task_dir in _discover_task_dirs(
        Path(config.tasks_dir),
        include_tasks=set(config.include_tasks),
        exclude_tasks=set(config.exclude_tasks),
    ):
        package = TaskPackage.from_task_dir(task_dir)
        rows.extend(_rows_for_package(package))
        if (
            config.max_tasks is not None
            and len({row.task_id for row in rows}) >= config.max_tasks
        ):
            break
    return rows


def _discover_task_dirs(
    tasks_dir: Path,
    *,
    include_tasks: set[str],
    exclude_tasks: set[str],
) -> list[Path]:
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {tasks_dir}")
    if _is_runnable_task_dir(tasks_dir):
        if tasks_dir.name in exclude_tasks:
            return []
        if include_tasks and tasks_dir.name not in include_tasks:
            return []
        return [tasks_dir]

    selected: list[Path] = []
    for child in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        if child.name in exclude_tasks:
            continue
        if include_tasks and child.name not in include_tasks:
            continue
        if _is_runnable_task_dir(child):
            selected.append(child)
            continue
        task_md = child / "task.md"
        parse_error = task_document_parse_error(task_md) if task_md.is_file() else None
        if parse_error is not None:
            raise ValueError(f"Malformed BenchFlow task {child}: {parse_error}")
    root_task_md = tasks_dir / "task.md"
    root_parse_error = (
        task_document_parse_error(root_task_md) if root_task_md.is_file() else None
    )
    if root_parse_error is not None and not selected:
        raise ValueError(f"Malformed BenchFlow task {tasks_dir}: {root_parse_error}")
    return selected


def _is_runnable_task_dir(path: Path) -> bool:
    if (path / "task.md").is_file():
        return check_task(path) == []
    return (path / "task.toml").is_file() and check_task(path) == []


def _rows_for_package(package: TaskPackage) -> list[_TaskRow]:
    prompt_plan = package.prompt_plan
    turns = prompt_plan.turns if prompt_plan is not None else ()
    if not turns:
        return []

    task_config = package.view.config.task
    task_name = task_config.name if task_config is not None else package.task_dir.name
    return [
        _TaskRow(
            task_id=package.task_dir.name,
            task_name=task_name,
            task_dir=package.task_dir,
            entrypoint=package.view.entrypoint,
            prompt_turn_index=index,
            prompt=[{"role": "user", "content": turn.prompt}],
        )
        for index, turn in enumerate(turns)
    ]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n[benchflow output truncated]\n"
    keep = max(0, max_chars - len(marker))
    return f"{text[:keep]}{marker}"


def _run_blocking(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async BenchFlow primitive from TRL's sync tool surface."""

    return _async_runner().run(coro)


__all__ = [
    "BashHarnessConfig",
    "BenchFlowOptionalDependencyError",
    "BenchFlowRuntimeEnvironment",
    "BenchFlowSpec",
    "BenchFlowSpecConfig",
    "bash_tool_schemas",
    "benchflow_environment_reward",
    "finish_rollout",
    "rollout_record",
    "write_rollout_record",
]

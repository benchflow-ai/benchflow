"""BenchFlow-owned adapter for TRL online training.

The adapter maps BenchFlow-compatible task directories onto the three slots
expected by TRL's environment-training APIs: ``train_dataset``,
``environment_factory``, and ``reward_funcs``. BenchFlow owns task loading,
sandbox lifecycle, verifier execution, and artifacts; TRL owns optimization.
"""

from __future__ import annotations

import asyncio
import shlex
import threading
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow._utils.task_authoring import check_task, task_document_parse_error
from benchflow.rollout import TaskRuntime, TaskRuntimeConfig
from benchflow.task.package import TaskPackage


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
        loop = self._loop
        if loop is None:
            raise RuntimeError("BenchFlow TRL async runner failed to start")
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


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
    """

    def __init__(self, harness: BashHarnessConfig) -> None:
        self._harness = harness.normalized()
        self._runtime: TaskRuntime | None = None
        self.task_id: str | None = None
        self.reward: float = 0.0
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
        self._runtime = _run_blocking(TaskRuntime.create(runtime_config))
        self.reward = 0.0
        self.rollout_dir = self._runtime.rollout_dir
        return self._harness.reset_message

    def run_bash(self, command: str) -> str:
        """Run a bash command in the BenchFlow task sandbox.

        Args:
            command: Bash command to execute in the task workspace.

        Returns:
            Combined standard output and standard error, truncated if needed.
        """

        runtime = self._require_runtime()
        result = _run_blocking(
            runtime.bash(command, timeout_sec=self._harness.bash_timeout_sec)
        )
        self.last_returncode = result.return_code
        output = result.stdout
        if result.stderr:
            output = f"{output}{result.stderr}"
        return _truncate(output, self._harness.max_output_chars)

    def submit(self, answer: str) -> str:
        """Write the final answer and run the BenchFlow verifier.

        Args:
            answer: Final answer string to write to the configured submission path.

        Returns:
            Confirmation text including the verifier reward.
        """

        runtime = self._require_runtime()
        answer_literal = shlex.quote(str(answer))
        submit_path = shlex.quote(self._harness.submit_path)
        _run_blocking(
            runtime.bash(
                f"mkdir -p $(dirname {submit_path}) && printf %s {answer_literal} > {submit_path}",
                timeout_sec=self._harness.bash_timeout_sec,
            )
        )
        self._finalize()
        return f"submission recorded; reward={self.reward:g}"

    def _finalize(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            result = _run_blocking(runtime.verify())
            self.reward = (
                float(result.reward) if isinstance(result.reward, int | float) else 0.0
            )
            self.rollout_dir = result.rollout_dir
        finally:
            self._runtime = None
            _run_blocking(runtime.close())

    def _close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            _run_blocking(runtime.close())

    def _require_runtime(self) -> TaskRuntime:
        if self._runtime is None:
            raise RuntimeError("environment must be reset before tool use")
        return self._runtime


def benchflow_environment_reward(
    completions: Sequence[Any],
    *,
    environments: Sequence[Any] | None = None,
    **_: Any,
) -> list[float]:
    """TRL custom reward function reading reward from BenchFlow environments."""

    if environments is None:
        return [0.0 for _ in completions]

    rewards: list[float] = []
    for index, _completion in enumerate(completions):
        env = environments[index] if index < len(environments) else None
        if isinstance(env, BenchFlowRuntimeEnvironment):
            env._finalize()
        value = getattr(env, "reward", 0.0)
        rewards.append(
            float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else 0.0
        )
    return rewards


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
    def reward_funcs(self) -> list[Callable[..., list[float]]]:
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
    "benchflow_environment_reward",
]

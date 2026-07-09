"""Synthetic coverage for the optional TRL GRPO integration."""

from __future__ import annotations

import builtins
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from benchflow.integrations.trl import (
    BashHarnessConfig,
    BenchFlowOptionalDependencyError,
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    BenchFlowSpecConfig,
)


def _make_task(parent: Path, name: str, instruction: str) -> Path:
    task = parent / name
    task.mkdir()
    (task / "task.toml").write_text(
        f"""
version = "1.0"

[task]
name = "benchflow/{name}"
description = "Synthetic TRL fixture"

[verifier]
timeout_sec = 60

[agent]
timeout_sec = 60

[environment]
""".lstrip()
    )
    (task / "instruction.md").write_text(instruction)
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests = task / "tests"
    tests.mkdir()
    test_sh = tests / "test.sh"
    test_sh.write_text("#!/usr/bin/env bash\necho 1 >/logs/verifier/reward.txt\n")
    test_sh.chmod(0o755)
    return task


@dataclass
class _FakeBashResult:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeRuntimeResult:
    reward: float
    rollout_dir: Path


class _FakeRuntime:
    created_configs: ClassVar[list[Any]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rollout_dir = Path(config.jobs_dir) / "fake-rollout"
        self.commands: list[str] = []
        self.closed = False

    @classmethod
    async def create(cls, config: Any) -> _FakeRuntime:
        cls.created_configs.append(config)
        return cls(config)

    async def bash(self, command: str, *, timeout_sec: int = 30) -> _FakeBashResult:
        self.commands.append(command)
        if command == "printf hi":
            return _FakeBashResult(stdout="hi")
        return _FakeBashResult()

    async def verify(self) -> _FakeRuntimeResult:
        reward = (
            1.0 if any("/workdir/answer.txt" in cmd for cmd in self.commands) else 0.0
        )
        return _FakeRuntimeResult(reward=reward, rollout_dir=self.rollout_dir)

    async def close(self) -> None:
        self.closed = True


def test_public_import_path_does_not_require_trl(monkeypatch: pytest.MonkeyPatch):
    """Guards PR #903: importing the public path must not import TRL."""

    monkeypatch.setitem(sys.modules, "trl", None)

    from benchflow.integrations.trl import BenchFlowSpec as Imported

    assert Imported is BenchFlowSpec


def test_spec_loads_task_rows_without_trl_dependency(tmp_path: Path):
    """Guards PR #903: task dataset rows load from BenchFlow packages."""

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    _make_task(tasks, "beta", "Answer beta")

    spec = BenchFlowSpec(tasks_dir=tasks, include_tasks=["beta"])

    rows = spec.train_dataset_rows
    assert len(rows) == 1
    assert rows[0]["benchflow_task_id"] == "beta"
    assert rows[0]["benchflow_task_name"] == "benchflow/beta"
    assert rows[0]["prompt"] == [{"role": "user", "content": "Answer beta"}]


def test_train_dataset_uses_hf_dataset_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Guards PR #903: train_dataset has the TRL/HF dataset shape."""

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")

    fake_datasets = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    fake_datasets.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    spec = BenchFlowSpec.from_tasks_dir(tasks)

    dataset = spec.train_dataset
    assert isinstance(dataset, FakeDataset)
    assert dataset.rows == list(spec.train_dataset_rows)


def test_train_dataset_missing_optional_dependency_is_actionable(tmp_path: Path):
    """Guards PR #903: missing datasets/TRL deps fail only when used."""

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec.from_tasks_dir(tasks)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("blocked datasets for test")
        return real_import(name, *args, **kwargs)

    import pytest

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(
            BenchFlowOptionalDependencyError, match="benchflow\\[trl\\]"
        ):
            _ = spec.train_dataset


def test_train_dataset_available_when_extra_is_installed(tmp_path: Path):
    """Guards PR #903 with installed optional dependencies."""

    try:
        import datasets  # noqa: F401
    except ImportError:
        pytest.skip("datasets optional dependency is not installed")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec.from_tasks_dir(tasks)

    assert len(spec.train_dataset) == 1


def test_environment_reset_lets_rollout_generate_unique_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Guards PR #903 against same-task GRPO artifact overwrite."""

    _FakeRuntime.created_configs = []
    monkeypatch.setattr("benchflow.integrations.trl.spec.TaskRuntime", _FakeRuntime)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec(
        tasks_dir=tasks,
        bash_harness=BashHarnessConfig(jobs_dir=tmp_path / "jobs"),
    )

    env = spec.environment_factory()
    row = spec.train_dataset_rows[0]
    env.reset(**row)
    first = _FakeRuntime.created_configs[-1]
    env.reset(**row)
    second = _FakeRuntime.created_configs[-1]

    assert first.rollout_name is None
    assert second.rollout_name is None


def test_environment_public_tools_have_transformers_schema() -> None:
    """Guards PR #903 so TRL sees only intended callable tools."""

    transformers_utils = pytest.importorskip("transformers.utils")

    public_callables = [
        name
        for name in dir(BenchFlowRuntimeEnvironment)
        if not name.startswith("_")
        and name != "reset"
        and callable(getattr(BenchFlowRuntimeEnvironment, name))
    ]

    assert public_callables == ["run_bash", "submit"]
    for name in public_callables:
        schema = transformers_utils.get_json_schema(
            getattr(BenchFlowRuntimeEnvironment, name)
        )
        assert schema["function"]["name"] == name


def test_reward_func_returns_zero_without_environment(tmp_path: Path):
    """Guards PR #903: reward funcs are safe before environment plumbing."""

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec.from_tasks_dir(tasks)

    assert spec.reward_funcs[0](completions=["ok"]) == [0.0]


def test_create_trainer_missing_trl_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Guards PR #903: TRL is imported lazily by create_trainer."""

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec.from_tasks_dir(tasks)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "trl" or name.startswith("trl."):
            raise ImportError("blocked trl for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(BenchFlowOptionalDependencyError, match="create_trainer"):
        spec.create_trainer(model="dummy-model")


def test_environment_factory_and_reward_func_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Guards PR #903: environment_factory and reward_funcs match GRPO shape."""

    _FakeRuntime.created_configs = []
    monkeypatch.setattr("benchflow.integrations.trl.spec.TaskRuntime", _FakeRuntime)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    task = _make_task(tasks, "alpha", "Answer alpha")
    spec = BenchFlowSpec(
        BenchFlowSpecConfig(
            tasks_dir=tasks,
            bash_harness=BashHarnessConfig(jobs_dir=tmp_path / "jobs"),
        )
    )

    env = spec.environment_factory()
    assert isinstance(env, BenchFlowRuntimeEnvironment)
    row = spec.train_dataset_rows[0]
    reset_message = env.reset(**row)
    assert reset_message is None
    assert _FakeRuntime.created_configs[-1].task_path == task
    assert _FakeRuntime.created_configs[-1].jobs_dir == tmp_path / "jobs"
    assert env.run_bash("printf hi") == "hi"
    assert env.submit("done") == "submission recorded; reward=1"

    rewards = spec.reward_funcs[0](completions=["ok"], environments=[env])
    assert rewards == [1.0]

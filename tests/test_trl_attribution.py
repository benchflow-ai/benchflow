"""The TRL adapter follows the RL cookbook attribution rule.

Infrastructure failures (a sandbox that never started, a verifier crash on a
sandbox the policy never touched) are dropped: the reward function returns
None, which TRL leaves out of the group baseline with zero advantage. Every
failure the policy could have caused scores 0.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from benchflow.integrations.trl import (
    BashHarnessConfig,
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    bash_tool_schemas,
    benchflow_environment_reward,
)


def _make_task(parent: Path, name: str) -> Path:
    task = parent / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        f'version = "1.0"\n\n[task]\nname = "benchflow/{name}"\n\n'
        "[verifier]\ntimeout_sec = 60\n\n[agent]\ntimeout_sec = 60\n\n[environment]\n"
    )
    (task / "instruction.md").write_text("Answer.")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text("#!/usr/bin/env bash\necho 1 >/logs/verifier/reward.txt\n")
    test_sh.chmod(0o755)
    return task


@dataclass
class _Bash:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _Verified:
    reward: float | None
    rollout_dir: Path
    error: str | None = None
    verifier_error: str | None = None
    rewards: dict | None = None


class _Runtime:
    """A TaskRuntime stand-in whose start and verify outcomes are scripted."""

    start_error: ClassVar[Exception | None] = None
    start_gate: ClassVar[threading.Event | None] = None
    verify_outcome: ClassVar[Any] = None  # an exception, or a dict of _Verified fields
    verify_barrier: ClassVar[asyncio.Barrier | None] = None
    created: ClassVar[list[_Runtime]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rollout_dir = Path(config.jobs_dir) / f"rollout-{len(_Runtime.created)}"
        self.commands: list[str] = []
        self.closed = False
        self.verify_count = 0

    @classmethod
    async def create(cls, config: Any) -> _Runtime:
        if cls.start_gate is not None:
            await asyncio.to_thread(cls.start_gate.wait, 5)
        if cls.start_error is not None:
            raise cls.start_error
        runtime = cls(config)
        cls.created.append(runtime)
        return runtime

    async def bash(self, command: str, *, timeout_sec: int = 30) -> _Bash:
        self.commands.append(command)
        return _Bash(stdout="ok")

    async def verify(self) -> _Verified:
        self.verify_count += 1
        if _Runtime.verify_barrier is not None:
            await asyncio.wait_for(_Runtime.verify_barrier.wait(), timeout=5)
        outcome = _Runtime.verify_outcome
        if isinstance(outcome, Exception):
            raise outcome
        fields = {"reward": 1.0} if outcome is None else dict(outcome)
        return _Verified(rollout_dir=self.rollout_dir, **fields)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _Runtime.start_error = None
    _Runtime.start_gate = None
    _Runtime.verify_outcome = None
    _Runtime.verify_barrier = None
    _Runtime.created = []
    monkeypatch.setattr("benchflow.integrations.trl.spec.TaskRuntime", _Runtime)
    tasks = tmp_path / "tasks"
    _make_task(tasks, "alpha")

    def build(**harness: Any) -> BenchFlowSpec:
        return BenchFlowSpec(
            tasks_dir=tasks,
            bash_harness=BashHarnessConfig(jobs_dir=tmp_path / "jobs", **harness),
        )

    return build


def _reward(spec: BenchFlowSpec, envs: list[Any], **kwargs: Any) -> list[float | None]:
    return spec.reward_funcs[0](
        completions=["done"] * len(envs), environments=envs, **kwargs
    )


def test_sandbox_start_failure_drops_the_rollout_without_crashing_reset(spec) -> None:
    """TRL calls reset() inside its batch loop, so a start failure must not raise there."""

    _Runtime.start_error = RuntimeError("Daytona: no capacity")
    s = spec()
    env = s.environment_factory()
    env.reset(**s.train_dataset_rows[0])

    with pytest.raises(RuntimeError, match="dropped"):
        env.run_bash("ls")
    assert _reward(s, [env]) == [None]
    assert env.decision is not None
    assert env.decision.reason == "sandbox_start"
    assert env.reward is None


def test_background_start_returns_before_the_sandbox_is_up(spec) -> None:
    """reset() must not block TRL's sequential reset loop on sandbox start."""

    gate = threading.Event()
    _Runtime.start_gate = gate
    s = spec(background_start=True)
    env = s.environment_factory()

    env.reset(**s.train_dataset_rows[0])  # would hang for 5 s without background start
    assert _Runtime.created == []
    gate.set()
    assert env.run_bash("ls") == "ok"
    assert _reward(s, [env]) == [1.0]


def test_background_start_failure_is_dropped(spec) -> None:
    _Runtime.start_error = RuntimeError("image pull failed")
    s = spec(background_start=True)
    env = s.environment_factory()
    env.reset(**s.train_dataset_rows[0])

    assert _reward(s, [env]) == [None]
    assert env.decision.reason == "sandbox_start"


def test_verifier_crash_after_the_policy_acted_scores_zero(spec) -> None:
    _Runtime.verify_outcome = {
        "reward": None,
        "verifier_error": "verifier crashed: exit 1",
    }
    s = spec()
    env = s.environment_factory()
    env.reset(**s.train_dataset_rows[0])
    env.run_bash("rm -rf /workdir")

    assert _reward(s, [env]) == [0.0]
    assert env.decision.reason == "verifier_error"


def test_verifier_crash_on_an_untouched_sandbox_is_dropped(spec) -> None:
    _Runtime.verify_outcome = {
        "reward": None,
        "verifier_error": "verifier crashed: exit 1",
    }
    s = spec()
    env = s.environment_factory()
    env.reset(**s.train_dataset_rows[0])

    assert _reward(s, [env]) == [None]
    assert env.decision.reason == "verifier_crash_clean_run"


def test_verify_raising_scores_by_who_could_have_caused_it(spec) -> None:
    _Runtime.verify_outcome = RuntimeError("connection lost")
    s = spec()
    acted = s.environment_factory()
    acted.reset(**s.train_dataset_rows[0])
    acted.run_bash("ls")
    idle = s.environment_factory()
    idle.reset(**s.train_dataset_rows[0])

    assert _reward(s, [acted, idle]) == [0.0, None]
    assert acted.decision.reason == "verifier_error"
    assert idle.decision.reason == "verifier_crash_clean_run"
    assert all(runtime.closed for runtime in _Runtime.created)


def test_submit_is_policy_action(spec) -> None:
    """Writing the answer is an action: a verifier crash afterwards scores 0."""

    _Runtime.verify_outcome = {
        "reward": None,
        "verifier_error": "verifier crashed: exit 1",
    }
    s = spec()
    env = s.environment_factory()
    env.reset(**s.train_dataset_rows[0])

    assert env.submit("42") == "submission recorded; reward=0"
    assert _reward(s, [env]) == [0.0]


def test_open_rollouts_are_verified_concurrently(spec) -> None:
    """Each verify waits for all the others, so this passes only when they overlap."""

    s = spec()
    envs = [s.environment_factory() for _ in range(3)]
    for env in envs:
        env.reset(**s.train_dataset_rows[0])
    _Runtime.verify_barrier = asyncio.Barrier(3)

    assert _reward(s, envs) == [1.0, 1.0, 1.0]


def test_reward_function_logs_drops_with_fixed_keys(spec) -> None:
    metrics: dict[str, list[float]] = {}
    extra: dict[str, list] = {}
    s = spec()
    ok = s.environment_factory()
    ok.reset(**s.train_dataset_rows[0])
    _Runtime.start_error = RuntimeError("no capacity")
    lost = s.environment_factory()
    lost.reset(**s.train_dataset_rows[0])

    rewards = _reward(
        s,
        [ok, lost],
        log_metric=lambda name, value: metrics.setdefault(name, []).append(value),
        log_extra=lambda column, values: extra.setdefault(column, []).extend(values),
    )

    assert rewards == [1.0, None]
    assert metrics["benchflow/dropped_frac"] == [0.5]
    assert metrics["benchflow/drop/sandbox_start"] == [0.5]
    # Every reason is logged every batch, so TRL averages over the same steps.
    assert metrics["benchflow/drop/model_endpoint"] == [0.0]
    assert metrics["benchflow/zero/timeout"] == [0.0]
    assert extra["benchflow_reason"] == ["scored", "sandbox_start"]


def test_every_rollout_is_recorded_for_audit_including_drops(
    spec, tmp_path: Path
) -> None:
    s = spec()
    ok = s.environment_factory()
    ok.reset(**s.train_dataset_rows[0])
    ok.run_bash("cat data.csv")
    _Runtime.start_error = RuntimeError("no capacity")
    lost = s.environment_factory()
    lost.reset(**s.train_dataset_rows[0])

    prompt = [{"role": "user", "content": "Answer."}]
    completion = [
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]}
    ]
    s.reward_funcs[0](
        completions=[completion, completion],
        prompts=[prompt, prompt],
        environments=[ok, lost],
        trainer_state=type("State", (), {"global_step": 7})(),
    )

    lines = (tmp_path / "jobs" / "rollouts.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["reason"] for r in records] == ["scored", "sandbox_start"]
    assert records[1]["rollout_dir"] is None
    assert records[1]["messages"] == [*prompt, *completion]
    assert all(r["step"] == 7 for r in records)
    saved = json.loads(
        (Path(records[0]["rollout_dir"]) / "policy" / "messages.json").read_text()
    )
    assert saved["messages"] == [*prompt, *completion]
    assert saved["policy_acted"] is True


def test_reward_without_benchflow_environments_is_unchanged() -> None:
    """Guards PR #903's contract for rows without BenchFlow environments."""

    assert benchflow_environment_reward(["a", "b"], environments=None) == [0.0, 0.0]


def test_tool_schemas_match_what_trl_derives() -> None:
    """The evaluator must show a policy the tools it was trained with."""

    transformers_utils = pytest.importorskip("transformers.utils")
    derived = [
        transformers_utils.get_json_schema(getattr(BenchFlowRuntimeEnvironment, name))
        for name in ("run_bash", "submit")
    ]
    assert bash_tool_schemas() == derived

"""Regression tests for #378 — RuntimeConfig fields must flow through.

The public ``Runtime`` / ``bf.run(Agent, Environment, RuntimeConfig)`` API
advertises an artifact-oriented ``RuntimeResult`` with ``rollout_dir``, and
``RuntimeConfig`` exposes ``rollout_name`` and ``timeout`` knobs. Before the
fix:

* ``RuntimeResult.rollout_dir`` was always ``None`` — callers had to scan
  ``jobs_dir/`` to find the artifacts.
* ``RuntimeConfig.rollout_name`` was silently dropped — the rollout was
  always named ``<task>__<uuid>``.
* ``RuntimeConfig.timeout`` was silently dropped — only ``task.toml``'s
  ``[agent] timeout_sec`` reached the rollout.

These tests fail without the wiring fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"


class _FakeInner:
    """Stand-in sandbox: records lifecycle without doing real work."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def start(self, *a: Any, **kw: Any) -> None:
        self.started += 1

    async def stop(self, *a: Any, **kw: Any) -> None:
        self.stopped += 1

    async def exec(self, cmd: str, **_: Any) -> Any:
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    async def upload_file(self, *a: Any, **kw: Any) -> None: ...

    async def upload_dir(self, *a: Any, **kw: Any) -> None: ...


def _stub_run_result(rollout_name: str = "rt") -> Any:
    """Build a minimal RolloutResult so the runtime layer can finish."""
    from benchflow.models import RolloutResult

    return RolloutResult(
        task_name=TASK_PATH.name,
        rollout_name=rollout_name,
        rewards={"reward": 1.0},
        trajectory=[],
        agent="",
        agent_name="",
        model="",
        n_tool_calls=0,
        n_prompts=0,
        error=None,
        verifier_error=None,
        partial_trajectory=False,
        trajectory_source=None,
        started_at=None,
        finished_at=None,
    )


# ---------------------------------------------------------------------------
# RuntimeResult.rollout_dir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_result_rollout_dir_set_from_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeResult.rollout_dir must point at Rollout._rollout_dir (#378)."""
    inner = _FakeInner()
    env = Environment(inner=inner, task_path=TASK_PATH, sandbox="docker")
    agent = Agent(name="claude-agent-acp", model="claude-haiku-4-5-20251001")

    fake_dir = Path("/tmp/fake-rollout-dir-for-test-378")

    async def fake_run(self: Rollout) -> Any:
        # Simulate what Rollout.setup() would have written.
        self._rollout_dir = fake_dir
        return _stub_run_result()

    monkeypatch.setattr(Rollout, "run", fake_run)

    result = await Runtime(env, agent, RuntimeConfig()).execute()

    assert result.rollout_dir == fake_dir, (
        "RuntimeResult.rollout_dir was not lifted from the Rollout — callers "
        "still cannot find result.json without scanning jobs_dir (#378)."
    )


# ---------------------------------------------------------------------------
# RuntimeConfig.rollout_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_config_rollout_name_threaded_into_rollout_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeConfig.rollout_name must populate RolloutConfig.rollout_name (#378).

    Before the fix the runtime built RolloutConfig without ``rollout_name``
    so the user's explicit ``"explicit-runtime-rollout"`` was silently
    replaced by ``<task>__<uuid>``.
    """
    inner = _FakeInner()
    env = Environment(inner=inner, task_path=TASK_PATH, sandbox="docker")
    agent = Agent(name="claude-agent-acp", model="claude-haiku-4-5-20251001")

    captured: dict[str, Any] = {}
    real_create = Rollout.create

    @classmethod
    async def fake_create(cls, cfg: RolloutConfig) -> Rollout:  # type: ignore[misc]
        captured["rollout_name"] = cfg.rollout_name
        captured["timeout"] = cfg.timeout
        return await real_create(cfg)

    async def fake_run(self: Rollout) -> Any:
        return _stub_run_result()

    monkeypatch.setattr(Rollout, "create", fake_create)
    monkeypatch.setattr(Rollout, "run", fake_run)

    await Runtime(
        env,
        agent,
        RuntimeConfig(rollout_name="explicit-runtime-rollout", timeout=123),
    ).execute()

    assert captured["rollout_name"] == "explicit-runtime-rollout"
    assert captured["timeout"] == 123


# ---------------------------------------------------------------------------
# RolloutConfig.timeout overrides task default
# ---------------------------------------------------------------------------


def test_rollout_config_timeout_field_defaults_to_none() -> None:
    """Default keeps the per-task budget — only an explicit override wins."""
    cfg = RolloutConfig(task_path=TASK_PATH)
    assert cfg.timeout is None


@pytest.mark.asyncio
async def test_rollout_config_timeout_overrides_task_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout.setup must honour cfg.timeout instead of task.config.agent.timeout_sec.

    This is the seam by which ``RuntimeConfig.timeout`` actually shortens
    or extends the agent's wall-clock budget. Without it the field is
    accepted but does nothing (#378).
    """
    cfg = RolloutConfig(task_path=TASK_PATH, environment="docker", timeout=42)
    rollout = Rollout(cfg)
    rollout.use_prebuilt_env(_FakeInner())

    await rollout.setup()

    assert rollout._timeout == 42, (
        "RolloutConfig.timeout did not override the task default — "
        "RuntimeConfig.timeout cannot bind the agent budget (#378)."
    )


@pytest.mark.asyncio
async def test_rollout_config_timeout_none_keeps_task_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative side: omitting timeout still falls back to the task value."""
    cfg = RolloutConfig(task_path=TASK_PATH, environment="docker")  # timeout=None
    rollout = Rollout(cfg)
    rollout.use_prebuilt_env(_FakeInner())

    await rollout.setup()

    # The hello-world-task fixture's task.toml owns this number; we only
    # assert that the rollout picked it up rather than zero/None.
    from benchflow.task import Task

    expected = int(Task(TASK_PATH).config.agent.timeout_sec or 0)
    assert rollout._timeout == expected

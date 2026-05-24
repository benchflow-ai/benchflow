"""Runtime honours the live sandbox of a caller-supplied Environment (#388).

Before this fix, ``Runtime(env, agent).execute()`` built a brand-new
sandbox via ``Rollout.setup()`` and silently discarded ``env.inner``.
Callers could prepare state, upload fixtures, or start services on one
Environment and unknowingly evaluate inside a different sandbox.

These tests guard the wiring at three layers:

* ``Rollout.use_prebuilt_env`` marks the sandbox externally-owned.
* ``Rollout.setup`` reuses the injected sandbox instead of creating one.
* ``Rollout.cleanup`` skips stopping a sandbox the caller owns.
* ``Runtime.execute`` wires ``env.inner`` into the Rollout and starts an
  unstarted Environment exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"


class _FakeInner:
    """Records lifecycle calls so we can assert sandbox identity is preserved."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.uploads: list[tuple[str, str]] = []
        self.execs: list[str] = []

    async def start(self, *args: Any, **kwargs: Any) -> None:
        self.started += 1

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        self.stopped += 1

    async def exec(self, cmd: str, **kwargs: Any) -> Any:
        self.execs.append(cmd)
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    async def upload_file(self, src: str | Path, dst: str) -> None:
        self.uploads.append((str(src), dst))

    async def upload_dir(self, src: str | Path, dst: str, **kwargs: Any) -> None:
        self.uploads.append((str(src), dst))


def test_use_prebuilt_env_marks_externally_owned() -> None:
    """use_prebuilt_env stashes the sandbox and flips the ownership flag."""
    cfg = RolloutConfig(task_path=TASK_PATH)
    rollout = Rollout(cfg)
    inner = _FakeInner()

    rollout.use_prebuilt_env(inner)

    assert rollout.env is inner
    assert rollout._env_externally_owned is True


def test_use_prebuilt_env_rejects_none() -> None:
    """Passing None is a programmer error — fail loud, don't silently rebuild."""
    rollout = Rollout(RolloutConfig(task_path=TASK_PATH))
    with pytest.raises(ValueError, match="non-None"):
        rollout.use_prebuilt_env(None)


@pytest.mark.asyncio
async def test_setup_reuses_prebuilt_env_no_new_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup() must NOT call _create_environment when a sandbox is injected.

    This is the core of #388: the bug was setup() always calling
    _create_environment(cfg.environment, ...) and replacing self._env.
    """
    cfg = RolloutConfig(task_path=TASK_PATH, environment="docker")
    rollout = Rollout(cfg)
    inner = _FakeInner()
    rollout.use_prebuilt_env(inner)

    create_calls: list[Any] = []

    def fake_create_environment(*args: Any, **kwargs: Any) -> Any:
        create_calls.append((args, kwargs))
        return _FakeInner()

    monkeypatch.setattr(
        "benchflow.rollout._create_environment", fake_create_environment
    )

    await rollout.setup()

    assert create_calls == [], "setup() created a new sandbox despite pre-built env"
    assert rollout.env is inner, "setup() must keep the injected sandbox identity"


@pytest.mark.asyncio
async def test_setup_without_prebuilt_env_still_creates_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative case: no pre-built env => setup() must build one.

    Guards against the obvious over-correction where we accidentally
    skip env creation in the default path too.
    """
    cfg = RolloutConfig(task_path=TASK_PATH, environment="docker")
    rollout = Rollout(cfg)

    created = _FakeInner()
    create_calls: list[Any] = []

    def fake_create_environment(*args: Any, **kwargs: Any) -> Any:
        create_calls.append((args, kwargs))
        return created

    monkeypatch.setattr(
        "benchflow.rollout._create_environment", fake_create_environment
    )

    await rollout.setup()

    assert len(create_calls) == 1, "setup() must create a sandbox when none injected"
    assert rollout.env is created
    assert rollout._env_externally_owned is False


@pytest.mark.asyncio
async def test_cleanup_does_not_stop_externally_owned_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cleanup() must not stop a caller-owned sandbox — caller owns lifecycle."""
    cfg = RolloutConfig(task_path=TASK_PATH)
    rollout = Rollout(cfg)
    inner = _FakeInner()
    rollout.use_prebuilt_env(inner)

    # Skip noise paths that cleanup() runs through.
    rollout._provider_runtime = None
    rollout._usage_runtime = None

    await rollout.cleanup()

    assert inner.stopped == 0, "cleanup() stopped a sandbox the caller owns"


@pytest.mark.asyncio
async def test_cleanup_stops_internally_created_env() -> None:
    """Default ownership: cleanup() stops the sandbox it created."""
    cfg = RolloutConfig(task_path=TASK_PATH)
    rollout = Rollout(cfg)
    inner = _FakeInner()
    rollout._env = inner  # simulate internally-created env, owned by rollout
    rollout._provider_runtime = None
    rollout._usage_runtime = None

    await rollout.cleanup()

    assert inner.stopped == 1, "cleanup() must stop a sandbox it created"


@pytest.mark.asyncio
async def test_runtime_execute_uses_environment_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime.execute() pipes env.inner through to the Rollout.

    Direct reproduction of the #388 issue body: a caller passes a live
    Environment, Runtime must use it and start it exactly once. Before
    the fix, env.inner.start was never called and the rollout failed
    with "Unknown sandbox_type: <whatever>".
    """
    inner = _FakeInner()
    env = Environment(inner=inner, task_path=TASK_PATH, sandbox="not-a-real-sandbox")
    agent = Agent(name="claude-agent-acp", model="claude-haiku-4-5-20251001")

    captured: dict[str, Any] = {}

    async def fake_run(self: Rollout) -> Any:
        captured["rollout_env"] = self._env
        captured["externally_owned"] = self._env_externally_owned
        from benchflow.models import RolloutResult

        return RolloutResult(
            task_name=TASK_PATH.name,
            rollout_name="rt-388",
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

    monkeypatch.setattr(Rollout, "run", fake_run)

    result = await Runtime(env, agent, RuntimeConfig()).execute()

    assert captured["rollout_env"] is inner, "Rollout did not receive env.inner"
    assert captured["externally_owned"] is True
    assert inner.started == 1, "Runtime should start the Environment exactly once"
    assert env._started is True, "Environment._started must stay consistent"
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_runtime_execute_does_not_restart_already_started_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the caller already started the Environment, don't start it again.

    Most sandbox backends (e.g. daytona) are not idempotent under
    repeated start() — a second call would build a second container.
    """
    inner = _FakeInner()
    env = Environment(inner=inner, task_path=TASK_PATH, sandbox="docker")
    await env.start()  # caller starts it themselves
    assert inner.started == 1

    agent = Agent(name="claude-agent-acp", model="claude-haiku-4-5-20251001")

    async def fake_run(self: Rollout) -> Any:
        from benchflow.models import RolloutResult

        return RolloutResult(
            task_name=TASK_PATH.name,
            rollout_name="rt-388-restart",
            rewards={"reward": 0.0},
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

    monkeypatch.setattr(Rollout, "run", fake_run)

    await Runtime(env, agent, RuntimeConfig()).execute()

    assert inner.started == 1, "Runtime restarted an already-started Environment"

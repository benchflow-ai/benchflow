"""Tests for runtime.py — Agent, Environment, Runtime, RuntimeResult."""

from datetime import datetime
from pathlib import Path

from benchflow.runtime import (
    Agent,
    Environment,
    Runtime,
    RuntimeConfig,
    RuntimeResult,
    run,
)


def test_agent_basic() -> None:
    a = Agent(name="claude-agent-acp", model="claude-haiku-4-5-20251001")
    assert a.name == "claude-agent-acp"
    assert a.model == "claude-haiku-4-5-20251001"
    assert a.config is not None
    assert a.launch_cmd != ""
    assert "Agent(" in repr(a)


def test_agent_unknown() -> None:
    a = Agent(name="nonexistent-agent", model="some-model")
    assert a.config is None
    assert a.launch_cmd == "nonexistent-agent"


def test_agent_env_default_empty() -> None:
    a = Agent(name="gemini", model="gemini-3.1-flash-lite-preview")
    assert a.env == {}


def test_runtime_config_defaults() -> None:
    c = RuntimeConfig()
    assert c.sandbox_user == "agent"
    assert c.sandbox_setup_timeout == 120
    assert c.max_rounds == 10
    assert c.snapshot_policy == "none"
    assert c.reward_stream is True
    assert c.timeout == 900


def test_runtime_result_passed() -> None:
    r = RuntimeResult(
        task_name="test-task",
        trial_name="trial-1",
        reward=1.0,
        rewards={"reward": 1.0},
        n_tool_calls=5,
        error=None,
        verifier_error=None,
        trajectory=[],
    )
    assert r.passed is True
    assert r.verified is True


def test_runtime_result_failed() -> None:
    r = RuntimeResult(
        task_name="test-task",
        trial_name="trial-1",
        reward=0.0,
        rewards={"reward": 0.0},
        n_tool_calls=3,
        error=None,
        verifier_error=None,
        trajectory=[],
    )
    assert r.passed is False
    assert r.verified is True


def test_runtime_result_error() -> None:
    r = RuntimeResult(
        task_name="test-task",
        trial_name="trial-1",
        reward=None,
        rewards=None,
        n_tool_calls=0,
        error="Agent timed out",
        verifier_error=None,
        trajectory=[],
    )
    assert r.passed is False
    assert r.verified is False


def test_environment_from_task() -> None:
    """Environment.from_task creates a wrapper with correct metadata."""
    # Use the conformance task as a real task.toml source
    task_path = Path(__file__).parent / "conformance" / "acp_smoke"
    if not (task_path / "task.toml").exists():
        return  # skip if not available
    env = Environment.from_task(task_path, backend="daytona")
    assert env.backend == "daytona"
    assert env.task_path == task_path
    assert not env._started
    assert "acp_smoke" in repr(env)


def test_environment_context_manager_interface() -> None:
    """Environment has async context manager methods."""
    assert hasattr(Environment, "__aenter__")
    assert hasattr(Environment, "__aexit__")


def test_runtime_result_to_run_result() -> None:
    r = RuntimeResult(
        task_name="test-task",
        trial_name="trial-1",
        reward=1.0,
        rewards={"reward": 1.0},
        n_tool_calls=5,
        error=None,
        verifier_error=None,
        trajectory=[{"type": "tool_call"}],
        started_at=datetime(2026, 4, 18),
        finished_at=datetime(2026, 4, 18),
    )
    legacy = r.to_run_result()
    assert legacy.task_name == "test-task"
    assert legacy.rewards == {"reward": 1.0}
    assert legacy.n_tool_calls == 5


def test_runtime_init() -> None:
    agent = Agent(name="gemini", model="gemini-3.1-flash-lite-preview")
    task_path = Path(__file__).parent / "conformance" / "acp_smoke"
    if not (task_path / "task.toml").exists():
        return
    env = Environment.from_task(task_path, backend="daytona")
    runtime = Runtime(env, agent)
    assert runtime.agent.name == "gemini"
    assert runtime.env.backend == "daytona"
    assert runtime.config.sandbox_user == "agent"


def test_runtime_custom_config() -> None:
    agent = Agent(name="gemini", model="gemini-3.1-flash-lite-preview")
    task_path = Path(__file__).parent / "conformance" / "acp_smoke"
    if not (task_path / "task.toml").exists():
        return
    env = Environment.from_task(task_path, backend="daytona")
    config = RuntimeConfig(sandbox_user=None, timeout=1800, sandbox_setup_timeout=45)
    runtime = Runtime(env, agent, config)
    assert runtime.config.sandbox_user is None
    assert runtime.config.sandbox_setup_timeout == 45
    assert runtime.config.timeout == 1800


def test_run_function_exists() -> None:
    """bf.run() convenience function is importable and callable."""
    assert callable(run)

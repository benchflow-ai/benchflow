"""K1's round-14 review.

K1-R14-1: an environment the SDK made (``Environment.from_task``) and
``Runtime.execute`` started never got the runner intent (only a rollout that
makes its own sandbox set it), so a kept-service oracle there had no freezer.
K1-R14-2: verifier recovery's child names an "oracle" that never runs
solve.sh, and asked for the freezer and the start-up probe as if it did.
K1-R14-3: the interim wrapper ran agent commands through bash, which a
minimal container (an Alpine sidecar) may not have; the launcher now runs
them with POSIX sh, the shell a backend runs an exec with.

The tests here run on e80ee98e unchanged and fail there by behaviour, except
where noted.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "keep_services"
SRC = Path(__file__).parents[1] / "src" / "benchflow"


class _Started(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("agent, runner", [("oracle", "oracle"), ("nop", None)])
async def test_k1_r14_1_runtime_execute_names_the_runner_before_start(
    monkeypatch, agent, runner
):
    from benchflow.runtime import Agent, Environment, Runtime

    inner = SimpleNamespace(kept_service_runner=None)
    env = Environment(inner, FIXTURES / "keep-service-agent", "docker")
    seen = []

    async def start(self, force_build=False):
        seen.append(self.inner.kept_service_runner)
        raise _Started

    monkeypatch.setattr(Environment, "start", start)
    monkeypatch.setattr(Runtime, "_ready_for_cloud_metadata_block", lambda self: None)
    monkeypatch.setattr(
        "benchflow.rollout.Rollout.create", AsyncMock(return_value=MagicMock())
    )
    with pytest.raises(_Started):
        await Runtime(env, Agent(agent, model="none")).execute()
    assert seen == [runner]


def test_k1_r14_1_a_caller_started_environment_can_name_its_runner():
    """New API (not on e80ee98e): the explicit setting before start()."""
    from benchflow.runtime import Environment

    inner = SimpleNamespace(kept_service_runner=None)
    env = Environment(inner, FIXTURES / "keep-service-agent", "docker")
    env.prepare_kept_services("oracle")
    assert inner.kept_service_runner == "oracle"
    env.prepare_kept_services("claude-agent-acp", sandbox_user="agent")
    assert inner.kept_service_runner == "agent"
    env.prepare_kept_services("nop")
    assert inner.kept_service_runner is None


@pytest.mark.asyncio
async def test_k1_r14_2_a_verifier_only_rollout_asks_for_no_freezer(tmp_path):
    """Recovery's and regrade's child: no runner intent at setup, no probe at
    install. ``setup(verifier_only=True)`` alone says so, which is all
    ``bench eval regrade`` passes (on 8a67dbda its child, without K1's
    attribute, asked for the freezer as an oracle)."""
    from benchflow.rollout import Rollout, RolloutConfig

    env = MagicMock()
    env.kept_service_runner = None
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "oracle"
    planes.create_environment.return_value = env
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=FIXTURES / "keep-service-agent",
            agent="oracle",
            jobs_dir=tmp_path / "jobs",
            planes=planes,
        )
    )
    await rollout.setup(verifier_only=True)
    assert env.kept_service_runner is None
    assert rollout._needs_kept_freezer() is False


@pytest.mark.parametrize(
    "path", ["rollout/_verifier_recovery.py", "eval_regrade.py"], ids=str
)
def test_k1_r14_2_every_verifier_only_child_says_so_at_setup(path):
    """Recovery and regrade both build an "oracle" child that never runs
    solve.sh, and both mark it through setup's own flag."""
    tree = ast.parse((SRC / path).read_text())
    calls = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "child.setup"
    ]
    assert calls == ["child.setup(verifier_only=True)"]


@pytest.mark.asyncio
async def test_k1_r14_3_an_agent_command_needs_no_bash():
    """The command handed to the backend runs with POSIX sh."""
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    inner = SimpleNamespace(
        exec_in_service=AsyncMock(return_value="done"), agent_identity_switch=()
    )
    await SessionFactorySandbox(inner, {}, None).exec_in_service(
        "target", "printf ok", user="nobody"
    )
    command = inner.exec_in_service.await_args.args[1]
    assert "bash" not in command.split(), command
    assert "bash -c" not in command, command

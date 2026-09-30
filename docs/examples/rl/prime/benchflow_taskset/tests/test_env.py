"""BenchFlowEnv: the sandbox lives exactly as long as the episode, whatever happens.

The env is built through Verifiers' own loaders from the same config shape prime-rl
sends, with the scripted bridge behind it. A stand-in agent drives the real
in-process MCP tool server with Verifiers' MCP client (what the null harness's
chat program uses), then scores the trace the way a rollout does; or it crashes,
or it is cancelled. Every case checks that the sandbox was closed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import benchflow_taskset.session as session_module
import pytest
import verifiers.v1 as vf
from benchflow_taskset import BenchFlowEnv, BenchFlowInfraError, BenchFlowTask
from benchflow_taskset.taskset import HARNESS_MESSAGE
from conftest import env_config
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.harnesses.utils.mcp import mcp_client
from verifiers.v1.utils.loaders import resolve_env_config


@pytest.fixture(autouse=True)
def quick_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "REPLY_SLACK_SEC", 1.0)
    monkeypatch.setattr(session_module, "CLOSE_REPLY_SEC", 5.0)
    monkeypatch.setattr(session_module, "EXIT_WAIT_SEC", 2.0)
    monkeypatch.setattr(session_module, "TERM_WAIT_SEC", 10.0)


def load_env(world, **task_overrides) -> BenchFlowEnv:
    env = vf.load_environment(resolve_env_config(env_config(world, **task_overrides)))
    assert isinstance(env, BenchFlowEnv)
    return env


def first_task(env: BenchFlowEnv) -> BenchFlowTask:
    return next(iter(env.taskset))


def outcomes(world) -> list[dict]:
    path = world.tmp / "jobs" / "outcomes.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


class Agents:
    def __init__(self, agent) -> None:
        self.agent = agent


class ToolAgent:
    """Calls the tools over MCP like the chat program, then scores like a rollout."""

    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls
        self.tools_seen: list[dict] = []
        self.results: list[str] = []
        self.runs = 0

    async def run(self, task, *, tools=None, **_):
        self.runs += 1
        (server,) = tools.values()
        assert server.external and server.local
        async with mcp_client({"url": server.url}) as client:
            listed = await client.list_tools()
            self.tools_seen = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.input_schema,
                }
                for tool in listed.tools
            ]
            for name, arguments in self.calls:
                result = await client.call_tool(name, arguments)
                self.results.append("\n".join(block.text for block in result.content))
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type="BenchFlowTask", data=task.data),
        )
        if await task.submitted(trace):
            trace.stop("submitted")
        try:
            async with boundary(TaskError, "scoring"):
                await task.score(trace)
        except Exception as exc:
            trace.record_error(exc)
        else:
            trace.ok = True
        return trace


class CrashingAgent:
    async def run(self, task, *, tools=None, **_):
        raise RuntimeError("the harness crashed")


class HangingAgent:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, task, *, tools=None, **_):
        self.started.set()
        await asyncio.sleep(3600)


async def test_taskset_rows_carry_the_shared_harness_prompt(world) -> None:
    task = first_task(load_env(world))
    assert task.data.prompt == "Solve t1." + HARNESS_MESSAGE
    assert task.data.system_prompt is None
    assert task.data.task_dir == "/tasks/t1"
    assert task.key == "fam/t1"


async def test_an_episode_serves_the_trl_tools_and_scores_the_verifier(world) -> None:
    env = load_env(world)
    agent = ToolAgent([("run_bash", {"command": "ls"}), ("submit", {"answer": "42"})])
    await env.run(first_task(env), Agents(agent))
    by_name = {tool["name"]: tool for tool in agent.tools_seen}
    assert set(by_name) == {"run_bash", "submit"}
    assert by_name["run_bash"]["description"] == (
        "Run a bash command in the task sandbox and return its output (stdout and stderr)."
    )
    assert by_name["submit"]["schema"]["required"] == ["answer"]
    assert agent.results == ["out\n", "submission recorded"]
    assert world.ops() == ["start", "bash", "write", "verify", "close"]
    assert world.closed() == ["close"]
    (row,) = outcomes(world)
    assert row["decision"]["reward"] == 1.0 and row["stop"] == "submitted" and row["ok"]


async def test_a_drop_decision_fails_the_trace_and_still_closes(world) -> None:
    world.set(
        verify={
            "ok": True,
            "decision": {
                "reward": None,
                "dropped": True,
                "reason": "verifier_crash_clean_run",
                "detail": "x",
            },
        }
    )
    env = load_env(world)
    agent = ToolAgent([])
    await env.run(first_task(env), Agents(agent))
    (row,) = outcomes(world)
    assert row["ok"] is False and row["error"] == "BenchFlowInfraError"
    assert row["decision"]["dropped"] is True
    assert world.closed() == ["close"]


async def test_a_crashing_agent_still_closes_the_sandbox(world) -> None:
    env = load_env(world)
    with pytest.raises(RuntimeError, match="harness crashed"):
        await env.run(first_task(env), Agents(CrashingAgent()))
    assert world.closed() == ["close"]
    assert len(outcomes(world)) == 1


async def test_a_cancelled_episode_still_closes_the_sandbox(world) -> None:
    # prime-rl cancels episodes (stale groups, shutdown); so does the episode timeout.
    env = load_env(world)
    agent = HangingAgent()
    episode = asyncio.create_task(env.run(first_task(env), Agents(agent)))
    await asyncio.wait_for(agent.started.wait(), 30)
    episode.cancel()
    with pytest.raises(asyncio.CancelledError):
        await episode
    assert world.closed() == ["close"]


async def test_a_sandbox_that_never_starts_is_dropped_before_the_agent_runs(
    world,
) -> None:
    world.set(
        start={
            "ok": False,
            "error": "quota",
            "decision": {
                "reward": None,
                "dropped": True,
                "reason": "sandbox_start",
                "detail": "quota",
            },
        }
    )
    env = load_env(world)
    agent = ToolAgent([])
    with pytest.raises(BenchFlowInfraError, match="sandbox_start"):
        await env.run(first_task(env), Agents(agent))
    assert agent.runs == 0
    assert world.closed() == ["close"]
    (row,) = outcomes(world)
    assert row["decision"]["reason"] == "sandbox_start"


async def test_a_bridge_that_cannot_start_is_dropped(world) -> None:
    env = load_env(world)
    task = first_task(env)
    task.config = task.config.model_copy(
        update={"benchflow_python": str(world.tmp / "no-such-python")}
    )
    with pytest.raises(BenchFlowInfraError, match="sandbox_start"):
        await env.run(task, Agents(ToolAgent([])))
    (row,) = outcomes(world)
    assert row["decision"]["reason"] == "sandbox_start"


async def test_the_sandbox_slot_is_released_after_every_episode(world) -> None:
    env = load_env(world, max_sandboxes=1)
    for _ in range(2):
        await asyncio.wait_for(env.run(first_task(env), Agents(ToolAgent([]))), 30)
    with pytest.raises(RuntimeError):
        await env.run(first_task(env), Agents(CrashingAgent()))
    await asyncio.wait_for(env.run(first_task(env), Agents(ToolAgent([]))), 30)
    assert world.closed() == ["close"] * 4


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ({"harness": {"id": "bash"}}, "executes commands"),
        ({"runtime": {"type": "docker"}}, "subprocess runtime"),
        ({"retries": {"max_retries": 1}}, "env.retries"),
    ],
)
def test_unsafe_seats_are_refused(world, agent, message) -> None:
    config = env_config(world)
    config["agent"] = agent
    with pytest.raises(ValueError, match=message):
        vf.load_environment(resolve_env_config(config))


def test_the_taskset_id_selects_this_env_with_a_safe_default_seat(world) -> None:
    config = resolve_env_config(env_config(world))
    assert config.agent.harness.id == "null"
    assert config.agent.runtime.type == "subprocess"
    assert config.agent.max_turns == 10
    assert Path(config.taskset.task.benchflow_python).name == "fake-benchflow-python"

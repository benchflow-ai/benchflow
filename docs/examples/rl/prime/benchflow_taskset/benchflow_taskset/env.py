"""The env that runs BenchFlow tasks: one sandbox per episode, owned from start to verdict.

A Verifiers rollout tears its tool servers down before it scores, so the sandbox
cannot live in a toolset. This env holds it for the whole episode instead:

1. take a machine-wide sandbox slot, start the bridge, start the sandbox;
2. serve ``run_bash`` and ``submit`` in-process and run the agent with them
   (``agents.agent.run(task, tools=...)``, the public way to lend live tools);
3. the task's ``@vf.reward`` runs BenchFlow's verifier while the sandbox is alive;
4. close the sandbox, whatever happened, and log the outcome.

The seat defaults to the ``null`` harness (a chat loop whose only tools are the
MCP ones) in the ``subprocess`` runtime: the harness only talks to the model, and
every command runs in the BenchFlow sandbox. Harnesses that execute commands
themselves (``bash``, ``codex``, ...) are refused, since those commands would run
on the trainer's machine.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.configs.env import TimeoutConfig
from verifiers.v1.harnesses.null import NullHarnessConfig
from verifiers.v1.mcp import SharedToolServer

from benchflow_taskset.session import (
    BenchFlowSession,
    Bridge,
    BridgeError,
    SandboxStartError,
)
from benchflow_taskset.slots import sandbox_slot
from benchflow_taskset.taskset import MAX_TURNS, BenchFlowInfraError, BenchFlowTask
from benchflow_taskset.tools import serve_tools

# Tools are advertised under their bare names (`run_bash`, `submit`), the shape of
# BenchFlow's TRL harness: the chat program prefixes a tool only when its server has a name.
TOOL_SERVER_NAME = ""


class BenchFlowEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(
        harness=NullHarnessConfig(id="null"),
        runtime=vf.SubprocessConfig(),
        max_turns=MAX_TURNS,
    )
    """The policy's seat: its chat loop runs locally; its commands run in the sandbox."""
    timeout: TimeoutConfig = TimeoutConfig(episode=3600.0)
    """A last-resort deadline for a stuck episode (dropped and counted as an error).
    The policy's own limits are ``max_turns`` and ``agent_budget_sec``, which score."""


class BenchFlowEnv(vf.Env[BenchFlowEnvConfig]):
    def __init__(self, config: BenchFlowEnvConfig) -> None:
        super().__init__(config)
        harness_config = config.agent_harnesses()["agent"]
        harness = vf.load_harness(harness_config)
        if harness.EXECUTES_CODE:
            raise ValueError(
                f"harness {harness_config.id!r} executes commands where it runs (this "
                "machine), not in the BenchFlow sandbox; use --env.agent.harness.id null"
            )
        if not harness.SUPPORTS_MCP:
            raise ValueError(
                f"harness {harness_config.id!r} cannot use MCP tools; use null"
            )
        if not isinstance(config.agent.runtime, vf.SubprocessConfig):
            raise ValueError(
                "run the harness in the subprocess runtime (--env.agent.runtime.type "
                "subprocess): it only talks to the model, and a remote runtime could "
                "not reach the tool server on 127.0.0.1"
            )
        if config.agent.retries.max_retries:
            raise ValueError(
                "--env.agent.retries would rerun the policy in a sandbox it already "
                "changed; use --env.retries, which reruns the whole episode"
            )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, BenchFlowTask):
            raise TypeError(
                f"BenchFlowEnv runs BenchFlow tasks, got {type(task).__name__}"
            )
        cfg = task.config
        rollout_name = f"{task.data.name}-{uuid.uuid4().hex[:10]}"
        jobs_dir = Path(cfg.jobs_dir).expanduser().resolve()
        outcome: dict[str, Any] = {
            "task": task.data.name,
            "rollout": rollout_name,
            "started_at": time.time(),
        }
        verify_timeout = cfg.verify_timeout_sec
        if task.data.verifier_timeout_sec:
            verify_timeout = max(
                verify_timeout, float(task.data.verifier_timeout_sec) + 300.0
            )
        try:
            async with sandbox_slot(cfg.slots_dir, cfg.max_sandboxes):
                session = BenchFlowSession(
                    Bridge(
                        cfg.benchflow_python,
                        idle_timeout_sec=cfg.idle_timeout_sec,
                        log_path=jobs_dir / "bridge-logs" / f"{rollout_name}.log",
                        env_passthrough=tuple(cfg.bridge_env),
                    ),
                    task_dir=task.data.task_dir,
                    rollout_name=rollout_name,
                    environment=cfg.sandbox,
                    sandbox_user=cfg.sandbox_user,
                    jobs_dir=str(jobs_dir),
                    bash_timeout_sec=cfg.bash_timeout_sec,
                    max_output_chars=cfg.max_output_chars,
                    submit_path=cfg.submit_path,
                    agent_budget_sec=cfg.agent_budget_sec,
                    sandbox_setup_timeout_sec=cfg.sandbox_setup_timeout_sec,
                    verify_timeout_sec=verify_timeout,
                )
                try:
                    try:
                        await session.start()
                    except SandboxStartError as exc:
                        outcome["decision"] = exc.decision or {
                            "reward": None,
                            "dropped": True,
                            "reason": "sandbox_start",
                            "detail": str(exc)[:500],
                        }
                        raise BenchFlowInfraError(
                            f"benchflow drop (sandbox_start): {exc}"
                        ) from exc
                    except BridgeError as exc:
                        outcome["decision"] = {
                            "reward": None,
                            "dropped": True,
                            "reason": "sandbox_start",
                            "detail": str(exc)[:500],
                        }
                        raise BenchFlowInfraError(
                            f"benchflow drop (sandbox_start): {exc}"
                        ) from exc
                    outcome["sandbox_ready_at"] = time.time()
                    async with serve_tools(session) as url:
                        bound = task.bind(session)
                        session.mark_agent_start()
                        trace = await agents.agent.run(
                            bound,
                            tools={
                                TOOL_SERVER_NAME: SharedToolServer(
                                    url=url, local=True, external=True
                                )
                            },
                        )
                    outcome.update(
                        {
                            "decision": session.decision,
                            "stop": trace.stop_condition,
                            "ok": trace.ok,
                            "error": trace.last_error.type
                            if trace.last_error
                            else None,
                            "turns": trace.num_turns,
                            "stats": session.stats,
                            "submitted": session.submitted,
                            "policy_acted": session.policy_acted,
                            "infra_error": session.infra_error,
                            "rollout_dir": session.rollout_dir,
                        }
                    )
                    if session.decision is None and trace.ok:
                        # Nothing scored the episode (it cannot happen through the task's
                        # own reward); never let it train as if it had.
                        raise BenchFlowInfraError(
                            "the episode ended without a BenchFlow verdict"
                        )
                finally:
                    await session.close()
        finally:
            outcome["ended_at"] = time.time()
            self._log_outcome(cfg, jobs_dir, outcome)

    @staticmethod
    def _log_outcome(cfg: Any, jobs_dir: Path, outcome: dict[str, Any]) -> None:
        path = (
            Path(cfg.outcomes_path).expanduser()
            if cfg.outcomes_path
            else jobs_dir / "outcomes.jsonl"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = (json.dumps(outcome, default=str) + "\n").encode()
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass

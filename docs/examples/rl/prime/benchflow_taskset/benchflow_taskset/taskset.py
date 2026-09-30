"""BenchFlow task folders as a Verifiers v1 taskset.

Each task is a BenchFlow task package (``task.md``, or legacy ``task.toml`` +
``instruction.md``). This module never imports BenchFlow: listing the tasks and
running them both go through ``bridge.py`` in a BenchFlow venv (see
``session.py``). It runs where prime-rl's orchestrator loads the taskset, and in
the env-server workers, which rebuild each task from its data.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field

import verifiers.v1 as vf

from benchflow_taskset.session import bridge_environment

if TYPE_CHECKING:
    from benchflow_taskset.session import BenchFlowSession

BRIDGE_SCRIPT = Path(__file__).with_name("bridge.py")

DEFAULT_SYSTEM_PROMPT = (
    "You are solving a task in a Linux sandbox. Use the run_bash tool to run shell "
    "commands in the task's working directory; each call starts a fresh shell there. "
    "When you are done, call the submit tool."
)


class BenchFlowInfraError(vf.SandboxError):
    """An infrastructure failure the policy could not have caused. The trace is
    marked failed, so prime-rl leaves it out of the batch and counts it."""


class BenchFlowTaskConfig(vf.TaskConfig):
    """How every BenchFlow task runs (``--env.taskset.task.*``)."""

    benchflow_python: str = Field(default_factory=lambda: os.environ.get("BENCHFLOW_PYTHON", "python3"))
    """Python of a BenchFlow venv (BenchFlow cannot share Verifiers' venv). Default:
    ``$BENCHFLOW_PYTHON``."""
    sandbox: str = "daytona"
    """BenchFlow sandbox backend for the task (``daytona`` or ``docker``)."""
    sandbox_user: str | None = "agent"
    """The sandbox user commands run as (``None``: root)."""
    jobs_dir: str = "jobs/benchflow-taskset"
    """Where BenchFlow writes each episode's rollout folder, and ``outcomes.jsonl``."""
    bash_timeout_sec: int = Field(60, ge=1)
    """Per-command limit. A command that runs out of time is the policy's: the model
    sees a timeout message and the episode goes on."""
    max_output_chars: int = Field(4096, ge=256)
    """Longest tool output the model sees (head and tail kept)."""
    submit_path: str | None = "/workdir/answer.txt"
    """Where ``submit(answer)`` writes a non-empty answer (``None``: nowhere)."""
    agent_budget_sec: float | None = Field(900.0, gt=0)
    """Wall-clock budget for the policy. When it runs out the episode stops and is
    verified as it stands: a budget stop scores like any other, never dropped."""
    sandbox_setup_timeout_sec: int = Field(300, ge=30)
    verify_timeout_sec: float = Field(900.0, gt=0)
    """How long to wait for the verifier's decision (BenchFlow enforces the task's
    own verifier timeout inside)."""
    idle_timeout_sec: float = Field(1800.0, gt=0)
    """The bridge closes the sandbox after this long without a request."""
    max_sandboxes: int = Field(16, ge=1)
    """Live sandboxes at once, across every env-server worker on this machine."""
    slots_dir: str = "/tmp/benchflow-taskset-slots"
    bridge_env: list[str] = Field(default_factory=list)
    """Extra environment variables the bridge may inherit, besides the Daytona and
    BenchFlow ones (model keys never need to)."""
    outcomes_path: str | None = None
    """One JSON line per episode: reward or drop and why. Default:
    ``<jobs_dir>/outcomes.jsonl``."""


class BenchFlowConfig(vf.TasksetConfig):
    """Which BenchFlow tasks to load (``--env.taskset.*``)."""

    tasks_dir: str = Field(default_factory=lambda: os.environ.get("BENCHFLOW_TASKS_DIR", ""))
    """A folder of BenchFlow task folders, or one task folder."""
    include: list[str] = Field(default_factory=list)
    """Task folder names to keep (empty: all)."""
    exclude: list[str] = Field(default_factory=list)
    task: BenchFlowTaskConfig = BenchFlowTaskConfig()


class BenchFlowData(vf.TaskData):
    task_dir: str
    """Absolute path of the task folder on this machine."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier_timeout_sec: float | None = None


class BenchFlowTask(vf.Task[BenchFlowData, vf.State, BenchFlowTaskConfig]):
    """One BenchFlow task. Its episode runs under ``BenchFlowEnv``, which binds a live
    ``BenchFlowSession`` (the sandbox) to a copy of the task; the stops and the
    reward below read that session."""

    session: BenchFlowSession | None = None

    @property
    def key(self) -> str:
        # The folder path differs between machines; the task's own name does not.
        return self.data.id or self.data.name or self.hash

    def bind(self, session: BenchFlowSession) -> Self:
        clone = copy.copy(self)
        clone.session = session
        return clone

    def _session(self) -> BenchFlowSession:
        if self.session is None:
            raise RuntimeError(
                "a BenchFlow task runs under the benchflow-taskset env (BenchFlowEnv), "
                "which owns its sandbox; do not pair this taskset with another env"
            )
        return self.session

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        """The policy called ``submit``."""
        return self._session().submitted

    @vf.stop
    async def time_budget(self, trace: vf.Trace) -> bool:
        """The policy's wall-clock budget ran out: stop and score what it left."""
        return self._session().over_budget()

    @vf.stop
    async def bridge_failed(self, trace: vf.Trace) -> bool:
        """The bridge process failed mid-episode: stop; scoring drops the episode."""
        return self._session().infra_error is not None

    @vf.reward(weight=1.0)
    async def benchflow(self, trace: vf.Trace) -> float:
        """BenchFlow's verifier reward, or a drop for an infrastructure failure."""
        session = self._session()
        from benchflow_taskset.session import BridgeError

        try:
            decision = await session.verify()
        except BridgeError as exc:
            raise BenchFlowInfraError(f"benchflow bridge failed: {exc}") from exc
        trace.info["benchflow"] = {
            "decision": decision,
            "result": session.verify_result,
            "stats": dict(session.stats),
            "policy_acted": session.policy_acted,
            "submitted": session.submitted,
            "rollout_dir": session.rollout_dir,
        }
        trace.record_metric("benchflow_submitted", float(session.submitted))
        trace.record_metric("benchflow_policy_acted", float(session.policy_acted))
        trace.record_metric("benchflow_bash_calls", float(session.stats["bash_calls"]))
        trace.record_metric("benchflow_bash_timeouts", float(session.stats["bash_timeouts"]))
        trace.record_metric("benchflow_exec_errors", float(session.stats["exec_errors"]))
        if decision.get("dropped") or decision.get("reward") is None:
            raise BenchFlowInfraError(
                f"benchflow drop ({decision.get('reason')}): {decision.get('detail')}"
            )
        return float(decision["reward"])


def list_tasks(config: BenchFlowConfig) -> list[dict[str, Any]]:
    """Ask the BenchFlow venv for the tasks under ``config.tasks_dir``."""
    if not config.tasks_dir:
        raise ValueError("set --env.taskset.tasks-dir (or $BENCHFLOW_TASKS_DIR) to a folder of BenchFlow tasks")
    command = [config.task.benchflow_python, str(BRIDGE_SCRIPT), "tasks", "--tasks-dir", config.tasks_dir]
    for name in config.include:
        command += ["--include", name]
    for name in config.exclude:
        command += ["--exclude", name]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=bridge_environment(tuple(config.task.bridge_env)),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"cannot run the BenchFlow venv's Python {config.task.benchflow_python!r}; "
            "set --env.taskset.task.benchflow-python or $BENCHFLOW_PYTHON"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"listing BenchFlow tasks failed: {result.stderr.strip()[-2000:]}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no BenchFlow tasks under {config.tasks_dir}")
    return rows


class BenchFlowTaskset(vf.Taskset[BenchFlowTask, BenchFlowConfig]):
    def load(self) -> list[BenchFlowTask]:
        tasks = []
        for row in list_tasks(self.config):
            data = BenchFlowData(
                id=row.get("id") or row["name"],
                name=row["name"],
                prompt=row["prompt"],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                task_dir=row["task_dir"],
                metadata=row.get("metadata") or {},
                verifier_timeout_sec=row.get("verifier_timeout_sec"),
            )
            tasks.append(BenchFlowTask(data, self.config.task))
        return tasks

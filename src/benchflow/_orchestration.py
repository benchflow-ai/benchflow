"""Trial-lifecycle helpers — pure, free functions.

Extracted from ``sdk.py`` (PLAN_V2_impl §13). These helpers were
originally staticmethods on ``SDK`` that ``trial.py`` back-edged into via
late imports. They live here so:

- ``Trial`` (canonical orchestrator) imports them at module level — no
  back-edge, no cycle.
- ``Job``, ``Runtime``, ``skill_eval``, and any future
  ``experimental/<orchestrator>/`` can import them without pulling in
  Trial's full sandbox/agent/ACP graph.

This file is a flat module, not a subpackage. Promote to
``orchestration/`` only when (a) a third orchestrator graduates from
experimental, OR (b) this file exceeds the 400-LOC per-file budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow.task import Task
from benchflow.verifier import Verifier
from benchflow.contracts.paths import TrialPaths
from benchflow.results import TrialResult, TrajectorySource
from benchflow.sandbox.verifier_harden import harden_before_verify

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000  # max chars for diagnostic stdout/stderr in logs


def _write_rewards_jsonl(
    trial_dir: Path,
    rewards: dict | None,
    finished_at: datetime,
) -> None:
    """Write rewards.jsonl — one JSON line per reward event.

    Emits rubric items (if present) as type="process" lines, then the
    terminal reward as the final type="terminal" line.  Schema is
    ORS-reward-signal compatible: one streamed event per line.

    Rubric format in rewards dict::

        {"reward": 0.75, "rubric": [
            {"name": "file_exists", "score": 1.0, "weight": 1.0},
            {"name": "content_correct", "score": 0.5, "weight": 1.0}
        ]}
    """
    if not rewards:
        return
    events: list[dict] = []
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            events.append(
                {
                    "ts": finished_at.isoformat(),
                    "type": "process",
                    "source": "verifier_rubric",
                    "value": item.get("score", 0.0),
                    "tag": item.get("name", f"rubric_{i}"),
                    "step_index": i,
                    "meta": {
                        k: v for k, v in item.items() if k not in ("score", "name")
                    },
                }
            )
    scalar = rewards.get("reward")
    if scalar is not None:
        non_event_keys = {"reward", "rubric"}
        events.append(
            {
                "ts": finished_at.isoformat(),
                "type": "terminal",
                "source": "verifier",
                "value": scalar,
                "tag": "reward",
                "step_index": None,
                "meta": {k: v for k, v in rewards.items() if k not in non_event_keys},
            }
        )
    if events:
        path = trial_dir / "rewards.jsonl"
        path.write_text("\n".join(json.dumps(e, default=str) for e in events) + "\n")


def init_trial(
    task_path: Path,
    job_name: str | None,
    trial_name: str | None,
    jobs_dir: str | Path,
) -> tuple[Task, Path, TrialPaths, datetime, str, str]:
    """Set up trial directory tree and return core trial objects."""
    from uuid import uuid4

    task = Task(task_path)
    job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    trial_name = trial_name or f"{task_path.name}__{uuid4().hex[:8]}"
    trial_dir = Path(jobs_dir) / job_name / trial_name
    trial_paths = TrialPaths(trial_dir)
    started_at = datetime.now()
    # Pre-create trial directory tree so Docker doesn't create them as root.
    trial_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("agent", "verifier", "artifacts", "trajectory"):
        (trial_dir / subdir).mkdir(exist_ok=True)
    return task, trial_dir, trial_paths, started_at, job_name, trial_name


def write_trial_config(
    trial_dir: Path,
    *,
    task_path: Path,
    agent: str,
    model: str | None,
    environment: str,
    skills_dir: str | Path | None,
    sandbox_user: str | None,
    context_root: str | Path | None,
    sandbox_locked_paths: list[str] | None = None,
    timeout: int,
    started_at: datetime,
    agent_env: dict[str, str],
) -> None:
    """Write config.json to trial_dir with secrets filtered out."""
    _secret_substrings = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS")
    recorded_env = {
        k: v
        for k, v in agent_env.items()
        if not any(s in k.upper() for s in _secret_substrings)
    }
    config_data = {
        "task_path": str(task_path),
        "agent": agent,
        "model": model,
        "environment": environment,
        "skills_dir": str(skills_dir) if skills_dir else None,
        "sandbox_user": sandbox_user,
        "sandbox_locked_paths": sandbox_locked_paths,
        "context_root": str(context_root) if context_root else None,
        "timeout_sec": timeout,
        "started_at": str(started_at),
        "agent_env": recorded_env,
    }
    (trial_dir / "config.json").write_text(json.dumps(config_data, indent=2))


def _coerce_int(value: object) -> int | None:
    """Best-effort int coercion for usage fields. None on missing/invalid."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _coerce_float(value: object) -> float | None:
    """Best-effort float coercion for cost_usd. None on missing/invalid/NaN."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    return None


def read_usage_sidecar(path: Path) -> dict[str, int | float] | None:
    """Read an agent's $BENCHFLOW_USAGE_PATH JSON, return parsed dict or None.

    Honest about partial reports: missing fields stay missing (caller projects
    to ``None``). Malformed JSON or non-dict payloads return None — the trial
    still finishes; the result.json carries explicit nulls.
    """
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Usage sidecar %s unreadable: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("Usage sidecar %s is not a dict; ignored", path)
        return None
    return raw


# Sandbox path where shims write the usage sidecar; matches the contract
# in tests/test_byoa_shim_contract_snapshot.py + docs/byoa.md.
SANDBOX_USAGE_PATH = "/tmp/benchflow_usage.json"


def build_result(
    trial_dir: Path,
    *,
    task_name: str,
    trial_name: str,
    agent: str,
    agent_name: str,
    model: str,
    n_tool_calls: int,
    prompts: list[str],
    error: str | None,
    verifier_error: str | None,
    trajectory: list[dict],
    partial_trajectory: bool,
    trajectory_source: TrajectorySource | None = None,
    rewards: dict | None,
    started_at: datetime,
    timing: dict[str, float],
    usage: dict[str, int | float] | None = None,
) -> TrialResult:
    """Build TrialResult and write result.json, timing.json, prompts.json, trajectory.

    *usage* is the optional cost/token report (PLAN_V2_byoa.md PR9). Populated
    upstream from one of: a BYOA shim's ``$BENCHFLOW_USAGE_PATH`` sidecar,
    an OTel collector (PR10), or future per-agent runners. Missing fields
    stay ``None`` — benchflow never pro-rata-computes cost.
    """
    finished_at = datetime.now()
    usage = usage or {}
    result = TrialResult(
        task_name=task_name,
        trial_name=trial_name,
        rewards=rewards,
        trajectory=trajectory,
        agent=agent,
        agent_name=agent_name,
        model=model,
        n_tool_calls=n_tool_calls,
        n_prompts=len(prompts),
        error=error,
        verifier_error=verifier_error,
        partial_trajectory=partial_trajectory,
        trajectory_source=trajectory_source,
        input_tokens=_coerce_int(usage.get("input_tokens")),
        output_tokens=_coerce_int(usage.get("output_tokens")),
        cache_tokens=_coerce_int(usage.get("cache_tokens")),
        cost_usd=_coerce_float(usage.get("cost_usd")),
        started_at=started_at,
        finished_at=finished_at,
    )
    # Finalize timing — use the locals (TrialResult fields are typed
    # datetime | None and would need narrowing)
    timing["total"] = (finished_at - started_at).total_seconds()
    timing = {k: round(v, 1) for k, v in timing.items()}
    # Save trajectory
    traj_dir = trial_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in trajectory)
    )
    # Save result.json, prompts.json, timing.json
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": result.task_name,
                "trial_name": result.trial_name,
                "rewards": result.rewards,
                "agent": result.agent,
                "agent_name": result.agent_name,
                "model": result.model,
                "n_tool_calls": result.n_tool_calls,
                "n_prompts": result.n_prompts,
                "error": result.error,
                "verifier_error": result.verifier_error,
                "partial_trajectory": result.partial_trajectory,
                "trajectory_source": result.trajectory_source,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cache_tokens": result.cache_tokens,
                "cost_usd": result.cost_usd,
                "started_at": str(result.started_at),
                "finished_at": str(result.finished_at),
                "timing": timing,
            },
            indent=2,
        )
    )
    (trial_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (trial_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    _write_rewards_jsonl(trial_dir, rewards, finished_at)
    return result


def resolve_prompts(
    task_path: Path, prompts: list[str | None] | None
) -> list[str]:
    """Read instruction.md and resolve prompt list."""
    instruction_path = task_path / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Task missing instruction.md: {task_path}")
    instruction = instruction_path.read_text().strip()
    if prompts is None:
        return [instruction]
    return [p if p is not None else instruction for p in prompts]


async def start_env_and_upload(env, task_path: Path, timing: dict) -> None:
    """Start environment and upload task files."""
    logger.info(f"Starting environment: {task_path.name}")
    t0 = datetime.now()
    await env.start(force_build=False)
    timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    if (task_path / "instruction.md").exists():
        await env.upload_file(task_path / "instruction.md", "/instruction.md")
    if (task_path / "solution").is_dir():
        await env.upload_dir(task_path / "solution", "/solution")


async def run_oracle(
    env, task_path: Path, timeout: int, sandbox_user: str | None = None
) -> tuple[list[dict], str]:
    """Run oracle mode (solution/solve.sh), return (trajectory, agent_name)."""
    logger.info("Oracle mode: running solution/solve.sh")
    if not (task_path / "solution" / "solve.sh").exists():
        raise FileNotFoundError(f"Oracle requires solution/solve.sh: {task_path}")
    if sandbox_user:
        oracle_cmd = "DEBIAN_FRONTEND=noninteractive bash /solution/solve.sh"
        cmd = (
            f"su -s /bin/bash {shlex.quote(sandbox_user)} "
            f"-c {shlex.quote(oracle_cmd)}"
        )
    else:
        cmd = "bash /solution/solve.sh"
    result = await env.exec(
        f"{cmd} > /logs/agent/oracle.txt 2>&1",
        env={"DEBIAN_FRONTEND": "noninteractive"},
        timeout_sec=timeout,
    )
    if result.return_code != 0:
        logger.warning(f"Oracle solve.sh exited with rc={result.return_code}")
    preview = await env.exec(
        f"tail -c {shlex.quote(str(_DIAG_TRUNCATE))} /logs/agent/oracle.txt 2>/dev/null || true",
        user="root",
        timeout_sec=10,
    )
    trajectory = [
        {
            "type": "oracle",
            "command": "solution/solve.sh",
            "return_code": result.return_code,
            "stdout": (preview.stdout or "")[:_DIAG_TRUNCATE],
        }
    ]
    return trajectory, "oracle"


async def verify_with_harden(
    env,
    task: Task,
    trial_paths: TrialPaths,
    timing: dict,
    sandbox_user: str | None = None,
    workspace: str | None = None,
) -> tuple[dict | None, str | None]:
    """Run verifier with pre-verification hardening."""
    trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    await harden_before_verify(env, task, sandbox_user, workspace=workspace)
    logger.info("Running verifier...")
    t0 = datetime.now()
    verifier_error = None
    try:
        verifier = Verifier(task=task, trial_paths=trial_paths, environment=env)
        verifier_result = await asyncio.wait_for(
            verifier.verify(),
            timeout=task.config.verifier.timeout_sec,
        )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = verifier_result.rewards
        logger.info(f"Rewards: {rewards}")
    except TimeoutError:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
        verifier_error = (
            f"verifier timed out after {task.config.verifier.timeout_sec}s"
        )
        rewards = None
        logger.error(verifier_error)
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
        verifier_error = f"verifier crashed: {e}"
        rewards = None
        logger.error(verifier_error)
    return rewards, verifier_error

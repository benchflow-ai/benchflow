"""Private rollout lifecycle helpers.

These functions are the rollout-native home for logic that used to live on
``SDK`` static/private methods. Keeping them module-level makes the lifecycle
independent from the legacy SDK shim while preserving current artifact shapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

import benchflow._harbor as harbor_compat
from benchflow._sandbox import harden_before_verify
from benchflow.models import RunResult
from benchflow.rewards import write_rewards_jsonl
from benchflow.rollouts.result import TrajectorySource

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000


def init_rollout(
    task_path: Path,
    job_name: str | None,
    rollout_name: str | None,
    jobs_dir: str | Path,
) -> tuple[Any, Path, Any, datetime, str, str]:
    """Set up rollout directory tree and return core objects."""

    from uuid import uuid4

    task = harbor_compat.make_task(task_path)
    job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    rollout_name = rollout_name or f"{task_path.name}__{uuid4().hex[:8]}"
    rollout_dir = Path(jobs_dir) / job_name / rollout_name
    rollout_paths = harbor_compat.make_trial_paths(rollout_dir)
    started_at = datetime.now()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("agent", "verifier", "artifacts", "trajectory"):
        (rollout_dir / subdir).mkdir(exist_ok=True)
    return task, rollout_dir, rollout_paths, started_at, job_name, rollout_name


def write_config(
    rollout_dir: Path,
    *,
    task_path: Path,
    agent: str,
    model: str | None,
    environment: str,
    skills_dir: str | Path | None,
    sandbox_user: str | None,
    context_root: str | Path | None,
    sandbox_locked_paths: list[str] | None = None,
    sandbox_setup_timeout: int = 120,
    timeout: int,
    started_at: datetime,
    agent_env: dict[str, str],
) -> None:
    """Write config.json with secrets filtered out."""

    secret_substrings = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS")
    recorded_env = {
        k: v
        for k, v in agent_env.items()
        if not any(s in k.upper() for s in secret_substrings)
    }
    config_data = {
        "task_path": str(task_path),
        "agent": agent,
        "model": model,
        "environment": environment,
        "skills_dir": str(skills_dir) if skills_dir else None,
        "sandbox_user": sandbox_user,
        "sandbox_locked_paths": sandbox_locked_paths,
        "sandbox_setup_timeout": sandbox_setup_timeout,
        "context_root": str(context_root) if context_root else None,
        "timeout_sec": timeout,
        "started_at": str(started_at),
        "agent_env": recorded_env,
    }
    (rollout_dir / "config.json").write_text(json.dumps(config_data, indent=2))


def build_result(
    rollout_dir: Path,
    *,
    task_name: str,
    rollout_name: str,
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
) -> RunResult:
    """Build RunResult and write result/timing/prompt/trajectory artifacts."""

    finished_at = datetime.now()
    result = RunResult(
        task_name=task_name,
        trial_name=rollout_name,
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
        started_at=started_at,
        finished_at=finished_at,
    )
    timing["total"] = (finished_at - started_at).total_seconds()
    timing = {k: round(v, 1) for k, v in timing.items()}
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in trajectory)
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": result.task_name,
                "trial_name": result.trial_name,
                "rollout_name": result.rollout_name,
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
                "started_at": str(result.started_at),
                "finished_at": str(result.finished_at),
                "timing": timing,
            },
            indent=2,
        )
    )
    (rollout_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (rollout_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    write_rewards_jsonl(rollout_dir, rewards, finished_at)
    return result


def resolve_prompts(
    task_path: Path,
    prompts: list[str | None] | None,
    skills_dir: str | Path | None = None,
    skill_nudge: str = "",
    agent: str | None = None,
) -> list[str]:
    """Read instruction.md and resolve prompt list."""

    instruction_path = task_path / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Task missing instruction.md: {task_path}")
    instruction = instruction_path.read_text().strip()

    if skill_nudge:
        from benchflow.agents.registry import AGENTS

        skill_display_path = "~/.claude/skills"
        if agent:
            agent_cfg = AGENTS.get(agent)
            if agent_cfg and agent_cfg.skill_paths:
                skill_display_path = agent_cfg.skill_paths[0].replace("$HOME", "~")

        skills = []
        for src in [skills_dir, task_path / "environment" / "skills"]:
            if src and Path(src).is_dir():
                for d in sorted(Path(src).iterdir()):
                    if d.is_dir() and (d / "SKILL.md").exists():
                        content = d.joinpath("SKILL.md").read_text()
                        name = d.name
                        desc = ""
                        if content.startswith("---"):
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                import yaml

                                try:
                                    fm = yaml.safe_load(parts[1])
                                    desc = fm.get("description", "") if fm else ""
                                except Exception:
                                    pass
                        skills.append({"name": name, "desc": desc, "content": content})
                if skills:
                    break

        if skills:
            if skill_nudge == "name":
                names = ", ".join(s["name"] for s in skills)
                nudge = f"Skills available at {skill_display_path}: {names}. Read them before starting."
                instruction = nudge + "\n\n" + instruction
            elif skill_nudge == "description":
                lines = [f"Skills available at {skill_display_path}:\n"]
                for s in skills:
                    lines.append(f"- **{s['name']}**: {s['desc']}")
                lines.append("\nRead the relevant skills before starting.")
                instruction = "\n".join(lines) + "\n\n" + instruction
            elif skill_nudge == "full":
                blocks = []
                for s in skills:
                    blocks.append(
                        f'<skill name="{s["name"]}">\n{s["content"]}\n</skill>'
                    )
                instruction = "\n\n".join(blocks) + "\n\n" + instruction

    if prompts is None:
        return [instruction]
    return [p if p is not None else instruction for p in prompts]


async def start_env_and_upload(env, task_path: Path, timing: dict) -> None:
    """Start environment and upload standard task files."""

    logger.info(f"Starting environment: {task_path.name}")
    t0 = datetime.now()
    await env.start(force_build=False)
    timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    if (task_path / "instruction.md").exists():
        await env.upload_file(task_path / "instruction.md", "/instruction.md")
    if (task_path / "solution").is_dir():
        await env.upload_dir(task_path / "solution", "/solution")


async def run_oracle(
    env,
    task_path: Path,
    timeout: int,
    sandbox_user: str | None = None,
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
    oracle_env: dict[str, str] = {"DEBIAN_FRONTEND": "noninteractive"}
    task = harbor_compat.make_task(task_path)
    if task.config.solution.env:
        oracle_env.update(harbor_compat.resolve_env_vars(task.config.solution.env))
    result = await env.exec(
        f"{cmd} > /logs/agent/oracle.txt 2>&1",
        env=oracle_env,
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


async def verify(
    env,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    sandbox_user: str | None = None,
    workspace: str | None = None,
) -> tuple[dict | None, str | None]:
    """Run verifier with pre-verification hardening."""

    rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    await harden_before_verify(env, task, sandbox_user, workspace=workspace)
    logger.info("Running verifier...")
    t0 = datetime.now()
    verifier_error = None
    try:
        verifier = harbor_compat.make_verifier(
            task=task,
            trial_paths=rollout_paths,
            environment=env,
        )
        verifier_result = await asyncio.wait_for(
            verifier.verify(),
            timeout=task.config.verifier.timeout_sec,
        )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = verifier_result.rewards
        logger.info(f"Rewards: {rewards}")
    except TimeoutError:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier timed out after {task.config.verifier.timeout_sec}s"
        rewards = None
        logger.error(verifier_error)
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier crashed: {e}"
        rewards = None
        logger.error(verifier_error)
    return rewards, verifier_error

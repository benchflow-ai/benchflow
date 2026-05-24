"""Rollout — the single execution path for one agent-on-task evaluation.

The ``Rollout`` class owns the 5-phase lifecycle::

    rollout = await Rollout.create(RolloutConfig(task_path=..., agent=..., model=...))
    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()
    await rollout.connect()
    await rollout.execute()
    result = await rollout.verify()
    await rollout.cleanup()

Or use ``rollout.run()`` for the full lifecycle.

Phases can be composed for multi-agent flows::

    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()

    # Coder turn
    await rollout.connect()
    await rollout.execute(prompts=[coder_prompt])
    await rollout.disconnect()

    # Reviewer turn (same sandbox, new ACP session)
    await rollout.connect()
    await rollout.execute(prompts=[reviewer_prompt])
    await rollout.disconnect()

    result = await rollout.verify()
    await rollout.cleanup()

See also: ``RolloutConfig`` for configuration dataclass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._types import Role, Scene, Turn
from benchflow._utils.config import normalize_agent_name, normalize_sandbox_user
from benchflow._utils.scoring import classify_error, classify_verifier_error
from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.runtime import connect_acp, execute_prompts
from benchflow.agents.credentials import (
    upload_subscription_auth,
    write_credential_files,
)
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.install import (
    _link_skill_paths,
    apply_web_tool_policy,
    deploy_skills,
    install_agent,
)
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.providers.runtime import (
    ensure_bedrock_proxy_runtime,
    ensure_usage_proxy_runtime,
    extract_usage,
    stop_provider_runtime,
)
from benchflow.rewards.validation import validate_reward_map
from benchflow.rollout_branch import ChildRunner
from benchflow.rollout_branch import branch as _branch_engine
from benchflow.sandbox.daytona import SandboxStartupError
from benchflow.sandbox.lockdown import (
    _resolve_locked_paths,
    _seed_verifier_workspace,
    _snapshot_build_config,
    lockdown_paths,
    setup_sandbox_user,
)
from benchflow.sandbox.setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    stage_dockerfile_deps,
)
from benchflow.sandbox.user import BaseUser, RoundResult
from benchflow.trajectories._capture import (
    _capture_session_trajectory,
    _scrape_agent_trajectory,
)
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step

logger = logging.getLogger(__name__)

_DISALLOW_WEB_TOOLS_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"
SKILL_MODE_DEFAULT = "default"
SKILL_MODE_SELF_GEN = "self-gen"
GENERATED_SKILLS_ROOT = "/app/generated-skills"


def _task_disallows_internet(task: Any) -> bool:
    """Return True when task.toml requests no internet for the agent task."""
    env_config = getattr(getattr(task, "config", None), "environment", None)
    return getattr(env_config, "allow_internet", True) is False


def _apply_web_policy(agent_env: dict[str, str], *, disallow: bool) -> dict[str, str]:
    """Inject BenchFlow's no-web policy marker into agent env when requested."""
    if not disallow:
        return agent_env
    return {**agent_env, _DISALLOW_WEB_TOOLS_ENV: "1"}


def _agent_launch_with_web_policy(agent: str, *, disallow: bool) -> str:
    """Return launch command, appending the agent's no-web launch knob if any."""
    launch = AGENT_LAUNCH.get(agent, agent)
    if not disallow:
        return launch
    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.disallow_web_tools_launch_suffix:
        return launch + agent_cfg.disallow_web_tools_launch_suffix
    return launch


def _skill_nudge(agent_env: dict[str, str] | None) -> str:
    """Read skill nudge from explicit agent env or the host environment."""
    return (agent_env or {}).get("BENCHFLOW_SKILL_NUDGE") or os.environ.get(
        "BENCHFLOW_SKILL_NUDGE", ""
    )


def _safe_skill_name(value: str) -> str:
    """Return a filesystem-safe generated skill directory name."""
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return name or "generated-task"


def _skill_frontmatter_name(skill_dir: Path) -> str:
    """Read a skill's frontmatter name, falling back to the directory name."""
    from benchflow.skills import parse_skill

    info = parse_skill(skill_dir / "SKILL.md")
    return info.name if info and info.name else skill_dir.name


def _resolve_skill_creator_root(path: str | Path | None) -> tuple[Path, str]:
    """Resolve a skills root containing the official skill-creator directory.

    BenchFlow mounts skills as a root directory whose children are individual
    skill directories. If the caller points directly at skill-creator, use its
    parent as the mounted root and leave the skill pack contents unchanged.
    """
    candidates: list[Path] = []
    scan_single_skill_roots: set[Path] = set()
    if path:
        explicit_path = Path(path).expanduser()
        candidates.append(explicit_path)
        scan_single_skill_roots.add(explicit_path)
    env_path = os.environ.get("BENCHFLOW_SKILL_CREATOR_DIR")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append(env_candidate)
        scan_single_skill_roots.add(env_candidate)
    repo_skill_creator = (
        Path(__file__).resolve().parents[2] / ".claude" / "skills" / "skill-creator"
    )
    cwd_skill_creator = Path.cwd() / ".claude" / "skills" / "skill-creator"
    candidates.append(repo_skill_creator)
    if cwd_skill_creator != repo_skill_creator:
        candidates.append(cwd_skill_creator)
    candidates.extend(
        [
            Path.home() / ".claude" / "skills" / "skill-creator",
            Path.home() / ".codex" / "skills" / ".system" / "skill-creator",
            Path.home() / ".agents" / "skills" / "skill-creator",
        ]
    )

    for candidate in candidates:
        if (candidate / "SKILL.md").exists():
            return candidate.parent, _skill_frontmatter_name(candidate)
        if (candidate / "skill-creator" / "SKILL.md").exists():
            skill_dir = candidate / "skill-creator"
            return candidate, _skill_frontmatter_name(skill_dir)
        if candidate in scan_single_skill_roots and candidate.is_dir():
            skill_dirs = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and (child / "SKILL.md").exists()
            ]
            if len(skill_dirs) == 1:
                skill_dir = skill_dirs[0]
                return candidate, _skill_frontmatter_name(skill_dir)

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find skill-creator. Pass --skill-creator-dir or set "
        f"BENCHFLOW_SKILL_CREATOR_DIR. Checked: {checked}"
    )


def _self_gen_prompt(
    task_path: Path, generated_skills_root: str, skill_creator_name: str
) -> str:
    """Prompt the clean creator agent to use the mounted skill-creator skill."""
    skill_dir_name = f"{_safe_skill_name(task_path.name)}-skill"
    target_dir = f"{generated_skills_root}/{skill_dir_name}"
    return f"""Use the {skill_creator_name} skill exactly as provided.

Read /instruction.md and inspect the task environment only as needed to understand the reusable workflow. Do not solve the task directly.

Create one or more complete Anthropic-standard skill packs as immediate child directories under:

{generated_skills_root}

Use this suggested path if one skill is enough:

{target_dir}

Each generated skill pack path must look like {generated_skills_root}/<skill-name>/SKILL.md. It may include scripts/, references/, assets/, examples, or other bundled resources when they help a fresh solver avoid repeated work.

The solver context will start with a clean agent session and only the generated skill packs mounted. Make the skills useful for solving this task type from the same sandbox environment."""


async def _ensure_sandbox_dir(
    env: Any, path: str | Path, sandbox_user: str | None = None
) -> None:
    """Create a sandbox directory and optionally make it writable by the agent."""
    q_path = shlex.quote(str(path))
    cmd = f"mkdir -p {q_path}"
    if sandbox_user:
        q_user = shlex.quote(sandbox_user)
        cmd += f" && chown -R {q_user}:{q_user} {q_path}"
    result = await env.exec(cmd, timeout_sec=10)
    if result.return_code != 0:
        raise RuntimeError(
            f"Failed to create sandbox directory {path}: "
            f"{result.stderr or result.stdout}"
        )


_DIAG_TRUNCATE = 2000


def _write_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict | None,
    finished_at: datetime,
) -> None:
    """Write rewards.jsonl — one JSON line per reward event."""
    from typing import cast

    if not rewards:
        return
    events: list[dict[str, Any]] = []
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            rubric_item = cast(dict[str, Any], item)
            events.append(
                {
                    "ts": finished_at.isoformat(),
                    "type": "process",
                    "source": "verifier_rubric",
                    "value": rubric_item.get("score", 0.0),
                    "tag": rubric_item.get("name", f"rubric_{i}"),
                    "step_index": i,
                    "meta": {
                        k: v
                        for k, v in rubric_item.items()
                        if k not in ("score", "name")
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
        path = rollout_dir / "rewards.jsonl"
        path.write_text("\n".join(json.dumps(e, default=str) for e in events) + "\n")


def _init_rollout(
    task_path: Path,
    job_name: str | None,
    rollout_name: str | None,
    jobs_dir: str | Path,
) -> tuple[Any, Path, Any, datetime, str, str]:
    """Set up trial directory tree and return core trial objects."""
    from uuid import uuid4

    from benchflow.task import RolloutPaths, Task

    task = Task(task_path)
    job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    rollout_name = rollout_name or f"{task_path.name}__{uuid4().hex[:8]}"
    rollout_dir = Path(jobs_dir) / job_name / rollout_name
    rollout_paths = RolloutPaths(rollout_dir=rollout_dir)
    started_at = datetime.now()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("agent", "verifier", "artifacts", "trajectory"):
        (rollout_dir / subdir).mkdir(exist_ok=True)
    return task, rollout_dir, rollout_paths, started_at, job_name, rollout_name


def _write_config(
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
    concurrency: int | None = None,
    agent_idle_timeout: int | None = None,
    scenes: list[Scene] | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> None:
    """Write config.json to rollout_dir with secrets filtered out."""
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
        "sandbox_setup_timeout": sandbox_setup_timeout,
        "context_root": str(context_root) if context_root else None,
        "timeout_sec": timeout,
        "concurrency": concurrency,
        "agent_idle_timeout_sec": agent_idle_timeout,
        "started_at": str(started_at),
        "agent_env": recorded_env,
        "scenes": _scene_metadata(scenes or []),
    }
    if source_provenance is not None:
        config_data["source"] = source_provenance
    (rollout_dir / "config.json").write_text(json.dumps(config_data, indent=2))


def _role_metadata(role: Role) -> dict[str, Any]:
    return {
        "name": role.name,
        "agent": role.agent,
        "model": role.model,
        "timeout_sec": role.timeout_sec,
        "idle_timeout_sec": role.idle_timeout_sec,
        "skills_dir": str(role.skills_dir) if role.skills_dir else None,
        "capabilities": role.capabilities,
        "env_keys": sorted(role.env),
    }


def _scene_metadata(scenes: list[Scene]) -> list[dict[str, Any]]:
    return [
        {
            "name": scene.name,
            "skills_dir": str(scene.skills_dir) if scene.skills_dir else None,
            "parallel_group": scene.parallel_group,
            "roles": [_role_metadata(role) for role in scene.roles],
            "turns": [
                {"role": turn.role, "has_prompt": turn.prompt is not None}
                for turn in scene.turns
            ],
        }
        for scene in scenes
    ]


def _parse_transport_error(e: ConnectionError) -> dict:
    """Extract structured fields from a ConnectionError message.

    Guards ENG-148: ACP transport rc=255 must carry actionable diagnostics.
    """
    import re as _re

    msg = str(e)
    info: dict[str, Any] = {"reason": "transport_closed", "raw_message": msg[:500]}

    rc_match = _re.search(r"rc=(\d+|None)", msg)
    if rc_match:
        raw = rc_match.group(1)
        info["process_exit_code"] = int(raw) if raw != "None" else None

    pid_match = _re.search(r"pid=(\d+)", msg)
    if pid_match:
        info["process_pid"] = int(pid_match.group(1))

    if "still alive but its stdout/transport closed" in msg:
        info["transport_diagnosis"] = "remote_session_killed"
    elif "exited with rc=" in msg:
        info["transport_diagnosis"] = "process_exited"
    elif "PTY readline" in msg:
        info["transport_diagnosis"] = "pty_error"
    else:
        info["transport_diagnosis"] = "unknown"

    stderr_match = _re.search(r"stderr: (.+)", msg, _re.DOTALL)
    if stderr_match:
        info["stderr_snippet"] = stderr_match.group(1).strip()[:500]

    return info


def _build_rollout_result(
    rollout_dir: Path,
    *,
    task_name: str,
    rollout_name: str,
    agent: str,
    agent_name: str,
    model: str | None,
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
    scenes: list[Scene] | None = None,
    n_input_tokens: int | None = None,
    n_output_tokens: int | None = None,
    n_cache_read_tokens: int | None = None,
    n_cache_creation_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    usage_source: str = "unavailable",
    price_source: str | None = None,
    evolved_skills: dict[str, str] | None = None,
    source_provenance: dict[str, Any] | None = None,
    idle_timeout_info: dict | None = None,
    sandbox_startup_info: dict | None = None,
    transport_error_info: dict | None = None,
    verifier_timeout_info: dict | None = None,
) -> RolloutResult:
    """Build RolloutResult and write result.json, timing.json, prompts.json, trajectory."""
    finished_at = datetime.now()
    result = RolloutResult(
        task_name=task_name,
        rollout_name=rollout_name,
        rewards=rewards,
        trajectory=trajectory,
        agent=agent,
        agent_name=agent_name,
        model=model,
        n_tool_calls=n_tool_calls,
        n_prompts=len(prompts),
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
        n_cache_read_tokens=n_cache_read_tokens,
        n_cache_creation_tokens=n_cache_creation_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        usage_source=usage_source,
        price_source=price_source,
        error=error,
        verifier_error=verifier_error,
        partial_trajectory=partial_trajectory,
        trajectory_source=trajectory_source,
        evolved_skills=evolved_skills,
        source_provenance=source_provenance,
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
                "rollout_name": result.rollout_name,
                "rewards": result.rewards,
                "agent": result.agent,
                "agent_name": result.agent_name,
                "model": result.model,
                "n_tool_calls": result.n_tool_calls,
                "n_prompts": result.n_prompts,
                "agent_result": {
                    "n_tool_calls": result.n_tool_calls,
                    "n_prompts": result.n_prompts,
                    "n_input_tokens": result.n_input_tokens,
                    "n_output_tokens": result.n_output_tokens,
                    "n_cache_read_tokens": result.n_cache_read_tokens,
                    "n_cache_creation_tokens": result.n_cache_creation_tokens,
                    "total_tokens": result.total_tokens,
                    "cost_usd": result.cost_usd,
                    "usage_source": result.usage_source,
                    "price_source": result.price_source,
                },
                "error": result.error,
                "error_category": classify_error(result.error),
                "verifier_error": result.verifier_error,
                "verifier_error_category": classify_verifier_error(
                    result.verifier_error
                ),
                "idle_timeout_info": idle_timeout_info,
                "sandbox_startup_info": sandbox_startup_info,
                "transport_error_info": transport_error_info,
                "verifier_timeout_info": verifier_timeout_info,
                "partial_trajectory": result.partial_trajectory,
                "trajectory_source": result.trajectory_source,
                "started_at": str(result.started_at),
                "finished_at": str(result.finished_at),
                "timing": timing,
                "scenes": _scene_metadata(scenes or []),
                **(
                    {"source": source_provenance}
                    if source_provenance is not None
                    else {}
                ),
            },
            indent=2,
        )
    )
    (rollout_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (rollout_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    _write_rewards_jsonl(rollout_dir, rewards, finished_at)
    return result


def _resolve_prompts(
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


async def _start_env_and_upload(env: Any, task_path: Path, timing: dict) -> None:
    """Start environment and upload task files."""
    logger.info(f"Starting environment: {task_path.name}")
    t0 = datetime.now()
    await env.start(force_build=False)
    timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    if (task_path / "instruction.md").exists():
        await env.upload_file(task_path / "instruction.md", "/instruction.md")
    task_skills = task_path / "environment" / "skills"
    if task_skills.is_dir():
        await env.upload_dir(task_skills, "/app/skills")
    if (task_path / "solution").is_dir():
        await env.upload_dir(task_path / "solution", "/solution")


async def _run_oracle(
    env: Any, task_path: Path, timeout: int, sandbox_user: str | None = None
) -> tuple[list[dict], str]:
    """Run oracle mode (solution/solve.sh), return (trajectory, agent_name)."""
    from benchflow.task import Task, resolve_env_vars

    logger.info("Oracle mode: running solution/solve.sh")
    if not (task_path / "solution" / "solve.sh").exists():
        raise FileNotFoundError(f"Oracle requires solution/solve.sh: {task_path}")
    if sandbox_user:
        oracle_cmd = "DEBIAN_FRONTEND=noninteractive bash /solution/solve.sh"
        cmd = (
            f"su -s /bin/bash {shlex.quote(sandbox_user)} -c {shlex.quote(oracle_cmd)}"
        )
    else:
        cmd = "bash /solution/solve.sh"
    oracle_env: dict[str, str] = {"DEBIAN_FRONTEND": "noninteractive"}
    task = Task(task_path)
    if task.config.solution.env:
        oracle_env.update(resolve_env_vars(task.config.solution.env))
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


async def _publish_trajectory_for_verifier(env, trajectory: list[dict]) -> None:
    """Make the captured ACP trajectory available inside /logs for verifiers."""
    if not trajectory:
        return
    await env.exec("mkdir -p /logs/agent", user="root", timeout_sec=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(e, default=str) for e in trajectory))
        f.write("\n")
        tmp_path = f.name
    try:
        await env.upload_file(tmp_path, "/logs/agent/acp_trajectory.jsonl")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)


async def _verify_rollout(
    env: Any,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    sandbox_user: str | None = None,
    workspace: str | None = None,
) -> tuple[dict | None, str | None, dict | None]:
    """Run verifier with pre-verification hardening.

    Returns (rewards, verifier_error, verifier_timeout_info).
    """
    from benchflow.sandbox.lockdown import harden_before_verify
    from benchflow.task import Verifier

    rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    verifier_error = None
    verifier_timeout_info: dict | None = None
    timeout_budget = task.config.verifier.timeout_sec
    try:
        await harden_before_verify(env, task, sandbox_user, workspace=workspace)
        logger.info("Running verifier...")
        verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=env)
        verifier_result = await asyncio.wait_for(
            verifier.verify(),
            timeout=timeout_budget,
        )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = _ensure_canonical_rewards(verifier_result.rewards)
        logger.info(f"Rewards: {rewards}")
    except TimeoutError:
        elapsed = (datetime.now() - t0).total_seconds()
        timing["verifier"] = elapsed
        verifier_error = f"verifier timed out after {timeout_budget}s"
        verifier_timeout_info = {
            "timeout_budget_sec": timeout_budget,
            "elapsed_sec": round(elapsed, 1),
            "task_name": task.config.name,
        }
        rewards = None
        logger.error(verifier_error)
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier crashed: {e}"
        rewards = None
        logger.error(verifier_error)
    return rewards, verifier_error, verifier_timeout_info


def _ensure_canonical_rewards(rewards: dict | None) -> dict:
    return validate_reward_map(rewards, source="verifier")


# Apply Docker DinD patch at import time.
def _apply_dind_patch() -> None:
    from benchflow.sandbox.setup import _patch_docker_dind

    _patch_docker_dind()


_apply_dind_patch()


__all__ = [
    "Role",
    "Scene",
    "Turn",
    "Rollout",
    "RolloutConfig",
]


@dataclass
class RolloutConfig:
    """Declarative rollout configuration.

    A rollout is a sequence of scenes executed in a shared sandbox.
    Single-agent runs are a rollout with one scene containing one role.
    """

    task_path: Path
    scenes: list[Scene] = field(default_factory=list)
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    services: list[str] | None = None
    job_name: str | None = None
    rollout_name: str | None = None
    jobs_dir: str | Path = "jobs"
    concurrency: int = 1
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None
    # Environment plane — when set, Rollout provisions a manifest-declared
    # stateful environment, gates on readiness before the agent runs, and
    # tears it down in cleanup. None => no separate environment plane (default).
    environment_manifest: EnvironmentManifest | None = None
    # Abort the prompt if no tool call arrives for this many seconds.
    # Catches agents that hung silently while the local process is alive
    # (e.g. gemini-cli not responding). None disables idle detection and
    # falls back to the agent's wall-clock timeout (task.toml [agent]).
    agent_idle_timeout: int | None = 600

    # User-driven progressive-disclosure loop
    user: BaseUser | None = None
    max_user_rounds: int = 5
    oracle_access: bool = False

    # Legacy compat fields — used by SDK.run() shim. Ignored when scenes is set.
    agent: str = "claude-agent-acp"
    prompts: list[str | None] | None = None
    model: str | None = None
    agent_env: dict[str, str] | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_DEFAULT
    skill_creator_dir: str | Path | None = None
    generated_skills_root: str = GENERATED_SKILLS_ROOT
    self_gen_no_internet: bool = False
    include_task_skills: bool = True
    skip_verify: bool = False
    export_generated_skills_to: str | Path | None = None
    source_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from benchflow._utils.config import normalize_agent_idle_timeout

        self.agent = normalize_agent_name(self.agent)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        for scene in self.scenes:
            for role in scene.roles:
                role.agent = normalize_agent_name(role.agent)

    @classmethod
    def from_legacy(
        cls,
        *,
        task_path: Path,
        agent: str = "claude-agent-acp",
        model: str | None = None,
        prompts: list[str | None] | None = None,
        skills_dir: str | Path | None = None,
        skill_mode: str = SKILL_MODE_DEFAULT,
        skill_creator_dir: str | Path | None = None,
        generated_skills_root: str = GENERATED_SKILLS_ROOT,
        self_gen_no_internet: bool = False,
        **kwargs,
    ) -> RolloutConfig:
        """Construct from flat SDK.run()-style args."""
        scenes = []
        if skill_mode not in {SKILL_MODE_DEFAULT, SKILL_MODE_SELF_GEN}:
            raise ValueError(f"Unknown skill_mode: {skill_mode}")
        if skill_mode == SKILL_MODE_DEFAULT:
            scenes = [
                Scene.single(
                    agent=agent, model=model, prompts=prompts, skills_dir=skills_dir
                )
            ]
        return cls(
            task_path=task_path,
            scenes=scenes,
            agent=agent,
            model=model,
            prompts=prompts,
            skills_dir=skills_dir,
            skill_mode=skill_mode,
            skill_creator_dir=skill_creator_dir,
            generated_skills_root=generated_skills_root,
            self_gen_no_internet=self_gen_no_internet,
            **kwargs,
        )

    @property
    def effective_scenes(self) -> list[Scene]:
        """Scenes to execute — falls back to legacy fields if scenes is empty."""
        if self.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout scenes."
            )
        if self.scenes:
            return self.scenes
        if self.skill_mode != SKILL_MODE_DEFAULT:
            raise ValueError(f"Unknown skill_mode: {self.skill_mode}")
        return [
            Scene.single(
                agent=self.agent,
                model=self.model,
                prompts=self.prompts,
                skills_dir=self.skills_dir,
            )
        ]

    @property
    def primary_agent(self) -> str:
        """Agent name for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].agent
        return self.agent

    @property
    def primary_model(self) -> str | None:
        """Model for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].model
        return self.model


class Rollout:
    """Decomposed trial lifecycle with independently-callable phases."""

    def __init__(self, config: RolloutConfig) -> None:
        self._config = config
        self._phase = "created"

        # Populated by setup()
        self._task: Any = None
        self._rollout_dir: Path | None = None
        self._rollout_paths: Any = None
        self._started_at: datetime | None = None
        self._job_name: str | None = None
        self._rollout_name: str | None = None
        self._agent_env: dict[str, str] = {}
        self._resolved_prompts: list[str] = []
        self._agent_launch: str = ""
        self._env: Any = None
        self._environment: ManifestEnvironment | None = None
        self._timeout: int = 0
        self._timing: dict[str, float] = {}
        self._effective_locked: list[str] = []
        self._disallow_web_tools: bool = False
        self._usage_runtime: Any = None
        self._usage_metrics: dict[str, Any] = extract_usage(None)

        # Populated by install_agent()
        self._agent_cfg: Any = None
        self._agent_cwd: str = "/app"

        # Populated by connect()
        self._acp_client: ACPClient | None = None
        self._session: Any = None
        self._agent_name: str = ""
        self._active_role: Role | None = None

        # Populated by execute()
        self._trajectory: list[dict] = []
        self._n_tool_calls: int = 0
        self._trajectory_source: TrajectorySource | None = None
        self._partial_trajectory: bool = False

        # The tree-native execution model (architecture.md, "tree-native").
        # A linear rollout is a degree-1 tree; execute() grows it one Step at a
        # time, and branch() forks a node into N children. The tree is additive
        # — it never alters linear behaviour or output.
        self._tree: RolloutTree = RolloutTree()
        self._cursor: RolloutNode = self._tree.root

        # Populated by verify()
        self._rewards: dict | None = None
        self._verifier_error: str | None = None
        self._error: str | None = None
        self._idle_timeout_info: dict | None = None
        self._sandbox_startup_info: dict | None = None
        self._transport_error_info: dict | None = None
        self._verifier_timeout_info: dict | None = None

        # Populated by _export_generated_skills() — the skills the agent
        # generated/evolved, captured for a continual-learning LearnerStore.
        self._evolved_skills: dict[str, str] | None = None

    @classmethod
    async def create(cls, config: RolloutConfig) -> Rollout:
        """Create a Rollout instance. Preferred over __init__ for consistency."""
        if config.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.create()."
            )
        return cls(config)

    @property
    def env(self) -> Any:
        return self._env

    @property
    def acp_client(self) -> ACPClient | None:
        return self._acp_client

    @property
    def trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def tree(self) -> RolloutTree:
        """The RolloutTree this rollout grows as it executes.

        A linear rollout is a degree-1 tree; :meth:`branch` forks a node into
        N children. ``tree.root`` is the start state s₀.
        """
        return self._tree

    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    @property
    def result(self) -> RolloutResult | None:
        if self._phase not in ("verified", "cleaned"):
            return None
        return self._build_result()

    def _require_rollout_dir(self) -> Path:
        if self._rollout_dir is None:
            raise RuntimeError("Rollout.setup() must run before this phase")
        return self._rollout_dir

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise RuntimeError("Rollout.setup() must run before building a result")
        return self._started_at

    # ── Phase 1: SETUP (host-side, no container yet) ──

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — /solution stays visible "
                "to the agent for the entire trial."
            )

        self._effective_locked = _resolve_locked_paths(
            cfg.sandbox_user, cfg.sandbox_locked_paths
        )

        (
            self._task,
            self._rollout_dir,
            self._rollout_paths,
            self._started_at,
            self._job_name,
            self._rollout_name,
        ) = _init_rollout(cfg.task_path, cfg.job_name, cfg.rollout_name, cfg.jobs_dir)

        self._disallow_web_tools = (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        ) and cfg.primary_agent != "oracle"
        self._agent_env = _apply_web_policy(
            resolve_agent_env(cfg.primary_agent, cfg.primary_model, cfg.agent_env),
            disallow=self._disallow_web_tools,
        )
        self._resolved_prompts = _resolve_prompts(
            cfg.task_path,
            cfg.prompts,
            skills_dir=cfg.skills_dir,
            skill_nudge=_skill_nudge(cfg.agent_env),
            agent=cfg.primary_agent,
        )
        self._agent_launch = _agent_launch_with_web_policy(
            cfg.primary_agent,
            disallow=self._disallow_web_tools,
        )

        # Copy task dir to temp when Dockerfile mutations are needed
        # (_inject_skills writes into environment/_deps/, stage_dockerfile
        # rewrites COPY paths — neither should modify the source tree)
        effective_task_path = cfg.task_path
        if cfg.context_root or cfg.skills_dir:
            import shutil
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            shutil.copytree(cfg.task_path, tmp / cfg.task_path.name, dirs_exist_ok=True)
            effective_task_path = tmp / cfg.task_path.name
            self._task_tmp = tmp

        if cfg.context_root:
            stage_dockerfile_deps(effective_task_path, Path(cfg.context_root))
        if cfg.skills_dir:
            _inject_skills_into_dockerfile(effective_task_path, Path(cfg.skills_dir))

        self._effective_task_path = effective_task_path

        self._env = _create_environment(
            cfg.environment,
            self._task,
            effective_task_path,
            self._rollout_name,
            self._rollout_paths,
            preserve_agent_network=self._disallow_web_tools,
        )
        self._timeout = int(self._task.config.agent.timeout_sec or 0)

        _write_config(
            self._rollout_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            environment=cfg.environment,
            skills_dir=cfg.skills_dir,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
            concurrency=cfg.concurrency,
            agent_idle_timeout=cfg.agent_idle_timeout,
            scenes=cfg.effective_scenes,
            source_provenance=cfg.source_provenance,
        )

        self._phase = "setup"

    # ── Phase 2: START (container comes up) ──

    async def start(self) -> None:
        """Start the environment and upload task files."""
        await _start_env_and_upload(self._env, self._config.task_path, self._timing)

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        # Environment plane: provision the manifest-declared stateful
        # environment and gate on its readiness before the agent runs.
        if self._config.environment_manifest is not None:
            self._environment = ManifestEnvironment(
                self._config.environment_manifest, sandbox=self._env
            )
            await self._environment.provision(
                ctx={"task_id": self._config.task_path.name}
            )
            probe = await self._environment.readiness()
            if not probe.ready:
                raise RuntimeError(
                    f"environment plane not ready: {probe.error} "
                    f"(checked: {probe.checked})"
                )
            logger.info(
                "environment '%s' ready (%d probe(s))",
                self._config.environment_manifest.name,
                len(probe.checked),
            )

        self._phase = "started"

    # ── Phase 3: INSTALL AGENT ──

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous multi-agent scenes (different agents per role),
        each role's agent is installed on-demand in _run_scene/connect_as.
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()

        cwd_result = await self._env.exec("pwd", timeout_sec=10)
        agent_cwd = (cwd_result.stdout or "").strip() or "/app"
        self._agent_cwd = agent_cwd

        if cfg.primary_agent == "oracle":
            if cfg.sandbox_user:
                await setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            await _snapshot_build_config(self._env, workspace=self._agent_cwd)
            await _seed_verifier_workspace(
                self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
            )
            await deploy_skills(
                self._env,
                getattr(self, "_effective_task_path", cfg.task_path),
                cfg.skills_dir,
                None,
                cfg.sandbox_user,
                self._agent_cwd,
                self._task,
                include_task_skills=cfg.include_task_skills,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        self._agent_cfg = await install_agent(self._env, agent_name, rollout_dir)
        if cfg.sandbox_user:
            self._agent_cwd = await setup_sandbox_user(
                self._env,
                cfg.sandbox_user,
                workspace=self._agent_cwd,
                timeout_sec=cfg.sandbox_setup_timeout,
            )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await write_credential_files(
            self._env,
            agent_name,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await upload_subscription_auth(self._env, agent_name, cred_home)
        await apply_web_tool_policy(
            self._env,
            agent_name,
            self._agent_cfg,
            cred_home,
            disallow=self._disallow_web_tools,
        )
        await _snapshot_build_config(self._env, workspace=self._agent_cwd)
        await _seed_verifier_workspace(
            self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
        )

        await deploy_skills(
            self._env,
            getattr(self, "_effective_task_path", cfg.task_path),
            cfg.skills_dir,
            self._agent_cfg,
            cfg.sandbox_user,
            self._agent_cwd,
            self._task,
            include_task_skills=cfg.include_task_skills,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # ── Phase 3b: CONNECT (ACP session — re-entrant) ──

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        self._agent_env, self._provider_runtime = await ensure_bedrock_proxy_runtime(
            agent=cfg.primary_agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            runtime=getattr(self, "_provider_runtime", None),
            environment=cfg.environment,
        )
        self._agent_env, self._usage_runtime = await ensure_usage_proxy_runtime(
            agent=cfg.primary_agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
        )
        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.primary_model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
        )

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    async def disconnect(self) -> None:
        """Close the ACP client and clean up agent process, keeping the environment alive."""
        self._capture_partial_acp_trajectory()
        if self._acp_client:
            try:
                await self._acp_client.close()
            except Exception as e:
                logger.warning(f"ACP client close failed: {e}")
            self._acp_client = None
            self._session = None
        # Kill any lingering agent processes to prevent context bleed between scenes
        if self._env and self._agent_launch.strip():
            agent_cmd = self._agent_launch.split()[0].split("/")[-1]
            with contextlib.suppress(Exception):
                await self._env.exec(f"pkill -f '{agent_cmd}' || true", timeout_sec=10)
        self._active_role = None
        self._session_tool_count = 0
        self._session_traj_count = 0
        self._phase = "installed"

    def _capture_partial_acp_trajectory(self) -> None:
        if self._trajectory or not self._acp_client or not self._acp_client.session:
            return
        try:
            captured = _capture_session_trajectory(self._acp_client.session)
            if captured:
                self._trajectory = captured
                self._partial_trajectory = True
                self._trajectory_source = "partial_acp"
                self._n_tool_calls = len(self._acp_client.session.tool_calls)
        except Exception as e:
            logger.warning(f"Partial trajectory capture failed: {e}")

    # ── Phase 3c: EXECUTE ──

    async def execute(
        self, prompts: list[str] | None = None, *, node: RolloutNode | None = None
    ) -> tuple[list[dict], int]:
        """Run prompts through the ACP session. Returns (new trajectory, new tool calls).

        execute_prompts returns cumulative session trajectory. We track
        what we've already captured to avoid duplication when the same
        session is reused across multiple turns.

        ``node`` — when given, a *pending* tree node (no incoming Step yet,
        from :meth:`RolloutTree.attach`) whose Step this call fills in place,
        instead of advancing the tree with a fresh child. The Branch engine
        passes a pre-attached branch-child node here so the child's real
        continuation Step lands on the child node itself.
        """
        effective_prompts = prompts or self._resolved_prompts
        if self._acp_client is None:
            raise RuntimeError("Rollout.connect() must run before execute()")
        prev_session_tools = getattr(self, "_session_tool_count", 0)
        t0 = datetime.now()
        active_role = getattr(self, "_active_role", None)
        timeout = (
            active_role.timeout_sec
            if active_role and active_role.timeout_sec is not None
            else self._timeout
        )
        idle_timeout = (
            active_role.idle_timeout_sec
            if active_role and active_role.idle_timeout_sec is not None
            else self._config.agent_idle_timeout
        )

        trajectory, n_tool_calls = await execute_prompts(
            self._acp_client,
            self._session,
            effective_prompts,
            timeout,
            idle_timeout=idle_timeout,
        )

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[getattr(self, "_session_traj_count", 0) :]
        self._session_tool_count = n_tool_calls
        self._session_traj_count = len(trajectory)

        self._trajectory.extend(new_events)
        self._n_tool_calls += new_tools
        self._trajectory_source = "acp"

        # Grow the tree by one Step. A linear rollout is a degree-1 tree — the
        # cursor walks down a chain, one node per execute() call. This is
        # additive: it never alters the trajectory or any linear output.
        step = Step(
            id=f"step-{len(self._trajectory)}",
            data={"events": new_events, "n_tool_calls": new_tools},
        )
        if node is not None:
            # Fill a pre-attached pending node (a branch child) in place — the
            # child's real continuation Step lands on the child node itself.
            self._cursor = self._tree.populate(node, step)
        else:
            self._cursor = self._tree.advance(self._cursor, step)

        if "agent_execution" not in self._timing:
            self._timing["agent_execution"] = (datetime.now() - t0).total_seconds()

        self._phase = "executed"
        return trajectory, n_tool_calls

    # ── Phase 3d: BRANCH ──

    async def branch(
        self,
        n: int,
        run_child: ChildRunner | None = None,
    ) -> float:
        """Branch the rollout at the cursor into ``n`` child continuations.

        Thin entry point — the Branch engine lives in
        :mod:`benchflow.rollout_branch`. It checkpoints the Environment at the
        cursor, runs each forked child as an isolated sub-rollout (its own
        scoped state, a fresh agent session), and aggregates the children's
        returns into V(parent). After this returns, the rollout's linear state
        is exactly what it was before.

        ``run_child`` is the per-child runner — injected for unit tests; the
        default restores the env, connects a fresh agent, runs the continuation,
        and scores it. A caller that needs per-child prompts binds them into the
        ``run_child`` closure.
        """
        return await _branch_engine(self, n, run_child)

    # ── Phase 4: VERIFY ──

    async def verify(self) -> dict | None:
        """Run the verifier and return rewards."""
        cfg = self._config

        if not self._trajectory and cfg.primary_agent != "oracle":
            scraped = await _scrape_agent_trajectory(
                self._env, cfg.primary_agent, cfg.sandbox_user
            )
            if scraped:
                self._trajectory = scraped
                self._trajectory_source = "scraped"
                logger.warning(
                    f"Using scraped trajectory ({len(scraped)} events) — UNTRUSTED"
                )

        await _publish_trajectory_for_verifier(self._env, self._trajectory)

        self._rewards, self._verifier_error, self._verifier_timeout_info = (
            await _verify_rollout(
                self._env,
                self._task,
                self._rollout_paths,
                self._timing,
                sandbox_user=cfg.sandbox_user,
                workspace=self._agent_cwd,
            )
        )

        self._phase = "verified"
        return self._rewards

    async def soft_verify(self) -> tuple[dict | None, str | None, str | None]:
        """Run the verifier without full hardening — for intermediate feedback.

        Skips process kill and workspace restore/chown (so the sandbox
        stays usable for the next round), but DOES purge agent-injected
        conftest.py / sitecustomize.py / .pth files to prevent the agent
        from gaming intermediate test results.

        Returns (rewards, verifier_output, verifier_error). The final
        verify() still does full hardening.
        """
        from benchflow.sandbox.lockdown import (
            cleanup_verifier_python_hooks,
            clear_verifier_output_dir,
            ensure_legacy_app_dir,
        )
        from benchflow.task import Verifier

        self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        try:
            await clear_verifier_output_dir(
                self._env,
                "Soft verifier setup failed: clearing verifier output directory",
                user="root",
                timeout_sec=10,
            )
            await ensure_legacy_app_dir(
                self._env,
                "Soft verifier setup failed: preparing /app",
                user="root",
                timeout_sec=10,
            )
            # Purge agent-injected conftest/sitecustomize/.pth without
            # killing processes or restoring workspace.
            # Honor per-task [verifier.hardening] opt-outs from task.toml.
            await cleanup_verifier_python_hooks(
                self._env,
                getattr(self._task, "task_dir", None),
                "Soft verifier setup failed: purging Python injection hooks",
                user="root",
                timeout_sec=10,
            )
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
            return None, None, verifier_error

        rewards = None
        verifier_output = None
        verifier_error = None
        try:
            verifier = Verifier(
                task=self._task,
                rollout_paths=self._rollout_paths,
                sandbox=self._env,
            )
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = _ensure_canonical_rewards(verifier_result.rewards)
            # Capture raw verifier output for the user
            cat = await self._env.exec(
                "cat /logs/verifier/*.log 2>/dev/null || "
                "cat /logs/verifier/output.txt 2>/dev/null || true",
                timeout_sec=10,
            )
            verifier_output = (cat.stdout or "").strip() or None
            logger.info(f"[soft_verify] rewards={rewards}")
        except TimeoutError:
            verifier_error = (
                f"soft verifier timed out after "
                f"{self._task.config.verifier.timeout_sec}s"
            )
            logger.error(verifier_error)
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
        return rewards, verifier_output, verifier_error

    # ── Phase 5: CLEANUP ──

    async def cleanup(self) -> None:
        """Close ACP client and stop the environment."""
        self._capture_partial_acp_trajectory()
        await self.disconnect()

        if self._env and self._config.export_generated_skills_to:
            try:
                await self._export_generated_skills()
            except Exception as e:
                # Surface export failure as a rollout error so it cannot be
                # confused with a successful-but-empty skill update (ENG/PR #389).
                # An export that was configured-but-failed must not collapse
                # into the same observable state as "agent honestly produced no
                # skills". Preserve any pre-existing agent/run error: skill
                # export happens during cleanup, after the agent already ran.
                export_error = f"Skill export failed: {e}"
                logger.error(export_error)
                if self._error is None:
                    self._error = export_error
                self._evolved_skills = None

        usage_runtime = getattr(self, "_usage_runtime", None)
        if usage_runtime is not None:
            try:
                await stop_provider_runtime(usage_runtime)
                self._usage_metrics = extract_usage(usage_runtime)
            except Exception as e:
                logger.warning(f"Usage telemetry runtime stop failed: {e}")
                self._usage_metrics = extract_usage(None)
            try:
                self._write_llm_trajectory(usage_runtime)
            except Exception as e:
                logger.warning(f"LLM trajectory write failed: {e}")
            finally:
                self._usage_runtime = None

        if self._environment is not None:
            with contextlib.suppress(Exception):
                await self._environment.teardown()
            self._environment = None

        if self._env:
            try:
                await stop_provider_runtime(getattr(self, "_provider_runtime", None))
                self._provider_runtime = None
            except Exception as e:
                logger.warning(f"Provider runtime stop failed: {e}")
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        if hasattr(self, "_task_tmp") and self._task_tmp:
            import shutil

            shutil.rmtree(self._task_tmp, ignore_errors=True)

        self._phase = "cleaned"

    # ── Full run ──

    async def run(self) -> RolloutResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        agent_timed_out = False
        if cfg.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.run()."
            )
        try:
            await self.setup()
            await self.start()

            if cfg.primary_agent == "oracle":
                await self.install_agent()
                # git safe.directory needed for SWE-bench tasks with sandbox_user
                import shlex

                await self._env.exec(
                    f"git config --global --add safe.directory "
                    f"{shlex.quote(self._agent_cwd)} 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
                self._trajectory, self._agent_name = await _run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
                )
            else:
                await self.install_agent()
                try:
                    try:
                        if cfg.user is not None:
                            await self._run_user_loop()
                        else:
                            for scene in cfg.effective_scenes:
                                await self._run_scene(scene)
                    except TimeoutError as e:
                        agent_timed_out = True
                        detail = str(e).strip()
                        self._error = (
                            detail or f"Agent timed out after {self._timeout}s"
                        )
                        self._idle_timeout_info = getattr(
                            e, "idle_timeout_info", None
                        )
                        logger.error(self._error)
                finally:
                    if cfg.oracle_access:
                        await self._env.exec(
                            "mv /solution_oracle_backup /solution 2>/dev/null || true",
                            user="root",
                            timeout_sec=10,
                        )

            if not cfg.skip_verify:
                await self.verify()
                if (
                    agent_timed_out
                    and self._rewards is None
                    and self._verifier_error is None
                ):
                    self._rewards = {"reward": 0.0}
                    self._verifier_error = None

        except TimeoutError as e:
            # Preserve the watchdog's diagnostic message ("Agent idle for 600s
            # with no new tool call ...") if it raised one. Fall back to the
            # generic wall-clock message only when there's no detail.
            detail = str(e).strip()
            self._error = detail or f"Agent timed out after {self._timeout}s"
            self._idle_timeout_info = getattr(
                e, "idle_timeout_info", None
            )
            logger.error(self._error)
        except ConnectionError as e:
            self._error = str(e)
            self._transport_error_info = _parse_transport_error(e)
            await self._probe_sandbox_health()
            logger.error(f"Agent connection lost: {self._error}")
        except SandboxStartupError as e:
            self._error = f"Sandbox startup failed: {e}"
            self._sandbox_startup_info = e.sandbox_startup_info
            logger.error(self._error)
        except ACPError as e:
            self._error = self._classify_acp_error(e)
            logger.error(self._error)
        except Exception as e:
            self._error = str(e)
            logger.error("Run failed", exc_info=True)
        finally:
            await self.cleanup()

        if self._rollout_dir is None:
            return RolloutResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return self._build_result()

    # ── Scene execution ──

    _OUTBOX_DIR = "/app/.outbox"

    async def _export_generated_skills(self) -> None:
        """Download creator-produced skills before sandbox cleanup.

        Also captures the exported skill packs into ``self._evolved_skills``
        — the ``name -> body`` dict a continual-learning Job commits to its
        persistent LearnerStore (capability 5).

        Retries transient download failures up to 3 times (guards ENG-147).
        """
        export_target = self._config.export_generated_skills_to
        if export_target is None:
            return
        target = Path(export_target)
        target.mkdir(parents=True, exist_ok=True)

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                await self._env.download_dir(
                    self._config.generated_skills_root, target
                )
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"Skill export attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
        else:
            raise RuntimeError(
                f"Skill export failed after 3 attempts: {last_err}"
            ) from last_err

        from benchflow.learner_skills import capture_skills

        self._evolved_skills = capture_skills(target)

    async def _activate_scene_skills(self, scene: Scene) -> None:
        """Activate scene-local skills by linking them into role discovery paths."""
        if not scene.skills_dir:
            return
        if self._env is None:
            raise RuntimeError("Environment is not started")

        source_value = scene.skills_dir
        source = str(source_value)
        local_source = Path(source).expanduser()
        if isinstance(source_value, os.PathLike) and local_source.is_dir():
            remote_source = f"/skills/{_safe_skill_name(scene.name)}"
            await _ensure_sandbox_dir(self._env, Path(remote_source).parent)
            await self._env.upload_dir(local_source, remote_source)
        elif source.startswith("/"):
            remote_source = source
        elif local_source.is_dir():
            remote_source = f"/skills/{_safe_skill_name(scene.name)}"
            await _ensure_sandbox_dir(self._env, Path(remote_source).parent)
            await self._env.upload_dir(local_source, remote_source)
        else:
            raise FileNotFoundError(f"Scene skills_dir not found: {scene.skills_dir}")

        home = (
            f"/home/{self._config.sandbox_user}"
            if self._config.sandbox_user
            else "/root"
        )
        for role in scene.roles:
            agent_cfg = AGENTS.get(role.agent)
            if not agent_cfg or not agent_cfg.skill_paths:
                continue
            await _link_skill_paths(
                self._env,
                remote_source,
                agent_cfg.skill_paths,
                home,
                self._agent_cwd,
                self._config.sandbox_user,
            )

    async def _run_scene(self, scene: Scene) -> None:
        """Execute one scene: for each turn, connect as the turn's role, execute, disconnect.

        For multi-role scenes, agents communicate via outbox files:
        an agent writes ``/app/.outbox/{recipient}.json`` with
        ``{"to": "role_name", "content": "..."}`` and the scheduler
        injects received messages into the next turn's prompt.

        Inter-role messages are persisted to ``rollout_dir/scene_messages.jsonl``.
        """
        cfg = self._config
        logger.info(
            f"[Scene] {scene.name} — {len(scene.turns)} turns, {len(scene.roles)} roles"
        )
        await self._activate_scene_skills(scene)

        role_map = {r.name: r for r in scene.roles}
        current_role: str | None = None
        multi_role = len(scene.roles) > 1
        scene_messages: list[dict] = []

        if multi_role:
            setup_cmd = f"rm -rf {self._OUTBOX_DIR} && mkdir -p {self._OUTBOX_DIR}"
            if cfg.sandbox_user:
                user = shlex.quote(cfg.sandbox_user)
                setup_cmd += f" && chown {user}:{user} {self._OUTBOX_DIR}"
            await self._env.exec(setup_cmd, timeout_sec=10)

        inbox: dict[str, list[str]] = {r.name: [] for r in scene.roles}
        turn_counter = 0

        try:
            for _i, turn in enumerate(scene.turns):
                role = role_map.get(turn.role)
                if not role:
                    raise ValueError(f"Turn references unknown role {turn.role!r}")

                if current_role != turn.role:
                    if current_role is not None:
                        await self.disconnect()
                    await self.connect_as(role)
                    current_role = turn.role

                if turn.prompt:
                    base_prompt = turn.prompt
                elif self._resolved_prompts:
                    base_prompt = self._resolved_prompts[0]
                else:
                    base_prompt = "Solve the task described in /app/instruction.md"

                pending = inbox.get(turn.role, [])
                if pending:
                    parts = [base_prompt, "\n---\nMessages from other agents:\n"]
                    parts.extend(pending)
                    prompts = ["\n".join(parts)]
                    inbox[turn.role] = []
                else:
                    prompts = [base_prompt]

                await self.execute(prompts=prompts)

                if multi_role:
                    if current_role is None:
                        raise RuntimeError("No active role after scene turn execution")
                    for recipient, content in await self._read_scene_outbox(
                        current_role
                    ):
                        turn_counter += 1
                        inbox.setdefault(recipient, []).append(
                            f"**From {current_role}:** {content}"
                        )
                        scene_messages.append(
                            {
                                "scene": scene.name,
                                "turn": turn_counter,
                                "sender": current_role,
                                "recipient": recipient,
                                "content": content,
                            }
                        )
        finally:
            if current_role is not None:
                await self.disconnect()

        if scene_messages and self._rollout_dir:
            msg_path = self._rollout_dir / "scene_messages.jsonl"
            with msg_path.open("a") as f:
                for m in scene_messages:
                    f.write(json.dumps(m) + "\n")
            logger.info(
                f"[Scene] {scene.name}: {len(scene_messages)} messages → {msg_path}"
            )

    async def _read_scene_outbox(self, sender: str) -> list[tuple[str, str]]:
        """Read and clear outbox files left by *sender*. Returns [(recipient, content), ...]."""
        result = await self._env.exec(
            f"ls {self._OUTBOX_DIR}/*.json 2>/dev/null || true",
            timeout_sec=10,
        )
        files = [
            f.strip() for f in (result.stdout or "").strip().splitlines() if f.strip()
        ]
        messages: list[tuple[str, str]] = []
        for fpath in files:
            quoted = shlex.quote(fpath)
            cat = await self._env.exec(f"cat {quoted}", timeout_sec=10)
            try:
                data = json.loads(cat.stdout or "{}")
                recipient = data.get("to", "")
                content = data.get("content", "")
                if recipient and content:
                    messages.append((recipient, content))
                    logger.info(
                        f"[Scene] outbox: {sender} → {recipient}: {content[:80]}"
                    )
            except json.JSONDecodeError:
                logger.warning(f"[Scene] invalid JSON in outbox: {fpath}")
            await self._env.exec(f"rm -f {quoted}", timeout_sec=10)
        return messages

    async def _run_user_loop(self) -> None:
        """Execute a user-driven progressive-disclosure loop.

        Each round: user.run() → connect → agent.execute() → disconnect →
        soft_verify() → build RoundResult → repeat. Stops when user.run()
        returns None or max_user_rounds is reached.
        """
        cfg = self._config
        user = cfg.user
        assert user is not None

        if len(cfg.effective_scenes) > 1:
            raise ValueError(
                "User-driven loops operate on a single scene. "
                f"Got {len(cfg.effective_scenes)} scenes."
            )
        scene = cfg.effective_scenes[0]
        if len(scene.roles) != 1:
            raise ValueError(
                "User-driven loops require a single-role scene. "
                f"Got {len(scene.roles)} roles."
            )
        role = scene.roles[0]

        instruction = (
            self._resolved_prompts[0]
            if self._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        # Oracle access: read /solution before the agent runs, then remove it
        solution: str | None = None
        if cfg.oracle_access:
            cat = await self._env.exec(
                "cat /solution/solve.sh 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )
            solution = (cat.stdout or "").strip() or None

        await user.setup(instruction, solution)

        # Hide oracle files from agent — move rather than delete so the
        # final verify() can still access /solution if the verifier needs it.
        if cfg.oracle_access:
            await self._env.exec(
                "mv /solution /solution_oracle_backup 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )

        round_result: RoundResult | None = None
        rounds_log: list[dict] = []

        for round_num in range(cfg.max_user_rounds):
            try:
                prompt = await user.run(round_num, instruction, round_result)
            except Exception as e:
                self._error = f"user.run() failed at round {round_num}: {e}"
                logger.error(self._error, exc_info=True)
                break

            if prompt is None:
                logger.info(f"[User] stopped at round {round_num}")
                break

            logger.info(
                f"[User] round {round_num}: prompt={prompt[:80]!r}..."
                if len(prompt) > 80
                else f"[User] round {round_num}: prompt={prompt!r}"
            )

            # Fresh ACP session each round — agent starts clean but sees
            # its previous workspace changes in the shared sandbox.
            traj_before = len(self._trajectory)
            try:
                await self.connect_as(role)
                await self.execute(prompts=[prompt])
            finally:
                await self.disconnect()

            round_trajectory = self._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            # Soft verify: run tests after agent disconnected but before
            # next round. Temporarily restore /solution so the verifier can
            # access it, then re-hide before the next agent round.
            if cfg.oracle_access:
                await self._env.exec(
                    "mv /solution_oracle_backup /solution 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
            try:
                rewards, verifier_output, verifier_error = await self.soft_verify()
            finally:
                if cfg.oracle_access:
                    await self._env.exec(
                        "mv /solution /solution_oracle_backup 2>/dev/null || true",
                        user="root",
                        timeout_sec=10,
                    )

            round_result = RoundResult(
                round=round_num,
                trajectory=round_trajectory,
                rewards=rewards,
                verifier_output=verifier_output,
                verifier_error=verifier_error,
                n_tool_calls=round_tools,
            )

            rounds_log.append(
                {
                    "round": round_num,
                    "prompt": prompt,
                    "rewards": rewards,
                    "verifier_error": verifier_error,
                    "n_tool_calls": round_tools,
                    "n_trajectory_events": len(round_trajectory),
                }
            )

            logger.info(
                f"[User] round {round_num} done: rewards={rewards}, tools={round_tools}"
            )

        # Persist round log
        if rounds_log and self._rollout_dir:
            log_path = self._rollout_dir / "user_rounds.jsonl"
            with log_path.open("w") as f:
                for entry in rounds_log:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"[User] {len(rounds_log)} rounds → {log_path}")

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role.

        Installs the role's agent binary and credentials if it differs
        from the primary agent (which was set up in install_agent()).
        Updates _agent_launch so disconnect() kills the correct process.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        # Merge cfg.agent_env (config-level) with role.env (role-specific) so
        # provider creds from YAML reach the agent. role.env wins on overlap.
        disallow_web_tools = getattr(self, "_disallow_web_tools", None)
        if disallow_web_tools is None:
            disallow_web_tools = _task_disallows_internet(getattr(self, "_task", None))
        disallow_web_tools = bool(disallow_web_tools and role.agent != "oracle")
        agent_launch = _agent_launch_with_web_policy(
            role.agent,
            disallow=disallow_web_tools,
        )
        agent_env = _apply_web_policy(
            resolve_agent_env(
                role.agent,
                role.model,
                {**(cfg.agent_env or {}), **(role.env or {})},
            ),
            disallow=disallow_web_tools,
        )
        agent_env, self._provider_runtime = await ensure_bedrock_proxy_runtime(
            agent=role.agent,
            agent_env=agent_env,
            model=role.model,
            runtime=getattr(self, "_provider_runtime", None),
            environment=cfg.environment,
        )
        agent_env, self._usage_runtime = await ensure_usage_proxy_runtime(
            agent=role.agent,
            agent_env=agent_env,
            model=role.model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
        )

        role_agent_differs = role.agent != cfg.primary_agent
        needs_role_credentials = (
            role_agent_differs or role.model != cfg.primary_model or bool(role.env)
        )
        if role_agent_differs:
            agent_cfg = await install_agent(self._env, role.agent, rollout_dir)
        else:
            agent_cfg = getattr(self, "_agent_cfg", None)
        if needs_role_credentials:
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await write_credential_files(
                self._env,
                role.agent,
                agent_env,
                agent_cfg,
                role.model,
                cred_home,
            )
            if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                await upload_subscription_auth(self._env, role.agent, cred_home)
            await apply_web_tool_policy(
                self._env,
                role.agent,
                agent_cfg,
                cred_home,
                disallow=disallow_web_tools,
            )

        self._agent_launch = agent_launch

        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=role.agent,
            agent_launch=agent_launch,
            agent_env=agent_env,
            sandbox_user=cfg.sandbox_user,
            model=role.model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
        )
        self._active_role = role

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # ── Internal helpers ──

    async def _probe_sandbox_health(self) -> None:
        """Quick health probe after transport death. Enriches _transport_error_info.

        Guards ENG-148: distinguishes Daytona session killed vs agent crash.
        """
        if self._transport_error_info is None or self._env is None:
            return
        try:
            result = await asyncio.wait_for(
                self._env.exec("echo __BENCHFLOW_HEALTH_OK__", timeout_sec=10),
                timeout=15,
            )
            stdout = str(getattr(result, "stdout", "") or "").strip()
            raw_rc = getattr(result, "return_code", None)
            rc = int(raw_rc) if isinstance(raw_rc, (int, float)) else None
            if "__BENCHFLOW_HEALTH_OK__" in stdout:
                self._transport_error_info["sandbox_reachable"] = True
                self._transport_error_info["sandbox_probe_rc"] = rc
            else:
                self._transport_error_info["sandbox_reachable"] = False
                self._transport_error_info["sandbox_probe_rc"] = rc
                self._transport_error_info["sandbox_probe_stdout"] = stdout[:200]
        except Exception as probe_err:
            self._transport_error_info["sandbox_reachable"] = False
            self._transport_error_info["sandbox_probe_error"] = str(probe_err)[:200]

    def _classify_acp_error(self, e: ACPError) -> str:
        if "Invalid API key" in e.message:
            from benchflow.agents.env import check_subscription_auth
            from benchflow.agents.registry import infer_env_key_for_model

            key = (
                infer_env_key_for_model(self._config.primary_model)
                if self._config.primary_model
                else None
            )
            if key and check_subscription_auth(self._config.primary_agent, key):
                return (
                    f"{key} was rejected as invalid. "
                    f"Subscription auth credentials exist — unset the env var "
                    f"to use them: env -u {key} <command>"
                )
        return str(e)

    def _write_llm_trajectory(self, usage_runtime: Any) -> None:
        """Persist captured provider HTTP exchanges as JSONL."""
        if self._rollout_dir is None:
            return
        trajectory = getattr(getattr(usage_runtime, "server", None), "trajectory", None)
        if trajectory is None or not trajectory.exchanges:
            return
        traj_dir = self._rollout_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "llm_trajectory.jsonl").write_text(
            trajectory.to_jsonl(redact_keys=True)
        )

    def _build_result(self) -> RolloutResult:
        rollout_dir = self._require_rollout_dir()
        return _build_rollout_result(
            rollout_dir,
            task_name=self._config.task_path.name,
            rollout_name=self._rollout_name or "",
            agent=self._config.primary_agent,
            agent_name=self._agent_name,
            model=self._config.primary_model,
            n_tool_calls=self._n_tool_calls,
            prompts=self._resolved_prompts,
            error=self._error,
            verifier_error=self._verifier_error,
            trajectory=self._trajectory,
            partial_trajectory=self._partial_trajectory,
            trajectory_source=self._trajectory_source,
            rewards=self._rewards,
            started_at=self._require_started_at(),
            timing=self._timing,
            scenes=self._config.effective_scenes,
            evolved_skills=self._evolved_skills,
            source_provenance=self._config.source_provenance,
            idle_timeout_info=self._idle_timeout_info,
            sandbox_startup_info=self._sandbox_startup_info,
            transport_error_info=self._transport_error_info,
            verifier_timeout_info=self._verifier_timeout_info,
            **self._usage_metrics,
        )

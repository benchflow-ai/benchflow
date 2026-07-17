"""Environment, agent, oracle, verify and prompt setup helpers for the rollout.

These free functions cover the *input* side of the 5-phase lifecycle: reading
the task instruction, applying the no-web policy, resolving the agent cwd,
initialising the rollout directory tree, resolving prompts, starting the
sandbox + uploading task files, running oracle mode, publishing the trajectory
for the verifier, and running the verifier itself.

Split out of ``rollout.py`` for cohesion; every name is re-exported from
:mod:`benchflow.rollout` so existing imports — ``_init_rollout``,
``_resolve_prompts``, ``_run_oracle``, ``_start_env_and_upload``,
``_verify_rollout`` (``sdk.py``), ``_resolve_agent_cwd`` /
``_start_env_and_upload`` / ``_verify_rollout`` (``task/acceptance_live.py``) —
keep resolving unchanged.

Note on patching: callers that patch ``benchflow.rollout.default_rollout_planes``
always thread an explicit ``planes`` into these helpers (via ``self._planes``),
so the ``planes or default_rollout_planes()`` fallback here is never the patched
seam — it only fires for direct, unpatched calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shlex
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.contracts import RolloutPlanes, default_rollout_planes
from benchflow.diagnostics import VerifierTimeoutDiagnostic
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.rewards.validation import (
    declared_reward_range,
    reward_lenient_from_env,
    validate_reward_map,
)
from benchflow.rollout._results import _DIAG_TRUNCATE
from benchflow.trajectories.types import redact_acp_trajectory_jsonl

logger = logging.getLogger(__name__)

_DISALLOW_WEB_TOOLS_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"
_TASK_SETUP_SANDBOX_DIR = PurePosixPath("/tmp/benchflow-task-setup")


def _task_disallows_internet(task: Any) -> bool:
    """Return True when task config requests no internet for the agent task."""
    env_config = getattr(getattr(task, "config", None), "environment", None)
    return getattr(env_config, "allow_internet", True) is False


def _read_task_instruction(task_path: Path) -> str:
    """Read the agent-facing instruction from legacy files or ``task.md``."""
    document_path = task_path / "task.md"
    if document_path.exists():
        from benchflow.task.document import TaskDocument

        return TaskDocument.from_path(document_path).instruction.strip()

    instruction_path = task_path / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text().strip()
    raise FileNotFoundError(f"Task missing instruction.md or task.md: {task_path}")


def _environment_uses_prebuilt_image(
    env_config: object | None, environment_manifest: EnvironmentManifest | None
) -> bool:
    """Return True when sandbox startup will skip the task Dockerfile build."""
    if env_config is not None and getattr(env_config, "docker_image", None):
        return True
    if environment_manifest is None:
        return False
    from benchflow.environment.manifest import resolve_manifest_image

    return bool(resolve_manifest_image(environment_manifest))


def _apply_web_policy(agent_env: dict[str, str], *, disallow: bool) -> dict[str, str]:
    """Inject BenchFlow's no-web policy marker into agent env when requested."""
    if not disallow:
        return agent_env
    return {**agent_env, _DISALLOW_WEB_TOOLS_ENV: "1"}


def _agent_launch_with_web_policy(
    agent: str, *, disallow: bool, planes: RolloutPlanes | None = None
) -> str:
    """Return launch command, appending the agent's no-web launch knob if any."""
    return (planes or default_rollout_planes()).agent_launch(
        agent, disallow_web_tools=disallow
    )


# Package runners whose *next* token is a subcommand (``uv run <agent>``,
# ``npx <agent>``), not the agent binary — skip both.
_PACKAGE_RUNNERS = frozenset({"uv", "uvx", "npx", "npm", "pnpm", "yarn", "pipx"})
# Interpreters that launch an agent but never *identify* it: keying a
# ``pkill -f`` pattern on one of these reaps every interpreter process in the
# sandbox, not just the agent. (e.g. ``pythonX.Y`` is matched by the
# ``python`` prefix in :func:`_is_generic_interpreter`.)
_GENERIC_INTERPRETERS = (
    frozenset(
        {
            "python",
            "pypy",
            "node",
            "nodejs",
            "deno",
            "bun",
            "ruby",
            "perl",
            "sh",
            "bash",
            "dash",
            "env",
        }
    )
    | _PACKAGE_RUNNERS
)
# Subcommands consumed by a package runner before the agent binary appears.
_RUNNER_SUBCOMMANDS = frozenset({"run", "tool", "exec", "x"})
# A leading ``FOO=bar`` environment-variable assignment (e.g. harvey-lab's
# ``HARVEY_LABS_ROOT=/opt/harvey-labs ... python <shim>``).
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
# Shell operators that separate commands; the agent invocation is the final
# command in a ``setup && … && <agent>`` launch (openhands).
_SHELL_SEP_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


def _is_generic_interpreter(basename: str) -> bool:
    return basename in _GENERIC_INTERPRETERS or basename.startswith("python")


def _agent_process_kill_pattern(agent_launch: str) -> str | None:
    """Return a ``pkill -f`` pattern that targets the *agent* process only.

    The pattern must identify the agent — never the interpreter that launches
    it. Several agents launch as ``<venv>/bin/python <shim>`` (deepagents,
    harvey-lab); naively keying on the first token then yields ``python``, and
    ``pkill -f python`` reaps **every** Python process in the sandbox. That
    includes Environment-plane services — mock APIs run as console scripts via
    their ``#!/usr/bin/python`` shebang, so their argv is
    ``/usr/bin/python /usr/local/bin/<svc> …`` — and Python verifiers. The
    service is then dead when the verifier reads its live state, scoring 0.0
    on an otherwise-correct rollout (BF-10).

    Resolution:

    1. Reduce to the **last shell-command segment** — for a
       ``export … && mkdir … && <agent>`` launch (openhands) the agent is the
       final command, not the leading ``export`` builtin.
    2. Walk that segment and key on the agent's own binary/shim token, skipping
       ``FOO=bar`` env assignments, generic interpreters, package-runner
       subcommands (``uv run <agent>``), and flags.

    Returns ``None`` when nothing specific enough is found — better to skip the
    cleanup pkill than to fire a sandbox-wide one.
    """
    segments = [s for s in _SHELL_SEP_RE.split(agent_launch.strip()) if s.strip()]
    if not segments:
        return None

    after_runner = False
    for token in segments[-1].split():
        if _ENV_ASSIGN_RE.match(token):  # FOO=bar prefix
            continue
        if token.startswith("-"):  # a flag (e.g. python -m, gemini --acp)
            continue
        basename = PurePosixPath(token).name
        if not basename:
            continue
        if _is_generic_interpreter(basename):  # too broad to pkill on
            after_runner = basename in _PACKAGE_RUNNERS
            continue
        if after_runner and basename in _RUNNER_SUBCOMMANDS:  # `uv run` etc.
            after_runner = False
            continue
        return rf"(^|[ /]){re.escape(basename)}( |$)"
    return None


def _configured_task_workdir(task: Any) -> str | None:
    """Return the task-declared sandbox workdir, if any."""

    env_config = getattr(getattr(task, "config", None), "environment", None)
    value = getattr(env_config, "workdir", None)
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return _dockerfile_task_workdir(task)


def _dockerfile_task_workdir(task: Any) -> str | None:
    """Return the task Dockerfile's effective WORKDIR, when simple to resolve."""

    paths = getattr(task, "paths", None)
    environment_dir = getattr(paths, "environment_dir", None)
    if environment_dir is None:
        return None
    dockerfile_path = Path(environment_dir) / "Dockerfile"
    if not dockerfile_path.exists():
        return None

    workdir = PurePosixPath("/")
    try:
        lines = dockerfile_path.read_text().splitlines()
    except OSError:
        return None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        instruction, _, rest = stripped.partition(" ")
        if instruction.upper() != "WORKDIR" or not rest.strip():
            continue
        try:
            parts = shlex.split(rest, comments=False, posix=True)
        except ValueError:
            return None
        if not parts or "$" in parts[0]:
            return None
        next_path = PurePosixPath(parts[0])
        workdir = next_path if next_path.is_absolute() else workdir / next_path
    if workdir == PurePosixPath("/"):
        return None
    return workdir.as_posix()


def _validate_agent_workdir(workdir: str) -> None:
    path = PurePosixPath(workdir)
    if not path.is_absolute() or path == PurePosixPath("/"):
        raise ValueError("environment.workdir must be an absolute non-root path")


async def _resolve_agent_cwd(env: Any, task: Any) -> str:
    """Resolve and materialize the workspace path used by agents and verifiers."""

    configured = _configured_task_workdir(task)
    if configured is None:
        cwd_result = await env.exec("pwd", timeout_sec=10)
        probed = (cwd_result.stdout or "").strip()
        if probed and probed != "/":
            return probed
        fallback = "/root"
        result = await env.exec(
            f"mkdir -p {fallback} && cd {fallback} && pwd",
            user="root",
            timeout_sec=10,
        )
        return (getattr(result, "stdout", "") or "").strip() or fallback

    _validate_agent_workdir(configured)
    quoted = shlex.quote(configured)
    result = await env.exec(
        f"mkdir -p {quoted} && cd {quoted} && pwd",
        user="root",
        timeout_sec=10,
    )
    return_code = getattr(result, "return_code", getattr(result, "exit_code", 0))
    if isinstance(return_code, int) and return_code != 0:
        stderr = (getattr(result, "stderr", "") or "").strip()
        raise RuntimeError(
            f"failed to prepare environment.workdir {configured!r}: {stderr}"
        )
    return (getattr(result, "stdout", "") or "").strip() or configured


def _skill_nudge(agent_env: dict[str, str] | None) -> str:
    """Read skill nudge from explicit agent env or the host environment."""
    return (agent_env or {}).get("BENCHFLOW_SKILL_NUDGE") or os.environ.get(
        "BENCHFLOW_SKILL_NUDGE", ""
    )


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


def _resolve_prompts(
    task_path: Path,
    prompts: list[str | None] | None,
    skills_dir: str | Path | None = None,
    task_skills_dir: str | Path | None = None,
    skill_nudge: str = "",
    agent: str | None = None,
    planes: RolloutPlanes | None = None,
) -> list[str]:
    """Read the task instruction and resolve prompt list."""
    instruction = _read_task_instruction(task_path)

    if skill_nudge:
        skill_display_path = "~/.claude/skills"
        if agent:
            agent_cfg = (planes or default_rollout_planes()).agent_config(agent)
            if agent_cfg and agent_cfg.skill_paths:
                skill_display_path = agent_cfg.skill_paths[0].replace("$HOME", "~")

        skills = []
        for src in [skills_dir, task_skills_dir]:
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


async def _start_env_and_upload(
    env: Any,
    task_path: Path,
    timing: dict,
    *,
    skip_start: bool = False,
    on_started: Callable[[], None] | None = None,
) -> None:
    """Start environment and upload task files.

    ``skip_start=True`` is used when the sandbox was created and started
    by the caller (Runtime with a live Environment, #388) — we still
    upload task files but must not re-run ``start()`` since most sandbox
    backends (e.g. daytona) are not idempotent.

    ``on_started`` runs once the sandbox exists but *before* any upload
    (#554/#563). Persisting the sandbox id here — not after upload —
    closes the failure window where an upload error or interrupt after
    Daytona creation would otherwise leave no ``sandbox.json`` to audit
    or clean up.
    """
    if skip_start:
        logger.info(f"Reusing caller-owned environment: {task_path.name}")
        timing["environment_setup"] = 0.0
    else:
        logger.info(f"Starting environment: {task_path.name}")
        t0 = datetime.now()
        await env.start(force_build=False)
        timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    if on_started is not None:
        on_started()
    instruction_path = task_path / "instruction.md"
    if instruction_path.exists() and not (task_path / "task.md").exists():
        await env.upload_file(instruction_path, "/instruction.md")
    else:
        instruction = _read_task_instruction(task_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(instruction)
            f.write("\n")
            temp_instruction = Path(f.name)
        try:
            await env.upload_file(temp_instruction, "/instruction.md")
        finally:
            temp_instruction.unlink(missing_ok=True)
    from benchflow.task.paths import SandboxPaths, TaskPaths

    paths = TaskPaths(task_path)
    if paths.solution_dir.is_dir():
        sandbox_paths = SandboxPaths()
        target_dir = (
            sandbox_paths.oracle_dir
            if paths.uses_native_oracle_dir
            else sandbox_paths.solution_dir
        )
        await env.upload_dir(paths.solution_dir, str(target_dir))
    await _run_task_setup_hooks(env, task_path)


async def _run_task_setup_hooks(env: Any, task_path: Path) -> None:
    """Run verifier/setup pre-command hooks before the agent starts."""
    from benchflow.task.paths import TaskPaths

    paths = TaskPaths(task_path)
    setup_dir = paths.tests_dir / "setup"
    pre_command = setup_dir / "pre_command.sh"
    if not pre_command.is_file():
        return

    remote_dir = str(_TASK_SETUP_SANDBOX_DIR)
    await env.exec(
        f"rm -rf -- {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}",
        user="root",
        timeout_sec=10,
    )
    try:
        await env.upload_dir(setup_dir, remote_dir)
        command = (
            f"chmod +x {shlex.quote(remote_dir)}/pre_command.sh && "
            f"cd {shlex.quote(remote_dir)} && "
            "DEBIAN_FRONTEND=noninteractive ./pre_command.sh"
        )
        result = await env.exec(command, user="root", timeout_sec=300)
        return_code = getattr(result, "return_code", getattr(result, "exit_code", 0))
        if isinstance(return_code, int) and return_code != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            stdout = (getattr(result, "stdout", "") or "").strip()
            detail = stderr or stdout or f"rc={return_code}"
            raise RuntimeError(f"task setup pre_command.sh failed: {detail}")
    finally:
        await env.exec(
            f"rm -rf -- {shlex.quote(remote_dir)}",
            user="root",
            timeout_sec=10,
        )


async def _run_oracle(
    env: Any, task_path: Path, timeout: int, sandbox_user: str | None = None
) -> tuple[list[dict], str]:
    """Run oracle mode (oracle/solve.sh or legacy solution/solve.sh)."""
    from benchflow.task import Task, resolve_env_vars
    from benchflow.task.paths import SandboxPaths

    logger.info("Oracle mode: running oracle solve.sh")
    task = Task(task_path)
    if not task.paths.solve_path.exists():
        raise FileNotFoundError(
            f"Oracle requires oracle/solve.sh or legacy solution/solve.sh: {task_path}"
        )
    sandbox_paths = SandboxPaths()
    oracle_dir = (
        sandbox_paths.oracle_dir
        if task.paths.uses_native_oracle_dir
        else sandbox_paths.solution_dir
    )
    oracle_command_label = (
        "oracle/solve.sh" if task.paths.uses_native_oracle_dir else "solution/solve.sh"
    )
    oracle_script = shlex.quote(str(oracle_dir / "solve.sh"))
    if sandbox_user:
        oracle_cmd = f"DEBIAN_FRONTEND=noninteractive bash {oracle_script}"
        cmd = (
            f"su -s /bin/bash {shlex.quote(sandbox_user)} -c {shlex.quote(oracle_cmd)}"
        )
    else:
        cmd = f"bash {oracle_script}"
    oracle_env: dict[str, str] = {"DEBIAN_FRONTEND": "noninteractive"}
    if task.config.solution.env:
        oracle_env.update(resolve_env_vars(task.config.solution.env))
    exec_kwargs: dict[str, object] = {
        "env": oracle_env,
        "timeout_sec": timeout,
    }
    if sandbox_user:
        exec_kwargs["user"] = "root"
    result = await env.exec(
        f"{cmd} > /logs/agent/oracle.txt 2>&1",
        **exec_kwargs,
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
            "command": oracle_command_label,
            "return_code": result.return_code,
            "stdout": (preview.stdout or "")[:_DIAG_TRUNCATE],
        }
    ]
    return trajectory, "oracle"


async def _publish_trajectory_for_verifier(
    env, trajectory: list[dict], agent_dir: Path
) -> None:
    """Make the captured ACP trajectory available inside /logs for verifiers.

    Also writes the same payload to the host rollout ``agent/`` dir so the
    artifact set matches across backends: Docker bind-mounts ``/logs/agent``
    to the host agent dir, while remote sandboxes (Daytona, Modal) never
    mirror the published file back.
    """
    if not trajectory:
        return
    payload = redact_acp_trajectory_jsonl(trajectory) + "\n"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "acp_trajectory.jsonl").write_text(payload)
    await env.exec("mkdir -p /logs/agent", user="root", timeout_sec=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(payload)
        tmp_path = f.name
    try:
        await env.upload_file(tmp_path, "/logs/agent/acp_trajectory.jsonl")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)


async def _download_rollout_artifacts(env: Any, rollout_paths: Any) -> None:
    """Copy sandbox-side /logs/artifacts back for non-mounted providers."""
    if getattr(env, "is_mounted", False):
        return
    from benchflow.task.paths import SandboxPaths

    try:
        await env.download_dir(
            str(SandboxPaths.artifacts_dir),
            rollout_paths.artifacts_dir,
        )
    except Exception as exc:
        logger.warning("Failed to download rollout artifacts: %s", exc)


async def _verify_rollout(
    env: Any,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    planes: RolloutPlanes,
    sandbox_user: str | None = None,
    workspace: str | None = None,
) -> tuple[dict | None, str | None, VerifierTimeoutDiagnostic | None]:
    """Run verifier with pre-verification hardening.

    Returns ``(rewards, verifier_error, verifier_timeout_diagnostic)``. The
    diagnostic is non-``None`` only when the verifier exceeded its timeout
    budget — the agent-error channel is unused (issue #503).
    """
    rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    verifier_error = None
    verifier_timeout: VerifierTimeoutDiagnostic | None = None
    timeout_budget = task.config.verifier.timeout_sec
    try:
        await planes.harden_before_verify(env, task, sandbox_user, workspace=workspace)
        logger.info("Running verifier...")
        verifier = planes.verifier(
            task=task,
            rollout_paths=rollout_paths,
            sandbox=env,
            workspace=workspace,
        )
        verifier_result = await asyncio.wait_for(
            verifier.verify(),
            timeout=timeout_budget,
        )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = _ensure_canonical_rewards(verifier_result.rewards, task=task)
        logger.info(f"Rewards: {rewards}")
    except TimeoutError:
        elapsed = (datetime.now() - t0).total_seconds()
        timing["verifier"] = elapsed
        verifier_error = f"verifier timed out after {timeout_budget}s"
        verifier_timeout = VerifierTimeoutDiagnostic(
            timeout_budget_sec=timeout_budget,
            elapsed_sec=round(elapsed, 1),
            task_name=task.name,
        )
        rewards = None
        logger.error(verifier_error)
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier crashed: {e}"
        rewards = None
        logger.error(verifier_error)
    return rewards, verifier_error, verifier_timeout


def _ensure_canonical_rewards(rewards: dict | None, *, task: Any = None) -> dict:
    # Honour the same BENCHFLOW_REWARD_LENIENT toggle and task-declared
    # ``[verifier] reward_range`` (BF-8) as the reward.json parse path so the
    # final canonicalization gate stays consistent with how the verifier
    # accepted the map (no-op unless the operator/task opts in).
    return validate_reward_map(
        rewards,
        source="verifier",
        lenient=reward_lenient_from_env(),
        reward_range=declared_reward_range(task),
    )


def _install_docker_compat(planes: RolloutPlanes | None = None) -> None:
    """Activate the Docker DinD compatibility shim.

    Called from ``Rollout.__init__`` so importing ``benchflow.rollout`` has
    no side effects on the Docker sandbox. The underlying patch is
    idempotent — safe to call once per rollout construction.
    """
    (planes or default_rollout_planes()).install_docker_compat()

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
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import re
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Coroutine
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow._types import Role, Scene, Turn

# --- Façade re-exports -------------------------------------------------------
# The helper groups live in private submodules. The ``import x as x`` aliases
# re-export the helpers that ``sdk.py``, ``self_gen.py``,
# ``task/acceptance_live.py`` and the tests import from ``benchflow.rollout``.
#
# Re-importing these into this module's namespace also preserves patching: tests
# that patch ``benchflow.rollout._verify_rollout`` / ``_scrape_agent_trajectory``
# / ``_capture_session_trajectory`` / ``default_rollout_planes`` keep affecting
# the ``Rollout`` methods that call those names, because those methods stay
# defined in this module.
from benchflow._utils.live_activity import ActivitySnapshot, SessionCounters
from benchflow._utils.text import describe_exception
from benchflow.acp.types import McpServerSpec
from benchflow.agents.codex_connector_policy import (
    effective_apps_policy,
    enforce_codex_apps_policy,
)
from benchflow.agents.credentials import upload_credential
from benchflow.agents.registry import (
    AGENTS,
    infer_env_key_for_model,
    is_scripted_agent,
    resolve_agent,
)
from benchflow.contracts import (
    AgentProtocolError,
    AskUserRequest,
    BaseUser,
    Environment,
    SandboxStartupFailure,
    default_rollout_planes,
)
from benchflow.diagnostics import (
    AgentModelNotOfferedError,
    AgentPromptTimeoutError,
    IntegrationFailureDiagnostic,
    ProviderApiErrorDiagnostic,
    RolloutDiagnostics,
    SuspectedApiErrorDiagnostic,
)
from benchflow.loop_strategies import (
    LoopStrategyUser,
    collect_loop_metadata,
    loop_block,
)
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.review.automatic import PreparedReview
from benchflow.review.outcome import ScoringResult, scoring_from_result
from benchflow.review.persistence import scoring_lock
from benchflow.rollout import _deadline as _deadline
from benchflow.rollout._artifacts import collect_rollout_artifacts
from benchflow.rollout._config import GENERATED_SKILLS_ROOT as GENERATED_SKILLS_ROOT
from benchflow.rollout._config import RolloutConfig as RolloutConfig
from benchflow.rollout._results import _build_rollout_result as _build_rollout_result
from benchflow.rollout._results import (
    _environment_manifest_metadata as _environment_manifest_metadata,
)
from benchflow.rollout._results import _is_secret_env_key as _is_secret_env_key
from benchflow.rollout._results import (
    _least_permissive_option_id as _least_permissive_option_id,
)
from benchflow.rollout._results import (
    _user_confirmation_policy as _user_confirmation_policy,
)
from benchflow.rollout._results import _write_config as _write_config
from benchflow.rollout._results import _write_rewards_jsonl as _write_rewards_jsonl
from benchflow.rollout._review import (
    capture_terminal_workspace,
    finish_terminal_review,
    prepare_capture_runtime,
    prepare_terminal_result,
    prepare_terminal_review,
)
from benchflow.rollout._separate_verifier import run_separate_verifier
from benchflow.rollout._setup import (
    _agent_launch_with_web_policy as _agent_launch_with_web_policy,
)
from benchflow.rollout._setup import (
    _agent_process_kill_pattern as _agent_process_kill_pattern,
)
from benchflow.rollout._setup import _apply_prompt_prefix as _apply_prompt_prefix
from benchflow.rollout._setup import _apply_web_policy as _apply_web_policy
from benchflow.rollout._setup import (
    _ensure_canonical_rewards as _ensure_canonical_rewards,
)
from benchflow.rollout._setup import _ensure_sandbox_dir as _ensure_sandbox_dir
from benchflow.rollout._setup import (
    _environment_uses_prebuilt_image as _environment_uses_prebuilt_image,
)
from benchflow.rollout._setup import _init_rollout as _init_rollout
from benchflow.rollout._setup import _install_docker_compat as _install_docker_compat
from benchflow.rollout._setup import (
    _publish_trajectory_for_verifier as _publish_trajectory_for_verifier,
)
from benchflow.rollout._setup import _resolve_agent_cwd as _resolve_agent_cwd
from benchflow.rollout._setup import _resolve_prompts as _resolve_prompts
from benchflow.rollout._setup import _run_oracle as _run_oracle
from benchflow.rollout._setup import _start_env_and_upload as _start_env_and_upload
from benchflow.rollout._setup import (
    _task_disallows_internet as _task_disallows_internet,
)
from benchflow.rollout._setup import _task_egress_denylist as _task_egress_denylist
from benchflow.rollout._setup import _verify_rollout as _verify_rollout
from benchflow.rollout._skills import (
    _resolve_skill_creator_root as _resolve_skill_creator_root,
)
from benchflow.rollout._skills import _safe_skill_name as _safe_skill_name
from benchflow.rollout._skills import _self_gen_prompt as _self_gen_prompt
from benchflow.rollout._skills import _skill_frontmatter_name as _skill_frontmatter_name
from benchflow.rollout._usage import (
    _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT as _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT,
)
from benchflow.rollout._usage import (
    ProviderFailure as ProviderFailure,
)
from benchflow.rollout._usage import _as_nonnegative_int as _as_nonnegative_int
from benchflow.rollout._usage import _native_acp_usage_delta as _native_acp_usage_delta
from benchflow.rollout._usage import (
    _provider_api_failure_summary_from_runtime as _provider_api_failure_summary_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_auth_status_from_runtime as _provider_auth_status_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_failure_from_runtime as _provider_failure_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_failure_from_status as _provider_failure_from_status,
)
from benchflow.rollout._usage import (
    _zero_native_acp_usage_metrics as _zero_native_acp_usage_metrics,
)
from benchflow.rollout._usage import classify_api_failure as classify_api_failure

# Step / user-loop drivers live in ``_user_loop`` as free functions taking the
# Rollout; the thin methods below delegate to these engine aliases.
from benchflow.rollout._user_loop import (
    _activate_step_skills as _activate_step_skills_engine,
)
from benchflow.rollout._user_loop import (
    _export_generated_skills as _export_generated_skills_engine,
)
from benchflow.rollout._user_loop import _run_steps as _run_steps_engine
from benchflow.rollout._user_loop import _run_user_loop as _run_user_loop_engine
from benchflow.rollout.task_runtime import BashToolResult as BashToolResult
from benchflow.rollout.task_runtime import TaskRuntime as TaskRuntime
from benchflow.rollout.task_runtime import TaskRuntimeConfig as TaskRuntimeConfig
from benchflow.rollout.task_runtime import TaskRuntimeResult as TaskRuntimeResult
from benchflow.rollout_branch import (
    BranchChild,
    ChildRunner,
    IdentifiedChildRunner,
    require_safe_branch_world,
)
from benchflow.rollout_branch import branch as _branch_engine
from benchflow.sandbox.egress_denylist import EgressDenylist, denylist_agent_env
from benchflow.sandbox.metadata import persist_sandbox_info
from benchflow.sandbox.native_oauth import (
    allowlist_model_transport,
    native_oauth_egress_policy,
    validate_native_oauth_transport,
)
from benchflow.scenes import compile_scenes_to_steps
from benchflow.skill_policy import SKILL_MODE_NO_SKILL as SKILL_MODE_NO_SKILL
from benchflow.skill_policy import (
    SKILL_MODE_SELF_GEN,
    TaskSkillPolicy,
    resolve_task_skill_policy,
    strip_task_bundled_skills,
    task_bundled_skills_dir,
)
from benchflow.skill_policy import SKILL_MODE_WITH_SKILL as SKILL_MODE_WITH_SKILL
from benchflow.task.verifier_sandbox import separate_verifier_requested
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _parse_provider_tool_evidence,
    _reconcile_tool_evidence,
    _scrape_agent_trajectory,
    make_trajectory_sink,
)
from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step
from benchflow.usage_tracking import (
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)


def _nop_trajectory() -> tuple[list[dict], str]:
    """The ``nop`` agent's one-event trajectory: nothing ran."""
    return (
        [
            {
                "type": "nop",
                "note": "nop agent: nothing ran; the verifier scores the untouched workspace",
            }
        ],
        "nop",
    )


_SETUP_COMMAND_LOCK_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Lifecycle phases from verify() onward. The agent will not run again in this
# rollout once one of these is set, so nothing may rewind ``_phase`` out of
# them — the live dashboard renders the phase as a label and a backwards step
# reads as the run having restarted (see disconnect()).
_TERMINAL_PHASES = frozenset({"verifying", "verified", "reviewing", "cleaned"})


_MCP_TRANSPORT_TO_ACP_TYPE = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "http",
}


def _task_mcp_specs(task: Any) -> list[McpServerSpec]:
    """Map the task's ``[[sandbox.mcp_servers]]`` entries to ACP specs.

    This is the composition seam between the task-config layer
    (``MCPServerConfig``) and the ACP protocol layer (``McpServerSpec``) — kept
    here, in the rollout, so ``acp/`` stays free of any task-config dependency.
    The resulting specs are attached to every ACP session the rollout opens
    (``session/new``), making task-declared MCP servers — e.g. a Playwright MCP
    — reachable by the agent. Returns ``[]`` when the task declares none,
    preserving the historical default of attaching no MCP servers.
    """
    env_config = getattr(getattr(task, "config", None), "sandbox", None)
    configs = getattr(env_config, "mcp_servers", None) or []
    return [
        McpServerSpec(
            name=config.name,
            type=_MCP_TRANSPORT_TO_ACP_TYPE.get(config.transport, config.transport),
            command=config.command,
            args=list(config.args),
            cwd=config.cwd,
            env=dict(config.env),
            url=config.url,
            headers=dict(config.headers),
            tools=list(config.tools) if config.tools is not None else None,
            include_tags=list(config.include_tags)
            if config.include_tags is not None
            else None,
            exclude_tags=list(config.exclude_tags)
            if config.exclude_tags is not None
            else None,
        )
        for config in configs
    ]


def _agent_uses_native_task_mcp_config(
    agent: str, agent_cfg: Any | None = None
) -> bool:
    if agent_cfg is None:
        agent_base = agent.split()[0]
        agent_cfg = AGENTS.get(agent_base)
    return getattr(agent_cfg, "task_mcp_transport", "acp") == "native-config"


def _task_mcp_specs_for_agent(
    agent: str, task: Any, agent_cfg: Any | None = None
) -> list[McpServerSpec]:
    """Return MCP specs to pass over ACP for this agent.

    Agents may declare a native task-MCP config path in their registry entry.
    Those agents load task MCP servers from that file, so BenchFlow must not
    also send the same servers over ACP ``session/new``.
    """

    if _agent_uses_native_task_mcp_config(agent, agent_cfg):
        return []
    return _task_mcp_specs(task)


def _fastmcp_task_mcp_config(task: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Return FastMCP ``mcp.json`` content for task MCP servers."""

    servers: dict[str, dict[str, Any]] = {}
    for spec in _task_mcp_specs(task):
        filters: dict[str, list[str]] = {}
        if spec.tools is not None:
            filters["tools"] = list(spec.tools)
        if spec.include_tags is not None:
            filters["include_tags"] = list(spec.include_tags)
        if spec.exclude_tags is not None:
            filters["exclude_tags"] = list(spec.exclude_tags)

        if spec.type == "stdio":
            server = {
                "command": spec.command,
                "args": list(spec.args),
                "env": dict(spec.env),
                "transport": "stdio",
                "enabled": True,
                **filters,
            }
            if spec.cwd is not None:
                server["cwd"] = spec.cwd
        else:
            server = {
                "url": spec.url,
                "transport": "sse" if spec.type == "sse" else "http",
                "headers": dict(spec.headers),
                "enabled": True,
                **filters,
            }
        servers[spec.name] = server
    return {"mcpServers": servers}


async def _install_native_task_mcp_config(
    env: Any,
    task: Any,
    *,
    agent_cfg: Any | None,
    cred_home: str,
    owner: str | None,
) -> None:
    if getattr(agent_cfg, "task_mcp_transport", "acp") != "native-config":
        return
    config_path = getattr(agent_cfg, "task_mcp_config_path", "")
    if not config_path:
        return
    config = _fastmcp_task_mcp_config(task)
    if not config["mcpServers"]:
        return
    target = (
        config_path if config_path.startswith("/") else f"{cred_home}/{config_path}"
    )
    await upload_credential(
        env,
        target,
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        owner=owner,
    )


def _setup_command_lock_path(lock_name: str) -> Path:
    lock_dir = Path(
        os.environ.get(
            "BENCHFLOW_SETUP_LOCK_DIR",
            str(Path(tempfile.gettempdir()) / "benchflow-setup-locks"),
        )
    )
    safe = _SETUP_COMMAND_LOCK_SAFE_RE.sub("-", lock_name).strip("-._") or "setup"
    digest = hashlib.sha256(lock_name.encode("utf-8")).hexdigest()[:12]
    return lock_dir / f"{safe[:80]}-{digest}.lock"


@contextlib.asynccontextmanager
async def _environment_setup_host_lock(lock_name: str):
    lock_path = _setup_command_lock_path(lock_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        logger.info("Waiting for environment setup host lock %s", lock_name)
        await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX)
        logger.info("Acquired environment setup host lock %s", lock_name)
        try:
            yield
        finally:
            await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_UN)
            logger.info("Released environment setup host lock %s", lock_name)


def _directory_to_tar_gz_b64(source_dir: Path) -> str:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(source_dir))
    return base64.b64encode(archive.getvalue()).decode("ascii")


def _quote_dotenv_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _upsert_dotenv_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = False
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        prefix = ""
        candidate = stripped
        if candidate.startswith("export "):
            prefix = "export "
            candidate = candidate[len("export ") :].lstrip()
        if candidate and not candidate.startswith("#") and "=" in candidate:
            existing_key = candidate.split("=", 1)[0].strip()
            if existing_key == key:
                out.append(f"{prefix}{key}={_quote_dotenv_value(value)}")
                seen = True
                continue
        out.append(raw)
    if not seen:
        if out and out[-1].strip():
            out.append("")
        out.append("# BenchFlow setup command capture")
        out.append(f"{key}={_quote_dotenv_value(value)}")
    path.write_text("\n".join(out) + "\n")


async def _capture_setup_command_dir(
    env: Any,
    *,
    source_dir: str,
    target_env_var: str,
    target_dotenv_path_env_var: str | None,
    service: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="benchflow-setup-capture-") as tmp:
        local_dir = Path(tmp) / "capture"
        await env.download_dir(source_dir, local_dir, service=service)
        encoded = await asyncio.to_thread(_directory_to_tar_gz_b64, local_dir)
    os.environ[target_env_var] = encoded
    if target_dotenv_path_env_var:
        dotenv_path = os.environ.get(target_dotenv_path_env_var)
        if dotenv_path:
            await asyncio.to_thread(
                _upsert_dotenv_value, Path(dotenv_path), target_env_var, encoded
            )
            logger.info(
                "Persisted setup command capture env var %s to dotenv file from %s",
                target_env_var,
                target_dotenv_path_env_var,
            )
    logger.info(
        "Captured setup command directory %s into host env var %s",
        source_dir,
        target_env_var,
    )


async def _run_one_environment_setup_command(
    env: Any,
    command_config: Any,
    *,
    env_vars: dict[str, str] | None,
    index: int,
) -> None:
    result = await env.exec(
        command_config.command,
        cwd=command_config.cwd,
        env=env_vars,
        timeout_sec=round(command_config.timeout_sec),
        user=command_config.user,
        service=command_config.service,
    )
    if getattr(result, "return_code", 0) != 0:
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        detail = "\n".join(part for part in (stdout, stderr) if part)
        if len(detail) > 4000:
            detail = detail[:4000] + "\n... truncated ..."
        raise RuntimeError(
            f"environment setup command {index} failed with exit code "
            f"{result.return_code}: {detail}"
        )

    capture_dir = getattr(command_config, "capture_dir", None)
    capture_dir_b64_env = getattr(command_config, "capture_dir_b64_env", None)
    if capture_dir and capture_dir_b64_env:
        await _capture_setup_command_dir(
            env,
            source_dir=capture_dir,
            target_env_var=capture_dir_b64_env,
            target_dotenv_path_env_var=getattr(
                command_config, "capture_dir_b64_env_file_var", None
            ),
            service=command_config.service,
        )


async def _run_environment_setup_commands(env: Any, task: Any) -> None:
    """Run task-authored setup commands after sandbox start, before agent install."""

    env_config = getattr(getattr(task, "config", None), "sandbox", None)
    commands = list(getattr(env_config, "setup_commands", []) or [])
    if not commands:
        return

    from benchflow.task.env import resolve_env_vars

    for index, command_config in enumerate(commands, start=1):
        logger.info(
            "Running environment setup command %d/%d on service %s",
            index,
            len(commands),
            command_config.service,
        )
        env_vars = resolve_env_vars(command_config.env) if command_config.env else None
        host_lock = getattr(command_config, "host_lock", None)
        if host_lock:
            async with _environment_setup_host_lock(host_lock):
                await _run_one_environment_setup_command(
                    env,
                    command_config,
                    env_vars=env_vars,
                    index=index,
                )
        else:
            await _run_one_environment_setup_command(
                env,
                command_config,
                env_vars=env_vars,
                index=index,
            )


async def _run_environment_healthcheck(env: Any, task: Any) -> None:
    """Gate rollout startup on the task-authored environment healthcheck."""

    env_config = getattr(getattr(task, "config", None), "sandbox", None)
    healthcheck = getattr(env_config, "healthcheck", None)
    if healthcheck is None:
        return

    loop = asyncio.get_running_loop()
    start_deadline = loop.time() + healthcheck.start_period_sec
    if healthcheck.start_period_sec > 0 and healthcheck.start_interval_sec > 0:
        await asyncio.sleep(
            min(healthcheck.start_interval_sec, healthcheck.start_period_sec)
        )

    failures = 0
    while True:
        result = await env.exec(
            healthcheck.command,
            user="root",
            timeout_sec=max(1, math.ceil(healthcheck.timeout_sec)),
        )
        if getattr(result, "return_code", 0) == 0:
            return

        now = loop.time()
        if now >= start_deadline:
            failures += 1
            if failures >= healthcheck.retries:
                stdout = (getattr(result, "stdout", "") or "").strip()
                stderr = (getattr(result, "stderr", "") or "").strip()
                detail = "\n".join(part for part in (stdout, stderr) if part)
                if len(detail) > 4000:
                    detail = detail[:4000] + "\n... truncated ..."
                raise RuntimeError(
                    "environment healthcheck failed after "
                    f"{healthcheck.retries} attempt(s): {detail}"
                )
            delay = healthcheck.interval_sec
        else:
            delay = min(healthcheck.start_interval_sec, start_deadline - now)
        if delay > 0:
            await asyncio.sleep(delay)


def _gateway_live_tokens(runtime: Any) -> int | None:
    """Cumulative provider tokens from the LiteLLM gateway's live capture.

    Piggybacks on the llm_trajectory.jsonl mirror the proxy runtime already
    runs for every rollout (``LiteLLMProcess.start_live_capture``, wired in
    ``connect()`` and cancelled by the proxy's ``stop()`` in cleanup) — this
    read generates no additional sandbox exec traffic, at any concurrency.

    ``runtime`` stays ``Any`` because the rollout kernel must not import the
    concrete provider plane (``benchflow.providers.runtime``; the #515
    architecture test forbids even a TYPE_CHECKING import, and the planes
    contract is deliberately Any-typed). Rename-safety for the
    ``server.live_usage_tokens`` accessor lives where typing IS allowed: the
    name is agreed in ``contracts.planes.LiveUsageGateway``, LiteLLMProcess
    statically asserts conformance in the providers plane
    (``_live_usage_gateway_conformance`` — a rename on either side fails
    ty), and ``test_gateway_live_tokens_reach_rollout_activity_snapshot``
    pins THIS call site to the real classes end-to-end (a drift here fails
    pytest). The attribute access is direct (no getattr default) so the
    failure mode under a fake without the accessor is the caught exception
    below, not a permanent silent None.
    Best-effort by the dashboard contract: any absence (no runtime, no
    server, the accessor raising, a non-int value) degrades to None —
    #963's ACP-only behavior — never to a render error.
    """
    if runtime is None:
        return None
    try:
        server = runtime.server
        if server is None:
            return None
        tokens = server.live_usage_tokens()
    except Exception:
        return None
    return tokens if isinstance(tokens, int) else None


def _classify_integration_failure(rollout: Any) -> bool:
    """Withhold the reward of a trial whose agent integration broke.

    See :mod:`benchflow.integration_health`: no tool call, message, genuine
    thought or output token after the prompt makes the trial an
    ``agent_integration`` execution failure with a named cause (auth, install,
    empty or truncated trajectory, immediate exit, no activity), unscored.
    Session-factory agents have no ACP event stream and are not judged;
    neither is a run that captured no events at all. A module function, not
    a method, so rollout doubles that borrow ``_maybe_classify_api_error``
    keep working.
    """
    from benchflow.integration_health import diagnose, read_agent_logs

    agent = rollout._config.agent
    try:
        cfg = resolve_agent(agent)
        if cfg.protocol == "session-factory" and cfg.session_factory:
            return False
    except Exception:
        pass
    try:
        usage = getattr(rollout, "_usage_metrics", None) or {}
        output_tokens = usage.get("n_output_tokens")
        rollout_dir = getattr(rollout, "_rollout_dir", None)
        failure = diagnose(
            getattr(rollout, "_trajectory", None) or [],
            agent=agent,
            n_tool_calls=getattr(rollout, "_n_tool_calls", 0),
            output_tokens=output_tokens
            if isinstance(output_tokens, int) and not isinstance(output_tokens, bool)
            else None,
            logs=read_agent_logs(rollout_dir) if rollout_dir is not None else {},
            agent_seconds=(getattr(rollout, "_timing", None) or {}).get(
                "agent_execution"
            ),
            prompt_sent=bool(getattr(rollout, "_executed_prompts", None)),
        )
    except Exception:  # the detector must never fail a finished rollout
        logger.debug("integration-health check failed", exc_info=True)
        return False
    if failure is None:
        return False
    rollout._bare_timeout = False
    rollout._diagnostics.set(failure.diagnostic(reward_withheld=rollout._rewards))
    rollout._error = failure.error_text()
    logger.warning(
        "Agent integration failure [%s]: %s — trial unscored",
        failure.cause,
        failure.evidence,
    )
    rollout._rewards = None
    return True


class Rollout:
    """Decomposed trial lifecycle with independently-callable phases."""

    def __init__(self, config: RolloutConfig) -> None:
        self._planes = config.planes or default_rollout_planes()
        # Activate Docker DinD compatibility shim on first rollout
        # construction (idempotent). Keeps `import benchflow.rollout`
        # side-effect free with respect to sandbox/provider behavior.
        _install_docker_compat(self._planes)

        self._config = config
        self._review_plan: PreparedReview | None = None
        self._scoring: ScoringResult | None = None
        self._completed_result: RolloutResult | None = None
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
        # When True, Rollout treats _env as caller-owned: setup() skips
        # creating a new sandbox and cleanup() skips stopping it. Set via
        # use_prebuilt_env() — see Runtime.execute() for the public path.
        self._env_externally_owned: bool = False
        self._environment: Environment | None = None
        self._timeout: int = 0
        self._timing: dict[str, float] = {}
        self._effective_locked: list[str] = []
        self._disallow_web_tools: bool = False
        self._egress_denylist: EgressDenylist | None = None
        self._active_egress_policy: EgressDenylist | None = None
        self._disallow_hosted_search: bool = False
        self._effective_skills_dir: Path | None = None
        self._effective_skills_sandbox_dir: str | None = None
        self._required_skill_names: tuple[str, ...] = ()
        # Task dir actually deployed: a temp copy (self._task_tmp) when
        # Dockerfile mutations are needed, otherwise config.task_path.
        # cleanup() removes the temp copy.
        self._effective_task_path: Path = config.task_path
        self._task_tmp: Path | None = None
        self._task_skill_policy: TaskSkillPolicy | None = None
        self._usage_runtime: Any = None
        self._usage_metrics: dict[str, Any] = self._planes.extract_usage(None)
        self._native_usage_metrics: dict[str, Any] = _zero_native_acp_usage_metrics()
        self._native_usage_checkpoint: dict[str, int | None] | None = None
        # Provider failure snapshotted during cleanup, after the usage proxy
        # imports its captures (Daytona's SandboxUsageProxy only fills trajectory
        # on stop()). Read by _provider_failure() so ACP-error classification can
        # expose a provider auth/rate-limit/outage failure (status code only)
        # instead of a generic ACP internal error (#546/#564).
        self._provider_failure_cached: ProviderFailure | None = None
        # Auth-only (401/403) view kept for callers added by PR #564.
        self._provider_auth_status_cached: int | None = None
        # Provider API failure summary (all statuses >= 400), snapshotted in
        # cleanup() alongside the auth status — consumed by the post-rollout
        # silent-API-failure classifier in _build_result().
        self._api_failure_summary_cached: dict[str, Any] | None = None

        # Populated by start()
        self._sandbox_id: str | None = None

        # Populated by install_agent()
        self._agent_cfg: Any = None
        self._agent_cwd: str = "/app"

        # Populated by connect()
        self._acp_client: Any = None
        self._session: Any = None
        self._is_session_factory: bool = False
        # ``_session_adapter`` carries the Agent-plane :class:`Session` contract
        # over the live ACP client (architecture.md, "The four contracts").
        # The kernel registers ``on_ask_user`` handlers through it so the live
        # ``session/request_permission`` path runs the handler instead of the
        # auto-approve fallback (#382 follow-up — instantiating the adapter
        # here is what makes the wire path honour the contract).
        self._session_adapter: Any = None
        # Sticky ``on_ask_user`` handler. Stored on the rollout (not the
        # adapter) because connect()/_reconnect_for_role() rebuild the adapter
        # each time we attach a new agent process; the handler the caller
        # registered before the first connect must follow across reconnects.
        # The "ever set" flag distinguishes "never registered" (default
        # auto-approve policy) from "explicitly cleared" (caller passed
        # ``None`` to roll back to auto-approve).
        self._ask_user_handler: Any = None
        self._ask_user_handler_set: bool = False
        self._agent_name: str = ""
        self._active_role: Role | None = None
        # Cursors into the live ACP session's cumulative trajectory and
        # tool-call totals so execute() and the partial-capture path extend
        # only the delta since the last read. Reset per session in
        # install_agent().
        self._session_traj_count: int = 0
        self._session_tool_count: int = 0

        # Populated by execute()
        self._trajectory: list[dict] = []
        self._n_tool_calls: int = 0
        self._trajectory_source: TrajectorySource | None = None
        self._partial_trajectory: bool = False
        # Set when a clean wall-clock prompt timeout (AgentPromptTimeoutError
        # with no pending tool calls) fired — its captured trajectory is a
        # complete terminal one, not a rerunnable partial (#640).
        self._terminal_timeout: bool = False
        # Detail-less agent-phase timeout, eligible for zero-activity
        # reclassification (#1071). One-shot: cleared once a verdict lands.
        self._bare_timeout: bool = False
        # Set by mark_solver_complete(); later failures keep the solver result.
        self._solver_execution_complete: bool = False
        # The running agent phase, so the host deadline can stop the agent
        # alone and still verify (#1134); the reason is set only by that stop.
        self._agent_phase: asyncio.Task | None = None
        self._agent_deadline_reason: str | None = None
        # Every prompt actually sent to the agent across all execute() calls —
        # this is what `n_prompts` and `prompts.json` should reflect for Scene
        # rollouts where each turn issues its own prompt. The original
        # `_resolved_prompts` is only the static base task prompt set.
        self._executed_prompts: list[str] = []
        # The user-loop engine's per-round log (aliased by _run_user_loop).
        # Rounds are appended as soon as their soft verify completes, so
        # _loop_strategy_metadata() can summarize whatever rounds finished
        # even when a later round timed out or crashed.
        self._user_rounds_log: list[dict[str, Any]] = []

        # The tree-native execution model (architecture.md, "tree-native").
        # A linear rollout is a degree-1 tree; execute() grows it one Step at a
        # time, and branch() forks a node into N children. The tree is additive
        # — it never alters linear behaviour or output.
        self._tree: RolloutTree = RolloutTree()
        self._cursor: RolloutNode = self._tree.root
        # Branch custody (rollout_branch.py) and verifier recovery state.
        self._branch_child_active = False
        self._branch_cleanup_unquiesced = False
        self._branch_world_unsafe = False
        self._branch_parent_discarded = False
        # Set on an isolated branch child's sub-rollout: the root trial whose
        # tree.json and branches/ record this rollout's own forks.
        self._lineage_dir: Path | None = None
        self._lineage_forks: list[dict[str, Any]] | None = None
        # Set by branch(resume_session=True) for a child's connect(): open
        # the parent's ACP session with session/load instead of a new one.
        self._resume_session_id: str | None = None
        # Set on an isolated branch child: the parent's installed agent
        # config, whose binary the snapshot already holds (install skipped).
        self._installed_agent_cfg: Any = None
        # Set when this rollout's sandbox is created from a branch snapshot
        # (an isolated child, or --from-checkpoint): start() and
        # install_agent() keep what the snapshot already holds, and
        # _snapshot_start records what was reused.
        self._from_branch_snapshot = False
        self._snapshot_start: dict[str, str] | None = None
        self._docker_recovery_baseline: Any | None = None
        self._recovery_lease_owned = False
        self._recovery_lease_released = False

        # Populated by verify()
        self._rewards: dict | None = None
        # Counts verify() calls; the branch engine reads it to tell whether a
        # custom child runner's number came from the verifier.
        self._verify_calls = 0
        # Canonical plan-review status/provenance, populated after verify().
        self._verifier_error: str | None = None
        self._error: str | None = None
        # Populated by _export_generated_skills() on failure (#389 follow-up).
        # Kept separate from self._error so classify_error() does not mis-tag
        # an export-time infra failure ("connection lost") as the agent's own
        # infra_failure category in dashboards.
        self._export_error: str | None = None
        # A separate verifier's clean control: the paths capture reads,
        # measured before the agent runs, and whether workspace capture then
        # hit its limits (see rollout._separate_verifier).
        self._pristine_outputs: dict[str, Any] | None = None
        self._capture_over_limit = False
        # Single bag for the four parallel diagnostic fields the old code
        # carried as separate attrs (issue #503). Each callsite that used
        # to assign to one of those slots now calls ``self._diagnostics.set(...)``
        # with a typed Diagnostic value.
        self._diagnostics: RolloutDiagnostics = RolloutDiagnostics()

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

    def use_prebuilt_env(self, inner: Any) -> None:
        """Inject a caller-owned sandbox; skip creation and teardown.

        When the public Runtime API receives a live ``Environment`` (one
        the caller already constructed, possibly started, and may want to
        reuse), the Rollout must evaluate inside that same sandbox rather
        than spinning up a second one. Call this before ``setup()``/
        ``run()``: ``setup()`` will then skip ``_create_environment`` and
        ``cleanup()`` will skip stopping the sandbox — the caller owns the
        lifecycle. Fixes #388.
        """
        if inner is None:
            raise ValueError("use_prebuilt_env() requires a non-None sandbox")
        self._env = inner
        self._env_externally_owned = True

    @property
    def env(self) -> Any:
        return self._env

    @property
    def acp_client(self) -> Any:
        return self._acp_client

    def activity_snapshot(self) -> ActivitySnapshot:
        """The eval dashboard's per-task :class:`ActivitySnapshot`: the
        current lifecycle phase plus the live :class:`SessionCounters` (tool
        calls, last tool title, total tokens, distinct tool titles).
        ``counters`` is None before
        the agent session exists — the phase then carries the cell
        ("creating sandbox…", "verifying…"), so a 90s sandbox create is not
        indistinguishable from a hang.

        ``total_tokens`` reconciles two cumulative live signals as
        ``max(acp_snapshot, gateway_live)``: the ACP session's self-reported
        usage (updates only when a *prompt* completes — a single-prompt
        rollout stays None for the whole agent phase) and the LiteLLM
        gateway's live callback capture (updates per completed LLM *request*,
        mid-prompt). max() because both inputs are non-decreasing counters, so
        the cell/footer never step down mid-run whichever signal leads —
        gateway-live can exceed the ACP self-report (requests the agent
        discarded, cache accounting) and vice versa (capture lag behind large
        log records) — and a dead gateway signal (None) degrades to exactly
        the ACP-only behavior. The winner is display-only: at completion the
        trusted scoring total replaces it (see the footer contract in
        cli/_live_progress.py for the sanctioned non-monotonic boundary).

        Rollout owns the client/session dig so a rename breaks here — in
        typed, tested code — instead of silently blanking the dashboard cell
        (see benchflow._utils.live_activity). Session-factory agents have no
        ACP client and always report counter-less snapshots (their
        gateway-live tokens are not surfaced — the counters seam is the only
        token carrier).
        """
        session = self._acp_client.session if self._acp_client else None
        if session is None:
            return ActivitySnapshot(self._phase, None)
        calls, last_title = session.progress_snapshot()
        usage = session.latest_usage_totals()
        acp_tokens = usage.get("total_tokens") if usage else None
        gateway_tokens = _gateway_live_tokens(self._usage_runtime)
        if acp_tokens is None and gateway_tokens is None:
            tokens = None
        else:
            tokens = max(acp_tokens or 0, gateway_tokens or 0)
        return ActivitySnapshot(
            self._phase,
            SessionCounters(calls, last_title, tokens, session.distinct_tool_titles),
        )

    @property
    def trajectory(self) -> list[dict]:
        return self._trajectory

    def record_external_tool_call(
        self,
        *,
        tool_name: str,
        event: dict,
    ) -> None:
        """Record a tool call driven outside the ACP prompt loop.

        Training integrations can own model generation while still preserving
        BenchFlow's verifier and rollout artifact contract. Normal ACP rollouts
        should continue to use ``execute``.
        """

        reserved = {"type", "tool_name"} & set(event)
        if reserved:
            reserved_text = ", ".join(sorted(reserved))
            raise ValueError(
                "record_external_tool_call event cannot contain reserved fields: "
                f"{reserved_text}"
            )

        self._n_tool_calls += 1
        self._trajectory.append(
            {
                "type": "tool_call",
                "tool_name": tool_name,
                **event,
            }
        )

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
        if self._completed_result is not None:
            return self._completed_result
        if self._phase not in ("verified", "cleaned"):
            return None
        if self._review_plan is not None and self._scoring is None:
            return None
        return prepare_terminal_result(self)

    def _require_rollout_dir(self) -> Path:
        if self._rollout_dir is None:
            raise RuntimeError("Rollout.setup() must run before this phase")
        return self._rollout_dir

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise RuntimeError("Rollout.setup() must run before building a result")
        return self._started_at

    # Phase 1: SETUP (host-side, no container yet)

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        require_safe_branch_world(self)
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — oracle files stay visible "
                "to the agent for the entire trial."
            )

        self._effective_locked = self._planes.resolve_locked_paths(
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

        # C-axis overlay: deep-merge cfg.config_override into the task's resolved
        # config here at the rollout layer (not in the Task constructor), so only
        # this run's tasks are patched and every downstream read sees it. No-op
        # when None.
        if cfg.config_override:
            from benchflow._utils.config_override import apply_config_override

            self._task.config = apply_config_override(
                self._task.config, cfg.config_override
            )

        prepare_terminal_review(self)
        if cfg.task_digest is None:
            from benchflow._utils.task_authoring import task_digest

            cfg.task_digest = task_digest(cfg.task_path)

        self._disallow_web_tools = (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        ) and not is_scripted_agent(cfg.primary_agent)
        self._egress_denylist = (
            None
            if self._disallow_web_tools or is_scripted_agent(cfg.primary_agent)
            else _task_egress_denylist(self._task)
        )
        if self._egress_denylist is not None and not cfg.sandbox_user:
            raise ValueError(
                f"network_mode={self._egress_denylist.mode!r} requires a sandbox_user"
            )
        self._disallow_hosted_search = self._egress_denylist is not None
        self._agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                cfg.primary_agent, cfg.primary_model, cfg.agent_env
            ),
            disallow=self._disallow_web_tools,
        )
        env_config = getattr(getattr(self._task, "config", None), "sandbox", None)
        task_skill_policy = resolve_task_skill_policy(
            task_path=cfg.task_path,
            skill_mode=cfg.recorded_skill_mode,
            runtime_skills_dir=cfg.skills_dir,
            declared_sandbox_skills_dir=getattr(env_config, "skills_dir", None),
        )
        self._task_skill_policy = task_skill_policy
        self._resolved_prompts = _apply_prompt_prefix(
            _resolve_prompts(cfg.task_path, cfg.prompts),
            self._task.config.agent.prompt_prefix,
        )
        self._agent_launch = self._planes.agent_launch(
            cfg.primary_agent,
            disallow_web_tools=self._disallow_web_tools,
            disallow_hosted_search=self._disallow_hosted_search,
        )

        # Copy task dir to temp when Dockerfile mutations are needed
        # (_inject_skills writes into environment/_deps/, stage_dockerfile
        # rewrites COPY paths — neither should modify the source tree)
        effective_task_path = cfg.task_path
        if (
            cfg.context_root
            or cfg.base_image_override
            or task_skill_policy.needs_task_copy
        ):
            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            shutil.copytree(cfg.task_path, tmp / cfg.task_path.name, dirs_exist_ok=True)
            effective_task_path = tmp / cfg.task_path.name
            self._task_tmp = tmp
            if task_skill_policy.strip_bundled_dir_from_copy:
                strip_task_bundled_skills(effective_task_path)

        if cfg.base_image_override:
            self._planes.override_dockerfile_base_image(
                effective_task_path, cfg.base_image_override
            )
        if cfg.context_root:
            self._planes.stage_dockerfile_deps(
                effective_task_path, Path(cfg.context_root)
            )
        effective_skills_dir = task_skill_policy.host_dir
        if (
            effective_skills_dir is not None
            and task_skill_policy.host_dir_is_bundled
            and effective_task_path != cfg.task_path
        ):
            effective_skills_dir = task_bundled_skills_dir(effective_task_path)
        if effective_skills_dir is not None and not _environment_uses_prebuilt_image(
            env_config, cfg.environment_manifest
        ):
            self._planes.inject_skills_into_dockerfile(
                effective_task_path,
                effective_skills_dir,
                sandbox_dir=task_skill_policy.sandbox_dir or "/skills",
            )

        task_skill_policy = replace(task_skill_policy, host_dir=effective_skills_dir)
        self._task_skill_policy = task_skill_policy
        self._effective_task_path = effective_task_path
        self._effective_skills_dir = effective_skills_dir
        self._effective_skills_sandbox_dir = task_skill_policy.sandbox_dir
        self._required_skill_names = (
            tuple(
                sorted(
                    path.parent.name for path in effective_skills_dir.glob("*/SKILL.md")
                )
            )
            if effective_skills_dir is not None
            else ()
        )

        # Honour an externally-supplied sandbox (use_prebuilt_env, set by
        # Runtime.execute() when the caller passes a live Environment).
        # Without this guard, every Runtime.execute() would build a second
        # sandbox and silently discard the caller's prepared one — #388.
        if self._env is None:
            self._env = self._planes.create_environment(
                cfg.environment,
                self._task,
                effective_task_path,
                self._rollout_name,
                self._rollout_paths,
                preserve_agent_network=self._disallow_web_tools,
                environment_manifest=cfg.environment_manifest,
            )
        # Caller-supplied wall-clock budget (e.g. RuntimeConfig.timeout)
        # wins over the task's own default. Without this override there is
        # no way to tighten/loosen the agent budget per run — see #378.
        if cfg.timeout is not None:
            self._timeout = int(cfg.timeout)
        else:
            self._timeout = int(self._task.config.agent.timeout_sec or 0)
        # Look on the type, not via instance getattr: permissive proxies and
        # AsyncMock-based test sandboxes manufacture arbitrary attributes,
        # which would turn this optional synchronous hook into an un-awaited
        # fake coroutine.
        configure_timeout = getattr(type(self._env), "configure_agent_timeout", None)
        if callable(configure_timeout):
            configure_timeout(self._env, self._timeout)

        from benchflow.rollout._verifier_recovery import recovery_ineligible_reason

        recovery_reason = recovery_ineligible_reason(self)
        _write_config(
            self._rollout_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            reasoning_effort=cfg.primary_reasoning_effort,
            codex_apps_policy=cfg.codex_apps_policy,
            environment=cfg.environment,
            environment_manifest=cfg.environment_manifest,
            skill_policy=task_skill_policy,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            skip_agent_install=cfg.skip_agent_install,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
            base_image_override=cfg.base_image_override,
            usage_tracking=cfg.usage_tracking.with_env_defaults(),
            concurrency=cfg.concurrency,
            agent_idle_timeout=cfg.agent_idle_timeout,
            scenes=cfg.effective_scenes,
            source_provenance=cfg.source_provenance,
            dataset=cfg.dataset,
            task_digest=cfg.task_digest,
            config_override=cfg.config_override,
            loop_strategy=cfg.loop_strategy_spec,
            review=self._review_plan.metadata() if self._review_plan else None,
            verifier_recovery={
                "eligible": recovery_reason is None,
                "reason": recovery_reason,
            },
            purpose=cfg.purpose,
            parent_rollout=cfg.parent_rollout,
            freeze_workspace=cfg.freeze_workspace,
        )

        self._phase = "setup"

    # Phase 2: START (container comes up)

    async def start(self) -> None:
        """Start the environment and upload task files."""

        async def _capture_and_persist_sandbox() -> None:
            # Persist the sandbox id the moment the sandbox exists, before any
            # upload that could fail or be interrupted (#554/#563). Otherwise a
            # mid-upload failure leaves a live Daytona sandbox with no
            # sandbox.json to audit or clean up.
            sid = getattr(self._env, "sandbox_id", None)
            self._sandbox_id = sid if isinstance(sid, str) else None
            persist_sandbox_info(self._env, self._rollout_dir)
            from benchflow.rollout._verifier_recovery import (
                capture_original_docker_baseline,
            )

            await capture_original_docker_baseline(self)

        await _start_env_and_upload(
            self._env,
            self._config.task_path,
            self._timing,
            skip_start=self._env_externally_owned,
            on_started=_capture_and_persist_sandbox,
            uploads=self._config.uploads,
            # A sandbox created from a branch snapshot already holds them.
            upload_task_files=not self._from_branch_snapshot,
        )

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        await _run_environment_healthcheck(self._env, self._task)

        # Environment plane: provision the manifest-declared stateful
        # environment and gate on its readiness before the agent runs.
        if self._config.environment_manifest is not None:
            self._environment = self._planes.manifest_environment(
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

        if self._from_branch_snapshot:
            # They ran before the snapshot was taken; re-running them would
            # act on the checkpoint's state, which in-place children never do.
            logger.info("Sandbox from a branch snapshot: setup commands not re-run")
        else:
            await _run_environment_setup_commands(self._env, self._task)

        self._phase = "started"

    # Phase 3: INSTALL AGENT

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous scene-authored steps (different agents per role),
        each role's agent is installed on-demand in connect_as().
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()

        self._agent_cwd = await _resolve_agent_cwd(self._env, self._task)

        await prepare_capture_runtime(self)

        present = {"agent": False, "baseline": False}
        if self._from_branch_snapshot:
            reuse = (
                None
                if is_scripted_agent(cfg.primary_agent)
                else self._installed_agent_cfg
            )
            launch = getattr(reuse, "launch_cmd", None) or ""
            present = await self._planes.probe_snapshot_start(
                self._env, agent_binary=launch.split()[0] if launch.split() else None
            )
            self._snapshot_start = {
                "agent": "reused"
                if reuse is not None and present["agent"]
                else "installed"
                if cfg.primary_agent != "oracle"
                else "none",
                "verifier_baseline": "inherited"
                if present["baseline"]
                else "recaptured",
                "setup_commands": "skipped",
            }
            if not present["baseline"]:
                logger.warning(
                    "Branch snapshot has no verifier baseline; re-capturing it from "
                    "the checkpoint state (tampering before the fork would not be "
                    "undone at verification)"
                )

        async def verifier_baseline() -> None:
            # A branch snapshot already holds the pre-agent baseline, captured
            # before the parent's agent ran; re-capturing would record the
            # parent agent's changes as pre-agent.
            if present["baseline"]:
                await self._planes.prepare_log_dirs(
                    self._env, sandbox_user=cfg.sandbox_user
                )
                return
            await self._planes.snapshot_build_config(
                self._env, workspace=self._agent_cwd
            )
            await self._planes.seed_verifier_workspace(
                self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
            )

        if is_scripted_agent(cfg.primary_agent):
            if cfg.sandbox_user:
                await self._planes.setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            await verifier_baseline()
            await self._planes.deploy_skills(
                self._env,
                self._effective_task_path,
                self._effective_skills_dir,
                None,
                cfg.sandbox_user,
                self._agent_cwd,
                skills_sandbox_dir=self._effective_skills_sandbox_dir,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await self._planes.lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        if cfg.skip_agent_install:
            self._agent_cfg = None
        elif self._installed_agent_cfg is not None and present["agent"]:
            # An isolated branch child: its sandbox came from a snapshot taken
            # after the parent installed this same agent (binary checked above;
            # same process, same registry pin, so the same version).
            self._agent_cfg = self._installed_agent_cfg
        else:
            self._agent_cfg = await self._planes.install_agent(
                self._env,
                agent_name,
                rollout_dir,
                sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            )
        if cfg.sandbox_user:
            self._agent_cwd = await self._planes.setup_sandbox_user(
                self._env,
                cfg.sandbox_user,
                workspace=self._agent_cwd,
                timeout_sec=cfg.sandbox_setup_timeout,
            )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await self._planes.write_credential_files(
            self._env,
            agent_name,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            cred_home,
        )
        await _install_native_task_mcp_config(
            self._env,
            self._task,
            agent_cfg=self._agent_cfg,
            cred_home=cred_home,
            owner=cfg.sandbox_user,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await self._planes.upload_subscription_auth(
                self._env, agent_name, cred_home
            )
        await self._planes.apply_web_tool_policy(
            self._env,
            agent_name,
            self._agent_cfg,
            cred_home,
            disallow=self._disallow_web_tools,
            disallow_hosted_search=self._disallow_hosted_search,
        )
        self._agent_env = await enforce_codex_apps_policy(
            self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            policy=effective_apps_policy(
                cfg.codex_apps_policy, purpose=cfg.purpose, skip_verify=cfg.skip_verify
            ),
            requested=cfg.codex_apps_policy,
            rollout_dir=rollout_dir,
        )
        await verifier_baseline()

        await self._planes.deploy_skills(
            self._env,
            self._effective_task_path,
            self._effective_skills_dir,
            self._agent_cfg,
            cfg.sandbox_user,
            self._agent_cwd,
            skills_sandbox_dir=self._effective_skills_sandbox_dir,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await self._planes.lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # Phase 3b: CONNECT (ACP session — re-entrant)

    def _session_factory_entrypoint(self, agent_name: str) -> str | None:
        """Return the ``session_factory`` "module:callable" if *agent_name* is a
        non-ACP session-factory agent, else None — the connect/drive dispatch key.

        A session-factory agent declares ``protocol="session-factory"`` + a
        ``session_factory`` entrypoint (e.g. omnigent's ``omnigent run`` CLI,
        which has no ACP server); everything else connects over ACP. Resolution
        failures degrade to ACP (None) rather than raising."""
        try:
            cfg = resolve_agent(agent_name)
        except Exception:
            return None
        if cfg.protocol == "session-factory" and cfg.session_factory:
            return cfg.session_factory
        return None

    async def _stop_active_egress(self) -> None:
        if self._active_egress_policy is not None:
            await self._planes.stop_egress_denylist(
                self._env,
                self._require_rollout_dir() / "network-transports" / uuid4().hex,
            )
            self._active_egress_policy = None

    async def _start_egress_denylist(
        self, denylist: EgressDenylist, *, agent_launch: str | None = None
    ) -> None:
        """Rebuild transport before each connection, including restored sandboxes."""
        runtime = self._usage_runtime
        receipt = self._require_rollout_dir() / "native-oauth-network.json"
        receipt.unlink(missing_ok=True)
        admission = None
        if denylist.native_claude_model_only:
            admission = await validate_native_oauth_transport(
                self._env, self._config.sandbox_user, agent_launch or self._agent_launch
            )
        await self._stop_active_egress()
        # Track attempted startup too: cleanup must remove partially staged policy.
        self._active_egress_policy = denylist
        await self._planes.start_egress_denylist(
            self._env,
            self._config.sandbox_user,
            denylist,
            model_gateway_url=runtime.agent_base_url if runtime is not None else None,
        )
        if admission is not None:
            receipt.write_text(
                json.dumps({**admission, "admitted": True}, indent=2) + "\n"
            )

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        require_safe_branch_world(self)
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        (rollout_dir / "native-oauth-network.json").unlink(missing_ok=True)
        t0 = datetime.now()
        egress_denylist = native_oauth_egress_policy(
            cfg.primary_agent,
            cfg.primary_model,
            self._agent_env,
            no_web=self._disallow_web_tools,
        ) or allowlist_model_transport(
            self._egress_denylist,
            cfg.primary_agent,
            cfg.primary_model,
            self._agent_env,
        )

        (
            self._agent_env,
            self._usage_runtime,
        ) = await self._planes.ensure_litellm_runtime(
            agent=cfg.primary_agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            runtime=self._usage_runtime,
            environment=cfg.environment,
            session_id=self._rollout_name or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            required_skill_names=self._required_skill_names,
            live_trajectory_path=rollout_dir / "trajectory" / "llm_trajectory.jsonl",
            force_sandbox_local=self._disallow_web_tools or egress_denylist is not None,
        )
        if egress_denylist is not None:
            self._agent_env = denylist_agent_env(self._agent_env, egress_denylist)
        self._agent_env = await enforce_codex_apps_policy(
            self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            policy=effective_apps_policy(
                cfg.codex_apps_policy, purpose=cfg.purpose, skip_verify=cfg.skip_verify
            ),
            requested=cfg.codex_apps_policy,
            rollout_dir=rollout_dir,
        )
        if egress_denylist is None:
            await self._stop_active_egress()
        sf_entrypoint = self._session_factory_entrypoint(cfg.primary_agent)
        self._is_session_factory = sf_entrypoint is not None
        if sf_entrypoint is not None:
            if egress_denylist is not None:
                raise RuntimeError(
                    f"network_mode={egress_denylist.mode!r} requires an ACP agent"
                )
            (
                self._acp_client,
                self._session,
                self._session_adapter,
                self._agent_name,
            ) = await self._planes.connect_session_factory(
                env=self._env,
                agent=cfg.primary_agent,
                session_factory=sf_entrypoint,
                agent_env=self._agent_env,
                sandbox_user=cfg.sandbox_user,
                model=cfg.primary_model,
                rollout_dir=rollout_dir,
                timeout=self._timeout,
                agent_cwd=self._agent_cwd,
            )
        else:
            if egress_denylist is not None:
                await self._start_egress_denylist(egress_denylist)
            (
                self._acp_client,
                self._session,
                self._session_adapter,
                self._agent_name,
            ) = await self._planes.connect_acp(
                env=self._env,
                agent=cfg.primary_agent,
                agent_launch=self._agent_launch,
                agent_env=self._agent_env,
                sandbox_user=cfg.sandbox_user,
                model=cfg.primary_model,
                rollout_dir=rollout_dir,
                environment=cfg.environment,
                agent_cwd=self._agent_cwd,
                reasoning_effort=cfg.primary_reasoning_effort,
                mcp_servers=_task_mcp_specs_for_agent(
                    cfg.primary_agent, self._task, self._agent_cfg
                ),
                resume_session_id=self._resume_session_id,
            )
            replayed = getattr(self._session, "replayed_prefix", None)
            if isinstance(replayed, tuple):
                # session/load replayed the parent's conversation: that is the
                # branch's shared prefix, not this child's continuation.
                self._session_traj_count, self._session_tool_count = replayed
        self._native_usage_checkpoint = None
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    def _attach_trajectory_writer(self, rollout_dir: Path) -> None:
        """Wire the current session's ``on_change`` to stream cumulative
        trajectory to ``rollout_dir/trajectory/acp_trajectory.jsonl``.

        The sink prepends ``self._trajectory`` (events from prior scenes,
        captured by value at wire-up time) so multi-scene rollouts don't
        overwrite earlier scenes' events with the current session's
        snapshot.
        """
        if self._session is None or rollout_dir is None:
            return
        traj_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        self._session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), self._trajectory
        )

    async def disconnect(self) -> None:
        """Close the ACP client and clean up agent process, keeping the environment alive."""
        if self._is_session_factory:
            self._capture_partial_session_factory_trajectory()
        else:
            self._capture_partial_acp_trajectory()
        self._collect_native_acp_usage()
        if self._acp_client:
            try:
                await self._acp_client.close()
            except Exception as e:
                logger.warning(f"ACP client close failed: {e}")
            self._acp_client = None
        self._session = None
        self._session_adapter = None
        self._is_session_factory = False
        # Kill any lingering agent processes to prevent context bleed between scenes
        agent_pattern = _agent_process_kill_pattern(self._agent_launch)
        if self._env and agent_pattern:
            with contextlib.suppress(Exception):
                await self._env.exec(
                    f"pkill -f {shlex.quote(agent_pattern)} || true",
                    timeout_sec=10,
                )
        self._active_role = None
        self._session_tool_count = 0
        self._session_traj_count = 0
        # Rewinding the phase to "installed" is right for the between-scenes
        # disconnect (another agent turn follows, and connect_as() will mark
        # "connected"), but disconnect() is ALSO called from cleanup(), after
        # verify() has already moved the rollout into its terminal phases. Left
        # unguarded, that rewind made the live dashboard walk backwards —
        # "verifying…" and then "running agent…" again for the whole teardown
        # stretch — and briefly blanked ``Rollout.result``, which is gated on
        # the same terminal phases.
        if self._phase not in _TERMINAL_PHASES:
            self._phase = "installed"

    def on_ask_user(self, handler: Any) -> None:
        """Register the agent-initiated ``session/request_permission`` handler.

        Forwards to :meth:`ACPSessionAdapter.on_ask_user` on the live adapter
        so the handler runs on the wire path; before #382's follow-up the
        adapter was never instantiated in production and the auto-approve
        policy ran unconditionally. The handler is sticky — stored on the
        rollout so reconnects (e.g. ``_reconnect_for_role``) re-register it
        on the freshly bound adapter via :meth:`_reapply_ask_user_handler`.

        Pass ``None`` to clear; the client's most-permissive auto-approve
        policy takes over (preserves the benchmark-mode default).
        """
        self._ask_user_handler = handler
        self._ask_user_handler_set = True
        self._reapply_ask_user_handler()

    def _reapply_ask_user_handler(self) -> None:
        """Re-bind any registered ``on_ask_user`` handler to the live surface.

        For an ACP agent that surface is the ``ACPSessionAdapter``; for a
        session-factory agent (``adapter is None``) it is the live ``Session``
        itself, which implements ``on_ask_user`` directly (protocol.py). The
        earlier ``adapter is None -> return`` guard silently dropped the handler
        for every session-factory agent — they return ``adapter=None`` from
        connect — so the agent-initiated branch hook never reached them (#825).
        """
        # No-op when the caller never touched on_ask_user — leaves the
        # default auto-approve path alone and avoids redundant client calls
        # from the connect()/_reconnect_for_role() hot paths.
        if not self._ask_user_handler_set:
            return
        adapter = self._session_adapter
        handler = self._ask_user_handler
        if adapter is None:
            # Session-factory path: no adapter, bind onto the live Session.
            if self._is_session_factory:
                session = self._session
                if session is not None and handler is not None:
                    session.on_ask_user(handler)
            return
        if handler is None:
            # Explicit clear — drop the bridge closure on the client so
            # the default most-permissive policy takes over.
            client = self._acp_client
            if client is not None:
                client.on_ask_user(None)
            return
        adapter.on_ask_user(handler)

    def _install_document_confirmation_handler(self, user: BaseUser) -> bool:
        """Install a fail-closed permission handler for document human policy.

        ``confirmation_policy: human`` means BenchFlow must not silently fall
        back to ACP's benchmark-mode auto-approve path. If a caller already
        registered an explicit ``on_ask_user`` handler we treat that as the
        human/policy hook and leave it alone; otherwise the non-interactive
        default denies/rejects permission requests when a deny option exists.
        """

        if _user_confirmation_policy(user) != "human":
            return False
        if self._ask_user_handler is not None:
            return False

        async def _deny_without_human(request: AskUserRequest) -> str:
            option = _least_permissive_option_id(
                request.options,
                request.option_kinds,
            )
            logger.info(
                "Document confirmation_policy=human denied ask_user request "
                "%s with option %s",
                request.request_id,
                option,
            )
            return option

        self.on_ask_user(_deny_without_human)
        return True

    def _capture_partial_acp_trajectory(self) -> None:
        """Append the live session's uncaptured tail to ``self._trajectory``.

        Runs on the disconnect / cleanup path when ``execute_prompts`` may
        have raised before the normal extend in :meth:`execute`. Uses
        ``_session_traj_count`` as the pointer to events already extended
        from this session so a partial scene's events are preserved on top
        of any prior scenes' (already-captured) events — see PR #566 review.
        """
        # Defensive lookup tolerates bare ``object()`` stubs used by older
        # rollout tests that pre-date the live-session partial-capture path.
        session = (
            getattr(self._acp_client, "session", None) if self._acp_client else None
        )
        if session is None:
            return
        try:
            captured = _capture_session_trajectory(session)
        except Exception as e:
            logger.warning(f"Partial trajectory capture failed: {e}")
            return
        delta = captured[self._session_traj_count :]
        if not delta:
            return
        self._trajectory.extend(delta)
        self._session_traj_count = len(captured)
        if self._terminal_timeout:
            # Clean wall-clock terminal timeout (#640): the captured tail is the
            # complete trajectory, so leave _partial_trajectory False.
            self._trajectory_source = "acp"
        else:
            self._partial_trajectory = True
            self._trajectory_source = "partial_acp"
        new_tools = len(session.tool_calls) - self._session_tool_count
        if new_tools > 0:
            self._n_tool_calls += new_tools
        self._session_tool_count = len(session.tool_calls)

    def _capture_partial_session_factory_trajectory(self) -> None:
        """Session-factory analogue of :meth:`_capture_partial_acp_trajectory`.

        A session-factory agent has no ACP client; its live trajectory lives
        directly on ``self._session.steps`` (the protocol-conformant Session).
        On the disconnect/cleanup path — where :meth:`execute` may have raised
        before its normal extend — append the session's uncaptured tail to
        ``self._trajectory``. ``_session_traj_count`` is the pointer to events
        already extended from this session, so a partial scene's steps land on
        top of prior scenes' (already-captured) events. Mirrors the
        terminal-vs-partial source labelling of the ACP path (#825).
        """
        session = self._session
        if session is None:
            return
        try:
            captured = list(session.steps)
        except Exception as e:
            logger.warning(f"Partial session-factory trajectory capture failed: {e}")
            return
        delta = captured[self._session_traj_count :]
        if not delta:
            return
        self._trajectory.extend(delta)
        self._session_traj_count = len(captured)
        if self._terminal_timeout:
            # Clean wall-clock terminal timeout: the captured tail is complete.
            self._trajectory_source = "acp"
        else:
            self._partial_trajectory = True
            self._trajectory_source = "partial_acp"

    # Phase 3c: EXECUTE

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
        continuation Step lands on the child node itself. Inside a branch
        child, a call without ``node`` fills the pending child node the same
        way, so a custom runner cannot hang the child's Steps under a
        grandchild by omitting it.
        """
        require_safe_branch_world(self)
        if (
            node is None
            and self._branch_child_active
            and self._cursor.parent is not None
            and self._cursor.step_in is None
        ):
            node = self._cursor
        effective_prompts = prompts or self._resolved_prompts
        # Protocol-agnostic "connected?" guard: ACP connect sets _acp_client;
        # a session-factory connect sets _session (no ACP client). Connected iff
        # at least one is present; both None means connect() never ran.
        if self._acp_client is None and self._session is None:
            raise RuntimeError("Rollout.connect() must run before execute()")
        prev_session_tools = self._session_tool_count
        t0 = datetime.now()
        active_role = self._active_role
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

        try:
            if self._is_session_factory:
                (
                    trajectory,
                    n_tool_calls,
                ) = await self._planes.execute_prompts_session_factory(
                    self._session,
                    effective_prompts,
                    timeout,
                    idle_timeout=idle_timeout,
                )
            else:
                trajectory, n_tool_calls = await self._planes.execute_prompts(
                    self._acp_client,
                    self._session,
                    effective_prompts,
                    timeout,
                    idle_timeout=idle_timeout,
                )
        except AgentPromptTimeoutError as e:
            self._diagnostics.set(e.diagnostic)
            self._commit_acp_execution(
                trajectory=e.trajectory,
                n_tool_calls=e.n_tool_calls,
                prev_session_tools=prev_session_tools,
                effective_prompts=e.executed_prompts or effective_prompts,
                started_at=t0,
                node=node,
                partial_trajectory=not e.terminal_trajectory_complete,
            )
            raise

        self._commit_acp_execution(
            trajectory=trajectory,
            n_tool_calls=n_tool_calls,
            prev_session_tools=prev_session_tools,
            effective_prompts=effective_prompts,
            started_at=t0,
            node=node,
        )
        return trajectory, n_tool_calls

    def _commit_acp_execution(
        self,
        *,
        trajectory: list[dict],
        n_tool_calls: int,
        prev_session_tools: int,
        effective_prompts: list[str],
        started_at: datetime,
        node: RolloutNode | None,
        partial_trajectory: bool = False,
    ) -> None:
        """Commit a finalized ACP snapshot into rollout state."""

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[self._session_traj_count :]
        self._session_tool_count = n_tool_calls
        self._session_traj_count = len(trajectory)

        self._trajectory.extend(new_events)
        self._n_tool_calls += new_tools
        self._executed_prompts.extend(effective_prompts)
        if partial_trajectory:
            self._partial_trajectory = True
            self._trajectory_source = "partial_acp"
        elif not self._partial_trajectory:
            self._trajectory_source = "acp"
        self._collect_native_acp_usage()

        # Grow the tree at Step-level granularity — one Step per ACP event
        # (tool_call, agent_message, agent_thought, user_message). A single
        # execute() call walks the cursor down N nodes when it produced N
        # events. Closes #414: branch/process-reward/value targets the
        # individual action, not a collapsed turn.
        #
        # Empty-event executes still emit one Step so the tree advances at
        # least once per execute() call — the cursor must move, and a branch
        # child's pending node must get populated (see rollout_branch).
        steps = self._build_step_batch(new_events, new_tools)
        first_step, *rest_steps = steps
        if node is not None:
            # Fill a pre-attached pending node (a branch child) in place — the
            # child's real continuation Step lands on the child node itself.
            self._cursor = self._tree.populate(node, first_step)
        else:
            self._cursor = self._tree.advance(self._cursor, first_step)
        for step in rest_steps:
            self._cursor = self._tree.advance(self._cursor, step)

        # Accumulate execution time across all execute() calls — Scene rollouts
        # invoke execute() once per turn, and the previous "set only on first
        # call" behaviour undercounted multi-turn agent time.
        elapsed = (datetime.now() - started_at).total_seconds()
        self._timing["agent_execution"] = (
            self._timing.get("agent_execution", 0.0) + elapsed
        )

        self._phase = "executed"

    def _collect_native_acp_usage(self) -> None:
        """Accumulate ACP PromptResponse.usage deltas for native subscription runs."""
        latest_fn = getattr(self._session, "latest_usage_totals", None)
        if not callable(latest_fn):
            return
        latest = latest_fn()
        if not latest:
            return
        delta = _native_acp_usage_delta(self._native_usage_checkpoint, latest)
        self._native_usage_checkpoint = dict(latest)
        if not any(delta.values()):
            return

        metrics = dict(self._native_usage_metrics)
        for (
            snapshot_field,
            result_field,
        ) in _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT.items():
            if result_field == "total_tokens":
                continue
            metrics[result_field] = _as_nonnegative_int(metrics.get(result_field)) + (
                delta.get(snapshot_field) or 0
            )
        metrics["total_tokens"] = _as_nonnegative_int(metrics.get("total_tokens")) + (
            delta.get("total_tokens") or 0
        )
        details = dict(metrics.get("usage_details") or {})
        details["thought_tokens"] = _as_nonnegative_int(
            details.get("thought_tokens")
        ) + (delta.get("thought_tokens") or 0)
        metrics["usage_details"] = details
        metrics["usage_source"] = USAGE_SOURCE_AGENT_NATIVE_ACP
        metrics["cost_usd"] = None
        metrics["price_source"] = None
        self._native_usage_metrics = metrics

    def _build_step_batch(self, new_events: list[dict], new_tools: int) -> list[Step]:
        """Build one Step per ACP event from the events appended this execute.

        Step-level granularity (closes #414) — each ACP event (tool_call,
        agent_message, agent_thought, user_message) becomes a Step the tree
        can address for branching, reward shaping, and value estimation.
        Empty-event executes still produce one Step so the cursor advances
        and any pending branch-child node gets populated.
        """
        base = len(self._trajectory) - len(new_events)
        if not new_events:
            return [
                Step(
                    id=f"step-{base}-empty",
                    data={"event": None, "n_tool_calls": 0},
                )
            ]
        steps: list[Step] = []
        for offset, event in enumerate(new_events):
            traj_index = base + offset
            event_type = (
                event.get("type", "event") if isinstance(event, dict) else "event"
            )
            is_tool_call = event_type == "tool_call"
            steps.append(
                Step(
                    id=f"step-{traj_index}-{event_type}",
                    data={
                        "event": event,
                        "event_type": event_type,
                        "n_tool_calls": 1 if is_tool_call else 0,
                    },
                )
            )
        # n_tool_calls across the batch should equal the new_tools reported
        # by execute_prompts. If they disagree (legacy/non-tool_call events
        # counted as tools by the agent shim) attribute the remainder to the
        # last step so the per-execute total still matches.
        batch_tools = sum(s.data["n_tool_calls"] for s in steps)
        if batch_tools != new_tools and steps:
            steps[-1].data["n_tool_calls"] += new_tools - batch_tools
        return steps

    # Phase 3d: BRANCH

    async def branch(
        self,
        n: int,
        run_child: ChildRunner | IdentifiedChildRunner | None = None,
        *,
        require_sandbox_snapshot: bool = False,
        snapshot_layers: frozenset[str] | set[str] | None = None,
        child_labels: list[str | None] | None = None,
        retain_snapshots: bool = False,
        restore_parent: bool = True,
        child_requests: list[str | None] | None = None,
        isolate_children: bool = False,
        concurrency: int = 1,
        resume_session: bool = False,
        child_retries: int = 0,
        continue_after_child_failure: bool = False,
        resume_session_id: str | None = None,
        reuse_snapshot: Any = None,
    ) -> float:
        """Branch the rollout at the cursor into ``n`` child continuations.

        Thin entry point; the engine lives in :mod:`benchflow.rollout_branch`.
        It disconnects the agent, checkpoints the requested
        ``snapshot_layers``, runs each child from that checkpoint with a fresh
        agent session, restores the parent's world, and returns V(parent), the
        mean of the children's rewards. The agent stays disconnected
        afterwards: call :meth:`connect` before continuing the parent.

        ``snapshot_layers``: ``{"environment"}`` (default; declared database
        state, needs an environment manifest), ``{"sandbox"}`` (the container
        filesystem, Docker or Daytona direct; the choice for a plain task), or
        both. Children do not inherit the agent's conversation, only the
        restored files and declared state, so child prompts must be
        self-contained.

        ``run_child(node)`` runs one child; declare a keyword-only ``child``
        parameter to also receive a :class:`~benchflow.rollout_branch.BranchChild`
        (index, label, node, fork id). The default runner uses the task's
        prompts and verifier. A runner that calls :meth:`verify` and returns
        that reward is recorded as ``reward_source: "verifier"``; a missing
        verifier reward or a ``None`` return leaves the child unscored.

        ``child_labels`` names the children (one per child) in ``tree.json``;
        ``child_requests`` describes what each child's runner does differently.
        ``retain_snapshots=True`` keeps the fork's container snapshot, which is
        otherwise deleted when the fork finishes. ``require_sandbox_snapshot``
        only checks that the sandbox supports container snapshots.
        ``restore_parent=False`` skips the parent restore after the last child
        (one restore fewer); the rollout then refuses to continue and only
        :meth:`finalize` or :meth:`cleanup` remain. ``isolate_children=True``
        runs each child in its own sandbox created from the snapshot, at most
        ``concurrency`` at once; the runner drives ``child.rollout``, which
        can branch again. ``resume_session=True`` makes each child's
        ``connect()`` resume the parent's agent conversation (ACP
        ``session/load``) instead of starting a fresh one. ``child_retries``
        retries a child that failed before its agent did anything;
        ``continue_after_child_failure`` lets in-place siblings run after a
        child failed. ``reuse_snapshot`` (sandbox layer only) is an existing
        image of the sandbox at the cursor, such as an automatic checkpoint,
        used as the fork's snapshot instead of taking another; the fork never
        deletes it.
        See ``docs/composed-checkpoints.md``.
        """
        return await _branch_engine(
            self,
            n,
            run_child,
            require_sandbox_snapshot=require_sandbox_snapshot,
            snapshot_layers=snapshot_layers,
            child_labels=child_labels,
            retain_snapshots=retain_snapshots,
            restore_parent=restore_parent,
            child_requests=child_requests,
            isolate_children=isolate_children,
            concurrency=concurrency,
            resume_session=resume_session,
            child_retries=child_retries,
            continue_after_child_failure=continue_after_child_failure,
            resume_session_id=resume_session_id,
            reuse_snapshot=reuse_snapshot,
        )

    # Phase 4: VERIFY

    async def verify(self) -> dict | None:
        """Run the verifier and return rewards."""
        require_safe_branch_world(self)
        # getattr: callers drive verify() on partially built stand-ins too.
        self._verify_calls = getattr(self, "_verify_calls", 0) + 1
        cfg = self._config

        from benchflow.rollout._verifier_recovery import (
            PRESERVED_SOLVER,
            mark_solver_complete,
            recovery_ineligible_reason,
        )

        mark_solver_complete(self)
        # Mark the phase at entry (the other transitions mark completion):
        # the verifier can run for minutes after disconnect() reset the phase
        # to "installed", and the dashboard's activity cell reads _phase to
        # label that stretch "verifying…" instead of going blank.
        self._phase = "verifying"
        await capture_terminal_workspace(self)
        await collect_rollout_artifacts(self)

        if not self._trajectory and not is_scripted_agent(cfg.primary_agent):
            scraped = await _scrape_agent_trajectory(
                self._env, cfg.primary_agent, cfg.sandbox_user
            )
            if scraped:
                self._trajectory = scraped
                self._trajectory_source = "scraped"
                logger.warning(
                    f"Using scraped trajectory ({len(scraped)} events) — UNTRUSTED"
                )

        if separate_verifier_requested(getattr(self._task, "config", None)):
            # The verifier never runs in the agent's sandbox: a fresh one gets
            # only the frozen workspace and the collected artifacts.
            self._rewards, self._verifier_error = await run_separate_verifier(self)
            self._phase = "verified"
            return self._rewards

        # Only a task with a verifier-only recovery contract keeps its verifier
        # failures off the solver-retry path. Every other task keeps main's
        # error text and exceptions, which evaluation retries as before.
        recovery_eligible = recovery_ineligible_reason(self) is None
        verifier_timeout_diag = None
        try:
            await _publish_trajectory_for_verifier(
                self._env, self._trajectory, self._rollout_paths.agent_dir
            )

            (
                self._rewards,
                self._verifier_error,
                verifier_timeout_diag,
            ) = await _verify_rollout(
                self._env,
                self._task,
                self._rollout_paths,
                self._timing,
                self._planes,
                sandbox_user=cfg.sandbox_user,
                workspace=self._agent_cwd,
                recovery_eligible=recovery_eligible,
            )
        except Exception as exc:
            if not recovery_eligible:
                raise
            self._rewards = None
            self._verifier_error = f"verifier crashed: {exc}"
        if recovery_eligible and self._verifier_error is not None:
            self._verifier_error = f"{PRESERVED_SOLVER} {self._verifier_error}"
        if verifier_timeout_diag is not None:
            self._diagnostics.set(verifier_timeout_diag)

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
        if separate_verifier_requested(getattr(self._task, "config", None)):
            # Soft verification would upload tests/ into the agent's sandbox.
            return (
                None,
                None,
                "soft_verify is unavailable: the task uses a separate verifier "
                "sandbox, so tests never enter the agent's sandbox",
            )
        require_safe_branch_world(self)
        self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        try:
            await self._planes.clear_verifier_output_dir(
                self._env,
                "Soft verifier setup failed: clearing verifier output directory",
                user="root",
                timeout_sec=10,
            )
            await self._planes.ensure_legacy_app_dir(
                self._env,
                "Soft verifier setup failed: preparing /app",
                user="root",
                timeout_sec=10,
            )
            # Purge agent-injected conftest/sitecustomize/.pth without
            # killing processes or restoring workspace.
            # Honor per-task [verifier.hardening] opt-outs from task config.
            # No timeout_sec here: the conftest purge walks the rootfs and can be
            # slow on network-backed FS (Daytona), so its budget is owned by
            # lockdown.cleanup_verifier_python_hooks (VERIFIER_SETUP_TIMEOUT_SEC),
            # shared with the scoring path in harden_before_verify. The except
            # below keeps the step fail-closed.
            await self._planes.cleanup_verifier_python_hooks(
                self._env,
                getattr(self._task, "task_dir", None),
                "Soft verifier setup failed: purging Python injection hooks",
                user="root",
            )
        except Exception as e:
            # describe_exception for the same reason as the funnel below: an
            # argument-less exception stringifies to nothing (#1065).
            verifier_error = f"soft verifier crashed: {describe_exception(e)}"
            logger.error(verifier_error)
            return None, None, verifier_error

        rewards = None
        verifier_output = None
        verifier_error = None
        try:
            verifier = self._planes.verifier(
                task=self._task,
                rollout_paths=self._rollout_paths,
                sandbox=self._env,
            )
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = _ensure_canonical_rewards(
                verifier_result.rewards, task=self._task
            )
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
            verifier_error = f"soft verifier crashed: {describe_exception(e)}"
            logger.error(verifier_error)
        return rewards, verifier_output, verifier_error

    # Phase 5: CLEANUP

    async def cleanup(self) -> None:
        """Close ACP client and stop the environment."""
        self._capture_partial_acp_trajectory()
        await self.disconnect()

        if self._env and self._config.export_generated_skills_to:
            try:
                await self._export_generated_skills()
            except Exception as e:
                # Surface export failure on a dedicated sibling channel
                # (#389 follow-up). Routing it through self._error caused
                # classify_error("Skill export failed: ... connection lost")
                # to mis-tag the rollout as agent infra_failure, polluting
                # the agent-error dashboards. Keep the agent/verifier error
                # channels untouched: export runs during cleanup, after the
                # agent already finished.
                export_error = f"Skill export failed: {e}"
                logger.error(export_error)
                if self._export_error is None:
                    self._export_error = export_error
                self._evolved_skills = None

        usage_runtime = self._usage_runtime
        if usage_runtime is not None:
            try:
                await self._planes.stop_provider_runtime(usage_runtime)
                self._usage_metrics = self._planes.extract_usage(usage_runtime)
            except Exception as e:
                logger.warning(f"Usage telemetry runtime stop failed: {e}")
                self._usage_metrics = self._planes.extract_usage(None)
            # Snapshot any provider failure (401/403/429/503) now that captures
            # are imported (stop() populated the trajectory). This must happen
            # before we drop the runtime reference below, and is read later by
            # ACP-error classification — for Daytona the trajectory is empty
            # until here (#546/#564).
            #
            # Coverage gap: only `self._usage_runtime` is scanned here. Bedrock
            # auth failures flow through `self._provider_runtime`, whose server
            # (BedrockProxyServer) exposes no `.trajectory`/`.exchanges`, so a
            # fallback scan of it would always return None — useless, so it's
            # not implemented. The direct-AWS-Bedrock case (remote sandbox,
            # runtime=None) bypasses both proxies entirely and is out of scope.
            self._provider_failure_cached = _provider_failure_from_runtime(
                usage_runtime
            )
            self._provider_auth_status_cached = (
                self._provider_failure_cached.status
                if self._provider_failure_cached is not None
                and self._provider_failure_cached.marker == "provider auth failed"
                else None
            )
            self._api_failure_summary_cached = (
                _provider_api_failure_summary_from_runtime(usage_runtime)
            )
            try:
                self._write_llm_trajectory(usage_runtime)
            except Exception as e:
                logger.warning(f"LLM trajectory write failed: {e}")
            try:
                self._reconcile_acp_tool_evidence(usage_runtime)
            except Exception as e:
                logger.warning(f"ACP tool-evidence reconciliation failed: {e}")
            finally:
                self._usage_runtime = None

        rollout_dir = self._rollout_dir
        if (
            (
                self._egress_denylist is not None
                or self._active_egress_policy is not None
            )
            and self._env is not None
            and rollout_dir is not None
        ):
            try:
                await self._planes.stop_egress_denylist(self._env, rollout_dir)
            except Exception as e:
                logger.warning(f"Egress denylist proxy stop failed: {e}")

        self._finalize_usage_metrics()
        self._enforce_required_usage_tracking()

        if self._environment is not None:
            with contextlib.suppress(Exception):
                await self._environment.teardown()
            self._environment = None

        if self._env and not self._env_externally_owned:
            # An externally-owned sandbox (use_prebuilt_env) belongs to the
            # caller — leave it running so they can reuse it or stop it
            # themselves. #388.
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        from benchflow.rollout._verifier_recovery import release_lease_at_teardown

        await release_lease_at_teardown(self)

        if self._task_tmp:
            shutil.rmtree(self._task_tmp, ignore_errors=True)

        self._phase = "cleaned"

    def _finalize_usage_metrics(self) -> None:
        """Prefer LiteLLM usage, otherwise use trusted native ACP usage."""
        if self._usage_metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE:
            return
        if is_token_usage_available(self._native_usage_metrics):
            self._usage_metrics = self._native_usage_metrics

    def _enforce_required_usage_tracking(self) -> None:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        if usage_cfg.mode != "required" or is_scripted_agent(
            self._config.primary_agent
        ):
            return
        if is_token_usage_available(self._usage_metrics):
            return
        if self._error is not None:
            return
        self._error = (
            "Token usage tracking is required, but no provider token usage was "
            "captured."
        )
        logger.error(self._error)

    # Full run

    def _record_agent_timeout(self, e: TimeoutError, *, agent_phase: bool) -> None:
        """Record a timed-out agent run on the rollout's error state.

        Shared by run()'s inner per-scene handler and the outer wall-clock
        handler. Preserves the watchdog's diagnostic message ("Agent idle
        for 600s with no new tool call ...") when it raised one, falling
        back to the generic wall-clock message only when there's no detail.

        A BenchFlow-owned wall-clock prompt timeout (``AgentPromptTimeoutError``)
        that fired with no pending tool calls is a *clean terminal* timeout:
        the trajectory is complete, not a rerunnable partial. Record that so
        the partial-capture path leaves ``_partial_trajectory`` False (#640).

        Adapted from PR #1131. A detail-less timeout reports measured
        rollout elapsed time, not the configured budget or agent-phase time,
        and — only when it came from the agent phase — is flagged for
        zero-activity reclassification (#1071). Setup/install/verify timeouts
        reach the outer handler with ``agent_phase=False`` and never reclassify.
        """
        detail = str(e).strip()
        self._bare_timeout = not detail and agent_phase
        if not detail and self._started_at is not None:
            elapsed = (datetime.now() - self._started_at).total_seconds()
            detail = (
                f"Agent timed out after {elapsed:.0f}s elapsed in rollout "
                f"(budget {self._timeout}s)"
            )
        self._error = detail or f"Agent timed out after {self._timeout}s"
        self._diagnostics.capture_idle(e)
        if isinstance(e, AgentPromptTimeoutError) and getattr(
            e, "terminal_trajectory_complete", False
        ):
            self._terminal_timeout = True
        logger.error(self._error)

    async def run(self) -> RolloutResult:
        """Run the complete trial lifecycle under a host-side hard deadline.

        The lifecycle itself lives in :meth:`_run_lifecycle`; the deadline is
        a backstop against awaits wedged below every phase-level timeout (a
        Daytona PTY kill on a dead websocket, a hung session exec in the
        post-verify export path) — see :mod:`benchflow.rollout._deadline`.
        A trip while the agent is still running stops only the agent and is
        reported as its timeout after verification (#1134); any other trip
        becomes a normal infra-retryable error result and the abandoned
        attempt's cleanup is bounded too.
        """
        result = await _deadline.enforce_hard_deadline(
            self._run_lifecycle(),
            config=self._config,
            stop_agent=self._stop_agent_at_deadline,
        )
        if self._solver_execution_complete and not result.rollout_name:
            from benchflow.rollout._verifier_recovery import interrupted_solver_result

            result = interrupted_solver_result(
                self, result.error or "post-solver deadline"
            )
        return await self._finish_scoring(result)

    async def _finish_scoring(self, result: RolloutResult) -> RolloutResult:
        """Serialize initial scoring and resumed scoring through the same lock."""
        if result.rollout_name and self._rollout_dir is not None:
            with scoring_lock(self._rollout_dir):
                result = await self._finish_scoring_locked(result)
            if result.rollout_dir is None:
                result.rollout_dir = self._rollout_dir
            return result
        self._completed_result = result
        return result

    async def _finish_scoring_locked(self, result: RolloutResult) -> RolloutResult:
        """Shared verifier-only recovery and review after solver cleanup."""
        from benchflow.rollout._verifier_recovery import (
            needs_verifier_recovery,
            recover_verifier,
            recovery_ineligible_reason,
            release_recovery_lease,
        )

        rollout_dir = self._require_rollout_dir()
        # A reviewed trial's result.json is written only when its scoring
        # commits; until then solver.json, plus verification.json after a
        # recovery, is its only record.
        publish = None if self._review_plan is not None else "result.json"
        # Another scorer may have finished while this caller waited to
        # enter finalization. Never replace its complete verdict with a
        # stale failed-attempt result.
        current_path = rollout_dir / "result.json"
        if current_path.is_file():
            current = json.loads(current_path.read_text())
            scoring = scoring_from_result(current)
            if scoring is not None and scoring.status == "complete":
                result.scoring = self._scoring = scoring
                result.rewards = self._rewards = current.get("rewards")
                result.verifier_error = self._verifier_error = None
                result.verifier_error_category = None
                await release_recovery_lease(self)
                self._completed_result = result
                return result
        if (rollout_dir / "verification.json").is_file():
            from benchflow.rollout._verifier_recovery import verification_source

            current = verification_source(rollout_dir)
            if current.get("rewards") is not None and not current.get("verifier_error"):
                self._rewards, self._verifier_error = current["rewards"], None
                result = self._build_result(result_filename=publish)
        if (
            result.rollout_name
            and self._phase == "cleaned"
            and (rollout_dir / "solver.json").is_file()
            and needs_verifier_recovery(result.verifier_error)
            # Without a recovery contract the original error stays retryable.
            and recovery_ineligible_reason(self) is None
        ):
            # A finished attempt releases the lease itself; an interrupted
            # one keeps it for `bench eval score` resume.
            try:
                self._rewards, self._verifier_error = await recover_verifier(self)
            except Exception as exc:
                self._rewards = None
                self._verifier_error = (
                    f"[solver-preserved] verifier recovery unavailable: {exc}"
                )
            result = self._build_result(result_filename=publish)
        else:
            await release_recovery_lease(self)
        # Scoring queues own no solver VM and may legitimately outlive its
        # deadline. Each reviewer has its own bounded rollout lifecycle.
        if (
            result.rollout_name
            and self._review_plan is not None
            and self._phase == "cleaned"
            and (rollout_dir / "solver.json").is_file()
        ):
            result = await finish_terminal_review(self, result=result, lock_held=True)
        self._completed_result = result
        return result

    async def finalize(self) -> RolloutResult:
        """Finish a manually driven rollout, including required rubric review."""
        if self._completed_result is not None:
            return self._completed_result
        if self._phase != "cleaned":
            await self.cleanup()
        if self._rollout_dir is None:
            # Keep the existing manually driven/test seam; real artifact
            # recovery requires a setup-created rollout directory.
            if self._review_plan is not None:
                self._completed_result = await finish_terminal_review(self)
            else:
                self._completed_result = self._build_result()
            return self._completed_result
        return await self._finish_scoring(prepare_terminal_result(self))

    async def _run_lifecycle(self) -> RolloutResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        agent_timed_out = False
        pending_acp_error: AgentProtocolError | None = None
        if cfg.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.run()."
            )
        try:
            await self.setup()
            await self.start()

            if cfg.primary_agent == "nop":
                # The empty control: nothing runs, the verifier scores the
                # untouched workspace.
                await self.install_agent()
                self._trajectory, self._agent_name = _nop_trajectory()
                from benchflow.rollout._verifier_recovery import mark_solver_complete

                mark_solver_complete(self)
            elif cfg.primary_agent == "oracle":
                await self.install_agent()
                # git safe.directory needed for SWE-bench tasks with sandbox_user
                await self._env.exec(
                    f"git config --global --add safe.directory "
                    f"{shlex.quote(self._agent_cwd)} 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
                oracle_started = datetime.now()
                self._trajectory, self._agent_name = await _run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
                )
                self._timing["agent_execution"] = (
                    datetime.now() - oracle_started
                ).total_seconds()
                from benchflow.rollout._verifier_recovery import mark_solver_complete

                mark_solver_complete(self)
            else:
                await self.install_agent()
                try:
                    try:
                        if cfg.user is not None:
                            await self._run_agent_phase(self._run_user_loop())
                        else:
                            await self._run_agent_phase(
                                self._run_steps(
                                    compile_scenes_to_steps(
                                        cfg.effective_scenes,
                                        default_prompt=(
                                            self._resolved_prompts[0]
                                            if self._resolved_prompts
                                            else None
                                        ),
                                    )
                                )
                            )
                        from benchflow.rollout._verifier_recovery import (
                            mark_solver_complete,
                        )

                        mark_solver_complete(self)
                    except TimeoutError as e:
                        agent_timed_out = True
                        self._record_agent_timeout(e, agent_phase=True)
                finally:
                    if cfg.oracle_access:
                        await self._env.exec(
                            "mv /oracle_backup /oracle 2>/dev/null || true; "
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
            if self._solver_execution_complete:
                self._verifier_error = (
                    f"[solver-preserved] post-solver stage timed out: {e}"
                )
            else:
                self._record_agent_timeout(e, agent_phase=False)
        except ConnectionError as e:
            if self._solver_execution_complete:
                self._verifier_error = f"[solver-preserved] verifier crashed: {e}"
            else:
                self._error = str(e)
                self._diagnostics.capture_transport(e)
                await self._probe_sandbox_health()
                logger.error(f"Agent connection lost: {self._error}")
        except SandboxStartupFailure as e:
            self._error = f"Sandbox startup failed: {e}"
            self._diagnostics.set(e.diagnostic)
            logger.error(self._error)
        except AgentModelNotOfferedError as e:
            # Already an agent-integration error text; keep it verbatim so
            # its category and the batch circuit breaker see the cause.
            self._error = str(e)
            self._diagnostics.set(
                IntegrationFailureDiagnostic(
                    cause="agent_model",
                    evidence=str(e).split(": ", 1)[-1],
                    evidence_source="session/new availableModels",
                )
            )
            logger.error(self._error)
        except AgentProtocolError as e:
            # Defer classification until after cleanup(): the provider 401/403
            # that distinguishes provider_auth from a generic retryable ACP
            # error lives in the usage-proxy trajectory, which Daytona's
            # SandboxUsageProxy only imports on stop() (#546/#564).
            pending_acp_error = e
            # Set a provisional error so cleanup()'s
            # _enforce_required_usage_tracking guard early-returns instead of
            # logging a misleading "no provider token usage was captured"
            # message — the agent failed with an ACP error, not a usage gap.
            # The post-cleanup block below still unconditionally refines
            # self._error to the provider_auth marker, so this is only a
            # placeholder during cleanup.
            self._error = str(e)
            logger.error(str(e))
        except Exception as e:
            # describe_exception, not str(e): this is the funnel every
            # unclassified rollout failure lands in, and some SDK errors
            # stringify to a bare wrapper prefix with no detail behind it.
            # Persisting those raw leaves an artifact that names neither what
            # failed nor that the detail was empty.
            if self._solver_execution_complete:
                self._verifier_error = f"[solver-preserved] post-solver stage failed: {describe_exception(e)}"
            else:
                self._error = describe_exception(e)
            logger.error("Run failed", exc_info=True)
        finally:
            try:
                await self.cleanup()
            except Exception as exc:
                if not self._solver_execution_complete:
                    raise
                self._verifier_error = f"[solver-preserved] post-solver cleanup failed: {describe_exception(exc)}"
                self._export_error = (
                    "Post-solver cleanup did not finalize telemetry/evidence"
                )

        # cleanup() has now imported usage-proxy captures and snapshotted any
        # provider auth status, so classification can see the real 401/403.
        if pending_acp_error is not None:
            self._error = self._classify_acp_error(pending_acp_error)
            logger.error(self._error)

        if self._rollout_dir is None:
            return RolloutResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return prepare_terminal_result(self)

    async def _run_agent_phase(self, phase: Coroutine[Any, Any, None]) -> None:
        """Run the agent in its own task so the host deadline can stop it alone.

        When :meth:`_stop_agent_at_deadline` cancelled it, the cancellation
        becomes the agent's ``TimeoutError`` and the lifecycle goes on to
        verify, as for any other agent timeout (#1134). Any other
        cancellation, including one of the lifecycle itself, propagates.
        """
        agent = asyncio.ensure_future(phase)
        self._agent_phase = agent
        try:
            await agent
        except asyncio.CancelledError:
            reason = self._agent_deadline_reason
            current = asyncio.current_task()
            if reason is None or (current is not None and current.cancelling()):
                raise
            raise TimeoutError(reason) from None
        finally:
            self._agent_phase = None

    def _stop_agent_at_deadline(self, reason: str) -> bool:
        """Stop a still-running agent phase; False once it has ended."""
        agent = self._agent_phase
        if agent is None or agent.done():
            return False
        self._agent_deadline_reason = reason
        agent.cancel()
        return True

    # Scene-authored Step execution
    #
    # The step / user-loop drivers and the generated-skill export hook live in
    # ``benchflow.rollout._user_loop`` as free functions taking this Rollout —
    # the same engine convention as ``rollout_branch.py``. These thin methods
    # keep instance-level patching and unbound ``Rollout._export_generated_skills``
    # calls working unchanged.

    async def _export_generated_skills(self) -> None:
        """Download creator-produced skills before sandbox cleanup.

        Also captures the exported skill packs into ``self._evolved_skills``
        — the ``name -> body`` dict a continual-learning Job commits to its
        persistent LearnerStore (capability 5).

        Retries transient download failures up to 3 times (guards ENG-147).
        """
        await _export_generated_skills_engine(self)

    async def _activate_step_skills(self, step: Step) -> None:
        """Activate scene-local skills attached by the Scene desugaring pass."""
        await _activate_step_skills_engine(self, step)

    async def _run_steps(self, steps: list[Step]) -> None:
        """Execute already-compiled rollout Steps in declaration order."""
        await _run_steps_engine(self, steps)

    async def _run_user_loop(self) -> None:
        """Execute a user-driven progressive-disclosure loop.

        Each round: user.run() → connect → agent.execute() → disconnect →
        soft_verify() → build RoundResult → repeat. Stops when user.run()
        returns None or max_user_rounds is reached.
        """
        await _run_user_loop_engine(self)

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role.

        Installs the role's agent binary and credentials if it differs
        from the primary agent (which was set up in install_agent()).
        Updates _agent_launch so disconnect() kills the correct process.
        """
        require_safe_branch_world(self)
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        (rollout_dir / "native-oauth-network.json").unlink(missing_ok=True)
        t0 = datetime.now()

        # Merge cfg.agent_env (config-level) with role.env (role-specific) so
        # provider creds from YAML reach the agent. role.env wins on overlap.
        disallow_web_tools = self._disallow_web_tools and not is_scripted_agent(
            role.agent
        )
        egress_denylist = (
            None
            if disallow_web_tools or is_scripted_agent(role.agent)
            else _task_egress_denylist(self._task)
        )
        disallow_hosted_search = egress_denylist is not None
        agent_launch = self._planes.agent_launch(
            role.agent,
            disallow_web_tools=disallow_web_tools,
            disallow_hosted_search=disallow_hosted_search,
        )
        agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                role.agent,
                role.model,
                {**(cfg.agent_env or {}), **(role.env or {})},
            ),
            disallow=disallow_web_tools,
        )
        egress_denylist = native_oauth_egress_policy(
            role.agent, role.model, agent_env, no_web=disallow_web_tools
        ) or allowlist_model_transport(
            egress_denylist, role.agent, role.model, agent_env
        )
        agent_env, self._usage_runtime = await self._planes.ensure_litellm_runtime(
            agent=role.agent,
            agent_env=agent_env,
            model=role.model,
            runtime=self._usage_runtime,
            environment=cfg.environment,
            session_id=self._rollout_name or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            required_skill_names=self._required_skill_names,
            live_trajectory_path=rollout_dir / "trajectory" / "llm_trajectory.jsonl",
            force_sandbox_local=disallow_web_tools or disallow_hosted_search,
        )
        if egress_denylist is not None:
            agent_env = denylist_agent_env(agent_env, egress_denylist)

        role_agent_differs = role.agent != cfg.primary_agent
        needs_role_credentials = (
            role_agent_differs or role.model != cfg.primary_model or bool(role.env)
        )
        if role_agent_differs:
            if cfg.skip_agent_install:
                agent_cfg = None
            else:
                agent_cfg = await self._planes.install_agent(
                    self._env,
                    role.agent,
                    rollout_dir,
                    sandbox_setup_timeout=cfg.sandbox_setup_timeout,
                )
        else:
            agent_cfg = self._agent_cfg
        if needs_role_credentials:
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await self._planes.write_credential_files(
                self._env,
                role.agent,
                agent_env,
                agent_cfg,
                role.model,
                cred_home,
            )
            await _install_native_task_mcp_config(
                self._env,
                self._task,
                agent_cfg=agent_cfg,
                cred_home=cred_home,
                owner=cfg.sandbox_user,
            )
            if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                await self._planes.upload_subscription_auth(
                    self._env, role.agent, cred_home
                )
            await self._planes.apply_web_tool_policy(
                self._env,
                role.agent,
                agent_cfg,
                cred_home,
                disallow=disallow_web_tools,
                disallow_hosted_search=disallow_hosted_search,
            )

        agent_env = await enforce_codex_apps_policy(
            self._env,
            agent=role.agent,
            agent_launch=agent_launch,
            agent_env=agent_env,
            sandbox_user=cfg.sandbox_user,
            policy=effective_apps_policy(
                cfg.codex_apps_policy, purpose=cfg.purpose, skip_verify=cfg.skip_verify
            ),
            requested=cfg.codex_apps_policy,
            rollout_dir=rollout_dir,
        )
        self._agent_launch = agent_launch

        if egress_denylist is None:
            await self._stop_active_egress()
        sf_entrypoint = self._session_factory_entrypoint(role.agent)
        self._is_session_factory = sf_entrypoint is not None
        if sf_entrypoint is not None:
            if egress_denylist is not None:
                raise RuntimeError(
                    f"network_mode={egress_denylist.mode!r} requires an ACP agent"
                )
            (
                self._acp_client,
                self._session,
                self._session_adapter,
                self._agent_name,
            ) = await self._planes.connect_session_factory(
                env=self._env,
                agent=role.agent,
                session_factory=sf_entrypoint,
                agent_env=agent_env,
                sandbox_user=cfg.sandbox_user,
                model=role.model,
                rollout_dir=rollout_dir,
                timeout=(
                    role.timeout_sec if role.timeout_sec is not None else self._timeout
                ),
                agent_cwd=self._agent_cwd,
            )
        else:
            if egress_denylist is not None:
                await self._start_egress_denylist(
                    egress_denylist, agent_launch=agent_launch
                )
            (
                self._acp_client,
                self._session,
                self._session_adapter,
                self._agent_name,
            ) = await self._planes.connect_acp(
                env=self._env,
                agent=role.agent,
                agent_launch=agent_launch,
                agent_env=agent_env,
                sandbox_user=cfg.sandbox_user,
                model=role.model,
                rollout_dir=rollout_dir,
                environment=cfg.environment,
                agent_cwd=self._agent_cwd,
                reasoning_effort=role.reasoning_effort,
                mcp_servers=_task_mcp_specs_for_agent(
                    role.agent, self._task, agent_cfg
                ),
            )
        self._native_usage_checkpoint = None
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)
        self._active_role = role

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # Internal helpers

    async def _probe_sandbox_health(self) -> None:
        """Quick health probe after transport death. Enriches transport diagnostic.

        Guards ENG-148: distinguishes Daytona session killed vs agent crash.
        """
        diag = self._diagnostics.transport_closed
        if diag is None or self._env is None:
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
                diag.sandbox_reachable = True
                diag.sandbox_probe_rc = rc
            else:
                diag.sandbox_reachable = False
                diag.sandbox_probe_rc = rc
                diag.sandbox_probe_stdout = stdout[:200]
        except Exception as probe_err:
            import traceback

            logger.exception("sandbox health probe failed")
            diag.sandbox_reachable = False
            diag.sandbox_probe_error = str(probe_err)[:200]
            diag.sandbox_probe_error_type = type(probe_err).__name__
            diag.sandbox_probe_traceback = traceback.format_exc()[-2000:]

    def _classify_acp_error(self, e: AgentProtocolError) -> str:
        # The base AgentProtocolError only annotates `message: str` without
        # assigning it, so a base instance has no `.message` (AttributeError
        # risk); ACPError subclasses do set it. Fall back to str(e) defensively.
        message = getattr(e, "message", str(e))
        if "Invalid API key" in message:
            from benchflow.agents.env import check_subscription_auth

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
        # A real provider failure often surfaces only as a generic
        # "ACP error -32603: Internal error" at this layer — the provider's
        # actual 401/403/429/503 is visible only in the proxy-captured
        # trajectory (#546/#564). Surface a sanitized marker (status code only —
        # never the response body or headers) so RetryConfig.should_retry can
        # classify it (auth/rate-limit fail fast; 503 stays retryable infra)
        # instead of burning retries on a generic ACP error.
        provider_failure = self._provider_failure()
        if provider_failure is not None:
            return f"{e} | {provider_failure.error_suffix}"
        return str(e)

    def _provider_failure(self) -> ProviderFailure | None:
        """Return the provider failure snapshotted during cleanup.

        Falls back to the auth-only status cache for partial Rollout doubles in
        tests that set ``_provider_auth_status_cached`` directly (#564).
        """
        failure = self._provider_failure_cached
        if failure is not None:
            return failure
        return _provider_failure_from_status(self._provider_auth_status())

    def _provider_auth_status(self) -> int | None:
        """Return the provider 401/403 status snapshotted during cleanup.

        The snapshot is taken in :meth:`cleanup` after the usage proxy imports
        its captures, so this is valid for both the host proxy (trajectory
        filled as requests complete) and Daytona's SandboxUsageProxy (filled
        only on ``stop()``). Only the integer status code is ever read — never
        response bodies or headers — so no credential material reaches
        ``result.error`` (#546/#564).
        """
        return self._provider_auth_status_cached

    def _write_llm_trajectory(self, usage_runtime: Any) -> None:
        """Persist captured provider HTTP exchanges as JSONL."""
        if self._rollout_dir is None:
            return
        trajectory = getattr(getattr(usage_runtime, "server", None), "trajectory", None)
        if trajectory is None or not trajectory.exchanges:
            return
        LiveLLMTrajectoryWriter(
            self._rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        ).reconcile(trajectory)

    def _reconcile_acp_tool_evidence(self, usage_runtime: Any) -> None:
        """Repair lossy ACP tool details from trusted provider capture."""

        if self._rollout_dir is None:
            return
        trajectory = getattr(getattr(usage_runtime, "server", None), "trajectory", None)
        exchanges = getattr(trajectory, "exchanges", None)
        if not isinstance(exchanges, list) or not exchanges:
            return
        provider_evidence = _parse_provider_tool_evidence(exchanges)
        self._trajectory, repaired = _reconcile_tool_evidence(
            self._trajectory, provider_evidence
        )
        if not repaired:
            return
        TrajectoryWriter(
            self._rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        ).write_final(self._trajectory)
        # info, not warning: trajectory repair is evidence-mutation an auditor
        # should find at default (non-TTY/CI) verbosity, but as a warning it
        # survived the live dashboard's WARNING+ replay and printed between
        # teardown and the score line.
        logger.info(
            "Repaired %d lossy ACP tool event(s) from trusted provider capture",
            repaired,
        )

    def _usage_tracking_metadata(self) -> dict[str, Any]:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        usage_source = str(self._usage_metrics.get("usage_source", "unavailable"))
        if usage_cfg.mode == "off":
            status = "off"
        elif is_token_usage_available(self._usage_metrics):
            status = "enabled"
        else:
            status = "unavailable"
        return usage_cfg.to_result_metadata(
            environment=self._config.environment,
            status=status,
            usage_source=usage_source,
        )

    def _current_sandbox_id(self) -> str | None:
        if isinstance(self._sandbox_id, str):
            return self._sandbox_id
        env_sandbox_id = getattr(self._env, "sandbox_id", None)
        return env_sandbox_id if isinstance(env_sandbox_id, str) else None

    def _maybe_classify_api_error(self) -> None:
        """Detect a silent provider API failure after the rollout finished.

        Runs only when no other error was recorded. Layer 1 (proxy-proven):
        every captured provider request failed and the agent produced zero
        tokens -> error_category "api_error". Layer 2 (zero-signal): no proxy
        failure evidence, but the agent ended with zero tokens AND zero tool
        calls -> "suspected_api_error" (e.g. the agent rejected the model id
        against its own catalog and never issued a request). Both null the
        reward so the slot is excluded from score denominators instead of
        polluting them as a fake healthy fail; the slot stays rerun-able and
        the batch is never interrupted.
        """
        # A bare agent-phase timeout is judged despite both gates: with zero
        # activity it is the same zero-signal shape (#1071). Otherwise skip
        # already-errored rollouts and setup/export failure paths, which own
        # their error channels (#389).
        bare_timeout = self._bare_timeout
        if bare_timeout and self._n_tool_calls > 0:
            # Missing token telemetry cannot erase observed work, even when
            # the provider's last captured request failed (PR #1131).
            return
        if not bare_timeout and (self._error is not None or not self._executed_prompts):
            return
        # Native-subscription runs have NO usage channel: the LiteLLM proxy is
        # deliberately skipped (Harbor-style split) and the CLI authenticates
        # itself, so zero tokens + zero tool calls is the expected shape of a
        # HEALTHY run for agents whose trajectory carries no tool telemetry
        # (e.g. omnigent's flat session events). The zero-signal heuristic is
        # meaningless there and would null verifier-granted rewards; real
        # failures still surface via the agent error channels.
        from benchflow.agents.env import uses_native_subscription_auth

        native = uses_native_subscription_auth(
            self._config.agent, self._config.model, self._agent_env
        )
        total_tokens = _as_nonnegative_int(self._usage_metrics.get("total_tokens"))
        verdict, info = (
            (None, {})
            if native
            else classify_api_failure(
                self._api_failure_summary_cached,
                total_tokens=total_tokens,
                n_tool_calls=self._n_tool_calls,
            )
        )
        # A broken agent integration is judged on what the agent did (tool
        # calls, messages, thoughts), so it also covers subscription runs; a
        # proxy-proven provider failure keeps its more specific verdict.
        if verdict != "api_error" and _classify_integration_failure(self):
            return
        if verdict is None:
            return
        self._bare_timeout = False
        if verdict == "api_error":
            subcategory = info.get("subcategory") or "provider_error"
            kind = "transient" if info.get("transient") else "permanent"
            diag = ProviderApiErrorDiagnostic(
                subcategory=subcategory,
                transient=bool(info.get("transient")),
                dominant_status=info.get("dominant_status"),
                status_counts=info.get("status_counts"),
                total_requests=info.get("total_requests") or 0,
                failed_requests=info.get("failed_requests") or 0,
                fingerprint=info.get("fingerprint") or "",
            )
            self._diagnostics.set(diag)
            self._error = (
                f"provider api error [{subcategory}/{kind}] "
                f"HTTP {info.get('dominant_status')} on "
                f"{diag.failed_requests}/{diag.total_requests} requests"
            )
        else:
            diag = SuspectedApiErrorDiagnostic(
                total_tokens=total_tokens,
                n_tool_calls=self._n_tool_calls,
                total_requests=info.get("total_requests") or 0,
                failed_requests=info.get("failed_requests") or 0,
            )
            self._diagnostics.set(diag)
            self._error = (
                "suspected provider api error: agent ended with zero tokens "
                "and zero tool calls (no scoreable model activity)"
            )
        # Unhealthy by definition: drop any verifier reward so the slot is
        # excluded from score denominators (rerun-able, never counted).
        self._rewards = None

    def _loop_strategy_metadata(self) -> dict[str, Any] | None:
        """Loop-strategy run summary for the result.json ``loop`` block.

        Computed at result-build time — after run() has finalized
        ``self._error`` on every path (agent timeout, ACP error, success) —
        from the engine's in-loop round log, so a mid-round crash still
        reports the rounds that completed.
        """
        user = self._config.user
        if self._config.loop_strategy_spec is None or not isinstance(
            user, LoopStrategyUser
        ):
            return None
        return collect_loop_metadata(
            user,
            self._user_rounds_log,
            max_rounds=self._config.max_user_rounds,
            error=self._error,
        )

    def _build_result(
        self, *, result_filename: str | None = "result.json"
    ) -> RolloutResult:
        rollout_dir = self._require_rollout_dir()
        # Provider telemetry is imported during cleanup; a pre-cleanup stage
        # checkpoint must not classify its temporarily missing usage as failure.
        if result_filename != "solver-complete.json":
            self._maybe_classify_api_error()
        # For Scene/multi-turn rollouts, each execute() call records the
        # prompt(s) it sent into self._executed_prompts. Use that as the
        # authoritative prompt list so n_prompts and prompts.json reflect
        # every prompt the agent actually received (issue #377). Fall back
        # to the resolved base prompts when no execute() ran (e.g. setup
        # failure paths).
        prompts = self._executed_prompts or self._resolved_prompts
        return _build_rollout_result(
            rollout_dir,
            task_name=self._config.task_path.name,
            rollout_name=self._rollout_name or "",
            agent=self._config.primary_agent,
            agent_name=self._agent_name,
            model=self._config.primary_model,
            n_tool_calls=self._n_tool_calls,
            prompts=prompts,
            error=self._error,
            verifier_error=self._verifier_error,
            export_error=self._export_error,
            trajectory=self._trajectory,
            partial_trajectory=self._partial_trajectory,
            trajectory_source=self._trajectory_source,
            rewards=self._rewards,
            started_at=self._require_started_at(),
            timing=self._timing,
            scenes=self._config.effective_scenes,
            evolved_skills=self._evolved_skills,
            source_provenance=self._config.source_provenance,
            dataset=self._config.dataset,
            task_digest=self._config.task_digest,
            scoring=self._scoring,
            purpose=self._config.purpose,
            parent_rollout=self._config.parent_rollout,
            result_filename=result_filename,
            diagnostics=self._diagnostics,
            usage_tracking=self._usage_tracking_metadata(),
            skill_policy=self._task_skill_policy
            or resolve_task_skill_policy(
                task_path=self._config.task_path,
                skill_mode=self._config.recorded_skill_mode,
                runtime_skills_dir=self._config.skills_dir,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_id=self._current_sandbox_id(),
            loop=loop_block(
                self._config.loop_strategy_spec,
                self._loop_strategy_metadata(),
            ),
            branches=self._branch_summary(),
            **self._usage_metrics,
        )

    def _branch_summary(self) -> dict[str, Any] | None:
        """result.json's ``branches`` block; None when the rollout never branched."""
        from benchflow.branch_lineage import branch_summary

        return branch_summary(getattr(self, "_branch_forks", []))


__all__ = [
    "BranchChild",
    "Role",
    "Scene",
    "Turn",
    "Rollout",
    "RolloutConfig",
    "BashToolResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TaskRuntimeResult",
]

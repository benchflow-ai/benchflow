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
import functools
import hashlib
import inspect
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
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from benchflow._diagnostics import render_text
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
from benchflow._utils import path_lock_record, timing_spans
from benchflow._utils.live_activity import ActivitySnapshot, SessionCounters
from benchflow._utils.log_context import trial_log_context
from benchflow._utils.reachable_write import ReachablePath, destination
from benchflow._utils.scoring import ORIGINAL_ERROR, PROVIDER_STALL
from benchflow._utils.text import describe_exception
from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.acp.types import McpServerSpec
from benchflow.agents.codex_connector_policy import (
    effective_apps_policy,
    enforce_codex_apps_policy,
)
from benchflow.agents.credentials import (
    prepare_launcher_credential,
    remove_muse_credentials,
    remove_staged_credentials,
    upload_credential,
)
from benchflow.agents.env import with_prompt_budget
from benchflow.agents.errors import UsageLimitError
from benchflow.agents.launch_env import agent_launch_env, credential_policy
from benchflow.agents.muse_config import validate_muse_roles
from benchflow.agents.registry import (
    AGENTS,
    has_builtin_install,
    infer_env_key_for_model,
    is_operator_script,
    is_scripted_agent,
    launcher_credential_plan,
    resolve_agent,
    session_factory_entrypoint,
)
from benchflow.agents.scripts import script_label
from benchflow.contracts import (
    AgentProtocolError,
    AskUserRequest,
    BaseUser,
    Environment,
    SandboxStartupFailure,
    default_rollout_planes,
)
from benchflow.contracts.planes import install_agent_with_policy
from benchflow.diagnostics import (
    AgentModelNotOfferedError,
    AgentPromptTimeoutError,
    IdleTimeoutError,
    IntegrationFailureDiagnostic,
    ProviderApiErrorDiagnostic,
    RolloutDiagnostics,
    SuspectedApiErrorDiagnostic,
    UsageLimitDiagnostic,
)
from benchflow.errors import UserError, user_message
from benchflow.loop_strategies import (
    LoopStrategyUser,
    collect_loop_metadata,
    loop_block,
)
from benchflow.models import AgentInstallError, RolloutResult, TrajectorySource
from benchflow.native_harness.harnesses import HARNESS_NATIVE, native_harness_for
from benchflow.review.automatic import PreparedReview
from benchflow.review.outcome import ScoringResult, scoring_from_result
from benchflow.review.persistence import scoring_lock
from benchflow.rollout import _deadline as _deadline
from benchflow.rollout._artifacts import (
    collect_rollout_artifacts,
    collect_unverified_rollout_artifacts,
)
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
    _record_reported_models as _record_reported_models,
)
from benchflow.rollout._results import (
    _user_confirmation_policy as _user_confirmation_policy,
)
from benchflow.rollout._results import _write_config as _write_config
from benchflow.rollout._results import _write_rewards_jsonl as _write_rewards_jsonl
from benchflow.rollout._review import (
    capture_terminal_workspace,
    finish_terminal_review,
    mark_agent_phase_start,
    prepare_capture_runtime,
    prepare_terminal_result,
    prepare_terminal_review,
    read_admitted_result,
)
from benchflow.rollout._separate_verifier import run_separate_verifier
from benchflow.rollout._session_log import price_from_session_log
from benchflow.rollout._setup import VerifierPublicationError
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
    _prepare_agent_temp_dirs as _prepare_agent_temp_dirs,
)
from benchflow.rollout._setup import (
    _publish_trajectory_for_verifier as _publish_trajectory_for_verifier,
)
from benchflow.rollout._setup import _resolve_agent_cwd as _resolve_agent_cwd
from benchflow.rollout._setup import _resolve_prompts as _resolve_prompts
from benchflow.rollout._setup import _run_oracle as _run_oracle
from benchflow.rollout._setup import _run_script as _run_script
from benchflow.rollout._setup import _start_env_and_upload as _start_env_and_upload
from benchflow.rollout._setup import _task_agent_offline as _task_agent_offline
from benchflow.rollout._setup import (
    _task_disallows_internet as _task_disallows_internet,
)
from benchflow.rollout._setup import _task_egress_denylist as _task_egress_denylist
from benchflow.rollout._setup import _verify_rollout as _verify_rollout
from benchflow.rollout._setup import trust_workspace_repos as trust_workspace_repos
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
from benchflow.rollout._user_loop import skill_export_root
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
from benchflow.sandbox._base import system_shell_exec
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox._in_order import in_order, runs_steps_at_once
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.cloud_metadata import (
    CLOUD_METADATA_RECORD,
    NOT_REACHABLE,
    PROVIDER_MANAGED,
    CloudMetadataPlan,
    CloudMetadataPowerError,
    cloud_metadata_plan,
    cloud_metadata_record,
    cloud_metadata_refusal,
    uncovered_services_note,
)
from benchflow.sandbox.egress_denylist import EgressDenylist, denylist_agent_env
from benchflow.sandbox.firewall_power import (
    FIREWALL_POWER_RECORD,
    FirewallPowerError,
    check_firewall_power,
    firewall_rules_hold_agent,
    forget_net_admin_drop,
)
from benchflow.sandbox.metadata import persist_sandbox_info
from benchflow.sandbox.native_oauth import (
    allowlist_model_transport,
    native_oauth_egress_policy,
    validate_native_oauth_transport,
)
from benchflow.scenes import compile_scenes_to_steps, scene_step_role
from benchflow.skill_policy import SKILL_MODE_NO_SKILL as SKILL_MODE_NO_SKILL
from benchflow.skill_policy import (
    SKILL_MODE_SELF_GEN,
    TaskSkillPolicy,
    resolve_task_skill_policy,
    strip_task_bundled_skills,
    task_bundled_skills_dir,
)
from benchflow.skill_policy import SKILL_MODE_WITH_SKILL as SKILL_MODE_WITH_SKILL
from benchflow.task import keep_services
from benchflow.task.verifier_errors import (
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
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
from benchflow.trajectories.types import Trajectory as ProviderTrajectory
from benchflow.usage_tracking import (
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)


def _write_path_lock_record(rollout_dir: Path, record: dict[str, Any]) -> None:
    """Write what the PATH lock did into the trial folder (when it ran)."""
    if not record:
        return
    try:
        (rollout_dir / path_lock_record.LOCKED_SYSTEM_PATH_RECORD).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    except OSError:
        logger.debug("locked-system-path record not written", exc_info=True)


ERROR_TRACEBACK_FILE = "error-traceback.txt"


def write_error_traceback(rollout_dir: Path | None, exc: BaseException) -> Path | None:
    """Keep an unexpected error's traceback with its trial, at every log level.

    The default level prints one line (DF-p5-cli-craft-7); a bug report needs
    the traceback, which the trial folder keeps, redacted like the crash log,
    so the user does not have to run the agent again under ``bench -v``
    (review L-A3). None when there is no trial folder or it cannot be
    written.
    """
    if rollout_dir is None:
        return None
    from benchflow.cli._errors import _secrets
    from benchflow.doctor import redacted_traceback

    path = Path(rollout_dir) / ERROR_TRACEBACK_FILE
    try:
        path.write_text(redacted_traceback(exc, _secrets()), encoding="utf-8")
    except OSError:
        return None
    return path


DAEMON_GUARD_LOG_FILE = "daemon-guard.log"


def _write_guard_progress(rollout_dir: Path | None, exc: BaseException) -> None:
    """Keep what a failed daemon guard logged (its steps, the package
    manager's output) with its trial, as ``daemon-guard.log`` (DFR-p3-4)."""
    progress = getattr(exc, "progress", None)
    if rollout_dir is None or not isinstance(progress, str) or not progress:
        return
    from benchflow._diagnostics import redact
    from benchflow.cli._errors import _secrets

    try:
        (Path(rollout_dir) / DAEMON_GUARD_LOG_FILE).write_text(
            redact(progress, _secrets()), encoding="utf-8"
        )
    except OSError:
        logger.debug("daemon guard log not written", exc_info=True)


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

# The separate verifier's wait for the agent sandbox's stop before the
# transfer. Only a wedged stop reaches it (Docker's own steps are bounded at
# 30 + 120 s, then a forced kill); cleanup() then stops the sandbox again.
_AGENT_STOP_TIMEOUT_SEC = 300


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
    # The task's own command (C8's task route).
    with as_task_command(env):
        result = await exec_raw(
            env.exec,
            command_config.command,
            cwd=command_config.cwd,
            env=env_vars,
            timeout_sec=round(command_config.timeout_sec),
            user=command_config.user,
            service=command_config.service,
        )
    if getattr(result, "return_code", 0) != 0:
        stdout = (diagnostic_result(result).stdout.render() or "").strip()
        stderr = (diagnostic_result(result).stderr.render() or "").strip()
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


async def _probe_integrity_access(rollout: Any, phase: str) -> None:
    """The integrity access probe: ``pre`` at the end of install_agent (after
    lockdown, before any agent turn), ``post`` at verify.

    getattr: callers drive verify() on partially built stand-ins too.
    """
    cfg = getattr(rollout, "_config", None)
    if (
        getattr(cfg, "integrity", "off") == "off"
        or getattr(cfg, "purpose", None) != "task"
    ):
        return
    if phase == "pre" and "pre" in (getattr(rollout, "_integrity_access", None) or {}):
        return  # once, before the first turn
    from benchflow.integrity.trial import record_access_probe

    await record_access_probe(rollout, phase)


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
        with as_task_command(env):
            result = await exec_raw(
                env.exec,
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
                stdout = (diagnostic_result(result).stdout.render() or "").strip()
                stderr = (diagnostic_result(result).stderr.render() or "").strip()
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


def _log_task(rollout: Any) -> str:
    """The task a rollout's log lines name (empty for a Rollout built
    without its config, as some callers and tests make one)."""
    path = getattr(getattr(rollout, "_config", None), "task_path", None)
    return path.name if isinstance(path, Path) else ""


@contextlib.contextmanager
def _keeping_trial_warnings(rollout: Any):
    """Keep the trial warnings a verifier run reports (in the agent's sandbox,
    a separate one, or a recovery) for the rollout's result.json."""
    with recording_trial_warnings() as warnings:
        try:
            yield
        finally:
            # vars(): a Rollout built without __init__ (tests) has none.
            kept = vars(rollout).setdefault("_warnings", [])
            kept.extend(w for w in warnings if w not in kept)


def _with_harness_cpus(method):
    """Run *method* inside the sandbox's ``harness_cpus`` window, when the
    sandbox's class has one (PERF-SBX); the body is the method's own."""

    @functools.wraps(method)
    async def run(self, *args, **kwargs):
        from benchflow.sandbox._hooks import sandbox_hook

        cpus = sandbox_hook(self._env, "harness_cpus")
        async with cpus() if cpus is not None else contextlib.nullcontext():
            return await method(self, *args, **kwargs)

    return run


class Rollout:
    """Decomposed trial lifecycle with independently-callable phases."""

    # A class default besides the __init__ assignment: a Rollout built without
    # __init__ (Rollout.__new__ in tests, an SDK subclass that skips
    # super().__init__) has no Muse login staged, and verify(), soft_verify()
    # and disconnect() read this flag first.
    _muse_credentials_staged: bool = False
    # Set by setup(verifier_only=True) (verifier recovery, regrade; K1-R14-2):
    # the rollout restores and grades, and runs no agent code (its "oracle"
    # is a stand-in), so it asks for no kept-service freezer and runs no probe.
    _verifier_only: bool = False
    # PERF-B's queued sandbox stop of this trial, when it was queued: a
    # discarded warm-pool trial removes its folder only after it.
    _queued_stop: asyncio.Task[None] | None = None

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
        self._prebuilt_rollout_paths: Any = None
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
        # The cacheable part of the primary agent's install, when start()
        # asked the sandbox to start from its cached layer (env_layers), and
        # the trial's ``layers`` record for config.json.
        self._agent_layer_plan: Any = None
        self._layer_records: dict[str, dict[str, Any]] = {}
        # Separate verifier mode: run()'s own verify() may stop the agent's
        # sandbox before the transfer (_verify_separate_overlap). Then
        # cleanup() skips the stop and every teardown step that already ran.
        self._release_agent_before_verify: bool = False
        self._agent_sandbox_stopped: bool = False
        self._teardown_done: set[str] = set()
        self._environment: Environment | None = None
        self._timeout: int = 0
        # Seconds per phase and span (timing_spans). A trial started by a job
        # begins with the seconds it waited in its concurrency slot.
        self._timing: dict[str, Any] = timing_spans.take_pending_trial_timing()
        # The job's warm pool (benchflow._warm_pool) may run setup(), start()
        # and the install ahead of the trial's turn (begin_prestart); the
        # lifecycle then waits for that work instead of doing it, and replays
        # its exception at the same point. _on_sandbox_ready tells the pool
        # the trial's sandbox is ready, so it prepares the next one.
        self._prestart: asyncio.Task[None] | None = None
        self._prestart_ready_at: float | None = None
        self._on_sandbox_ready: Callable[[], None] | None = None
        self._trial_clock_start: float | None = None
        # Separate verifier mode on Docker: the verifier sandbox started as the
        # agent's turn began (_begin_verifier_warmup), until verify() takes it
        # or cleanup() deletes it.
        self._verifier_warmup: Any = None
        self._effective_locked: list[str] = []
        self._disallow_web_tools: bool = False
        # What the run does about the cloud metadata endpoints (setup() sets
        # it; None: no agent, or a network that is not open).
        self._cloud_metadata: CloudMetadataPlan | None = None
        # What result.json's ``cloud_metadata`` says: the block's state and
        # the run's opt-out, once the block step ran.
        self._cloud_metadata_result: dict[str, Any] | None = None
        # What result.json's ``warnings`` lists (trial_warnings), never a value.
        self._warnings: list[str] = []
        self._egress_denylist: EgressDenylist | None = None
        self._active_egress_policy: EgressDenylist | None = None
        self._disallow_hosted_search: bool = False
        self._effective_skills_dir: Path | None = None
        self._skills_agent_derived = False
        self._skills_baked = False
        self._effective_skills_sandbox_dir: str | None = None
        self._required_skill_names: tuple[str, ...] = ()
        # Task dir actually deployed: a temp copy (self._task_tmp) when
        # Dockerfile mutations are needed, otherwise config.task_path.
        # cleanup() removes the temp copy.
        self._effective_task_path: Path = config.task_path
        self._task_tmp: Path | None = None
        self._task_skill_policy: TaskSkillPolicy | None = None
        # Where the verifier runs and why (resolve_verifier_isolation), set in
        # setup() once the task is loaded; result.json records both.
        self._verifier_isolation: tuple[str, str] | None = None
        self._retired_usage_runtimes: list[Any] = []
        self._usage_runtime: Any = None
        # The warm host proxy asked for when the sandbox starts, for the
        # first connection to take (_reserve_host_proxy).
        self._host_proxy_reservation: Any = None
        self._usage_metrics: dict[str, Any] = self._planes.extract_usage(None)
        self._native_usage_metrics: dict[str, Any] = _zero_native_acp_usage_metrics()
        self._unproxied_native_usage_metrics: dict[str, Any] = (
            _zero_native_acp_usage_metrics()
        )
        self._native_usage_proxy_covered = False
        self._native_usage_agents: set[str] = set()
        self._has_unproxied_native_session = False
        self._mixed_usage_finalized = False
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
        # The models that answered (P6, G-28), recorded in config.json at
        # cleanup: from the provider's responses the proxy captured, else from
        # Claude Code's session log (``_session_log.price_from_session_log``).
        self._proxy_reported_models: list[str] = []
        self._session_reported_models: list[str] = []

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
        # True even after partial staging, so teardown cannot strand a login.
        self._muse_credentials_staged = False
        # The credential lifecycle (C23, C26-C28): every credential
        # destination a connection stages, recorded before it is written and
        # kept until its removal is confirmed; and the identity of the last
        # connection, kept past disconnect, so the next role's staging stops
        # the previous agent first.
        self._credential_ledger: list[str] = []
        self._connection_key: tuple[Any, ...] | None = None
        self._connection_needs_quiescence = False
        # Whether disconnect() ran K1's quiesce kill, and after it the agent
        # pkill, for the current connection; connect() and connect_as() clear
        # both.
        self._agent_quiesced = False
        self._agent_pkill_done = False
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
        # Set when the agent's wall-clock budget ran out (not an idle abort);
        # a task.md draft 2 package's reward.json records it as timed_out.
        self._agent_ran_out_of_time: bool = False
        # Detail-less agent-phase timeout, eligible for zero-activity
        # reclassification (#1071). One-shot: cleared once a verdict lands.
        self._bare_timeout: bool = False
        # The agent's wall-clock budget ran out with a timeout message. It
        # keeps its error for the API verdicts, but an agent that did nothing
        # before it is an integration failure. One-shot, like _bare_timeout.
        self._judge_timed_out_agent: bool = False
        # The agent's deadline ended while a model request had no response.
        self._provider_stall: bool = False
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
        self._skill_export_root: Any = None
        # A separate verifier's clean control: the paths capture reads,
        # measured before the agent runs, and whether workspace capture then
        # hit its limits (see rollout._separate_verifier).
        self._pristine_outputs: dict[str, Any] | None = None
        self._capture_over_limit = False
        # What the quiesce before a workspace capture killed (K1); verify()
        # hands it to the hardening kill, which adds to it.
        self._kill_record: dict[str, Any] | None = None
        # Separate verifier only: the sandbox clock as the agent phase began,
        # and the files written outside everything that crosses since then.
        self._agent_phase_started: float | None = None
        self._outside_workspace_writes: dict | None = None
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
        # Docker binds these folders when the sandbox is constructed. They
        # cannot be moved after start(); use the very same paths for scoring
        # and all host artifacts, including caller-started environments.
        from benchflow.task import RolloutPaths

        paths = getattr(inner, "rollout_paths", None)
        self._prebuilt_rollout_paths = (
            paths if isinstance(paths, RolloutPaths) else None
        )

    @property
    def env(self) -> Any:
        """The rollout's sandbox, once :meth:`setup` created it."""
        return self._env

    @property
    def acp_client(self) -> Any:
        """The agent's ACP client while connected, else None."""
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
            # A slow image build inside setup reads as its own stage
            # (DF-p1-first-run-6). Compared with literals: mocked sandboxes
            # manufacture attributes.
            build = getattr(getattr(self, "_env", None), "image_build", "")
            if self._phase in ("created", "setup") and build in ("building", "built"):
                return ActivitySnapshot(f"image-{build}", None)
            return ActivitySnapshot(self._phase, None)
        calls, last_title = session.progress_snapshot()
        usage = session.latest_usage_totals()
        acp_tokens = usage.get("total_tokens") if usage else None
        gateway_tokens = _gateway_live_tokens(self._usage_runtime)
        retired_tokens = sum(
            trajectory.total_provider_tokens
            for runtime in getattr(self, "_retired_usage_runtimes", ())
            if (
                trajectory := getattr(
                    getattr(runtime, "server", None), "trajectory", None
                )
            )
            is not None
        )
        if retired_tokens:
            gateway_tokens = (gateway_tokens or 0) + retired_tokens
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
        """The ACP events captured so far."""
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
    def timing(self) -> dict[str, Any]:
        """Seconds per phase and span so far (see ``timing_spans``)."""
        return self._timing

    @property
    def result(self) -> RolloutResult | None:
        """The final result once the rollout is verified or finished, else None."""
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

    async def setup(self, *, verifier_only: bool = False) -> None:
        """Resolve config, create environment object (not yet started).

        Verifier-only children use oracle's sandbox provisioning, but never
        execute its solution and do not need the solution's credentials.
        They run no agent code either, so they ask for no kept-service
        freezer and run no probe (K1-R14-2): every caller (verifier recovery,
        ``bench regrade``) gets that from this flag alone.
        """
        if verifier_only:
            self._verifier_only = True
        require_safe_branch_world(self)
        cfg = self._config

        # Check every selected connection before credentials, proxy resolution
        # or sandbox creation. Use execution's compiler so later scenes, role
        # names and repeated turns cannot bypass admission through the primary
        # agent's session-factory exemption. Scripted runs bypass scene turns.
        from benchflow.task.plan_validation import validate_agent_transport

        validate_agent_transport(
            sandbox=cfg.environment, agent=cfg.primary_agent, harness=cfg.harness
        )
        if not is_scripted_agent(cfg.primary_agent):
            for step in compile_scenes_to_steps(cfg.effective_scenes):
                validate_agent_transport(
                    sandbox=cfg.environment,
                    agent=scene_step_role(step).agent,
                    harness=cfg.harness,
                )

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — it has no effect: only "
                "a User reads the solution."
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
        ) = _init_rollout(
            cfg.task_path,
            cfg.job_name,
            cfg.rollout_name,
            cfg.jobs_dir,
            **(
                {"rollout_paths": self._prebuilt_rollout_paths}
                if self._prebuilt_rollout_paths is not None
                else {}
            ),
        )
        # TASKMD-R14 B-M2: the skill export is a user-selected output root,
        # held before any agent runs.
        if cfg.export_generated_skills_to is not None:
            self._skill_export_root = skill_export_root(
                Path(cfg.export_generated_skills_to)
            )

        # C-axis overlay: deep-merge cfg.config_override into the task's resolved
        # config here at the rollout layer (not in the Task constructor), so only
        # this run's tasks are patched and every downstream read sees it. No-op
        # when None.
        if cfg.config_override:
            from benchflow._utils.config_override import apply_config_override

            self._task.config = apply_config_override(
                self._task.config, cfg.config_override
            )

        from benchflow.integrity.trial import resolve_verifier_isolation
        from benchflow.task.verifier_sandbox import effective_verifier_sandbox

        # One setting before resolution, the plan check and config.json.
        cfg.verifier_sandbox = effective_verifier_sandbox(
            cfg.verifier_sandbox, cfg.separate_verifier
        )
        self._verifier_isolation = resolve_verifier_isolation(
            self._task.config, cfg.verifier_sandbox, cfg.integrity
        )
        if self._verifier_isolation[1] != "task":
            # --verifier-sandbox separate, or strict integrity (which implies
            # it), runs the verifier in the separate verifier sandbox. A job
            # refuses unsupported tasks at plan time (check_separate_plan);
            # this is the backstop for SDK callers: refused here, before any
            # sandbox exists, where that sandbox cannot run this task.
            from benchflow.integrity.trial import (
                force_separate_verifier,
                separate_launch_issues,
            )

            issues = separate_launch_issues(
                self._task.config,
                sandbox=cfg.environment,
                task_dir=cfg.task_path,
                environment_manifest=cfg.environment_manifest,
            )
            if issues:
                setting = (
                    "integrity=strict"
                    if self._verifier_isolation[1] == "integrity-strict"
                    else "separate_verifier=True"
                    if cfg.separate_verifier
                    else "verifier_sandbox=separate"
                )
                raise ValueError(
                    f"{setting} runs the verifier in a separate verifier "
                    "sandbox, which this task cannot use here:\n- "
                    + "\n- ".join(issues)
                )
            self._task.config = force_separate_verifier(self._task.config)

        if cfg.separate_verifier_include and not separate_verifier_requested(
            self._task.config
        ):
            logger.warning(
                "--separate-verifier-include has no effect for %s: its verifier "
                "runs in the agent's sandbox",
                self._task.name,
            )

        if not verifier_only:
            # After the verifier's sandbox is resolved: a separate verifier
            # keeps nothing, so is never refused.
            from benchflow.task.keep_services import preflight_daytona

            preflight_daytona(
                self._task,
                cfg.task_path,
                sandbox=cfg.environment,
                agent=cfg.primary_agent,
                sandbox_user=cfg.sandbox_user,
                base_image=cfg.base_image_override,
            )

        prepare_terminal_review(self)
        if cfg.task_digest is None:
            from benchflow._utils.task_authoring import task_digest

            cfg.task_digest = task_digest(cfg.task_path)

        # Scripted runners (oracle, nop) and the task runtime launch no agent:
        # no launch-command lookup, and no agent-only web or egress policy.
        has_agent_launch = not (
            is_scripted_agent(cfg.primary_agent) or cfg.primary_agent == "task-runtime"
        )
        self._disallow_web_tools = has_agent_launch and (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        )
        from benchflow.task.plan_validation import agent_egress_policy

        self._egress_denylist = agent_egress_policy(
            self._task,
            agent=cfg.primary_agent,
            sandbox_user=cfg.sandbox_user,
            disallow_web_tools=self._disallow_web_tools,
        )
        if (
            has_agent_launch
            and not cfg.sandbox_user
            and _task_agent_offline(self._task)
        ):
            # Only the agent user's uid firewall holds an agent offline in an
            # online sandbox; an agent running as root would keep the network.
            raise ValueError(
                "[agent] network_mode='no-network' requires a sandbox_user "
                "(--sandbox-user agent): only the agent user can be kept offline "
                "in an online sandbox"
            )
        self._disallow_hosted_search = self._egress_denylist is not None
        # The compose file the sandbox factory stacks (setup.py picks the
        # same folder): main's network_mode and the other services.
        task_environment_dir = cfg.task_path / "environment"
        if not task_environment_dir.exists():
            task_environment_dir = getattr(
                getattr(self._task, "paths", None), "environment_dir", None
            )
        self._cloud_metadata = cloud_metadata_plan(
            getattr(self._task, "config", None),
            sandbox_user=cfg.sandbox_user,
            launches_agent=has_agent_launch,
            environment_dir=task_environment_dir,
            allow_cloud_metadata=bool(getattr(cfg, "allow_cloud_metadata", False)),
        )
        self._agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                cfg.primary_agent, cfg.primary_model, cfg.agent_env
            ),
            disallow=self._disallow_web_tools,
        )
        # The same sandbox-free plan gate as `bench tasks check`, using the
        # resolved auth route (native logins do not use proxy extras). Its
        # proxy install record goes in config.json. A resolve can take
        # minutes, so the gate runs off the event loop, before any sandbox.
        from benchflow.task.plan_validation import validate_task_plan

        self._litellm_proxy_record = await asyncio.to_thread(
            validate_task_plan,
            cfg.task_path,
            sandbox=cfg.environment,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            harness=cfg.harness,
            sandbox_user=cfg.sandbox_user,
            agent_env=self._agent_env,
            task=self._task,
            custom_backend=cfg.planes is not None or self._env_externally_owned,
            sandbox_instance=self._env,
            force_sandbox_local=self._disallow_web_tools
            or self._egress_denylist is not None,
            disallow_web_tools=self._disallow_web_tools,
            execute_solution=cfg.primary_agent == "oracle" and not verifier_only,
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
        if cfg.prompt_suffix:
            self._resolved_prompts = [
                f"{prompt}\n\n{cfg.prompt_suffix}" for prompt in self._resolved_prompts
            ]
        if has_agent_launch:
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
            from benchflow.task.compose_env import host_access_allowed
            from benchflow.task.env import TaskEnvNotAllowedError
            from benchflow.task.staging import StagingRefused, copy_tree

            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            # Links are copied as links, never read through: a link to an
            # operator file must not become a task file.
            try:
                copy_tree(
                    cfg.task_path,
                    tmp / cfg.task_path.name,
                    label=cfg.task_path.name,
                    allow_outside=host_access_allowed(),
                )
            except StagingRefused as exc:
                shutil.rmtree(tmp, ignore_errors=True)
                raise TaskEnvNotAllowedError(
                    {cfg.task_path.name: exc.refused}
                ) from None
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
        # A learner store's skills were written by earlier agents; the task's
        # bundled skills are the task's own.
        self._skills_agent_derived = (
            cfg.skills_agent_derived
            and effective_skills_dir is not None
            and not task_skill_policy.host_dir_is_bundled
        )
        self._skills_baked = False
        if effective_skills_dir is not None and not _environment_uses_prebuilt_image(
            env_config, cfg.environment_manifest
        ):
            self._skills_baked = True
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
                block_cloud_metadata=self._cloud_metadata is not None
                and self._cloud_metadata.armed,
                # Named only when set, so planes that predate it still take
                # every other call.
                **(
                    {"cloud_metadata_root_agent": True}
                    if self._cloud_metadata is not None
                    and self._cloud_metadata.armed
                    and getattr(self._cloud_metadata, "root_agent", False)
                    else {}
                ),
            )
            # A custom factory can use a built-in backend name without its
            # container barrier. Check the returned instance before start
            # (K1-R15-2 B1); the earlier named gate cannot establish this.
            from benchflow.task import identity_switch

            if identity_switch.declared(self._task.config):
                identity_switch.require_barrier(self._env, backend=cfg.environment)
            # What runs agent code here, for the sandbox's start to decide the
            # kept-service freezer; none for a verifier-only
            # rollout (recovery, K1-R14-2).
            from benchflow.task import keep_services

            keep_services.assign_runner_intent(
                self._env,
                self._task,
                cfg.primary_agent,
                cfg.sandbox_user,
                verifier_only=self._verifier_only,
            )
            # Daytona builds the proxy and agent installs into the image
            # recipe as cached layers (sandbox.daytona_layers). Looked up on
            # the type so a mocked sandbox never grows the hook.
            self._configure_daytona_layers()
        # Caller-supplied wall-clock budget (e.g. RuntimeConfig.timeout)
        # wins over the task's own default. Without this override there is
        # no way to tighten/loosen the agent budget per run — see #378.
        # A task with no agent timeout of its own (and a caller with none, or
        # 0) runs bounded by the rollout's whole deadline.
        if cfg.timeout:
            self._timeout = int(cfg.timeout)
        else:
            from benchflow.rollout._deadline import uncapped_agent_budget_sec
            from benchflow.task.config import effective_agent_timeout_sec

            task_timeout = effective_agent_timeout_sec(self._task.config.agent)
            self._timeout = (
                int(task_timeout) if task_timeout else uncapped_agent_budget_sec(cfg)
            )
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
            verifier_sandbox_mode=(
                "separate"
                if separate_verifier_requested(getattr(self._task, "config", None))
                else "shared"
            ),
            verifier_include=(
                cfg.separate_verifier_include
                if separate_verifier_requested(getattr(self._task, "config", None))
                else ()
            ),
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
            integrity=cfg.integrity,
            verifier_sandbox=cfg.verifier_sandbox,
            harness=cfg.harness,
            litellm_proxy=self._litellm_proxy_record,
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
            self._record_layers()
            from benchflow.rollout._verifier_recovery import (
                capture_original_docker_baseline,
            )

            await capture_original_docker_baseline(self)

        # Lifecycle phases are public: hooks or registry entries can change
        # after setup() planned the image, but before start() builds it.
        self._configure_daytona_layers()
        self._request_layers()
        self._reserve_host_proxy()
        await _start_env_and_upload(
            self._env,
            self._config.task_path,
            self._timing,
            skip_start=self._env_externally_owned,
            on_started=_capture_and_persist_sandbox,
            uploads=self._config.uploads,
            # A sandbox created from a branch snapshot already holds them.
            upload_task_files=not self._from_branch_snapshot,
            # The solution only where the oracle runs (GAP-ORACLE-READABLE).
            upload_solution=self._config.primary_agent == "oracle",
            # A learner store's skills baked into the image (M3).
            image_holds_agent_bytes=self._skills_agent_derived and self._skills_baked,
        )
        # Beside the sandbox's own setup, where this process mostly waits on
        # the sandbox; not beside its start, whose host work it would slow.
        self._prewarm_acp()

        # An SDK caller's hooks run the task's commands (C8's task route).
        for hook in self._config.pre_agent_hooks or []:
            with as_task_command(self._env):
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

    def _prewarm_acp(self) -> None:
        """Import the ACP SDK after sandbox start, overlapping agent setup.

        Only for agents that connect over ACP (acp.types.prewarm_sdk).
        """
        agent = self._config.primary_agent
        if (
            self._config.harness == HARNESS_NATIVE
            or not agent
            or is_scripted_agent(agent)
        ):
            return
        from benchflow.acp.types import prewarm_sdk

        prewarm_sdk()

    def _reserve_host_proxy(self) -> None:
        """Ask the job's warm host proxies for the one this trial's agent
        will connect through, so its start overlaps the sandbox's instead of
        following it (providers.litellm_runtime._HostProxyPool). Only for the
        first connection of an agent that gets a host proxy; connect() takes
        it, and a trial that never connects gives it back at clean-up. Only on
        a sandbox whose type says its agents reach a host proxy (Docker),
        looked up on the type so a mocked sandbox never starts one."""
        reserve = getattr(self._planes, "reserve_host_proxy", None)
        cfg = self._config
        agent = cfg.primary_agent
        if (
            reserve is None
            or not getattr(type(self._env), "starts_host_proxy_ahead", False)
            or self._host_proxy_reservation is not None
            or self._usage_runtime is not None
            or not agent
            or is_scripted_agent(agent)
        ):
            return
        self._host_proxy_reservation = reserve(
            agent=agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            environment=cfg.environment,
            required_skill_names=self._required_skill_names,
            force_sandbox_local=self._disallow_web_tools
            or self._egress_denylist is not None,
        )

    def _release_host_proxy(self) -> None:
        reservation, self._host_proxy_reservation = self._host_proxy_reservation, None
        release = getattr(self._planes, "release_host_proxy", None)
        if reservation is not None and release is not None:
            release(reservation)

    def _request_layers(self) -> None:
        """Ask the sandbox for the cached layers this trial can start from.

        The model proxy's (when it runs in the sandbox) and the agent's. On a
        backend that builds no layers (Modal, Apple Container, AgentCore)
        both install in the trial and the record says ``unsupported``.
        """
        from benchflow.sandbox.env_layers import AGENT, PROXY, unsupported_records

        self._request_verifier_layer()
        self._request_proxy_layer()
        self._request_agent_layer()
        cfg = self._config
        agent = cfg.primary_agent
        asked = []
        if self._proxy_in_sandbox():
            asked.append(PROXY)
        if agent and not is_scripted_agent(agent) and not cfg.skip_agent_install:
            asked.append(AGENT)
        if self._env_externally_owned or self._from_branch_snapshot:
            return
        self._layer_records.update(unsupported_records(cfg.environment, asked))

    def _request_verifier_layer(self) -> None:
        """Record why shared grading uses its ordinary image and tools.

        Separate verifiers request image reuse in ``_separate_verifier``.
        Caller-owned sandboxes and restored snapshots are never replanned.
        """
        from benchflow.sandbox.env_layers import VERIFIER, fresh_install_requested
        from benchflow.sandbox.verifier_layers import plan_verifier_layer

        self._layer_records.pop(VERIFIER, None)
        use_layer = getattr(type(self._env), "use_verifier_layer", None)
        if not callable(use_layer):
            return
        task_config = getattr(self._task, "config", None)
        if (
            self._env_externally_owned
            or self._from_branch_snapshot
            or task_config is None
            or separate_verifier_requested(task_config)
        ):
            use_layer(self._env, None)
            return
        sandbox_config = getattr(self._env, "task_env_config", None)
        recipe, record = plan_verifier_layer(
            self._task,
            separate=False,
            fresh_install=fresh_install_requested(
                getattr(self._config, "fresh_install", None)
            ),
            allow_internet=bool(getattr(sandbox_config, "allow_internet", True)),
            sandbox_env=getattr(sandbox_config, "env", None),
        )
        if record is not None:
            self._layer_records[VERIFIER] = record
        use_layer(self._env, recipe)

    def _proxy_in_sandbox(self) -> bool:
        """Whether the primary agent's model proxy runs (and installs) in the sandbox."""
        record = getattr(self, "_litellm_proxy_record", None)
        return isinstance(record, dict) and record.get("location") == "sandbox"

    def _request_proxy_layer(self) -> None:
        """Ask the sandbox to start from the model proxy's cached layer.

        Only when the proxy will run in the sandbox (its install is then the
        trial's longest setup step, about 104 packages) and the sandbox
        builds layers (``use_proxy_layer``: Docker and remote Docker). The
        layer holds the venv of the same locked set and extras the setup
        check recorded (``litellm_proxy``); the proxy's config, keys and logs
        stay per trial. A fresh install, a caller-owned sandbox and one from a
        branch snapshot start without it.
        """
        from benchflow.providers.litellm_runtime import proxy_layer_recipe
        from benchflow.providers.proxy_requirements import ProxyInstall
        from benchflow.sandbox.env_layers import (
            PROXY,
            fresh_install_requested,
            fresh_record,
        )

        self._layer_records.pop(PROXY, None)
        use_layer = getattr(type(self._env), "use_proxy_layer", None)
        if not callable(use_layer):
            return
        record = getattr(self, "_litellm_proxy_record", None)
        if (
            not self._proxy_in_sandbox()
            or not isinstance(record, dict)
            or self._env_externally_owned
            or self._from_branch_snapshot
        ):
            use_layer(self._env, None)
            return
        if fresh_install_requested(getattr(self._config, "fresh_install", None)):
            use_layer(self._env, None)
            self._layer_records[PROXY] = fresh_record(
                "fresh", "fresh install asked for"
            )
            return
        reason = self._proxy_layer_bypass_reason()
        if reason is not None:
            use_layer(self._env, None)
            self._layer_records[PROXY] = fresh_record("fresh", reason)
            return
        file_name = str(record.get("requirements") or "")
        install = ProxyInstall(
            group=file_name.removesuffix(".txt"),
            extra_requirements=tuple(
                str(extra) for extra in record.get("extra_requirements") or ()
            ),
        )
        if install.extra_requirements:
            # Extras resolve at install time (only the locked set is pinned
            # and hashed), so a layer built last week can hold another
            # version of an extra, or of its dependencies, than a fresh
            # install under the same key: no layer for them.
            use_layer(self._env, None)
            self._layer_records[PROXY] = fresh_record(
                "fresh",
                "proxy extras resolve at install time, so they are never cached",
            )
            return
        try:
            recipe = proxy_layer_recipe(install)
        except ValueError as e:
            use_layer(self._env, None)
            self._layer_records[PROXY] = fresh_record("fresh", f"no proxy layer: {e}")
            return
        use_layer(self._env, recipe)

    def _add_reported_models(self, runtime: Any) -> None:
        """Add the models a stopped proxy's responses named to the rollout's list.

        A Scene role on another model or route replaces the proxy (its
        callback log is imported into its trajectory when it stops), so each
        replaced runtime's models are added here before it is dropped, and the
        last one's at cleanup. Bounded and deduplicated by
        :func:`add_model`.
        """
        from benchflow._utils.reported_models import add_model

        reported = getattr(
            getattr(getattr(runtime, "server", None), "trajectory", None),
            "metadata",
            None,
        )
        if isinstance(reported, dict) and isinstance(
            reported.get("reported_models"), list
        ):
            for name in reported["reported_models"]:
                add_model(self._proxy_reported_models, name)

    def _record_proxy_layer_use(self) -> None:
        """After the proxy started: whether it ran from its layer (config.json)."""
        from benchflow.sandbox.env_layers import PROXY

        self._record_restricted_web_model()
        server = getattr(self._usage_runtime, "server", None)
        use = getattr(server, "layer_use", None)
        if not isinstance(use, dict):
            return
        self._layer_records.setdefault(PROXY, {}).update(use)
        self._record_layers()

    def _record_restricted_web_model(self) -> None:
        """How each web-restricted model was admitted, override included."""
        admission = getattr(self._usage_runtime, "restricted_web_model", None)
        if not isinstance(admission, dict) or self._rollout_dir is None:
            return
        path = self._rollout_dir / "config.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        records = data.get("restricted_web_model")
        records = (
            [r for r in records if r != admission] if isinstance(records, list) else []
        )
        data["restricted_web_model"] = [*records, admission]
        from benchflow.review.persistence import write_json_atomic

        write_json_atomic(path, data)

    def _proxy_layer_bypass_reason(self) -> str | None:
        """Keep Docker's private proxy recipe independent of agent setup.

        Docker and remote Docker build only the locked proxy recipe into a
        fixed venv; reuse checks that recipe's stamp and executables. Task
        provisioning and the caller's agent install plane do not select its
        packages. They still disable agent reuse, whose install writes into
        the task environment. Daytona retains its conservative shared policy.
        """
        from benchflow.sandbox.env_layers import fresh_install_requested

        if self._config.environment in {"docker", "remote-docker"}:
            if fresh_install_requested(self._config.fresh_install):
                return "fresh install asked for"
            return None
        return self._agent_layer_bypass_reason()

    def _agent_layer_bypass_reason(self) -> str | None:
        """A recipe stamp cannot authenticate bytes after arbitrary setup.

        Keep the ordinary install after caller/task mutation and preserve the
        caller's install plane. Recheck at use too: lifecycle phases are public
        and their configuration can change after layer selection.
        """
        from benchflow.sandbox.env_layers import fresh_install_requested

        cfg = self._config
        if fresh_install_requested(cfg.fresh_install):
            return "fresh install asked for"
        if cfg.planes is not None:
            return "caller-supplied install plane"
        if cfg.pre_agent_hooks or cfg.environment_manifest is not None or cfg.uploads:
            return "pre-install hooks, provisioning or uploads can change the install"
        task_config = getattr(self._task, "config", None)
        env_config = getattr(task_config, "sandbox", None)
        if getattr(env_config, "setup_commands", None) or getattr(
            env_config, "healthcheck", None
        ):
            return "task setup or healthcheck can change the install"
        if getattr(env_config, "env", None) or getattr(
            self._env, "_persistent_env", None
        ):
            return "runtime environment overrides require an in-trial install"
        return None

    def _configure_daytona_layers(self) -> None:
        """Plan only installs whose inputs and outputs stay valid until use."""
        if self._env_externally_owned or self._from_branch_snapshot:
            return
        configure_layers = getattr(type(self._env), "configure_install_layers", None)
        if not callable(configure_layers):
            return
        from benchflow.sandbox.daytona_layers import fresh_install_requested

        cfg = self._config
        configure_layers(
            self._env,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            agent_env=self._agent_env,
            fresh_install=fresh_install_requested(cfg.fresh_install)
            or self._agent_layer_bypass_reason() is not None,
            install_agent=not cfg.skip_agent_install,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            only_if_reused=True,
        )

    def _request_agent_layer(self) -> None:
        """Ask the sandbox to start from the agent's cached install layer.

        Only sandboxes that build layers have ``use_agent_layer`` (Docker and
        remote Docker); it is looked up on the type, so a mocked sandbox never
        grows it. The layer holds the cacheable part of the install (system
        packages, Node, the agent's pinned package); install_agent() still runs
        everything per trial. A caller-owned sandbox, one created from a branch
        snapshot, a scripted agent, ``skip_agent_install`` and a fresh install
        (``--fresh-install``, ``BENCHFLOW_FRESH_INSTALL=1``) start without one.
        """
        from benchflow.agents.install import agent_layer_plan
        from benchflow.sandbox.env_layers import (
            AGENT,
            fresh_install_requested,
            fresh_record,
        )

        self._agent_layer_plan = None
        self._layer_records.pop(AGENT, None)
        use_layer = getattr(type(self._env), "use_agent_layer", None)
        if not callable(use_layer):
            return
        if self._env_externally_owned or self._from_branch_snapshot:
            use_layer(self._env, None)
            return
        cfg = self._config
        agent = cfg.primary_agent
        if not agent or is_scripted_agent(agent) or cfg.skip_agent_install:
            use_layer(self._env, None)
            return
        if fresh_install_requested(cfg.fresh_install):
            use_layer(self._env, None)
            self._layer_records[AGENT] = fresh_record(
                "fresh", "fresh install asked for"
            )
            return
        reason = self._agent_layer_bypass_reason()
        if reason is not None:
            use_layer(self._env, None)
            self._layer_records[AGENT] = fresh_record("fresh", reason)
            return
        plan = agent_layer_plan(
            agent,
            harness=cfg.harness,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
        )
        if plan is None:
            use_layer(self._env, None)
            self._layer_records[AGENT] = fresh_record(
                "fresh", f"{agent.split()[0]} has no pinned install to cache"
            )
            return
        self._agent_layer_plan = plan
        use_layer(self._env, plan.recipe)

    def _record_layers(self) -> None:
        """Write the trial's ``layers`` record into its config.json.

        config.json is written before the sandbox exists; the layer's key and
        whether it was a hit, a miss, fresh or unsupported are known after
        start() and after install_agent().
        """
        records = {k: dict(v) for k, v in self._layer_records.items()}
        sandbox_records = getattr(self._env, "layer_records", None)
        if isinstance(sandbox_records, dict):
            for kind, record in sandbox_records.items():
                if isinstance(record, dict):
                    records[kind] = {**record, **records.get(kind, {})}
        # How the sandbox started its containers: the direct Docker path, or
        # Compose and why (PERF-E).
        direct = getattr(self._env, "direct_docker_record", None)
        if not isinstance(direct, dict):
            direct = None
        if not (records or direct) or self._rollout_dir is None:
            return
        path = self._rollout_dir / "config.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        if records:
            data["layers"] = records
        if direct:
            data["direct_docker"] = dict(direct)
        from benchflow.review.persistence import write_json_atomic

        write_json_atomic(path, data)

    async def _install_from_agent_layer(self, rollout_dir: Path) -> bool:
        """True when the sandbox started from the agent's layer and holds it.

        The install itself then does not run in the trial; the record says
        whether it was used, or why not (the trial then installs as before).
        """
        from benchflow.agents.install import agent_layer_plan, install_from_layer
        from benchflow.sandbox.env_layers import AGENT, LayerResult

        plan = self._agent_layer_plan
        if plan is None:
            return False
        layer = getattr(self._env, "agent_layer", None)
        if not isinstance(layer, LayerResult):
            return False
        why_not = self._agent_layer_bypass_reason()
        if why_not is None:
            cfg = self._config
            current_plan = agent_layer_plan(
                cfg.primary_agent,
                harness=cfg.harness,
                sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            )
            if current_plan != plan:
                why_not = "the effective agent or native install recipe changed after layer selection"
        if why_not is None:
            why_not = await install_from_layer(
                self._env, plan, rollout_dir, layer_key=layer.key
            )
        # Added to the sandbox's own record of the layer (_record_layers).
        use = self._layer_records.setdefault(AGENT, {})
        use["install"] = "from layer" if why_not is None else "in trial"
        if why_not is not None:
            logger.warning("Installing %s in the trial: %s", plan.agent, why_not)
            use["reason"] = why_not
        self._record_layers()
        return why_not is None

    # Phase 3: INSTALL AGENT

    def _needs_kept_freezer(self) -> bool:
        """Whether this trial can leave a process the kills keep frozen,
        from the identities :func:`keep_services.prepare_identities` measured.

        Only when the task keeps services and agent code runs as someone
        other than root beside a root verifier (K1-R10-1, DT2-R12-1): nop
        runs nothing, a verifier-only rollout runs no agent code, and the
        kills never keep a root process (a root oracle's service outlives
        them unfrozen, as before). The oracle is judged by the account its
        own su selects, the agent by ``--sandbox-user``.
        """
        from benchflow.task import keep_services

        cfg = self._config
        if self._verifier_only or cfg.primary_agent == "nop":
            return False
        if cfg.primary_agent == "oracle":
            return keep_services.kept_by_identity(self._task) is not None
        return keep_services.trial_keep(self._task, cfg.sandbox_user) is not None

    async def _prove_kept_freezer(self) -> None:
        """Refuse or probe a trial that keeps services, right after its
        identities are measured and before any agent code runs (K1-R9-1).

        A trial whose agent code can leave a process the kills keep needs a
        cgroup root can freeze it in; proving it here, rather than failing
        the kill after the agent's turn. Whether it keeps anything is known
        only now, so a sandbox that cannot give the freezer is
        refused here, not before it starts: a non-root agent on a sandbox
        with no freezer (``keep_services.provider_refusal``) or on an Engine
        that cannot give it (the start's reason); everything else, the
        oracle included, meets the freezer probe.
        """
        if not self._needs_kept_freezer():
            return
        cfg = self._config
        if cfg.primary_agent != "oracle":
            from benchflow.sandbox._compose import engine_freezer_unavailable
            from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
            from benchflow.sandbox.providers import is_known_provider
            from benchflow.task import keep_services

            if cfg.planes is None or is_known_provider(cfg.environment):
                refusal = keep_services.provider_refusal(
                    self._task,
                    getattr(self, "_effective_task_path", None) or cfg.task_path,
                    sandbox=cfg.environment,
                    keep=keep_services.trial_keep(self._task, cfg.sandbox_user),
                    agent=cfg.primary_agent,
                )
                if refusal is not None:
                    raise KeptServiceFreezerUnavailable(refusal)
            why = getattr(self._env, "kept_service_freezer_refusal", None)
            if isinstance(why, str) and why:
                raise engine_freezer_unavailable(why)
        with timing_spans.span(timing_spans.KEPT_FREEZER_PROBE, into=self._timing):
            await self._planes.probe_kept_service_freezer(self._env)

    @_with_harness_cpus
    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous scene-authored steps (different agents per role),
        each role's agent is installed on-demand in connect_as().
        This method installs the primary agent to set up the sandbox baseline.

        BenchFlow's steps here run with the sandbox's harness CPUs where it
        has them (``harness_cpus``); the task's limit is back before the
        agent's (or the solution's) first command.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()

        self._agent_cwd = await _resolve_agent_cwd(self._env, self._task)

        from benchflow.task import identity_switch

        # Before any agent code: the accounts the task lets the agent switch
        # to, which agent commands and the kills then follow (K1-R11-1).
        switch = identity_switch.for_task(self._task)
        if switch:
            # The plan gate refuses a named backend without the container's
            # no_new_privs; a plane outside the registry is asked here, by
            # the sandbox's own property (K1-R15-2).
            identity_switch.require_barrier(self._env)
            self._env.agent_identity_switch = switch

        await prepare_capture_runtime(self)

        present = {"agent": False, "baseline": False}
        # A restored checkpoint can already contain agent-written utilities.
        # Recapturing build files must not bless them as BASH_ENV inspectors.
        self._env._benchflow_bash_env_capture_allowed = not self._from_branch_snapshot
        self._env._benchflow_verifier_service = self._task.config.verifier.service
        self._env._benchflow_sandbox_user = cfg.sandbox_user
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

            async def seed_then_runtime() -> None:
                await self._planes.seed_verifier_workspace(
                    self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
                )
                # The verifier's plugins, from a root-only tree built while no
                # agent process has run yet. Its probe reads the
                # seeded copy, so it comes after the seed.
                await self._planes.prepare_verifier_runtime(
                    self._env, task=self._task, sandbox_user=cfg.sandbox_user
                )

            # The build-config snapshot (and its inspectors) and the seed with
            # the runtime write disjoint places and read nothing the other
            # writes, so on a sandbox that runs execs side by side they go at
            # once; the first failure in this order is raised (PERF-SBX).
            await in_order(
                self._env,
                lambda: self._planes.snapshot_build_config(
                    self._env, workspace=self._agent_cwd
                ),
                seed_then_runtime,
            )

        if is_scripted_agent(cfg.primary_agent):
            if cfg.sandbox_user:
                # Nothing runs as the sandbox user (the oracle runs as root),
                # so the sandbox-daemon guard has nobody to keep away.
                path_lock: dict[str, Any] = {}
                with (
                    path_lock_record.recording_into(path_lock),
                    timing_spans.recording_into(self._timing),
                ):
                    script_cwd = await self._planes.setup_sandbox_user(
                        self._env,
                        cfg.sandbox_user,
                        workspace=self._agent_cwd,
                        timeout_sec=cfg.sandbox_setup_timeout,
                        guard_daemon=is_operator_script(cfg.primary_agent),
                    )
                _write_path_lock_record(rollout_dir, path_lock)
                if is_operator_script(cfg.primary_agent):
                    self._agent_cwd = script_cwd
            from benchflow.task.keep_services import prepare_identities, run_identity

            # The oracle's processes run as its own account, not
            # --sandbox-user (DT2-R9-2); the same one _run_oracle resolves.
            await prepare_identities(
                self._env,
                self._task,
                run_identity(self._task, cfg.primary_agent, cfg.sandbox_user),
                inherit=cfg.primary_agent == "oracle",
            )
            # Before solve.sh or the operator script (K1-R9-1).
            await self._prove_kept_freezer()
            await verifier_baseline()
            await self._planes.deploy_skills(
                self._env,
                self._effective_task_path,
                self._effective_skills_dir,
                None,
                None,  # Scripted agents consume skills as root.
                self._agent_cwd,
                skills_sandbox_dir=self._effective_skills_sandbox_dir,
                agent_derived=self._skills_agent_derived,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await self._planes.lockdown_paths(self._env, self._effective_locked)
            await _probe_integrity_access(self, "pre")
            await mark_agent_phase_start(self)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        # Custom installers can repair loader custody before the seal.
        # Custom planes without this method also keep only the final check.
        if (
            cfg.sandbox_user
            and has_builtin_install(agent_name)
            and hasattr(type(self._planes), "check_loader_custody")
        ):
            # Before the install: an image whose loader files another user
            # owns, which the seal below would refuse, is refused now.
            await self._planes.check_loader_custody(
                self._env, self._agent_cwd, timeout_sec=cfg.sandbox_setup_timeout
            )
        if cfg.skip_agent_install:
            self._agent_cfg = None
        elif self._installed_agent_cfg is not None and present["agent"]:
            # An isolated branch child: its sandbox came from a snapshot taken
            # after the parent installed this same agent (binary checked above;
            # same process, same registry pin, so the same version).
            self._agent_cfg = self._installed_agent_cfg
        else:
            with timing_spans.span(timing_spans.INSTALL_AGENT, into=self._timing):
                # The cacheable part (system packages, Node, the pinned
                # package, a native harness's CLI) comes from the sandbox's
                # agent layer when it started from one; everything below
                # (user, credentials, skills, lockdown) runs every trial.
                if await self._install_from_agent_layer(rollout_dir):
                    self._agent_cfg = AGENTS.get(agent_name.split()[0])
                else:
                    self._agent_cfg = await install_agent_with_policy(
                        self._planes,
                        self._env,
                        agent_name,
                        rollout_dir,
                        sandbox_setup_timeout=cfg.sandbox_setup_timeout,
                        install_layer_bypass_reason=self._agent_layer_bypass_reason(),
                    )
                    await self._install_native_cli(agent_name, rollout_dir)
        if cfg.sandbox_user:
            path_lock: dict[str, Any] = {}
            # The daemon guard records its span here.
            with (
                path_lock_record.recording_into(path_lock),
                timing_spans.recording_into(self._timing),
            ):
                self._agent_cwd = await self._planes.setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            _write_path_lock_record(rollout_dir, path_lock)
            # After the install (it can leave them root-owned, #1183) and once
            # the user exists.
            await _prepare_agent_temp_dirs(
                self._env,
                self._agent_cfg,
                cfg.sandbox_user,
                locked_paths=self._effective_locked,
            )
        from benchflow.task.keep_services import prepare_identities

        await prepare_identities(self._env, self._task, cfg.sandbox_user)
        # Before the agent launches (K1-R9-1).
        await self._prove_kept_freezer()
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        # Credentials live only for a connection, not for the lifetime of the
        # installed sandbox: connect()/connect_as() stage them after proxy
        # wiring, from the values the credential table admits (C23, C24).
        await _install_native_task_mcp_config(
            self._env,
            self._task,
            agent_cfg=self._agent_cfg,
            cred_home=cred_home,
            owner=cfg.sandbox_user,
        )
        # The web-tool policy setup runs per connection, after that
        # connection's credentials are written (C30).
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
            harness=cfg.harness,
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
            agent_derived=self._skills_agent_derived,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await self._planes.lockdown_paths(self._env, self._effective_locked)
        # The integrity pre-turn probe, on every lifecycle that installs an
        # agent (the automatic one and a manually composed one alike), before
        # the agent phase's clock mark: files it writes are not the agent's.
        await _probe_integrity_access(self, "pre")
        await mark_agent_phase_start(self)

        self._phase = "installed"

    # Phase 3b: CONNECT (ACP session — re-entrant)

    def _session_factory_entrypoint(self, agent_name: str) -> str | None:
        """Return the ``session_factory`` "module:callable" if *agent_name* is a
        non-ACP session-factory agent, else None — the connect/drive dispatch key.

        A session-factory agent declares ``protocol="session-factory"`` + a
        ``session_factory`` entrypoint (e.g. omnigent's ``omnigent run`` CLI,
        which has no ACP server); everything else connects over ACP. Resolution
        failures degrade to ACP (None) rather than raising."""
        return session_factory_entrypoint(agent_name, resolver=resolve_agent)

    async def _install_native_cli(self, agent: str, rollout_dir: Path) -> None:
        """Add the native harness's CLI to an agent install that lacks it.

        Claude Code's install already brings the pinned CLI (the adapter runs
        it); Codex's native harness installs the pinned ``codex`` next to
        ``codex-acp``. A no-op for the ACP harness.
        """
        if self._config.harness != HARNESS_NATIVE or is_scripted_agent(agent):
            return
        harness = native_harness_for(agent)
        if not harness.install_cmd:
            return
        from benchflow.agents.install import effective_install_timeout

        # Root installs the CLI on the plumbing route, as every agent install
        # (no user: a root-only PATH, no PYTHON*; FF-install-C15).
        result = await exec_raw(
            self._env.exec,
            harness.install_cmd,
            timeout_sec=effective_install_timeout(
                agent, self._config.sandbox_setup_timeout
            ),
        )
        log = ReachablePath.rollout(rollout_dir, "agent") / "install-native-stdout.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        output = f"$ {harness.install_cmd}\n{diagnostic_result(result).stdout.render() or ''}{diagnostic_result(result).stderr.render() or ''}"
        log.write_text(output)
        if result.return_code != 0:
            raise AgentInstallError(
                agent=f"{agent} (native {harness.cli})",
                return_code=result.return_code,
                stdout=output,
                diagnostics="",
                log_path=str(log),
            )

    async def _connect_live_agent(self, **kwargs: Any) -> Any:
        """Connect the run's harness: the ACP adapter, or the agent's own CLI."""
        if self._config.harness == HARNESS_NATIVE:
            return await self._planes.connect_native(**kwargs)
        with timing_spans.span(timing_spans.ACP_CONNECT, into=self._timing):
            return await self._planes.connect_acp(**kwargs)

    def _check_cloud_metadata_protocol(self, session_factory: str | None) -> None:
        """Refuse an unprotected transport before any agent-side audit runs.

        Not for an agent that runs as root: the launcher's barrier is not
        what holds it. On Docker the block holds root only in a main whose
        root has neither NET_ADMIN nor NET_RAW, whatever starts the agent;
        on a sandbox provider the block is provider-managed.

        Also refuses a session-factory agent that runs as root under the
        no-web policy: its commands run as root without the restricted
        launch, so nothing can hold it to loopback."""
        if (
            session_factory is not None
            and self._disallow_web_tools
            and not self._config.sandbox_user
        ):
            raise FirewallPowerError(
                "sandbox startup refused: under the no-web policy an agent "
                "that runs as root is held only through the restricted "
                "ACP or native launch, and session-factory agents run "
                "commands outside it. Choose a non-root --sandbox-user, or "
                "an ACP or native agent."
            )
        plan = self._cloud_metadata
        if (
            plan is not None
            and plan.armed
            and not getattr(plan, "root_agent", False)
            and session_factory is not None
        ):
            raise CloudMetadataPowerError(
                "sandbox startup refused: cloud metadata protection requires "
                "the kernel's ACP or native privilege-drop launcher; "
                "session-factory agents can execute outside that barrier. "
                "Choose an ACP or native agent."
            )

    async def _check_firewall_power(
        self,
        rollout_dir: Path,
        *,
        agent: str,
        agent_env: dict[str, str],
        session_factory: str | None = None,
    ) -> None:
        """Before the agent starts, wherever a firewall rule of BenchFlow's
        will hold its user: disarm an image that would hand the agent that
        firewall (NET_ADMIN dropped from its commands) or refuse it, and
        record which in the trial folder's firewall-power.json (see
        benchflow.sandbox.firewall_power).

        Runs on every connection, before the cloud metadata block (whose
        own check then starts the way the agent does): a restored snapshot
        is a new container. A root agent, the oracle and nop are not checked.
        """
        user = self._config.sandbox_user
        if not user or is_scripted_agent(agent):
            forget_net_admin_drop(self._env)
            return
        holds = self._firewall_holds(agent_env)
        if not holds:
            forget_net_admin_drop(self._env)
            return
        with timing_spans.span(timing_spans.FIREWALL_POWER, into=self._timing):
            record, refusal = await check_firewall_power(
                self._env,
                user,
                holds=holds,
                can_drop=session_factory is None,
            )
        (rollout_dir / FIREWALL_POWER_RECORD).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        if refusal is not None:
            raise FirewallPowerError(refusal)

    def _firewall_holds(self, agent_env: Mapping[str, str]) -> tuple[str, ...]:
        """The firewall rules of BenchFlow's that hold the agent user under
        ``agent_env`` (what ``_check_firewall_power`` checks for)."""
        plan = self._cloud_metadata
        return firewall_rules_hold_agent(
            self._env,
            agent_env,
            cloud_metadata_armed=plan is not None and plan.armed,
        )

    def _guards_beside_proxy(
        self,
        *,
        force_sandbox_local: bool,
        egress_denylist: EgressDenylist | None,
        role_changes_sandbox: bool = False,
    ) -> bool:
        """Whether this connection starts its model proxy while the sandbox
        guards (the firewall-power check, then the cloud metadata block) run,
        instead of before them.

        A host proxy (``proxy_runs_on_host``) runs nothing in the sandbox, and
        its start is mostly the health wait for the new process (about 4 s on
        a build VM), so the guards' commands can fill it. Only where nothing
        else changes the sandbox between the two: the connection's first
        proxy (stopping an earlier one may run in the sandbox), no egress
        policy of its own or of an earlier connection (that policy's stop
        comes before the block's proof, and its env reaches the check), and
        no install or credentials for another role's agent (the
        firewall-power check must see what that install adds). Not with
        ``BENCHFLOW_HOST_PROXY_AHEAD=0``, which restores the start as it was.
        """
        from benchflow.providers.litellm_runtime import (
            host_proxy_ahead,
            proxy_runs_on_host,
        )

        return (
            host_proxy_ahead()
            and egress_denylist is None
            and not role_changes_sandbox
            and self._usage_runtime is None
            and self._active_egress_policy is None
            and proxy_runs_on_host(
                self._config.environment, force_sandbox_local=force_sandbox_local
            )
        )

    async def _start_proxy_beside_guards(
        self,
        start: Callable[[], Awaitable[tuple[dict[str, str], Any]]],
        guards: Callable[[], Awaitable[None]],
    ) -> tuple[dict[str, str], Any]:
        """Start the host proxy (``start``) and run ``guards`` meanwhile;
        return what the proxy start returns.

        The order's outcomes stay as they were with the proxy first: a guard
        refusal raises once the proxy start has ended, with a started proxy
        kept on the rollout for clean-up; the proxy's own failure raises once
        the guards are done. Cancelled, the proxy start is cancelled too (it
        stops a proxy that never became healthy).
        """
        proxy = asyncio.ensure_future(start())
        try:
            # Let the proxy's process start before the guards' first command.
            await asyncio.sleep(0)
            await guards()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                proxy.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                _, self._usage_runtime = await proxy
                self._record_proxy_layer_use()
            raise
        return await proxy

    async def _apply_cloud_metadata_block(
        self, rollout_dir: Path, *, session_factory: str | None = None
    ) -> None:
        """Refuse the cloud metadata endpoints to the agent user before the
        agent starts, under an open network (see lockdown.cloud_metadata_plan),
        and record what holds in the trial folder's cloud-metadata.json (and
        result.json's ``cloud_metadata``).

        Runs on every connection: a restored snapshot is a new container, so
        its rules are installed again (running it twice adds nothing). The
        agent starts only when the block is ``on`` (proven), a root agent's
        included, or when the run opted out; otherwise the trial stops here,
        before the agent, with the record's reason.
        """
        plan = self._cloud_metadata
        if plan is None:
            return
        # Only an explicit opt-out lets the agent start without the block.
        opt_out = str(getattr(plan, "opt_out", "") or "")
        if plan.armed:
            # S2-B18-CONF-1: the construction check proves the kernel's
            # launcher, not raw exec(user=agent) calls made by an in-process
            # session factory. Refuse that protocol before optional audits
            # can admit it, on initial connect and every role/reconnect.
            # Dispatch must use this same resolved entrypoint: resolving again
            # after the awaited audit could turn an ACP fallback into a factory.
            self._check_cloud_metadata_protocol(session_factory)
            with timing_spans.span(
                timing_spans.CLOUD_METADATA_BLOCK, into=self._timing
            ):
                # A root agent (no sandbox user) is held as root.
                record = await self._planes.block_cloud_metadata(
                    self._env, self._config.sandbox_user
                )
            if plan.uncovered:
                record = {
                    **record,
                    "why": record["why"] + uncovered_services_note(plan.uncovered),
                    "uncovered_services": list(plan.uncovered),
                }
        else:
            record = cloud_metadata_record(
                plan.state,
                f"{plan.why}, so the agent can reach the cloud metadata endpoints",
                **({"opt_out": opt_out} if opt_out else {}),
            )
        (rollout_dir / CLOUD_METADATA_RECORD).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        self._cloud_metadata_result = {
            "state": record["state"],
            **({"opt_out": opt_out} if opt_out else {}),
        }
        if record["state"] == NOT_REACHABLE and not opt_out:
            # The provider gives the sandbox no route to the endpoints, and
            # this trial's connects proved it: nothing to block.
            logger.info(
                "Cloud metadata: not reachable from this sandbox (proven); "
                "no block is needed"
            )
            return
        if record["state"] == PROVIDER_MANAGED and not opt_out:
            if getattr(plan, "require_block", False):
                raise cloud_metadata_refusal(
                    str(record["state"]),
                    f"{record['why']}; the task requires the block "
                    "(require_cloud_metadata_block = true)",
                )
            logger.warning(
                "Cloud metadata: the agent runs as root on a sandbox provider; "
                "the provider's own metadata policy applies (no BenchFlow block)"
            )
            return
        if record["state"] != "on" and not opt_out:
            raise cloud_metadata_refusal(str(record["state"]), str(record["why"]))

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
                self._env,
                self._config.sandbox_user,
                agent_launch or self._agent_launch,
                harness=self._config.harness,
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
        await self._end_previous_connection((cfg.primary_agent, cfg.primary_model, ()))
        # Probes and proxy setup can execute with the next login too. Mark
        # this attempt pending before either, including failed admissions.
        self._connection_needs_quiescence = True
        self._agent_pkill_done = self._agent_quiesced = False
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

        previous_runtime = self._usage_runtime
        force_sandbox_local = self._disallow_web_tools or egress_denylist is not None
        # A host proxy starts while the sandbox guards run (see
        # _guards_beside_proxy); they then check the env before its wiring.
        beside = self._guards_beside_proxy(
            force_sandbox_local=force_sandbox_local, egress_denylist=egress_denylist
        )
        guard_env = dict(self._agent_env)
        sf_entrypoint: str | None = None
        if beside:
            # Resolve once for admission, the firewall audit and dispatch. An
            # unsupported session factory must not run even an audit candidate.
            sf_entrypoint = self._session_factory_entrypoint(cfg.primary_agent)
            self._check_cloud_metadata_protocol(sf_entrypoint)

        async def guards() -> None:
            # Before the first command the agent user runs (the Codex Apps
            # probe), then the block before anything the agent user runs.
            await self._check_firewall_power(
                rollout_dir,
                agent=cfg.primary_agent,
                agent_env=guard_env,
                session_factory=sf_entrypoint,
            )
            await self._apply_cloud_metadata_block(
                rollout_dir, session_factory=sf_entrypoint
            )

        async def start_proxy() -> tuple[dict[str, str], Any]:
            # The proxy's install and health wait record their spans here. The
            # first connection may take the proxy reserved when the sandbox
            # started (_reserve_host_proxy).
            claim = getattr(self._planes, "host_proxy_claim", None)
            with (
                timing_spans.recording_into(self._timing),
                claim(self._host_proxy_reservation)
                if claim is not None
                else contextlib.nullcontext(),
            ):
                return await self._planes.ensure_litellm_runtime(
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
                    prior_runtimes=self._usage_runtimes(),
                    live_trajectory_path=rollout_dir
                    / "trajectory"
                    / "llm_trajectory.jsonl",
                    force_sandbox_local=force_sandbox_local,
                    disallow_web_tools=self._disallow_web_tools
                    or egress_denylist is not None,
                    install_layer_bypass_reason=self._proxy_layer_bypass_reason(),
                )

        if beside:
            (
                self._agent_env,
                self._usage_runtime,
            ) = await self._start_proxy_beside_guards(start_proxy, guards)
        else:
            self._agent_env, self._usage_runtime = await start_proxy()
        # A reservation this connection did not take goes back.
        self._release_host_proxy()
        if previous_runtime is not None and previous_runtime is not self._usage_runtime:
            self._retain_usage_runtime(previous_runtime)
            self._add_reported_models(previous_runtime)
        self._record_proxy_layer_use()
        if egress_denylist is not None:
            self._agent_env = denylist_agent_env(self._agent_env, egress_denylist)
        self._pin_credential_homes(cfg.primary_agent, self._agent_env)
        if not beside:
            sf_entrypoint = self._session_factory_entrypoint(cfg.primary_agent)
            self._check_cloud_metadata_protocol(sf_entrypoint)
            # Before the first command the agent user runs (the Codex Apps probe).
            await self._check_firewall_power(
                rollout_dir,
                agent=cfg.primary_agent,
                agent_env=self._agent_env,
                session_factory=sf_entrypoint,
            )
        elif self._firewall_holds(self._agent_env) != self._firewall_holds(guard_env):
            # The check ran on the env before the proxy's wiring; when that
            # wiring changes which rules hold the agent, check again on it.
            await self._check_firewall_power(
                rollout_dir,
                agent=cfg.primary_agent,
                agent_env=self._agent_env,
                session_factory=sf_entrypoint,
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
            harness=cfg.harness,
        )
        # After proxy wiring and admission: a slot holding the proxy key is
        # written as a file (C24), and a refused connection stages nothing.
        await self._stage_connection_credentials(
            cfg.primary_agent,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            key=(cfg.primary_agent, cfg.primary_model, ()),
            disallow_web_tools=self._disallow_web_tools,
            disallow_hosted_search=self._disallow_hosted_search,
        )
        if egress_denylist is None:
            await self._stop_active_egress()
        if not beside:
            await self._apply_cloud_metadata_block(
                rollout_dir, session_factory=sf_entrypoint
            )
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
            ) = await self._connect_live_agent(
                env=self._env,
                agent=cfg.primary_agent,
                agent_launch=self._agent_launch,
                # The primary prompt runs with no active role: self._timeout.
                agent_env=with_prompt_budget(
                    cfg.primary_agent, self._agent_env, self._timeout
                ),
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
        self._native_usage_proxy_covered = bool(
            getattr(self._usage_runtime, "proxy_active", False)
        )
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

    async def _stage_muse_credentials(
        self, agent_env: dict[str, str], model: str | None
    ) -> None:
        """Pin credential discovery to the managed home for this connection."""
        cfg = self._config
        validate_muse_roles([("muse-code", model)], sandbox_user=cfg.sandbox_user)
        assert cfg.sandbox_user is not None
        home = f"/home/{cfg.sandbox_user}"
        if self._muse_credentials_staged:
            # A caller opening a second connection must end the first one.
            await self.disconnect()
        self._record_credential(f"{home}/.config/muse/auth.json")
        await remove_muse_credentials(self._env, cfg.sandbox_user)
        self._muse_credentials_staged = True
        await self._planes.write_credential_files(
            self._env,
            "muse-code",
            agent_env,
            AGENTS["muse-code"],
            model,
            home,
            record=self._record_credential,
        )
        # Override caller-supplied homes too: the shim and binary must agree
        # with the credential writer and the fixed evidence/snapshot exclusion.
        agent_env["BENCHFLOW_AGENT_HOME"] = home
        agent_env["XDG_CONFIG_HOME"] = f"{home}/.config"

    def _pin_credential_homes(self, agent: str, agent_env: dict[str, str]) -> str:
        """The launcher and the ledger must address the same managed home.

        Host login discovery may use relocated folders, but sandbox logins
        use the fixed destination (agents.login_dirs). These internal launch
        inputs cannot redirect a new credential outside the cleanup ledger.
        """
        user = self._config.sandbox_user
        home = f"/home/{user}" if user else "/root"
        registered = AGENTS.get(agent)
        if "BENCHFLOW_AGENT_HOME" in agent_env or (
            registered and registered.launcher_credential_files
        ):
            agent_env["BENCHFLOW_AGENT_HOME"] = home
        if agent.removeprefix("acpx:") == "codex-acp":
            agent_env["CODEX_HOME"] = f"{home}/.codex"
        return home

    async def _stage_connection_credentials(
        self,
        agent: str,
        agent_env: dict[str, str],
        agent_cfg: Any,
        model: str | None,
        *,
        key: tuple[Any, ...],
        disallow_web_tools: bool,
        disallow_hosted_search: bool,
    ) -> None:
        """Stage one connection's credentials from the admitted values.

        Runs after proxy wiring and admission, so a slot holding the proxy
        key is written as a file (C24). First the previous connection ends:
        on a role transition its agent is stopped (C26), then every ledger
        entry is removed. Each destination, including the ones the launch
        command writes (C28), is recorded before it is written (C27). The
        web-tool policy setup, which may edit a credential-backed config,
        runs after the files exist (C30).
        """
        self._connection_key = key
        cfg = self._config
        cred_home = self._pin_credential_homes(agent, agent_env)
        # The registry entry the credential table reads, also when the
        # install was skipped and no config was returned.
        agent_cfg = agent_cfg or AGENTS.get(agent)
        # Disarm every launcher path before any writer (including web policy)
        # sees it. The same ledger owns native, uploaded and launcher files.
        writer_plan = launcher_credential_plan(
            agent,
            list(getattr(agent_cfg, "launcher_credential_files", None) or ()),
            agent_launch_env(agent, model, agent_env),
            codex_acp=cfg.harness != HARNESS_NATIVE,
        )
        for path in writer_plan.paths:
            await prepare_launcher_credential(
                self._env,
                path.format(home=cred_home, workspace=self._agent_cwd),
                owner=cfg.sandbox_user,
                record=self._record_credential,
            )
        if agent == "muse-code":
            await self._stage_muse_credentials(agent_env, model)
        else:
            record = self._record_credential
            await self._planes.write_credential_files(
                self._env, agent, agent_env, agent_cfg, model, cred_home, record=record
            )
            if credential_policy(agent, model, agent_env).host_login:
                await self._planes.upload_subscription_auth(
                    self._env, agent, cred_home, record=record
                )
        await self._planes.apply_web_tool_policy(
            self._env,
            agent,
            agent_cfg,
            cred_home,
            disallow=disallow_web_tools,
            disallow_hosted_search=disallow_hosted_search,
        )
        # Uploads consume the original credential table, including file-only
        # sources. Apply launch-only empties after every staging consumer.
        agent_env.update(writer_plan.env_overrides)

    def _record_credential(self, path: str) -> None:
        if path not in self._credential_ledger:
            self._credential_ledger.append(path)

    async def _end_previous_connection(self, key: tuple[Any, ...]) -> None:
        """Stop the previous role's agent, then remove every staged credential.

        This barrier precedes *all* credential consumers, not just upload.
        Pending process cleanup is independent of both the role key and the
        file ledger: an empty ledger or identical role is not proof of death.
        """
        if self._connection_needs_quiescence:
            await self.disconnect()
            if (
                self._connection_needs_quiescence
                and keep_services.trial_keep(self._task, self._config.sandbox_user)
                is not None
            ):
                # Keeping listeners also keeps processes whose access to
                # the next login cannot be revoked by deleting old files.
                # Refuse before probes, proxy setup or credential staging.
                raise RuntimeError(
                    "Cannot reconnect while x-keep-services preserves processes "
                    "from a previous connection; use a fresh rollout"
                )
        await self._remove_ledger()

    async def _remove_ledger(self) -> None:
        """Remove every ledger entry; entries stay until removal is confirmed."""
        if not self._credential_ledger:
            return
        try:
            await remove_staged_credentials(
                self._env, self._config.sandbox_user, tuple(self._credential_ledger)
            )
        except RuntimeError:
            if self._muse_credentials_staged:
                raise RuntimeError(
                    "Muse credential cleanup failed; refusing another role or verifier"
                ) from None
            raise
        self._credential_ledger = []
        self._muse_credentials_staged = False

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
        sandbox_user = self._config.sandbox_user
        if (
            self._connection_needs_quiescence
            and sandbox_user
            and (
                self._muse_credentials_staged
                or keep_services.trial_keep(self._task, sandbox_user) is None
            )
        ):
            # Reuse K1's verified UID stop; do not clean while a writer can
            # rename or recreate a login. Kept-service role transitions still
            # need K1's descendant tracking, not a UID-wide service kill.
            await self._planes.quiesce_agent(self._env, sandbox_user)
            self._connection_needs_quiescence = False
            self._agent_quiesced = True
        # Kill any lingering agent processes to prevent context bleed between scenes
        agent_pattern = _agent_process_kill_pattern(self._agent_launch)
        # A stopped sandbox has no agent process left to kill.
        stopped = getattr(self, "_agent_sandbox_stopped", False)
        # Once per connection when K1's kill stopped every process of the
        # sandbox user first (the verifier's capture disconnects again after
        # the run loop did): the agent ran as that user, so no process of it
        # is left to match, and nothing was launched since (PERF-G).
        repeated = bool(sandbox_user) and getattr(self, "_agent_pkill_done", False)
        if self._env and agent_pattern and not stopped and not repeated:
            with contextlib.suppress(Exception):
                await exec_raw(
                    self._env.exec,
                    f"pkill -f {shlex.quote(agent_pattern)} || true",
                    timeout_sec=10,
                )
            self._agent_pkill_done = getattr(self, "_agent_quiesced", False)
        if self._env is not None:
            try:
                await self._remove_ledger()
            except Exception as e:
                if self._muse_credentials_staged:
                    raise
                # Kept pending: the next connection retries and fails closed.
                logger.warning(f"Credential cleanup at disconnect failed: {e}")
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
        # A cancelled run never reaches the final write: publish the tail the
        # live sink held back, as public main's #1165 does (it writes at most
        # every 250 ms). After the empty-delta return, so a completed prompt's
        # file is never rewritten.
        mark_prompt_end = getattr(session, "mark_prompt_end", None)
        if callable(mark_prompt_end):
            mark_prompt_end()
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
        server = getattr(self._usage_runtime, "server", None)
        activity = getattr(type(server), "model_request_activity", None)
        if self._session is not None and server is not None and callable(activity):
            self._session.model_request_activity = server.model_request_activity

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
        agent = (
            getattr(self._active_role, "agent", "")
            if self._active_role is not None
            else self._config.primary_agent
        ) or ""
        # Missing telemetry is missing cost coverage too. Record the harness
        # before inspecting its optional usage, including zero-token sessions.
        # A proxy can also fail to return telemetry. Its ACP fallback still
        # needs harness ownership before any session-log estimate is allowed.
        if self._session is not None:
            self._native_usage_agents.add(agent)
            if not self._native_usage_proxy_covered:
                self._has_unproxied_native_session = True
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
        if not self._native_usage_proxy_covered:
            uncovered = dict(self._unproxied_native_usage_metrics)
            for (
                snapshot_field,
                result_field,
            ) in _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT.items():
                uncovered[result_field] = _as_nonnegative_int(
                    uncovered.get(result_field)
                ) + (delta.get(snapshot_field) or 0)
            details = dict(uncovered.get("usage_details") or {})
            details["thought_tokens"] = _as_nonnegative_int(
                details.get("thought_tokens")
            ) + (delta.get("thought_tokens") or 0)
            uncovered.update(
                usage_source=USAGE_SOURCE_AGENT_NATIVE_ACP, usage_details=details
            )
            self._unproxied_native_usage_metrics = uncovered

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
        See ``docs/jobs/branching-reference.md``.
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
        from benchflow.rollout._verifier_recovery import mark_solver_complete

        if self._muse_credentials_staged:
            await self.disconnect()
        mark_solver_complete(self)
        # From the solver's completion to the rewards. Stand-ins built
        # without __init__ have no timing and none of the helpers, which is
        # why they are called through the class.
        with timing_spans.span(
            timing_spans.AGENT_STOP_TO_REWARD, into=getattr(self, "_timing", None)
        ):
            try:
                return await Rollout._score_finished_solver(self)
            except Exception as exc:
                if verifier_infrastructure_failure(exc):
                    self._rewards = None
                    self._verifier_error = (
                        "verifier crashed: preparation failed: "
                        + verifier_failure_reason(exc)
                    )
                    self._phase = "verified"
                    return None
                if (
                    getattr(self._env, "_benchflow_test_command_issued", False)
                    is not True
                ):
                    raise
                warning = f"Verifier scored 0: preparation after an earlier test command failed: {describe_exception(exc)}"
                logger.warning(warning)
                vars(self).setdefault("_warnings", []).append(warning)
                self._rewards, self._verifier_error = {"reward": 0.0}, None
                self._phase = "verified"
                return self._rewards

    async def _score_finished_solver(self) -> dict | None:
        cfg = self._config

        from benchflow.rollout._verifier_recovery import (
            PRESERVED_SOLVER,
            recovery_ineligible_reason,
        )

        # What the agent user could reach, before anything is hardened or
        # collected (benchflow.integrity.access).
        await _probe_integrity_access(self, "post")
        # Mark the phase at entry (the other transitions mark completion):
        # the verifier can run for minutes after disconnect() reset the phase
        # to "installed", and the dashboard's activity cell reads _phase to
        # label that stretch "verifying…" instead of going blank.
        self._phase = "verifying"
        self._kill_record = None
        separate = separate_verifier_requested(getattr(self._task, "config", None))
        order = Rollout._separate_verifier_order(self) if separate else None
        if order is not None:
            # The agent's sandbox stops before anything crosses, and the
            # verifier's starts meanwhile.
            with _keeping_trial_warnings(self):
                (
                    self._rewards,
                    self._verifier_error,
                ) = await self._verify_separate_overlap(order)
            self._phase = "verified"
            return self._rewards

        await capture_terminal_workspace(self)
        published: asyncio.Future[None] | None = None
        if not separate and Rollout._publish_beside_collection(self):
            # The trajectory goes up while the artifacts are collected: the
            # collection reads only the mounted /logs/artifacts and writes
            # only the host's artifacts folder; the publication writes only
            # /logs/agent, its stage beside it and the host's agent folder.
            # Its outcome is awaited where the publication ran before, so a
            # failure is handled there as before (PERF-G).
            await Rollout._scrape_trajectory_if_missing(self)
            published = asyncio.ensure_future(
                _publish_trajectory_for_verifier(
                    self._env,
                    self._trajectory,
                    self._rollout_paths.agent_dir,
                    sandbox_user=cfg.sandbox_user,
                    publish_reply=not is_scripted_agent(cfg.primary_agent),
                )
            )
            try:
                await collect_rollout_artifacts(self)
            except BaseException:
                published.cancel()
                with contextlib.suppress(BaseException):
                    await published
                raise
        else:
            await collect_rollout_artifacts(self)
            await Rollout._scrape_trajectory_if_missing(self)

        if separate:
            # The verifier never runs in the agent's sandbox: a fresh one gets
            # only the frozen workspace and the collected artifacts. The
            # serial order makes its own; a warmed-up one is not used.
            await self._discard_verifier_warmup()
            with _keeping_trial_warnings(self):
                self._rewards, self._verifier_error = await run_separate_verifier(self)
            self._phase = "verified"
            return self._rewards

        # Only a task with a verifier-only recovery contract keeps its verifier
        # failures off the solver-retry path. Every other task keeps main's
        # error text and exceptions, which evaluation retries as before.
        recovery_eligible = recovery_ineligible_reason(self) is None
        verifier_timeout_diag = None
        try:
            if published is not None:
                await published
            else:
                await _publish_trajectory_for_verifier(
                    self._env,
                    self._trajectory,
                    self._rollout_paths.agent_dir,
                    sandbox_user=cfg.sandbox_user,
                    publish_reply=not is_scripted_agent(cfg.primary_agent),
                )

            with _keeping_trial_warnings(self):
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
                    kill_record=self._kill_record,
                    # A branch child's own folder, not the parent root its
                    # paths keep (host writers follow _rollout_dir).
                    record_dir=self._require_rollout_dir(),
                )
        except Exception as exc:
            if getattr(
                self._env, "_benchflow_test_command_issued", False
            ) is True and not verifier_infrastructure_failure(exc):
                warning = f"Verifier scored 0: publication after an earlier test command failed: {describe_exception(exc)}"
                logger.warning(warning)
                vars(self).setdefault("_warnings", []).append(warning)
                self._rewards, self._verifier_error = {"reward": 0.0}, None
            else:
                if (
                    not recovery_eligible
                    and not isinstance(exc, VerifierPublicationError)
                    and not verifier_infrastructure_failure(exc)
                ):
                    raise
                self._rewards = None
                self._verifier_error = (
                    str(exc)
                    if isinstance(exc, VerifierPublicationError)
                    else "verifier crashed: " + verifier_failure_reason(exc)
                )
        if recovery_eligible and self._verifier_error is not None:
            self._verifier_error = f"{PRESERVED_SOLVER} {self._verifier_error}"
        if verifier_timeout_diag is not None:
            self._diagnostics.set(verifier_timeout_diag)
        # Whether hardening filled the verifier's tool state from the
        # verifier layer (layers.verifier.seed). getattr: stand-ins built
        # without setup() have no records.
        if isinstance(getattr(self, "_layer_records", None), dict):
            self._record_layers()

        self._phase = "verified"
        return self._rewards

    def _publish_beside_collection(self) -> bool:
        """Whether the trajectory may be published while the artifacts are
        collected: on a sandbox that runs steps side by side
        (sandbox._in_order), with /logs mounted from the host, when the task
        declares no artifacts, so the collection reads /logs/artifacts alone
        and nothing either one writes is read by the other (PERF-G)."""
        env = getattr(self, "_env", None)
        task = getattr(self, "_task", None)
        artifacts = getattr(getattr(task, "config", None), "artifacts", None)
        return (
            env is not None
            and runs_steps_at_once(env)
            and getattr(env, "is_mounted", False) is True
            and isinstance(artifacts, (list, tuple))
            and not artifacts
            and not getattr(self, "_branch_child_active", False)
        )

    async def _scrape_trajectory_if_missing(self) -> None:
        cfg = self._config
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

    def _separate_verifier_order(self) -> str | None:
        """How a separate verifier overlaps the agent's release, or None.

        ``prestart``: the verifier sandbox starts from task files at verify()
        entry (Docker, remote Docker). ``after_stop``: it starts once the
        agent's sandbox stopped (Daytona, until an agent there is shown unable
        to reach an idle verifier sandbox; Modal, so its GPUs are released
        first). None keeps the serial order, where
        the agent's sandbox is still up while the verifier runs: a verify()
        outside run()'s lifecycle (a branch child, an SDK caller who may still
        use the sandbox), an externally owned sandbox, or
        ``BENCHFLOW_VERIFIER_PREPARE=serial``.
        """
        from benchflow.rollout._separate_verifier import serial_order_requested

        if (
            not getattr(self, "_release_agent_before_verify", False)
            or getattr(self, "_env_externally_owned", False)
            or getattr(self, "_env", None) is None
            or serial_order_requested()
        ):
            return None
        from benchflow._utils.sandbox_providers import PROVIDERS_BY_NAME

        provider = PROVIDERS_BY_NAME.get(str(getattr(self._config, "environment", "")))
        return provider.separate_verifier_start if provider is not None else None

    async def _verify_separate_overlap(
        self, order: str
    ) -> tuple[dict | None, str | None]:
        """Separate mode with the agent's sandbox released before the transfer.

        In order: the verifier sandbox starts (``prestart``); capture,
        artifact collection and the trajectory scrape run as before; the
        trajectory the verifier reads is kept; the teardown steps that read
        the agent's sandbox run (:meth:`_steps_before_agent_stop`) and the
        agent's sandbox stops; the verifier sandbox starts (``after_stop``);
        then the transfer is packed and the verifier runs. Any exception or
        cancellation before the verifier's own run deletes the verifier
        sandbox once and keeps the trial's error as it would be without it.
        """
        from benchflow.rollout._separate_verifier import OVERLAP, VerifierSandboxRun

        # The sandbox the warm-up started as the agent's turn began, if any.
        run = getattr(self, "_verifier_warmup", None) or VerifierSandboxRun(
            self, order=OVERLAP
        )
        self._verifier_warmup = None
        # A warmed-up sandbox is charged from here, as one started here is.
        run.begin_billing()
        try:
            if order == "prestart":
                run.begin_start()  # nothing when the warm-up started it
            await capture_terminal_workspace(self)
            await collect_rollout_artifacts(self)
            # Read after capture, which writes it (the record was made before).
            run.record["outside_workspace_writes"] = getattr(
                self, "_outside_workspace_writes", None
            )
            await self._scrape_trajectory_if_missing()
            run.snapshot_trajectory(self._trajectory)
            with timing_spans.span(timing_spans.AGENT_RELEASE, into=self._timing):
                if await self._steps_before_agent_stop(defer_failures=True):
                    await self._stop_agent_sandbox_before_verify(run)
                else:
                    run.agent_stopped(
                        "not stopped before the verifier: a teardown step "
                        "before the stop failed (cleanup runs it again)"
                    )
            if order == "after_stop":
                if not self._agent_sandbox_stopped:
                    # These providers require release before allocation, not
                    # merely an attempted stop. Keep the solver's result
                    # unscored and leave the retained handle to cleanup.
                    reason = run.record.get("custody", {}).get("agent_stop_error")
                    error = (
                        "separate verifier not started: agent sandbox release "
                        f"was not confirmed: {reason or 'stop did not complete'}"
                    )
                    run.record["error"] = error
                    await run.discard(cancelled=False)
                    return None, error
                run.begin_start()
            return await run.finish()
        except BaseException as exc:
            await run.discard(cancelled=isinstance(exc, asyncio.CancelledError))
            raise

    async def _stop_agent_sandbox_before_verify(self, run: Any) -> None:
        """Stop (``keep_containers``: only stop) the agent's sandbox.

        A failed or wedged stop is logged and recorded as ``agent_stop_error``;
        the sandbox is left to ``cleanup()``, which stops it again, and the
        verifier runs only if its start order permits overlap. An
        ``after_stop`` verifier stays unscored until release is confirmed.
        The providers log a failed delete instead of raising, so a stop that
        returned counts only once :func:`agent_sandbox_left_running` finds
        the sandbox gone.
        """
        from benchflow.rollout._separate_verifier import agent_sandbox_left_running

        env: Any = self._env
        # Daytona's stop drops its sandbox handle whether or not the delete
        # went through (or was cancelled at the bound); keep it, so the check
        # can look the sandbox up and cleanup() can delete it again.
        handle = getattr(env, "_sandbox", None)
        error: str | None = None
        # The handle is this stop's until the check finds the sandbox gone;
        # on every other way out (a failed or wedged stop, a cancellation or
        # error during the delete or the check) the finally below hands it
        # back to cleanup(), which deletes it again (SV-ALL-1: a cancel during
        # Daytona's delete left the trial with no handle and the box running).
        gone = False
        try:
            with timing_spans.span(
                timing_spans.AGENT_SANDBOX_STOP_BEFORE_VERIFY, into=self._timing
            ):
                try:
                    await asyncio.wait_for(
                        env.stop(delete=True), timeout=_AGENT_STOP_TIMEOUT_SEC
                    )
                except Exception as exc:
                    error = describe_exception(exc)
                left = await agent_sandbox_left_running(env, handle)
            gone = left is None
            if error is None:
                error = left
            if error is None:
                self._agent_sandbox_stopped = True
                run.agent_stopped(None)
                return
            logger.warning(
                "The agent's sandbox did not stop before the verifier; "
                "cleanup stops it again: %s",
                error,
            )
            run.agent_stopped(error)
        except BaseException as exc:
            # Recorded, then the cancellation (or error) goes on as before.
            run.agent_stopped(
                "the stop before the verifier was interrupted "
                f"({type(exc).__name__}); cleanup stops it again"
            )
            raise
        finally:
            if (
                not gone
                and handle is not None
                and getattr(env, "_sandbox", None) is None
            ):
                env._sandbox = handle

    async def soft_verify(self) -> tuple[dict | None, str | None, str | None]:
        """Run the verifier without full hardening — for intermediate feedback.

        Skips process kill and workspace restore/chown (so the sandbox
        stays usable for the next round), but DOES purge agent-injected
        conftest.py / sitecustomize.py / .pth files to prevent the agent
        from gaming intermediate test results.

        Returns (rewards, verifier_output, verifier_error). The final
        verify() still does full hardening.

        The agent's processes stay alive throughout, so the whole phase runs
        as beside kept services: exec adds no persistent env, and the
        verifier hands its env (persistent env included) to its commands
        through a file only root can read, never a command line.
        """
        services_kept = getattr(getattr(self, "_env", None), "services_kept", None)
        with services_kept() if callable(services_kept) else contextlib.nullcontext():
            return await self._soft_verify()

    async def _soft_verify(self) -> tuple[dict | None, str | None, str | None]:
        if separate_verifier_requested(getattr(self._task, "config", None)):
            # Soft verification would upload tests/ into the agent's sandbox.
            return (
                None,
                None,
                "soft_verify is unavailable: the task uses a separate verifier "
                "sandbox, so tests never enter the agent's sandbox",
            )
        require_safe_branch_world(self)
        if self._muse_credentials_staged:
            await self.disconnect()
        self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        try:
            await _publish_trajectory_for_verifier(
                self._env,
                self._trajectory,
                self._rollout_paths.agent_dir,
                sandbox_user=self._config.sandbox_user,
                publish_reply=not is_scripted_agent(self._config.primary_agent),
            )
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
                workspace=self._agent_cwd,
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
            verifier.agent_processes_alive = True
            # Hardening's BASH_ENV policy, as the scored path applies it; a
            # planes object without it gives none (b127 PY8-MAIN-26).
            resolve_bash_env = getattr(self._planes, "soft_verifier_bash_env", None)
            bash_env = (
                resolve_bash_env(
                    self._env, self._task, self._config.sandbox_user, self._agent_cwd
                )
                if callable(resolve_bash_env)
                else ""
            )
            if inspect.isawaitable(bash_env):
                bash_env = await bash_env
            verifier.bash_env = bash_env if isinstance(bash_env, str) else ""
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = _ensure_canonical_rewards(
                verifier_result.rewards, task=self._task
            )
            # Capture raw verifier output for the user. The agent is alive and
            # the image's PATH may lead into its folders: root's cat comes
            # from the system shell (C8).
            cat = await exec_raw(
                system_shell_exec,
                self._env,
                "cat /logs/verifier/*.log 2>/dev/null || "
                "cat /logs/verifier/output.txt 2>/dev/null || true",
                what="Soft verify's log read",
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
        """Close ACP client and stop the environment.

        A rollout that never reached its verifier (the agent errored) has its
        artifacts collected here, before the sandbox stops. Its seconds are
        the ``cleanup`` span.
        """
        with timing_spans.span(timing_spans.CLEANUP, into=self._timing):
            await self._cleanup_steps()

    async def _cleanup_steps(self) -> None:
        # A trial that ended before its agent connected gives back the warm
        # proxy it asked for (_reserve_host_proxy).
        self._release_host_proxy()
        # Destroying an owned sandbox and releasing its lease never wait on the
        # steps before them (AG-R2-3): a disconnect that cannot remove a staged
        # Muse login raises, and that sandbox still holds the login.
        try:
            await self._cleanup_before_teardown()
        finally:
            try:
                # A trial that never verified: its warmed-up verifier sandbox.
                await self._discard_verifier_warmup()
            finally:
                await self._teardown_sandbox()

    async def _cleanup_before_teardown(self) -> None:
        self._capture_partial_acp_trajectory()
        disconnect_error: Exception | None = None
        try:
            await self.disconnect()
        except Exception as exc:
            # Raised after teardown. Nothing more is read from this sandbox:
            # it may still hold the login the disconnect could not remove.
            disconnect_error = exc
            logger.error(f"Disconnect failed; the sandbox is still torn down: {exc}")
        if disconnect_error is None:
            await collect_unverified_rollout_artifacts(self)

        if disconnect_error is None:
            await self._steps_before_agent_stop()
        else:
            # Drain telemetry even when staged credentials could not be removed;
            # never read skills or session logs from that sandbox.
            await self._teardown_usage_runtime()
            await self._teardown_egress()
            await self._teardown_finalize_usage()
            raise disconnect_error
        self._enforce_required_usage_tracking()
        if self._proxy_reported_models:
            _record_reported_models(
                self._rollout_dir, self._proxy_reported_models, "provider_response"
            )
        else:
            _record_reported_models(
                self._rollout_dir, self._session_reported_models, "agent_session_log"
            )

    async def _steps_before_agent_stop(self, *, defer_failures: bool = False) -> bool:
        """The teardown steps that read the agent's sandbox or its usage proxy.

        In this order: skill export, the usage runtime's stop (with the ACP
        reconcile), the egress proxy's stop, usage finalization and the
        session-log price. cleanup() runs them; in separate verifier mode
        verify() runs them first, before it stops the agent's sandbox, and
        each one that finished is marked done so cleanup() skips it.
        ``_enforce_required_usage_tracking`` stays in cleanup(), after them,
        so it cannot fail a trial before it is graded.

        With ``defer_failures`` a step that raises is logged and left, with
        the steps after it, for cleanup() to run (and raise) as before; the
        return value says whether every step finished.
        """
        steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] = (
            ("skill_export", self._teardown_skill_export),
            ("usage_runtime_stop", self._teardown_usage_runtime),
            ("egress_stop", self._teardown_egress),
            ("finalize_usage", self._teardown_finalize_usage),
            ("price_session_log", self._teardown_price),
        )
        for name, step in steps:
            if name in self._teardown_done:
                continue
            try:
                await step()
            except Exception:
                if not defer_failures:
                    raise
                logger.warning(
                    "Teardown step %s failed before the agent's sandbox stopped; "
                    "cleanup runs it again",
                    name,
                    exc_info=True,
                )
                return False
            self._teardown_done.add(name)
        return True

    async def _teardown_skill_export(self) -> None:
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

    async def _teardown_usage_runtime(self) -> None:
        usage_runtime = self._usage_runtime
        if usage_runtime is not None or self._retired_usage_runtimes:
            usage_stopped = True
            try:
                if usage_runtime is not None:
                    await self._planes.stop_provider_runtime(usage_runtime)
            except Exception as e:
                logger.warning(f"Usage telemetry runtime stop failed: {e}")
                usage_stopped = False
            # Retired proxies have already imported their final callback log.
            # Do not stop them again, or count a reused runtime twice.
            for runtime in self._usage_runtimes():
                server = getattr(runtime, "server", None)
                summary = getattr(type(server), "request_summary", None)
                if callable(summary):
                    timing, invalid = summary(server)
                    for name, seconds in timing.items():
                        self._timing[name] = self._timing.get(name, 0.0) + seconds
                    if invalid is None or getattr(
                        self, "_invalid_provider_response_cached", None
                    ):
                        continue
                    self._invalid_provider_response_cached = invalid
                    if self._error is not None and (
                        "expecting value" in self._error.lower()
                        or "provider api error" in self._error.lower()
                    ):
                        self._error = f"{invalid}{ORIGINAL_ERROR}{self._error}"
            if "agent_execution" in self._timing and "provider_wait" in self._timing:
                self._timing["agent_without_provider_wait"] = max(
                    0.0, self._timing["agent_execution"] - self._timing["provider_wait"]
                )
            usage_runtime = self._combined_usage_runtime()
            self._usage_metrics = self._planes.extract_usage(
                usage_runtime if usage_stopped else None
            )
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
            self._add_reported_models(usage_runtime)
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
                self._retired_usage_runtimes.clear()

    async def _teardown_egress(self) -> None:
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

    async def _teardown_finalize_usage(self) -> None:
        self._finalize_usage_metrics()

    async def _teardown_price(self) -> None:
        # Unpriced Claude Code usage (a subscription): estimate its USD from
        # Claude Code's session log while the sandbox is still up.
        await price_from_session_log(self)

    async def _teardown_sandbox(self) -> None:
        if self._environment is not None:
            if not self._agent_sandbox_stopped:
                with contextlib.suppress(Exception):
                    await self._environment.teardown()
            self._environment = None

        env = (
            self._env
            if (
                self._env
                and not self._env_externally_owned
                and not self._agent_sandbox_stopped
            )
            else None
        )
        # An externally-owned sandbox (use_prebuilt_env) belongs to the
        # caller — leave it running so they can reuse it or stop it
        # themselves. #388.
        task_tmp = self._task_tmp
        lease = getattr(self, "_recovery_lease_owned", False)

        async def stop_and_remove() -> None:
            if env is not None:
                try:
                    await env.stop(delete=True)
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}")
            if lease:
                from benchflow.rollout._verifier_recovery import (
                    release_lease_at_teardown,
                )

                await release_lease_at_teardown(self)
            if task_tmp:
                shutil.rmtree(task_tmp, ignore_errors=True)

        from benchflow._warm_pool import task_shares_nothing
        from benchflow.rollout._teardown_queue import teardown_queue

        # In a job a Docker sandbox's stop runs in the background, so the
        # trial's result is not held for it (PERF-B, rollout/_teardown_queue.py);
        # the job drains it, and the trial's budget counts its seconds. A
        # provider's sandbox, one holding a verifier-recovery lease, and a
        # task reserving shared host resources stop inline. The next trial
        # cannot claim a fixed port, device or name until this stop ends.
        queue = teardown_queue()
        if (
            queue is not None
            and env is not None
            and not lease
            and getattr(env, "background_stop", False) is True
            and task_shares_nothing(self._config.task_path)
        ):
            self._queued_stop = await queue.submit(
                stop_and_remove, what=self._config.task_path.name
            )
        else:
            await stop_and_remove()

        self._teardown_done.clear()
        self._agent_sandbox_stopped = False
        self._phase = "cleaned"

    def _finalize_usage_metrics(self) -> None:
        """Prefer provider usage for covered roles; add only uncovered ACP usage."""
        if self._usage_metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE:
            native = self._unproxied_native_usage_metrics
            if (
                self._has_unproxied_native_session or is_token_usage_available(native)
            ) and not self._mixed_usage_finalized:
                provider = dict(self._usage_metrics)
                combined = dict(provider)
                for field in _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT.values():
                    combined[field] = _as_nonnegative_int(
                        provider.get(field)
                    ) + _as_nonnegative_int(native.get(field))
                combined["usage_details"] = {
                    **(provider.get("usage_details") or {}),
                    "provider_response": provider,
                    "agent_native_acp": dict(native),
                }
                # Provider spend covers only some roles; never present it as
                # the complete cost of a run containing subscription usage,
                # even if timeout/disconnect preceded its first usage report.
                combined.update(cost_usd=None, price_source=None)
                self._usage_metrics = combined
                self._mixed_usage_finalized = True
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

    def _record_taskmd_timed_out(self) -> None:
        """Record in a task.md draft 2 package's reward.json whether the agent ran out of time.

        task.md: "In every case reward.json records timed_out" (docs/document.md,
        "Timeouts"). The rewards themselves are unchanged, and a trial without a
        reward gets no reward file.
        """
        from benchflow.taskmd.materialize import taskmd_metadata

        paths = getattr(self, "_rollout_paths", None)
        if (
            paths is None
            or self._rewards is None
            or taskmd_metadata(self._config.task_path) is None
        ):
            return
        path = paths.reward_json_path
        try:
            if path.is_file():
                from benchflow.task.verifier_core import _REWARD_FILE_MAX_BYTES

                with destination(path).open("rb") as stream:
                    data = stream.read(_REWARD_FILE_MAX_BYTES + 1)
                if len(data) > _REWARD_FILE_MAX_BYTES:
                    raise ValueError("reward.json exceeds the reward file limit")
                recorded = json.loads(data)
            else:
                recorded = {}
        except (OSError, ValueError, RecursionError, OverflowError):
            recorded = None
        if not isinstance(recorded, dict):
            logger.warning(f"Cannot record timed_out in unreadable {path}")
            return
        recorded = (recorded or dict(self._rewards)) | {
            "timed_out": bool(getattr(self, "_agent_ran_out_of_time", False))
        }
        # A verifier that ran as root in the container leaves reward.json root-owned,
        # so the host cannot open it for writing; replacing it needs only a writable
        # folder. The new file is written privately and placed through the held
        # verifier folder, never through a link at the name (R4-1). Recording
        # timed_out must never fail a trial that has its rewards.
        try:
            path.write(json.dumps(recorded, indent=2) + "\n")
        except OSError as e:
            logger.warning(f"Cannot record timed_out in {path}: {e}")

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
        # The agent's wall-clock budget ran out (an idle-watchdog abort is not
        # running out of time); a task.md draft 2 reward.json records it.
        ran_out_of_time = (
            agent_phase
            and not isinstance(e, IdleTimeoutError)
            and (
                isinstance(e, AgentPromptTimeoutError)
                or "exceeded wall-clock budget" in detail
                or not detail
            )
        )
        if ran_out_of_time:
            self._agent_ran_out_of_time = True
        # Judged on what the agent did, like a bare timeout. An idle timeout
        # is left out: it is retried, an integration failure is not.
        self._judge_timed_out_agent = ran_out_of_time and bool(detail)
        if not detail and self._started_at is not None:
            elapsed = (datetime.now() - self._started_at).total_seconds()
            detail = (
                f"Agent timed out after {elapsed:.0f}s elapsed in rollout "
                f"(budget {self._timeout}s)"
            )
        self._error = detail or f"Agent timed out after {self._timeout}s"
        self._diagnostics.capture_idle(e)
        # Read with getattr: continue_run's Trial records its timeouts here too.
        server = getattr(getattr(self, "_usage_runtime", None), "server", None)
        stalled = getattr(type(server), "stalled_model_requests", None)
        count, age = stalled(server) if agent_phase and callable(stalled) else (0, 0)
        role = getattr(self, "_active_role", None)
        idle = (
            role.idle_timeout_sec
            if role and role.idle_timeout_sec is not None
            else getattr(getattr(self, "_config", None), "agent_idle_timeout", None)
        )
        if count and idle and age >= idle:
            # The provider has not begun to answer for longer than the idle
            # budget (the stall dc509087a's idle deadline retried): a provider
            # stall, retried and resumed like any transient provider error.
            # A younger pending request leaves the timeout as it was.
            ending = (
                "went idle" if isinstance(e, IdleTimeoutError) else "ran out of time"
            )
            self._error = (
                f"{PROVIDER_STALL} the agent {ending} while {count} model "
                f"request(s) had no response; the oldest was pending {age:.1f}s\n"
                "next: check the model endpoint and "
                "trajectory/provider_requests.jsonl, then resume the job"
            )
            self._bare_timeout = False
            self._judge_timed_out_agent = False
            self._provider_stall = True
        if isinstance(e, AgentPromptTimeoutError) and getattr(
            e, "terminal_trajectory_complete", False
        ):
            self._terminal_timeout = True
        logger.error(self._error)

    async def run(self) -> RolloutResult:
        """Run the complete trial lifecycle under a host-side hard deadline.

        Cancellation leaves an unscored spend receipt after cleanup, so a
        later resume cannot forget the attempt's trial and verifier seconds.
        """
        from benchflow.budget import cancelled_attempt_receipt
        from benchflow.task.staging import remove_released
        from benchflow.trajectories.types import trial_redaction_memo

        with trial_redaction_memo():
            try:
                with (
                    trial_log_context(_log_task(self)),
                    cancelled_attempt_receipt(
                        lambda: (
                            self._rollout_dir
                            if self._config.purpose == "task"
                            else None
                        )
                    ),
                ):
                    return await self._run_with_scoring()
            finally:
                # A task copy a collected sandbox's finalizer had to queue goes
                # now, whoever runs the trial (Runtime.execute, branch, regrade).
                remove_released()

    async def _run_with_scoring(self) -> RolloutResult:
        """Execute, clean up, and publish the attempt's scoring when complete.

        The lifecycle itself lives in :meth:`_run_lifecycle`; the deadline is
        a backstop against awaits wedged below every phase-level timeout (a
        Daytona PTY kill on a dead websocket, a hung session exec in the
        post-verify export path) — see :mod:`benchflow.rollout._deadline`.
        A trip while the agent is still running stops only the agent and is
        reported as its timeout after verification (#1134); any other trip
        becomes a normal infra-retryable error result and the abandoned
        attempt's cleanup is bounded too.
        """
        started_at = datetime.now()
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
        elif not result.rollout_name and result.error_category == "infra_failure":
            # The deadline synthesizes a terminal result even when setup never
            # returned. Publish it before aggregation, or later job readers
            # silently drop the failed trial. Cancellation still propagates
            # above and never fabricates a completed attempt here.
            if self._rollout_dir is None:
                (
                    self._task,
                    self._rollout_dir,
                    self._rollout_paths,
                    self._started_at,
                    self._job_name,
                    self._rollout_name,
                ) = _init_rollout(
                    self._config.task_path,
                    self._config.job_name,
                    self._config.rollout_name,
                    self._config.jobs_dir,
                )
                self._started_at = started_at
            self._error = result.error
            self._rewards = self._scoring = None
            self._verifier_error = None
            self._partial_trajectory = True
            self._rollout_name = self._rollout_name or self._rollout_dir.name
            self._started_at = self._started_at or started_at
            with scoring_lock(self._rollout_dir):
                result = read_admitted_result(self) or self._build_result()
            self._completed_result = result
            return result
        return await self._finish_scoring(result)

    async def _finish_scoring(self, result: RolloutResult) -> RolloutResult:
        """Serialize initial scoring and resumed scoring through the same lock."""
        if result.rollout_name and self._rollout_dir is not None:
            with scoring_lock(self._rollout_dir):
                result = await self._finish_scoring_locked(result)
                # Synchronous CPU and file IO over the whole trajectory (about
                # 0.4 ms per recorded tool call), so it runs off the event loop:
                # it is outside the hard deadline and inside the scoring lock,
                # and every other trial in this process would stall for it.
                await asyncio.to_thread(self._write_integrity, result)
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
                with _keeping_trial_warnings(self):
                    self._rewards, self._verifier_error = await recover_verifier(self)
            except Exception as exc:
                self._rewards = None
                self._verifier_error = (
                    f"[solver-preserved] verifier recovery unavailable: {exc}"
                )
            result = self._build_result(result_filename=publish)
        elif (
            self._phase == "cleaned"
            or not (rollout_dir / "solver.json").is_file()
            or not needs_verifier_recovery(result.verifier_error)
            or recovery_ineligible_reason(self) is not None
        ):
            # A post-solver deadline can interrupt cleanup before recovery
            # may run. Keep its original image lease for scoring-only resume.
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

    def _write_integrity(self, result: RolloutResult) -> None:
        """Audit the finished trial (``integrity`` audit/strict); rewards untouched.

        Runs after verifier recovery and review, so it sees the final scoring.
        Reviewer rollouts are never audited.
        """
        if self._config.integrity == "off" or self._config.purpose != "task":
            return
        from benchflow.integrity.trial import write_rollout_integrity

        verdict = write_rollout_integrity(self, result)
        if verdict is not None:
            result.integrity = verdict.as_dict()

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
            await self._setup_and_start()

            if cfg.primary_agent == "nop":
                # The empty control: nothing runs, the verifier scores the
                # untouched workspace.
                await self._install_agent_step()
                self._trajectory, self._agent_name = _nop_trajectory()
                from benchflow.rollout._verifier_recovery import mark_solver_complete

                mark_solver_complete(self)
            elif cfg.primary_agent == "oracle":
                await self._install_agent_step()
                # The sandbox-user grant chowned the workspace away from the
                # user solve.sh runs as (root, or the task's [agent] user):
                # trust the workspace and its repositories again for that
                # user, and nothing outside it (DF-p3-eval-runner-2), on the
                # task route solve.sh runs on.
                await trust_workspace_repos(self._env, self._agent_cwd, self._task)
                # That harness step writes the oracle user's ~/.gitconfig:
                # restart the agent phase clock so it is not listed as the
                # solver's output.
                await mark_agent_phase_start(self)
                oracle_started = datetime.now()
                # A task network policy covers the oracle's turn; its arming
                # records the oracle_network_policy span here.
                with timing_spans.recording_into(self._timing):
                    self._trajectory, self._agent_name = await _run_oracle(
                        self._env,
                        cfg.task_path,
                        self._timeout,
                        sandbox_user=None,
                        rollout_dir=self._rollout_dir,
                        task=self._task,
                    )
                self._timing["agent_execution"] = (
                    datetime.now() - oracle_started
                ).total_seconds()
                from benchflow.rollout._verifier_recovery import mark_solver_complete

                mark_solver_complete(self)
            elif is_operator_script(cfg.primary_agent):
                await self._install_agent_step()
                script_env: dict[str, str] = {}
                script_started = datetime.now()
                try:
                    if self._egress_denylist is not None:
                        await self._start_egress_denylist(self._egress_denylist)
                        script_env = denylist_agent_env({}, self._egress_denylist)
                    await self._planes.enforce_agent_egress_firewall(
                        self._env, cfg.sandbox_user, script_env
                    )
                    assert (
                        cfg.sandbox_user is not None and cfg._script_bytes is not None
                    )
                    self._trajectory, self._agent_name = await _run_script(
                        self._env,
                        cfg._script_bytes,
                        script_label(cfg.primary_agent),
                        self._timeout,
                        cfg.sandbox_user,
                        self._agent_cwd,
                        script_env,
                        planes=self._planes,
                        task=self._task,
                    )
                    self._agent_ran_out_of_time = self._trajectory[0]["timed_out"]
                finally:
                    self._timing["agent_execution"] = (
                        datetime.now() - script_started
                    ).total_seconds()
                    await self._stop_active_egress()
                from benchflow.rollout._verifier_recovery import mark_solver_complete

                mark_solver_complete(self)
            else:
                await self._install_agent_step()
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

            if not cfg.skip_verify:
                # The trial's own last verify(): a separate verifier may stop
                # the agent's sandbox before its transfer (cleanup() follows).
                self._release_agent_before_verify = True
                try:
                    await self.verify()
                finally:
                    self._release_agent_before_verify = False
                if (
                    agent_timed_out
                    and self._rewards is None
                    and self._verifier_error is None
                ):
                    self._rewards = {"reward": 0.0}
                    self._verifier_error = None
                self._record_taskmd_timed_out()

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
            if not isinstance(e, UsageLimitError):
                # A usage limit is logged once, below, with its login.
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
            if isinstance(e, UserError):
                # Expected (a missing login, a task file that does not
                # parse, an image that does not build): its message says it
                # all, a traceback would not.
                logger.error(f"Run failed: {user_message(e)}")
            else:
                # One line at the default level; the traceback only with
                # `bench -v` (DEBUG), once per attempt (DF-p5-cli-craft-7).
                saved = write_error_traceback(self._rollout_dir, e)
                _write_guard_progress(self._rollout_dir, e)
                logger.error(
                    "Run failed: %s%s",
                    render_text(" ".join(describe_exception(e).split()), 300),
                    f" (traceback: {saved})" if saved is not None else "",
                )
                logger.debug("Run failed", exc_info=True)
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
            invalid = getattr(self, "_invalid_provider_response_cached", None)
            if invalid:
                self._error = f"{invalid}{ORIGINAL_ERROR}{self._error}"
            logger.error(self._error)

        if self._rollout_dir is None:
            return RolloutResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return prepare_terminal_result(self)

    # Warm pool: the steps before the agent, ahead of the trial's turn.

    def begin_prestart(self) -> asyncio.Task[None]:
        """Run setup(), start() and the install now, in a task of their own.

        The job's warm pool (``benchflow._warm_pool``) calls this for a trial
        still waiting for its concurrency slot. They are the same calls a
        cold start makes, in the same order, so the sandbox is created and
        hardened exactly as then (user, lockdown, freezer proof, verifier
        baseline). run() waits for this task instead of making the calls and
        raises its exception where the calls would have raised it. Call it
        once, before run().
        """
        if self._prestart is not None or self._phase != "created":
            raise RuntimeError("begin_prestart() must run once, before run()")
        # Wall clock (time.monotonic, as the budget counts), for the trial's
        # sandbox seconds at adoption.
        self._prestart_began_at = time.monotonic()
        # The task copies this context: its lines name the trial's task.
        with trial_log_context(_log_task(self)):
            self._prestart = asyncio.create_task(self._prestart_steps())
        self._prestart.add_done_callback(self._note_prestart_done)
        return self._prestart

    async def _prestart_steps(self) -> None:
        await self.setup()
        await self.start()
        await self.install_agent()
        self._prestart_ready_at = timing_spans.clock()

    def on_sandbox_ready(self, callback: Callable[[], None]) -> None:
        """Call *callback* once the trial's sandbox is started and its agent
        installed (the job's warm pool then prepares the next trial)."""
        self._on_sandbox_ready = callback

    def _note_prestart_done(self, _task: asyncio.Task[None]) -> None:
        self._prestart_done_at = time.monotonic()

    @property
    def prestarted(self) -> bool:
        """Whether begin_prestart() ran for this rollout."""
        return self._prestart is not None

    @property
    def adopted(self) -> bool:
        """Whether run()'s lifecycle began (it then owns the sandbox's clean-up)."""
        return self._trial_clock_start is not None

    async def discard_prestart(self) -> None:
        """Remove a prestarted trial the job never ran: its sandbox and folder.

        Cancels the prestart if it is still running, then cleans up as a
        cancelled trial does (sandbox stop and delete, lease, staging copy),
        and removes the trial folder setup() made, which then holds no result.
        Never raises: the job's end must reach every other entry.
        """
        task = self._prestart
        if task is None:
            return
        if not task.done():
            task.cancel()
        # wait() never raises the task's own outcome (a cancellation or the
        # start's error): the entry is dropped either way.
        await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved, so asyncio does not log it as lost
        if self._phase != "cleaned":
            # Safe in any phase: run()'s lifecycle also cleans up after a
            # setup() that failed on its first line.
            try:
                await self.cleanup()
            except Exception:
                logger.warning(
                    "Could not clean up an unused warm-pool sandbox", exc_info=True
                )
        from benchflow.task.staging import remove_released

        # With PERF-B's queue the sandbox stop may still be running: the
        # folder goes only after it (PERF-ALL §2, merge wave batch 31).
        queued = getattr(self, "_queued_stop", None)
        if queued is not None and not queued.done():
            await asyncio.wait({queued})
        remove_released()
        rollout_dir = self._rollout_dir
        if rollout_dir is not None and not (rollout_dir / "result.json").exists():
            shutil.rmtree(rollout_dir, ignore_errors=True)

    async def _setup_and_start(self) -> None:
        """setup() and start(), or wait for the warm pool's run of them."""
        self._trial_clock_start = timing_spans.clock()
        task = getattr(self, "_prestart", None)
        if task is None:
            await self.setup()
            await self.start()
            return
        # The slot waits the job handed this trial's own task (the rollout
        # itself was made in the pool's task, which had none).
        self._timing.update(timing_spans.take_pending_trial_timing())
        ready_at = self._prestart_ready_at
        _credit_prestart_seconds(self)
        try:
            await task
        finally:
            timing_spans.add_into(
                self._timing,
                timing_spans.WARM_POOL_WAIT,
                timing_spans.clock() - self._trial_clock_start,
            )
            if ready_at is not None:
                timing_spans.add_into(
                    self._timing,
                    timing_spans.WARM_POOL_IDLE,
                    max(0.0, self._trial_clock_start - ready_at),
                )

    async def _install_agent_step(self) -> None:
        """install_agent(), unless the warm pool already ran it for this trial."""
        if getattr(self, "_prestart", None) is None:
            await self.install_agent()
        else:
            # The agent phase starts now, not when the prepared sandbox began
            # to wait for its slot (separate verifier mode only).
            await mark_agent_phase_start(self)
        trial_start = getattr(self, "_trial_clock_start", None)
        if trial_start is not None:
            timing_spans.add_into(
                self._timing,
                timing_spans.SANDBOX_READY,
                timing_spans.clock() - trial_start,
            )
        self._begin_verifier_warmup()
        notify = getattr(self, "_on_sandbox_ready", None)
        if notify is not None:
            try:
                notify()
            except Exception:
                logger.debug("warm pool notification failed", exc_info=True)

    def _begin_verifier_warmup(self) -> None:
        """Separate verifier on Docker: start its sandbox as the agent's turn begins.

        The same start SV3's overlap order makes at verify() entry
        (:meth:`VerifierSandboxRun.begin_start`: task files only, nothing
        from the agent's run), made earlier so its seconds overlap the
        agent's turn. Only where that order starts it at entry (Docker and
        remote Docker, run()'s own lifecycle, a sandbox BenchFlow owns) and
        not under ``BENCHFLOW_VERIFIER_PREPARE=serial`` or ``entry``. The
        verifier sandbox is its own Compose project on its own network with
        no published port, so the agent cannot reach it. verify() takes it;
        a trial that never verifies deletes it in cleanup().
        """
        from benchflow.rollout._separate_verifier import (
            OVERLAP,
            VerifierSandboxRun,
            warmup_requested,
        )

        cfg = self._config
        if (
            getattr(self, "_verifier_warmup", None) is not None
            or cfg.skip_verify
            or getattr(self, "_env_externally_owned", False)
            or getattr(self, "_env", None) is None
            or cfg.environment not in ("docker", "remote-docker")
            or not separate_verifier_requested(getattr(self._task, "config", None))
            or not warmup_requested()
        ):
            return
        run = VerifierSandboxRun(self, order=OVERLAP)
        run.begin_start(warm=True)
        self._verifier_warmup = run

    async def _discard_verifier_warmup(self) -> None:
        """Delete a warmed-up verifier sandbox no verify() took (exactly once)."""
        run = getattr(self, "_verifier_warmup", None)
        self._verifier_warmup = None
        if run is not None:
            await run.discard(cancelled=False)

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

        Retries transient download failures up to 3 times.
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
        validate_muse_roles(
            [(cfg.primary_agent, cfg.primary_model), (role.agent, role.model)],
            sandbox_user=cfg.sandbox_user,
        )
        await self._end_previous_connection((role.agent, role.model, ()))
        self._connection_needs_quiescence = True
        self._agent_pkill_done = self._agent_quiesced = False
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
        previous_runtime = self._usage_runtime
        role_agent_differs = role.agent != cfg.primary_agent
        needs_role_credentials = (
            role_agent_differs or role.model != cfg.primary_model or bool(role.env)
        )
        force_sandbox_local = disallow_web_tools or disallow_hosted_search
        # A host proxy starts while the sandbox guards run (see
        # _guards_beside_proxy); they then check the env before its wiring.
        beside = self._guards_beside_proxy(
            force_sandbox_local=force_sandbox_local,
            egress_denylist=egress_denylist,
            role_changes_sandbox=needs_role_credentials,
        )
        guard_env = dict(agent_env)
        sf_entrypoint: str | None = None
        if beside:
            sf_entrypoint = self._session_factory_entrypoint(role.agent)
            self._check_cloud_metadata_protocol(sf_entrypoint)

        async def guards() -> None:
            # Before the first command the agent user runs (the Codex Apps
            # probe), then the block before anything the agent user runs.
            await self._check_firewall_power(
                rollout_dir,
                agent=role.agent,
                agent_env=guard_env,
                session_factory=sf_entrypoint,
            )
            if not is_scripted_agent(role.agent):
                await self._apply_cloud_metadata_block(
                    rollout_dir, session_factory=sf_entrypoint
                )

        async def start_proxy() -> tuple[dict[str, str], Any]:
            # The proxy's install and health wait record their spans here. The
            # trial's first connection, a role's included, may take the proxy
            # reserved when the sandbox started (_reserve_host_proxy); only one
            # started from this role's launch is handed out.
            claim = getattr(self._planes, "host_proxy_claim", None)
            with (
                timing_spans.recording_into(self._timing),
                claim(self._host_proxy_reservation)
                if claim is not None
                else contextlib.nullcontext(),
            ):
                return await self._planes.ensure_litellm_runtime(
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
                    prior_runtimes=self._usage_runtimes(),
                    live_trajectory_path=rollout_dir
                    / "trajectory"
                    / "llm_trajectory.jsonl",
                    force_sandbox_local=force_sandbox_local,
                    disallow_web_tools=disallow_web_tools or disallow_hosted_search,
                    install_layer_bypass_reason=self._proxy_layer_bypass_reason(),
                )

        if beside:
            agent_env, self._usage_runtime = await self._start_proxy_beside_guards(
                start_proxy, guards
            )
        else:
            agent_env, self._usage_runtime = await start_proxy()
        # A reservation this connection did not take goes back.
        self._release_host_proxy()
        # A role on another model or route stopped the previous proxy: keep
        # the models it recorded.
        if previous_runtime is not None and previous_runtime is not self._usage_runtime:
            self._retain_usage_runtime(previous_runtime)
            self._add_reported_models(previous_runtime)
        self._record_proxy_layer_use()
        if egress_denylist is not None:
            agent_env = denylist_agent_env(agent_env, egress_denylist)

        if role_agent_differs:
            if cfg.skip_agent_install:
                agent_cfg = None
            else:
                with timing_spans.span(timing_spans.INSTALL_AGENT, into=self._timing):
                    agent_cfg = await install_agent_with_policy(
                        self._planes,
                        self._env,
                        role.agent,
                        rollout_dir,
                        sandbox_setup_timeout=cfg.sandbox_setup_timeout,
                        install_layer_bypass_reason=self._agent_layer_bypass_reason(),
                    )
                    await self._install_native_cli(role.agent, rollout_dir)
                if cfg.sandbox_user:
                    # The role's install ran as root too (#1183); the user
                    # exists since the primary install.
                    await _prepare_agent_temp_dirs(
                        self._env,
                        agent_cfg,
                        cfg.sandbox_user,
                        locked_paths=self._effective_locked,
                    )
        else:
            agent_cfg = self._agent_cfg
        if needs_role_credentials:
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await _install_native_task_mcp_config(
                self._env,
                self._task,
                agent_cfg=agent_cfg,
                cred_home=cred_home,
                owner=cfg.sandbox_user,
            )

        self._pin_credential_homes(role.agent, agent_env)
        if not beside:
            sf_entrypoint = self._session_factory_entrypoint(role.agent)
            self._check_cloud_metadata_protocol(sf_entrypoint)
            # Before the first command the agent user runs (the Codex Apps probe).
            await self._check_firewall_power(
                rollout_dir,
                agent=role.agent,
                agent_env=agent_env,
                session_factory=sf_entrypoint,
            )
        elif self._firewall_holds(agent_env) != self._firewall_holds(guard_env):
            # The check ran on the env before the proxy's wiring; when that
            # wiring changes which rules hold the agent, check again on it.
            await self._check_firewall_power(
                rollout_dir,
                agent=role.agent,
                agent_env=agent_env,
                session_factory=sf_entrypoint,
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
            harness=cfg.harness,
        )
        # Every connection stages its own credentials, after proxy wiring
        # and admission; the previous connection's are removed first.
        await self._stage_connection_credentials(
            role.agent,
            agent_env,
            agent_cfg,
            role.model,
            key=(role.agent, role.model, tuple(sorted((role.env or {}).items()))),
            disallow_web_tools=disallow_web_tools,
            disallow_hosted_search=disallow_hosted_search,
        )
        self._agent_launch = agent_launch

        if egress_denylist is None:
            await self._stop_active_egress()
        if not beside and not is_scripted_agent(role.agent):
            await self._apply_cloud_metadata_block(
                rollout_dir, session_factory=sf_entrypoint
            )
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
            ) = await self._connect_live_agent(
                env=self._env,
                agent=role.agent,
                agent_launch=agent_launch,
                # The same budget execute() gives this role's prompts.
                agent_env=with_prompt_budget(
                    role.agent,
                    agent_env,
                    role.timeout_sec if role.timeout_sec is not None else self._timeout,
                ),
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
        self._native_usage_proxy_covered = bool(
            getattr(self._usage_runtime, "proxy_active", False)
        )
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)
        self._active_role = role

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # Internal helpers

    async def _probe_sandbox_health(self) -> None:
        """Quick health probe after transport death. Enriches transport diagnostic.

        distinguishes Daytona session killed vs agent crash.
        """
        diag = self._diagnostics.transport_closed
        if diag is None or self._env is None:
            return
        try:
            result = await asyncio.wait_for(
                exec_raw(
                    self._env.exec, "echo __BENCHFLOW_HEALTH_OK__", timeout_sec=10
                ),
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
        if isinstance(e, UsageLimitError):
            # Name the login (a label, never the token) and record the typed
            # diagnostic: the trial is an unscored usage_limit, never retried.
            from benchflow.agents.env import login_label

            named = e.with_login(
                login_label(
                    self._config.primary_agent,
                    self._config.primary_model,
                    self._agent_env,
                    self._config.agent_env,
                )
            )
            self._diagnostics.set(UsageLimitDiagnostic.from_error(named))
            return str(named)
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

    def _retain_usage_runtime(self, runtime: Any) -> None:
        if not any(old is runtime for old in self._retired_usage_runtimes):
            self._retired_usage_runtimes.append(runtime)

    def _usage_runtimes(self) -> tuple[Any, ...]:
        runtimes = list(self._retired_usage_runtimes)
        if self._usage_runtime is not None and not any(
            old is self._usage_runtime for old in runtimes
        ):
            runtimes.append(self._usage_runtime)
        return tuple(runtimes)

    def _usage_trajectories(self) -> tuple[ProviderTrajectory, ...]:
        return tuple(
            trajectory
            for runtime in self._usage_runtimes()
            if (
                trajectory := getattr(
                    getattr(runtime, "server", None), "trajectory", None
                )
            )
            is not None
        )

    def _combined_usage_runtime(self) -> Any:
        if not self._retired_usage_runtimes:
            return self._usage_runtime
        trajectories = self._usage_trajectories()
        combined = ProviderTrajectory(
            session_id=self._rollout_name or "",
            exchanges=[
                exchange
                for trajectory in trajectories
                for exchange in trajectory.exchanges
            ],
        )
        priced = [t for t in trajectories if t.has_provider_usage]
        if priced and all(t.total_cost_usd is not None for t in priced):
            combined.metadata["cost_usd"] = sum(t.total_cost_usd or 0.0 for t in priced)
        return SimpleNamespace(
            kind="litellm", server=SimpleNamespace(trajectory=combined)
        )

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
        if is_operator_script(self._config.agent):
            return
        if bare_timeout and self._n_tool_calls > 0:
            # Missing token telemetry cannot erase observed work, even when
            # the provider's last captured request failed (PR #1131).
            return
        if getattr(self, "_provider_stall", False):
            # Like any provider error: no scoreable model activity to score.
            self._rewards = None
            return
        if not bare_timeout and (self._error is not None or not self._executed_prompts):
            # A wall-clock timeout with a message keeps it for the API
            # verdicts, but an agent that did nothing before it (no tool call,
            # message, thought or output token) is an integration failure, as
            # bf.load_job reads the result: a model that never answered is
            # not a 0. Judged once.
            if getattr(self, "_judge_timed_out_agent", False):
                self._judge_timed_out_agent = False
                if self._n_tool_calls == 0:
                    _classify_integration_failure(self)
            return
        # Native-subscription runs have NO usage channel: the LiteLLM proxy is
        # deliberately skipped (the subscription-auth split) and the CLI authenticates
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
            if getattr(self, "_invalid_provider_response_cached", None):
                self._error = (
                    f"{self._invalid_provider_response_cached}"
                    f"{ORIGINAL_ERROR}{self._error}"
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
            agent=script_label(self._config.primary_agent),
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
            scenes=[
                replace(
                    scene,
                    roles=[
                        replace(role, agent=script_label(role.agent))
                        for role in scene.roles
                    ],
                )
                for scene in self._config.effective_scenes
            ],
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
            verifier_isolation=self._verifier_isolation,
            cloud_metadata=self._cloud_metadata_result,
            warnings=getattr(self, "_warnings", None),
            **self._usage_metrics,
        )

    def _branch_summary(self) -> dict[str, Any] | None:
        """result.json's ``branches`` block; None when the rollout never branched."""
        from benchflow.branch_lineage import branch_summary

        return branch_summary(getattr(self, "_branch_forks", []))


def _credit_prestart_seconds(rollout: Rollout) -> None:
    """Charge the trial, at adoption, for its sandbox's life before its turn.

    A warm-pool prestart ran before the job admitted the trial, so the
    budget's clock (from the trial's start, sandbox creation included) did
    not see it: the seconds from the prestart's start to its end, or to now
    when it is still running, are added to the trial (PERF-C review L1).
    """
    from benchflow.budget import credit_trial_seconds

    began = getattr(rollout, "_prestart_began_at", None)
    if began is None:
        return
    now = time.monotonic()
    done = getattr(rollout, "_prestart_done_at", None)
    credit_trial_seconds(max(0.0, min(now if done is None else done, now) - began))


# What decides whether a trial makes its sandbox through setup() and start()
# (benchflow._warm_pool prepares a trial through those same two calls).
_STOCK_LIFECYCLE = {
    name: Rollout.__dict__[name]
    for name in ("create", "run", "_run_lifecycle", "begin_prestart")
}


def lifecycle_is_stock(rollout_class: Any) -> bool:
    """Whether trials of *rollout_class* make their sandbox as the pool would.

    The warm pool calls setup(), start() and install_agent() ahead of the
    trial, whatever they are. A caller or test that replaced how a trial is
    made or run (``create``, ``run``, the lifecycle) would not call them,
    and the pool would make a sandbox the trial never uses: it stays off.
    """
    return rollout_class is Rollout and all(
        Rollout.__dict__.get(name) is method
        for name, method in _STOCK_LIFECYCLE.items()
    )


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

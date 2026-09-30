"""Connect a native harness: the counterpart of :func:`benchflow.acp.runtime.connect_acp`.

``connect_native`` prepares everything the ACP path prepares before its first
prompt, from the same inputs and with the same helpers: the Codex launch
config (``apply_codex_launch_config``), model and effort selection (the ACP
runtime's rules for when the launch environment owns the model), the task's
MCP servers, and the sandbox-user egress firewall. It adds two checks the ACP
adapter makes implicitly:

* the CLI in the sandbox is the pinned one (``--version``), so a run never
  drives an unpinned binary;
* the model route is BenchFlow's: the model proxy for an API-key run, or, for
  Claude Code only, its own subscription login. A launch environment holding
  anything else (a raw provider key, a ChatGPT login for Codex) is refused
  before a process starts, so no model call can bypass the proxy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchflow.acp.runtime import (
    _format_acp_model,
    _model_selection_owned_by_env,
    _resolve_acp_model_input,
)
from benchflow.acp.session import ACPSession
from benchflow.acp.types import McpServerSpec
from benchflow.agents.codex_config import apply_codex_launch_config
from benchflow.agents.env import uses_native_subscription_auth
from benchflow.native_harness.client import NativeCLIClient
from benchflow.native_harness.harnesses import native_harness_for
from benchflow.native_harness.session import NativeSession
from benchflow.native_harness.spec import NativeHarness
from benchflow.providers.litellm_config import LITELLM_MASTER_KEY_ENV
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

logger = logging.getLogger(__name__)

# Codex login material the native CLI must not see: native Codex runs only
# through the proxy, and with one of these it could sign in to ChatGPT instead.
_CODEX_LOGIN_ENV = ("CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON")


class NativeRouteRefused(ValueError):
    """The launch environment would let the CLI reach a model outside BenchFlow's route."""


def _proxied(env: dict[str, str]) -> bool:
    return bool(env.get(LITELLM_MASTER_KEY_ENV))


def check_model_route(
    harness: NativeHarness, agent: str, model: str | None, env: dict[str, str]
) -> dict[str, str]:
    """The launch env, stripped of credentials the CLI must not hold; raise if unroutable."""
    launch = {k: v for k, v in env.items() if k not in _CODEX_LOGIN_ENV}
    master = env.get(LITELLM_MASTER_KEY_ENV, "")
    if harness.cli == "codex":
        if not _proxied(env):
            raise NativeRouteRefused(
                "native Codex runs only through BenchFlow's model proxy (an API-key "
                "route); this run has no proxy route (a ChatGPT login or no key)"
            )
        base = env.get("OPENAI_BASE_URL", "")
        if env.get("OPENAI_API_KEY") != master or not base:
            raise NativeRouteRefused(
                "native Codex must hold only the proxy's key and endpoint"
            )
        return launch
    if env.get("ANTHROPIC_API_KEY"):
        raise NativeRouteRefused(
            "a raw ANTHROPIC_API_KEY would reach the Claude Code CLI; "
            "API-key runs go through BenchFlow's model proxy"
        )
    if _proxied(env):
        if env.get("ANTHROPIC_AUTH_TOKEN") != master or not env.get(
            "ANTHROPIC_BASE_URL"
        ):
            raise NativeRouteRefused(
                "native Claude Code must hold only the proxy's key and endpoint"
            )
        return launch
    if not uses_native_subscription_auth(agent, model, env):
        raise NativeRouteRefused(
            "native Claude Code needs BenchFlow's model proxy (an API key) or its "
            "own subscription login (CLAUDE_CODE_OAUTH_TOKEN)"
        )
    return launch


async def verify_cli(env: Any, harness: NativeHarness) -> str:
    """Refuse a CLI other than the pin; return its version line."""
    result = await env.exec(
        f"{harness.executable} --version 2>&1", user="root", timeout_sec=60
    )
    lines = [
        line.strip() for line in (result.stdout or "").splitlines() if line.strip()
    ]
    found = lines[-1] if lines else ""
    if result.return_code != 0 or found != harness.version_output:
        raise RuntimeError(
            f"native harness needs {harness.package} at {harness.executable} "
            f"('{harness.version_output}'); found {found!r} "
            f"(exit code {result.return_code})"
        )
    return found


def _model_flag(
    agent: str, model: str | None, env: dict[str, str], *, launch_owns: bool
) -> str | None:
    """The model to pass on the command line, None when the launch env selects it.

    The ACP runtime's rules: with BenchFlow's proxy the model rides the
    environment (``ANTHROPIC_MODEL``, Codex's config), so no flag; a
    subscription run names the model itself.
    """
    if not model or launch_owns or _model_selection_owned_by_env(agent, model, env):
        return None
    return _format_acp_model(_resolve_acp_model_input(agent, model, env), agent)


async def connect_native(
    env: Any,
    agent: str,
    agent_env: dict[str, str],
    sandbox_user: str | None,
    model: str | None,
    rollout_dir: Path,
    environment: str,
    agent_cwd: str,
    reasoning_effort: str | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    resume_session_id: str | None = None,
    **_ignored: Any,
) -> tuple[NativeCLIClient, ACPSession, NativeSession, str]:
    """Prepare a native harness and return ``(client, session, Session, name)``.

    The same four slots ``connect_acp`` fills: the client carries the live
    verbs ``execute_prompts`` drives, the ``ACPSession`` the trajectory, and
    :class:`NativeSession` the Agent-plane contract. No process starts here;
    each prompt runs the CLI for one turn.
    """
    del environment
    harness = native_harness_for(agent)
    if reasoning_effort and reasoning_effort not in harness.efforts:
        raise ValueError(
            f"reasoning_effort={reasoning_effort!r} is not a {harness.cli} effort "
            f"({', '.join(sorted(harness.efforts))})"
        )
    launch_env, launch_owns_model = apply_codex_launch_config(
        agent,
        agent_env,
        model=model,
        reasoning_effort=reasoning_effort,
        sandboxed=bool(sandbox_user),
    )
    launch_env = check_model_route(harness, agent, model, dict(launch_env))
    version = await verify_cli(env, harness)
    model_flag = _model_flag(agent, model, launch_env, launch_owns=launch_owns_model)
    effort = reasoning_effort
    if harness.cli == "codex" and launch_owns_model and reasoning_effort:
        effort = None  # apply_codex_launch_config put it in CODEX_CONFIG
    if sandbox_user is None:
        # Claude Code refuses bypassPermissions as root unless told it is in
        # a sandbox, which it is.
        launch_env.setdefault("IS_SANDBOX", "1")
    await enforce_agent_egress_firewall(env, sandbox_user, launch_env)
    client = NativeCLIClient(
        env=env,
        harness=harness,
        agent=agent,
        launch_env=launch_env,
        sandbox_user=sandbox_user,
        cwd=agent_cwd,
        rollout_dir=rollout_dir,
        model=model_flag,
        reasoning_effort=effort,
        mcp_servers=tuple(mcp_servers or ()),
        resume_id=resume_session_id,
    )
    logger.info(
        "Native harness: %s (%s)%s",
        harness.cli,
        version,
        f", resuming {resume_session_id}" if resume_session_id else "",
    )
    name = f"{harness.cli} {harness.version}"
    return client, client.session, NativeSession(client), name


__all__ = [
    "NativeRouteRefused",
    "check_model_route",
    "connect_native",
    "verify_cli",
]

"""Controller-owned model-only transport for the verified native Claude client."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from typing import Any

from benchflow.agents.env import uses_native_subscription_auth
from benchflow.agents.registry import (
    CLAUDE_AGENT_ACP_LAUNCHER,
    CLAUDE_CODE_EXECUTABLE_PATH,
    pinned_npm_package,
)
from benchflow.sandbox.egress_denylist import EgressDenylist

_LAUNCHER = "/opt/benchflow/bin/claude-agent-acp"

_ROUTING_CONFLICTS = (
    "BENCHFLOW_PROVIDER_BASE_URL",
    "LLM_BASE_URL",
    "BENCHFLOW_PROVIDER_API_KEY",
    "ANTHROPIC_CUSTOM_HEADERS",
    "NODE_OPTIONS",
    "NODE_PATH",
    "CLAUDE_CODE_EXECUTABLE",
    "CLAUDE_CODE_CUSTOM_OAUTH_URL",
    "CLAUDE_LOCAL_OAUTH_API_BASE",
    "CLAUDE_LOCAL_OAUTH_APPS_BASE",
    "CLAUDE_LOCAL_OAUTH_CONSOLE_BASE",
)
_ROUTING_SWITCHES = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_GATEWAY",
)


def native_oauth_egress_policy(
    agent: str, model: str | None, agent_env: dict[str, str], *, no_web: bool
) -> EgressDenylist | None:
    if not no_web or not uses_native_subscription_auth(agent, model, agent_env):
        return None
    if agent != "claude-agent-acp":
        raise ValueError(
            "Native subscription no-web transport is supported only for direct Claude ACP"
        )
    for key in _ROUTING_CONFLICTS:
        if agent_env.get(key):
            raise ValueError(f"Native Claude no-web transport conflicts with {key}")
    for key in _ROUTING_SWITCHES:
        if agent_env.get(key, "").lower() not in ("", "0", "false"):
            raise ValueError(f"Native Claude no-web transport conflicts with {key}")
    if (
        agent_env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        != "https://api.anthropic.com"
    ):
        raise ValueError(
            "Native Claude no-web transport requires the canonical first-party origin"
        )
    return EgressDenylist((), (), native_claude_model_only=True)


def allowlist_model_transport(
    policy: EgressDenylist | None,
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> EgressDenylist | None:
    """Admit the model endpoint an agent needs under ``network_mode='allowlist'``.

    Agents on an API key reach their model through the controller's loopback
    gateway, which the proxy always admits (``start_egress_denylist`` registers
    its port). Native Claude subscription auth has no gateway: it calls
    api.anthropic.com itself, so that origin is admitted for model requests
    only (POST /v1/messages, no URL sources). Other native subscription
    clients are refused here rather than failing silently on their first
    model call (harbor-framework/harbor#2146).
    """
    if policy is None or not policy.allow_mode:
        return policy
    if not uses_native_subscription_auth(agent, model, agent_env):
        return policy
    if agent == "claude-agent-acp":
        return replace(policy, native_claude_model_origin=True)
    raise ValueError(
        f"network_mode='allowlist' does not admit the native subscription endpoint "
        f"of {agent}; run it with an API key so model calls use the sandbox gateway"
    )


async def validate_native_oauth_transport(
    env: Any, sandbox_user: str | None, agent_launch: str, *, harness: str = "acp"
) -> dict[str, Any]:
    """Check actual UID and installed native version before admitting transport.

    ACP must be the registry pin and its SDK the exact version that ACP pins.
    Native Claude is the separately pinned Claude Code CLI: its package and
    its binary's ``--version`` must both be the pin, and the managed launcher
    must be the one that hands exactly that binary to the adapter. With
    ``harness="native"`` BenchFlow launches that CLI itself, so the adapter
    and launcher checks give way to the CLI's own (package, ``--version``,
    and the executable resolving into the pinned package).
    """
    if harness == "native":
        return await _validate_native_cli_transport(env, sandbox_user)
    if not sandbox_user or agent_launch != _LAUNCHER:
        raise ValueError(
            "Native Claude no-web transport requires a managed launcher and nonroot sandbox user"
        )
    uid = await env.exec(
        f"id -u {shlex.quote(sandbox_user)}", user="root", timeout_sec=30
    )
    value = (uid.stdout or "").strip()
    if uid.return_code != 0 or not value.isdecimal() or int(value) == 0:
        raise ValueError(
            "Native Claude no-web transport requires a verified nonzero sandbox UID"
        )
    inherited_guard = (
        "const conflict=" + json.dumps((*_ROUTING_CONFLICTS, "ANTHROPIC_API_KEY")) + ";"
        "const switches=" + json.dumps(_ROUTING_SWITCHES) + ";"
        'if(conflict.some(k=>process.env[k])||switches.some(k=>!["","0","false"].includes((process.env[k]||"").toLowerCase()))||'
        '(process.env.ANTHROPIC_BASE_URL && process.env.ANTHROPIC_BASE_URL.replace(/\\/$/,"")!=="https://api.anthropic.com"))process.exit(2);'
    )
    package, pinned = pinned_npm_package("claude-agent-acp")
    cli_package, cli_pinned = pinned_npm_package("claude-code")
    modules = "/opt/benchflow/js-agents/lib/node_modules"
    script = (
        inherited_guard
        + "const acp="
        + json.dumps(f"{modules}/{package}")
        + ",cli="
        + json.dumps(f"{modules}/{cli_package}")
        + ",exe="
        + json.dumps(CLAUDE_CODE_EXECUTABLE_PATH)
        + ",launcher="
        + json.dumps(_LAUNCHER)
        + ";"
        + """const fs=require('fs'),p=require('path'),cp=require('child_process');
const sdkName='@anthropic-ai/claude-agent-sdk';
const dir=p.dirname(require.resolve(sdkName,{paths:[acp]}));
const r=cp.spawnSync(exe,['--version'],{encoding:'utf8',timeout:10000});
if(r.status!==0)process.exit(1);
const read=f=>JSON.parse(fs.readFileSync(f));
const a=read(p.join(acp,'package.json')),s=read(p.join(dir,'package.json')),c=read(p.join(cli,'package.json'));
console.log(JSON.stringify({acp:a.version,acp_sdk:(a.dependencies||{})[sdkName],sdk:s.version,cli:c.version,native:r.stdout.trim(),launcher:fs.readFileSync(launcher,'utf8')}));"""
    )
    result = await env.exec(
        'test -z "${NODE_OPTIONS-}" && test -z "${NODE_PATH-}" && env -u NODE_OPTIONS -u NODE_PATH /opt/benchflow/node/bin/node -e '
        + shlex.quote(script),
        user="root",
        timeout_sec=30,
    )
    try:
        found = json.loads(result.stdout or "")
    except ValueError:
        found = None
    if (
        result.return_code != 0
        or not isinstance(found, dict)
        or found.get("acp") != pinned
        or not isinstance(found.get("sdk"), str)
        or found.get("acp_sdk") != found["sdk"]
        or found.get("cli") != cli_pinned
        or found.get("native") != f"{cli_pinned} (Claude Code)"
        or found.get("launcher") != CLAUDE_AGENT_ACP_LAUNCHER
    ):
        raise ValueError(
            "Native Claude no-web transport client version is not verified"
        )
    return {
        "mechanism": "root_owned_tls_model_only_proxy",
        "versions": {key: found[key] for key in ("acp", "sdk", "native")},
        "sandbox_uid": int(value),
        "origin": "https://api.anthropic.com",
        "method": "POST",
        "paths": ["/v1/messages", "/v1/messages?beta=true"],
    }


async def _sandbox_uid(env: Any, sandbox_user: str | None) -> int:
    if not sandbox_user:
        raise ValueError(
            "Native Claude no-web transport requires a nonroot sandbox user"
        )
    uid = await env.exec(
        f"id -u {shlex.quote(sandbox_user)}", user="root", timeout_sec=30
    )
    value = (uid.stdout or "").strip()
    if uid.return_code != 0 or not value.isdecimal() or int(value) == 0:
        raise ValueError(
            "Native Claude no-web transport requires a verified nonzero sandbox UID"
        )
    return int(value)


async def _validate_native_cli_transport(
    env: Any, sandbox_user: str | None
) -> dict[str, Any]:
    """Admission for the native harness: BenchFlow runs the pinned CLI directly."""
    uid = await _sandbox_uid(env, sandbox_user)
    cli_package, cli_pinned = pinned_npm_package("claude-code")
    modules = "/opt/benchflow/js-agents/lib/node_modules"
    script = (
        "const conflict="
        + json.dumps((*_ROUTING_CONFLICTS, "ANTHROPIC_API_KEY"))
        + ";const switches="
        + json.dumps(_ROUTING_SWITCHES)
        + ";"
        'if(conflict.some(k=>process.env[k])||switches.some(k=>!["","0","false"].includes((process.env[k]||"").toLowerCase()))||'
        '(process.env.ANTHROPIC_BASE_URL && process.env.ANTHROPIC_BASE_URL.replace(/\\/$/,"")!=="https://api.anthropic.com"))process.exit(2);'
        + "const cli="
        + json.dumps(f"{modules}/{cli_package}")
        + ",exe="
        + json.dumps(CLAUDE_CODE_EXECUTABLE_PATH)
        + ";"
        + """const fs=require('fs'),p=require('path'),cp=require('child_process');
const r=cp.spawnSync(exe,['--version'],{encoding:'utf8',timeout:10000});
if(r.status!==0)process.exit(1);
const real=fs.realpathSync(exe),root=fs.realpathSync(cli);
const c=JSON.parse(fs.readFileSync(p.join(cli,'package.json')));
console.log(JSON.stringify({cli:c.version,native:r.stdout.trim(),inside:real.startsWith(root+p.sep)}));"""
    )
    result = await env.exec(
        'test -z "${NODE_OPTIONS-}" && test -z "${NODE_PATH-}" && env -u NODE_OPTIONS -u NODE_PATH /opt/benchflow/node/bin/node -e '
        + shlex.quote(script),
        user="root",
        timeout_sec=30,
    )
    try:
        found = json.loads(result.stdout or "")
    except ValueError:
        found = None
    if (
        result.return_code != 0
        or not isinstance(found, dict)
        or found.get("cli") != cli_pinned
        or found.get("native") != f"{cli_pinned} (Claude Code)"
        or found.get("inside") is not True
    ):
        raise ValueError(
            "Native Claude no-web transport client version is not verified"
        )
    return {
        "mechanism": "root_owned_tls_model_only_proxy",
        "harness": "native",
        "versions": {"native": found["native"]},
        "sandbox_uid": uid,
        "origin": "https://api.anthropic.com",
        "method": "POST",
        "paths": ["/v1/messages", "/v1/messages?beta=true"],
    }

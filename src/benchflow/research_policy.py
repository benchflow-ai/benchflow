"""Private, run-scoped research access policies.

Research policies are operator inputs, not task inputs.  Keeping them outside
the task package prevents a blocked paper title or URL from becoming part of
the agent-visible benchmark itself.  Only a non-reversible digest and counts
are written to rollout artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from benchflow.task.config import MCPServerConfig

RESEARCH_MCP_NAME = "benchflow-research"
RESEARCH_RUNTIME_PATH = "/opt/benchflow/research_gateway.py"
RESEARCH_POLICY_PATH = "/run/benchflow/research-policy.json"
RESEARCH_GATEWAY_PORT = 8765
RESEARCH_ANTHROPIC_RELAY_BASE = (
    f"http://127.0.0.1:{RESEARCH_GATEWAY_PORT}/provider/anthropic"
)
_DEFAULT_SEARCH_ENDPOINT = "https://lite.duckduckgo.com/lite/"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ResolvedResearchPolicy:
    """Validated policy for exactly one task."""

    task_id: str
    blocked_urls: tuple[str, ...]
    blocked_url_prefixes: tuple[str, ...]
    blocked_hosts: tuple[str, ...]
    blocked_terms: tuple[str, ...]
    blocked_content_sha256: tuple[str, ...]
    search_endpoint: str
    sha256: str

    def runtime_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "task_id": self.task_id,
            "blocked_urls": list(self.blocked_urls),
            "blocked_url_prefixes": list(self.blocked_url_prefixes),
            "blocked_hosts": list(self.blocked_hosts),
            "blocked_terms": list(self.blocked_terms),
            "blocked_content_sha256": list(self.blocked_content_sha256),
            "search_endpoint": self.search_endpoint,
            "policy_sha256": self.sha256,
        }

    def artifact_metadata(self, *, enforced: bool) -> dict[str, Any]:
        """Return safe provenance without policy values or the private path."""

        return {
            "version": 1,
            "sha256": self.sha256,
            "enforced": enforced,
            "blocked_url_count": len(self.blocked_urls),
            "blocked_url_prefix_count": len(self.blocked_url_prefixes),
            "blocked_host_count": len(self.blocked_hosts),
            "blocked_term_count": len(self.blocked_terms),
            "blocked_content_hash_count": len(self.blocked_content_sha256),
        }


def _string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"research policy {key!r} must be a list of strings")
    normalized = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
    if len(normalized) != len(value):
        raise ValueError(
            f"research policy {key!r} contains an empty or duplicate value"
        )
    return normalized


def _validate_public_url(value: str, *, field: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"research policy {field!r} entries must be HTTP(S) URLs")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"research policy {field!r} entries must not contain userinfo")
    try:
        parsed.hostname.encode("idna")
        _ = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError(
            f"research policy {field!r} contains an invalid authority"
        ) from exc


def load_research_policy(path: str | Path, *, task_id: str) -> ResolvedResearchPolicy:
    """Load and resolve a private YAML policy for ``task_id``.

    A policy-enabled batch must explicitly cover every selected task.  Missing
    task entries fail closed rather than silently restoring unrestricted web.
    """

    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise ValueError(f"research policy file does not exist: {policy_path}")
    raw = yaml.safe_load(policy_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("research policy must be a YAML mapping")
    if raw.get("version") != 1:
        raise ValueError("research policy version must be 1")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("research policy must contain a 'tasks' mapping")
    task_raw = tasks.get(task_id)
    if not isinstance(task_raw, dict):
        raise ValueError(f"research policy has no entry for task {task_id!r}")

    blocked_urls = _string_list(task_raw, "blocked_urls")
    blocked_prefixes = _string_list(task_raw, "blocked_url_prefixes")
    try:
        blocked_hosts = tuple(
            host.encode("idna").decode("ascii").lower().rstrip(".")
            for host in _string_list(task_raw, "blocked_hosts")
        )
    except UnicodeError as exc:
        raise ValueError(
            "research policy 'blocked_hosts' contains an invalid hostname"
        ) from exc
    blocked_terms = _string_list(task_raw, "blocked_terms")
    blocked_hashes = tuple(
        value.lower() for value in _string_list(task_raw, "blocked_content_sha256")
    )
    for value in (*blocked_urls, *blocked_prefixes):
        _validate_public_url(value, field="blocked_urls")
    for host in blocked_hosts:
        if not host or "://" in host or "/" in host or "@" in host:
            raise ValueError(
                "research policy 'blocked_hosts' entries must be hostnames"
            )
    for value in blocked_hashes:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(
                "research policy 'blocked_content_sha256' entries must be SHA-256 hex"
            )
    if not any(
        (blocked_urls, blocked_prefixes, blocked_hosts, blocked_terms, blocked_hashes)
    ):
        raise ValueError(f"research policy entry for task {task_id!r} blocks nothing")

    search_endpoint = raw.get("search_endpoint", _DEFAULT_SEARCH_ENDPOINT)
    if not isinstance(search_endpoint, str) or not search_endpoint.strip():
        raise ValueError("research policy 'search_endpoint' must be a URL string")
    search_endpoint = search_endpoint.strip()
    _validate_public_url(search_endpoint, field="search_endpoint")

    canonical = {
        "version": 1,
        "task_id": task_id,
        "blocked_urls": list(blocked_urls),
        "blocked_url_prefixes": list(blocked_prefixes),
        "blocked_hosts": list(blocked_hosts),
        "blocked_terms": list(blocked_terms),
        "blocked_content_sha256": list(blocked_hashes),
        "search_endpoint": search_endpoint,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResolvedResearchPolicy(
        task_id=task_id,
        blocked_urls=blocked_urls,
        blocked_url_prefixes=blocked_prefixes,
        blocked_hosts=blocked_hosts,
        blocked_terms=blocked_terms,
        blocked_content_sha256=blocked_hashes,
        search_endpoint=search_endpoint,
        sha256=digest,
    )


def attach_research_mcp(task: Any) -> None:
    """Attach the policy-enforcing gateway without embedding policy data."""

    sandbox = getattr(getattr(task, "config", None), "sandbox", None)
    if sandbox is None:
        raise RuntimeError("research policy requires a task sandbox configuration")
    existing = {server.name for server in sandbox.mcp_servers}
    if RESEARCH_MCP_NAME in existing:
        raise RuntimeError(
            f"task MCP server name {RESEARCH_MCP_NAME!r} is reserved by BenchFlow"
        )
    sandbox.mcp_servers.append(
        MCPServerConfig(
            name=RESEARCH_MCP_NAME,
            transport="stdio",
            command="python3",
            args=[
                RESEARCH_RUNTIME_PATH,
                "mcp",
                "--endpoint",
                f"http://127.0.0.1:{RESEARCH_GATEWAY_PORT}",
            ],
        )
    )


def route_native_subscription_auth(
    agent: str, model: str | None, agent_env: dict[str, str]
) -> dict[str, str]:
    """Keep native Claude auth usable after the agent UID loses Internet access.

    Subscription-authenticated Claude cannot use LiteLLM because BenchFlow does
    not own an upstream API key.  Route its native Anthropic protocol through
    the root-owned, fixed-destination relay exposed by the research gateway.
    The sandbox user can still reach only loopback; the relay can reach only
    api.anthropic.com and cannot be repurposed as a general web proxy.
    """

    from benchflow.agents.env import uses_native_subscription_auth
    from benchflow.agents.registry import AGENTS

    if not uses_native_subscription_auth(agent, model, agent_env):
        return agent_env

    config = AGENTS.get(agent)
    if (
        config is None
        or config.subscription_auth is None
        or config.subscription_auth.replaces_env != "ANTHROPIC_API_KEY"
    ):
        raise RuntimeError(
            "research policy does not support native subscription auth for "
            f"agent {agent!r}; use provider API-key auth"
        )

    updated = dict(agent_env)
    updated["ANTHROPIC_BASE_URL"] = RESEARCH_ANTHROPIC_RELAY_BASE
    updated["BENCHFLOW_PROVIDER_BASE_URL"] = RESEARCH_ANTHROPIC_RELAY_BASE
    return updated


async def install_research_gateway(env: Any, policy: ResolvedResearchPolicy) -> None:
    """Install and start the root-owned gateway inside a prepared sandbox."""

    runtime_source = (
        Path(__file__).with_name("sandbox") / "_research_gateway_runtime.py"
    )
    await env.upload_file(runtime_source, RESEARCH_RUNTIME_PATH, mode="755")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="benchflow-research-policy-", suffix=".json", delete=False
        ) as handle:
            json.dump(policy.runtime_payload(), handle, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        await env.upload_file(temp_path, RESEARCH_POLICY_PATH, mode="600")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    command = (
        "set -e; command -v python3 >/dev/null 2>&1 || "
        "{ echo 'research policy requires python3' >&2; exit 86; }; "
        "mkdir -p /run/benchflow; chmod 700 /run/benchflow; "
        f"chmod 600 {RESEARCH_POLICY_PATH}; "
        f"nohup python3 {RESEARCH_RUNTIME_PATH} serve "
        f"--policy {RESEARCH_POLICY_PATH} --port {RESEARCH_GATEWAY_PORT} "
        ">/logs/agent/research-gateway.log 2>&1 & "
        "echo $! >/run/benchflow/research-gateway.pid"
    )
    result = await env.exec(command, user="root", timeout_sec=20)
    if getattr(result, "return_code", 0) != 0:
        raise RuntimeError("failed to start the research gateway")

    probe = (
        'python3 -c "import json,urllib.request; '
        f"d=json.load(urllib.request.urlopen('http://127.0.0.1:{RESEARCH_GATEWAY_PORT}/health', timeout=2)); "
        f"assert d.get('policy_sha256') == '{policy.sha256}'\""
    )
    for _ in range(20):
        check = await env.exec(probe, user="root", timeout_sec=5)
        if getattr(check, "return_code", 0) == 0:
            return
        import asyncio

        await asyncio.sleep(0.1)
    raise RuntimeError("research gateway did not become healthy")

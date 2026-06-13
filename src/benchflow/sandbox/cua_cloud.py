"""Cua cloud (warm-pool claim) sandbox provider.

This backend reaches Cua *cloud* desktop environments through the supported
Kubernetes warm-pool **claim API** (the "cyclops-cs" backend), driven by the
``cua_train`` SDK. It is distinct from :mod:`benchflow.sandbox.cua`, which talks
to the local/SDK ``Sandbox`` surface — that surface's cloud command endpoint
currently 404s (``cloud-computer-server-cmd-404``), so the claim API is the
supported path for cloud desktops and gets its own provider rather than being
bolted onto the local one.

How it works (verified against a live pool; see the proven ``claim_and_connect``
reference shared by the Cua team):

* ``cua_train.TrainClient.from_key`` performs the OAuth2 client-credentials
  exchange against the ``cyclops-cs`` realm and hands back an authenticated
  ``httpx.Client`` (via ``get_httpx_client``) that transparently refreshes the
  bearer token — we fetch it per request so a long rollout never carries a
  stale token.
* ``start`` creates an ``OSGymSandboxClaim`` custom resource via the
  ``/api/k8s`` kubectl-proxy; the pool operator binds it to a warm sandbox.
  We poll the claim until ``status.phase == "Bound"`` and a sandbox name is
  set, then poll the sandbox's service endpoint until the in-guest service
  stops answering with a proxy 502/503/504 (the guest OS boots after the bind).
* the claimed sandbox is reached through ``/api/svc/{pool}/{sandbox}-{service}/``.
  The control plane is the ``cua-driver`` service's Streamable-HTTP MCP endpoint
  at ``/api/svc/{pool}/{sandbox}-cua-driver/mcp`` — desktop actions are driven
  over MCP tool calls, not a Unix shell.
* ``stop`` releases the sandbox by DELETE-ing the claim CR. BenchFlow owns every
  claim it creates under a ``benchflow-`` name prefix and tracks the live claim
  so cleanup never leaks a warm sandbox back to the pool.

Scope (the *infra*): the claim lifecycle (create -> bind -> service-ready ->
release with no leak), an authed reach to the ``cua-driver`` MCP endpoint, the
MCP handshake (initialize -> notifications/initialized -> tools/list), and a
capability/metadata probe. The cyclops pool is a **desktop** environment driven
over the ``cua-driver`` MCP control plane, *not* a general Unix shell, so the
shell-shaped ``BaseSandbox`` operations (``exec`` and base64-over-shell file
transfer) raise a structured ``NotImplementedError`` naming the MCP control
plane rather than pretending an unsupported shell exists — the same honest
pattern :mod:`benchflow.sandbox.macos_ios_simulator` uses for in-guest paths
``simctl`` cannot serve.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxStartupError

if TYPE_CHECKING:
    from pathlib import Path

# The OSGymSandboxClaim custom resource coordinates (group/version/plural) and
# the auth/run defaults — kept here as the single source of truth so the URL
# shapes match the proven reference exactly.
_CLAIM_GROUP = "osgym.cua.ai"
_CLAIM_VERSION = "v1alpha1"
_CLAIM_PLURAL = "osgymsandboxclaims"
_CLAIM_KIND = "OSGymSandboxClaim"

_DEFAULT_TOKEN_URL = (
    "https://auth.cua.ai/realms/cyclops-cs/protocol/openid-connect/token"
)
_DEFAULT_BASE_URL = "https://run.cua.ai"

# The control-plane service. The cua-driver exposes a Streamable-HTTP MCP
# endpoint that desktop actions are driven through.
_DRIVER_SERVICE = "cua-driver"

# BenchFlow owns every claim it creates under this name prefix so post-run
# cleanup/audit can find (and never leak) them — mirrors the iOS provider's
# device-name prefix discipline.
_CLAIM_NAME_PREFIX = "benchflow-"

# Claim names are RFC-1123 label-ish; keep ours to a tidy lowercase slug so
# they are shell-safe and greppable in ``kubectl get osgymsandboxclaims``.
_NAME_INVALID = re.compile(r"[^a-z0-9-]+")

# The original SDK-cloud command endpoint failure this provider supersedes.
# Surfaced verbatim in the diagnostic so an operator hitting the old path knows
# to switch ``--sandbox cua`` (cloud) over to ``--sandbox cua-cloud``.
_SUPERSEDED_CMD_404 = "cloud-computer-server-cmd-404"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds") from exc


def _name_part(value: str) -> str:
    slug = _NAME_INVALID.sub("-", value.strip().lower()).strip("-")
    return slug or "task"


def _load_train_client() -> Any:
    """Import the optional ``cua_train`` SDK with an actionable dependency error."""
    try:
        from cua_train import TrainClient

        return TrainClient
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Missing optional dependency for the 'cua-cloud' sandbox. Install "
            "it with `uv sync --extra sandbox-cua-cloud` for local development, "
            "or `pip install 'benchflow[sandbox-cua-cloud]'` for a packaged "
            "install. The cyclops-cs claim API is driven by the `cua-train` SDK "
            "(`pip install cua-train --extra-index-url "
            "https://wheels.cua.ai/simple/`)."
        ) from exc


class CuaCloudSandbox(BaseSandbox):
    """Sandbox backend for Cua cloud desktops via the cyclops-cs claim API."""

    def __init__(
        self,
        *args: Any,
        pool: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve the pool (== Kubernetes namespace) from the constructor or the
        # environment. No default: a wrong namespace would 404 confusingly, so
        # we require an explicit pool.
        self._pool: str | None = pool or _env("BENCHFLOW_CUA_CLOUD_POOL")
        # The authed httpx client factory (``client.get_httpx_client``) — called
        # per request so the bearer token is always fresh.
        self._http_factory: Any | None = None
        # The claim we own and must release. Set before we wait for Bound so a
        # bind failure still releases the (created-but-unbound) claim in stop()
        # — no leak on partial start. ``None`` => nothing to release.
        self._claim_name: str | None = None
        # The bound sandbox name, set once the claim reaches phase Bound.
        self._sandbox: str | None = None
        super().__init__(*args, **kwargs)

    @classmethod
    def preflight(cls) -> None:
        """Fail with an actionable error when the claim path cannot be used."""
        _load_train_client()
        missing = [
            name for name in ("CUA_CLIENT_ID", "CUA_CLIENT_SECRET") if not _env(name)
        ]
        if missing:
            raise SystemExit(
                "The cua-cloud sandbox requires a per-USER cyclops-cs key. Set "
                f"{' and '.join(missing)} (an OAuth2 client-credentials pair "
                "whose client id starts with 'ukey-', from POST /api/user-keys). "
                "Override CUA_TOKEN_URL / CUA_BASE_URL if your pool is not on the "
                "defaults."
            )
        if not _env("BENCHFLOW_CUA_CLOUD_POOL"):
            raise SystemExit(
                "The cua-cloud sandbox requires a pool name. Set "
                "BENCHFLOW_CUA_CLOUD_POOL to the warm-pool namespace to claim "
                "from (the claim template is '<pool>-template')."
            )

    def _validate_definition(self) -> None:
        # The cyclops pool decides the guest OS via its template; BenchFlow does
        # not pick an image here. Any task OS is accepted — the pool, not the
        # task config, governs the desktop. (No-op validation, declared so the
        # abstract method is satisfied.)
        return None

    @property
    def sandbox_id(self) -> str | None:
        """The bound cyclops sandbox name (the provider-side identifier)."""
        return self._sandbox

    @property
    def pool(self) -> str | None:
        """The warm-pool namespace this sandbox claims from."""
        return self._pool

    @property
    def claim_name(self) -> str | None:
        """The OSGymSandboxClaim CR name BenchFlow owns, or ``None``."""
        return self._claim_name

    # ---- claim CR plumbing ------------------------------------------------

    def _require_pool(self) -> str:
        if not self._pool:
            raise SandboxStartupError(
                "cua-cloud sandbox has no pool. Set BENCHFLOW_CUA_CLOUD_POOL or "
                "pass pool=... to claim a warm sandbox."
            )
        return self._pool

    @staticmethod
    def _claims_url(pool: str, name: str | None = None) -> str:
        base = (
            f"/api/k8s/apis/{_CLAIM_GROUP}/{_CLAIM_VERSION}"
            f"/namespaces/{pool}/{_CLAIM_PLURAL}"
        )
        return f"{base}/{name}" if name else base

    def _service_base(self, service: str) -> str:
        """Reverse-proxy base path for a service on the bound sandbox."""
        pool = self._require_pool()
        sandbox = self._require_sandbox()
        return f"/api/svc/{pool}/{sandbox}-{service}"

    def _http(self) -> Any:
        """Return a freshly-tokened authed httpx client (auto-refreshing)."""
        if self._http_factory is None:
            raise RuntimeError("cua-cloud sandbox is not started")
        return self._http_factory()

    def _require_sandbox(self) -> str:
        if self._sandbox is None:
            raise RuntimeError("cua-cloud sandbox is not started")
        return self._sandbox

    def _make_claim_name(self) -> str:
        return (
            f"{_CLAIM_NAME_PREFIX}{_name_part(self.environment_name)}-{uuid4().hex[:8]}"
        )[:63].strip("-") or f"{_CLAIM_NAME_PREFIX}task"

    def _create_claim(self, http: Any, pool: str, name: str) -> None:
        resp = http.post(
            self._claims_url(pool),
            json={
                "apiVersion": f"{_CLAIM_GROUP}/{_CLAIM_VERSION}",
                "kind": _CLAIM_KIND,
                "metadata": {"name": name},
                "spec": {"sandboxTemplateRef": {"name": f"{pool}-template"}},
            },
        )
        if resp.status_code == 403:
            raise SandboxStartupError(
                f"403 creating claim on pool {pool!r} — this looks like a "
                "per-pool key. Claims need a per-USER key (client id starting "
                "with 'ukey-', from POST /api/user-keys); /api/k8s impersonates "
                "the token owner so tenant RBAC applies.",
                sandbox_state="claim-forbidden",
            )
        resp.raise_for_status()

    def _wait_bound(self, http_factory: Any, pool: str, name: str) -> str:
        """Poll the claim until ``status.phase == Bound``; return the sandbox name."""
        timeout = _env_float("BENCHFLOW_CUA_CLOUD_BIND_TIMEOUT_SEC", 300.0)
        poll = _env_float("BENCHFLOW_CUA_CLOUD_BIND_POLL_SEC", 2.0)
        deadline = time.monotonic() + timeout
        while True:
            resp = http_factory().get(self._claims_url(pool, name))
            resp.raise_for_status()
            status = resp.json().get("status") or {}
            phase = status.get("phase", "Pending")
            sandbox = (status.get("sandbox") or {}).get("name")
            if phase == "Bound" and sandbox:
                return str(sandbox)
            if phase == "Failed":
                raise SandboxStartupError(
                    f"cua-cloud claim {name!r} on pool {pool!r} failed to bind: "
                    f"{status}",
                    sandbox_state="claim-failed",
                )
            if time.monotonic() >= deadline:
                raise SandboxStartupError(
                    f"cua-cloud claim {name!r} did not reach 'Bound' within "
                    f"{timeout:.0f}s (last phase: {phase!r})",
                    sandbox_state="bind-timeout",
                )
            time.sleep(poll)

    def _wait_service_ready(self, http_factory: Any, url: str) -> Any:
        """GET ``url`` until the in-guest service answers with a non-proxy code.

        While the guest OS boots after the bind, the reverse proxy returns
        502/503/504; anything else means the in-guest service is answering.
        """
        timeout = _env_float("BENCHFLOW_CUA_CLOUD_READY_TIMEOUT_SEC", 420.0)
        poll = _env_float("BENCHFLOW_CUA_CLOUD_READY_POLL_SEC", 15.0)
        deadline = time.monotonic() + timeout
        last_status = 0
        while True:
            resp = http_factory().get(url)
            last_status = resp.status_code
            if resp.status_code not in (502, 503, 504):
                return resp
            if time.monotonic() >= deadline:
                raise SandboxStartupError(
                    f"cua-cloud service at {url!r} did not become ready within "
                    f"{timeout:.0f}s (last proxy status: HTTP {last_status})",
                    sandbox_id=self._sandbox,
                    sandbox_state="service-not-ready",
                )
            time.sleep(poll)

    async def start(self, force_build: bool = False) -> None:
        TrainClient = _load_train_client()
        pool = self._require_pool()

        client_id = _env("CUA_CLIENT_ID")
        client_secret = _env("CUA_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise SandboxStartupError(
                "cua-cloud sandbox requires CUA_CLIENT_ID and CUA_CLIENT_SECRET "
                "(a per-user cyclops-cs key); see preflight()."
            )

        try:
            client = TrainClient.from_key(
                token_url=_env("CUA_TOKEN_URL", _DEFAULT_TOKEN_URL),
                client_id=client_id,
                client_secret=client_secret,
                base_url=_env("CUA_BASE_URL", _DEFAULT_BASE_URL),
            )
        except Exception as exc:  # surface a structured startup error
            raise SandboxStartupError(
                "cua-cloud sandbox could not authenticate against the "
                "cyclops-cs realm. Check CUA_CLIENT_ID/CUA_CLIENT_SECRET and "
                f"CUA_TOKEN_URL ({_env('CUA_TOKEN_URL', _DEFAULT_TOKEN_URL)!r})."
            ) from exc

        # get_httpx_client is a factory: call it per request for a fresh token.
        self._http_factory = client.get_httpx_client

        # Track the claim name *before* creating it so a create/bind failure
        # still releases the claim in stop() — no leak on partial start.
        claim_name = self._make_claim_name()
        self._claim_name = claim_name

        self._create_claim(self._http(), pool, claim_name)
        self._sandbox = self._wait_bound(self._http_factory, pool, claim_name)

        # Wait for the cua-driver control-plane service to answer before we hand
        # the sandbox back — the MCP endpoint is what callers drive.
        driver_root = f"{self._service_base(_DRIVER_SERVICE)}/"
        self._wait_service_ready(self._http_factory, driver_root)

    async def stop(self, delete: bool = True) -> None:
        # Release == DELETE the claim CR. Best-effort and idempotent: a missing
        # or already-released claim must not raise during cleanup, and we clear
        # state first so a failure mid-teardown cannot strand a half-released
        # instance. ``delete=False`` keeps the claim (and warm sandbox) alive.
        claim_name = self._claim_name
        pool = self._pool
        self._sandbox = None
        if not delete:
            return
        self._claim_name = None
        if claim_name is None or pool is None or self._http_factory is None:
            return
        try:
            self._http().delete(self._claims_url(pool, claim_name))
        except Exception as exc:  # cleanup is best-effort
            self.logger.warning(
                "cua-cloud: best-effort release of claim %r on pool %r failed: %s",
                claim_name,
                pool,
                exc,
            )

    # ---- control plane: cua-driver MCP ------------------------------------

    def driver_mcp_url(self) -> str:
        """Absolute-path URL of the bound sandbox's cua-driver MCP endpoint.

        Desktop actions are driven by POSTing JSON-RPC to this Streamable-HTTP
        endpoint (see :meth:`mcp_handshake`). The path is relative to the authed
        httpx client's ``base_url`` (``CUA_BASE_URL``).
        """
        return f"{self._service_base(_DRIVER_SERVICE)}/mcp"

    def mcp_handshake(self) -> dict[str, Any]:
        """Run the cua-driver MCP handshake and return discovered metadata.

        Streamable-HTTP MCP: POST ``initialize`` (reading the ``mcp-session-id``
        response header), POST ``notifications/initialized`` with that session,
        then POST ``tools/list``. SSE responses carry the JSON-RPC payload on
        ``data:`` lines, so we parse the first such line. Returns
        ``{"session_id": ..., "tools": [...]}``.
        """
        url = self.driver_mcp_url()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        resp = self._http().post(
            url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "benchflow-cua-cloud", "version": "0.1.0"},
                },
            },
        )
        resp.raise_for_status()
        session = resp.headers.get("mcp-session-id")
        if session:
            headers["Mcp-Session-Id"] = session

        self._http().post(
            url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        listing = self._http().post(
            url,
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        listing.raise_for_status()
        tools = [
            t.get("name") for t in _parse_mcp_result(listing.text).get("tools", [])
        ]
        return {"session_id": session, "tools": tools}

    async def capabilities(self) -> dict[str, Any]:
        """Probe the bound sandbox's control plane and return its metadata.

        The cyclops pool is a desktop environment, so the capability surface is
        the cua-driver MCP tool set rather than a shell. Returns the pool,
        bound sandbox name, the MCP endpoint, and the discovered MCP session +
        tool names — what an adapter needs to decide how to drive the desktop.
        """
        handshake = self.mcp_handshake()
        return {
            "provider": "cua-cloud",
            "pool": self._pool,
            "sandbox": self._sandbox,
            "claim_name": self._claim_name,
            "control_plane": "cua-driver-mcp",
            "mcp_url": self.driver_mcp_url(),
            "mcp_session_id": handshake.get("session_id"),
            "tools": handshake.get("tools", []),
        }

    # ---- honestly-unsupported shell-shaped BaseSandbox ops ----------------
    #
    # The cyclops pool is a desktop driven over the cua-driver MCP control
    # plane, not a Unix shell. There is no in-guest ``/bin/sh`` to run commands
    # in and no base64-over-shell file channel. Rather than fake a shell that
    # does not exist (and silently "succeed"), every shell-shaped op raises a
    # structured NotImplementedError naming the MCP control plane — the same
    # honest pattern macos_ios_simulator.py uses for in-guest paths simctl
    # cannot serve. The old SDK-cloud path's ``cloud-computer-server-cmd-404``
    # is surfaced so callers know that failure is superseded by this provider.

    def _unsupported(self, op: str) -> NotImplementedError:
        return NotImplementedError(
            f"cua-cloud {op}: the cyclops warm-pool is a desktop environment "
            f"driven over the cua-driver MCP control plane ({self.driver_mcp_url()} "
            "once started), not a Unix shell. There is no in-guest /bin/sh for "
            f"command exec or base64-over-shell file transfer. (The old "
            f"SDK-cloud command endpoint that returned {_SUPERSEDED_CMD_404!r} is "
            "superseded by this claim-based provider.) Drive the desktop via "
            "mcp_handshake()/driver_mcp_url() and the cua-driver MCP tools "
            "instead; use capabilities() to enumerate them."
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        raise self._unsupported("exec")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        raise self._unsupported("upload_file")

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        raise self._unsupported("upload_dir")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        raise self._unsupported("download_file")

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        raise self._unsupported("download_dir")


def _parse_mcp_result(text: str) -> dict[str, Any]:
    """Extract the JSON-RPC ``result`` object from an MCP HTTP response body.

    Streamable-HTTP MCP servers may answer either with a bare JSON body or an
    SSE stream whose JSON-RPC payload rides on ``data:`` lines; handle both.
    """
    payload = next(
        (
            line[len("data:") :].strip()
            for line in text.splitlines()
            if line.startswith("data:")
        ),
        text,
    )
    try:
        return json.loads(payload).get("result", {}) or {}
    except (json.JSONDecodeError, AttributeError):
        return {}

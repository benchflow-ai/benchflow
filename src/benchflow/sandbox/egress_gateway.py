"""A task's denylist on Modal, held at the egress gateway (the controller's side).

A Modal sandbox (gVisor) has no netfilter, so the uid firewall that holds the
agent to the loopback egress proxy on Docker cannot go in. Modal does enforce,
outside the sandbox and for root too, a create-time
``outbound_cidr_allowlist``. With ``BENCHFLOW_EGRESS_GATEWAY`` naming a
running gateway's control socket (:mod:`benchflow.sandbox.egress_gateway_server`),
a Modal sandbox whose own network is the task's ``denylist`` is *held*:

- it is created with ``outbound_cidr_allowlist = [<gateway IPv4>/32]``
  (:func:`create_kwargs`), so no process in it reaches any other address;
- once started (:func:`attach`), its rollout is registered at the gateway,
  which runs BenchFlow's egress proxy for it twice (``open``, no policy, and
  ``deny``, the task's), the proxy's CA is trusted in the sandbox, and root's
  relay (``_egress_gateway_relay.py``) answers on ``127.0.0.1:18628``, the
  proxy's port, and splices to the gateway after an HMAC challenge. The proxy
  variables go into the sandbox's persistent env, and root proves that a
  direct connection fails, the relay answers and a URL the task blocks is
  refused in ``deny``. Any failure stops the sandbox before the agent;
- the rollout's phase follows BenchFlow's own steps: ``deny`` from each agent
  connection's egress start (:func:`deny`), for the oracle's turn and for a
  ``denylist`` verifier; ``open`` before that (installs) and for a ``public``
  verifier. Where Docker adds the agent's uid rules, a held sandbox proves
  its agent confined (:func:`enforce_agent`);
- stopping it deregisters the rollout and writes ``modal-egress.json`` in the
  trial folder (:func:`detach`).

Every process of a held sandbox is under the phase's policy, root's model
proxy included; the policies a gateway cannot hold stay refused before any
sandbox (``task.runtime_capabilities``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandbox import egress_gateway_server as server

logger = logging.getLogger(__name__)

#: The control socket of the gateway that holds Modal denylists (unset: none).
ENV = "BENCHFLOW_EGRESS_GATEWAY"
#: Where a held sandbox keeps its :class:`Hold`, and the gateway's answer to
#: ``hello`` between the sandbox's create and its attach.
STATE_ATTR = "_benchflow_egress_gateway"
_PENDING_ATTR = "_benchflow_egress_gateway_pending"
#: The trial folder's record of a held sandbox.
RECORD_NAME = "modal-egress.json"
#: Root's folder for the relay and its configuration (mode 700).
RELAY_DIR = "/opt/benchflow-egress-relay"
RELAY_SOURCE = Path(__file__).with_name("_egress_gateway_relay.py")
#: ``OracleEgress.match`` and ``VerifierNetwork.proxy`` of a held sandbox.
ORACLE_MATCH = "every process (egress gateway)"
PROXY = "egress gateway"
#: Public addresses that accept TCP on 443: a failed connect to them is the
#: allowlist's refusal, not the endpoint's.
_DIRECT_V4 = ("1.1.1.1", 443)
_DIRECT_V6 = ("2606:4700:4700::1111", 443)
HEALTH_TIMEOUT_SEC = 60
#: Longer than the gateway can take, so a registration never completes after
#: attach gave up on it.
REGISTER_TIMEOUT_SEC = server.register_max_sec() + 60.0
#: A rollout lives at the gateway the sandbox's lifetime plus this.
TTL_MARGIN_SEC = 3600
_ROLLOUT_ID_MAX = 128
#: Whether ``hello`` may name a loopback address (the tests' gateway).
_LOOPBACK_PUBLIC_OK = False
#: IETF protocol assignments, not global but for these two anycast addresses
#: (IANA); named since cpython before 3.12.4 classed most of the block global.
_IETF_PROTOCOL = ipaddress.IPv4Network("192.0.0.0/24")
_IETF_GLOBAL = (ipaddress.IPv4Address("192.0.0.9"), ipaddress.IPv4Address("192.0.0.10"))


class EgressGatewayError(RuntimeError):
    """The egress gateway refused a request, did not answer, or could not hold a sandbox."""


def control_socket(environ: Any = None) -> str | None:
    """``BENCHFLOW_EGRESS_GATEWAY``, or None when unset or empty."""
    env = os.environ if environ is None else environ
    value = (env.get(ENV) or "").strip()
    return value or None


def configured(environ: Any = None) -> bool:
    """Whether a gateway is named for this run."""
    return control_socket(environ) is not None


def wanted(config: Any) -> bool:
    """Whether a sandbox created with *config* is held: a gateway is named,
    and the sandbox's own network is the task's denylist with internet."""
    return (
        configured()
        and getattr(config, "network_mode", None) == "denylist"
        and getattr(config, "allow_internet", True) is not False
    )


@dataclass
class Hold:
    """One held sandbox's place at the gateway (kept on the sandbox)."""

    control: str
    public: str
    rollout: str
    token: str = field(repr=False)
    gateway_started_at: float | None = None
    sandbox_id: str | None = None
    phase: str = "open"
    events: list[dict[str, Any]] = field(default_factory=list)
    proofs: list[dict[str, Any]] = field(default_factory=list)
    #: How far each of the gateway's logs was read (``log`` offsets).
    offsets: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(server.LOGS, 0)
    )
    #: The deny log's offset when the verifier's network was armed.
    verifier_from: int | None = None
    start_proof_log: list[str] = field(default_factory=list)
    open_refusals: int = 0

    @property
    def allowlist(self) -> list[str]:
        return [f"{server.parse_address(self.public)[0]}/32"]

    def note(self, event: str, **fields: Any) -> None:
        self.events.append({"ts": round(time.time(), 3), "event": event, **fields})

    def record(self) -> dict[str, Any]:
        """The trial folder's record (the token never in it)."""
        return {
            "gateway": self.public,
            "outbound_cidr_allowlist": self.allowlist,
            "rollout": self.rollout,
            "modal_sandbox_id": self.sandbox_id,
            "phase": self.phase,
            "gateway_started_at": self.gateway_started_at,
            "events": self.events,
            "proofs": self.proofs,
            "start_proof_log": self.start_proof_log,
            "open_phase_refusals": self.open_refusals,
        }


def held(env: Any) -> Hold | None:
    """The sandbox's :class:`Hold`, None for every sandbox the gateway does not hold.

    Read from the instance's own attributes, so a mocked sandbox, which
    answers any attribute, never looks held."""
    try:
        value = vars(env).get(STATE_ATTR)
    except TypeError:
        return None
    return value if isinstance(value, Hold) else None


def rollout_id(session_id: str) -> str:
    """*session_id* as a gateway rollout id ([A-Za-z0-9._-], 128 at most, not
    only dots); any other is cleaned and ends in a hash of the whole."""
    name = re.sub(r"[^A-Za-z0-9._-]", "-", session_id) or "rollout"
    if name == session_id and len(name) <= _ROLLOUT_ID_MAX and name.strip("."):
        return name
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    return f"{name[: _ROLLOUT_ID_MAX - len(digest) - 1]}-{digest}"


def call(
    control: str, payload: dict[str, Any], timeout: float = 60.0
) -> dict[str, Any]:
    """One control request; EgressGatewayError when it fails or is refused."""
    try:
        reply = server.request(control, payload, timeout=timeout)
    except (OSError, ValueError) as exc:
        raise EgressGatewayError(
            f"egress gateway at {control}: {type(exc).__name__}: {exc}"
        ) from exc
    if not reply.get("ok"):
        raise EgressGatewayError(
            f"egress gateway at {control} refused {payload.get('command')!r}: "
            f"{reply.get('error', reply)}"
        )
    return reply


async def acall(
    control: str, payload: dict[str, Any], timeout: float = 60.0
) -> dict[str, Any]:
    """:func:`call` off the event loop."""
    return await asyncio.to_thread(call, control, payload, timeout)


async def hello(control: str) -> dict[str, Any]:
    """The gateway's ``hello``: its protocol and a public ``IPv4:port``."""
    reply = await acall(control, {"command": "hello"}, timeout=10.0)
    if reply.get("protocol") != server.PROTOCOL:
        raise EgressGatewayError(
            f"egress gateway at {control} speaks {reply.get('protocol')!r}, "
            f"not {server.PROTOCOL}"
        )
    public = str(reply.get("public"))
    try:
        host, _ = server.parse_address(public)
        address = ipaddress.IPv4Address(host)
    except ValueError as exc:
        raise EgressGatewayError(
            f"egress gateway at {control} names {public!r}: a Modal allowlist "
            "needs an IPv4 address and a port"
        ) from exc
    # The allowlist admits the address on every port: a private, shared,
    # loopback, link-local, multicast or reserved one would open a host that
    # is not the gateway, such as a provider-internal one.
    reserved = address in _IETF_PROTOCOL and address not in _IETF_GLOBAL
    if not (address.is_global and not address.is_multicast and not reserved) and not (
        _LOOPBACK_PUBLIC_OK and address.is_loopback
    ):
        raise EgressGatewayError(
            f"egress gateway at {control} names {public!r}: a Modal allowlist "
            "needs the gateway's global IPv4 address, not a private, shared, "
            "loopback, link-local, multicast or reserved one"
        )
    return reply


async def create_kwargs(env: Any) -> dict[str, Any]:
    """``Sandbox.create``'s allowlist for a sandbox the gateway will hold
    (the gateway's address alone), {} for any other. Asks the gateway first:
    one that does not answer fails the create."""
    if not wanted(getattr(env, "task_env_config", None)):
        return {}
    control = control_socket()
    assert control is not None
    reply = await hello(control)
    public = str(reply["public"])
    vars(env)[_PENDING_ATTR] = (control, public, reply.get("started_at"))
    return {"outbound_cidr_allowlist": [f"{server.parse_address(public)[0]}/32"]}


def _policy(denylist: Any) -> dict[str, Any]:
    """The ``policy.json`` the gateway's deny proxy loads for *denylist*."""
    from benchflow.sandbox.egress_denylist import _policy_record

    if (
        getattr(denylist, "allow_mode", False)
        or getattr(denylist, "native_claude_model_only", False)
        or getattr(denylist, "native_claude_model_origin", False)
        or getattr(denylist, "native_codex_model_only", False)
    ):
        raise EgressGatewayError(
            "the egress gateway holds network_mode='denylist' only, not "
            f"{getattr(denylist, 'mode', 'this policy')!r} or a native model transport"
        )
    return _policy_record(denylist, None)


def proxy_env() -> dict[str, str]:
    """The proxy and CA variables, without the agent's markers (the oracle's set)."""
    from benchflow.sandbox.egress_denylist import EgressDenylist, denylist_agent_env
    from benchflow.sandbox.oracle_egress import _AGENT_ONLY_ENV

    return {
        k: v
        for k, v in denylist_agent_env({}, EgressDenylist((), ())).items()
        if k not in _AGENT_ONLY_ENV
    }


#: Root's probe: a direct connect to two public addresses, then the relay's
#: health (the phase proxy's ``/healthz``).
_PROBE_PY = """\
import json, socket, urllib.request
out = {}
for key, family, address in (
    ("direct", socket.AF_INET, %r),
    ("direct6", socket.AF_INET6, %r),
):
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(address)
        out[key] = "connected"
    except OSError as exc:
        out[key] = type(exc).__name__
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    out["relay"] = opener.open("http://127.0.0.1:%d/healthz", timeout=10).status
except Exception as exc:
    out["relay"] = type(exc).__name__
print("BF egress-gateway-probe " + json.dumps(out))
"""

#: Root's request for a URL the task blocks, through the relay, trusting the
#: rollout's CA (a path rule is judged inside the intercepted TLS).
_DENY_PY = """\
import json, ssl, urllib.error, urllib.request
url = %r
proxy = "http://127.0.0.1:%d"
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
    urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=%r)),
)
out = {"url": url}
try:
    out["status"] = opener.open(url, timeout=30).status
except urllib.error.HTTPError as exc:
    out["status"] = exc.code
    out["blocked"] = exc.headers.get("X-BenchFlow-Blocked")
except Exception as exc:
    out["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
print("BF egress-gateway-deny " + json.dumps(out))
"""


def probe_cmd() -> str:
    """The confinement probe as root's shell text (root's protected Python)."""
    from benchflow._root_python import root_python_sh
    from benchflow.sandbox.egress_denylist import EGRESS_PORT

    return root_python_sh(_PROBE_PY % (_DIRECT_V4, _DIRECT_V6, EGRESS_PORT))


def deny_probe_cmd(url: str) -> str:
    from benchflow._root_python import root_python_sh
    from benchflow.sandbox.egress_denylist import CA_BUNDLE_PATH, EGRESS_PORT

    return root_python_sh(_DENY_PY % (url, EGRESS_PORT, CA_BUNDLE_PATH))


def marked(stdout: str, marker: str) -> dict[str, Any] | None:
    """The JSON object after *marker* on a line of *stdout*, else None."""
    for line in (stdout or "").splitlines():
        if line.startswith(marker + " "):
            with contextlib.suppress(ValueError):
                value = json.loads(line[len(marker) + 1 :])
                return value if isinstance(value, dict) else None
            return None
    return None


def confined(result: dict[str, Any] | None) -> bool:
    """A probe that proves confinement: neither direct connect got through,
    and the relay answered 200."""
    return (
        result is not None
        and result.get("direct") != "connected"
        and result.get("direct6") != "connected"
        and result.get("relay") == 200
    )


def denied_probe_url(denylist: Any) -> str | None:
    """A URL *denylist* blocks, or None when it lists no plain one: the first
    ``blocked_urls`` entry, else the first ``blocked_hosts`` entry; wildcard
    entries are skipped."""
    for entry in getattr(denylist, "blocked_urls", ()) or ():
        text = str(entry).strip()
        if text and "*" not in text:
            return text if "://" in text else f"https://{text}"
    for entry in getattr(denylist, "blocked_hosts", ()) or ():
        text = str(entry).strip().strip(".")
        if text and "*" not in text and "/" not in text:
            return f"https://{text}/"
    return None


def refused(result: dict[str, Any] | None) -> bool:
    """A deny probe the proxy refused: its 403 inside the intercepted TLS
    (``X-BenchFlow-Blocked: 1``), or a refused CONNECT."""
    if not result:
        return False
    if result.get("status") == 403 and result.get("blocked") == "1":
        return True
    return "Tunnel connection failed: 403" in str(result.get("error", ""))


async def _root(env: Any, command: str, timeout_sec: int = 120) -> tuple[int, str, str]:
    from benchflow.sandbox._exec_output import exec_raw

    result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout_sec)
    code = getattr(result, "return_code", None)
    return (
        code if isinstance(code, int) else -1,
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )


def _relay_start_cmd() -> str:
    """Root shell: the relay started on root's interpreter, picked as the
    egress proxy's is (the kit's Python once the kit exists, else the
    image's python3), in its own session."""
    from benchflow._root_python import FLAGS, root_python_pick_sh

    q = shlex.quote
    d = RELAY_DIR
    owner = "The egress gateway relay"
    image_py = (
        'PY="$(command -v python3 || true)"; '
        f'[ -n "$PY" ] || {{ echo "{owner} needs python3 in the task image" >&2; exit 87; }}'
    )
    no_kit_py = f'echo "{owner} needs python3 in the protected runtime" >&2; exit 87'
    return (
        "set -e; cd /; "
        f"chmod 700 {q(d)}; chmod 600 {q(d)}/relay.json {q(d)}/relay.py; "
        + root_python_pick_sh("PY", fallback="", refuse=no_kit_py)
        + f'; [ -n "$PY" ] || {{ {image_py}; }}; '
        f'if [ -s {q(d)}/relay.pid ]; then kill -TERM "$(cat {q(d)}/relay.pid)" '
        "2>/dev/null || true; fi; "
        "if command -v setsid >/dev/null 2>&1; then S=setsid; else S=; fi; "
        f'$S nohup "$PY" {" ".join(FLAGS)} {q(d)}/relay.py {q(d)}/relay.json '
        f"</dev/null >{q(d)}/relay.log 2>&1 & echo $! > {q(d)}/relay.pid"
    )


async def _install(env: Any, hold: Hold, ca_cert: bytes) -> None:
    """The rollout's CA trusted where the proxy variables name it, then the
    relay uploaded (root's alone) and started."""
    from benchflow.sandbox.egress_denylist import (
        RUNTIME_DIR,
        _ca_only_cmd,
        _run,
        _runtime_dir_cmd,
        _upload,
    )
    from benchflow.trust.state import TRUSTED_TRANSFER, admit

    await _run(
        env, _runtime_dir_cmd(env, RUNTIME_DIR), "egress runtime dir", timeout_sec=30
    )
    await _upload(env, {"ca.crt": ca_cert})
    await _run(env, _ca_only_cmd(), "egress CA install", timeout_sec=120)
    code, _, err = await _root(
        env, f"mkdir -p {RELAY_DIR} && chmod 700 {RELAY_DIR}", timeout_sec=30
    )
    if code != 0:
        raise EgressGatewayError(f"cannot make {RELAY_DIR} (rc={code}): {err[-500:]}")
    from benchflow.sandbox.egress_denylist import EGRESS_PORT

    config = {
        "gateway": hold.public,
        "rollout": hold.rollout,
        "token": hold.token,
        "port": EGRESS_PORT,
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "relay.py").write_bytes(RELAY_SOURCE.read_bytes())
        (Path(tmp) / "relay.json").write_text(json.dumps(config))
        await env.upload_dir(
            tmp,
            RELAY_DIR,
            admission=admit(
                env, "main", TRUSTED_TRANSFER, reason="BenchFlow's egress gateway relay"
            ),
        )
    code, _, err = await _root(env, _relay_start_cmd(), timeout_sec=120)
    if code != 0:
        raise EgressGatewayError(f"the relay did not start (rc={code}): {err[-1000:]}")


async def _wait_relay(env: Any) -> dict[str, Any] | None:
    """Root's probe, again until the relay answers or the time is up."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
    while True:
        _, out, _ = await _root(env, probe_cmd(), timeout_sec=30)
        result = marked(out, "BF egress-gateway-probe")
        if (result and result.get("relay") == 200) or time.monotonic() >= deadline:
            return result
        await asyncio.sleep(1.0)


#: The most ``log`` requests one read makes (each returns up to
#: ``server.MAX_LOG`` bytes of each log).
_MAX_LOG_READS = 64


async def _read_logs(
    hold: Hold, *, advance: tuple[str, ...], start: dict[str, int] | None = None
) -> dict[str, Any]:
    """Every log's lines from the held offsets (or *start*'s) to its end:
    ``texts``, ``offsets`` (the ends), the gateway's ``phase`` and
    ``counters``. The named logs' held offsets move to their ends."""
    offsets = {**hold.offsets, **(start or {})}
    texts = dict.fromkeys(server.LOGS, "")
    reply: dict[str, Any] = {}
    for _ in range(_MAX_LOG_READS):
        reply = await acall(
            hold.control,
            {"command": "log", "rollout": hold.rollout, "offsets": dict(offsets)},
        )
        logs = reply["logs"]
        for name in server.LOGS:
            texts[name] += logs[name]["text"]
            offsets[name] = int(logs[name]["offset"])
        if not any(logs[name]["text"] for name in server.LOGS):
            break
    for name in advance:
        hold.offsets[name] = offsets[name]
    return {
        "texts": texts,
        "offsets": offsets,
        "phase": reply.get("phase"),
        "counters": reply.get("counters", {}),
    }


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


async def _prove_denied(env: Any, hold: Hold, denylist: Any) -> None:
    """In ``deny``, as root, a URL the task blocks is refused; then back to
    ``open``. The probe's log lines are kept out of the trajectory's."""
    url = denied_probe_url(denylist)
    if url is None:
        hold.note("deny_probe_skipped", why="the policy lists no plain URL or host")
        return
    await deny(env, denylist, at="start proof")
    code, out, err = await _root(env, deny_probe_cmd(url), timeout_sec=90)
    result = marked(out, "BF egress-gateway-deny")
    hold.proofs.append(
        {"user": "root", "at": "start, deny", **(result or {"rc": code})}
    )
    if not refused(result):
        raise EgressGatewayError(
            f"the task's policy did not refuse {url} at the gateway "
            f"({result or err[-500:]})"
        )
    read = await _read_logs(hold, advance=("deny", "deny_allowed"))
    hold.start_proof_log = _lines(read["texts"]["deny"])
    await open_phase(env, "start proof done")


async def attach(env: Any) -> Hold | None:
    """Put a started sandbox the gateway holds behind its relay and prove it.

    None for a sandbox the gateway does not hold. EgressGatewayError when any
    step fails: the caller stops the sandbox (its stop deregisters), and
    nothing of the task runs unconfined.
    """
    if not wanted(getattr(env, "task_env_config", None)):
        return None
    pending = vars(env).pop(_PENDING_ATTR, None)
    if pending is None:
        raise EgressGatewayError(
            "the sandbox was created without the gateway's allowlist "
            f"({ENV} was set after its create)"
        )
    control, public, started = pending
    from benchflow.sandbox.egress_denylist import _runtime_files, egress_denylist_for

    denylist = egress_denylist_for(env.task_env_config)
    if denylist is None:
        raise EgressGatewayError("the sandbox's network declares no denylist")
    _policy(denylist)
    files = await asyncio.to_thread(_runtime_files, denylist, None)
    hold = Hold(
        control,
        public,
        rollout_id(str(env.session_id)),
        secrets.token_hex(32),
        gateway_started_at=started,
        sandbox_id=getattr(getattr(env, "_sandbox", None), "object_id", None),
    )
    lifetime = getattr(env, "_sandbox_timeout", None)
    ttl = server.DEFAULT_TTL_SEC
    if isinstance(lifetime, int) and lifetime > 0:
        ttl = min(lifetime + TTL_MARGIN_SEC, server.MAX_TTL_SEC)
    registering = asyncio.ensure_future(
        acall(
            control,
            {
                "command": "register",
                "rollout": hold.rollout,
                "token": hold.token,
                "ttl_sec": ttl,
                "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
            },
            timeout=REGISTER_TIMEOUT_SEC,
        )
    )
    try:
        await asyncio.shield(registering)
    except asyncio.CancelledError:
        # The request runs on in its thread: wait for the gateway's answer,
        # so the stop deregisters a rollout it registered.
        with contextlib.suppress(Exception):
            await registering
            vars(env)[STATE_ATTR] = hold
        raise
    # From here the sandbox's stop deregisters it.
    vars(env)[STATE_ATTR] = hold
    hold.note("registered", phase="open", ttl_sec=ttl)
    await _install(env, hold, files["ca.crt"])
    result = await _wait_relay(env)
    hold.proofs.append({"user": "root", "at": "start", **(result or {})})
    if not confined(result):
        _, log, _ = await _root(env, f"tail -c 2000 {RELAY_DIR}/relay.log", 30)
        raise EgressGatewayError(
            f"the sandbox is not confined to the gateway at start ({result}); "
            f"relay log: {log[-1000:]}"
        )
    env._persistent_env = {
        **(getattr(env, "_persistent_env", None) or {}),
        **proxy_env(),
    }
    await _prove_denied(env, hold, denylist)
    return hold


async def start_hold(env: Any) -> None:
    """:func:`attach` at the end of a Modal sandbox's start; on any failure
    the sandbox is stopped (and deregistered), then the error is raised."""
    try:
        await attach(env)
    except BaseException:
        with contextlib.suppress(Exception):
            await env.stop(True)
        raise


async def deny(env: Any, denylist: Any, *, at: str) -> None:
    """The rollout's new connections reach *denylist*'s proxy; connections
    opened in ``open`` are cut."""
    hold = held(env)
    assert hold is not None
    reply = await acall(
        hold.control,
        {"command": "deny", "rollout": hold.rollout, "policy": _policy(denylist)},
    )
    hold.phase = "deny"
    hold.note(
        "deny", at=at, cut=reply.get("cut", 0), policy_sha256=reply.get("policy_sha256")
    )


async def open_phase(env: Any, why: str) -> None:
    """The rollout's new connections reach the open proxy (no policy)."""
    hold = held(env)
    assert hold is not None
    await acall(hold.control, {"command": "open", "rollout": hold.rollout})
    hold.phase = "open"
    hold.note("open", why=why)


async def prove_confined(env: Any, user: str, *, at: str) -> dict[str, Any]:
    """As *user*, a direct connection fails and the relay answers, and the
    gateway holds the rollout in ``deny``. RuntimeError when not."""
    from benchflow.sandbox.lockdown import (
        _cloud_metadata_agent_exec,
        _cloud_metadata_exec,
    )

    hold = held(env)
    assert hold is not None
    if user in ("root", "0"):
        out, why = await _cloud_metadata_exec(
            env, probe_cmd(), user="root", timeout_sec=60
        )
    else:
        # The agent's own verified privilege drop, as the metadata probe's.
        out, why = await _cloud_metadata_agent_exec(
            env, probe_cmd(), user, timeout_sec=60
        )
    result = (
        marked(str(getattr(out, "stdout", "") or ""), "BF egress-gateway-probe")
        if out is not None
        else None
    )
    reply = await _read_logs(hold, advance=())
    proof = {
        "user": user,
        "at": at,
        "phase": reply.get("phase"),
        **(result or {"exec_error": why}),
    }
    hold.proofs.append(proof)
    if not confined(result) or reply.get("phase") != "deny":
        raise RuntimeError(
            f"egress gateway: {user!r} is not held in the deny phase at {at} "
            f"(gateway phase {reply.get('phase')!r}, probe {result or why})"
        )
    return proof


async def enforce_agent(
    env: Any, sandbox_user: str | None, agent_env: dict[str, str]
) -> None:
    """In place of the agent's uid rules: refuse what the gateway cannot hold
    (no-web, an allowlist), then prove the agent's user confined in ``deny``."""
    from benchflow.sandbox.lockdown import EGRESS_ALLOWLIST_ENV, EGRESS_DENYLIST_ENV

    if agent_env.get("BENCHFLOW_DISALLOW_WEB_TOOLS") == "1":
        raise RuntimeError(
            "the no-web policy holds the agent's uid alone, which the egress "
            "gateway cannot do: run it on Docker"
        )
    if (
        agent_env.get(EGRESS_ALLOWLIST_ENV) == "1"
        or agent_env.get(EGRESS_DENYLIST_ENV) != "1"
    ):
        raise RuntimeError("the egress gateway holds network_mode='denylist' only")
    if not sandbox_user:
        raise RuntimeError("network_mode='denylist' requires a sandbox_user")
    await prove_confined(env, sandbox_user, at="agent launch")


async def collect_logs(env: Any, rollout_dir: Any) -> None:
    """The deny proxy's blocked and allowed lines since the last read into the
    trajectory (``egress_denylist.jsonl``, ``egress_allowed.jsonl``); the open
    proxy's refusals are counted. Never raises."""
    from benchflow.sandbox.egress_denylist import ALLOWED_LOG_NAME, TRAJECTORY_LOG_NAME

    hold = held(env)
    if hold is None:
        return
    try:
        read = await _read_logs(hold, advance=server.LOGS)
    except EgressGatewayError as exc:
        hold.note("log_failed", error=str(exc)[:300])
        return
    hold.open_refusals += len(_lines(read["texts"]["open"]))
    if rollout_dir is None:
        return
    for name, log in (
        (TRAJECTORY_LOG_NAME, "deny"),
        (ALLOWED_LOG_NAME, "deny_allowed"),
    ):
        _write_trajectory(rollout_dir, name, read["texts"][log])


def _write_trajectory(rollout_dir: Any, name: str, text: str) -> None:
    """``trajectory/<name>`` under the rollout's held root; never raises."""
    from benchflow._utils.reachable_write import ReachablePath, destination

    try:
        relative = Path("trajectory", name)
        if isinstance(rollout_dir, ReachablePath):
            target = rollout_dir / relative
        else:
            target = destination(Path(rollout_dir) / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Egress log %s not collected: %s", name, exc)


async def arm_oracle(env: Any, denylist: Any) -> Any:
    """The oracle's turn under *denylist*, every process of the sandbox held."""
    from benchflow.sandbox.oracle_egress import OracleEgress

    await deny(env, denylist, at="oracle")
    return OracleEgress(policy=denylist, match=ORACLE_MATCH, gid=None, env=proxy_env())


async def arm_verifier(env: Any, task: Any, mode: str, *, where: str) -> Any:
    """Hold test.sh to the verifier's ``denylist`` at the gateway and prove it
    as root. ``VerifierNetworkUnheldError`` when it cannot."""
    from benchflow.sandbox.egress_denylist import egress_denylist_for
    from benchflow.sandbox.verifier_network import (
        VerifierNetwork,
        _verifier_policy_config,
    )
    from benchflow.task.verifier_errors import VerifierNetworkUnheldError

    hold = held(env)
    assert hold is not None
    try:
        if mode != "denylist":
            raise RuntimeError(
                f"the egress gateway holds denylist and public, not {mode}"
            )
        denylist = egress_denylist_for(_verifier_policy_config(task.config, mode))
        await deny(env, denylist, at="verifier")
        proof = await prove_confined(env, "root", at="verifier")
        # From the deny log's end: what came before is not test.sh's.
        hold.verifier_from = (await _read_logs(hold, advance=()))["offsets"]["deny"]
    except Exception as exc:
        raise VerifierNetworkUnheldError(
            f"Verifier hardening failed: the verifier's network ({where} under "
            f"{mode}) could not be held at the egress gateway ({exc}); test.sh "
            "runs the agent's code and does not run outside the task's network"
        ) from exc
    return VerifierNetwork(
        mode,
        "held",
        match=None,
        where=where,
        launch=False,
        proxy=PROXY,
        env=proxy_env(),
        proof={k: v for k, v in proof.items() if k not in ("user", "at")},
    )


async def refusals_since_arm(env: Any) -> list[dict[str, Any]]:
    """The deny proxy's log entries since the verifier's arm."""
    hold = held(env)
    if hold is None or hold.verifier_from is None:
        return []
    try:
        read = await _read_logs(hold, advance=(), start={"deny": hold.verifier_from})
    except EgressGatewayError as exc:
        # As an in-sandbox proxy's log that cannot be read: none counted.
        hold.note("log_failed", error=str(exc)[:300])
        return []
    found: list[dict[str, Any]] = []
    for line in _lines(read["texts"]["deny"]):
        with contextlib.suppress(ValueError):
            entry = json.loads(line)
            if isinstance(entry, dict):
                found.append(entry)
    return found


async def detach(env: Any) -> None:
    """Deregister a held sandbox's rollout (its proxies stop, its CA key is
    deleted) and write the trial folder's record. Never raises."""
    hold = held(env)
    if hold is None:
        return
    vars(env).pop(STATE_ATTR, None)
    record: dict[str, Any] = {}
    try:
        read = await _read_logs(hold, advance=server.LOGS)
        hold.open_refusals += len(_lines(read["texts"]["open"]))
        record["gateway_counters"] = read["counters"]
        record["gateway_started_at_stop"] = (await hello(hold.control)).get(
            "started_at"
        )
    except EgressGatewayError as exc:
        hold.note("log_failed", error=str(exc)[:300])
    try:
        await acall(hold.control, {"command": "deregister", "rollout": hold.rollout})
        hold.note("deregistered")
    except EgressGatewayError as exc:
        hold.note("deregister_failed", error=str(exc)[:300])
        logger.warning(
            "egress gateway: rollout %s not deregistered: %s", hold.rollout, exc
        )
    paths = getattr(env, "rollout_paths", None)
    rollout_dir = getattr(paths, "rollout_dir", None)
    if rollout_dir is None:
        return
    try:
        from benchflow._utils.reachable_write import destination

        target = destination(Path(rollout_dir) / RECORD_NAME)
        target.write_text(json.dumps({**hold.record(), **record}, indent=2) + "\n")
    except Exception as exc:
        logger.warning("egress gateway: %s not written: %s", RECORD_NAME, exc)

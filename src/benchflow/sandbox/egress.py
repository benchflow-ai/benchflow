"""Agent egress blocklist — ``network_mode = "blocklist"`` enforcement.

The task keeps full internet access, but a declared list of ``host`` /
``host/path-prefix`` entries must stay unreachable from the agent. Enforcement
is layered because a single mechanism cannot see every path an agent has to
the web:

1. **Sandbox-local filtering proxy** (this module's ``_EGRESS_PROXY_SOURCE``).
   A stdlib-only forward proxy started as root on loopback before the agent
   launches. Plain HTTP requests carry the full URL, so host *and* path rules
   apply. HTTPS ``CONNECT`` tunnels only expose the hostname, so a host rule
   rejects the tunnel outright; a host that carries *path* rules is TLS-
   inspected instead — the proxy terminates TLS with a leaf certificate signed
   by a per-run CA that is installed into the sandbox trust store, reads the
   request path, and forwards allowed requests upstream over a fresh TLS
   connection. The agent user never sees the CA private key.
2. **Agent-UID firewall** (``lockdown._agent_egress_firewall_cmd``). After the
   ACP handshake the agent's UID may only reach loopback, so every tool that
   ignores ``HTTP(S)_PROXY`` fails closed instead of bypassing the filter.
3. **Server-side web tools** live outside the sandbox (Claude Code WebSearch,
   Codex ``web_search``, Gemini grounding). The LiteLLM pre-call hook injects
   Anthropic ``blocked_domains`` and strips OpenAI Responses search tools; the
   per-agent ``blocklist_web_tools_*`` registry knobs disable the rest.

Blocked requests answer ``404 Not Found`` rather than ``403`` so that a
research agent cannot distinguish a hidden paper from a missing one. Every
decision is appended to an egress log the rollout downloads for auditing.

Because the proxy runs as root (outside the agent-UID firewall) it vets every
resolved upstream address and refuses loopback, link-local (cloud metadata),
unspecified, multicast, reserved, and private ranges with ``403`` — it must
never be the hop that turns the blocklist into an SSRF primitive. The only
private addresses reachable are the container's own directly-connected
subnets (read from ``/proc/net/route``), which is where compose side-services
live; ``allow_private_networks`` re-opens every private range for tasks that
genuinely need it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Agent-env marker carrying the JSON rule list. Present in the LiteLLM proxy
#: process env (it drives the server-side tool rewrite) and stripped from the
#: agent's own env so the agent cannot read which URLs are hidden from it.
EGRESS_BLOCKED_URLS_ENV = "BENCHFLOW_EGRESS_BLOCKED_URLS"
#: Agent-env marker that the blocklist policy is active (safe to expose).
EGRESS_POLICY_ENV = "BENCHFLOW_EGRESS_POLICY"
EGRESS_POLICY_BLOCKLIST = "blocklist"

DEFAULT_EGRESS_PROXY_PORT = 61380
EGRESS_RUNTIME_DIR = "/opt/benchflow/egress"
EGRESS_CA_BUNDLE_PATH = f"{EGRESS_RUNTIME_DIR}/ca-bundle.crt"
EGRESS_LOG_PATH = f"{EGRESS_RUNTIME_DIR}/egress.jsonl"
EGRESS_LOG_ARTIFACT_NAME = "egress.jsonl"
#: Status returned for blocked requests — indistinguishable from "no such page".
BLOCKED_STATUS = 404

_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "NODE_USE_ENV_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "GIT_SSL_CAINFO",
)


def _split_rule(rule: str) -> tuple[str, str]:
    host, _, path = rule.partition("/")
    return host.lower(), path.strip("/")


def _host_matches(rule_host: str, host: str) -> bool:
    host = host.lower().rstrip(".")
    return host == rule_host or host.endswith("." + rule_host)


def normalize_request_path(path: str) -> str:
    """Canonical form of a request path for rule matching.

    Mirrors what an upstream HTTP server does before routing, so an agent
    cannot dodge a rule with ``%32%34...`` percent-encoding, ``//`` runs, or
    ``/other/../abs/…`` dot segments: drop the query/fragment, percent-decode,
    collapse with ``posixpath.normpath``, strip slashes, lower-case.
    """
    raw = path.split("?", 1)[0].split("#", 1)[0]
    decoded = _unquote_fully(raw)
    collapsed = posixpath.normpath("/" + decoded).strip("/")
    return "" if collapsed == "." else collapsed.lower()


def _unquote_fully(value: str, *, rounds: int = 4) -> str:
    """Percent-decode until stable (bounded) so a double-encoded separator
    (``%252f`` → ``%2f`` → ``/``) cannot reach an upstream that decodes twice
    while the proxy judged the single-decoded form."""
    for _ in range(rounds):
        decoded = urllib.parse.unquote(value)
        if decoded == value:
            return value
        value = decoded
    return value


def match_blocked(
    rules: tuple[str, ...] | list[str], host: str, path: str
) -> str | None:
    """Return the first rule that blocks ``host`` + ``path``, else ``None``.

    Host rules match the host and its subdomains. Path rules additionally
    require the request path to start with the rule's path prefix (the
    ``host/path-prefix`` contract from ``blocked_urls``). ``path`` is the
    request target without scheme/host; the query string is ignored.
    """
    clean_path = normalize_request_path(path)
    for rule in rules:
        rule_host, rule_path = _split_rule(rule)
        rule_path = rule_path.lower()
        if not _host_matches(rule_host, host):
            continue
        if not rule_path:
            return rule
        if clean_path == rule_path or clean_path.startswith(rule_path):
            return rule
    return None


@dataclass(frozen=True)
class EgressBlocklist:
    """Resolved blocklist for one rollout (already schema-normalized rules)."""

    rules: tuple[str, ...]
    proxy_port: int = DEFAULT_EGRESS_PROXY_PORT

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("EgressBlocklist requires at least one rule")

    @property
    def tls_inspection_hosts(self) -> tuple[str, ...]:
        """Hosts that carry path rules and therefore need TLS inspection."""
        hosts: list[str] = []
        for rule in self.rules:
            host, path = _split_rule(rule)
            if path and host not in hosts:
                hosts.append(host)
        return tuple(hosts)

    @property
    def needs_tls_inspection(self) -> bool:
        return bool(self.tls_inspection_hosts)

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}"

    def to_env_value(self) -> str:
        return json.dumps(list(self.rules))

    @classmethod
    def from_env(cls, env: dict[str, str] | None) -> EgressBlocklist | None:
        raw = (env or {}).get(EGRESS_BLOCKED_URLS_ENV, "")
        if not raw:
            return None
        rules = json.loads(raw)
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            raise ValueError(
                f"{EGRESS_BLOCKED_URLS_ENV} must be a JSON list of strings"
            )
        port = int(
            (env or {}).get("BENCHFLOW_EGRESS_PROXY_PORT", DEFAULT_EGRESS_PROXY_PORT)
        )
        return cls(rules=tuple(rules), proxy_port=port)

    @classmethod
    def from_task_config(cls, config: Any) -> EgressBlocklist | None:
        """Resolve the agent-phase blocklist from a parsed ``TaskConfig``.

        The ``agent`` section overrides ``sandbox`` when it declares its own
        ``network_mode``; otherwise the sandbox policy applies to the agent.
        """
        from benchflow.task.config import NetworkMode

        agent = getattr(config, "agent", None)
        sandbox = getattr(config, "sandbox", None)
        agent_mode = getattr(agent, "network_mode", None) if agent is not None else None
        sandbox_mode = (
            getattr(sandbox, "network_mode", None) if sandbox is not None else None
        )
        if agent_mode is not None:
            section = agent
            if (
                sandbox_mode == NetworkMode.BLOCKLIST
                and agent_mode != NetworkMode.BLOCKLIST
            ):
                # A run-level --block-url lands in the sandbox section; an
                # explicit agent-level mode would silently win and the hidden
                # URLs would be served. Refuse instead of running open.
                raise ValueError(
                    "sandbox.network_mode='blocklist' is overridden by the "
                    f"agent-level network_mode={agent_mode.value!r}; the "
                    "blocklist would not apply to the agent. Drop the agent "
                    "override or set agent.network_mode='blocklist' too."
                )
        else:
            section = sandbox
        if section is None or getattr(section, "network_mode", None) != (
            NetworkMode.BLOCKLIST
        ):
            return None
        rules = tuple(getattr(section, "blocked_urls", None) or ())
        if not rules:
            return None
        return cls(rules=rules)

    def agent_env(self) -> dict[str, str]:
        """Env the *agent* process needs: route through the proxy, trust the CA.

        Deliberately excludes the rule list itself.
        """
        env = {
            EGRESS_POLICY_ENV: EGRESS_POLICY_BLOCKLIST,
            "BENCHFLOW_EGRESS_PROXY_PORT": str(self.proxy_port),
            "HTTP_PROXY": self.proxy_url,
            "HTTPS_PROXY": self.proxy_url,
            "http_proxy": self.proxy_url,
            "https_proxy": self.proxy_url,
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
            # Node's global fetch (undici) ignores proxy env unless told to.
            "NODE_USE_ENV_PROXY": "1",
        }
        if self.needs_tls_inspection:
            env.update(
                {
                    "SSL_CERT_FILE": EGRESS_CA_BUNDLE_PATH,
                    "REQUESTS_CA_BUNDLE": EGRESS_CA_BUNDLE_PATH,
                    "CURL_CA_BUNDLE": EGRESS_CA_BUNDLE_PATH,
                    "NODE_EXTRA_CA_CERTS": EGRESS_CA_BUNDLE_PATH,
                    "GIT_SSL_CAINFO": EGRESS_CA_BUNDLE_PATH,
                }
            )
        return env

    def config_metadata(self) -> dict[str, Any]:
        """Block recorded in the rollout's ``config.json``."""
        return {
            "mode": EGRESS_POLICY_BLOCKLIST,
            "blocked_urls": list(self.rules),
            "tls_inspection_hosts": list(self.tls_inspection_hosts),
            "blocked_status": BLOCKED_STATUS,
            "proxy_port": self.proxy_port,
        }


def apply_blocklist_env(
    agent_env: dict[str, str], blocklist: EgressBlocklist | None
) -> dict[str, str]:
    """Return ``agent_env`` with the blocklist policy applied (or unchanged)."""
    if blocklist is None:
        return agent_env
    return {
        **agent_env,
        **blocklist.agent_env(),
        EGRESS_BLOCKED_URLS_ENV: blocklist.to_env_value(),
    }


def strip_blocklist_secret(agent_env: dict[str, str]) -> dict[str, str]:
    """Drop the rule list before the env reaches the agent process."""
    if EGRESS_BLOCKED_URLS_ENV not in agent_env:
        return agent_env
    return {k: v for k, v in agent_env.items() if k != EGRESS_BLOCKED_URLS_ENV}


def strip_proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Drop proxy/CA routing vars — for root-run helpers (LiteLLM) that must
    reach providers directly rather than through the agent's filter."""
    return {k: v for k, v in env.items() if k not in _PROXY_ENV_NAMES}


def blocklist_active(agent_env: dict[str, str] | None) -> bool:
    return bool((agent_env or {}).get(EGRESS_BLOCKED_URLS_ENV))


# ---------------------------------------------------------------------------
# In-sandbox proxy (stdlib only; must run on whatever python3 the image has).
# ---------------------------------------------------------------------------

_EGRESS_PROXY_SOURCE = r'''
"""BenchFlow egress blocklist proxy. Runs as root on loopback inside the sandbox."""
import ipaddress
import json
import os
import posixpath
import select
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse

with open(sys.argv[1], encoding="utf-8") as _cfg_fh:
    CONFIG = json.load(_cfg_fh)
RULES = list(CONFIG["rules"])
PORT = int(CONFIG["port"])
LOG_PATH = CONFIG["log_path"]
BLOCKED_STATUS = int(CONFIG.get("blocked_status", 404))
CA_DIR = CONFIG.get("ca_dir")  # None => no TLS inspection
UPSTREAM_CA_FILE = CONFIG.get("upstream_ca_file")  # tests only
# The proxy runs as root and is NOT confined by the agent-UID firewall, so it
# must refuse to become a hop into places the agent could not otherwise reach:
# loopback (root-only local services), link-local (cloud instance metadata,
# 169.254.169.254), unspecified/multicast/reserved ranges. RFC1918 private
# ranges stay reachable by default because task compose side-services (mock
# APIs, vulnerable targets) live there; flip allow_private_networks to block.
# RFC1918/ULA ranges are refused by default: the root proxy is not confined by
# the agent-UID firewall, so it must not become a hop into internal services.
# The container's OWN directly-connected subnets (the compose network with the
# task's side-services) are allowed automatically; allow_private_networks=true
# re-opens every private range for tasks that genuinely need it.
ALLOW_PRIVATE_NETWORKS = bool(CONFIG.get("allow_private_networks", False))
ALLOW_LOCAL_SUBNETS = bool(CONFIG.get("allow_local_subnets", True))
ALLOW_LOOPBACK_UPSTREAM = bool(CONFIG.get("allow_loopback_upstream", False))  # tests only
CONNECT_TIMEOUT = float(CONFIG.get("connect_timeout", 20))
IDLE_TIMEOUT = float(CONFIG.get("idle_timeout", 300))
MAX_HEAD = 256 * 1024
# Cap on simultaneously handled connections (one thread each). A burst of
# concurrent fetches from the agent waits in the kernel listen backlog instead
# of exhausting threads or file descriptors in the proxy (review P1 #5).
MAX_CONNECTIONS = int(CONFIG.get("max_connections", 256))
_SLOTS = threading.BoundedSemaphore(MAX_CONNECTIONS)

_LOG_LOCK = threading.Lock()
_LEAF_LOCK = threading.Lock()


def log(event, **fields):
    record = {"ts": time.time(), "event": event}
    record.update(fields)
    line = json.dumps(record, sort_keys=True)
    with _LOG_LOCK:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")


def split_rule(rule):
    host, _, path = rule.partition("/")
    return host.lower(), path.strip("/")


def host_matches(rule_host, host):
    host = host.lower().rstrip(".")
    return host == rule_host or host.endswith("." + rule_host)


def normalize_request_path(path):
    # Same canonicalization an upstream server applies before routing, so
    # percent-encoding, "//" runs, and "/x/../" segments cannot dodge a rule.
    raw = path.split("?", 1)[0].split("#", 1)[0]
    decoded = raw
    for _ in range(4):  # decode until stable: %252f -> %2f -> / (bounded)
        again = urllib.parse.unquote(decoded)
        if again == decoded:
            break
        decoded = again
    collapsed = posixpath.normpath("/" + decoded).strip("/")
    return "" if collapsed == "." else collapsed.lower()


def match_blocked(host, path):
    clean = normalize_request_path(path)
    for rule in RULES:
        rule_host, rule_path = split_rule(rule)
        rule_path = rule_path.lower()
        if not host_matches(rule_host, host):
            continue
        if not rule_path:
            return rule
        if clean == rule_path or clean.startswith(rule_path):
            return rule
    return None


def request_hosts(primary_host, headers):
    """Every host name a request names: the URL/CONNECT target AND the Host
    header. Upstream routes on the Host header, so an agent that sends
    ``GET http://<ip>/paper`` with ``Host: arxiv.org`` must be judged on both."""
    hosts = [primary_host]
    host_hdr = header(headers, "host")
    if host_hdr:
        hdr_host = host_hdr.rpartition(":")[0] if ":" in host_hdr and not host_hdr.endswith("]") else host_hdr
        hdr_host = hdr_host.strip("[]").strip().lower()
        if hdr_host and hdr_host not in hosts:
            hosts.append(hdr_host)
    return hosts


def first_blocking_rule(hosts, path):
    for candidate in hosts:
        rule = match_blocked(candidate, path)
        if rule:
            return rule
    return None


def needs_inspection(host):
    if not CA_DIR:
        return False
    for rule in RULES:
        rule_host, rule_path = split_rule(rule)
        if rule_path and host_matches(rule_host, host):
            return True
    return False


def blocked_response(head_only=False):
    body = b"Not Found\n" if BLOCKED_STATUS == 404 else b"Forbidden\n"
    reason = "Not Found" if BLOCKED_STATUS == 404 else "Forbidden"
    head = (
        "HTTP/1.1 %d %s\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n"
        "Connection: close\r\n\r\n" % (BLOCKED_STATUS, reason, len(body))
    ).encode()
    return head if head_only else head + body


def simple_response(status, reason, body=b""):
    return (
        "HTTP/1.1 %d %s\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n"
        "Connection: close\r\n\r\n" % (status, reason, len(body))
    ).encode() + body


def read_head(conn):
    """Read up to the end of the HTTP head. Returns (head_bytes, leftover)."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return buf, b""
        buf += chunk
        if len(buf) > MAX_HEAD:
            raise ValueError("request head too large")
    head, rest = buf.split(b"\r\n\r\n", 1)
    return head + b"\r\n\r\n", rest


def parse_head(head):
    text = head.decode("latin-1")
    lines = text.split("\r\n")
    request_line = lines[0]
    parts = request_line.split(" ")
    if len(parts) < 3:
        raise ValueError("malformed request line")
    method, target, version = parts[0], parts[1], parts[2]
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers.append((name.strip(), value.strip()))
    return method, target, version, headers


def header(headers, name):
    for k, v in headers:
        if k.lower() == name.lower():
            return v
    return None


def relay(a, b):
    """Bidirectional copy until either side closes."""
    socks = [a, b]
    a.setblocking(True)
    b.setblocking(True)
    last = time.time()
    while True:
        readable, _, errored = select.select(socks, [], socks, 5.0)
        if errored:
            break
        if not readable:
            if time.time() - last > IDLE_TIMEOUT:
                break
            continue
        last = time.time()
        done = False
        for s in readable:
            other = b if s is a else a
            try:
                data = s.recv(65536)
            except (OSError, ssl.SSLError):
                done = True
                break
            if not data:
                done = True
                break
            try:
                other.sendall(data)
            except OSError:
                done = True
                break
        if done:
            break


class UpstreamRefused(Exception):
    """The resolved upstream address is one the proxy must not reach for."""


def _local_ipv4_subnets(route_table=None):
    """Directly-connected IPv4 networks from /proc/net/route (Linux). Lines
    with a gateway of 0 are on-link routes: the container's own subnets."""
    if route_table is None:
        try:
            with open("/proc/net/route", encoding="ascii") as fh:
                route_table = fh.read()
        except OSError:
            return []
    nets = []
    for line in route_table.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        dest_hex, gateway_hex, mask_hex = parts[1], parts[2], parts[7]
        try:
            if int(gateway_hex, 16) != 0:
                continue  # via a gateway: not on-link
            dest = ipaddress.IPv4Address(int.from_bytes(bytes.fromhex(dest_hex)[::-1], "big"))
            mask = ipaddress.IPv4Address(int.from_bytes(bytes.fromhex(mask_hex)[::-1], "big"))
            net = ipaddress.IPv4Network("%s/%s" % (dest, mask), strict=False)
        except (ValueError, OverflowError):
            continue
        if net.prefixlen == 0 or net.is_loopback:
            continue
        nets.append(net)
    return nets


_LOCAL_SUBNETS = _local_ipv4_subnets() if ALLOW_LOCAL_SUBNETS else []


def _address_refusal(ip):
    if ip.is_loopback:
        return None if ALLOW_LOOPBACK_UPSTREAM else "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private and not ALLOW_PRIVATE_NETWORKS:
        if any(ip in net for net in _LOCAL_SUBNETS):
            return None  # the container's own compose network
        return "private"
    return None


def resolve_upstream(host, port):
    """Resolve host and vet EVERY address (DNS-rebinding safe: we connect to
    the vetted IPs, never re-resolve). Returns the candidate list ordered
    IPv4-first: a default docker bridge has no IPv6 route, and glibc tends to
    list AAAA records first, so trying v6 first would fail with ENETUNREACH."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise OSError("resolve %s: %s" % (host, exc))
    if not infos:
        raise OSError("resolve %s: no addresses" % host)
    candidates = []
    for family, _, _, _, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        why = _address_refusal(ip)
        if why:
            raise UpstreamRefused("%s resolves to %s address %s" % (host, why, ip))
        candidates.append((family, sockaddr))
    candidates.sort(key=lambda item: 0 if item[0] == socket.AF_INET else 1)
    return candidates


def upstream_connect(host, port, tls=False):
    sock = None
    last_err = None
    for family, sockaddr in resolve_upstream(host, port):
        candidate = socket.socket(family, socket.SOCK_STREAM)
        try:
            candidate.settimeout(CONNECT_TIMEOUT)
            candidate.connect(sockaddr)
        except OSError as exc:  # e.g. ENETUNREACH on a v6 address: try the next
            last_err = exc
            candidate.close()
            continue
        sock = candidate
        break
    if sock is None:
        raise OSError("connect %s:%s: %s" % (host, port, last_err))
    sock.settimeout(IDLE_TIMEOUT)
    if tls:
        ctx = ssl.create_default_context()
        if UPSTREAM_CA_FILE:
            ctx.load_verify_locations(UPSTREAM_CA_FILE)
        ctx.set_alpn_protocols(["http/1.1"])  # we only speak HTTP/1.1 in the middle
        sock = ctx.wrap_socket(sock, server_hostname=host)
    return sock


def refused_response(exc):
    body = ("Forbidden: %s\n" % exc).encode()
    return simple_response(403, "Forbidden", body)


def _leaf_key(leaf_dir):
    """One shared RSA key for every leaf: keygen is the expensive step, and
    per-host uniqueness buys nothing here (the CA is per-run and root-only)."""
    key = os.path.join(leaf_dir, "leaf.key")
    if not os.path.exists(key):
        subprocess.run(
            ["openssl", "genrsa", "-out", key, "2048"], check=True, capture_output=True
        )
        os.chmod(key, 0o600)
    return key


def leaf_cert(host):
    """Return (cert_path, key_path) for host, signing a new leaf on first use."""
    safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in host)
    leaf_dir = os.path.join(CA_DIR, "leaf")
    crt = os.path.join(leaf_dir, safe + ".crt")
    with _LEAF_LOCK:
        os.makedirs(leaf_dir, exist_ok=True)
        key = _leaf_key(leaf_dir)
        if os.path.exists(crt):
            return crt, key
        csr = os.path.join(leaf_dir, safe + ".csr")
        ext = os.path.join(leaf_dir, safe + ".ext")
        # RFC 5280: an IP-literal host needs an IP: SAN, not DNS: (clients
        # reject the name match otherwise). Rules may name bare IPs.
        try:
            ipaddress.ip_address(host)
            san = "IP:%s" % host
        except ValueError:
            san = "DNS:%s" % host
        with open(ext, "w") as fh:
            fh.write("subjectAltName=%s\nextendedKeyUsage=serverAuth\n" % san)
        subprocess.run(
            ["openssl", "req", "-new", "-key", key, "-out", csr, "-subj", "/CN=%s" % host],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["openssl", "x509", "-req", "-in", csr,
             "-CA", os.path.join(CA_DIR, "ca.crt"),
             "-CAkey", os.path.join(CA_DIR, "ca.key"), "-CAcreateserial",
             "-out", crt, "-days", "3", "-sha256", "-extfile", ext],
            check=True, capture_output=True,
        )
        return crt, key


def forward_request(client, upstream, method, target, version, headers, leftover):
    """Send one request (origin-form target) upstream and relay the response.

    ``Connection: close`` is forced so the response ends when upstream closes
    and per-request policy decisions stay simple; clients reconnect per request.
    """
    hop_by_hop = {"proxy-connection", "connection", "keep-alive", "proxy-authorization",
                  "te", "trailer", "transfer-encoding", "upgrade", "expect"}
    out_headers = [(k, v) for k, v in headers if k.lower() not in hop_by_hop]
    chunked = (header(headers, "transfer-encoding") or "").lower() == "chunked"
    if chunked:
        out_headers.append(("Transfer-Encoding", "chunked"))
    out_headers.append(("Connection", "close"))
    head = "%s %s %s\r\n" % (method, target, version)
    head += "".join("%s: %s\r\n" % (k, v) for k, v in out_headers)
    head += "\r\n"
    upstream.sendall(head.encode("latin-1"))
    # Expect: 100-continue — the client waits for an interim response before
    # sending its body while we would wait for that body before reading
    # upstream: a deadlock (until the client's own timeout). Answer the
    # interim ourselves and forward the request WITHOUT the Expect header, so
    # the upstream reply is a normal final response. Clients that send the body
    # straight away are unaffected (the body arrives as leftover/recv anyway).
    if (header(headers, "expect") or "").lower() == "100-continue" and not leftover:
        client.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
    # Body: Content-Length or chunked; otherwise none.
    length = header(headers, "content-length")
    if chunked:
        upstream.sendall(leftover)
        buf = leftover
        while not buf.endswith(b"0\r\n\r\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            upstream.sendall(chunk)
            buf = (buf + chunk)[-8:]
    elif length:
        remaining = int(length) - len(leftover)
        upstream.sendall(leftover)
        while remaining > 0:
            chunk = client.recv(min(65536, remaining))
            if not chunk:
                break
            upstream.sendall(chunk)
            remaining -= len(chunk)
    while True:
        data = upstream.recv(65536)
        if not data:
            break
        client.sendall(data)


def handle_inspected(tls_client, host, port):
    """Serve HTTP requests inside a terminated TLS session, one at a time."""
    while True:
        try:
            head, leftover = read_head(tls_client)
        except (OSError, ValueError, ssl.SSLError):
            return
        if not head.strip():
            return
        method, target, version, headers = parse_head(head)
        path = target if target.startswith("/") else "/" + target.split("://", 1)[-1].partition("/")[2]
        rule = first_blocking_rule(request_hosts(host, headers), path)
        if rule:
            log("block", scheme="https", host=host, port=port, method=method, path=path, rule=rule)
            tls_client.sendall(blocked_response(head_only=(method == "HEAD")))
            return
        try:
            upstream = upstream_connect(host, port, tls=True)
        except UpstreamRefused as exc:
            log("refuse", scheme="https", host=host, port=port, method=method, path=path, reason=str(exc))
            tls_client.sendall(refused_response(exc))
            return
        except Exception as exc:
            tls_client.sendall(simple_response(502, "Bad Gateway", str(exc).encode()))
            return
        log("allow", scheme="https", host=host, port=port, method=method, path=path)
        try:
            forward_request(tls_client, upstream, method, path, version, headers, leftover)
        finally:
            upstream.close()
        return  # Connection: close semantics


def handle_connect(client, target):
    host, _, port = target.rpartition(":")
    host = host.strip("[]").lower()
    port = int(port or 443)
    rule = match_blocked(host, "")
    if rule:
        log("block", scheme="connect", host=host, port=port, rule=rule)
        client.sendall(blocked_response())
        return
    if needs_inspection(host):
        crt, key = leaf_cert(host)
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        # We parse HTTP/1.1 text inside the tunnel; refuse h2 at the handshake
        # so curl/httpx/Node negotiate down instead of sending binary frames.
        ctx.set_alpn_protocols(["http/1.1"])
        try:
            tls_client = ctx.wrap_socket(client, server_side=True)
        except ssl.SSLError as exc:
            log("tls-error", host=host, port=port, error=str(exc))
            return
        try:
            handle_inspected(tls_client, host, port)
        finally:
            try:
                tls_client.close()
            except OSError:
                pass
        return
    try:
        upstream = upstream_connect(host, port)
    except UpstreamRefused as exc:
        log("refuse", scheme="connect", host=host, port=port, reason=str(exc))
        client.sendall(refused_response(exc))
        return
    except Exception as exc:
        log("error", scheme="connect", host=host, port=port, error=str(exc))
        client.sendall(simple_response(502, "Bad Gateway", str(exc).encode()))
        return
    log("allow", scheme="connect", host=host, port=port)
    client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
    try:
        relay(client, upstream)
    finally:
        upstream.close()


def handle_plain(client, method, target, version, headers, leftover):
    if target.startswith("/"):
        if target == "/healthz":
            client.sendall(simple_response(200, "OK", b"ok\n"))
        else:
            client.sendall(simple_response(400, "Bad Request", b"absolute URL required\n"))
        return
    scheme, _, rest = target.partition("://")
    hostport, _, path = rest.partition("/")
    path = "/" + path
    host, _, port = hostport.rpartition(":") if ":" in hostport else (hostport, "", "")
    host = host.strip("[]").lower()
    port = int(port) if port else (443 if scheme == "https" else 80)
    rule = first_blocking_rule(request_hosts(host, headers), path)
    if rule:
        log("block", scheme=scheme, host=host, port=port, method=method, path=path, rule=rule)
        client.sendall(blocked_response(head_only=(method == "HEAD")))
        return
    try:
        upstream = upstream_connect(host, port, tls=(scheme == "https"))
    except UpstreamRefused as exc:
        log("refuse", scheme=scheme, host=host, port=port, method=method, path=path, reason=str(exc))
        client.sendall(refused_response(exc))
        return
    except Exception as exc:
        client.sendall(simple_response(502, "Bad Gateway", str(exc).encode()))
        return
    log("allow", scheme=scheme, host=host, port=port, method=method, path=path)
    try:
        forward_request(client, upstream, method, path, version, headers, leftover)
    finally:
        upstream.close()


def handle(client):
    try:
        _handle(client)
    finally:
        _SLOTS.release()


def _handle(client):
    client.settimeout(IDLE_TIMEOUT)
    try:
        head, leftover = read_head(client)
        if not head.strip():
            return
        method, target, version, headers = parse_head(head)
        if method == "CONNECT":
            handle_connect(client, target)
        else:
            handle_plain(client, method, target, version, headers, leftover)
    except Exception as exc:
        log("error", error=str(exc))
    finally:
        try:
            client.close()
        except OSError:
            pass


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.listen(128)
    log(
        "start",
        port=PORT,
        rules=len(RULES),
        tls_inspection=bool(CA_DIR),
        max_connections=MAX_CONNECTIONS,
    )
    while True:
        _SLOTS.acquire()  # back-pressure: block accept() until a slot frees up
        try:
            client, _ = server.accept()
        except OSError:
            _SLOTS.release()
            continue
        try:
            threading.Thread(target=handle, args=(client,), daemon=True).start()
        except RuntimeError as exc:  # can't start new thread
            _SLOTS.release()
            log("error", error="thread start failed: %s" % exc)
            client.close()


if __name__ == "__main__":
    main()
'''


def egress_proxy_source() -> str:
    """The in-sandbox proxy script (exposed for tests)."""
    return _EGRESS_PROXY_SOURCE


def _ensure_tool_cmd(binary: str, package: str) -> str:
    return (
        f"if ! command -v {binary} >/dev/null 2>&1; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        f"apt-get update -qq && apt-get install -y -qq {package} >/dev/null; "
        "elif command -v dnf >/dev/null 2>&1; then "
        f"dnf -y install {package} >/dev/null; "
        "elif command -v apk >/dev/null 2>&1; then "
        f"apk add --no-cache {package} >/dev/null; "
        f"else echo 'No supported package manager to install {package}' >&2; exit 86; fi; fi"
    )


def _ca_setup_cmd() -> str:
    """Generate the per-run CA (root-only key) and publish a trust bundle."""
    d = shlex.quote(EGRESS_RUNTIME_DIR)
    ca_dir = f"{d}/ca"
    bundle = shlex.quote(EGRESS_CA_BUNDLE_PATH)
    return (
        f"mkdir -p {ca_dir}/leaf && chmod 700 {ca_dir} && "
        f"if [ ! -s {ca_dir}/ca.key ]; then "
        f"openssl req -x509 -newkey rsa:2048 -nodes -keyout {ca_dir}/ca.key "
        f"-out {ca_dir}/ca.crt -days 3 -sha256 -subj '/CN=BenchFlow Egress CA' "
        "-addext 'basicConstraints=critical,CA:TRUE' "
        "-addext 'keyUsage=critical,keyCertSign,cRLSign' >/dev/null 2>&1; fi && "
        f"chmod 600 {ca_dir}/ca.key && "
        "sys_bundle=''; for c in /etc/ssl/certs/ca-certificates.crt "
        "/etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do "
        'if [ -s "$c" ]; then sys_bundle="$c"; break; fi; done; '
        f'{{ [ -n "$sys_bundle" ] && cat "$sys_bundle"; cat {ca_dir}/ca.crt; }} '
        f"> {bundle} && chmod 644 {bundle} && "
        "if [ -d /usr/local/share/ca-certificates ]; then "
        f"cp {ca_dir}/ca.crt /usr/local/share/ca-certificates/benchflow-egress.crt && "
        "(update-ca-certificates >/dev/null 2>&1 || true); "
        "elif [ -d /etc/pki/ca-trust/source/anchors ]; then "
        f"cp {ca_dir}/ca.crt /etc/pki/ca-trust/source/anchors/benchflow-egress.crt && "
        "(update-ca-trust >/dev/null 2>&1 || true); fi"
    )


def _exec_return_code(result: Any) -> int:
    code = getattr(result, "return_code", None)
    return int(code) if code is not None else 1


def _exec_detail(result: Any) -> str:
    out = (getattr(result, "stdout", "") or "").strip()
    err = (getattr(result, "stderr", "") or "").strip()
    parts = [p for p in (err, out) if p]
    return (" " + " | ".join(parts)[:800]) if parts else ""


async def _upload_text(env: Any, text: str, target_path: str, *, suffix: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        await env.upload_file(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


async def start_egress_proxy(
    env: Any,
    blocklist: EgressBlocklist,
    *,
    timeout_sec: int = 180,
) -> None:
    """Install and start the filtering proxy as root; idempotent per sandbox.

    Runs before the agent launches so ``HTTP(S)_PROXY`` already resolves. The
    rule file and log are root-only (mode 0600): the agent must not learn the
    hidden URLs by reading the proxy's configuration.
    """
    runtime_dir = EGRESS_RUNTIME_DIR
    q_dir = shlex.quote(runtime_dir)
    script_path = f"{runtime_dir}/egress_proxy.py"
    config_path = f"{runtime_dir}/policy.json"
    pid_path = f"{runtime_dir}/proxy.pid"
    stderr_path = f"{runtime_dir}/proxy.stderr"

    health = (
        'python3 -c "import urllib.request,sys;'
        f"sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{blocklist.proxy_port}/healthz',"
        ' timeout=2).status==200 else 1)"'
    )
    already = await env.exec(health, user="root", timeout_sec=10)
    if _exec_return_code(already) == 0:
        logger.info("Egress blocklist proxy already running on %s", blocklist.proxy_url)
        return

    prep = (
        "set -e; "
        f"mkdir -p {q_dir} && chmod 700 {q_dir}; "
        "if ! command -v python3 >/dev/null 2>&1; then "
        "echo 'python3 is required in the task image for the egress blocklist proxy' >&2; "
        "exit 87; fi"
    )
    if blocklist.needs_tls_inspection:
        prep += "; " + _ensure_tool_cmd("openssl", "openssl") + "; " + _ca_setup_cmd()
    result = await env.exec(prep, user="root", timeout_sec=timeout_sec)
    if _exec_return_code(result) != 0:
        raise RuntimeError(
            f"Failed to prepare the egress blocklist runtime.{_exec_detail(result)}"
        )

    config = {
        "rules": list(blocklist.rules),
        "port": blocklist.proxy_port,
        "log_path": EGRESS_LOG_PATH,
        "blocked_status": BLOCKED_STATUS,
        "ca_dir": f"{runtime_dir}/ca" if blocklist.needs_tls_inspection else None,
    }
    await _upload_text(env, _EGRESS_PROXY_SOURCE, script_path, suffix=".py")
    await _upload_text(env, json.dumps(config), config_path, suffix=".json")

    start = (
        f"chmod 600 {shlex.quote(config_path)} && "
        f"touch {shlex.quote(EGRESS_LOG_PATH)} && chmod 600 {shlex.quote(EGRESS_LOG_PATH)} && "
        f"(nohup python3 {shlex.quote(script_path)} {shlex.quote(config_path)} "
        f">/dev/null 2>{shlex.quote(stderr_path)} </dev/null & echo $! > {shlex.quote(pid_path)})"
    )
    result = await env.exec(start, user="root", timeout_sec=30)
    if _exec_return_code(result) != 0:
        raise RuntimeError(
            f"Failed to start the egress blocklist proxy.{_exec_detail(result)}"
        )

    last = ""
    for _ in range(80):
        probe = await env.exec(health, user="root", timeout_sec=10)
        if _exec_return_code(probe) == 0:
            logger.info(
                "Egress blocklist proxy active on %s (%d rules, tls_inspection=%s)",
                blocklist.proxy_url,
                len(blocklist.rules),
                blocklist.needs_tls_inspection,
            )
            return
        last = _exec_detail(probe)
        await asyncio.sleep(0.25)
    stderr = await env.exec(
        f"cat {shlex.quote(stderr_path)} 2>/dev/null | tail -n 20",
        user="root",
        timeout_sec=10,
    )
    raise RuntimeError(
        f"Egress blocklist proxy did not become healthy.{last}{_exec_detail(stderr)}"
    )


def _probe_cmd(blocklist: EgressBlocklist) -> str:
    """Shell probe run AS THE AGENT USER: the first blocked host **or path**
    must answer the blocked status through the proxy, and direct egress must
    be rejected.

    The probe targets the rule's full ``host/path`` — a path-only rule such as
    ``arxiv.org/abs/2401.12345`` leaves ``arxiv.org/`` legitimately open, so
    probing the root would report 200 and fail a correctly configured run.
    """
    rule_host, rule_path = _split_rule(blocklist.rules[0])
    probe_target = f"/{rule_path}" if rule_path else "/"
    return (
        "python3 - <<'PY'\n"
        "import json, socket, sys, urllib.error, urllib.request\n"
        f"proxy = {blocklist.proxy_url!r}\n"
        f"host = {rule_host!r}\n"
        f"target = {probe_target!r}\n"
        f"expected = {BLOCKED_STATUS}\n"
        "out = {'blocked_target': host + target}\n"
        "opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))\n"
        "try:\n"
        "    resp = opener.open('http://' + host + target, timeout=15)\n"
        "    out['via_proxy_status'] = resp.status\n"
        "except urllib.error.HTTPError as exc:\n"
        "    out['via_proxy_status'] = exc.code\n"
        "except Exception as exc:\n"
        "    out['via_proxy_status'] = None\n"
        "    out['via_proxy_error'] = str(exc)[:200]\n"
        "try:\n"
        "    s = socket.create_connection(('1.1.1.1', 443), timeout=5)\n"
        "    s.close()\n"
        "    out['direct_egress'] = 'open'\n"
        "except Exception as exc:\n"
        "    out['direct_egress'] = 'blocked'\n"
        "    out['direct_egress_error'] = str(exc)[:200]\n"
        "out['ok'] = out['via_proxy_status'] == expected and out['direct_egress'] == 'blocked'\n"
        "print(json.dumps(out))\n"
        "sys.exit(0 if out['ok'] else 1)\n"
        "PY\n"
    )


async def verify_egress_blocklist(
    env: Any, sandbox_user: str | None, agent_env: dict[str, str]
) -> dict[str, Any] | None:
    """Post-firewall self-check, run as the sandbox user. Returns the probe.

    Fails loudly when the blocked host is reachable or direct egress is still
    open: an experiment that silently ran without its blocklist is worthless.
    """
    blocklist = EgressBlocklist.from_env(agent_env)
    if blocklist is None:
        return None
    if not sandbox_user:
        raise RuntimeError(
            "network_mode='blocklist' requires a sandbox_user: the agent-UID "
            "firewall is what makes the proxy mandatory"
        )
    result = await env.exec(_probe_cmd(blocklist), user=sandbox_user, timeout_sec=60)
    stdout = (getattr(result, "stdout", "") or "").strip().splitlines()
    probe: dict[str, Any] = {}
    for line in reversed(stdout):
        try:
            probe = json.loads(line)
            break
        except ValueError:
            continue
    record = json.dumps({"event": "probe", "user": sandbox_user, **probe})
    await env.exec(
        f"printf '%s\\n' {shlex.quote(record)} >> {shlex.quote(EGRESS_LOG_PATH)}",
        user="root",
        timeout_sec=10,
    )
    if _exec_return_code(result) != 0 or not probe.get("ok"):
        raise RuntimeError(
            "Egress blocklist self-check failed: "
            f"{json.dumps(probe) if probe else _exec_detail(result)}"
        )
    logger.info("Egress blocklist verified for %s: %s", sandbox_user, probe)
    return probe


async def download_egress_log(env: Any, target_dir: Path) -> None:
    """Copy the root-only egress log into the rollout's agent artifacts.

    Callers gate on whether the proxy was ever started for this sandbox (any
    role, any scene) — not on the primary agent's env, which an oracle primary
    never carries even when a role agent ran under the blocklist.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / EGRESS_LOG_ARTIFACT_NAME
    try:
        await env.download_file(EGRESS_LOG_PATH, target)
    except Exception as exc:  # audit artifact, never fatal
        logger.warning("Could not download egress log: %s", exc)

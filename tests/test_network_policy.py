"""Network policy: config, rule matching, the egress filter, and its wiring."""

from __future__ import annotations

import http.client
import http.server
import json
import os
import socket
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.providers.litellm_logging import callback_module_source
from benchflow.providers.litellm_runtime import _litellm_proxy_env
from benchflow.sandbox.egress import (
    NETWORK_POLICY_ENV,
    NETWORK_POLICY_MARKER_ENV,
    EgressFilterProcess,
    NetworkPolicy,
    issue_tls_material,
    start_egress_filter,
)
from benchflow.sandbox.egress_filter import (
    BLOCKED_HEADER,
    EgressFilter,
    Policy,
    Rule,
    canonical_path,
)
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
from benchflow.task.config import NetworkMode, TaskConfig
from benchflow.task.runtime_capabilities import validate_task_runtime_support

PAPER = "arxiv.org/abs/2401.01234"


# Task config


def test_blocklist_entries_are_normalised_and_required() -> None:
    config = TaskConfig.model_validate(
        {
            "agent": {
                "network_mode": "blocklist",
                "blocked_urls": [
                    "https://ArXiv.org/abs/2401.01234",
                    "openreview.net/forum?id=AbC123",
                    "Example-Lab.org.",
                    "pypi.org/",
                    "pypi.org/simple/../simple/requests/",
                ],
            }
        }
    )

    assert config.agent.blocked_urls == [
        PAPER,
        "openreview.net/forum?id=AbC123",
        "example-lab.org",
        "pypi.org",
        "pypi.org/simple/requests/",
    ]
    with pytest.raises(ValueError, match="blocked_urls must be non-empty"):
        TaskConfig.model_validate({"agent": {"network_mode": "blocklist"}})
    with pytest.raises(ValueError, match="only valid for network_mode='blocklist'"):
        TaskConfig.model_validate({"sandbox": {"blocked_urls": ["arxiv.org"]}})
    for bad in ("/abs/1", "arxiv.org:8443/abs/1", "not a host", "ftp://arxiv.org/x"):
        with pytest.raises(ValueError, match="start with a valid hostname"):
            TaskConfig.model_validate(
                {"sandbox": {"network_mode": "blocklist", "blocked_urls": [bad]}}
            )
    tolerated = TaskConfig.model_validate(
        {
            "sandbox": {
                "network_mode": "blocklist",
                "blocked_urls": ["https://user@ArXiv.org:443/abs/1#frag"],
            }
        }
    )
    assert tolerated.sandbox.blocked_urls == ["arxiv.org/abs/1"]


def test_network_policy_prefers_the_agent_override() -> None:
    config = TaskConfig.model_validate(
        {
            "sandbox": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
            "agent": {"network_mode": "blocklist", "blocked_urls": [PAPER]},
        }
    )

    policy = NetworkPolicy.resolve(config)

    assert policy == NetworkPolicy(NetworkMode.BLOCKLIST, (PAPER,))
    assert policy.to_json() == {"mode": "blocklist", "rules": [PAPER]}
    # The agent learns that a policy exists, the proxy learns what it is.
    assert policy.agent_env({"KEEP": "1"}) == {
        "KEEP": "1",
        NETWORK_POLICY_MARKER_ENV: "1",
    }
    assert json.loads(policy.proxy_env({})[NETWORK_POLICY_ENV]) == policy.to_json()


def test_network_policy_is_absent_for_public_and_no_network() -> None:
    public = TaskConfig.model_validate({"sandbox": {"network_mode": "public"}})
    isolated = TaskConfig.model_validate(
        {
            "sandbox": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
            "agent": {"network_mode": "no-network"},
        }
    )

    assert NetworkPolicy.resolve(public) is None
    assert NetworkPolicy.resolve(isolated) is None


@pytest.mark.parametrize(
    ("sandbox", "expected"),
    [
        ("docker", []),
        ("daytona", []),
        (
            "agentcore",
            [
                (
                    "agent.network_mode",
                    "network_mode='blocklist' is not enforced by agentcore",
                )
            ],
        ),
    ],
)
def test_capability_gate_reports_backends_that_cannot_filter(
    sandbox: str, expected: list[tuple[str, str]]
) -> None:
    config = TaskConfig.model_validate(
        {"agent": {"network_mode": "blocklist", "blocked_urls": [PAPER]}}
    )

    issues = validate_task_runtime_support(config, sandbox=sandbox)

    assert [(issue.path, issue.reason) for issue in issues] == expected


def test_capability_gate_says_the_verifier_is_not_filtered() -> None:
    config = TaskConfig.model_validate(
        {"verifier": {"network_mode": "blocklist", "blocked_urls": [PAPER]}}
    )

    issues = validate_task_runtime_support(config, sandbox="docker")

    assert [issue.reason for issue in issues] == [
        "network_mode='blocklist' is enforced for the agent, not the verifier"
    ]


# Rules


def test_canonical_paths_decode_collapse_and_keep_directory_slashes() -> None:
    assert canonical_path("/abs/%32401.01234") == "/abs/2401.01234"
    assert canonical_path("/pdf/../abs/2401.01234?x=%41") == "/abs/2401.01234?x=A"
    assert canonical_path("//simple/./requests/") == "/simple/requests/"
    assert canonical_path("") == "/"


def test_rules_match_hosts_subdomains_and_path_prefixes() -> None:
    paper = Rule.parse(f"https://{PAPER}")
    lab = Rule.parse("Example-Lab.org")
    directory = Rule.parse("pypi.org/simple/requests/")

    assert paper == Rule("arxiv.org", "/abs/2401.01234")
    assert paper.matches("arxiv.org", "/abs/2401.01234v2")
    assert paper.matches("export.arxiv.org", "/abs/2401.01234")
    assert not paper.matches("arxiv.org", "/abs/2401.0123")
    assert not paper.matches("notarxiv.org", "/abs/2401.01234")
    assert lab.matches("www.example-lab.org", "/anything?x=1")
    assert directory.matches("pypi.org", "/simple/requests/")
    assert not directory.matches("pypi.org", "/simple/requests-foo/")
    assert Rule.parse("pypi.org/") == Rule("pypi.org")


def test_blocklist_policy_decides_on_canonical_paths_and_inspects_path_hosts() -> None:
    policy = Policy.from_json(
        {"mode": "blocklist", "rules": [PAPER, "example-lab.org"]}
    )

    assert policy.decide("arxiv.org", "/abs/2401.01234").allowed is False
    assert policy.decide("ArXiv.org.", "/pdf/../abs/%32401.01234").allowed is False
    assert policy.decide("arxiv.org", "/abs/2309.00001").allowed is True
    assert policy.decide("example-lab.org", "/").rule == Rule("example-lab.org")
    assert policy.inspects("arxiv.org") and policy.inspects("Export.arxiv.org")
    assert not policy.inspects("example-lab.org")
    assert policy.inspected_hosts == ("arxiv.org",)


def test_allowlist_policy_admits_listed_hosts_only() -> None:
    policy = Policy.from_json({"mode": "allowlist", "rules": ["pypi.org"]})

    assert policy.decide("files.pypi.org", "/simple/").allowed is True
    assert policy.decide("arxiv.org", "/").allowed is False
    assert policy.inspected_hosts == ()


# The filter itself, against local upstreams.


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = f"upstream saw {self.path} for {self.headers['Host']}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "12345")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def upstream() -> Iterator[http.server.HTTPServer]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def tls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    material = issue_tls_material(("localhost",))
    paths = {name: str(tmp_path / f"{name}.pem") for name in ("ca", "cert", "key")}
    Path(paths["ca"]).write_bytes(material.ca_pem)
    Path(paths["cert"]).write_bytes(material.cert_pem)
    Path(paths["key"]).write_bytes(material.key_pem)
    # The filter verifies upstream TLS with the default trust store, which
    # honours SSL_CERT_FILE; the test upstream presents the same leaf.
    monkeypatch.setenv("SSL_CERT_FILE", paths["ca"])
    return paths


@pytest.fixture
def egress(tmp_path: Path, tls: dict[str, str]) -> Iterator[EgressFilter]:
    policy = Policy.from_json(
        {"mode": "blocklist", "rules": ["localhost/paper", "blocked.example"]}
    )
    server = EgressFilter(
        ("127.0.0.1", 0),
        policy=policy,
        log_path=str(tmp_path / "decisions.jsonl"),
        certificate=(tls["cert"], tls["key"]),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _via_proxy(
    egress: EgressFilter, method: str, url: str, headers: dict[str, str] | None = None
) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", egress.server_port, timeout=10)
    connection.request(method, url, headers=headers or {})
    return connection.getresponse()


def _raw(egress: EgressFilter, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", egress.server_port)) as sock:
        sock.sendall(request)
        reply = b""
        while chunk := sock.recv(4096):
            reply += chunk
    return reply


def test_filter_forwards_plain_http_and_refuses_blocked_paths(
    egress: EgressFilter, upstream: http.server.HTTPServer, tmp_path: Path
) -> None:
    base = f"http://localhost:{upstream.server_port}"

    allowed = _via_proxy(egress, "GET", f"{base}/other?x=1")
    refused = _via_proxy(egress, "GET", f"{base}/paper/123")

    assert allowed.status == 200
    assert (
        allowed.read()
        == f"upstream saw /other?x=1 for localhost:{upstream.server_port}".encode()
    )
    assert refused.status == 403
    assert refused.getheader(BLOCKED_HEADER) == "blocked"
    assert refused.read() == b"benchflow: blocked by network policy\n"
    decisions = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text().splitlines()
    ]
    # Allowed requests are logged without their query string.
    assert [(d["path"], d["decision"], d["rule"]) for d in decisions] == [
        ("/other", "allow", None),
        ("/paper/123", "block", "localhost/paper"),
    ]
    assert (tmp_path / "decisions.jsonl").stat().st_mode & 0o777 == 0o600


def test_filter_talks_to_the_decided_host_not_the_client_host_header(
    egress: EgressFilter, upstream: http.server.HTTPServer
) -> None:
    response = _via_proxy(
        egress,
        "GET",
        f"http://localhost:{upstream.server_port}/x",
        headers={"Host": "blocked.example"},
    )

    assert response.status == 200
    assert response.read().endswith(f"for localhost:{upstream.server_port}".encode())


def test_filter_passes_head_headers_and_needs_a_content_length(
    egress: EgressFilter, upstream: http.server.HTTPServer
) -> None:
    base = f"http://localhost:{upstream.server_port}"

    head = _via_proxy(egress, "HEAD", f"{base}/x")
    assert (head.status, head.getheader("Content-Length"), head.read()) == (
        200,
        "12345",
        b"",
    )

    chunked = _raw(
        egress,
        f"POST {base}/x HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n".encode(),
    )
    assert chunked.startswith(b"HTTP/1.1 411")


def test_filter_refuses_ip_literals_and_blocked_tunnel_hosts(
    egress: EgressFilter,
) -> None:
    for request in (
        b"CONNECT blocked.example:443 HTTP/1.1\r\nHost: blocked.example:443\r\n\r\n",
        b"CONNECT 151.101.1.42:443 HTTP/1.1\r\nHost: 151.101.1.42:443\r\n\r\n",
        b"CONNECT [2606:4700::1]:8443 HTTP/1.1\r\nHost: [2606:4700::1]:8443\r\n\r\n",
        b"GET http://151.101.1.42/abs/2401.01234 HTTP/1.1\r\nHost: arxiv.org\r\n\r\n",
    ):
        reply = _raw(egress, request)
        assert reply.startswith(b"HTTP/1.1 403"), request
        assert b"blocked by network policy" in reply


def test_filter_opens_tunnels_to_inspected_hosts_and_decides_per_path(
    egress: EgressFilter, upstream: http.server.HTTPServer, tls: dict[str, str]
) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tls["cert"], tls["key"])
    upstream.socket = context.wrap_socket(upstream.socket, server_side=True)

    def fetch(path: str) -> tuple[int, bytes]:
        client = ssl.create_default_context(cafile=tls["ca"])
        connection = http.client.HTTPSConnection(
            "127.0.0.1", egress.server_port, context=client, timeout=10
        )
        connection.set_tunnel("localhost", upstream.server_port)
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()

    assert fetch("/notes") == (
        200,
        f"upstream saw /notes for localhost:{upstream.server_port}".encode(),
    )
    assert fetch("/paper/2401.01234") == (
        403,
        b"benchflow: blocked by network policy\n",
    )


def test_filter_never_relays_an_inspected_host_blind(tmp_path: Path) -> None:
    policy = Policy.from_json({"mode": "blocklist", "rules": ["localhost/paper"]})
    server = EgressFilter(
        ("127.0.0.1", 0), policy=policy, log_path=str(tmp_path / "log.jsonl")
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reply = _raw(
            server,
            b"CONNECT localhost:9 HTTP/1.1\r\nHost: localhost:9\r\n\r\n"
            b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
    finally:
        server.shutdown()
        server.server_close()

    # The tunnel opens (the SNI has to be read first) and then closes with
    # nothing relayed; the log records the refusal.
    assert reply.startswith(b"HTTP/1.1 200 Connection established")
    assert reply.endswith(b"\r\n\r\n")
    decisions = [
        json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert [(d["host"], d["decision"]) for d in decisions] == [("localhost", "block")]


# Provider-side search inside the LiteLLM proxy.


async def test_proxy_hook_removes_provider_side_search_under_a_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    logger = namespace["proxy_handler_instance"]
    data = {
        "model": "gemini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"google_search": {}},
            {"type": "function", "function": {"name": "sh"}},
        ],
        "web_search_options": {"search_context_size": "low"},
    }

    assert (
        await logger.async_pre_call_hook(None, None, dict(data), "completion") is None
    )

    monkeypatch.setenv(
        NETWORK_POLICY_ENV, json.dumps({"mode": "blocklist", "rules": [PAPER]})
    )
    cleaned = await logger.async_pre_call_hook(None, None, dict(data), "completion")

    assert cleaned["tools"] == [{"type": "function", "function": {"name": "sh"}}]
    assert "web_search_options" not in cleaned
    assert cleaned["messages"] == data["messages"]


def test_proxy_environment_carries_the_policy_but_not_the_filter_variables() -> None:
    policy = NetworkPolicy(NetworkMode.BLOCKLIST, (PAPER,))
    agent_env = {
        "OPENAI_API_KEY": "k",
        "HTTPS_PROXY": "http://127.0.0.1:4242",
        "SSL_CERT_FILE": "/tmp/x/ca-bundle.pem",
        NETWORK_POLICY_MARKER_ENV: "1",
    }

    proxy_env = _litellm_proxy_env(
        agent="opencode",
        agent_env=agent_env,
        required_skill_names=(),
        network_policy=policy,
    )

    assert proxy_env["OPENAI_API_KEY"] == "k"
    assert json.loads(proxy_env[NETWORK_POLICY_ENV]) == policy.to_json()
    assert not {"HTTPS_PROXY", "SSL_CERT_FILE"} & proxy_env.keys()
    assert NETWORK_POLICY_ENV not in _litellm_proxy_env(
        agent="opencode", agent_env=agent_env, required_skill_names=()
    )


# The firewall marker.


async def test_network_policy_marker_turns_on_the_loopback_firewall() -> None:
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0))

    await enforce_agent_egress_firewall(
        env,
        "agent",
        {
            NETWORK_POLICY_MARKER_ENV: "1",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1",
        },
    )

    env.exec.assert_awaited_once()
    assert env.exec.await_args.kwargs == {"user": "root", "timeout_sec": 120}


# Starting the filter inside a sandbox.


class _FakeSandbox:
    def __init__(self) -> None:
        self.uploads: dict[str, tuple[bytes, str | None]] = {}
        self.commands: list[str] = []
        self.downloads: list[tuple[str, Path]] = []

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30) -> Any:
        self.commands.append(cmd)
        stdout = '{"pid": 7, "port": 4242}' if cmd.startswith("cat ") else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def upload_file(
        self, src: Path, dst: str, *, mode: str | None = None
    ) -> None:
        self.uploads[dst.rsplit("/", 1)[1]] = (src.read_bytes(), mode)

    async def download_file(self, src: str, dst: Path) -> None:
        self.downloads.append((src, dst))


async def test_start_egress_filter_uploads_policy_and_certificate(
    tmp_path: Path,
) -> None:
    sandbox = _FakeSandbox()
    policy = NetworkPolicy(NetworkMode.BLOCKLIST, (PAPER, "example-lab.org"))

    process = await start_egress_filter(sandbox, policy, python="/venv/bin/python")

    assert process.port == 4242 and process.inspecting is True
    assert json.loads(sandbox.uploads["policy.json"][0]) == policy.to_json()
    assert sandbox.uploads["policy.json"][1] == sandbox.uploads["key.pem"][1] == "600"
    assert sandbox.uploads["ca.pem"][1] == "644"
    assert b"BEGIN CERTIFICATE" in sandbox.uploads["ca.pem"][0]
    assert sandbox.uploads["egress_filter.py"][0].startswith(
        b'"""Loopback egress filter'
    )
    assert any("chmod 711" in cmd for cmd in sandbox.commands)
    assert any(
        cmd.startswith("/venv/bin/python ") and " launch " in cmd
        for cmd in sandbox.commands
    )
    env = process.agent_env
    assert env["HTTPS_PROXY"] == env["http_proxy"] == "http://127.0.0.1:4242"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"
    assert env["NODE_EXTRA_CA_CERTS"] == process.paths["ca"]
    assert env["SSL_CERT_FILE"] == process.paths["ca_bundle"]
    assert NETWORK_POLICY_ENV not in env

    await process.stop(log_destination=tmp_path / "network_policy.jsonl")

    assert sandbox.downloads == [
        (process.paths["log"], tmp_path / "network_policy.jsonl")
    ]
    assert sandbox.commands[-1] == f"rm -rf {process.runtime_dir}"


def test_host_only_policy_needs_no_certificate() -> None:
    process = EgressFilterProcess(
        sandbox=None, runtime_dir="/tmp/x", port=1, inspecting=False, paths={}
    )

    assert "SSL_CERT_FILE" not in process.agent_env
    assert process.agent_env["https_proxy"] == "http://127.0.0.1:1"


# Rollout wiring.


def _rollout_with_policy() -> Any:
    from benchflow.rollout import Rollout, _resolve_network_policy

    task = SimpleNamespace(
        config=TaskConfig.model_validate(
            {"agent": {"network_mode": "blocklist", "blocked_urls": [PAPER]}}
        )
    )
    rollout = Rollout.__new__(Rollout)
    rollout._disallow_web_tools = False
    rollout._network_policy = _resolve_network_policy(task)
    rollout._egress_filter = None
    rollout._env = object()
    rollout._usage_runtime = SimpleNamespace(
        server=SimpleNamespace(python="/venv/bin/python")
    )
    rollout._planes = MagicMock()
    return rollout


async def test_rollout_routes_the_agent_through_the_filter_once_per_sandbox() -> None:
    rollout = _rollout_with_policy()
    started = EgressFilterProcess(
        sandbox=None,
        runtime_dir="/tmp/x",
        port=4242,
        inspecting=True,
        paths={"ca": "/tmp/x/ca.pem", "ca_bundle": "/tmp/x/ca-bundle.pem"},
    )
    rollout._planes.start_egress_filter = AsyncMock(return_value=started)

    env = rollout._with_network_policy({"KEEP": "1"}, "opencode")
    env = await rollout._start_egress_filter(env, "opencode")
    env = await rollout._start_egress_filter(env, "opencode")

    assert env["KEEP"] == "1"
    assert env[NETWORK_POLICY_MARKER_ENV] == "1"
    assert NETWORK_POLICY_ENV not in env
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:4242"
    rollout._planes.start_egress_filter.assert_awaited_once_with(
        rollout._env, rollout._network_policy, python="/venv/bin/python"
    )


async def test_rollout_keeps_the_oracle_outside_the_policy() -> None:
    rollout = _rollout_with_policy()

    assert rollout._network_policy_for("oracle") is None
    assert rollout._with_network_policy({"KEEP": "1"}, "oracle") == {"KEEP": "1"}
    assert await rollout._start_egress_filter({"KEEP": "1"}, "oracle") == {"KEEP": "1"}
    rollout._planes.start_egress_filter.assert_not_called()


def test_rollout_refuses_a_policy_it_cannot_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout = _rollout_with_policy()
    monkeypatch.setattr(rollout, "_session_factory_entrypoint", lambda agent: None)

    with pytest.raises(RuntimeError, match="needs a sandbox user"):
        rollout._require_enforceable_network_policy(
            SimpleNamespace(primary_agent="opencode", sandbox_user=None)
        )
    rollout._require_enforceable_network_policy(
        SimpleNamespace(primary_agent="opencode", sandbox_user="agent")
    )
    rollout._require_enforceable_network_policy(
        SimpleNamespace(primary_agent="oracle", sandbox_user=None)
    )

    monkeypatch.setattr(
        rollout, "_session_factory_entrypoint", lambda agent: "/entry.py"
    )
    with pytest.raises(RuntimeError, match="session-factory agent"):
        rollout._require_enforceable_network_policy(
            SimpleNamespace(primary_agent="deepagents", sandbox_user="agent")
        )


def test_tunnels_are_decided_on_the_tls_server_name_not_the_connect_target(
    egress: EgressFilter, upstream: http.server.HTTPServer, tls: dict[str, str]
) -> None:
    """An alias or an address in CONNECT cannot stand in for a blocked name."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tls["cert"], tls["key"])
    upstream.socket = context.wrap_socket(upstream.socket, server_side=True)
    client = ssl.create_default_context(cafile=tls["ca"])
    client.check_hostname = False

    def handshake(connect_to: str, server_name: str) -> ssl.SSLSocket:
        sock = socket.create_connection(("127.0.0.1", egress.server_port), timeout=10)
        sock.sendall(
            f"CONNECT {connect_to} HTTP/1.1\r\nHost: {connect_to}\r\n\r\n".encode()
        )
        assert sock.recv(4096).startswith(b"HTTP/1.1 200")
        return client.wrap_socket(sock, server_hostname=server_name)

    # CONNECT names an innocent host; the handshake asks for a blocked one.
    with pytest.raises(OSError):
        handshake(f"allowed.example:{upstream.server_port}", "blocked.example")
    # CONNECT names an innocent host; the handshake asks for an inspected one,
    # so the path is decided and the upstream is the named host.
    secured = handshake(f"allowed.example:{upstream.server_port}", "localhost")
    secured.sendall(b"GET /paper/1 HTTP/1.1\r\nHost: localhost\r\n\r\n")
    assert b"403" in secured.recv(4096)
    secured.close()
    # A tunnel that does not start with TLS is closed without relaying.
    with socket.create_connection(
        ("127.0.0.1", egress.server_port), timeout=10
    ) as sock:
        sock.sendall(
            f"CONNECT allowed.example:{upstream.server_port} HTTP/1.1\r\n\r\n".encode()
        )
        assert sock.recv(4096).startswith(b"HTTP/1.1 200")
        sock.sendall(b"GET / HTTP/1.1\r\nHost: blocked.example\r\n\r\n")
        assert sock.recv(4096) == b""


@pytest.mark.parametrize(
    "address", ["127.1", "0x7f000001", "2130706433", "[::1]", "10.0.0.1"]
)
def test_numeric_hosts_are_refused_in_every_spelling(
    egress: EgressFilter, address: str
) -> None:
    reply = _raw(
        egress,
        f"CONNECT {address}:443 HTTP/1.1\r\nHost: {address}:443\r\n\r\n".encode(),
    )

    assert reply.startswith(b"HTTP/1.1 403")


def test_filter_rejects_a_bad_content_length(egress: EgressFilter) -> None:
    reply = _raw(
        egress,
        b"POST http://localhost:1/x HTTP/1.1\r\nHost: localhost:1\r\nContent-Length: -5\r\n\r\n",
    )

    assert reply.startswith(b"HTTP/1.1 400")


def test_filter_reframes_chunked_upstream_bodies(egress: EgressFilter) -> None:
    class _Chunked(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Length", "999")  # lies; chunked wins
            self.end_headers()
            for piece in (b"first ", b"second"):
                self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *args: Any) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Chunked)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        response = _via_proxy(
            egress, "GET", f"http://localhost:{server.server_port}/stream"
        )
        assert response.status == 200
        assert response.getheader("Content-Length") is None
        assert response.read() == b"first second"
    finally:
        server.shutdown()


async def test_production_litellm_wrapper_forwards_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planes call the wrapper in providers.runtime, not the implementation."""
    from benchflow.providers import litellm_runtime, runtime

    captured: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> tuple[dict[str, str], None]:
        captured.update(kwargs)
        return kwargs["agent_env"], None

    monkeypatch.setattr(litellm_runtime, "ensure_litellm_runtime", fake)
    policy = NetworkPolicy(NetworkMode.BLOCKLIST, (PAPER,))

    await runtime.ensure_litellm_runtime(
        agent="opencode",
        agent_env={},
        model="m",
        runtime=None,
        environment="daytona",
        network_policy=policy,
    )

    assert captured["network_policy"] is policy


def test_capability_gate_judges_the_mode_the_agent_runs_under() -> None:
    config = TaskConfig.model_validate(
        {
            "sandbox": {"network_mode": "blocklist", "blocked_urls": [PAPER]},
            "agent": {"network_mode": "public"},
        }
    )

    assert validate_task_runtime_support(config, sandbox="agentcore") == []


# Docker: the firewall needs a capability the daemon withholds by default.


def test_docker_grants_the_firewall_capability_last(tmp_path: Path) -> None:
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    sandbox._egress_firewall_compose_path = None
    sandbox._mounts_compose_path = None
    sandbox._use_prebuilt = False
    sandbox.environment_dir = tmp_path
    sandbox.task_env_config = SimpleNamespace(allow_internet=True)

    sandbox.grant_egress_firewall()

    override = sandbox._egress_firewall_compose_path
    assert override is not None
    assert json.loads(override.read_text()) == {
        "services": {"main": {"cap_add": ["NET_ADMIN"]}}
    }
    assert sandbox._docker_compose_paths[-1] == override


def test_rollout_grants_the_capability_only_to_backends_that_need_it() -> None:
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._env_externally_owned = False
    rollout._env = SimpleNamespace(grant_egress_firewall=MagicMock())
    rollout._grant_egress_firewall()
    rollout._env.grant_egress_firewall.assert_called_once_with()

    rollout._env = object()  # Daytona-like: nothing to grant
    rollout._grant_egress_firewall()

    rollout._env = SimpleNamespace(grant_egress_firewall=MagicMock())
    rollout._env_externally_owned = True
    with pytest.raises(RuntimeError, match="already-started sandbox"):
        rollout._grant_egress_firewall()


# Live canary: a real sandbox, the filter, the firewall, and curl as the
# sandbox user. Skipped by default; run with ``-m integration -k canary``.


def _backend_available(backend: str) -> bool:
    import shutil
    import subprocess

    if backend == "daytona":
        return bool(os.environ.get("DAYTONA_API_KEY"))
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=15, check=False
    )
    return probe.returncode == 0


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["docker", "daytona"])
async def test_network_policy_sandbox_canary(backend: str, tmp_path: Path) -> None:
    """The blocklist holds inside a real sandbox for the sandbox user."""
    import shlex
    import shutil

    from benchflow.sandbox.lockdown import build_priv_drop_cmd, setup_sandbox_user
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task.paths import RolloutPaths
    from benchflow.task.task import Task

    if not _backend_available(backend):
        pytest.skip(f"{backend} is not available here")
    task_dir = tmp_path / "blocklist-canary"
    shutil.copytree(Path(__file__).parent / "examples" / "hello-world-task", task_dir)
    dockerfile = task_dir / "environment" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text().replace(
            "apt-get install -y -qq curl",
            "apt-get install -y -qq curl python3 ca-certificates iptables",
        )
    )
    # Keep the native task.md only, so the policy edit below is the config.
    for legacy in ("task.toml", "instruction.md"):
        (task_dir / legacy).unlink(missing_ok=True)
    front_matter = (task_dir / "task.md").read_text()
    (task_dir / "task.md").write_text(
        front_matter.replace(
            "agent:\n",
            "agent:\n  network_mode: blocklist\n  blocked_urls:\n"
            "  - pypi.org/simple/requests/\n  - api.github.com\n",
            1,
        )
    )
    task = Task(task_dir)
    policy = NetworkPolicy.resolve(task.config)
    assert policy is not None
    rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    rollout_paths.mkdir()
    env = _create_sandbox_environment(
        backend, task, task_dir, "blocklist-canary", rollout_paths
    )
    grant = getattr(env, "grant_egress_filter", None) or getattr(
        env, "grant_egress_firewall", None
    )
    if callable(grant):
        grant()
    await env.start(force_build=False)
    try:
        await setup_sandbox_user(env, "agent", "/app")
        process = await start_egress_filter(env, policy, python="python3")
        await enforce_agent_egress_firewall(
            env,
            "agent",
            policy.agent_env(
                {
                    **process.agent_env,
                    "BENCHFLOW_PROVIDER_BASE_URL": f"http://127.0.0.1:{process.port}/v1",
                }
            ),
        )
        exports = " ".join(
            f"export {k}={shlex.quote(v)};" for k, v in process.agent_env.items()
        )

        async def as_agent(command: str) -> str:
            result = await env.exec(
                build_priv_drop_cmd(f"{exports} {command}", "agent"), timeout_sec=120
            )
            return " ".join(
                ((result.stdout or "") + " " + (result.stderr or "")).split()
            )

        code = "curl -sS -o /dev/null -w '%{http_code}' --max-time 40 "
        assert await as_agent(code + "https://pypi.org/simple/requests/") == "403"
        assert (
            await as_agent(code + "'https://pypi.org/simple/../simple/%72equests/'")
            == "403"
        )
        assert await as_agent(code + "https://pypi.org/simple/pip/") == "200"
        assert "403" in await as_agent(code + "https://api.github.com/")
        assert "403" in await as_agent(code + "https://151.101.0.223/simple/pip/")
        direct = await as_agent(
            "env -u HTTPS_PROXY -u https_proxy " + code + "https://pypi.org/simple/pip/"
        )
        assert "Could not resolve host" in direct or direct.endswith("000")
        assert "denied" in await as_agent(
            f"head -c 1 {process.paths['policy']} 2>&1 || echo denied"
        )
        await process.stop(log_destination=tmp_path / "network_policy.jsonl")
        decisions = [
            json.loads(line)
            for line in (tmp_path / "network_policy.jsonl").read_text().splitlines()
        ]
        assert any(
            d["decision"] == "block" and d["rule"] == "pypi.org/simple/requests/"
            for d in decisions
        )
    finally:
        await env.stop(delete=True)

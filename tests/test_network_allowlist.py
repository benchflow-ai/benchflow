"""network_mode='allowlist': config, capability gate, proxy allow mode, DNS filter, firewall.

The allowlist reuses the denylist egress proxy (``_egress_denylist_proxy``) in an
allow mode. The battery covers the failure modes in Harbor's allowlist bug tail:
harbor-framework/harbor#2146 (allowlisted agent gets ConnectionRefused / model API
unreachable), #2527 (kernel lacks the filtering feature and the run proceeds
anyway), #583 (install and verifier phases need the network; hostname allowlists
resolved once go stale), plus blocked IP literals, DNS rebinding, IPv6 and
redirects to a blocked host.
"""

from __future__ import annotations

import http.server
import json
import socket
import ssl
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import (
    DNS_PORT,
    EGRESS_ALLOW_NETWORKS_ENV,
    EGRESS_ALLOWLIST_ENV,
    EGRESS_DENYLIST_ENV,
    EGRESS_PORT,
    EgressDenylist,
    agent_network_sandbox_config,
    certificate_material,
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
)
from benchflow.sandbox.lockdown import (
    _agent_egress_firewall_cmd,
    enforce_agent_egress_firewall,
)
from benchflow.sandbox.providers import ALLOWLIST_UNSUPPORTED_PROVIDERS
from benchflow.task import TaskConfig, validate_task_runtime_support
from benchflow.task.config import SandboxConfig

# ---------------------------------------------------------------- config


class TestAllowedHostsConfig:
    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("Example.COM.", "example.com"),
            ("*.Example.com", "*.example.com"),
            ("192.0.2.10", "192.0.2.10"),
            ("10.0.0.0/8", "10.0.0.0/8"),
            ("2001:DB8::1", "2001:db8::1"),
            ("2001:db8::/32", "2001:db8::/32"),
        ],
    )
    def test_harbor_entry_kinds_are_accepted(self, raw, normalized):
        """Harbor accepts hostnames, leading wildcards, IPs and CIDRs (harbor config.py:45-64)."""
        config = SandboxConfig(network_mode="allowlist", allowed_hosts=[raw])
        assert config.allowed_hosts == [normalized]

    @pytest.mark.parametrize(
        "raw",
        [
            "*example.com",
            "a.*.example.com",
            "*",
            "*.",
            "10.0.0.1/8",  # host bits set: strict CIDR only
            "10.0.0.0/33",
            "fe80::1%eth0",
            "example.com/24",
        ],
    )
    def test_malformed_entries_are_rejected(self, raw):
        with pytest.raises(ValueError, match="allowed_hosts"):
            SandboxConfig(network_mode="allowlist", allowed_hosts=[raw])

    def test_url_error_string_is_unchanged(self):
        with pytest.raises(ValueError, match="allowed_hosts entries must be hostnames"):
            SandboxConfig(network_mode="allowlist", allowed_hosts=["https://x.com"])


class TestCapabilityGate:
    def test_docker_and_daytona_enforce_sandbox_and_agent_allowlists(self):
        """Lifts the refusal of runtime_capabilities.py:278-283 for validated backends."""
        config = TaskConfig.model_validate(
            {
                "agent": {
                    "network_mode": "allowlist",
                    "allowed_hosts": ["api.example.com"],
                },
                "sandbox": {
                    "network_mode": "allowlist",
                    "allowed_hosts": ["repo.example.com"],
                },
            }
        )
        for sandbox in ("docker", "daytona"):
            assert validate_task_runtime_support(config, sandbox=sandbox) == []

    @pytest.mark.parametrize("sandbox", sorted(ALLOWLIST_UNSUPPORTED_PROVIDERS))
    def test_unvalidated_backends_still_refuse(self, sandbox):
        config = TaskConfig.model_validate(
            {"sandbox": {"network_mode": "allowlist", "allowed_hosts": ["x.com"]}}
        )
        paths = [i.path for i in validate_task_runtime_support(config, sandbox=sandbox)]
        assert "sandbox.network_mode" in paths

    def test_unsupported_set_is_exactly_the_unvalidated_backends(self):
        assert (
            frozenset({"modal", "apple-container", "agentcore"})
            == ALLOWLIST_UNSUPPORTED_PROVIDERS
        )

    def test_verifier_allowlist_is_refused(self):
        """The verifier runs as root outside the uid firewall; its own rule is not an allowlist."""
        config = TaskConfig.model_validate(
            {"verifier": {"network_mode": "allowlist", "allowed_hosts": ["x.com"]}}
        )
        issues = validate_task_runtime_support(config, sandbox="docker")
        assert [i.path for i in issues] == ["verifier.network_mode"]

    @pytest.mark.parametrize("sandbox_mode", ["no-network", "denylist"])
    def test_agent_allowlist_cannot_combine_with_other_sandbox_policies(
        self, sandbox_mode
    ):
        sandbox = {"network_mode": sandbox_mode}
        if sandbox_mode == "denylist":
            sandbox["blocked_hosts"] = ["x.com"]
        config = TaskConfig.model_validate(
            {
                "agent": {"network_mode": "allowlist", "allowed_hosts": ["y.com"]},
                "sandbox": sandbox,
            }
        )
        issues = validate_task_runtime_support(config, sandbox="docker")
        assert "agent.network_mode" in [i.path for i in issues]


class TestPolicyResolution:
    def test_sandbox_allowlist_maps_to_allow_mode_policy(self):
        config = SandboxConfig(
            network_mode="allowlist", allowed_hosts=["pypi.org", "10.0.0.0/8"]
        )
        policy = egress_denylist_for(config)
        assert policy == EgressDenylist(
            (), (), allowed_hosts=("pypi.org", "10.0.0.0/8")
        )
        assert policy.allow_mode

    def test_agent_override_is_folded_into_the_sandbox_config(self):
        """Harbor's recommended form (#2146): [agent] network_mode=allowlist over a public baseline."""
        config = TaskConfig.model_validate(
            {"agent": {"network_mode": "allowlist", "allowed_hosts": ["example.com"]}}
        )
        folded = agent_network_sandbox_config(config)
        assert folded.network_mode == "allowlist"
        assert folded.allowed_hosts == ["example.com"]
        assert config.sandbox.network_mode == "public"  # original untouched
        assert egress_denylist_for(folded).allowed_hosts == ("example.com",)

    def test_no_override_returns_the_same_object(self):
        config = TaskConfig.model_validate({})
        assert agent_network_sandbox_config(config) is config.sandbox

    def test_compose_main_gets_net_admin_for_allowlists(self):
        from benchflow.sandbox._compose import compose_needs_net_admin

        assert compose_needs_net_admin(
            SandboxConfig(network_mode="allowlist", allowed_hosts=["x.com"])
        )

    def test_agent_env_carries_allow_markers(self):
        policy = EgressDenylist(
            (), (), allowed_hosts=("x.com", "10.0.0.0/8", "2001:db8::/32")
        )
        env = denylist_agent_env({}, policy)
        assert env[EGRESS_DENYLIST_ENV] == "1"
        assert env[EGRESS_ALLOWLIST_ENV] == "1"
        assert env[EGRESS_ALLOW_NETWORKS_ENV] == "10.0.0.0/8,2001:db8::/32"
        assert EGRESS_ALLOWLIST_ENV not in denylist_agent_env({})


# ---------------------------------------------------------------- proxy policy


def _allow(*hosts: str, **kwargs) -> proxy_mod.Policy:
    return proxy_mod.Policy([], [], allowed_hosts=list(hosts), **kwargs)


class TestAllowPolicy:
    def test_exact_host_only(self):
        policy = _allow("example.com")
        assert policy.host_rule("example.com", 443) is None
        assert policy.host_rule("EXAMPLE.com.", 443) is None
        assert policy.host_rule("www.example.com", 443) == "not-allowlisted"
        assert policy.host_rule("example.com.evil.test", 443) == "not-allowlisted"
        assert policy.host_rule("other.test", 443) == "not-allowlisted"

    def test_wildcard_matches_subdomains_not_apex(self):
        policy = _allow("*.example.com")
        assert policy.host_rule("a.example.com", 443) is None
        assert policy.host_rule("a.b.example.com", 443) is None
        assert policy.host_rule("example.com", 443) == "not-allowlisted"
        assert policy.host_rule("badexample.com", 443) == "not-allowlisted"

    @pytest.mark.parametrize(
        "host",
        ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946", "::ffff:93.184.216.34"],
    )
    def test_ip_literal_of_an_allowlisted_name_is_refused(self, host):
        """Blocked-by-IP-literal: naming the host lets its name through, not its address."""
        assert _allow("example.com").host_rule(host, 443) == "not-allowlisted"

    @pytest.mark.parametrize(
        "host",
        ["3232235777", "0xc0a80101", "0300.0250.1.1", "127.1", "1-2-3-4.sslip.io"],
    )
    def test_non_canonical_address_notations_are_refused(self, host):
        assert _allow("192.168.1.0/24", "example.com").host_rule(host, 80) in {
            "ip-literal",
            "not-allowlisted",
        }

    def test_cidr_entries_admit_literals_in_range(self):
        policy = _allow("192.0.2.0/24", "2001:db8::/32")
        assert policy.host_rule("192.0.2.77", 443) is None
        assert policy.host_rule("192.0.3.1", 443) == "not-allowlisted"
        assert policy.host_rule("2001:db8::5", 443) is None
        assert policy.host_rule("2001:db9::5", 443) == "not-allowlisted"
        # IPv4-mapped IPv6 is judged by its IPv4 address.
        assert policy.host_rule("::ffff:192.0.2.9", 443) is None

    def test_cidr_entries_do_not_admit_unlisted_names(self):
        """Names never resolve through CIDR entries: DNS for unlisted names would leak."""
        assert _allow("10.0.0.0/8").host_rule("db.internal", 443) == "not-allowlisted"

    def test_address_checks(self):
        policy = _allow("example.com", "10.1.0.0/16", "fd00::/8")
        assert policy.address_allowed("93.184.216.34")
        assert policy.address_allowed("10.1.2.3")
        assert policy.address_allowed("fd00::1")
        assert not policy.address_allowed("10.2.0.1")
        assert not policy.address_allowed("169.254.169.254")
        assert not policy.address_allowed("fe80::1")
        # Loopback is never reached through the root proxy, even when listed.
        assert not _allow("127.0.0.0/8").address_allowed("127.0.0.1")
        assert not _allow("::1").address_allowed("::1")

    def test_gateway_stays_reachable(self):
        """#2146: the agent's model API must be reachable under an allowlist."""
        policy = _allow("example.com", model_gateway_port=4000)
        assert policy.host_rule("127.0.0.1", 4000) is None
        assert policy.host_rule("127.0.0.1", 4001) == "not-allowlisted"

    def test_native_claude_origin_is_model_only(self):
        policy = _allow("example.com", native_claude_model_origin=True)
        assert policy.host_rule("api.anthropic.com", 443) is None
        assert policy.model_only("api.anthropic.com")
        assert not policy.model_only("example.com")
        assert (
            policy.request_rule("api.anthropic.com", 443, "/v1/messages", "POST", True)
            is None
        )
        assert (
            policy.request_rule("api.anthropic.com", 443, "/v1/files", "GET", True)
            == "native-model-request"
        )
        assert policy.host_rule("statsig.anthropic.com", 443) == "not-allowlisted"

    def test_listed_anthropic_is_not_narrowed(self):
        policy = _allow("api.anthropic.com", native_claude_model_origin=True)
        assert not policy.model_only("api.anthropic.com")

    def test_dns_rule(self):
        policy = _allow("example.com", "*.pypi.org", "10.0.0.0/8")
        assert policy.dns_rule("example.com") is None
        assert policy.dns_rule("files.pypi.org.") is None
        assert policy.dns_rule("pypi.org") == "dns-not-allowlisted"
        assert policy.dns_rule("exfil.attacker.test") == "dns-not-allowlisted"

    def test_policy_json_round_trip(self, tmp_path: Path):
        path = tmp_path / "policy.json"
        path.write_text(
            json.dumps(
                {
                    "blocked_urls": [],
                    "blocked_hosts": [],
                    "allowed_hosts": ["example.com"],
                    "model_gateway_port": None,
                    "native_claude_model_only": False,
                    "native_claude_model_origin": False,
                }
            )
        )
        assert proxy_mod.Policy.load(str(path)).allow_mode

    def test_denylist_behaviour_is_unchanged(self):
        policy = proxy_mod.Policy([], ["mirror.test"])
        assert not policy.allow_mode
        assert policy.host_rule("other.test", 443) is None
        assert policy.dns_rule("anything.test") is None


# ---------------------------------------------------------------- in-process proxy battery


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/redirect"):
            target = self.path.split("to=", 1)[1]
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = f"hello {self.headers['Host']} {self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def allow_stack(tmp_path: Path, monkeypatch):
    """Allow-mode proxy; test names resolve to fixed addresses, sockets go to local servers."""
    names = ("allowed.test", "blocked.test", "rebind.test", "v6.test")
    upstream_ca = certificate_material(names)
    proxy_ca = certificate_material(names)
    (tmp_path / "upstream-ca.crt").write_bytes(upstream_ca["ca.crt"])
    (tmp_path / "client-ca.crt").write_bytes(upstream_ca["ca.crt"] + proxy_ca["ca.crt"])
    certs = tmp_path / "certs"
    certs.mkdir()
    for name, material in proxy_ca.items():
        if name.endswith(".pem"):
            (certs / name).write_bytes(material)

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    def sni(sock, server_name, _ctx):
        pem = tmp_path / f"upstream-{server_name}.pem"
        pem.write_bytes(upstream_ca[f"{server_name}.pem"])
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(pem))
        sock.context = ctx

    default_pem = tmp_path / "upstream-default.pem"
    default_pem.write_bytes(upstream_ca["allowed.test.pem"])
    server_ctx.load_cert_chain(str(default_pem))
    server_ctx.sni_callback = sni
    tls = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    tls.socket = server_ctx.wrap_socket(tls.socket, server_side=True)
    plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)

    answers = {
        "allowed.test": [["93.184.216.34"]],
        "blocked.test": [["93.184.216.35"]],
        # DNS rebinding: public at the pre-check, metadata address at connect.
        "rebind.test": [["93.184.216.36"], ["169.254.169.254"]],
        "v6.test": [["fd00::5"]],
    }
    resolved: list[str] = []
    connected: list[str] = []

    def fake_resolve(host, port):
        resolved.append(host)
        try:
            return [host] if proxy_mod.ipaddress.ip_address(host) else []
        except ValueError:
            pass
        seq = answers[host]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    real_create_connection = socket.create_connection

    def fake_create_connection(address, timeout=None, *args, **kwargs):
        if address[0] == "127.0.0.1":  # the test client reaching the proxy
            return real_create_connection(address, timeout, *args, **kwargs)
        connected.append(address[0])
        port = tls.server_address[1] if address[1] == 443 else plain.server_address[1]
        return real_create_connection(("127.0.0.1", port), timeout)

    monkeypatch.setattr(proxy_mod, "_resolve", fake_resolve)
    monkeypatch.setattr(proxy_mod.socket, "create_connection", fake_create_connection)
    log = tmp_path / "blocked.jsonl"
    policy = proxy_mod.Policy(
        [], [], allowed_hosts=["allowed.test", "rebind.test", "v6.test", "192.0.2.0/24"]
    )
    server = proxy_mod.Proxy(
        ("127.0.0.1", 0),
        policy,
        proxy_mod.CertStore(str(certs)),
        proxy_mod.Log(str(log)),
        str(tmp_path / "upstream-ca.crt"),
    )
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True)
        for s in (tls, plain, server)
    ]
    for t in threads:
        t.start()
    proxy_url = f"http://127.0.0.1:{server.server_address[1]}"
    client_ctx = ssl.create_default_context(cafile=str(tmp_path / "client-ca.crt"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}),
        urllib.request.HTTPSHandler(context=client_ctx),
    )
    yield SimpleNamespace(
        opener=opener,
        log=log,
        resolved=resolved,
        connected=connected,
        proxy_url=proxy_url,
    )
    for s in (server, tls, plain):
        s.shutdown()
        s.server_close()


def _status(opener, url: str) -> tuple[int, str]:
    try:
        with opener.open(url, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _rules(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [json.loads(line)["rule"] for line in log.read_text().splitlines()]


class TestAllowProxyBattery:
    def test_allowed_host_is_reachable_over_https_and_http(self, allow_stack):
        assert _status(allow_stack.opener, "https://allowed.test/x") == (
            200,
            "hello allowed.test /x",
        )
        assert _status(allow_stack.opener, "http://allowed.test/y")[0] == 200
        assert _rules(allow_stack.log) == []

    def test_blocked_host_is_refused_without_resolving_it(self, allow_stack):
        with pytest.raises(OSError, match="403"):
            allow_stack.opener.open("https://blocked.test/", timeout=10)
        assert _status(allow_stack.opener, "http://blocked.test/")[0] == 403
        assert "blocked.test" not in allow_stack.resolved
        assert _rules(allow_stack.log) == ["not-allowlisted", "not-allowlisted"]

    def test_blocked_by_ip_literal(self, allow_stack):
        assert _status(allow_stack.opener, "http://93.184.216.34/")[0] == 403
        with pytest.raises(OSError, match="403"):
            allow_stack.opener.open("https://93.184.216.34/", timeout=10)
        assert allow_stack.connected == []

    def test_listed_cidr_literal_is_tunnelled(self, allow_stack):
        code, body = _status(allow_stack.opener, "http://192.0.2.10/z")
        assert code == 200 and body.endswith("/z")
        assert allow_stack.connected == ["192.0.2.10"]

    def test_dns_rebinding_to_metadata_is_refused(self, allow_stack):
        with pytest.raises(OSError, match="403"):
            allow_stack.opener.open("https://rebind.test/", timeout=10)
        assert "169.254.169.254" not in allow_stack.connected
        assert _rules(allow_stack.log)[-1] == "private-address"

    def test_ipv6(self, allow_stack):
        # Listed name with a private ULA answer and no covering CIDR: refused.
        assert _status(allow_stack.opener, "http://v6.test/")[0] == 403
        # Unlisted IPv6 literal: refused before any socket.
        assert _status(allow_stack.opener, "http://[2001:db8::1]/")[0] == 403
        assert allow_stack.connected == []

    def test_redirect_to_blocked_host_is_refused(self, allow_stack):
        code, _ = _status(
            allow_stack.opener,
            "http://allowed.test/redirect?to=http://blocked.test/secret",
        )
        assert code == 403
        assert _rules(allow_stack.log) == ["not-allowlisted"]
        code, _ = _status(
            allow_stack.opener,
            "https://allowed.test/redirect?to=https://allowed.test/ok",
        )
        assert code == 200

    def test_fronting_through_an_allowed_connect_is_refused(self, allow_stack):
        """Shared-CDN fronting: CONNECT allowed.test, inner Host blocked.test."""
        from urllib.parse import urlsplit

        proxy = urlsplit(allow_stack.proxy_url)
        raw = socket.create_connection((proxy.hostname, proxy.port), timeout=10)
        raw.sendall(
            b"CONNECT allowed.test:443 HTTP/1.1\r\nHost: allowed.test:443\r\n\r\n"
        )
        assert raw.recv(4096).startswith(b"HTTP/1.1 200")
        ctx = ssl._create_unverified_context()
        with ctx.wrap_socket(raw, server_hostname="allowed.test") as tls:
            tls.sendall(b"GET / HTTP/1.1\r\nHost: blocked.test\r\n\r\n")
            head = tls.recv(4096)
        assert head.split(b" ", 2)[1] == b"403"


# ---------------------------------------------------------------- DNS filter


def _query(name: str, qtype: int = 1, ident: int = 0x1234) -> bytes:
    qname = b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\0"
    return (
        struct.pack("!HHHHHH", ident, 0x0100, 1, 0, 0, 0)
        + qname
        + struct.pack("!HH", qtype, 1)
    )


class TestDnsFilter:
    def test_unlisted_name_is_refused_without_upstream(self):
        forwarded: list[bytes] = []
        response, rule = proxy_mod.dns_answer(
            _query("exfil-4a6f.attacker.test"), _allow("example.com"), forwarded.append
        )
        assert forwarded == []
        assert rule == "dns-not-allowlisted"
        ident, flags, qd, an = struct.unpack("!HHHH", response[:8])
        assert ident == 0x1234 and flags & 0x8000 and flags & 0xF == 5
        assert (qd, an) == (1, 0)

    def test_listed_name_is_forwarded(self):
        upstream_reply = _query("example.com")[:2] + b"\x81\x80" + b"rest"

        def forward(query: bytes) -> bytes:
            assert query == _query("example.com", qtype=28)
            return upstream_reply

        response, rule = proxy_mod.dns_answer(
            _query("example.com", qtype=28), _allow("example.com"), forward
        )
        assert (response, rule) == (upstream_reply, None)

    @pytest.mark.parametrize(
        "query",
        [
            b"",
            b"\x12\x34",
            _query("example.com")[:-2],
            # Two questions: resolvers disagree about which one counts.
            _query("example.com")[:4] + b"\x00\x02" + _query("example.com")[6:],
            # Compression pointer in the question.
            _query("x")[:12] + b"\xc0\x0c\x00\x01\x00\x01",
        ],
    )
    def test_malformed_queries_are_refused(self, query):
        response, rule = proxy_mod.dns_answer(query, _allow("example.com"), lambda q: q)
        assert rule == "dns-malformed"
        if len(query) >= 2:
            assert response[:2] == query[:2]

    def test_udp_server_end_to_end(self, tmp_path: Path, monkeypatch):
        upstream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        upstream.bind(("127.0.0.1", 0))

        def serve_upstream():
            data, addr = upstream.recvfrom(4096)
            upstream.sendto(data[:2] + b"\x81\x80" + data[4:], addr)

        threading.Thread(target=serve_upstream, daemon=True).start()
        log = tmp_path / "blocked.jsonl"
        port = _free_port()
        servers = proxy_mod.serve_dns(
            port,
            _allow("example.com"),
            proxy_mod.Log(str(log)),
            [f"127.0.0.1:{upstream.getsockname()[1]}"],
        )
        for s in servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(5)
            client.sendto(_query("example.com"), ("127.0.0.1", port))
            reply, _ = client.recvfrom(4096)
            assert reply[2:4] == b"\x81\x80"
            client.sendto(_query("leak.attacker.test"), ("127.0.0.1", port))
            reply, _ = client.recvfrom(4096)
            assert reply[3] & 0xF == 5
            assert _rules(log) == ["dns-not-allowlisted"]
            # TCP: two-byte length prefix.
            with socket.create_connection(("127.0.0.1", port), timeout=5) as tcp:
                q = _query("leak2.attacker.test")
                tcp.sendall(struct.pack("!H", len(q)) + q)
                size = struct.unpack("!H", tcp.recv(2))[0]
                assert tcp.recv(size)[3] & 0xF == 5
        finally:
            for s in servers:
                s.shutdown()
                s.server_close()
            upstream.close()

    def test_resolver_retries_are_logged_once(self, tmp_path: Path):
        """glibc retries each nameserver and A/AAAA: one refusal line per name and burst."""
        log = tmp_path / "blocked.jsonl"
        port = _free_port()
        servers = proxy_mod.serve_dns(
            port, _allow("example.com"), proxy_mod.Log(str(log)), []
        )
        for s in servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(5)
            for qtype in (1, 28, 1, 28):
                client.sendto(_query("leak.attacker.test", qtype), ("127.0.0.1", port))
                client.recvfrom(4096)
            client.sendto(_query("other.attacker.test"), ("127.0.0.1", port))
            client.recvfrom(4096)
        finally:
            for s in servers:
                s.shutdown()
                s.server_close()
        names = [json.loads(line)["url"] for line in log.read_text().splitlines()]
        assert names == ["leak.attacker.test", "other.attacker.test"]

    def test_upstreams_come_from_resolv_conf(self, tmp_path: Path):
        conf = tmp_path / "resolv.conf"
        conf.write_text("# c\nnameserver 127.0.0.11\nnameserver ::1\noptions ndots:0\n")
        assert proxy_mod.resolv_conf_upstreams(str(conf)) == [
            "127.0.0.11:53",
            "[::1]:53",
        ]


# ---------------------------------------------------------------- firewall


class TestAllowFirewall:
    def test_base_command_unchanged_without_allow_mode(self):
        assert "REDIRECT" not in _agent_egress_firewall_cmd("agent")

    def test_allow_mode_redirects_dns_and_admits_listed_networks(self):
        cmd = _agent_egress_firewall_cmd(
            "agent", allow_networks=("10.0.0.0/8", "2001:db8::/32")
        )
        assert f"--to-ports {DNS_PORT}" in cmd
        assert "iptables -t nat" in cmd
        assert "-p udp --dport 53" in cmd and "-p tcp --dport 53" in cmd
        assert "-d 10.0.0.0/8" in cmd
        assert "ip6tables" in cmd and "-d 2001:db8::/32" in cmd
        assert "-d 127.0.0.11" in cmd
        # Redirected DNS is not "-o lo" in the filter chain (as on
        # Daytona): it needs its own accept on the post-NAT destination.
        for proto in ("udp", "tcp"):
            assert f"-d 127.0.0.1 -p {proto} --dport {DNS_PORT}" in cmd
        # #2527: a kernel without the nat table must fail closed, not run open.
        assert "exit 86" in cmd.split("iptables -t nat", 1)[1]

    def test_allow_networks_are_validated_before_shell_use(self):
        with pytest.raises(ValueError):
            _agent_egress_firewall_cmd("agent", allow_networks=("10.0.0.0/8; reboot",))

    async def test_marker_passes_networks_to_the_firewall(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(return_code=0))
        policy = EgressDenylist((), (), allowed_hosts=("x.com", "10.0.0.0/8"))
        await enforce_agent_egress_firewall(
            env, "agent", denylist_agent_env({}, policy)
        )
        env.exec.assert_awaited_once()
        cmd = env.exec.await_args.args[0]
        assert "-d 10.0.0.0/8" in cmd and "REDIRECT" in cmd

    async def test_firewall_failure_is_raised(self):
        """#2527: missing kernel support surfaces as a setup error before the agent runs."""
        env = MagicMock()
        env.exec = AsyncMock(
            return_value=MagicMock(return_code=86, stdout="", stderr="no nat table")
        )
        policy = EgressDenylist((), (), allowed_hosts=("x.com",))
        with pytest.raises(RuntimeError, match="egress firewall"):
            await enforce_agent_egress_firewall(
                env, "agent", denylist_agent_env({}, policy)
            )


class TestStart:
    async def test_policy_json_and_dns_port_reach_the_sandbox(self):
        env = MagicMock()
        env.exec = AsyncMock(
            return_value=MagicMock(return_code=0, stdout="", stderr="")
        )
        env.exec_transient = env.exec
        env.upload_file = AsyncMock()
        uploaded: dict[str, bytes] = {}

        async def upload(local, remote, mode=None):
            uploaded[remote] = Path(local).read_bytes()

        env.upload_file = AsyncMock(side_effect=upload)
        policy = EgressDenylist((), (), allowed_hosts=("example.com",))
        await start_egress_denylist(
            env, "agent", policy, model_gateway_url="http://127.0.0.1:4000"
        )
        data = json.loads(uploaded["/opt/benchflow-egress/policy.json"])
        assert data["allowed_hosts"] == ["example.com"]
        assert data["model_gateway_port"] == 4000
        setup = env.exec.await_args_list[1].args[0]
        assert f"--dns-port {DNS_PORT}" in setup
        assert EGRESS_PORT != DNS_PORT


# ---------------------------------------------------------------- model endpoint admission


class TestModelTransport:
    """#2146: the agent's model API must stay reachable under an allowlist, or fail loudly."""

    POLICY = EgressDenylist((), (), allowed_hosts=("example.com",))

    def test_gateway_agents_keep_the_policy(self, monkeypatch):
        from benchflow.sandbox import native_oauth

        monkeypatch.setattr(
            native_oauth, "uses_native_subscription_auth", lambda *a: False
        )
        assert (
            native_oauth.allowlist_model_transport(
                self.POLICY, "codex-acp", "gpt-5", {}
            )
            is self.POLICY
        )

    def test_native_claude_gets_the_model_origin(self, monkeypatch):
        from benchflow.sandbox import native_oauth

        monkeypatch.setattr(
            native_oauth, "uses_native_subscription_auth", lambda *a: True
        )
        policy = native_oauth.allowlist_model_transport(
            self.POLICY, "claude-agent-acp", "claude-opus-4-8", {}
        )
        assert policy.native_claude_model_origin
        assert policy.allowed_hosts == ("example.com",)
        assert "api.anthropic.com" in policy.inspect_hosts

    def test_other_native_clients_are_refused(self, monkeypatch):
        from benchflow.sandbox import native_oauth

        monkeypatch.setattr(
            native_oauth, "uses_native_subscription_auth", lambda *a: True
        )
        with pytest.raises(ValueError, match="API key"):
            native_oauth.allowlist_model_transport(
                self.POLICY, "codex-acp", "gpt-5", {}
            )

    def test_denylist_is_untouched(self, monkeypatch):
        from benchflow.sandbox import native_oauth

        monkeypatch.setattr(
            native_oauth, "uses_native_subscription_auth", lambda *a: True
        )
        deny = EgressDenylist((), ("x.test",))
        assert (
            native_oauth.allowlist_model_transport(deny, "codex-acp", None, {}) is deny
        )

    def test_task_policy_includes_agent_override(self):
        from benchflow.rollout._setup import _task_egress_denylist

        task = SimpleNamespace(
            config=TaskConfig.model_validate(
                {"agent": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}}
            )
        )
        assert _task_egress_denylist(task) == EgressDenylist(
            (), (), allowed_hosts=("pypi.org",)
        )


# ---------------------------------------------------------------- separate verifier sandbox


class TestSeparateVerifier:
    """The allowlist binds the agent uid; a separate verifier sandbox has no agent."""

    def test_agent_allowlist_does_not_reach_the_verifier_sandbox(self):
        from benchflow.rollout._separate_verifier import _verifier_task
        from benchflow.task.verifier_sandbox import VerifierImage

        config = TaskConfig.model_validate(
            {"agent": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}}
        )
        task = SimpleNamespace(config=config, document=None)
        plan = VerifierImage(source="image", sandbox=SandboxConfig(), context_dir=None)
        view = _verifier_task(task, plan)
        assert agent_network_sandbox_config(view.config).network_mode == "public"
        assert config.agent.network_mode == "allowlist"  # original untouched

    @pytest.mark.parametrize("mode", ["allowlist", "denylist"])
    def test_egress_filters_on_the_verifier_sandbox_are_refused(self, mode):
        sandbox = {"network_mode": mode}
        sandbox["allowed_hosts" if mode == "allowlist" else "blocked_hosts"] = ["x.com"]
        config = TaskConfig.model_validate(
            {"verifier": {"sandbox_mode": "separate", "sandbox": sandbox}}
        )
        issues = validate_task_runtime_support(config, sandbox="docker")
        assert "verifier.sandbox.network_mode" in [i.path for i in issues]

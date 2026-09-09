"""Denylist egress mode: proxy policy, in-process proxy, agent env, start/stop, firewall gate.

Guards the denylist egress mode added for benchflow-ai/FrontierPhysics#365.
"""

from __future__ import annotations

import http.server
import json
import socket
import ssl
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import (
    CA_BUNDLE_PATH,
    CA_CERT_PATH,
    EGRESS_DENYLIST_ENV,
    EGRESS_PORT,
    TRAJECTORY_LOG_NAME,
    EgressDenylist,
    _health_cmd,
    _setup_cmd,
    certificate_material,
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall


class TestPolicy:
    def test_url_prefix_matches_across_scheme_case_and_www(self):
        policy = proxy_mod.Policy(["https://arxiv.org/abs/2401.12345"], [])
        assert (
            policy.url_rule("arxiv.org", "/abs/2401.12345")
            == "url:arxiv.org/abs/2401.12345"
        )
        assert policy.url_rule("www.arxiv.org", "/abs/2401.12345v2?x=1") is not None
        assert policy.url_rule("ARXIV.ORG", "/ABS/2401.12345") is not None
        assert policy.url_rule("arxiv.org", "/abs/2401.12346") is None
        assert policy.url_rule("arxiv.org", "/") is None

    def test_url_prefix_without_scheme_and_percent_encoding(self):
        policy = proxy_mod.Policy(["example.org/Blocked Dir/"], [])
        assert policy.url_rule("example.org", "/blocked%20dir/paper.pdf") is not None
        assert policy.url_rule("example.org", "/blocked") is None

    @pytest.mark.parametrize(
        "path",
        [
            "/abs/../abs/2401.12345",
            "//abs/2401.12345",
            "/abs/./2401.12345",
            "/./abs//2401.12345",
            "/x/../abs/2401.12345",
            "/abs/%2e%2e/abs/2401.12345",
            "/abs/%252e%252e/abs/2401.12345",
            "/%61bs/2401.12345",
            "\\abs\\2401.12345",
            "/abs;v=1/2401.12345;jsessionid=1",
            "/ABS/2401.12345/",
        ],
    )
    def test_path_normalization_defeats_traversal_variants(self, path):
        """Guards the traversal bypass found in review of the denylist mode (FrontierPhysics#365)."""
        policy = proxy_mod.Policy(["https://arxiv.org/abs/2401.12345"], [])
        assert policy.url_rule("arxiv.org", path) == "url:arxiv.org/abs/2401.12345"

    def test_path_normalization_keeps_siblings_open(self):
        policy = proxy_mod.Policy(["https://arxiv.org/abs/2401.12345"], [])
        for path in (
            "/abs/2401.1234",
            "/abs/../pdf/2401.12345",
            "/abs/2401.12345/../2401.99999",
        ):
            assert policy.url_rule("arxiv.org", path) is None

    def test_root_prefix_blocks_whole_host_and_trailing_slash_is_dropped(self):
        assert (
            proxy_mod.Policy(["https://example.org"], []).url_rule(
                "example.org", "/any"
            )
            is not None
        )
        policy = proxy_mod.Policy(["https://example.org/blocked/"], [])
        assert policy.url_rule("example.org", "/blocked/x") is not None
        assert policy.url_rule("example.org", "/blocked") is not None
        assert policy.url_rule("example.org", "/block") is None

    @pytest.mark.parametrize(
        "host",
        [
            "1-2-3-4.sslip.io",
            "1.2.3.4.nip.io",
            "c0a80101.sslip.io",
            "app.10.0.0.1.xip.io",
            "paper.localtest.me",
        ],
    )
    def test_wildcard_dns_names_count_as_addresses(self, host):
        """Guards the wildcard-DNS route around the address rule (FrontierPhysics#365)."""
        assert proxy_mod.Policy([], []).host_rule(host) == "ip-literal"

    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.17.0.1",
            "192.168.1.2",
            "169.254.169.254",
            "::1",
            "fd00::1",
            "100.64.0.1",
            "0.0.0.0",
        ],
    )
    def test_non_global_upstreams_are_refused(self, address):
        assert not proxy_mod._upstream_allowed(address)

    def test_global_upstreams_are_allowed(self):
        assert proxy_mod._upstream_allowed("93.184.216.34")
        assert proxy_mod._upstream_allowed("2606:2800:220:1:248:1893:25c8:1946")

    def test_connect_upstream_refuses_when_any_answer_is_private(self, monkeypatch):
        monkeypatch.setattr(
            proxy_mod, "_resolve", lambda host, port: ["93.184.216.34", "10.0.0.5"]
        )
        with pytest.raises(proxy_mod._PrivateDestination):
            proxy_mod._connect_upstream("rebind.test", 443)

    def test_body_prefix_drops_pipelined_requests(self):
        rest = b"abc" + b"GET /blocked HTTP/1.1\r\nHost: x\r\n\r\n"
        assert proxy_mod._body_prefix([("Content-Length", "3")], rest) == b"abc"
        assert proxy_mod._body_prefix([], rest) == b""
        assert proxy_mod._body_prefix([("Transfer-Encoding", "chunked")], rest) == rest

    def test_host_rule_covers_subdomains_but_not_suffix_lookalikes(self):
        policy = proxy_mod.Policy([], ["example.org"])
        assert policy.host_rule("example.org") == "host:example.org"
        assert policy.host_rule("a.b.example.org") == "host:example.org"
        assert policy.host_rule("evil-example.org") is None
        assert policy.host_rule("example.org.evil.com") is None

    @pytest.mark.parametrize(
        "host",
        [
            "93.184.216.34",
            "2606:2800:220:1:248:1893:25c8:1946",
            "3232235777",
            "0xc0a80101",
            "0300.0250.1.1",
            "127.1",
            "0x7f.1",
            "192.168.1.1.",
        ],
    )
    def test_addresses_in_every_resolver_notation_are_refused(self, host):
        """Guards the non-canonical IPv4 bypass found in review (FrontierPhysics#365)."""
        assert proxy_mod.Policy([], []).host_rule(host) == "ip-literal"

    def test_names_with_numeric_labels_are_still_names(self):
        policy = proxy_mod.Policy([], [])
        assert policy.host_rule("1e100.net") is None
        assert policy.host_rule("3.example.org") is None

    def test_blocked_hosts_keep_www_exact(self):
        policy = proxy_mod.Policy([], ["www.example.org"])
        assert policy.host_rule("www.example.org") == "host:www.example.org"
        assert policy.host_rule("a.www.example.org") == "host:www.example.org"
        assert policy.host_rule("example.org") is None
        assert policy.host_rule("docs.example.org") is None

    def test_only_url_hosts_are_inspected(self):
        policy = proxy_mod.Policy(["https://arxiv.org/abs/1"], ["alphaxiv.org"])
        assert policy.inspect("www.arxiv.org")
        assert not policy.inspect("alphaxiv.org")
        assert not policy.inspect("example.com")

    def test_entry_without_host_is_rejected(self):
        with pytest.raises(ValueError, match="no host"):
            proxy_mod.Policy(["https:///abs"], [])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"hello {self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stack(tmp_path: Path, monkeypatch):
    """Proxy plus TLS and plain upstreams; test hostnames resolve to the local servers."""
    upstream_ca = certificate_material(("paper.test", "other.test"))
    proxy_ca = certificate_material(("paper.test",))
    (tmp_path / "upstream-ca.crt").write_bytes(upstream_ca["ca.crt"])
    (tmp_path / "proxy-ca.crt").write_bytes(proxy_ca["ca.crt"])
    (tmp_path / "client-ca.crt").write_bytes(upstream_ca["ca.crt"] + proxy_ca["ca.crt"])
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / "paper.test.pem").write_bytes(proxy_ca["paper.test.pem"])

    def tls_server(host: str) -> http.server.ThreadingHTTPServer:
        pem = tmp_path / f"upstream-{host}.pem"
        pem.write_bytes(upstream_ca[f"{host}.pem"])
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(pem))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        return server

    tls = tls_server("paper.test")
    tls_other = tls_server("other.test")
    plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    ports = {
        "paper.test": tls.server_address[1],
        "other.test": tls_other.server_address[1],
        "plain.test": plain.server_address[1],
    }

    def fake_connect_upstream(host, port, *, model_gateway_port=None):
        if host == "internal.test":
            raise proxy_mod._PrivateDestination(host)
        return socket.create_connection(("127.0.0.1", ports[host]), timeout=10)

    monkeypatch.setattr(proxy_mod, "_connect_upstream", fake_connect_upstream)
    log = tmp_path / "blocked.jsonl"
    server = proxy_mod.serve(
        _free_port(),
        proxy_mod.Policy(["https://paper.test/abs/2401.12345"], ["mirror.test"]),
        proxy_mod.CertStore(str(certs)),
        proxy_mod.Log(str(log)),
        upstream_ca=str(tmp_path / "upstream-ca.crt"),
    )
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True)
        for s in (tls, tls_other, plain, server)
    ]
    for t in threads:
        t.start()
    proxy_url = f"http://127.0.0.1:{server.server_address[1]}"
    client_ctx = ssl.create_default_context(cafile=str(tmp_path / "client-ca.crt"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}),
        urllib.request.HTTPSHandler(context=client_ctx),
    )
    yield SimpleNamespace(opener=opener, proxy_url=proxy_url, log=log)
    for s in (server, tls, tls_other, plain):
        s.shutdown()
        s.server_close()


def _status(opener, url: str) -> tuple[int, str]:
    try:
        with opener.open(url, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


class TestProxy:
    def test_blocked_url_is_refused_and_logged(self, stack):
        code, body = _status(stack.opener, "https://paper.test/abs/2401.12345v3")
        assert code == 403
        assert "Blocked by the task network policy" in body
        entry = json.loads(stack.log.read_text().splitlines()[-1])
        assert entry["action"] == "blocked"
        assert entry["url"] == "https://paper.test/abs/2401.12345v3"
        assert entry["rule"] == "url:paper.test/abs/2401.12345"

    def test_sibling_path_on_inspected_host_is_served(self, stack):
        code, body = _status(stack.opener, "https://paper.test/abs/1706.03762")
        assert (code, body) == (200, "hello /abs/1706.03762")
        assert not stack.log.exists()

    def test_other_host_is_tunnelled_end_to_end(self, stack):
        code, body = _status(stack.opener, "https://other.test/anything")
        assert (code, body) == (200, "hello /anything")

    def test_plain_http_is_forwarded(self, stack):
        code, body = _status(stack.opener, "http://plain.test/x?y=1")
        assert (code, body) == (200, "hello /x?y=1")

    def test_query_with_scheme_is_not_mistaken_for_absolute_form(self, stack):
        code, body = _status(stack.opener, "https://paper.test/search?q=http://x/y")
        assert (code, body) == (200, "hello /search?q=http://x/y")

    def test_private_destination_is_refused_and_logged(self, stack):
        """Guards the SSRF flag raised on PR #1113: the root proxy must not bridge to internal services."""
        with pytest.raises(OSError, match="403"):
            stack.opener.open("https://internal.test/", timeout=10)
        assert json.loads(stack.log.read_text())["rule"] == "private-address"
        code, _body = _status(stack.opener, "http://internal.test/")
        assert code == 403

    def test_blocked_host_connect_is_refused(self, stack):
        with pytest.raises(OSError, match="403"):
            stack.opener.open("https://mirror.test/", timeout=10)
        assert json.loads(stack.log.read_text())["rule"] == "host:mirror.test"

    def test_healthz(self, stack):
        with urllib.request.urlopen(f"{stack.proxy_url}/healthz", timeout=5) as resp:
            assert resp.status == 200


class TestCertificateMaterial:
    def test_leaf_is_signed_by_ca_and_covers_www(self, tmp_path: Path):
        from cryptography import x509

        material = certificate_material(("paper.test",))
        assert set(material) == {"ca.crt", "paper.test.pem"}
        ca = x509.load_pem_x509_certificate(material["ca.crt"])
        leaf = x509.load_pem_x509_certificate(material["paper.test.pem"])
        assert leaf.issuer == ca.subject
        san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert set(san.get_values_for_type(x509.DNSName)) == {
            "paper.test",
            "www.paper.test",
        }
        assert b"PRIVATE KEY" in material["paper.test.pem"]
        assert b"PRIVATE KEY" not in material["ca.crt"]


class TestAgentEnv:
    def test_env_routes_through_loopback_proxy_and_trusts_ca(self):
        env = denylist_agent_env({"KEEP": "1"})
        assert env["KEEP"] == "1"
        assert env[EGRESS_DENYLIST_ENV] == "1"
        assert (
            env["HTTPS_PROXY"]
            == env["https_proxy"]
            == f"http://127.0.0.1:{EGRESS_PORT}"
        )
        assert "localhost" in env["NO_PROXY"] and "127.0.0.1" in env["no_proxy"]
        assert env["SSL_CERT_FILE"] == env["REQUESTS_CA_BUNDLE"] == CA_BUNDLE_PATH
        assert env["NODE_EXTRA_CA_CERTS"] == CA_CERT_PATH

    def test_input_is_not_mutated(self):
        original = {"A": "1"}
        denylist_agent_env(original)
        assert original == {"A": "1"}

    def test_denylist_for_config(self):
        assert egress_denylist_for(SimpleNamespace(network_mode="public")) is None
        found = egress_denylist_for(
            SimpleNamespace(
                network_mode="denylist",
                blocked_urls=["https://a.test/x"],
                blocked_hosts=None,
            )
        )
        assert found == EgressDenylist(("https://a.test/x",), ())
        assert found.inspect_hosts == ("a.test",)


class TestShellCommands:
    @pytest.mark.parametrize("shell", ["bash", "sh"])
    def test_commands_parse(self, shell):
        for cmd in (_setup_cmd(), _health_cmd()):
            subprocess.run([shell, "-n", "-c", cmd], check=True)

    def test_setup_replaces_a_running_proxy_and_keeps_the_log(self, tmp_path: Path):
        """Guards the restart path found in review: a stale or live pid file must not abort setup, and the block log must survive."""
        runtime, ca, fake_bin = tmp_path / "rt", tmp_path / "ca", tmp_path / "bin"
        (runtime / "certs").mkdir(parents=True)
        fake_bin.mkdir()
        (runtime / "ca.crt").write_text("cert\n")
        python = fake_bin / "python3"
        python.write_text("#!/bin/sh\nexec sleep 60\n")
        python.chmod(0o755)
        (runtime / "proxy.pid").write_text("999999\n")
        (runtime / "blocked.jsonl").write_text('{"action": "blocked"}\n')
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin"}
        cmd = _setup_cmd(runtime_dir=str(runtime), ca_dir=str(ca))
        for _ in range(2):
            result = subprocess.run(
                ["/bin/sh", "-c", cmd],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
        pid = int((runtime / "proxy.pid").read_text())
        subprocess.run(["kill", str(pid)], check=False)
        assert (runtime / "blocked.jsonl").read_text() == '{"action": "blocked"}\n'
        assert (ca / "ca-bundle.crt").read_text().endswith("cert\n")

    def test_setup_fails_closed_without_python(self, tmp_path: Path):
        empty_bin = tmp_path / "bin"
        empty_bin.mkdir()
        result = subprocess.run(
            ["/bin/sh", "-c", _setup_cmd()],
            env={"PATH": str(empty_bin)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 87
        assert "needs python3" in result.stderr
        assert not (tmp_path / "proxy.pid").exists()


def _fake_env(health_rc: int = 0):
    env = MagicMock()

    async def exec_(command, **kwargs):
        rc = health_rc if "healthz" in command else 0
        return MagicMock(return_code=rc, stdout="", stderr="")

    env.exec = AsyncMock(side_effect=exec_)
    env.exec_transient = env.exec
    env.upload_file = AsyncMock()
    env.download_file = AsyncMock()
    return env


class TestStartStop:
    async def test_start_requires_sandbox_user(self):
        with pytest.raises(RuntimeError, match="sandbox_user"):
            await start_egress_denylist(
                _fake_env(), None, EgressDenylist(("https://a.test/x",), ())
            )

    async def test_start_uploads_policy_certs_and_script_then_probes_health(self):
        env = _fake_env()
        await start_egress_denylist(
            env, "agent", EgressDenylist(("https://a.test/x",), ("b.test",))
        )
        uploaded = {call.args[1]: call for call in env.upload_file.await_args_list}
        assert set(uploaded) == {
            "/opt/benchflow-egress/policy.json",
            "/opt/benchflow-egress/proxy.py",
            "/opt/benchflow-egress/ca.crt",
            "/opt/benchflow-egress/certs/a.test.pem",
        }
        assert all(call.kwargs == {"mode": "600"} for call in uploaded.values())
        commands = [call.args[0] for call in env.exec.await_args_list]
        assert all(call.kwargs["user"] == "root" for call in env.exec.await_args_list)
        assert commands[0].startswith("mkdir -p /opt/benchflow-egress/certs")
        assert commands[1] == _setup_cmd()
        assert "healthz" in commands[2]

    async def test_start_raises_with_stderr_when_unhealthy(self):
        env = _fake_env(health_rc=1)
        with pytest.raises(RuntimeError, match="did not become healthy"):
            await start_egress_denylist(
                env, "agent", EgressDenylist((), ("b.test",)), timeout_sec=1
            )

    async def test_stop_downloads_log_and_kills_proxy(self, tmp_path: Path):
        env = _fake_env()
        await stop_egress_denylist(env, tmp_path)
        target = tmp_path / "trajectory" / TRAJECTORY_LOG_NAME
        env.download_file.assert_awaited_once_with(
            "/opt/benchflow-egress/blocked.jsonl", target
        )
        (kill_cmd,) = [c.args[0] for c in env.exec.await_args_list]
        assert (
            "kill -TERM" in kill_cmd
            and "rm -rf /opt/benchflow-egress /etc/benchflow-egress" in kill_cmd
        )

    async def test_stop_falls_back_to_cat_and_never_raises(self, tmp_path: Path):
        env = _fake_env()
        env.download_file = AsyncMock(side_effect=RuntimeError("no cp"))
        env.exec = AsyncMock(
            side_effect=[
                MagicMock(return_code=0, stdout='{"action": "blocked"}\n'),
                RuntimeError("gone"),
            ]
        )
        await stop_egress_denylist(env, tmp_path)
        assert (
            tmp_path / "trajectory" / TRAJECTORY_LOG_NAME
        ).read_text() == '{"action": "blocked"}\n'


class TestFirewallGate:
    async def test_denylist_marker_arms_firewall_without_provider_url(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(return_code=0))
        await enforce_agent_egress_firewall(
            env,
            "agent",
            {
                EGRESS_DENYLIST_ENV: "1",
                "HTTPS_PROXY": f"http://127.0.0.1:{EGRESS_PORT}",
            },
        )
        env.exec.assert_awaited_once()
        assert "iptables" in env.exec.await_args.args[0]
        assert env.exec.await_args.kwargs == {"user": "root", "timeout_sec": 120}

    async def test_denylist_marker_accepts_loopback_provider_url(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(return_code=0))
        await enforce_agent_egress_firewall(
            env,
            "agent",
            {
                EGRESS_DENYLIST_ENV: "1",
                "HTTPS_PROXY": "http://127.0.0.1:18628",
                "LLM_BASE_URL": "http://127.0.0.1:4000",
            },
        )
        env.exec.assert_awaited_once()

    async def test_denylist_marker_rejects_missing_proxy(self):
        env = MagicMock()
        env.exec = AsyncMock()
        with pytest.raises(RuntimeError, match="HTTPS_PROXY"):
            await enforce_agent_egress_firewall(
                env, "agent", {EGRESS_DENYLIST_ENV: "1"}
            )
        env.exec.assert_not_called()

    async def test_denylist_marker_rejects_remote_provider_url(self):
        env = MagicMock()
        env.exec = AsyncMock()
        with pytest.raises(RuntimeError, match="loopback provider base URL"):
            await enforce_agent_egress_firewall(
                env,
                "agent",
                {
                    EGRESS_DENYLIST_ENV: "1",
                    "HTTPS_PROXY": "http://127.0.0.1:18628",
                    "LLM_BASE_URL": "http://172.17.0.1:4000",
                },
            )

    async def test_denylist_marker_requires_sandbox_user(self):
        env = MagicMock()
        env.exec = AsyncMock()
        with pytest.raises(RuntimeError, match="sandbox_user"):
            await enforce_agent_egress_firewall(
                env,
                None,
                {EGRESS_DENYLIST_ENV: "1", "HTTPS_PROXY": "http://127.0.0.1:1"},
            )

    async def test_no_web_without_sandbox_user_still_skips(self):
        env = MagicMock()
        env.exec = AsyncMock()
        await enforce_agent_egress_firewall(
            env, None, {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}
        )
        env.exec.assert_not_called()

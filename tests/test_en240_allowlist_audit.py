"""What an allowlist lets out is logged, and the task's own services stay reachable and closed.

The allowlist itself is 0.8's (``test_network_allowlist.py``). Later work adds the
pieces a real-data task needs before its traffic can be trusted:

* every request the proxy lets out goes to ``egress_allowed.jsonl`` with the
  rule that admitted it, its status and its byte counts;
* a compose service the allowlist names is admitted at its own container
  addresses, so the agent reaches it without listing a private range;
* a task service with a route out of the sandbox refuses the run, because the
  agent could have it fetch what the allowlist refuses.
"""

from __future__ import annotations

import hashlib
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

from benchflow._utils.scoring import is_unrecoverable_startup_error
from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox import egress_denylist as egress_mod
from benchflow.sandbox._docker_task_services import (
    TaskServiceInspectError,
    task_services_from_inspect,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.egress_denylist import (
    EgressDenylist,
    TaskService,
    TaskServiceAddressError,
    TaskServiceEgressError,
    _setup_cmd,
    admit_task_services,
    certificate_material,
    start_egress_denylist,
    task_service_policy,
)
from benchflow.task import TaskConfig, validate_task_runtime_support

# ---------------------------------------------------------------- allowed log


class _Origin(http.server.BaseHTTPRequestHandler):
    def _reply(self, extra: bytes = b"") -> None:
        self.server.paths.append(self.path)  # ty: ignore[unresolved-attribute]
        body = f"hello {self.headers['Host']} {self.path}".encode() + extra
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply()

    def do_POST(self):
        self._reply(self.rfile.read(int(self.headers["Content-Length"])))

    def log_message(self, *args):
        pass


@pytest.fixture
def audit_stack(tmp_path: Path, monkeypatch):
    """An allow-mode proxy with an outside host, a task service and both logs.

    Names resolve to fixed addresses and every upstream socket lands on a
    local origin, so nothing leaves the machine.
    """
    upstream_ca = certificate_material(("allowed.test",))
    proxy_ca = certificate_material(("allowed.test",))
    (tmp_path / "upstream-ca.crt").write_bytes(upstream_ca["ca.crt"])
    (tmp_path / "client-ca.crt").write_bytes(proxy_ca["ca.crt"])
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / "allowed.test.pem").write_bytes(proxy_ca["allowed.test.pem"])
    origin_pem = tmp_path / "origin.pem"
    origin_pem.write_bytes(upstream_ca["allowed.test.pem"])
    origin_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    origin_ctx.load_cert_chain(str(origin_pem))
    tls = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Origin)
    tls.socket = origin_ctx.wrap_socket(tls.socket, server_side=True)
    plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Origin)
    origin_paths: list[str] = []
    tls.paths = plain.paths = origin_paths  # ty: ignore[unresolved-attribute]
    answers = {
        "allowed.test": ["93.184.216.34"],
        "blocked.test": ["93.184.216.35"],
        # The task's service and an unlisted one on the same private network.
        "gitea": ["172.30.0.5"],
        "gitea-db": ["172.30.0.6"],
    }

    def fake_resolve(host, port):
        try:
            return [str(proxy_mod.ipaddress.ip_address(host))]
        except ValueError:
            return answers[host]

    real_create_connection = socket.create_connection

    def fake_create_connection(address, timeout=None, *args, **kwargs):
        if address[0] == "127.0.0.1":  # the test client reaching the proxy
            return real_create_connection(address, timeout, *args, **kwargs)
        port = tls.server_address[1] if address[1] == 443 else plain.server_address[1]
        return real_create_connection(("127.0.0.1", port), timeout)

    monkeypatch.setattr(proxy_mod, "_resolve", fake_resolve)
    monkeypatch.setattr(proxy_mod.socket, "create_connection", fake_create_connection)
    blocked, allowed = tmp_path / "blocked.jsonl", tmp_path / "allowed.jsonl"

    def start(policy: proxy_mod.Policy) -> SimpleNamespace:
        server = proxy_mod.Proxy(
            ("127.0.0.1", 0),
            policy,
            proxy_mod.CertStore(str(certs)),
            proxy_mod.Log(str(blocked)),
            str(tmp_path / "upstream-ca.crt"),
            allowed_log=proxy_mod.Log(str(allowed)),
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": url, "https": url}),
            urllib.request.HTTPSHandler(
                context=ssl.create_default_context(
                    cafile=str(tmp_path / "client-ca.crt")
                )
            ),
        )
        return SimpleNamespace(opener=opener, address=server.server_address)

    servers: list = [tls, plain]
    for s in (tls, plain):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield SimpleNamespace(
        start=start,
        blocked=blocked,
        allowed=allowed,
        answers=answers,
        origin_paths=origin_paths,
        allow=lambda: proxy_mod.Policy(
            [],
            [],
            allowed_hosts=["allowed.test", "gitea", "172.30.0.5/32", "192.0.2.0/24"],
            task_services=["gitea"],
        ),
    )
    for s in servers:
        s.shutdown()
        s.server_close()


def _open(opener, url: str, data: bytes | None = None) -> tuple[int, bytes]:
    try:
        with opener.open(url, data=data, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _rows(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _exchanges(log: Path, count: int = 1) -> list[dict]:
    """The allowed log's exchanges once ``count`` are closed, each record pair
    joined by its id. The ``allowed`` record goes in before the request is
    forwarded; ``closed`` follows when the relay ends, a moment after the client
    read the whole response."""
    for _ in range(100):
        rows = _rows(log)
        closed = [r for r in rows if r["action"] == "closed"]
        if len(closed) >= count:
            break
        threading.Event().wait(0.05)
    rows = _rows(log)
    opened = {r["id"]: r for r in rows if r["action"] == "allowed"}
    return [
        {**opened[r["id"]], **{k: v for k, v in r.items() if k != "action"}}
        for r in rows
        if r["action"] == "closed"
    ]


class TestAllowedLog:
    def test_an_allowed_https_request_is_logged_with_rule_status_and_bytes(
        self, audit_stack
    ):
        """An allowed request leaves a record, not only a refusal."""
        proxy = audit_stack.start(audit_stack.allow())
        code, body = _open(proxy.opener, "https://allowed.test/page?token=abc")
        assert code == 200
        (row,) = _exchanges(audit_stack.allowed)
        assert row["action"] == "allowed" and row["method"] == "GET"
        # Queries are dropped, as in the block log: they can carry credentials.
        assert row["url"] == "https://allowed.test/page"
        assert row["rule"] == "allowed-host" and row["status"] == 200
        assert row["bytes_received"] > len(body)  # the body plus the head
        assert row["bytes_sent"] > len("GET /page?token=abc HTTP/1.1\r\n")
        assert row["ms"] >= 0
        assert _rows(audit_stack.blocked) == []

    def test_the_record_is_written_before_the_request_leaves(self, audit_stack):
        """No record, no egress. A log the proxy cannot append to
        stops the request before the origin sees a byte of it."""
        audit_stack.allowed.mkdir()  # opening it for append now fails
        proxy = audit_stack.start(audit_stack.allow())
        with pytest.raises(OSError):
            proxy.opener.open("http://allowed.test/never", timeout=10)
        threading.Event().wait(0.3)
        assert audit_stack.origin_paths == []

    def test_a_refused_request_is_only_in_the_block_log(self, audit_stack):
        """A refused request is only in the block log."""
        proxy = audit_stack.start(audit_stack.allow())
        assert _open(proxy.opener, "http://blocked.test/")[0] == 403
        with pytest.raises(OSError, match="403"):
            proxy.opener.open("https://blocked.test/", timeout=10)
        assert [r["rule"] for r in _rows(audit_stack.blocked)] == [
            "not-allowlisted",
            "not-allowlisted",
        ]
        threading.Event().wait(0.3)  # a late write would land by now
        assert _rows(audit_stack.allowed) == []

    def test_a_request_body_is_counted(self, audit_stack):
        """A request body is counted."""
        proxy = audit_stack.start(audit_stack.allow())
        payload = b"x" * 5000
        code, body = _open(proxy.opener, "http://allowed.test/upload", data=payload)
        assert code == 200 and body.endswith(payload)
        (row,) = _exchanges(audit_stack.allowed)
        assert row["method"] == "POST" and row["bytes_sent"] > len(payload)
        assert row["bytes_received"] > len(payload)

    def test_a_task_service_is_reached_through_the_proxy_and_labelled(
        self, audit_stack
    ):
        """A named service at its own private address is admitted."""
        proxy = audit_stack.start(audit_stack.allow())
        code, body = _open(proxy.opener, "http://gitea:3000/api/v1/version")
        assert code == 200 and body.startswith(b"hello gitea:3000")
        (row,) = _exchanges(audit_stack.allowed)
        assert row["rule"] == "task-service"
        assert row["url"] == "http://gitea:3000/api/v1/version"

    def test_an_unlisted_neighbour_stays_refused_by_name_and_by_address(
        self, audit_stack
    ):
        """Only the named service's own address is admitted."""
        proxy = audit_stack.start(audit_stack.allow())
        assert _open(proxy.opener, "http://gitea-db:5432/")[0] == 403
        assert _open(proxy.opener, "http://172.30.0.6:5432/")[0] == 403
        assert [r["rule"] for r in _rows(audit_stack.blocked)] == [
            "not-allowlisted",
            "not-allowlisted",
        ]
        threading.Event().wait(0.3)  # a late write would land by now
        assert _rows(audit_stack.allowed) == []

    def test_a_listed_name_that_resolves_to_the_neighbour_is_refused(self, audit_stack):
        """The address check runs at connect time, so a listed
        service name pointed at an unlisted neighbour does not reach it."""
        audit_stack.answers["gitea"] = ["172.30.0.6"]
        proxy = audit_stack.start(audit_stack.allow())
        assert _open(proxy.opener, "http://gitea:3000/")[0] == 403
        assert [r["rule"] for r in _rows(audit_stack.blocked)] == ["private-address"]

    def test_an_opaque_tunnel_is_logged_with_its_byte_counts(self, audit_stack):
        """An opaque tunnel is logged with its byte counts."""
        proxy = audit_stack.start(audit_stack.allow())
        with socket.create_connection(proxy.address, timeout=10) as raw:
            raw.sendall(b"CONNECT 192.0.2.10:443 HTTP/1.1\r\nHost: 192.0.2.10\r\n\r\n")
            assert raw.recv(4096).startswith(b"HTTP/1.1 200")
            ctx = ssl._create_unverified_context()
            with ctx.wrap_socket(raw, server_hostname="allowed.test") as tunnel:
                tunnel.sendall(b"GET / HTTP/1.1\r\nHost: allowed.test\r\n\r\n")
                assert tunnel.recv(4096).startswith(b"HTTP/1.0 200")
        (row,) = _exchanges(audit_stack.allowed)
        assert (row["method"], row["url"], row["rule"]) == (
            "CONNECT",
            "192.0.2.10:443",
            "allowed-range",
        )
        assert row["status"] is None
        assert row["bytes_sent"] > 0 and row["bytes_received"] > 0

    def test_a_tunnel_still_open_is_already_in_the_log(self, audit_stack):
        """A connection open when the proxy stops is recorded."""
        proxy = audit_stack.start(audit_stack.allow())
        with socket.create_connection(proxy.address, timeout=10) as raw:
            raw.sendall(b"CONNECT 192.0.2.10:443 HTTP/1.1\r\nHost: 192.0.2.10\r\n\r\n")
            assert raw.recv(4096).startswith(b"HTTP/1.1 200")
            (row,) = _rows(audit_stack.allowed)
            assert (row["action"], row["url"]) == ("allowed", "192.0.2.10:443")

    def test_the_denylist_logs_what_it_lets_out_too(self, audit_stack):
        """The denylist logs what it lets out too."""
        proxy = audit_stack.start(proxy_mod.Policy([], ["blocked.test"]))
        assert _open(proxy.opener, "http://allowed.test/x")[0] == 200
        (row,) = _exchanges(audit_stack.allowed)
        assert row["rule"] == "not-denylisted"


class TestPolicyLabels:
    def test_task_services_must_be_listed_names(self):
        """Task services must be listed names."""
        with pytest.raises(ValueError, match="task services must be listed"):
            proxy_mod.Policy([], [], allowed_hosts=["a.test"], task_services=["gitea"])

    def test_admitted_by(self):
        """Each record names the rule that let it out."""
        policy = proxy_mod.Policy(
            [],
            [],
            model_gateway_port=4000,
            allowed_hosts=["a.test", "*.b.test", "gitea", "10.0.0.5/32"],
            native_claude_model_origin=True,
            task_services=["gitea"],
        )
        assert policy.admitted_by("127.0.0.1", 4000) == "model-gateway"
        assert policy.admitted_by("api.anthropic.com", 443) == "native-model"
        assert policy.admitted_by("gitea", 3000) == "task-service"
        assert policy.admitted_by("10.0.0.5", 443) == "allowed-range"
        assert policy.admitted_by("x.b.test", 443) == "allowed-host"

    def test_policy_json_carries_the_task_services(self, tmp_path: Path):
        """The policy file carries the task services."""
        path = tmp_path / "policy.json"
        path.write_text(
            json.dumps(
                {
                    "allowed_hosts": ["gitea", "172.30.0.5/32"],
                    "task_services": ["gitea"],
                }
            )
        )
        assert proxy_mod.Policy.load(str(path)).task_services == frozenset({"gitea"})


# ------------------------------------------------------------- task services


def _cid(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _container(service, name, networks, mode=None, v6=None):
    return {
        "Id": _cid(name),
        "Name": f"/{name}",
        "Config": {"Labels": {"com.docker.compose.service": service}},
        "HostConfig": {"NetworkMode": mode or next(iter(networks), "none")},
        "NetworkSettings": {
            "Networks": {
                net: {
                    "NetworkID": f"id-{net}",
                    "IPAddress": ip,
                    "GlobalIPv6Address": (v6 or {}).get(net, ""),
                    "Aliases": [service, name],
                    "DNSNames": [name, service, _cid(name)[:12]],
                }
                for net, ip in networks.items()
            }
        },
    }


_NETWORKS = {
    "id-p_default": {"Id": "id-p_default", "Internal": False, "Driver": "bridge"},
    "id-p_tracker": {"Id": "id-p_tracker", "Internal": True, "Driver": "bridge"},
    "id-p_backend": {"Id": "id-p_backend", "Internal": True, "Driver": "bridge"},
}
_MAIN = _container(
    "main", "p-main-1", {"p_default": "172.20.0.2", "p_tracker": "10.9.0.2"}
)


class TestInspect:
    def test_the_realsvc_layout_has_no_route_out(self):
        """main on default and an internal network; the service on the internal one only."""
        services = task_services_from_inspect(
            [_MAIN, _container("gitea", "p-gitea-1", {"p_tracker": "10.9.0.3"})],
            _NETWORKS,
        )
        (gitea,) = services
        assert gitea.service == "gitea" and gitea.outbound == ()
        assert gitea.addresses == ("10.9.0.3",)
        assert {"gitea", "p-gitea-1"} <= set(gitea.names)

    def test_a_service_on_a_routed_network_has_a_route_out(self):
        """A service on a routed network has a route out."""
        (db,) = task_services_from_inspect(
            [_MAIN, _container("db", "p-db-1", {"p_default": "172.20.0.3"})],
            _NETWORKS,
        )
        assert db.outbound == ("network 'p_default' is not internal",)

    def test_the_host_network_is_a_route_out(self):
        """The host network is a route out."""
        (relay,) = task_services_from_inspect(
            [_MAIN, _container("relay", "p-relay-1", {"host": ""}, mode="host")],
            _NETWORKS,
        )
        assert relay.outbound == ("it uses the host's network",)

    def test_a_borrowed_namespace_has_its_owner_s_routes(self):
        """``network_mode: service:x`` (``container:<id>`` in Docker): main's
        namespace and unknown owners are routes out; an internal-only owner's is not."""
        cache = _container("cache", "p-cache-1", {"p_tracker": "10.9.0.4"})
        routed = _container("web", "p-web-1", {"p_default": "172.20.0.5"})
        borrowers = [
            _container("on-main", "p-on-main-1", {}, mode=f"container:{_MAIN['Id']}"),
            _container(
                "on-cache", "p-on-cache-1", {}, mode=f"container:{cache['Id'][:12]}"
            ),
            _container("on-web", "p-on-web-1", {}, mode=f"container:{routed['Id']}"),
            _container("on-other", "p-on-other-1", {}, mode="container:" + "e" * 64),
        ]
        services = {
            s.service: s
            for s in task_services_from_inspect(
                [_MAIN, cache, routed, *borrowers], _NETWORKS
            )
        }
        assert services["on-main"].outbound == ("it shares main's network namespace",)
        assert services["on-cache"].outbound == ()
        assert services["on-web"].outbound == (
            "through 'web': network 'p_default' is not internal",
        )
        assert services["on-other"].outbound == (
            "it shares the namespace of a container outside the task",
        )

    def test_a_service_without_a_network_has_no_route_out(self):
        """``network_mode: none`` joins the null-driver network, which routes nowhere."""
        none = _container("batch", "p-batch-1", {"none": ""}, mode="none")
        (batch,) = task_services_from_inspect(
            [_MAIN, none],
            {
                **_NETWORKS,
                "id-none": {"Id": "id-none", "Internal": False, "Driver": "null"},
            },
        )
        assert batch.outbound == () and batch.addresses == ()

    def test_an_ipv6_only_service_is_addressed_by_its_v6_address(self):
        """An IPv6-only service is admitted at its IPv6 address."""
        v6 = _container("v6", "p-v6-1", {"p_tracker": ""}, v6={"p_tracker": "fd00::7"})
        (service,) = task_services_from_inspect([_MAIN, v6], _NETWORKS)
        assert service.addresses == ("fd00::7",)

    def test_a_service_main_cannot_reach_gets_no_names(self):
        """A service main cannot reach gets no names."""
        (worker,) = task_services_from_inspect(
            [_MAIN, _container("worker", "p-worker-1", {"p_backend": "10.8.0.4"})],
            _NETWORKS,
        )
        assert worker.outbound == () and worker.names == () and worker.addresses == ()

    def test_missing_or_doubled_main_or_network_record_refuses(self):
        """Missing or doubled main or network record refuses."""
        with pytest.raises(TaskServiceInspectError, match="one running main"):
            task_services_from_inspect([], _NETWORKS)
        other_main = _container("main", "q-main-1", {"p_tracker": "10.9.0.9"})
        with pytest.raises(TaskServiceInspectError, match="found 2"):
            task_services_from_inspect([_MAIN, other_main], _NETWORKS)
        with pytest.raises(TaskServiceInspectError, match="was not inspected"):
            task_services_from_inspect(
                [_MAIN, _container("x", "p-x-1", {"p_other": "10.7.0.2"})], _NETWORKS
            )


_ALLOW = EgressDenylist((), (), allowed_hosts=("allowed.test", "gitea"))


class TestAdmission:
    def test_a_named_service_is_admitted_at_its_own_addresses(self):
        """A named service is admitted at its own addresses."""
        policy = admit_task_services(
            _ALLOW,
            [
                TaskService("gitea", ("gitea", "p-gitea-1"), ("10.9.0.3", "fd00::3")),
                TaskService("gitea-db", ("gitea-db",), ("10.9.0.4",)),
            ],
        )
        assert policy.allowed_hosts == (
            "allowed.test",
            "gitea",
            "10.9.0.3/32",
            "fd00::3/128",
        )
        assert policy.task_services == ("gitea",)
        # Single addresses: the rest of the network, its gateway included, stays closed.
        assert policy.allow_networks == ("10.9.0.3/32", "fd00::3/128")

    def test_nothing_named_leaves_the_policy_as_declared(self):
        """Nothing named leaves the policy as declared."""
        empty = EgressDenylist((), (), allowed_hosts=())
        services = [TaskService("plane-api", ("plane-api",), ("10.9.0.3",))]
        assert admit_task_services(empty, services) is empty

    def test_any_service_with_a_route_out_refuses_the_run(self):
        """A relay the task runs in main is invisible, so every service counts."""
        services = [
            TaskService("gitea", ("gitea",), ("10.9.0.3",)),
            TaskService("cache", outbound=("network 'p_default' is not internal",)),
        ]
        with pytest.raises(TaskServiceEgressError) as refused:
            admit_task_services(EgressDenylist((), (), allowed_hosts=()), services)
        message = str(refused.value)
        assert message.startswith("sandbox startup refused: network_mode='allowlist'")
        assert "'cache' (network 'p_default' is not internal)" in message
        assert "internal: true" in message
        assert is_unrecoverable_startup_error(message)

    @pytest.mark.parametrize(
        "services",
        [
            # Restarting: Docker shows no address until it runs again.
            [TaskService("gitea", ("gitea",), ())],
            # Not on a network main joins: no shared names either.
            [TaskService("gitea")],
        ],
    )
    def test_a_named_service_without_an_address_refuses(self, services):
        """An unreachable named service is the task's, not the agent's, failure."""
        with pytest.raises(TaskServiceAddressError, match="'gitea'") as refused:
            admit_task_services(_ALLOW, services)
        assert str(refused.value).startswith("sandbox startup refused: ")
        # A trial retry can find the service up.
        assert not is_unrecoverable_startup_error(str(refused.value))

    def test_one_addressed_replica_serves_its_service(self):
        """One addressed replica serves its service."""
        policy = admit_task_services(
            _ALLOW,
            [
                TaskService("gitea", ("gitea",), ()),
                TaskService("gitea", ("gitea",), ("10.9.0.3",)),
            ],
        )
        assert policy.allow_networks == ("10.9.0.3/32",)


class _Described:
    """A sandbox class that can describe its task services, one answer per read."""

    def __init__(self, *answers: list[TaskService]) -> None:
        self.answers = list(answers)
        self.reads = 0

    async def egress_task_services(self) -> list[TaskService]:
        self.reads += 1
        return self.answers[min(self.reads, len(self.answers)) - 1]


class TestTaskServicePolicy:
    async def test_denylist_and_no_policy_are_untouched(self):
        """Denylist and no policy are untouched."""
        env = _Described([])
        deny = EgressDenylist((), ("blocked.test",))
        assert await task_service_policy(env, deny) is deny
        assert await task_service_policy(env, None) is None
        assert env.reads == 0

    async def test_a_backend_without_a_description_keeps_the_policy(self):
        """Single-container backends: the capability gate refuses compose tasks there."""
        # A mock answers any attribute; only the sandbox's class is consulted.
        env = MagicMock()
        env.egress_task_services = AsyncMock(side_effect=AssertionError("called"))
        assert await task_service_policy(env, _ALLOW) is _ALLOW

    async def test_a_described_backend_admits_and_refuses(self):
        """A described backend admits and refuses."""
        admitted = await task_service_policy(
            _Described([TaskService("gitea", ("gitea",), ("10.9.0.3",))]), _ALLOW
        )
        assert admitted is not None and admitted.task_services == ("gitea",)
        with pytest.raises(TaskServiceEgressError):
            await task_service_policy(
                _Described(
                    [TaskService("x", outbound=("it uses the host's network",))]
                ),
                _ALLOW,
            )

    async def test_a_named_service_still_starting_is_waited_for(self, monkeypatch):
        """A named service still starting is waited for."""
        monkeypatch.setattr(egress_mod, "_ADDRESS_POLL_SEC", 0)
        env = _Described(
            [TaskService("gitea", ("gitea",), ())],
            [TaskService("gitea", ("gitea",), ("10.9.0.3",))],
        )
        admitted = await task_service_policy(env, _ALLOW)
        assert admitted is not None and admitted.allow_networks == ("10.9.0.3/32",)
        assert env.reads == 2

    async def test_a_named_service_that_never_gets_an_address_refuses(
        self, monkeypatch
    ):
        """A named service that never gets an address refuses."""
        monkeypatch.setattr(egress_mod, "_ADDRESS_POLL_SEC", 0)
        monkeypatch.setattr(egress_mod, "_ADDRESS_ATTEMPTS", 3)
        env = _Described([TaskService("gitea", ("gitea",), ())])
        with pytest.raises(TaskServiceAddressError):
            await task_service_policy(env, _ALLOW)
        assert env.reads == 3


def _docker(results: list) -> DockerSandbox:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._compose_project = "p"
    sandbox._docker_cli = AsyncMock(side_effect=results)  # ty: ignore[invalid-assignment]
    return sandbox


def _result(stdout: str, rc: int = 0):
    return SimpleNamespace(stdout=stdout, stderr="", return_code=rc)


class TestDockerDescription:
    async def test_reads_the_project_from_the_daemon(self):
        """Reads the project from the daemon."""
        containers = [
            _MAIN,
            _container("gitea", "p-gitea-1", {"p_tracker": "10.9.0.3"}),
        ]
        ids = [c["Id"] for c in containers]
        sandbox = _docker(
            [
                _result("\n".join(ids) + "\n"),
                _result(json.dumps(containers)),
                _result(json.dumps(list(_NETWORKS.values()))),
            ]
        )
        (gitea,) = await sandbox.egress_task_services()
        assert gitea.addresses == ("10.9.0.3",) and gitea.outbound == ()
        calls = [c.args[0] for c in sandbox._docker_cli.await_args_list]
        assert calls[0][:4] == ["ps", "-q", "--no-trunc", "--filter"]
        assert calls[0][4] == "label=com.docker.compose.project=p"
        assert calls[1] == ["inspect", *ids]
        assert calls[2] == ["network", "inspect", "id-p_default", "id-p_tracker"]

    async def test_main_alone_is_still_checked_to_be_main(self):
        """Main alone is still checked to be main."""
        sandbox = _docker(
            [
                _result(_MAIN["Id"] + "\n"),
                _result(json.dumps([_MAIN])),
                _result(json.dumps(list(_NETWORKS.values()))),
            ]
        )
        assert await sandbox.egress_task_services() == []

    @pytest.mark.parametrize(
        ("results", "why"),
        [
            ([_result("", rc=1)], "docker ps failed"),
            ([_result("")], "lists no running container"),
            (
                [_result("a" * 64 + "\n" + "b" * 64), _result("", rc=1)],
                "inspect failed",
            ),
            (
                [_result("a" * 64 + "\n" + "b" * 64), _result("not json")],
                "no container records",
            ),
            # The one running container is not main: never read as "no services".
            (
                [
                    _result("a" * 64),
                    _result(
                        json.dumps(
                            [_container("db", "p-db-1", {"p_tracker": "10.9.0.5"})]
                        )
                    ),
                    _result(json.dumps(list(_NETWORKS.values()))),
                ],
                "one running main",
            ),
        ],
    )
    async def test_an_unreadable_project_refuses(self, results, why):
        """An unreadable project refuses."""
        with pytest.raises(
            RuntimeError,
            match="sandbox startup refused: network_mode='allowlist' could not check",
        ) as refused:
            await _docker(results).egress_task_services()
        assert why in str(refused.value)


class TestCapabilityGate:
    """Where the services go unchecked, the task is refused up front."""

    @staticmethod
    def _task(tmp_path: Path, *, compose: bool) -> Path:
        env = tmp_path / "environment"
        env.mkdir(parents=True)
        (env / "Dockerfile").write_text("FROM python:3.12-slim\n")
        if compose:
            (env / "docker-compose.yaml").write_text(
                "services:\n  gitea:\n    image: gitea/gitea\n"
            )
        return tmp_path

    @staticmethod
    def _issues(task_dir: Path | None, sandbox: str) -> list[str]:
        config = TaskConfig.model_validate(
            {"sandbox": {"network_mode": "allowlist", "allowed_hosts": ["gitea"]}}
        )
        return [
            issue.reason
            for issue in validate_task_runtime_support(
                config, sandbox=sandbox, task_dir=task_dir
            )
            if issue.path == "sandbox.network_mode"
        ]

    def test_compose_services_under_an_allowlist_run_only_where_checked(self, tmp_path):
        """Compose services under an allowlist run only where checked."""
        task = self._task(tmp_path, compose=True)
        assert self._issues(task, "docker") == []
        assert self._issues(task, "remote-docker") == []
        (reason,) = self._issues(task, "daytona")
        assert "docker and remote-docker" in reason

    def test_a_single_container_task_is_not_affected(self, tmp_path):
        """A single container task is not affected."""
        assert self._issues(self._task(tmp_path, compose=False), "daytona") == []

    def test_a_config_only_check_leaves_it_to_the_launch_gate(self):
        """The launch gate always has the task's folder (sandbox/setup.py)."""
        assert self._issues(None, "daytona") == []


# ------------------------------------------------------------ proxy plumbing


class TestPlumbing:
    def test_setup_opens_the_allowed_log_root_only(self, tmp_path: Path):
        """Setup opens the allowed log root only."""
        runtime, ca, fake_bin = tmp_path / "rt", tmp_path / "ca", tmp_path / "bin"
        (runtime / "certs").mkdir(parents=True)
        fake_bin.mkdir()
        (runtime / "ca.crt").write_text("cert\n")
        for tool in ("python3", "openssl"):
            exe = fake_bin / tool
            exe.write_text("#!/bin/sh\nexec sleep 60\n")
            exe.chmod(0o755)
        cmd = _setup_cmd(runtime_dir=str(runtime), ca_dir=str(ca))
        assert f"--allowed-log {runtime}/allowed.jsonl" in cmd
        result = subprocess.run(
            ["/bin/sh", "-c", cmd],
            env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        subprocess.run(
            ["kill", (runtime / "proxy.pid").read_text().strip()], check=False
        )
        assert ((runtime / "allowed.jsonl").stat().st_mode & 0o777) == 0o600

    async def test_start_writes_the_task_services_into_the_policy(self, tmp_path: Path):
        """Start writes the task services into the policy."""
        uploaded: dict[str, bytes] = {}

        async def upload_dir(source, target, **_):
            for path in Path(source).rglob("*"):
                if path.is_file():
                    uploaded[str(path.relative_to(source))] = path.read_bytes()

        env = MagicMock()
        env.upload_dir = AsyncMock(side_effect=upload_dir)
        env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="ok"))
        env.exec_transient = env.exec
        policy = EgressDenylist(
            (),
            (),
            allowed_hosts=("gitea", "10.9.0.3/32"),
            task_services=("gitea",),
        )
        await start_egress_denylist(env, "agent", policy, timeout_sec=1)
        written = json.loads(uploaded["policy.json"])
        assert written["allowed_hosts"] == ["gitea", "10.9.0.3/32"]
        assert written["task_services"] == ["gitea"]


# ---------------------------------------------------------------- the rollout


class _ServiceSandbox:
    """A sandbox whose class describes its task services, as ``DockerSandbox``'s does."""

    def __init__(self, services: list[TaskService]) -> None:
        self.services = services

    async def egress_task_services(self) -> list[TaskService]:
        return self.services

    async def exec(self, command: str, **kwargs):
        from benchflow.sandbox._base import ExecResult

        if command.startswith("id -u"):
            return ExecResult(stdout="1500\n", return_code=0)
        return ExecResult(stdout="", return_code=0)


async def test_connect_hands_the_completed_policy_to_the_proxy_and_the_firewall(
    tmp_path: Path,
):
    """connect() admits the named service for both enforcement
    points, the proxy's policy and the uid firewall's listed ranges."""
    from benchflow.acp.session import ACPSession
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox.egress_denylist import EGRESS_ALLOW_NETWORKS_ENV

    r = Rollout(
        RolloutConfig(
            task_path=tmp_path,
            agent="claude-agent-acp",
            model="claude-sonnet-4-6",
            environment="docker",
            sandbox_user="agent",
        )
    )
    r._rollout_dir = tmp_path
    r._rollout_name = "run"
    r._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "fake-oauth"}
    r._agent_launch = "/opt/runtime/bin/claude-agent-acp"
    r._disallow_web_tools = False
    r._egress_denylist = EgressDenylist((), (), allowed_hosts=("gitea",))
    r._reapply_ask_user_handler = lambda: None
    r._attach_trajectory_writer = lambda _: None
    # The firewall-power audit needs a real shell; its own tests cover it.
    r._check_firewall_power = AsyncMock()  # ty: ignore[invalid-assignment]
    r._env = _ServiceSandbox([TaskService("gitea", ("gitea",), ("10.9.0.3",))])
    seen: dict = {}

    async def runtime(**kwargs):
        return kwargs["agent_env"], None

    async def start(env, user, policy, **kwargs):
        seen["policy"] = policy

    async def connect(**kwargs):
        seen["agent_env"] = kwargs["agent_env"]
        session = ACPSession("dummy-session")
        return AsyncMock(session=session), session, AsyncMock(), "claude-agent-acp"

    r._planes = SimpleNamespace(
        ensure_litellm_runtime=runtime,
        start_egress_denylist=start,
        stop_egress_denylist=AsyncMock(),
        connect_acp=connect,
        write_credential_files=AsyncMock(return_value=[]),
        upload_subscription_auth=AsyncMock(return_value=[]),
        apply_web_tool_policy=AsyncMock(),
        quiesce_agent=AsyncMock(),
    )
    await r.connect()
    assert seen["policy"].task_services == ("gitea",)
    assert seen["policy"].native_claude_model_origin is True
    assert "10.9.0.3/32" in seen["policy"].allowed_hosts
    assert seen["agent_env"][EGRESS_ALLOW_NETWORKS_ENV] == "10.9.0.3/32"

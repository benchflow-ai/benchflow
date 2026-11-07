"""The agent-phase network policies hold from inside a real sandbox.

One probe tries each way out of the sandbox as the agent's user and prints
``GI-NETPROBE <json>``: DNS through the sandbox's resolver, its nameservers
and public resolvers (UDP and TCP), direct TCP over IPv4 and IPv6 (and an
IPv4-mapped IPv6 address), ICMP, raw and packet sockets, the host through the
sandbox's gateway, the proxy variables dropped, every loopback listener used
as a forward proxy or as a model proxy told to call elsewhere, and, under an
allowlist, a TLS name that differs from the CONNECT host. A cloud metadata
address is tried only when the rest show egress closed, so no run reaches a
real metadata server. The host end of the gateway check is a listener this
test runs: a connection it accepts from a restricted sandbox fails the test,
whichever relay carried it.

The probe runs as an operator script (``script:``) and, for the agent path a
policy default would cover, as ``claude-agent-acp``'s Bash tool driven by the
deterministic tier's fake model inside the sandbox (no model is billed).

Marked ``deterministic``: a real ``bench`` and a real sandbox. It needs the
internet for the public control (1.1.1.1, example.com).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()

PROBE = r"""
import errno, json, os, random, socket, ssl, struct, urllib.request

HOST_PORT = __HOST_PORT__
out = {"uid": os.getuid()}


def mark(name, state, detail=""):
    out[name] = {"state": state, "detail": str(detail)[:160]}


def why(e):
    return errno.errorcode.get(getattr(e, "errno", None) or -1, type(e).__name__)


def tcp(addr, port, family=socket.AF_INET, timeout=6):
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((addr, port))
        return "open", "connected"
    except OSError as e:
        return "blocked", why(e)


def dns(server, name, proto):
    family = socket.AF_INET6 if ":" in server else socket.AF_INET
    kind = socket.SOCK_DGRAM if proto == "udp" else socket.SOCK_STREAM
    q = struct.pack(">HHHHHH", random.getrandbits(16), 0x0100, 1, 0, 0, 0)
    q += b"".join(bytes([len(p)]) + p.encode() for p in name.split("."))
    q += b"\0\0\1\0\1"
    try:
        with socket.socket(family, kind) as s:
            s.settimeout(5)
            if proto == "udp":
                s.sendto(q, (server, 53))
                data = s.recv(4096)
            else:
                s.connect((server, 53))
                s.sendall(struct.pack(">H", len(q)) + q)
                data = s.recv(4096)[2:]
    except OSError as e:
        return "blocked", why(e)
    if len(data) < 4:
        return "blocked", "short answer"
    rcode = data[3] & 15
    # REFUSED is BenchFlow's DNS filter turning an unlisted name away; any
    # other answer means the name was looked up outside the sandbox.
    return ("blocked" if rcode == 5 else "open"), "rcode %d" % rcode


try:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(("192.0.2.1", 9))  # no packet is sent
    out["address"] = probe.getsockname()[0]
    probe.close()
except OSError as e:
    out["address"] = None
gateway = None
for line in open("/proc/net/route").read().splitlines()[1:]:
    f = line.split()
    if f[1] == "00000000":
        gateway = socket.inet_ntoa(struct.pack("<L", int(f[2], 16)))
out["gateway"] = gateway

# Direct TCP: the host through the gateway, then the internet.
if not HOST_PORT:
    mark("host_route", "n/a", "no host listener on this backend")
elif gateway:
    mark("host_route", *tcp(gateway, HOST_PORT))
else:
    mark("host_route", "blocked", "no default route")
mark("tcp_public", *tcp("1.1.1.1", 443))
mark("tcp6_public", *tcp("2606:4700:4700::1111", 443, socket.AF_INET6))
mark("tcp6_v4_mapped", *tcp("::ffff:1.1.1.1", 443, socket.AF_INET6))
egress_open = "open" in (out["host_route"]["state"], out["tcp_public"]["state"])

# ICMP needs no capability where ping_group_range allows it; raw and packet
# sockets need CAP_NET_RAW.
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP)
    s.settimeout(5)
    s.sendto(struct.pack(">BBHHH", 8, 0, 0, 0, 1) + b"gi", ("1.1.1.1", 0))
    s.recv(1024)
    mark("icmp", "open", "echo reply")
    s.close()
except OSError as e:
    mark("icmp", "blocked", why(e))
for name, args in (
    ("raw_socket", (socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)),
    ("packet_socket", (socket.AF_PACKET, socket.SOCK_RAW, 0x0300)),
):
    try:
        socket.socket(*args).close()
        mark(name, "open", "created")
    except OSError as e:
        mark(name, "blocked", why(e))

# DNS, for a name no list holds.
unlisted = "gi-%08x.example.org" % random.getrandbits(32)
try:
    mark("dns_resolver", "open", socket.getaddrinfo("example.org", 443)[0][4][0])
except OSError as e:
    mark("dns_resolver", "blocked", str(e)[:80])
servers = [l.split()[1] for l in open("/etc/resolv.conf") if l.startswith("nameserver")]
for server in [*servers, "8.8.8.8", "1.1.1.1"]:
    for proto in ("udp", "tcp"):
        if server.startswith("169.254.") and egress_open:
            mark("dns_%s_%s" % (proto, server), "skipped", "egress is open")
        else:
            mark("dns_%s_%s" % (proto, server), *dns(server, unlisted, proto))

# A metadata address, only once the checks above show egress closed.
if egress_open:
    mark("metadata", "skipped", "egress is open")
else:
    mark("metadata", *tcp("169.254.169.254", 80))

# The proxy variables dropped: straight to a host an allowlist may name.
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with direct.open("https://example.com/", timeout=10) as r:
        mark("direct_example_com", "open", "HTTP %d" % r.status)
except Exception as e:
    mark("direct_example_com", "blocked", type(e).__name__ + " " + str(e)[:80])
env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
for url, name in (("https://example.com/", "proxy_listed"), ("https://example.org/", "proxy_unlisted")):
    if not env_proxy:
        mark(name, "n/a", "no proxy variable")
        continue
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            mark(name, "open", "HTTP %d" % r.status)
    except urllib.error.HTTPError as e:
        mark(name, "blocked" if e.code == 403 else "open", "HTTP %d" % e.code)
    except Exception as e:
        mark(name, "blocked", type(e).__name__ + " " + str(e)[:80])
if env_proxy:
    # CONNECT a listed host, then name an unlisted one in TLS.
    host, _, port = env_proxy.split("://", 1)[-1].rstrip("/").rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=10) as s:
            s.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
            head = s.recv(4096)
            if b" 200" not in head.split(b"\r\n")[0]:
                raise OSError("CONNECT refused: %r" % head[:40])
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with ctx.wrap_socket(s, server_hostname="example.org") as t:
                t.sendall(b"GET / HTTP/1.1\r\nHost: example.org\r\nConnection: close\r\n\r\n")
                status = t.recv(64).split(b"\r\n")[0].decode("latin-1")
        mark("tls_name_mismatch", "open" if " 200" in status else "blocked", status)
    except (OSError, ssl.SSLError) as e:
        mark("tls_name_mismatch", "blocked", type(e).__name__ + " " + str(e)[:80])

# Every loopback listener as a forward proxy, and as a model proxy asked to
# call the host listener instead of its upstream.
relays = {}
for path, family in (("/proc/net/tcp", socket.AF_INET), ("/proc/net/tcp6", socket.AF_INET6)):
    try:
        rows = open(path).read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        f = row.split()
        if f[3] != "0A":
            continue
        hexaddr, hexport = f[1].split(":")
        port = int(hexport, 16)
        if family == socket.AF_INET:
            addr = socket.inet_ntoa(struct.pack("<L", int(hexaddr, 16)))
            addr = "127.0.0.1" if addr == "0.0.0.0" else addr
        else:
            addr = "::1"
        if not (addr.startswith("127.") or addr == "::1"):
            continue
        key = "%s:%d" % (addr, port)
        try:
            with socket.create_connection((addr, port), timeout=3) as s:
                s.sendall(b"CONNECT example.org:443 HTTP/1.1\r\nHost: example.org:443\r\n\r\n")
                head = s.recv(200).split(b"\r\n")[0].decode("latin-1")
        except OSError as e:
            head = "error " + why(e)
        relays[key] = head
        if gateway and HOST_PORT:
            target = "http://%s:%d/v1" % (gateway, HOST_PORT)
            body = json.dumps({
                "model": "claude-haiku-4-5", "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
                "api_base": target, "base_url": target,
            }).encode()
            for route in ("/v1/messages", "/v1/chat/completions"):
                try:
                    request = urllib.request.Request(
                        "http://%s:%d%s" % (addr if family == socket.AF_INET else "[::1]", port, route),
                        data=body, headers={"content-type": "application/json"})
                    direct.open(request, timeout=5).close()
                except Exception:
                    pass
out["loopback_relays"] = relays
mark(
    "loopback_forward",
    "open" if any(" 200" in v for v in relays.values()) else "blocked",
    json.dumps(relays)[:160],
)
print("GI-NETPROBE " + json.dumps(out, sort_keys=True))
"""

# Each mode's task.md network lines.
MODES = {
    "public": ("", ""),
    "agent-offline": ("  network_mode: no-network\n", ""),
    "allowlist": (
        "  network_mode: allowlist\n  allowed_hosts:\n    - example.com\n",
        "",
    ),
    "sandbox-offline": ("", "  network_mode: no-network\n"),
}
RESTRICTED = ("agent-offline", "allowlist", "sandbox-offline")
# Vectors that must be closed under every restricted mode.
CLOSED = (
    "host_route",
    "tcp_public",
    "tcp6_public",
    "tcp6_v4_mapped",
    "icmp",
    "raw_socket",
    "packet_socket",
    "dns_resolver",
    "metadata",
    "direct_example_com",
    "loopback_forward",
)

DOCKERFILE = """\
FROM ubuntu:24.04
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \\
    python3 python3-pytest curl wget ca-certificates >/dev/null \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
WORKDIR /app
"""

TASK_MD = """\
---
schema_version: "1.3"
metadata:
  author_name: benchflow tests
  difficulty: easy
  category: sanity
agent:
  timeout_sec: 300
{agent}verifier:
  timeout_sec: 60
sandbox:
  cpus: 1
  memory_mb: 1024
{sandbox}---
# net-{mode}

## prompt

Probe the network.
"""


@contextmanager
def host_listener() -> Iterator[tuple[int, list[str]]]:
    """A TCP listener on every host address; yields its port and the peers it accepted."""
    peers: list[str] = []
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 0))
    server.listen(64)
    server.settimeout(0.5)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, peer = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            peers.append(peer[0])
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1], peers
    finally:
        stop.set()
        thread.join(5)
        server.close()


def _probe_result(text: str) -> dict:
    lines = re.findall(r"GI-NETPROBE (\{.*\})", text)
    assert lines, text[-2000:]
    return json.loads(lines[-1])


def _closed(result: dict, mode: str) -> None:
    """Every way out is closed, and every DNS query was refused or never answered."""
    for vector in CLOSED:
        # The host listener runs beside a local Docker daemon only.
        want = "n/a" if vector == "host_route" and SANDBOX != "docker" else "blocked"
        assert result[vector]["state"] == want, (mode, vector, result[vector])
    for key, value in result.items():
        if key.startswith("dns_"):
            assert value["state"] in {"blocked", "skipped"}, (mode, key, value)
    if "tls_name_mismatch" in result:
        assert result["tls_name_mismatch"]["state"] == "blocked", (mode, result)


@pytest.fixture(scope="module")
def script_probes(tmp_path_factory) -> dict[str, tuple[dict, list[str]]]:
    """The probe as an operator script, one task per mode, in one job."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "script: agents run on Docker")
    root = tmp_path_factory.mktemp("net-script")
    tasks = root / "tasks"
    for mode, (agent, sandbox) in MODES.items():
        task = tasks / f"net-{mode}"
        (task / "environment").mkdir(parents=True)
        (task / "verifier").mkdir()
        (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
        (task / "task.md").write_text(
            TASK_MD.format(agent=agent, sandbox=sandbox, mode=mode)
        )
        test_sh = task / "verifier" / "test.sh"
        test_sh.write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        test_sh.chmod(0o755)
    with host_listener() as (port, peers):
        script = root / "probe.sh"
        script.write_text(
            "#!/bin/bash\npython3 - <<'GIPROBE'\n"
            + PROBE.replace("__HOST_PORT__", str(port))
            + "GIPROBE\n"
        )
        jobs = root / "jobs"
        proc = subprocess.run(
            [
                h.bench_executable(),
                "eval",
                "run",
                "--tasks-dir",
                str(tasks),
                "--agent",
                f"script:{script}",
                "--sandbox",
                "docker",
                "--concurrency",
                "4",
                "--retry-attempts",
                "0",
                "--jobs-dir",
                str(jobs),
            ],
            cwd=h.REPO_ROOT,
            env={**os.environ, "BENCHFLOW_SKIP_UPDATE_CHECK": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
    probes = {}
    for mode in MODES:
        found = list(jobs.glob(f"*/net-{mode}__*/agent/script.txt"))
        assert len(found) == 1, (mode, proc.stdout[-3000:])
        probes[mode] = (_probe_result(found[0].read_text()), list(peers))
    keep = os.environ.get("BENCHFLOW_GI_KEEP")
    if keep:
        shutil.copytree(jobs, Path(keep) / "net-script-jobs", dirs_exist_ok=True)
    return probes


def test_the_probe_sees_an_open_network_as_open(script_probes):
    """The public control: the probe reaches the host listener, the internet and DNS."""
    result, peers = script_probes["public"]
    assert result["host_route"]["state"] == "open", result["host_route"]
    assert result["tcp_public"]["state"] == "open", result["tcp_public"]
    assert result["dns_resolver"]["state"] == "open", result["dns_resolver"]
    assert result["address"] in peers, (result["address"], peers)


@pytest.mark.parametrize("mode", RESTRICTED)
def test_a_restricted_script_agent_finds_no_way_out(script_probes, mode):
    """Raw sockets, DNS, IPv6, dropped proxy variables, a metadata IP and the host route stay closed."""
    result, peers = script_probes[mode]
    _closed(result, mode)
    assert result["address"] not in peers, (mode, result["address"], peers)


def test_the_allowlist_still_admits_its_host_through_the_proxy(script_probes):
    """The allowlist control: example.com through the proxy, example.org refused."""
    result, _ = script_probes["allowlist"]
    assert result["proxy_listed"]["state"] == "open", result["proxy_listed"]
    assert result["proxy_unlisted"]["state"] == "blocked", result["proxy_unlisted"]


# The probe as claude-agent-acp's Bash tool, driven by the fake model inside
# the sandbox: the path a policy default covers.
AGENT_MODES = ("agent-offline", "allowlist")


def _agent_task(dest: Path, mode: str, port: int) -> Path:
    task = dest / f"agent-net-{mode}"
    shutil.copytree(h.TEMPLATE_TASK, task)
    scripts_path = task / "environment" / "fake_llm" / "scripts.json"
    scripts = json.loads(scripts_path.read_text())
    command = (
        "python3 - <<'GIPROBE'\n"
        + PROBE.replace("__HOST_PORT__", str(port))
        + "GIPROBE\nprintf 'Hello, world!\\n' > /app/hello.txt"
    )
    scripts["net-probe"] = [
        {
            "text": "Probing the network.",
            "tool": "Bash",
            "input": {
                "command": command,
                "description": "Probe the network",
                "timeout": 300000,
            },
        },
        {"text": "Probed."},
    ]
    scripts_path.write_text(json.dumps(scripts))
    task_md = task / "task.md"
    text = task_md.read_text().replace(
        "[[fake-llm:hello-pass]]", "[[fake-llm:net-probe]]"
    )
    agent_lines = MODES[mode][0]
    text, n = re.subn(
        r"(?m)^(agent:\n  timeout_sec: [0-9.]+)$",
        rf"\g<1>\n{agent_lines.rstrip()}",
        text,
    )
    assert n == 1
    task_md.write_text(text)
    return task


def _bash_output(trial: Path) -> str:
    for event in h.normalized_acp(trial):
        if event.get("type") == "tool_call" and event.get("tool_name") == "Bash":
            return str(event.get("raw_output") or "")
    return ""


@pytest.fixture(scope="module")
def agent_probes(tmp_path_factory) -> dict[str, tuple[dict, list[str]]]:
    if SANDBOX is None:
        pytest.skip(SKIP_REASON)
    root = tmp_path_factory.mktemp("net-agent")
    tasks = root / "tasks"
    tasks.mkdir()
    with host_listener() as (port, peers):
        for mode in AGENT_MODES:
            _agent_task(tasks, mode, port if SANDBOX == "docker" else 0)
        jobs = root / "jobs"
        args, env = h.bench_command(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=jobs,
            sandbox=SANDBOX,
            host_fake_url=None,
            route="native",
            extra=["--concurrency", "2", "--retry-attempts", "0"],
        )
        # Both policies run the proxy in the sandbox, so its upstream is the
        # sandbox's own fake provider on any backend.
        args = [a for a in args]
        for i, value in enumerate(args):
            if value.startswith("ANTHROPIC_AUTH_TOKEN="):
                args[i] = f"ANTHROPIC_API_KEY={h.DUMMY_KEY}"
            elif value.startswith("ANTHROPIC_BASE_URL="):
                args[i] = f"BENCHFLOW_PROVIDER_BASE_URL={h.IN_SANDBOX_FAKE_URL}"
        proc = subprocess.run(
            args,
            cwd=h.REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2400,
        )
    probes = {}
    for mode in AGENT_MODES:
        found = [p.parent for p in jobs.glob(f"*/agent-net-{mode}__*/result.json")]
        assert len(found) == 1, (mode, proc.stdout[-3000:])
        probes[mode] = (_probe_result(_bash_output(found[0])), list(peers))
    keep = os.environ.get("BENCHFLOW_GI_KEEP")
    if keep:
        shutil.copytree(
            jobs, Path(keep) / f"net-agent-jobs-{SANDBOX}", dirs_exist_ok=True
        )
    return probes


@pytest.mark.parametrize("mode", AGENT_MODES)
def test_a_restricted_model_agent_finds_no_way_out(agent_probes, mode):
    """claude-agent-acp's own tool calls under agent-phase no-network and an allowlist."""
    result, peers = agent_probes[mode]
    _closed(result, mode)
    assert result["address"] not in peers, (mode, result["address"], peers)
    if mode == "allowlist":
        assert result["proxy_listed"]["state"] == "open", result["proxy_listed"]


# The cloud metadata endpoints under the open network, for a script: agent.
# They are mocked inside the sandbox (tests/_metadata_mock.py), so no run
# connects to a real metadata server; the probe is the deterministic tier's.
METADATA_TASK = "net-metadata"


@pytest.fixture(scope="module")
def script_metadata(tmp_path_factory) -> Path:
    from tests._metadata_mock import MockWatcher

    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "the mock runs on Docker only")
    root = tmp_path_factory.mktemp("net-metadata")
    task = root / "tasks" / METADATA_TASK
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "task.md").write_text(TASK_MD.format(agent="", sandbox="", mode="metadata"))
    test_sh = task / "verifier" / "test.sh"
    test_sh.write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    test_sh.chmod(0o755)
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    script = root / "metadata-probe.sh"
    script.write_text(
        "#!/bin/bash\n" + scripts["metadata-probe"][0]["input"]["command"] + "\n"
    )

    def wanted(_cid: str, labels: dict[str, str]) -> bool:
        return labels.get("com.docker.compose.service") == "main" and labels.get(
            "com.docker.compose.project", ""
        ).startswith(f"{METADATA_TASK}__")

    jobs = root / "jobs"
    with MockWatcher(wanted, "bf.test.gi-metadata-mock=1") as mock:
        proc = subprocess.run(
            [
                h.bench_executable(),
                "eval",
                "run",
                "--tasks-dir",
                str(root / "tasks"),
                "--agent",
                f"script:{script}",
                "--sandbox",
                "docker",
                "--retry-attempts",
                "0",
                "--jobs-dir",
                str(jobs),
            ],
            cwd=h.REPO_ROOT,
            env={**os.environ, "BENCHFLOW_SKIP_UPDATE_CHECK": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
    assert not mock.errors, mock.errors
    assert len(mock.mains) == 1, (mock.mains, proc.stdout[-3000:])
    found = [p.parent for p in jobs.glob(f"*/{METADATA_TASK}__*/result.json")]
    assert len(found) == 1, proc.stdout[-3000:]
    return found[0]


def test_a_script_agent_on_an_open_network_cannot_reach_the_metadata_endpoints(
    script_metadata,
):
    """Fails on 42bb217b, where a script: agent under the public network got no metadata block."""
    output = (script_metadata / "agent" / "script.txt").read_text()
    probes = dict(re.findall(r"(?m)^probe (\S+) (\S+)$", output))
    assert probes and set(probes.values()) == {"refused"}, probes
    record = json.loads((script_metadata / "cloud-metadata.json").read_text())
    assert record["state"] == "on", record

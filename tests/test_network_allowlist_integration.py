"""Live canary for network_mode='allowlist'.

Run with ``uv run pytest -m integration tests/test_network_allowlist_integration.py``.
Docker needs a local daemon; Daytona needs DAYTONA_API_KEY and sandbox-daytona.
No model API: the probe runs actual sockets and DNS lookups as the sandbox user
after the proxy and uid firewall are installed, the way a rollout arms them.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow.sandbox.egress_denylist import (
    agent_network_sandbox_config,
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task

ALLOWED = ["example.com", "httpbin.org", "*.sslip.io", "1.1.1.1/32"]
# Compose variant: a sibling service by name plus the bridge ranges Docker
# assigns to compose networks.
COMPOSE_ALLOWED = [*ALLOWED, "sidecar", "172.16.0.0/12", "192.168.0.0/16"]


def write_allowlist_task(root: Path, *, compose: bool = False) -> Path:
    task = root / "allowlist-canary"
    environment = task / "environment"
    environment.mkdir(parents=True)
    if compose:
        (environment / "docker-compose.yaml").write_text(
            "services:\n"
            "  sidecar:\n"
            "    image: python:3.12-slim\n"
            '    command: ["python", "-m", "http.server", "8080"]\n'
        )
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl iptables openssl && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m -s /bin/bash agent\nWORKDIR /app\n"
    )
    allowed = COMPOSE_ALLOWED if compose else ALLOWED
    hosts = "".join(f"    - {json.dumps(h)}\n" for h in allowed)
    (task / "task.md").write_text(
        '---\nschema_version: "1.3"\n'
        "agent:\n  network_mode: allowlist\n  allowed_hosts:\n"
        + hosts
        + "sandbox:\n  cpus: 1\n  memory_mb: 2048\n  storage_mb: 4096\n"
        "  workdir: /app\n---\nCheck the network policy.\n"
    )
    return task


# Every check records its outcome; the controller asserts on the JSON.
PROBE = dedent("""\
    import json, os, socket, urllib.error, urllib.request
    assert os.getuid() != 0, 'probe must run as the agent'
    out = {'uid': os.getuid()}

    def status(url):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.status, r.read(2000).decode('utf-8', 'replace'), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(2000).decode('utf-8', 'replace'), dict(e.headers)
        except OSError as e:
            return 'error', str(e), {}

    code, body, _ = status('https://example.com/')
    out['allowed_https'] = code == 200 and 'Example Domain' in body
    code, body, _ = status('http://example.com/')
    out['allowed_http'] = code == 200
    code, body, headers = status('http://example.org/')
    out['blocked_http'] = code == 403 and headers.get('X-BenchFlow-Blocked') == '1'
    code, body, _ = status('https://example.org/')
    out['blocked_https'] = code != 200 and '403' in str(body)
    ip = os.environ['ALLOW_CANARY_IP']
    code, _, headers = status('http://' + ip + '/')
    out['blocked_ip_literal'] = code == 403 and headers.get('X-BenchFlow-Blocked') == '1'
    ip6 = os.environ.get('ALLOW_CANARY_IP6', '')
    if ip6:
        code, _, headers = status('http://[' + ip6 + ']/')
        out['blocked_ipv6_literal'] = code == 403
    # A listed wildcard name that points at the metadata service / loopback:
    # the proxy re-checks the resolved address at connect time.
    for name in ('169-254-169-254.sslip.io', '127-0-0-1.sslip.io'):
        code, _, headers = status('http://' + name + '/')
        out['rebind_' + name.split('.')[0]] = code == 403
    # Redirect from an allowed host to a blocked one: the client's second
    # request is judged on its own.
    code, body, headers = status('https://httpbin.org/redirect-to?url=http%3A%2F%2Fexample.org%2F')
    out['redirect_status'] = code
    out['redirect_to_blocked_refused'] = code == 403 and headers.get('X-BenchFlow-Blocked') == '1'
    # DNS: listed names resolve for the agent, unlisted ones do not.
    try:
        out['dns_allowed'] = bool(socket.getaddrinfo('example.com', 443))
    except OSError as e:
        out['dns_allowed'] = 'error: ' + str(e)
    for key, name in (('dns_refused_unlisted', 'example.org'),
                      ('dns_refused_random', 'leak-%s.example.net' % os.urandom(4).hex())):
        try:
            socket.getaddrinfo(name, 443)
            out[key] = False
        except OSError:
            out[key] = True
    # Direct sockets: a listed CIDR is reachable, anything else is not.
    def direct(addr, family=socket.AF_INET):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(8)
                s.connect((addr, 443))
                return True
        except OSError:
            return False
    out['direct_cidr_listed'] = direct('1.1.1.1')
    out['direct_cidr_unlisted'] = direct('1.0.0.1')
    out['direct_allowed_name_ip'] = direct(ip)
    if ip6:
        out['direct_ipv6'] = direct(ip6, socket.AF_INET6)
    if os.environ.get('ALLOW_CANARY_COMPOSE') == '1':
        import time
        for _ in range(30):
            code, body, _ = status('http://sidecar:8080/')
            if code == 200:
                break
            time.sleep(2)
        out['compose_sibling_via_proxy'] = code == 200
        try:
            with socket.create_connection(('sidecar', 8080), timeout=8):
                out['compose_sibling_direct'] = True
        except OSError as e:
            out['compose_sibling_direct'] = 'error: ' + str(e)
    print(json.dumps(out))
    """)


RESOLVE = dedent("""\
    import socket
    print(socket.gethostbyname('example.com'))
    try:
        print(socket.getaddrinfo('example.com', 443, socket.AF_INET6)[0][4][0])
    except OSError:
        print('')
    """)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("compose", [False, True], ids=["single", "compose"])
@pytest.mark.parametrize("backend", ["docker", "daytona"])
async def test_allowlist_sandbox_canary(tmp_path: Path, backend: str, compose: bool):
    """Guards the allowlist against the known allowlist failure modes.

    The compose variant runs Docker's engine semantics (embedded DNS at
    127.0.0.11, the NET_ADMIN overlay); on Daytona it uses the DinD strategy.
    """
    if backend == "daytona":
        if not os.environ.get("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        pytest.importorskip("daytona")
    else:
        if not shutil.which("docker"):
            pytest.skip("Docker not installed")
        if subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        ).returncode:
            pytest.skip("Docker daemon unavailable")

    task_path = write_allowlist_task(tmp_path, compose=compose)
    task = Task(task_path)
    policy = egress_denylist_for(agent_network_sandbox_config(task.config))
    assert policy is not None and policy.allow_mode
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        backend,
        task,
        task_path,
        f"allowlist-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    evidence: dict = {}
    try:
        await sandbox.start(force_build=False)
        resolve = await sandbox.exec(
            "python3 -c " + shlex.quote(RESOLVE),
            user="root",
            timeout_sec=30,
        )
        assert resolve.return_code == 0, resolve.stderr
        lines = [*(resolve.stdout or "").strip().splitlines(), "", ""]
        await start_egress_denylist(sandbox, "agent", policy)
        agent_env = denylist_agent_env(
            {
                "ALLOW_CANARY_IP": lines[0].strip(),
                "ALLOW_CANARY_IP6": lines[1].strip(),
                "ALLOW_CANARY_COMPOSE": "1" if compose else "0",
            },
            policy,
        )
        await enforce_agent_egress_firewall(sandbox, "agent", agent_env)
        result = await sandbox.exec(
            "python3 -c " + shlex.quote(PROBE),
            user="agent",
            env=agent_env,
            timeout_sec=300,
        )
        assert result.return_code == 0, (result.stdout or "") + (result.stderr or "")
        evidence = json.loads((result.stdout or "").strip().splitlines()[-1])
        # The verifier's own rule: root is outside the uid firewall.
        root = await sandbox.exec(
            "curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 30 https://example.org/",
            user="root",
            timeout_sec=45,
        )
        evidence["root_unfiltered"] = (root.stdout or "").strip() == "200"
        firewall = await sandbox.exec(
            "iptables -t nat -S OUTPUT; iptables -S OUTPUT; ip6tables -S OUTPUT 2>&1 | head -20",
            user="root",
            timeout_sec=30,
        )
        evidence["firewall"] = (firewall.stdout or "").splitlines()
    finally:
        try:
            await stop_egress_denylist(sandbox, rollout_dir)
        finally:
            await sandbox.stop(delete=True)

    out_dir = os.environ.get("ALLOWLIST_CANARY_EVIDENCE")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        lane = f"{backend}-{'compose' if compose else 'single'}"
        (Path(out_dir) / f"{lane}.json").write_text(json.dumps(evidence, indent=2))
        log = rollout_dir / "trajectory" / "egress_denylist.jsonl"
        if log.exists():
            (Path(out_dir) / f"{lane}-blocked.jsonl").write_text(log.read_text())

    for key in (
        "allowed_https",
        "allowed_http",
        "blocked_http",
        "blocked_https",
        "blocked_ip_literal",
        "rebind_169-254-169-254",
        "rebind_127-0-0-1",
        "redirect_to_blocked_refused",
        "dns_refused_unlisted",
        "dns_refused_random",
        "direct_cidr_listed",
        "root_unfiltered",
    ):
        assert evidence.get(key) is True, (key, evidence)
    assert evidence["dns_allowed"] is True, evidence
    if compose:
        assert evidence["compose_sibling_via_proxy"] is True, evidence
        assert evidence["compose_sibling_direct"] is True, evidence
    assert evidence["direct_cidr_unlisted"] is False, evidence
    assert evidence["direct_allowed_name_ip"] is False, evidence
    if "blocked_ipv6_literal" in evidence:
        assert evidence["blocked_ipv6_literal"] is True, evidence
        assert evidence["direct_ipv6"] is False, evidence
    rules = [
        json.loads(line)["rule"]
        for line in (rollout_dir / "trajectory" / "egress_denylist.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "not-allowlisted" in rules
    assert "private-address" in rules
    assert "dns-not-allowlisted" in rules

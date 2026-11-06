"""Live Docker canary: an allowlist with the task's own compose services.

Run with ``uv run pytest -m integration tests/test_en240_allowlist_integration.py``
on a machine with a Docker daemon. No model API. The task puts two services on
an ``internal: true`` network that ``main`` also joins (the layout of the
real-data tasks: Gitea, Plane, Paperless, Grafana); the allowlist names one of
them and no outside host but ``example.com``. The probe runs as the sandbox
user after the proxy and the uid firewall are armed the way a rollout arms
them, and tests the property rather than a list: any host that is not listed
is refused, whatever it is, and the named service stays reachable.
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
    ALLOWED_LOG_NAME,
    TRAJECTORY_LOG_NAME,
    TaskServiceEgressError,
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
    task_service_policy,
)
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_SERVICE = '    image: python:3.12-slim\n    command: ["python", "-m", "http.server", "8080"]\n'


def write_task(root: Path, *, sidecar_networks: str) -> Path:
    task = root / "allowlist-services"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n    networks: [default, svc]\n"
        f"  sidecar:\n{_SERVICE}    networks: {sidecar_networks}\n"
        f"  neighbour:\n{_SERVICE}    networks: [svc]\n"
        "networks:\n  svc:\n    internal: true\n"
    )
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl iptables openssl && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m -s /bin/bash agent\nWORKDIR /app\n"
    )
    (task / "task.md").write_text(
        '---\nschema_version: "1.3"\n'
        "sandbox:\n  network_mode: allowlist\n"
        '  allowed_hosts: ["example.com", "sidecar"]\n'
        "  cpus: 1\n  memory_mb: 2048\n  storage_mb: 4096\n  workdir: /app\n"
        "---\nCheck the network policy.\n"
    )
    return task


PROBE = dedent("""\
    import json, os, socket, time, urllib.error, urllib.request
    assert os.getuid() != 0, 'probe must run as the agent'
    out = {}

    def status(url):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError as e:
            return 'error: ' + str(e)

    def direct(addr, port):
        try:
            with socket.create_connection((addr, port), timeout=5):
                return True
        except OSError:
            return False

    def resolves(name):
        try:
            return bool(socket.getaddrinfo(name, 443))
        except OSError:
            return False

    out['allowed_host'] = status('https://example.com/')
    # Any other host, whatever it is: never listed, so refused.
    out['unlisted'] = {h: status('https://' + h + '/') for h in (
        'example.org', 'www.wikipedia.org', 'leak-%s.example.net' % os.urandom(4).hex())}
    out['unlisted_dns'] = {h: resolves(h) for h in ('example.org', 'www.wikipedia.org')}
    for _ in range(30):
        out['sidecar_via_proxy'] = status('http://sidecar:8080/')
        if out['sidecar_via_proxy'] == 200:
            break
        time.sleep(2)
    out['sidecar_dns'] = resolves('sidecar')
    out['sidecar_direct'] = direct('sidecar', 8080) if out['sidecar_dns'] else False
    out['neighbour_via_proxy'] = status('http://neighbour:8080/')
    out['neighbour_dns'] = resolves('neighbour')
    out['neighbour_direct'] = direct(os.environ['NEIGHBOUR_IP'], 8080)
    gateway = os.environ.get('SVC_GATEWAY', '')
    out['gateway_direct'] = {p: direct(gateway, p) for p in (22, 80, 2375)} if gateway else {}
    out['public_direct'] = direct(os.environ['PUBLIC_IP'], 443)
    # UDP to port 53 goes to the DNS filter by design; any other port is refused.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(3)
            s.sendto(b'\\x00', (os.environ['PUBLIC_IP'], 443))
            s.recvfrom(512)
            out['public_udp'] = True
    except OSError:
        out['public_udp'] = False
    print(json.dumps(out))
    """)


SIDECAR_PROBE = dedent("""\
    import json, socket, sys
    def reach(addr, port):
        try:
            with socket.create_connection((addr, port), timeout=5):
                return True
        except OSError:
            return False
    public, gateway = sys.argv[1], sys.argv[2]
    print(json.dumps({"public": reach(public, 443),
                      "gateway": {p: reach(gateway, p) for p in (22, 80)} if gateway else {}}))
    """)


def _host(url: str) -> str:
    """The host of a logged ``scheme://host/path`` or ``host:port`` (a tunnel)."""
    return url.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]


def _docker_ready() -> None:
    if not shutil.which("docker"):
        pytest.skip("Docker not installed")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker daemon unavailable")


async def _root(sandbox, command: str) -> str:
    result = await sandbox.exec(command, user="root", timeout_sec=60)
    assert result.return_code == 0, (result.stdout or "") + (result.stderr or "")
    return (result.stdout or "").strip()


async def test_named_service_reachable_everything_else_refused(tmp_path: Path):
    """Guards the property, the services and the logs on Docker."""
    _docker_ready()
    task_path = write_task(tmp_path, sidecar_networks="[svc]")
    task = Task(task_path)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"allowlist-svc-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    evidence: dict = {}
    try:
        await sandbox.start(force_build=False)
        resolve = (
            "python3 -c 'import socket, sys; print(socket.gethostbyname(sys.argv[1]))' "
        )
        public_ip = await _root(sandbox, resolve + "example.com")
        neighbour_ip = await _root(sandbox, resolve + "neighbour")
        network = f"{sandbox.compose_project_name}_svc"
        inspected = subprocess.run(
            ["docker", "network", "inspect", network],
            capture_output=True,
            text=True,
            timeout=30,
        )
        evidence["svc_network"] = json.loads(inspected.stdout or "[{}]")[0]
        gateways = [
            c.get("Gateway", "")
            for c in (evidence["svc_network"].get("IPAM") or {}).get("Config") or []
        ]
        evidence["gateway"] = next((g for g in gateways if g), "")
        policy = await task_service_policy(
            sandbox, egress_denylist_for(task.config.sandbox)
        )
        assert policy is not None and policy.task_services == ("sidecar",)
        evidence["allowed_hosts"] = list(policy.allowed_hosts or ())
        await start_egress_denylist(sandbox, "agent", policy)
        agent_env = denylist_agent_env(
            {
                "PUBLIC_IP": public_ip,
                "NEIGHBOUR_IP": neighbour_ip,
                "SVC_GATEWAY": evidence["gateway"],
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
        evidence["probe"] = json.loads((result.stdout or "").strip().splitlines()[-1])
        evidence["iptables"] = (
            await _root(
                sandbox,
                "iptables -t nat -S OUTPUT; iptables -S OUTPUT; "
                "ip6tables -S OUTPUT 2>/dev/null || true",
            )
        ).splitlines()
        # The task service itself: no route to the internet. The Docker host
        # still answers at the internal network's gateway (a documented limit).
        sidecar = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={sandbox.compose_project_name}",
                "--filter",
                "label=com.docker.compose.service=sidecar",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.split()[0]
        evidence["from_sidecar"] = json.loads(
            subprocess.run(
                [
                    "docker",
                    "exec",
                    sidecar,
                    "python",
                    "-c",
                    SIDECAR_PROBE,
                    public_ip,
                    evidence["gateway"],
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            ).stdout
        )
    finally:
        try:
            await stop_egress_denylist(sandbox, rollout_dir)
        finally:
            await sandbox.stop(delete=True)

    trajectory = rollout_dir / "trajectory"
    blocked = [
        json.loads(x)
        for x in (trajectory / TRAJECTORY_LOG_NAME).read_text().splitlines()
    ]
    allowed = [
        json.loads(x) for x in (trajectory / ALLOWED_LOG_NAME).read_text().splitlines()
    ]
    out_dir = os.environ.get("ALLOWLIST_CANARY_EVIDENCE")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "docker-services.json").write_text(
            json.dumps({**evidence, "blocked": blocked, "allowed": allowed}, indent=2)
        )

    probe = evidence["probe"]
    assert probe["allowed_host"] == 200, probe
    assert all(code != 200 for code in probe["unlisted"].values()), probe
    assert not any(probe["unlisted_dns"].values()), probe
    assert probe["sidecar_via_proxy"] == 200, probe
    assert probe["sidecar_dns"] is True and probe["sidecar_direct"] is True, probe
    assert probe["neighbour_via_proxy"] == 403, probe
    assert probe["neighbour_dns"] is False and probe["neighbour_direct"] is False, probe
    assert not any(probe["gateway_direct"].values()), probe
    assert probe["public_direct"] is False and probe["public_udp"] is False, probe
    assert evidence["from_sidecar"]["public"] is False, evidence["from_sidecar"]
    rules = {row["rule"] for row in blocked}
    assert {"not-allowlisted", "dns-not-allowlisted"} <= rules, rules
    opened = [row for row in allowed if row["action"] == "allowed"]
    closed = {row["id"]: row for row in allowed if row["action"] == "closed"}
    admitted = {(row["rule"], _host(row["url"])) for row in opened}
    assert ("allowed-host", "example.com") in admitted, allowed
    assert ("task-service", "sidecar") in admitted, allowed
    # Every admitted request is closed, with what it carried.
    assert {row["id"] for row in opened} == set(closed), allowed
    assert all(row["bytes_received"] > 0 for row in closed.values()), allowed


async def test_a_service_with_a_route_out_refuses_the_run(tmp_path: Path):
    """A task service on a routed network fails closed before any agent."""
    _docker_ready()
    task_path = write_task(tmp_path, sidecar_networks="[default, svc]")
    task = Task(task_path)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"allowlist-out-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        with pytest.raises(TaskServiceEgressError, match="'sidecar' \\(network '"):
            await task_service_policy(sandbox, egress_denylist_for(task.config.sandbox))
    finally:
        await sandbox.stop(delete=True)

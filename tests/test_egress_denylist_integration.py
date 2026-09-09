"""Live adoption canaries for the denylist introduced by PR #1113.

Run with ``uv run pytest -m integration tests/test_egress_denylist_integration.py``.
Docker needs a local daemon; Daytona needs DAYTONA_API_KEY and sandbox-daytona.
These tests exercise actual sockets as the sandbox user without a model API.
Set BENCHFLOW_DENYLIST_MODEL and provider credentials to also run a full
model rollout (Gemini by default). BENCHFLOW_DENYLIST_AGENT and
BENCHFLOW_DENYLIST_EFFORT select a different native harness and effort.
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
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task


def write_denylist_task(root: Path) -> Path:
    """A small native task using the same config path as a research task."""
    task = root / "denylist-canary"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl iptables && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m -s /bin/bash agent\nWORKDIR /app\n"
    )
    (task / "task.md").write_text(
        dedent("""\
        ---
        schema_version: "1.3"
        sandbox:
          network_mode: denylist
          blocked_urls:
            - https://example.com/paper/2401.12345
            - https://arxiv.org/abs/2401.12345
            - https://arxiv.org/pdf/2401.12345
            - https://arxiv.org/html/2401.12345
          blocked_hosts:
            - example.net
          cpus: 1
          memory_mb: 2048
          storage_mb: 4096
          workdir: /app
        ---
        Check the network policy.
        """)
    )
    return task


# No mocked DNS, proxy or firewall: a blocked request must have the policy
# header, an allowed sibling must return origin content, and direct sockets
# must fail even when the caller deliberately ignores every proxy variable.
PROBE = dedent("""\
    import json, os, socket, urllib.error, urllib.request
    assert os.getuid() != 0, 'probe must run as the agent'
    with urllib.request.urlopen('https://example.com/', timeout=30) as r:
        assert r.status == 200
        assert b'Example Domain' in r.read(), 'missing origin content'
    urls = [
        'http://example.com/paper/2401.12345',
        'https://example.com/paper/2401.12345v2?download=1',
        'https://example.com/%70aper/2401.12345',
        'https://arxiv.org/abs/2401.12345v2',
        'https://arxiv.org/pdf/2401.12345.pdf?download=1',
        'https://arxiv.org/html/2401.12345',
    ]
    for url in urls:
        try:
            urllib.request.urlopen(url, timeout=30)
        except urllib.error.HTTPError as e:
            assert e.code == 403, (url, e.code)
            assert e.headers.get('X-BenchFlow-Blocked') == '1', url
        else:
            raise AssertionError('blocked URL was reachable: ' + url)
    for host in ('example.net', 'mirror.example.net'):
        # Plain HTTP exposes the refusal header (CONNECT errors do not).
        try:
            urllib.request.urlopen('http://' + host + '/', timeout=30)
        except urllib.error.HTTPError as e:
            assert e.code == 403 and e.headers.get('X-BenchFlow-Blocked') == '1'
        else:
            raise AssertionError('blocked host was reachable: ' + host)
    address = os.environ['DENYLIST_CANARY_IP']
    try:
        with socket.create_connection((address, 443), timeout=5):
            raise AssertionError('direct egress bypassed the firewall')
    except OSError:
        pass
    print(json.dumps({'allowed_origin': True, 'blocked_requests': len(urls) + 2,
                      'direct_egress_blocked': True, 'uid': os.getuid()}))
    """)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "daytona"])
async def test_research_denylist_sandbox_canary(tmp_path: Path, backend: str):
    """Guards PR #1113's real Docker/Daytona enforcement for FrontierPhysics #366."""
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

    task_path = write_denylist_task(tmp_path)
    task = Task(task_path)
    policy = egress_denylist_for(task.config.sandbox)
    assert policy is not None
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    paths = RolloutPaths(rollout_dir)
    sandbox = _create_sandbox_environment(
        backend, task, task_path, f"denylist-{uuid.uuid4().hex[:12]}", paths
    )
    try:
        await sandbox.start(force_build=False)
        address = await sandbox.exec(
            "python3 -c 'import socket; print(socket.gethostbyname(\"example.com\"))'",
            user="root",
            timeout_sec=30,
        )
        assert address.return_code == 0, address.stderr
        await start_egress_denylist(sandbox, "agent", policy)
        # Daytona's DNS server is outside loopback: resolution as the agent
        # is correctly blocked. Resolve as root to test TCP bypass separately.
        agent_env = denylist_agent_env({"DENYLIST_CANARY_IP": address.stdout.strip()})
        await enforce_agent_egress_firewall(sandbox, "agent", agent_env)
        result = await sandbox.exec(
            "python3 -c " + shlex.quote(PROBE),
            user="agent",
            env=agent_env,
            timeout_sec=180,
        )
        assert result.return_code == 0, result.stdout + result.stderr
        evidence = json.loads(result.stdout)
        assert evidence["blocked_requests"] == 8
        # The verifier/oracle UID must still be able to use direct egress.
        result = await sandbox.exec(
            "curl --noproxy '*' --fail --max-time 30 https://example.com/",
            user="root",
            timeout_sec=45,
        )
        assert result.return_code == 0 and "Example Domain" in result.stdout
        # Reconnecting another scene must preserve the earlier block log.
        await start_egress_denylist(sandbox, "agent", policy)
    finally:
        try:
            await stop_egress_denylist(sandbox, rollout_dir)
        finally:
            await sandbox.stop(delete=True)

    log = rollout_dir / "trajectory" / "egress_denylist.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 8
    assert all(event["action"] == "blocked" for event in events)
    assert any(event["rule"] == "host:example.net" for event in events)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "daytona"])
async def test_research_denylist_model_rollout(tmp_path: Path, backend: str):
    """Guards PR #1113's model-gateway lifecycle with an actual native agent."""
    import socket

    from benchflow import RolloutConfig, Scene, run

    model = os.environ.get("BENCHFLOW_DENYLIST_MODEL")
    if not model:
        pytest.skip("Set BENCHFLOW_DENYLIST_MODEL and its provider credentials")
    if backend == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
        pytest.skip("DAYTONA_API_KEY not set")
    agent = os.environ.get("BENCHFLOW_DENYLIST_AGENT", "gemini")
    task = write_denylist_task(tmp_path)
    probe = PROBE.replace(
        "os.environ['DENYLIST_CANARY_IP']", repr(socket.gethostbyname("example.com"))
    )
    (task / "environment" / "probe.py").write_text(probe)
    with (task / "environment" / "Dockerfile").open("a") as f:
        f.write("COPY probe.py /app/probe.py\nRUN chmod 444 /app/probe.py\n")
    document = task / "task.md"
    document.write_text(
        document.read_text().replace(
            "---\nCheck the network policy.",
            "verifier:\n  type: test-script\n  timeout_sec: 60\n---\n"
            "Run `python3 /app/probe.py > /app/network_probe.json` once. "
            "Do not modify probe.py. Report the observed result and stop.",
        )
    )
    verifier = task / "verifier"
    verifier.mkdir()
    (verifier / "test.sh").write_text(
        dedent("""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p /logs/verifier
        python3 - <<'PYTHON'
        import json
        from pathlib import Path
        data = json.loads(Path('/app/network_probe.json').read_text())
        assert data['allowed_origin'] is True
        assert data['blocked_requests'] == 8
        assert data['direct_egress_blocked'] is True
        assert data['uid'] != 0
        Path('/logs/verifier/network_probe.json').write_text(json.dumps(data))
        Path('/logs/verifier/reward.txt').write_text('1.0')
        PYTHON
        """)
    )
    result = await run(
        RolloutConfig(
            task_path=task,
            scenes=[
                Scene.single(
                    agent=agent,
                    model=model,
                    reasoning_effort=os.environ.get("BENCHFLOW_DENYLIST_EFFORT"),
                )
            ],
            environment=backend,
            jobs_dir=tmp_path / "jobs",
            timeout=600,
        )
    )
    assert result.error is None and result.verifier_error is None
    assert result.n_tool_calls > 0 and result.rewards == {"reward": 1.0}
    logs = list((tmp_path / "jobs").rglob("egress_denylist.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert len(events) == 8 and all(e["action"] == "blocked" for e in events)

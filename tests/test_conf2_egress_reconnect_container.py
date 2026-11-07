"""PT-C8-CONF-2: after the agent's turn, the egress
proxy's reconnect never starts the task image's Python as root.

A real Docker sandbox with root's protected kit runs the denylist proxy,
then has an agent's turn: the agent leaves ``urllib.py`` and ``ssl.py`` in
the working folder, and every interpreter of the image records each start
(its user and arguments) before running as before. The proxy then dies (a
lost route). The reconnect's health check must see it, the reconnect must
bring the proxy back and the route must block again as the agent, with no
root start of the image's Python and no import of the agent's modules: the
proxy, its CA probe and its health check run on the kit's Python. Fails on
c210416e4, where all three start ``python3`` from root's route in the
working folder.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow._runtime_tree import KIT
from benchflow.sandbox import egress_denylist
from benchflow.sandbox.egress_denylist import (
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import prepare_runtime_tree
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except Exception:
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _docker_ready(), reason="Docker daemon required")

STARTS = "/var/tmp/conf2-python-starts"
SHADOW = "/var/tmp/conf2-shadow-imported"

# Every python of the image (python3.12 and the links to it) becomes a
# recorder that runs the real interpreter: "<uid> <args>" per start.
RECORDER = dedent(f"""\
    set -e
    real=/usr/local/bin/.conf2-real-python3.12
    mv /usr/local/bin/python3.12 "$real"
    printf '%s\\n' '#!/bin/sh' \\
      'echo "$(id -u) $*" >> {STARTS}' \\
      "exec $real \\"\\$@\\"" > /usr/local/bin/python3.12
    chmod 755 /usr/local/bin/python3.12
    : > {STARTS}; chmod 666 {STARTS}
    """)

# What the agent leaves in its working folder: stdlib names that, imported,
# mark the import and fail.
SHADOW_MODULE = f"open({SHADOW!r}, 'w').write('shadow')\nraise RuntimeError('shadow')\n"

# The route as the agent: a blocked host gets the proxy's refusal.
AGENT_PROBE = dedent("""\
    import os, urllib.error, urllib.request
    assert os.getuid() != 0
    try:
        urllib.request.urlopen('http://example.net/', timeout=20)
    except urllib.error.HTTPError as e:
        assert e.code == 403 and e.headers.get('X-BenchFlow-Blocked') == '1', e.code
    else:
        raise AssertionError('blocked host was reachable')
    print('blocked')
    """)


def _write_task(root: Path) -> Path:
    task = root / "conf2-egress"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "openssl ca-certificates && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m -s /bin/bash agent && mkdir -p /app && chown agent /app\n"
        "WORKDIR /app\n"
    )
    (task / "task.md").write_text(
        dedent("""\
        ---
        schema_version: "1.3"
        sandbox:
          network_mode: denylist
          blocked_hosts:
            - example.net
          cpus: 1
          memory_mb: 2048
          storage_mb: 4096
          workdir: /app
        ---
        Reconnect the egress proxy.
        """)
    )
    return task


async def _root(sandbox, command: str):
    return await sandbox.exec(command, user="root", timeout_sec=60)


@pytest.mark.asyncio
async def test_the_reconnect_check_finds_a_lost_route_and_recovers_it_on_the_kit(
    tmp_path: Path,
):
    task_path = _write_task(tmp_path)
    task = Task(task_path)
    policy = egress_denylist_for(task.config.sandbox)
    assert policy is not None
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"conf2-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        # Before the agent's turn: root's kit, then the proxy.
        await prepare_runtime_tree(sandbox)
        kit = await _root(sandbox, f"test -x {KIT}/py/python3 && cat {KIT}/RECORD")
        assert kit.return_code == 0, kit.stdout + kit.stderr
        await start_egress_denylist(sandbox, "agent", policy, timeout_sec=60)

        # The agent's turn: its modules in its working folder; from here on
        # every start of the image's Python is recorded.
        for name in ("urllib.py", "ssl.py"):
            planted = await sandbox.exec(
                f"printf %s {shlex.quote(SHADOW_MODULE)} > /app/{name}",
                user="agent",
                timeout_sec=30,
            )
            assert planted.return_code == 0, planted.stderr
        recorder = await _root(sandbox, RECORDER)
        assert recorder.return_code == 0, recorder.stderr

        # The route is lost: the proxy dies.
        pid = (await _root(sandbox, f"cat {egress_denylist._PID_PATH}")).stdout.strip()
        assert pid.isdecimal()
        killed = await _root(
            sandbox,
            f"kill -KILL {pid}; for i in 1 2 3 4 5 6 7 8 9 10; do "
            f"kill -0 {pid} 2>/dev/null || exit 0; sleep 0.2; done; exit 1",
        )
        assert killed.return_code == 0

        # The reconnect's check, from the agent's working folder, sees it.
        lost = await _root(sandbox, egress_denylist._health_cmd())
        assert lost.return_code != 0

        # The reconnect brings the proxy back and the route blocks again.
        await start_egress_denylist(sandbox, "agent", policy, timeout_sec=30)
        healthy = await _root(sandbox, egress_denylist._health_cmd())
        assert healthy.return_code == 0, healthy.stdout + healthy.stderr
        new_pid = (
            await _root(sandbox, f"cat {egress_denylist._PID_PATH}")
        ).stdout.strip()
        # The kit's interpreter (started through the system loader when it
        # carries its own libraries), not the image's.
        argv = await _root(sandbox, f"tr '\\000' ' ' < /proc/{new_pid}/cmdline")
        assert f"{KIT}/libexec/python3 " in argv.stdout, argv.stdout
        blocked = await sandbox.exec(
            "cd / && python3 -I -c " + shlex.quote(AGENT_PROBE),
            user="agent",
            env=denylist_agent_env({}, policy),
            timeout_sec=60,
        )
        assert blocked.return_code == 0, blocked.stdout + blocked.stderr
        assert blocked.stdout.strip() == "blocked"

        # No root start of the image's Python, no import of the agent's files.
        starts = (await _root(sandbox, f"cat {STARTS}")).stdout.splitlines()
        assert starts, "the recorder saw no start at all"
        assert all(not line.startswith("0 ") for line in starts), starts
        shadow = await _root(sandbox, f"test ! -e {SHADOW}")
        assert shadow.return_code == 0, "an agent module was imported"
    finally:
        try:
            await stop_egress_denylist(sandbox, rollout_dir)
        finally:
            await sandbox.stop(delete=True)

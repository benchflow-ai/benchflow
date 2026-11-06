"""PT-C8-CONF-2b round 3: on an image nobody changed after root's kit, the
pre-agent record holds every interpreter the pytest guard installer meets,
so none is "unrecorded": each takes its guard copy as before, with no
``benchflow:`` line and no guard on the verifier's PYTHONPATH.

Real Docker sandboxes, each image as built, kit and an agent turn that
changes nothing, then the installer on PID 1's PATH and on the fallback
verifier PATH:

- ``last``: python:3.12-slim + pytest whose PATH ends in ``/usr/local/bin``,
  the folder of its only Python. An honest test.sh,
  ``export PYTHONPATH=$(pwd); pytest``, run as root with the verifier's
  PYTEST_ADDOPTS and PYTHONPATH, scores 1;
- ``merged-usr``: debian:bookworm-slim with apt's python3 and pytest, where
  ``/bin`` (PATH's last folder) is a link to ``/usr/bin``;
- the four TB2 images review A named (chess-best-move,
  constraints-scheduling, db-wal-recovery, fix-code-vulnerability).

On e4090a15f the kit's capture never visited PATH's last folder or the
system folders: ``last`` has no copy and its test.sh fails to import the
guard, and the other five name ``/bin/python3*`` as unrecorded.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow._image_python_paths import GUARD_SYSTEM_PATH
from benchflow.sandbox.lockdown import (
    _install_pytest_plugin_guard,
    prepare_runtime_tree,
)
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

_AGENT = (
    "RUN id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
    "mkdir -p /app && chown agent /app\nWORKDIR /app\n"
)

IMAGES = {
    "last": "FROM python:3.12-slim\n"
    "RUN pip install --no-cache-dir pytest==8.3.4\n"
    "ENV PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin\n" + _AGENT,
    "merged-usr": "FROM debian:bookworm-slim\n"
    "RUN apt-get update && apt-get install -y --no-install-recommends "
    "python3 python3-pytest && rm -rf /var/lib/apt/lists/*\n" + _AGENT,
    **{
        task: f"FROM alexgshaw/{task}:20251031\n" + _AGENT
        for task in (
            "chess-best-move",
            "constraints-scheduling",
            "db-wal-recovery",
            "fix-code-vulnerability",
        )
    },
}

# The honest test.sh of #1180's shape: its own PYTHONPATH, then pytest.
TEST_SH = dedent("""\
    #!/bin/bash
    cd /tests
    export PYTHONPATH=$(pwd)
    if pytest -q -p no:cacheprovider test_outputs.py; then
        echo 1 > /logs/verifier/reward.txt
    else
        echo 0 > /logs/verifier/reward.txt
    fi
    """)


def _write_task(root: Path, dockerfile: str) -> Path:
    task = root / "conf2b-r3"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "task.md").write_text(
        dedent("""\
        ---
        schema_version: "1.3"
        sandbox:
          cpus: 1
          memory_mb: 2048
          storage_mb: 4096
          workdir: /app
        ---
        Install the pytest guard.
        """)
    )
    return task


async def _run(sandbox, command: str, user: str = "root"):
    done = await sandbox.exec(command, user=user, timeout_sec=180)
    done.stdout = done.stdout or ""
    return done


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(IMAGES))
async def test_an_unchanged_image_has_no_unrecorded_interpreter(
    tmp_path: Path, kind, caplog
):
    task_path = _write_task(tmp_path, IMAGES[kind])
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_path),
        task_path,
        f"conf2b-r3-{kind}-{uuid.uuid4().hex[:10]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        await prepare_runtime_tree(sandbox)
        turn = await _run(sandbox, "echo done > /app/out.txt", "agent")
        assert turn.return_code == 0, turn.stderr
        path = (
            await _run(
                sandbox, "tr '\\000' '\\n' < /proc/1/environ | sed -n 's/^PATH=//p'"
            )
        ).stdout.strip()
        assert path.startswith("/"), path

        installs = {}
        for verifier_path in (path, GUARD_SYSTEM_PATH):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="benchflow.sandbox.lockdown"):
                (
                    directory,
                    flags,
                    guarded,
                    unrecorded,
                ) = await _install_pytest_plugin_guard(
                    sandbox, "agent", "/app", "", verifier_path=verifier_path
                )
            notes = [
                r.getMessage() for r in caplog.records if "benchflow:" in r.getMessage()
            ]
            installs[verifier_path] = (directory, flags, guarded, unrecorded, notes)

        # The installer on PID 1's PATH, as hardening runs it.
        directory, flags, guarded, unrecorded, _ = installs[path]
        reward = None
        if kind == "last":
            reward = await _test_sh_reward(
                sandbox, path, directory, flags, guarded, unrecorded
            )

        for verifier_path, (_, _, guarded, unrecorded, notes) in installs.items():
            assert unrecorded == (), (verifier_path, unrecorded)
            assert notes == [], (verifier_path, notes)
            assert guarded, verifier_path
        # PATH's last folder holds a Python on each of these images; it is
        # guarded too.
        last = path.rsplit(":", 1)[-1]
        assert any(p.startswith(last + "/") for p in installs[path][2]), (
            last,
            installs[path][2],
        )
        if kind == "last":
            assert reward == "1", reward
    finally:
        await sandbox.stop(delete=True)


async def _test_sh_reward(sandbox, path, directory, flags, guarded, unrecorded):
    """TEST_SH's reward, run as root with the verifier's pytest settings."""
    # The verifier's PYTHONPATH carries the guard only for a Python without
    # a copy, or one the record does not hold (lockdown._assemble_verifier_env).
    pythonpath = directory if (not guarded or unrecorded) else ""
    made = await _run(
        sandbox,
        "mkdir -p /tests /logs/verifier && printf %s "
        + shlex.quote("def test_x():\n    assert True\n")
        + " > /tests/test_outputs.py && printf %s "
        + shlex.quote(TEST_SH)
        + " > /tests/test.sh && chmod 755 /tests/test.sh",
    )
    assert made.return_code == 0, made.stderr
    ran = await _run(
        sandbox,
        "env "
        + shlex.quote(f"PATH={path}")
        + " "
        + shlex.quote(f"PYTEST_ADDOPTS={flags}")
        + " "
        + shlex.quote(f"PYTHONPATH={pythonpath}")
        + " /tests/test.sh 2>&1; echo REWARD $(cat /logs/verifier/reward.txt)",
    )
    print(ran.stdout)
    return ran.stdout.rsplit("REWARD ", 1)[-1].strip()

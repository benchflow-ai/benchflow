"""PT-C8-CONF-2b round 3 (review B of round 2, M1): a
separate verifier sandbox never starts an interpreter the transfer brought,
as any uid, whether or not the task pins pytest plugins.

Real Docker, review B's layout: python:3.12-slim with a workspace venv
``/app/.venv`` holding pytest, and a regular (copied, not linked)
``/usr/local/bin/pytest`` whose shebang is ``#!/app/.venv/bin/python``.
The task has no plugin pins and no ctrf. Then, as the separate verifier
runs:

1. ``prepare_separate_verifier_runtime`` on the fresh image (``kit``), or
   nothing, as on a sandbox that has no kit (``no-kit``);
2. the "transfer": ``/app/.venv/bin/python`` becomes a root-owned 0755
   script that records each start;
3. the PATH judge with the separate verifier's blocked prefixes, then the
   guard installer on the judged PATH with ``ownership=False``.

The transferred file is never started; it is reported as unrecorded with a
``benchflow:`` line (so the verifier's PYTHONPATH carries its guard); the
image's own ``/usr/local/bin/python3`` is guarded as before.

On e4090a15f the separate verifier built no kit without pins, and the
no-kit installer started the transferred file twice as root (the version
check, then the install) and reported it guarded.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow.sandbox import lockdown
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

STARTS = "/var/tmp/conf2b-r3-sv-starts"
DOCKERFILE = (
    "FROM python:3.12-slim\n"
    "RUN mkdir -p /app && python -m venv /app/.venv "
    "&& /app/.venv/bin/pip install --no-cache-dir pytest==8.3.4 "
    "&& cp /app/.venv/bin/pytest /usr/local/bin/pytest\n"
    "WORKDIR /app\n"
)
# The transfer installs workspace files as root with the mode they had.
TRANSFER = (
    f": > {STARTS} && chmod 666 {STARTS} && rm -f /app/.venv/bin/python && "
    f"printf '#!/bin/sh\\necho \"$(id -u) $0\" >> {STARTS}\\nexit 0\\n' "
    "> /app/.venv/bin/python && chmod 755 /app/.venv/bin/python "
    "&& chown root:root /app/.venv/bin/python"
)
TRANSFERRED = "/app/.venv/bin/python"
AGENT_PATHS = ("/app", "/logs/artifacts")


def _write_task(root: Path) -> Path:
    task = root / "conf2b-r3-sv"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
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
        Verify in a sandbox of its own.
        """)
    )
    return task


async def _run(sandbox, command: str):
    done = await sandbox.exec(command, user="root", timeout_sec=300)
    done.stdout = done.stdout or ""
    return done


@pytest.mark.asyncio
@pytest.mark.parametrize("kit", ["kit", "no-kit"])
async def test_a_separate_verifier_never_starts_a_transferred_interpreter(
    tmp_path: Path, kit, caplog
):
    task_path = _write_task(tmp_path)
    task = Task(task_path)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"conf2b-r3-sv-{kit}-{uuid.uuid4().hex[:10]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        # From the fresh image, before any of the agent's bytes arrive.
        if kit == "kit":
            await lockdown.prepare_separate_verifier_runtime(
                sandbox, task, workspace="/app"
            )
        prefixes = lockdown._separate_verifier_blocked_prefixes(AGENT_PATHS)
        await lockdown._prepare_bash_env_inspector(sandbox, blocked=prefixes)
        has_kit = await _run(sandbox, f"[ -f {lockdown.KIT_RECORD} ]")

        moved = await _run(sandbox, TRANSFER)
        assert moved.return_code == 0, moved.stderr
        blocked = await lockdown._path_exclusions_before_freeze(sandbox, prefixes)
        report = await lockdown._judge_verifier_path(
            sandbox, None, "/app", blocked=blocked
        )
        with caplog.at_level(logging.DEBUG, logger="benchflow.sandbox.lockdown"):
            _, _, guarded, unrecorded = await lockdown._install_pytest_plugin_guard(
                sandbox,
                None,
                "/app",
                "",
                verifier_path=report.path,
                blocked=blocked.paths(),
                ownership=False,
            )
        notes = [
            r.getMessage() for r in caplog.records if "benchflow:" in r.getMessage()
        ]
        starts = (await _run(sandbox, f"cat {STARTS}")).stdout.splitlines()

        assert starts == [], starts
        assert TRANSFERRED not in guarded, guarded
        assert TRANSFERRED in unrecorded, unrecorded
        assert any(TRANSFERRED in note for note in notes), notes
        # The line is DEBUG; its reason goes to the verifier error instead.
        assert TRANSFERRED in sandbox._runtime_guard_why
        # The image's own Python keeps its copy.
        assert "/usr/local/bin/python3" in guarded, guarded
        # Built from the fresh image even with no plugin pins.
        assert (has_kit.return_code == 0) is (kit == "kit")
    finally:
        await sandbox.stop(delete=True)

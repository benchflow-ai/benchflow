"""PT-C8-CONF-2b round 2: the pre-agent record is the
only source for where an interpreter looks. After the agent's turn hardening
starts no image interpreter, as any uid; one without a valid record gets the
guard from the verifier's PYTHONPATH and a ``benchflow:`` line, never
silence.

Real Docker sandboxes, python:3.12-slim + pytest. Review A's cases: the
PATH starts with ``/opt/v2/bin`` and ``/opt/v3/bin``, both empty when root's
kit is built, and root adds an interpreter there after it, as an oracle's
install may:

- ``cfg600``: a venv whose ``pyvenv.cfg`` only root reads;
- ``wrapper``: a ``python3`` that starts but exits 1 when not root;
- ``ownlib``: a copy of ``/usr/local`` whose standard library only root
  reads, so another uid sees another prefix than root's run does;
- ``control``: a readable venv.

Review B's cases:

- ``survivor``: ``/usr/local/bin/pytest`` links to a workspace venv's, whose
  ``python`` the agent replaces with a script that leaves a process behind
  to write ``/logs/verifier/reward.txt``;
- ``cut``: a venv on PATH whose ``pyvenv.cfg`` holds a byte that is not
  UTF-8 before the kit, so the record's answer stops right after ``py3 1``;
  hardening fails, as the installer before CONF-2b failed there.

For the others: no image interpreter starts during the install (every
standard library records its starts, and the agent's script its own), no
process but root's is left, the added interpreter is guarded or named by a
``benchflow:`` line, and its pytest, run as root with the ``-p`` flags and
the PYTHONPATH the verifier gets, loads the guard and passes.

On 5d1241139 ``cfg600`` and ``wrapper`` are skipped with no line, ``ownlib``
gets a copy its root run never finds, ``control`` and ``survivor`` start as
uid 65534 (the survivor's process outlives the install) and ``cut`` is
skipped with no error.

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

PATH = "/opt/v2/bin:/opt/v3/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
STARTS = "/var/tmp/conf2b-r2-starts"

_BASE = (
    "FROM python:3.12-slim\n"
    "RUN useradd -m -s /bin/bash agent && mkdir -p /app && chown agent /app "
    "&& pip install --no-cache-dir pytest==8.3.4\n"
)
IMAGE_A = _BASE + f"ENV PATH={PATH}\nWORKDIR /app\n"
IMAGE_SURVIVOR = (
    "FROM python:3.12-slim\n"
    "RUN useradd -m -s /bin/bash agent && mkdir -p /app "
    "&& python -m venv /app/.venv "
    "&& /app/.venv/bin/pip install --no-cache-dir pytest==8.3.4 "
    "&& ln -s /app/.venv/bin/pytest /usr/local/bin/pytest && chown -R agent /app\n"
    "WORKDIR /app\n"
)
IMAGE_CUT = (
    _BASE + "RUN python -m venv --system-site-packages /opt/v2 "
    "&& printf '\\377\\n' >> /opt/v2/pyvenv.cfg\n"
    f"ENV PATH={PATH}\nWORKDIR /app\n"
)

_SCRIPT = "mkdir -p /opt/v3/bin && printf %s {body} > /opt/v3/bin/python3 && chmod 755 /opt/v3/bin/python3"

# Per case: the image, what runs after the kit (and as whom), the
# interpreter test.sh would run.
CASES = {
    "cfg600": (
        IMAGE_A,
        "python3 -m venv --system-site-packages /opt/v2 && chmod 600 /opt/v2/pyvenv.cfg",
        "root",
        "/opt/v2/bin/python",
    ),
    "wrapper": (
        IMAGE_A,
        _SCRIPT.format(
            body=shlex.quote(
                '#!/bin/sh\n[ "$(id -u)" = 0 ] || exit 1\n'
                'exec /usr/local/bin/python3.12 "$@"\n'
            )
        ),
        "root",
        "/opt/v3/bin/python3",
    ),
    "ownlib": (
        IMAGE_A,
        "cp -a /usr/local /opt/v3 && chmod 700 /opt/v3/lib/python3.12 "
        "&& rm -f /opt/v3/bin/python /opt/v3/bin/python3",
        "root",
        "/opt/v3/bin/python3.12",
    ),
    "control": (
        IMAGE_A,
        "python3 -m venv --system-site-packages /opt/v2",
        "root",
        "/opt/v2/bin/python",
    ),
    "survivor": (
        IMAGE_SURVIVOR,
        "rm -f /app/.venv/bin/python && printf %s "
        + shlex.quote(
            "#!/bin/sh\n"
            f'echo "$(id -u) $0" >> {STARTS}\n'
            "( sleep 4; echo 1 > /logs/verifier/reward.txt ) "
            "</dev/null >/dev/null 2>&1 &\n"
            "exit 0\n"
        )
        + " > /app/.venv/bin/python && chmod 755 /app/.venv/bin/python",
        "agent",
        "/app/.venv/bin/python",
    ),
    "cut": (IMAGE_CUT, "true", "root", "/opt/v2/bin/python"),
}

# Appended to each standard library's encodings package, which every start
# imports: "<uid> <executable>" per start.
RECORD_LINE = (
    "\ntry:\n"
    "    import posix as _bf_r, sys as _bf_s\n"
    f"    _bf_f = _bf_r.open({STARTS!r}, _bf_r.O_WRONLY | _bf_r.O_APPEND)\n"
    "    _bf_r.write(_bf_f, ('%d %s\\n' % (_bf_r.getuid(), _bf_s.executable)).encode())\n"
    "    _bf_r.close(_bf_f)\n"
    "except Exception:\n"
    "    pass\n"
)
RECORDER = dedent(f"""\
    set -e
    find / -xdev \\( -path /proc -o -path /opt/benchflow \\) -prune -o \\
        -path '*/encodings/__init__.py' -print > /var/tmp/conf2b-r2-stdlibs
    while IFS= read -r f; do printf %s {shlex.quote(RECORD_LINE)} >> "$f"; done \\
        < /var/tmp/conf2b-r2-stdlibs
    """)

# Every process that is not root's and not a zombie: "<uid> <name>".
_NON_ROOT = (
    "for s in /proc/[0-9]*/status; do "
    'awk \'$1=="Name:"{n=$2} $1=="State:"{z=$2} '
    '$1=="Uid:"{if ($2!=0 && z!="Z") print $2, n}\' "$s" 2>/dev/null; done'
)


def _write_task(root: Path, dockerfile: str) -> Path:
    task = root / "conf2b-r2"
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
@pytest.mark.parametrize("case", sorted(CASES))
async def test_an_unrecorded_interpreter_is_not_started_and_never_silent(
    tmp_path: Path, case, caplog
):
    dockerfile, after_kit, as_user, python = CASES[case]
    task_path = _write_task(tmp_path, dockerfile)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_path),
        task_path,
        f"conf2b-r2-{case}-{uuid.uuid4().hex[:10]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        await prepare_runtime_tree(sandbox)
        made = await _run(sandbox, f": > {STARTS} && chmod 666 {STARTS}")
        assert made.return_code == 0, made.stderr
        added = await _run(sandbox, after_kit, as_user)
        assert added.return_code == 0, added.stderr
        recorder = await _run(sandbox, RECORDER)
        assert recorder.return_code == 0, recorder.stderr
        # As hardening's clear leaves it.
        logs = await _run(
            sandbox, "mkdir -p /logs/verifier && chmod 777 /logs/verifier"
        )
        assert logs.return_code == 0, logs.stderr
        path = (
            await _run(
                sandbox, "tr '\\000' '\\n' < /proc/1/environ | sed -n 's/^PATH=//p'"
            )
        ).stdout.strip()

        if case == "cut":
            # Recorded, but stopped before end: hardening fails, as before.
            with pytest.raises(
                RuntimeError,
                match=f"Cannot install the pytest plugin guard for {python}",
            ):
                await _install_pytest_plugin_guard(
                    sandbox, "agent", "/app", "", verifier_path=path
                )
            return

        with caplog.at_level(logging.DEBUG, logger="benchflow.sandbox.lockdown"):
            result = await _install_pytest_plugin_guard(
                sandbox, "agent", "/app", "", verifier_path=path
            )
        directory, flags, guarded = result[:3]
        unrecorded = result[3] if len(result) > 3 else ()
        notes = [
            r.getMessage() for r in caplog.records if "benchflow:" in r.getMessage()
        ]
        starts = (await _run(sandbox, f"cat {STARTS}")).stdout.splitlines()
        left = (await _run(sandbox, _NON_ROOT)).stdout.strip()

        # Never silence: guarded, or a benchflow: line naming it.
        assert python in guarded or any(python in note for note in notes), (
            guarded,
            notes,
        )
        if case != "survivor":
            # test.sh's pytest on it, as root, with the verifier's PYTHONPATH.
            pythonpath = directory if (not guarded or unrecorded) else ""
            ran = await _run(
                sandbox,
                "cd /tmp && mkdir -p t && printf 'def test_x():\\n    assert True\\n' "
                "> t/test_x.py && env "
                + shlex.quote(f"PYTEST_ADDOPTS={flags}")
                + " "
                + shlex.quote(f"PYTHONPATH={pythonpath}")
                + f" {python} -m pytest -q -p no:cacheprovider t",
            )
            assert ran.return_code == 0, ran.stdout + (ran.stderr or "")
            assert "1 passed" in ran.stdout, ran.stdout
        # No image interpreter started by the install, as any uid, and
        # nothing it could have started is left.
        assert starts == [], starts
        assert left == "", left
        assert unrecorded and python in unrecorded, unrecorded
        if case == "survivor":
            reward = await _run(sandbox, "sleep 5; ls /logs/verifier")
            assert "reward.txt" not in reward.stdout, reward.stdout
    finally:
        await sandbox.stop(delete=True)

"""PT-C8-CONF-2b: after the agent's turn, the pytest
guard installer never starts the task image's Python as root, and puts the
guard where it did before.

A real Docker sandbox per image kind (a system Python, a uv-installed Python
reached only through a venv, a conda base, no Python at all) builds root's
kit, then has the agent's turn. From then on every standard library of the
image records each start of its interpreter (the start's uid) before going
on as before. The guard installer runs on the image's own PATH; it must
start no image Python as root. The installer as it was (the branch it keeps
where no kit exists) then runs with the kit's RECORD set aside: both must
guard the same interpreters, in the same folders. On the system image a
``python3`` added after the kit (as an agent's install adds one) is not in
the record: it is not started by anyone and gets no copy; it is reported as
unrecorded, so the verifier's PYTHONPATH carries its guard (CONF-2b round
2). Fails on d4acaaea2, where the installer starts every interpreter as
root.

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

from benchflow._runtime_tree import KIT_RECORD
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

STARTS = "/var/tmp/conf2b-python-starts"

_AGENT = "RUN useradd -m -s /bin/bash agent && mkdir -p /app && chown agent /app\nWORKDIR /app\n"

IMAGES = {
    "system": (
        "FROM python:3.12-slim\n"
        "RUN pip install --no-cache-dir pytest==8.3.4\n" + _AGENT,
        True,
    ),
    # The only Python is uv's, under root's home, reached through a venv.
    "uv-venv": (
        "FROM debian:bookworm-slim\n"
        "COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv\n"
        "RUN uv python install 3.12 && uv venv /opt/venv --python 3.12 "
        "&& uv pip install --python /opt/venv/bin/python pytest==8.3.4\n"
        "ENV PATH=/opt/venv/bin:$PATH\n" + _AGENT,
        True,
    ),
    "conda": (
        "FROM mambaorg/micromamba:1.5.10-bookworm-slim\n"
        "USER root\n"
        "RUN micromamba install -y -n base -c conda-forge python=3.12 pytest "
        "&& micromamba clean -afy\n"
        "ENV PATH=/opt/conda/bin:$PATH\n"
        "ENTRYPOINT []\n" + _AGENT,
        True,
    ),
    "none": ("FROM debian:bookworm-slim\n" + _AGENT, False),
}

# Appended to each standard library's encodings package, which every start
# imports: "<uid> <executable>" per start, before it goes on as before.
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
    : > {STARTS}; chmod 666 {STARTS}
    find / -xdev \\( -path /proc -o -path /opt/benchflow \\) -prune -o \\
        -path '*/encodings/__init__.py' -print > /var/tmp/conf2b-stdlibs
    while IFS= read -r f; do printf %s {shlex.quote(RECORD_LINE)} >> "$f"; done \\
        < /var/tmp/conf2b-stdlibs
    cat /var/tmp/conf2b-stdlibs
    """)


def _write_task(root: Path, dockerfile: str) -> Path:
    task = root / "conf2b-guard"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(dockerfile)
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


async def _root(sandbox, command: str):
    done = await sandbox.exec(command, user="root", timeout_sec=120)
    done.stdout = done.stdout or ""
    return done


async def _copies(sandbox, directory: str) -> set[str]:
    """The folders holding a copy of the guard whose folder is *directory*."""
    name = directory.rstrip("/").rsplit("/", 1)[-1]
    found = await _root(
        sandbox,
        "find / -xdev \\( -path /proc -o -path " + shlex.quote(directory) + " \\) "
        "-prune -o -name " + shlex.quote(name + ".py") + " -print",
    )
    return {line.rsplit("/", 1)[0] for line in found.stdout.splitlines() if line}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(IMAGES))
async def test_the_guard_installer_starts_no_image_python_as_root(tmp_path: Path, kind):
    dockerfile, has_python = IMAGES[kind]
    task_path = _write_task(tmp_path, dockerfile)
    task = Task(task_path)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"conf2b-{kind}-{uuid.uuid4().hex[:10]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        # Before the agent's turn: root's kit.
        await prepare_runtime_tree(sandbox)

        # The agent's turn.
        turn = await sandbox.exec(
            "echo done > /app/out.txt", user="agent", timeout_sec=30
        )
        assert turn.return_code == 0, turn.stderr
        # After it: an interpreter the record never saw (as an agent's install
        # adds one), and from here on every image Python start is recorded.
        if kind == "system":
            added = await _root(
                sandbox, "ln -s /usr/local/bin/python3.12 /usr/bin/python3"
            )
            assert added.return_code == 0, added.stderr
        recorder = await _root(sandbox, RECORDER)
        assert recorder.return_code == 0, recorder.stderr
        assert bool(recorder.stdout.strip()) is has_python, recorder.stdout
        path = (
            await _root(
                sandbox, "tr '\\000' '\\n' < /proc/1/environ | sed -n 's/^PATH=//p'"
            )
        ).stdout.strip()
        assert path.startswith("/"), path

        directory, _flags, guarded, unrecorded = await _install_pytest_plugin_guard(
            sandbox, "agent", "/app", "", verifier_path=path
        )
        starts = (await _root(sandbox, f"cat {STARTS}")).stdout.splitlines()
        # No image Python started after the agent's turn, as any uid.
        assert starts == [], starts
        # The kit's record, taken before the agent's turn, held the answers.
        record = await _root(sandbox, "cat /opt/benchflow/rt/IMAGE_PYTHON_GUARD")
        assert record.return_code == 0, record.stderr
        assert ("py3 1" in record.stdout) is has_python, record.stdout
        if kind == "system":
            # One start for the three names of one binary, an entry for each.
            entries = record.stdout.split("exe ")[1:]
            names = [entry.split("\n", 1)[0] for entry in entries]
            for python in ("python", "python3", "python3.12"):
                assert f"/usr/local/bin/{python}" in names, names
            answers = {entry.split("\n", 2)[2] for entry in entries}
            assert len(answers) == 1, answers
        assert bool(guarded) is has_python, guarded
        if kind == "system":
            # Not started, no copy: the verifier's PYTHONPATH carries its
            # guard (CONF-2b round 2).
            # (/bin is a link to /usr/bin: the same file by both names.)
            assert "/usr/bin/python3" in unrecorded, unrecorded
            assert set(unrecorded) <= {"/usr/bin/python3", "/bin/python3"}
            assert not set(unrecorded) & set(guarded), guarded
        else:
            assert unrecorded == (), unrecorded
        copies = await _copies(sandbox, directory)

        # The installer as it was, where no kit exists: the same interpreters,
        # the same folders.
        aside = await _root(sandbox, f"mv {KIT_RECORD} {KIT_RECORD}.conf2b")
        assert aside.return_code == 0, aside.stderr
        try:
            before_dir, _, before, _ = await _install_pytest_plugin_guard(
                sandbox, "agent", "/app", "", verifier_path=path
            )
        finally:
            back = await _root(sandbox, f"mv {KIT_RECORD}.conf2b {KIT_RECORD}")
            assert back.return_code == 0, back.stderr
        # The same, but for the interpreter the record does not vouch for.
        assert sorted(guarded) == sorted(p for p in before if p not in unrecorded)
        assert set(unrecorded) <= set(before)
        assert copies == await _copies(sandbox, before_dir)

        # Each guarded interpreter finds the copy by its own finder.
        name = directory.rstrip("/").rsplit("/", 1)[-1]
        for python in guarded:
            found = await _root(
                sandbox,
                "cd / && "
                + shlex.quote(python)
                + " -I -c 'import importlib.util, sys; "
                "print(importlib.util.find_spec(sys.argv[1]).origin)' "
                + shlex.quote(name),
            )
            assert found.return_code == 0, found.stderr
            assert found.stdout.strip().rsplit("/", 1)[0] in copies, (
                python,
                found.stdout,
            )
    finally:
        await sandbox.stop(delete=True)

"""Every read and call a held start makes ends, and so does the start.

- A ``docker cp`` read cut off after its first member returns even when its
  pipe holds more than the reader's buffer (its transport paused): the rest
  is read and dropped before the exit is awaited. Some Python builds (3.12.3)
  report a process's exit only once its pipes have closed, so a bare wait
  there never returns.
- A read that never answers ends at its bound, as BenchFlow's own failure.
- bfx's digest read that stalls past its timeout ends too.
- A start with no deadline of its own ends at the default bound.
- A healthcheck budget of 0 still ends the wait.
- Stopping a process that has already exited reports its own status: it
  is not signalled (the signal's poll would reap it first, and asyncio would
  report 255 and warn).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._bfx import delivery
from benchflow.sandbox import _held_mounts, _inert_bootstrap
from benchflow.sandbox.docker import DockerSandbox

#: A ``docker`` that writes an archive whose first member is the folder
#: /.bfx, then 2 MiB more, then never exits; or, with ``SILENT``, writes
#: nothing and never exits.
FAKE_DOCKER = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import io, os, sys, tarfile, time
    if os.environ.get("SILENT"):
        time.sleep(600)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        folder = tarfile.TarInfo(".bfx")
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        data = os.urandom(2 << 20)
        member = tarfile.TarInfo(".bfx/bfx")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    sys.stdout.buffer.write(buffer.getvalue())
    sys.stdout.flush()
    time.sleep(600)
    """
)


@pytest.fixture
def fake_docker(tmp_path):
    folder = tmp_path / "bin"
    folder.mkdir()
    docker = folder / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    return folder


def _paused(monkeypatch, *, first_read: bool) -> None:
    """Every process the code under test starts gets its stdout transport
    paused, as a busy loop leaves one whose reader fell behind: after its
    first read returns (*first_read*), or 0.3 s after it starts."""
    spawn = asyncio.create_subprocess_exec

    async def paused(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        transport = process._transport.get_pipe_transport(1)
        if not first_read:
            asyncio.get_running_loop().call_later(0.3, transport.pause_reading)
            return process
        read = process.stdout.read
        calls = 0

        async def read_then_pause(n=-1):
            nonlocal calls
            calls += 1
            data = await read(n)
            if calls == 1:
                await asyncio.sleep(0.5)
                transport.pause_reading()
            return data

        process.stdout.read = read_then_pause
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", paused)


def _sandbox(folder: Path, **env: str) -> SimpleNamespace:
    environ = {**os.environ, "PATH": f"{folder}:{os.environ['PATH']}", **env}
    return SimpleNamespace(_docker_args=lambda: [], _docker_process_env=lambda: environ)


async def test_a_read_cut_off_with_its_pipe_full_still_returns(
    fake_docker, monkeypatch
):
    _paused(monkeypatch, first_read=True)
    entry = await asyncio.wait_for(
        DockerSandbox._path_entry(_sandbox(fake_docker), "c" * 64, "/.bfx"), 30
    )
    assert entry == ("dir", "")


async def test_a_read_that_never_answers_ends_at_its_bound(fake_docker, monkeypatch):
    monkeypatch.setattr(_held_mounts, "READ_SEC", 1.0)
    with pytest.raises(_held_mounts.HeldStartUnread, match="took over 1s"):
        await asyncio.wait_for(
            DockerSandbox._path_entry(
                _sandbox(fake_docker, SILENT="1"), "c" * 64, "/.bfx"
            ),
            30,
        )


async def test_bfx_s_digest_read_that_stalls_ends_at_its_timeout(
    fake_docker, monkeypatch
):
    _paused(monkeypatch, first_read=False)
    environ = _sandbox(fake_docker)._docker_process_env()
    began = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            delivery._docker(
                ["docker"], ["cp", "c:/.bfx/bfx", "-"], environ, timeout=1.0
            ),
            30,
        )
    # Its own timeout and the stop's bound, not the outer 30 s.
    assert time.monotonic() - began < 15


class _WedgedStart:
    """A sandbox whose start never returns."""

    def __init__(self, config) -> None:
        self.task_env_config = config

    async def start(self, force_build: bool) -> None:
        await asyncio.Event().wait()


async def test_a_start_with_no_deadline_of_its_own_ends_at_the_default(
    tmp_path, monkeypatch
):
    from benchflow.rollout import _setup
    from benchflow.sandbox.protocol import EnvironmentStartTimeoutError
    from benchflow.task.config import SandboxConfig

    monkeypatch.setattr(_setup, "START_ALLOWANCE_SEC", 0.5)
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("hi\n")
    config = SandboxConfig(build_timeout_sec=0.5)
    assert config.start_deadline_sec is None
    with pytest.raises(EnvironmentStartTimeoutError, match=r"within 1s"):
        await asyncio.wait_for(
            _setup._start_env_and_upload(
                _WedgedStart(config), task, {}, task=None, agent=None
            ),
            30,
        )


def test_the_default_start_bound_follows_the_build_budget():
    from benchflow.rollout import _setup
    from benchflow.task.config import SandboxConfig

    allowance = _setup.START_ALLOWANCE_SEC
    assert _setup.default_start_deadline(SandboxConfig(build_timeout_sec=900)) == (
        900 + allowance
    )
    assert _setup.default_start_deadline(SandboxConfig(build_timeout_sec=0)) == (
        600 + allowance
    )


def test_a_healthcheck_budget_of_0_still_ends_the_wait():
    from benchflow.task.config import SandboxConfig

    def env(config):
        return SimpleNamespace(task_env_config=config)

    assert _inert_bootstrap._health_cap(env(SandboxConfig(build_timeout_sec=30))) == 30
    assert _inert_bootstrap._health_cap(env(SandboxConfig(build_timeout_sec=0))) == (
        _inert_bootstrap.DEFAULT_HEALTH_CAP_SEC
    )
    assert _inert_bootstrap._health_cap(SimpleNamespace()) is None


def _exited(pid: int) -> bool:
    """Whether the child has exited: its status still there (``WNOWAIT``),
    or already read by asyncio's watcher thread (Python 3.12.3's)."""
    try:
        found = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return found is not None


async def test_stopping_a_process_that_has_exited_reports_its_own_status(caplog):
    """Where asyncio's child watcher reads a status only when the loop runs
    (the pidfd watcher: a Python with ``os.pidfd_open``, 3.12.15 here), a
    signal sent to an exited child would reap it in the signal's own poll,
    and asyncio would then warn and report 255. A build without
    ``os.pidfd_open`` (uv's 3.12.3) uses a watcher thread that reaps at once:
    the signal's poll then finds nothing to reap, and this row passes there
    with or without the check."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('x' * 100)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Exited, and not yet reported by asyncio: no await between here and the
    # stop.
    end = time.monotonic() + 10
    while not _exited(process.pid) and time.monotonic() < end:
        time.sleep(0.01)
    assert process.returncode is None
    with caplog.at_level(logging.WARNING, logger="asyncio"):
        await asyncio.wait_for(delivery.stop_process(process), 10)
    assert process.returncode == 0
    assert not [r for r in caplog.records if "already read" in r.getMessage()]


async def test_stopping_a_running_process_kills_and_reaps_it():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(600)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    began = time.monotonic()
    await delivery.stop_process(process)
    assert process.returncode == -9
    assert time.monotonic() - began < delivery.STOP_SEC + 1

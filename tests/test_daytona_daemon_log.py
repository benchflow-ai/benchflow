"""Daytona's daemon log is made root-only before the agent starts (K1, Rust
57213637). Daytona's toolbox daemon writes ``/tmp/daytona-daemon.log`` readable
by every user, with its routes and the client's address in it; an agent read it
while looking for a way to root. A single-container sandbox's ``start`` makes
the log root's with mode 0600 as root, checks the result, and refuses a log it
cannot protect (a link, a non-file, a failed change), and so does a
restore from a snapshot (K1-R1-B-1), so the agent,
which starts later, cannot read it.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect

# A checkpoint carries the toolbox startup receipt a restore checks first.
_STARTUP_META = {
    "benchflow_toolbox_startup_v1": json.dumps(
        dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    )
}


def _sandbox() -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox.environment_name = "hello-world-task"
    sandbox.task_env_config = SimpleNamespace(
        cpus=1, memory_mb=2048, storage_mb=10240, docker_image="img:latest"
    )
    sandbox._snapshot_template_name = None
    sandbox._start_snapshot_ref = None
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._strategy = _DaytonaDirect(sandbox)
    return sandbox


@pytest.mark.asyncio
async def test_start_makes_the_daemon_log_root_only(monkeypatch):
    """The single-container start runs the daemon-log chmod after the dir setup."""
    manager = SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    # The Daytona SDK handles are loaded lazily and may be None here.
    monkeypatch.setattr(daytona_mod, "Resources", lambda **kw: kw)
    monkeypatch.setattr(daytona_mod, "Image", SimpleNamespace(base=lambda img: img))
    monkeypatch.setattr(
        daytona_mod, "CreateSandboxFromImageParams", lambda **kw: SimpleNamespace(**kw)
    )
    sandbox = _sandbox()
    sandbox._client_manager = None
    sandbox._create_sandbox = AsyncMock()
    # This fixture records startup chmods; it has no toolbox transport.
    sandbox._toolbox_startup_env = AsyncMock(
        return_value=dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    )
    sandbox._create_attempts = 0
    sandbox._sandbox_exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0, stdout="BF daemon-log root-only\n", stderr=""
        )
    )

    await sandbox._strategy.start(force_build=False)

    calls = sandbox._sandbox_exec.await_args_list
    log_calls = [c for c in calls if "/tmp/daytona-daemon.log" in c.args[0]]
    # The start makes the daemon log root's and readable by root only (fails
    # on the base, whose start only makes the log dirs).
    assert len(log_calls) == 1
    assert "chmod 600" in log_calls[0].args[0]
    assert "chown 0:0" in log_calls[0].args[0]
    # As root, bounded: the image's default user may not be root.
    assert log_calls[0].kwargs.get("user") == "root"
    assert log_calls[0].kwargs.get("timeout_sec")
    # The log dirs are made first, in one exec with the default user's pwd
    # and id -u (PY8 U12). This mock's answer has none of that exec's
    # markers, so the start command then runs again alone, unchanged.
    probe, alone = (c.args[0] for c in calls[:2])
    assert "@@benchflow-merged-" in probe
    assert alone in probe
    assert "mkdir" in alone
    assert calls.index(log_calls[0]) == 2


@pytest.mark.parametrize(
    "stdout, code",
    [
        ("BF daemon-log link\n", 1),
        ("BF daemon-log unprotected\n", 1),
        ("BF daemon-log not-root\n", 1),
        ("", 0),
        ("BF daemon-log root-only\n", None),
        ("BF daemon-log root-only\nBF daemon-log absent\n", 0),
    ],
    ids=["link", "unprotected", "not-root", "no-line", "no-status", "two-lines"],
)
@pytest.mark.asyncio
async def test_start_refuses_a_log_it_cannot_protect(monkeypatch, stdout, code):
    """A refused, unchecked or unclear daemon-log step stops the start before
    the agent runs; the base ignored the step's result."""
    manager = SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    monkeypatch.setattr(daytona_mod, "Resources", lambda **kw: kw)
    monkeypatch.setattr(daytona_mod, "Image", SimpleNamespace(base=lambda img: img))
    monkeypatch.setattr(
        daytona_mod, "CreateSandboxFromImageParams", lambda **kw: SimpleNamespace(**kw)
    )
    sandbox = _sandbox()
    sandbox._client_manager = None
    sandbox._create_sandbox = AsyncMock()
    sandbox._toolbox_startup_env = AsyncMock(
        return_value=dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    )
    sandbox._create_attempts = 0

    async def fake_exec(command, **kwargs):
        if "/tmp/daytona-daemon.log" in command:
            return SimpleNamespace(return_code=code, stdout=stdout, stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox._sandbox_exec = fake_exec

    with pytest.raises(RuntimeError, match="could not be made root-only"):
        await sandbox._strategy.start(force_build=False)


# K1-R1-B-1: a branch child's sandbox is built from a snapshot by restore(),
# which may be a kept checkpoint from before the step, and its daemon starts
# again; restore must protect the log too.


def _restoring_sandbox(monkeypatch, order, stdout, code):
    from benchflow.sandbox import daytona_strategies

    monkeypatch.setattr(
        daytona_mod,
        "CreateSandboxFromSnapshotParams",
        lambda **kw: SimpleNamespace(**kw),
    )

    async def put_back(ops, stash):
        order.append("credentials")

    monkeypatch.setattr(daytona_strategies, "put_back_credentials", put_back)
    sandbox = _sandbox()
    sandbox._sandbox = None
    # The creation seam: these tests check what runs after it, and in what
    # order (the toolbox startup checks are tests/test_daytona_toolbox_startup.py).
    sandbox._create_sandbox_without_toolbox_bash_env = AsyncMock(
        side_effect=lambda params, startup_env=None: order.append(
            ("create", params.snapshot)
        )
    )

    async def fake_exec(command, **kwargs):
        order.append(("exec", command, kwargs.get("user")))
        return SimpleNamespace(return_code=code, stdout=stdout, stderr="")

    sandbox._sandbox_exec = fake_exec
    strategy = sandbox._strategy
    strategy._credential_ops = lambda: None
    strategy._snapshot_credentials["bf-snap-kept"] = {"stash": "kept"}
    return sandbox


@pytest.mark.asyncio
async def test_restore_makes_the_daemon_log_root_only(monkeypatch):
    """A sandbox restored from a snapshot gets the step as root, after it is
    created and before the credentials go back (fails on 602fd52e, whose
    restore never ran it)."""
    from benchflow.sandbox.protocol import SandboxImage

    order: list = []
    sandbox = _restoring_sandbox(monkeypatch, order, "BF daemon-log root-only\n", 0)

    await sandbox.restore(SandboxImage("daytona", "bf-snap-kept", meta=_STARTUP_META))

    assert order[0] == ("create", "bf-snap-kept")
    log_calls = [e for e in order if e[0] == "exec" and DAEMON_LOG in e[1]]
    assert len(log_calls) == 1
    assert "chmod 600" in log_calls[0][1] and "chown 0:0" in log_calls[0][1]
    assert log_calls[0][2] == "root"
    assert order.index(log_calls[0]) < order.index("credentials")


@pytest.mark.asyncio
async def test_restore_refuses_a_log_it_cannot_protect(monkeypatch):
    """A restored sandbox whose log cannot be protected is refused before its
    credentials go back and before any agent runs in it."""
    from benchflow.sandbox.protocol import SandboxImage

    order: list = []
    sandbox = _restoring_sandbox(monkeypatch, order, "BF daemon-log link\n", 1)

    with pytest.raises(RuntimeError, match="could not be made root-only"):
        await sandbox.restore(
            SandboxImage("daytona", "bf-snap-kept", meta=_STARTUP_META)
        )
    assert "credentials" not in order


DAEMON_LOG = "/tmp/daytona-daemon.log"


def _run(path) -> subprocess.CompletedProcess:
    from benchflow.sandbox.daytona_strategies import daemon_log_root_only_cmd

    return subprocess.run(
        ["/bin/sh", "-c", daemon_log_root_only_cmd(str(path))],
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.skipif(os.geteuid() != 0, reason="makes a file root's: needs root")
def test_a_world_readable_log_becomes_root_only(tmp_path):
    log = tmp_path / "daytona-daemon.log"
    log.write_text("GET /process/session 200\n")
    os.chown(log, 65534, 65534)
    log.chmod(0o644)
    out = _run(log)
    st = log.stat()
    assert (stat.S_IMODE(st.st_mode), st.st_uid) == (0o600, 0), out
    assert (out.returncode, out.stdout.strip()) == (0, "BF daemon-log root-only")


@pytest.mark.skipif(os.geteuid() == 0, reason="checks the step refuses non-root")
def test_the_step_refuses_to_run_as_a_user_other_than_root(tmp_path):
    """Not root, the step can make the log readable only by its current
    owner, who may be the agent: it refuses instead."""
    log = tmp_path / "daytona-daemon.log"
    log.write_text("x")
    log.chmod(0o644)
    out = _run(log)
    assert (out.returncode, out.stdout.strip()) == (1, "BF daemon-log not-root")
    assert stat.S_IMODE(log.stat().st_mode) == 0o644


def _run_as_root(path) -> subprocess.CompletedProcess:
    """The step with ``id -u`` answering 0, ``chown`` and ``chmod`` doing
    nothing: what it does with what ``ls`` reports after its changes."""
    from benchflow.sandbox.daytona_strategies import daemon_log_root_only_cmd

    stubs = "id() { echo 0; }; chown() { :; }; chmod() { :; }; "
    return subprocess.run(
        ["/bin/sh", "-c", stubs + daemon_log_root_only_cmd(str(path))],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_a_log_still_readable_after_the_changes_is_refused(tmp_path):
    """The step checks the result: a log that is still not root's alone (here
    chown and chmod did nothing) is refused, not reported done."""
    log = tmp_path / "daytona-daemon.log"
    log.write_text("x")
    log.chmod(0o644)
    out = _run_as_root(log)
    assert (out.returncode, out.stdout.strip()) == (1, "BF daemon-log unprotected")


def test_a_symlink_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("x")
    target.chmod(0o644)
    link = tmp_path / "link.log"
    link.symlink_to(target)
    out = _run_as_root(link)
    assert (out.returncode, out.stdout.strip()) == (1, "BF daemon-log link")
    # Nothing followed the link: the target keeps its mode.
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_a_dangling_symlink_is_refused(tmp_path):
    link = tmp_path / "link.log"
    link.symlink_to(tmp_path / "missing")
    out = _run_as_root(link)
    assert (out.returncode, out.stdout.strip()) == (1, "BF daemon-log link")


def test_a_folder_in_its_place_is_refused(tmp_path):
    (tmp_path / "d.log").mkdir()
    out = _run_as_root(tmp_path / "d.log")
    assert (out.returncode, out.stdout.strip()) == (1, "BF daemon-log not-a-file")


def test_no_log_is_reported_absent(tmp_path):
    out = _run_as_root(tmp_path / "none.log")
    assert (out.returncode, out.stdout.strip()) == (0, "BF daemon-log absent")

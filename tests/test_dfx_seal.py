"""DFX-SEAL (PY8 batch 49): the daemon guard says where it was, and the
loader proof judges what an image really lets another user change.

- DFR-p3-4: three Daytona trials lost the guard to its 180 s timeout with
  "the guard did not report (no report)": the receipt holds every byte the
  guard prints, so a stopped guard left nothing. The guard now logs each
  step and the package manager's output to a root-only file as it goes; a
  failed guard's log reaches the trial folder (``daemon-guard.log``), its
  message names the last step, and its seconds are ``daemon_guard``.
- DFR-p3-5: ``sam-cell-seg``'s ``libopenjpip.so`` is a root-owned link to a
  file the image does not have; the proof judged the link's own 0777 mode
  bits and refused. A link to a missing file is now judged as a missing file
  is. ``make-mips-interpreter`` really ships loader paths uid 2952 owns: the
  refusal stays, now before the agent's install, naming owner and mode, with
  a ``next:`` line.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import timing_spans
from tests import test_sdk_lockdown as sdk_lockdown
from tests._receipt_fakes import inner_command, receipt_answer
from tests.test_timing_spans import FakeClock, _install_trial, _quiet_planes


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(timing_spans, "_clock", fake)
    return fake


# --- DFR-p3-4: the guard's log, timeout message and span -----------------------


def test_the_guard_logs_each_step_and_the_install_output(tmp_path):
    """An image without iptables: the log holds the steps in order, with
    their seconds, and apt-get's own lines between them."""
    from benchflow.sandbox.lockdown import _daemon_guard_cmd

    t = sdk_lockdown.TestSandboxDaemonGuard()
    apt = (
        'if [ "$1" = update ]; then echo "Get:1 http://archive.ubuntu.com/ubuntu '
        'noble InRelease"; exit 0; fi\n'
        '[ -f "$(dirname "$0")/.updated" ] || { touch "$(dirname "$0")/.updated"; '
        "echo 'E: Unable to locate package iptables' >&2; exit 100; }\n"
        'for n in iptables ip6tables; do cp "$(dirname "$0")/fw-template" '
        '"$(dirname "$0")/$n"; chmod 755 "$(dirname "$0")/$n"; done\n'
        "echo 'Setting up iptables (1.8.10-3ubuntu2) ...'\n"
    )
    bin_dir = t._bin(tmp_path, tools=(), extra={"apt-get": apt})
    (bin_dir / "fw-template").write_text(
        sdk_lockdown.FAKE_IPTABLES.format(python=sys.executable)
    )
    proc = t._proc(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    log = run / "daemon-guard.log"
    out = t._sh(
        tmp_path, _daemon_guard_cmd("agent", proc=str(proc), log=str(log)), bin_dir
    )
    assert "status=0" in out
    text = log.read_text()
    steps = [ln.split(" ", 1)[1] for ln in text.splitlines() if ln.startswith("+")]
    assert steps == [
        "start",
        "ports 2280 22220 22222 33333 ",
        "install iptables: apt-get install",
        "install iptables: apt-get update",
        "install iptables: apt-get install",
        "rules: iptables",
        "rules: ip6tables",
        "done status=0",
    ]
    assert "E: Unable to locate package iptables" in text
    assert "Get:1 http://archive.ubuntu.com/ubuntu noble InRelease" in text
    assert "Setting up iptables" in text


def test_a_guard_without_its_log_folder_still_guards(tmp_path):
    """No root-only folder (no protected runtime): no log, same guard."""
    from benchflow.sandbox.lockdown import _daemon_guard_cmd

    t = sdk_lockdown.TestSandboxDaemonGuard()
    bin_dir = t._bin(tmp_path)
    proc = t._proc(tmp_path)
    log = tmp_path / "absent" / "daemon-guard.log"
    out = t._sh(
        tmp_path, _daemon_guard_cmd("agent", proc=str(proc), log=str(log)), bin_dir
    )
    assert "status=0" in out
    assert not log.parent.exists()


def _timed_out_env(progress: str):
    """A Daytona-like env whose guard exec the sandbox's timeout stopped
    (status 124, no receipt), and whose log read answers *progress*."""
    calls: list[str] = []

    async def execute(command, **kwargs):
        inner = inner_command(command)
        calls.append(inner)
        if "BF daemon-guard" in inner:
            return SimpleNamespace(stdout="", stderr="", return_code=124)
        return receipt_answer(
            command, SimpleNamespace(stdout=progress, stderr="", return_code=0)
        )

    env = SimpleNamespace(
        supports_system_shell=True,
        sandbox_daemon_guard="container",
        exec=AsyncMock(side_effect=execute),
    )
    return env, calls


@pytest.mark.asyncio
async def test_a_guard_its_timeout_stopped_names_its_last_step():
    from benchflow.sandbox.lockdown import (
        DAEMON_GUARD_LOG,
        SandboxDaemonGuardError,
        guard_sandbox_daemon,
    )

    progress = (
        "+0s start\n+0s ports 2280 22220 22222 33333 \n"
        "+0s install iptables: apt-get install\n"
        "E: Unable to locate package iptables\n"
        "+1s install iptables: apt-get update\n"
        "Get:1 http://archive.ubuntu.com/ubuntu noble InRelease\n"
    )
    env, calls = _timed_out_env(progress)
    with pytest.raises(SandboxDaemonGuardError) as raised:
        await guard_sandbox_daemon(env, "agent")
    message = str(raised.value)
    assert message.startswith("sandbox startup refused: ")
    assert (
        "the guard did not finish in 180 s (its last step: +1s install iptables: "
        "apt-get update)" in message
    )
    assert "may pass on a retry" in message
    assert raised.value.progress == progress
    assert calls[-1] == f"cat {DAEMON_GUARD_LOG}"


def test_a_failed_guards_log_is_kept_in_the_trial_folder(tmp_path):
    from benchflow.rollout import _write_guard_progress
    from benchflow.sandbox.lockdown import SandboxDaemonGuardError

    _write_guard_progress(
        tmp_path, SandboxDaemonGuardError("x", progress="+0s start\n")
    )
    assert (tmp_path / "daemon-guard.log").read_text() == "+0s start\n"
    _write_guard_progress(tmp_path / "none", RuntimeError("no progress"))
    assert not (tmp_path / "none").exists()


@pytest.mark.asyncio
async def test_the_guard_records_its_span(clock):
    from benchflow.sandbox.lockdown import guard_sandbox_daemon

    async def execute(command, **kwargs):
        clock.advance(4.0)
        return receipt_answer(
            command,
            SimpleNamespace(
                stdout="BF daemon-guard ports=2280 status=0", stderr="", return_code=0
            ),
        )

    env = SimpleNamespace(
        supports_system_shell=True,
        sandbox_daemon_guard="container",
        exec=AsyncMock(side_effect=execute),
    )
    timing: dict = {}
    with timing_spans.recording_into(timing):
        await guard_sandbox_daemon(env, "agent")
    assert timing == {"daemon_guard": 4.0}


@pytest.mark.asyncio
async def test_the_sandbox_user_setup_records_into_the_trials_timing(
    tmp_path, monkeypatch, clock
):
    """The trial's timing is the recorder while the sandbox user is set up,
    so the guard's span (inside it) reaches timing.json."""
    trial = _install_trial(tmp_path, "claude-agent-acp")
    _quiet_planes(monkeypatch, trial, clock)

    async def setup(*args, **kwargs):
        with timing_spans.span(timing_spans.DAEMON_GUARD):
            clock.advance(3.0)
        return "/home/agent"

    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup)
    await trial.install_agent()
    assert trial.timing == {"install_agent": 7.0, "daemon_guard": 3.0}


# --- DFR-p3-5: the early loader check -------------------------------------------


@pytest.mark.asyncio
async def test_the_loader_check_runs_before_the_agent_install(
    tmp_path, monkeypatch, clock
):
    from benchflow.sandbox import lockdown

    trial = _install_trial(tmp_path, "claude-agent-acp")
    install = _quiet_planes(monkeypatch, trial, clock)
    checked = []

    async def refuse(env, workspace, **kwargs):
        checked.append(env)
        raise lockdown.LoaderCustodyError(
            "sandbox startup refused: in the task's image"
        )

    monkeypatch.setattr(trial._planes, "check_loader_custody", refuse)
    with pytest.raises(lockdown.LoaderCustodyError):
        await trial.install_agent()
    assert checked == [trial._env]
    install.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_run_as_root_gets_no_loader_check(tmp_path, monkeypatch, clock):
    trial = _install_trial(tmp_path, "claude-agent-acp")
    trial._config.sandbox_user = None
    install = _quiet_planes(monkeypatch, trial, clock)
    refuse = AsyncMock(side_effect=AssertionError("checked"))
    monkeypatch.setattr(trial._planes, "check_loader_custody", refuse)
    await trial.install_agent()
    refuse.assert_not_awaited()
    install.assert_awaited_once()


def test_a_loader_refusal_is_not_retried():
    from benchflow._utils.scoring import classify_error, is_unrecoverable_startup_error

    early = (
        "sandbox startup refused: in the task's image, the system loader searches "
        "/lib/mips-linux-gnu (owner 2952:1009, mode drwxr-xr-x), which is not root's "
        "alone, so BenchFlow will not run root steps"
    )
    seal = (
        "sandbox startup refused: BenchFlow could not seal its protected runtime: "
        "the system loader loads /lib/x/liba.so (owner 0:0, mode -rw-rw-rw-), which "
        "is not root's alone"
    )
    for error in (early, seal):
        assert classify_error(error) == "sandbox_setup"
        assert is_unrecoverable_startup_error(error)


# --- DFR-p3-5: the loader proof in a real container ------------------------------

IMAGE = "python:3.12-slim"
LIBDIR = "/usr/lib/x86_64-linux-gnu"


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture
def box():
    """A fresh container of IMAGE (no network); yields a function that runs
    a root shell command in it."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    import os

    label = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
    name = "dfxseal-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", label, "--network",
         "none", IMAGE, "sleep", "900"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")

    def sh(command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", command],
            capture_output=True,
            text=True,
        )

    sh.name = name  # type: ignore[attr-defined]
    try:
        yield sh
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _seal(box, walk: bool) -> subprocess.CompletedProcess:
    from benchflow._runtime_tree import kit_prepare_command

    done = box(kit_prepare_command("all", walk=walk))
    # Each proof says which ran: the walk on the kit's Python, or the sh.
    if done.returncode == 0:
        assert ("BF kit-seal walk" if walk else "BF kit-seal sh") in done.stdout
    return done


@pytest.mark.parametrize("walk", [True, False])
def test_a_root_link_to_a_missing_library_seals(box, walk):
    """sam-cell-seg: a -dev package's link to a library the image lacks, in
    a root-only loader folder: only root could create the file it names."""
    assert box(f"ln -s libbfmissing.so.7 {LIBDIR}/libbfmissing.so").returncode == 0
    done = _seal(box, walk)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("walk", [True, False])
def test_a_link_to_a_missing_file_another_user_could_create_refuses(box, walk):
    prep = f"mkdir -p /srv/open && chmod 777 /srv/open && ln -s /srv/open/libx.so.1 {LIBDIR}/libx.so"
    assert box(prep).returncode == 0
    done = _seal(box, walk)
    assert done.returncode != 0
    assert (
        "the system loader loads /lib/x86_64-linux-gnu/libx.so (owner 0:0, mode lrwxrwxrwx)"
        in done.stderr
    )


@pytest.mark.parametrize("walk", [True, False])
def test_a_library_another_user_owns_is_named_with_its_owner(box, walk):
    assert (
        box(
            f"echo x > {LIBDIR}/libbfmips.so && chown 2952:1009 {LIBDIR}/libbfmips.so"
        ).returncode
        == 0
    )
    done = _seal(box, walk)
    assert done.returncode != 0
    assert (
        "the system loader loads /lib/x86_64-linux-gnu/libbfmips.so "
        "(owner 2952:1009, mode -rw-r--r--), which is not root's alone"
    ) in done.stderr
    assert "/./" not in done.stderr


@pytest.mark.parametrize("walk", [True, False])
def test_a_loader_folder_another_user_owns_is_named_with_its_owner(box, walk):
    """make-mips-interpreter: /usr/lib/mips-linux-gnu belongs to uid 2952."""
    assert (
        box(
            "mkdir -p /usr/lib/mips-linux-gnu && chown 2952:1009 /usr/lib/mips-linux-gnu"
        ).returncode
        == 0
    )
    done = _seal(box, walk)
    assert done.returncode != 0
    assert (
        "the system loader searches /lib/mips-linux-gnu "
        "(owner 2952:1009, mode drwxr-xr-x), which is not root's alone"
    ) in done.stderr


def _docker_sandbox(name: str):
    from tests.test_c8_protected_runtime_container import _sandbox

    return _sandbox(name)


@pytest.mark.asyncio
async def test_the_early_check_refuses_with_the_owner_and_a_next_line(box):
    from benchflow.sandbox.lockdown import LoaderCustodyError, check_loader_custody

    sandbox = _docker_sandbox(box.name)
    assert box(f"ln -s libbfmissing.so.7 {LIBDIR}/libbfmissing.so").returncode == 0
    await check_loader_custody(sandbox, "/app")
    assert (
        box(
            "mkdir -p /usr/lib/mips-linux-gnu && chown 2952:1009 /usr/lib/mips-linux-gnu"
        ).returncode
        == 0
    )
    with pytest.raises(LoaderCustodyError) as raised:
        await check_loader_custody(sandbox, "/app")
    message = str(raised.value)
    assert message.startswith(
        "sandbox startup refused: in the task's image, the system loader searches "
        "/lib/mips-linux-gnu (owner 2952:1009, mode drwxr-xr-x), which is not "
        "root's alone"
    )
    assert "did not install the agent" in message
    assert "\nnext: make that path root's in the task's image" in message
    assert "--sandbox-user none" in message
    # Read only: the image is as it was.
    assert (
        box("ls -ldn /usr/lib/mips-linux-gnu | awk '{print $3}'").stdout.strip()
        == "2952"
    )
    assert box("[ -e /opt/benchflow ] || echo none").stdout.strip() == "none"


@pytest.mark.asyncio
async def test_the_seal_refuses_with_the_owner_and_a_next_line(box):
    from benchflow.sandbox.lockdown import LoaderCustodyError, setup_sandbox_user

    sandbox = _docker_sandbox(box.name)
    assert (
        box(
            "mkdir -p /srv/ws /usr/lib/mips-linux-gnu && chown 2952:1009 /usr/lib/mips-linux-gnu"
        ).returncode
        == 0
    )
    with pytest.raises(LoaderCustodyError) as raised:
        await setup_sandbox_user(sandbox, "agent", "/srv/ws", guard_daemon=False)
    message = str(raised.value)
    assert message.startswith(
        "sandbox startup refused: BenchFlow could not seal its protected runtime: "
        "the system loader searches /lib/mips-linux-gnu (owner 2952:1009, mode "
        "drwxr-xr-x), which is not root's alone"
    )
    assert "\nnext: " in message and "--sandbox-user none" in message


@pytest.mark.asyncio
async def test_the_early_check_leaves_what_the_path_lock_repairs_to_the_seal(box):
    """A loader folder the PATH lock repairs before the seal (it lies above
    the folder a PATH program's link points into) is no early refusal: the
    early check refuses only what the seal is sure to refuse."""
    from benchflow.sandbox.lockdown import check_loader_custody, setup_sandbox_user

    prep = (
        f"mkdir -p {LIBDIR}/bfjdk/bin /srv/ws && printf '#!/bin/sh\\n' > {LIBDIR}/bfjdk/bin/bfjava && "
        f"chmod 755 {LIBDIR}/bfjdk/bin/bfjava && ln -s {LIBDIR}/bfjdk/bin/bfjava /usr/local/bin/bfjava && "
        f"chmod g+w {LIBDIR}"
    )
    assert box(prep).returncode == 0
    sandbox = _docker_sandbox(box.name)
    await check_loader_custody(sandbox, "/srv/ws")
    await setup_sandbox_user(sandbox, "agent", "/srv/ws", guard_daemon=False)
    assert sandbox.runtime_kit is not None
    assert box(f"stat -c %a {LIBDIR}").stdout.strip() == "755"


def test_the_dry_path_lock_reports_and_changes_nothing(box):
    from benchflow.sandbox.lockdown import _lock_system_path_cmd, _path_lock_report

    assert box("mkdir -p /srv/ws && chmod 777 /usr/local/bin").returncode == 0
    done = box(_lock_system_path_cmd("/srv/ws", dry=True))
    assert "/usr/local/bin" in _path_lock_report(done.stdout)[0]
    assert box("stat -c %a /usr/local/bin").stdout.strip() == "777"


@pytest.mark.asyncio
async def test_planes_without_the_early_check_keep_the_seal(
    tmp_path, monkeypatch, clock
):
    """A plane whose class has no check_loader_custody (a custom plane, a
    test's MagicMock) installs as before; its seal still decides."""
    from unittest.mock import MagicMock

    trial = _install_trial(tmp_path, "claude-agent-acp")
    install = _quiet_planes(monkeypatch, trial, clock)
    planes = MagicMock(wraps=trial._planes)
    planes.install_agent = install
    planes.setup_sandbox_user = AsyncMock(return_value="/home/agent")
    for name in (
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_verifier_runtime",
        "deploy_skills",
        "lockdown_paths",
        "write_credential_files",
        "upload_subscription_auth",
        "apply_web_tool_policy",
    ):
        setattr(planes, name, AsyncMock())
    trial._planes = planes
    await trial.install_agent()
    install.assert_awaited_once()

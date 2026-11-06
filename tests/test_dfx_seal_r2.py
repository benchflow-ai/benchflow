"""DFX-SEAL r2: regressions against 6e31325b7 and the compatible controls.

All custody checks and installers below run in a real Docker container.
"""

from __future__ import annotations

import shlex
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from tests.test_dfx_seal import LIBDIR, _docker_sandbox, _seal
from tests.test_dfx_seal import box as box
from tests.test_timing_spans import FakeClock, _install_trial, _quiet_planes


@pytest.mark.parametrize("walk", [False, True])
@pytest.mark.parametrize("suffix", ["\n", "\n\n"])
@pytest.mark.parametrize("middle", [False, True])
def test_exact_target_bytes_cannot_name_an_agent_creatable_file(
    box, walk, suffix, middle
):
    """Guard the target-byte bypass present at 6e31325b7, through every hop."""
    target = "/tmp/bf-safe" + suffix
    link = f"{LIBDIR}/libbf-tail.so"
    prep = "touch /tmp/bf-safe; chmod 644 /tmp/bf-safe; "
    if middle:
        prep += f"ln -s {shlex.quote(target)} /srv/bf-hop; "
        target = "/srv/bf-hop"
    prep += f"ln -s {shlex.quote(target)} {link}"
    assert box(prep).returncode == 0
    done = _seal(box, walk)
    assert done.returncode != 0, done.stdout
    assert "the system loader loads" in done.stderr


@pytest.mark.parametrize("walk", [False, True])
def test_a_dangling_link_with_a_newline_name_refuses(box, walk):
    """Guard the newline-name bypass present at 6e31325b7 in both seals."""
    link = f"{LIBDIR}/libbf-name.so\n"
    assert box(f"ln -s /tmp/bf-agent-library {shlex.quote(link)}").returncode == 0
    done = _seal(box, walk)
    assert done.returncode != 0, done.stdout
    assert "name with a newline" in done.stderr


@pytest.mark.asyncio
async def test_a_newline_name_is_a_nonretryable_loader_refusal(box):
    """Guard the 6275acf63 classification fix for the 6e31325b7 name bypass."""
    from benchflow._utils.scoring import classify_error, is_unrecoverable_startup_error
    from benchflow.sandbox.lockdown import LoaderCustodyError, setup_sandbox_user

    link = f"{LIBDIR}/libbf-name.so\n"
    assert (
        box(
            f"mkdir -p /srv/ws; ln -s /tmp/bf-agent-library {shlex.quote(link)}"
        ).returncode
        == 0
    )
    with pytest.raises(LoaderCustodyError, match="name with a newline") as caught:
        await setup_sandbox_user(
            _docker_sandbox(box.name), "agent", "/srv/ws", guard_daemon=False
        )
    assert classify_error(str(caught.value)) == "sandbox_setup"
    assert is_unrecoverable_startup_error(str(caught.value))


@pytest.mark.parametrize("walk", [False, True])
def test_exact_missing_target_in_a_root_folder_still_seals(box, walk):
    """Control: retain 6e31325b7's acceptance of a safe missing target."""
    target = f"{LIBDIR}/libbf-missing.so.1\n"
    assert box(f"ln -s {shlex.quote(target)} {LIBDIR}/libbf-safe.so").returncode == 0
    done = _seal(box, walk)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("walk", [False, True])
def test_a_root_owned_link_loop_keeps_the_existing_seal_result(box, walk):
    """Retain 6e31325b7's immutable-loop result, regressed at 2dd646551."""
    from benchflow._runtime_tree import kit_prepare_command

    assert box(f"ln -s libbf-loop.so {LIBDIR}/libbf-loop.so").returncode == 0
    done = box(kit_prepare_command("all", walk=walk))
    assert done.returncode == 0, done.stderr
    assert "BF kit-seal" in done.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_available", [True, False])
async def test_a_hard_link_repaired_by_the_path_lock_waits_for_the_seal(
    box, monkeypatch, identity_available
):
    """Guard the false early refusal of repaired hard links at 6e31325b7."""
    from benchflow.sandbox import lockdown

    library = f"{LIBDIR}/libbf-alias.so"
    assert (
        box(
            f"mkdir -p /srv/ws; touch {library}; chmod 644 {library}; "
            f"chown 2952:1009 {library}; ln {library} /usr/local/bin/bf-alias"
        ).returncode
        == 0
    )
    sandbox = _docker_sandbox(box.name)
    if not identity_available:
        original = lockdown._read_exec

        async def unavailable(env, command, **kwargs):
            # A real stat failure during the dry walk, before any mutation.
            if command.startswith("w="):
                command = (
                    'stat() { case "$*" in *%d:%i*) return 1;; '
                    '*) command stat "$@";; esac; }; ' + command
                )
            return await original(env, command, **kwargs)

        monkeypatch.setattr(lockdown, "_read_exec", unavailable)
    await lockdown.check_loader_custody(sandbox, "/srv/ws")
    assert box(f"stat -c %u {library}").stdout.strip() == "2952"
    await lockdown.setup_sandbox_user(sandbox, "agent", "/srv/ws", guard_daemon=False)
    assert sandbox.runtime_kit is not None
    assert box(f"stat -c '%u:%a' {library}").stdout.strip() == "0:644"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret", ["synthetic-guard-key-8392", "synthetic-guard\nkey-8392"]
)
async def test_failed_guard_output_is_redacted_in_every_trial_file(
    box, tmp_path, monkeypatch, secret
):
    """Guard 6e31325b7 artifact leaks and 94996123e result-message leaks."""
    from benchflow.rollout import _write_guard_progress, write_error_traceback
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.sandbox import lockdown

    monkeypatch.setenv("BENCHFLOW_TEST_API_KEY", secret)
    progress = "+0s install iptables\npackage output: " + secret + "\n"
    # Execute a failing guard in Docker and read its actual log through the
    # sandbox receipt. No network or firewall changes are needed.
    command = (
        "mkdir -p -m 700 /opt/benchflow/run; "
        f"printf %s {shlex.quote(progress)} > {lockdown.DAEMON_GUARD_LOG}; "
        f"printf '%s\\n' {shlex.quote('BF daemon-guard install: ' + secret)}; "
        "echo 'BF daemon-guard ports=2280 status=5'"
    )
    monkeypatch.setattr(lockdown, "_daemon_guard_cmd", lambda user: command)
    sandbox = _docker_sandbox(box.name)
    sandbox.sandbox_daemon_guard = "container"
    with pytest.raises(lockdown.SandboxDaemonGuardError) as caught:
        await lockdown.guard_sandbox_daemon(sandbox, "agent")
    assert secret in caught.value.progress
    trial = tmp_path / "trial"
    trial.mkdir()
    write_error_traceback(trial, caught.value)
    _write_guard_progress(trial, caught.value)
    _build_rollout_result(
        trial,
        task_name="guard-probe",
        rollout_name="guard-probe",
        agent="test-agent",
        agent_name="test-agent",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=str(caught.value),
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=None,
        started_at=datetime.now(),
        timing={},
    )
    files = [p for p in trial.rglob("*") if p.is_file()]
    assert {"error-traceback.txt", "daemon-guard.log", "result.json"} <= {
        p.name for p in files
    }
    for path in files:
        for part in secret.splitlines():
            assert part.encode() not in path.read_bytes(), path.name
    log = (trial / "daemon-guard.log").read_text()
    assert "package output: ***" in log
    assert "+0s install iptables" in log


@pytest.mark.asyncio
@pytest.mark.parametrize("read_error", [TimeoutError, OSError])
async def test_a_failed_diagnostic_read_keeps_the_guard_refusal(
    box, monkeypatch, read_error
):
    """Guard the secondary-read error masking the setup refusal at 6e31325b7."""
    from benchflow._utils.scoring import classify_error, is_unrecoverable_startup_error
    from benchflow.sandbox import lockdown

    monkeypatch.setattr(lockdown, "_daemon_guard_cmd", lambda user: "exit 124")
    sandbox = _docker_sandbox(box.name)
    sandbox.sandbox_daemon_guard = "container"
    with pytest.raises(lockdown.SandboxDaemonGuardError) as original:
        await lockdown.guard_sandbox_daemon(sandbox, "agent")
    read_exec = lockdown._read_exec

    async def unavailable(env, command, **kwargs):
        if command == f"cat {shlex.quote(lockdown.DAEMON_GUARD_LOG)}":
            raise read_error("diagnostic read unavailable")
        return await read_exec(env, command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", unavailable)
    with pytest.raises(lockdown.SandboxDaemonGuardError) as caught:
        await lockdown.guard_sandbox_daemon(sandbox, "agent")
    assert str(caught.value) == str(original.value)
    assert "the guard did not finish in 180 s" in str(caught.value)
    assert caught.value.progress == ""
    assert classify_error(str(caught.value)) == "sandbox_setup"
    assert not is_unrecoverable_startup_error(str(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["bf-custom-repair", "claude-agent-acp"])
async def test_a_custom_installer_repairs_the_library_before_the_seal(
    box, tmp_path, monkeypatch, agent
):
    """Guard the false refusal before a custom repair install at 6e31325b7."""
    from benchflow.agents import registry
    from benchflow.agents.install import install_agent
    from benchflow.sandbox import lockdown

    # Also cover a custom recipe registered under a built-in agent's name.
    for name in ("AGENTS", "AGENT_INSTALLERS", "AGENT_LAUNCH"):
        table = getattr(registry, name)
        for key in (agent,):
            monkeypatch.setitem(table, key, table.get(key))
    library = f"{LIBDIR}/libbf-repair.so"
    registry.register_agent(
        agent,
        f"chown 0:0 {library} && chmod 0644 {library} && touch /srv/installed",
        "true",
    )
    assert (
        box(
            f"mkdir -p /srv/ws; touch {library}; chmod 644 {library}; chown 2952:1009 {library}"
        ).returncode
        == 0
    )
    trial = _install_trial(tmp_path, agent)
    _quiet_planes(monkeypatch, trial, FakeClock())
    trial._env = _docker_sandbox(box.name)
    trial._agent_cwd = "/srv/ws"
    monkeypatch.setattr(
        trial, "_install_from_agent_layer", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(trial, "_install_native_cli", AsyncMock())
    monkeypatch.setattr(trial, "_prove_kept_freezer", AsyncMock())
    monkeypatch.setattr(trial._planes, "install_agent", install_agent)

    async def setup(env, user, workspace, **kwargs):
        return await lockdown.setup_sandbox_user(
            env, user, workspace, guard_daemon=False
        )

    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup)
    await trial.install_agent()
    assert box("test -f /srv/installed").returncode == 0
    assert trial._env.runtime_kit is not None
    assert box(f"stat -c '%u:%a' {library}").stdout.strip() == "0:644"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["identity", "dry"])
@pytest.mark.parametrize("read_error", [TimeoutError, OSError])
async def test_an_unavailable_repair_read_defers_to_the_seal(
    box, monkeypatch, stage, read_error
):
    """Guard an unavailable repair read stopping setup at 6e31325b7."""
    from benchflow.sandbox import lockdown

    library = f"{LIBDIR}/libbf-read.so"
    assert (
        box(
            f"mkdir -p /srv/ws; touch {library}; chmod 644 {library}; chown 2952:1009 {library}; ln {library} /usr/local/bin/bf-read"
        ).returncode
        == 0
    )
    sandbox = _docker_sandbox(box.name)
    original = lockdown._read_exec
    injected = False

    async def unavailable(env, command, **kwargs):
        nonlocal injected
        match = (
            command.startswith("readlink -f ")
            if stage == "identity"
            else command.startswith("w=")
        )
        if match and not injected:
            injected = True
            raise read_error("repair report unavailable")
        return await original(env, command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", unavailable)
    await lockdown.check_loader_custody(sandbox, "/srv/ws")
    assert injected
    assert box(f"stat -c %u {library}").stdout.strip() == "2952"
    await lockdown.setup_sandbox_user(sandbox, "agent", "/srv/ws", guard_daemon=False)
    assert sandbox.runtime_kit is not None
    assert box(f"stat -c %u {library}").stdout.strip() == "0"

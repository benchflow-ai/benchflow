"""Tests for ManifestEnvironment over a fake sandbox.

ManifestEnvironment runs the in-sandbox topology: it starts the manifest's
services and health-checks them via sandbox.exec. One manifest serves a
whole benchmark — per-task images carrying only a subset of the services
are handled by entry-point (`<binary> --help`) detection.
"""

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import (
    EnvironmentSnapshotError,
    ManifestEnvironment,
)
from benchflow.environment.protocol import Environment, StateSnapshot
from benchflow.sandbox.protocol import ExecResult

CLAWS = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "clawsbench"
base_image     = "kywch/smolclaws-base:latest"
owns_lifecycle = false

[[environment.services]]
name    = "gmail"
command = "claw-gmail --db /data/gmail.db serve --host 0.0.0.0 --port 9001 --no-mcp"
port    = 9001

[[environment.services]]
name    = "gcal"
command = "claw-gcal --db /data/gcal.db serve --host 0.0.0.0 --port 9003 --no-mcp"
port    = 9003

[environment.readiness]
timeout_sec = 5
"""
)

CHI = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
ports          = [8020, 8023]
owns_lifecycle = true

[environment.readiness]
http        = ["http://localhost:8023/health"]
timeout_sec = 5
"""
)

# A stateful ClawsBench manifest — declares [environment.state] so the
# environment supports snapshot/restore (Feature A).
CLAWS_STATEFUL = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "clawsbench"
base_image     = "kywch/smolclaws-base:latest"
owns_lifecycle = false

[[environment.services]]
name    = "gmail"
command = "claw-gmail --db /data/gmail.db serve --host 0.0.0.0 --port 9001 --no-mcp"
port    = 9001

[environment.state]
kind  = "sqlite"
paths = ["/data/gmail.db", "/data/gcal.db"]
"""
)


class FakeSandbox:
    """Minimal Sandbox stand-in recording exec calls.

    ``absent_binaries`` — the `<binary> --help` detection probe returns
    non-zero for these: the service's package is not installed in this
    per-task image (a bare PATH stub, or genuinely missing).
    ``fail_health_for`` — the curl health-check returns non-zero for any
    command mentioning this substring.
    """

    def __init__(
        self,
        *,
        absent_binaries: set[str] | None = None,
        fail_health_for: str | None = None,
    ) -> None:
        self.exec_calls: list[str] = []
        self._absent = absent_binaries or set()
        self._fail_health_for = fail_health_for
        self.host = "localhost"
        self.expose_ports: list[int] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        self.exec_calls.append(cmd)
        if "--help" in cmd:  # the service-detection probe: `<binary> --help`
            binary = cmd.split()[0]
            rc = 1 if binary in self._absent else 0
            return ExecResult(return_code=rc, stdout="", stderr="")
        if "curl" in cmd and self._fail_health_for and self._fail_health_for in cmd:
            return ExecResult(return_code=1, stdout="", stderr="")
        return ExecResult(return_code=0, stdout="", stderr="")


async def test_satisfies_environment_protocol():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    assert isinstance(env, Environment)


async def test_provision_starts_present_services():
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx={"task_id": "multi-mail-cal-sync"})
    starts = [c for c in sandbox.exec_calls if "serve" in c]
    assert len(starts) == 2
    assert all(c.rstrip().endswith("&") for c in starts)


async def test_provision_skips_absent_service_binary():
    sandbox = FakeSandbox(absent_binaries={"claw-gcal"})
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    starts = [c for c in sandbox.exec_calls if "serve" in c]
    assert len(starts) == 1
    assert "claw-gmail" in starts[0]


async def test_provision_detects_services_by_running_the_entry_point():
    """Detection probes `<binary> --help`, never bare `command -v`.

    Regression: smolclaws-style base images ship a console-script *stub* for
    every claw-* service, so `command -v` over-detects a service whose package
    a per-task image never installed — it then starts a crashing process and
    the readiness gate hangs. The probe must run the entry point.
    """
    sandbox = FakeSandbox(absent_binaries={"claw-gcal"})  # gcal: PATH stub, no pkg
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    assert any("claw-gmail --help" in c for c in sandbox.exec_calls)
    assert not any(c.startswith("command -v ") for c in sandbox.exec_calls)
    # the gcal stub is skipped — only the real gmail service starts
    assert [c for c in sandbox.exec_calls if "serve" in c] == [
        c for c in sandbox.exec_calls if "serve" in c and "claw-gmail" in c
    ]


async def test_provision_returns_handle_with_all_declared_endpoints():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    handle = await env.provision(ctx=None)
    assert handle.name == "clawsbench"
    assert handle.endpoints == {
        9001: "http://localhost:9001",
        9003: "http://localhost:9003",
    }


async def test_provision_skips_service_start_when_owns_lifecycle():
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CHI, sandbox=sandbox)
    await env.provision(ctx=None)
    assert sandbox.exec_calls == []  # entrypoint owns the lifecycle


async def test_readiness_ready_when_all_health_checks_pass():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    await env.provision(ctx=None)
    probe = await env.readiness()
    assert probe.ready is True
    assert probe.checked == [
        "http://localhost:9001/health",
        "http://localhost:9003/health",
    ]


async def test_readiness_only_probes_started_services():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox(absent_binaries={"claw-gcal"}))
    await env.provision(ctx=None)
    probe = await env.readiness()
    assert probe.ready is True
    assert probe.checked == ["http://localhost:9001/health"]


async def test_readiness_fails_when_a_service_never_healthy():
    sandbox = FakeSandbox(fail_health_for="9003")
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    probe = await env.readiness()
    assert probe.ready is False
    assert "9003" in (probe.error or "")


async def test_readiness_owns_lifecycle_uses_manifest_http():
    env = ManifestEnvironment(CHI, sandbox=FakeSandbox())
    await env.provision(ctx=None)
    probe = await env.readiness()
    assert probe.ready is True
    assert probe.checked == ["http://localhost:8023/health"]


async def test_teardown_stops_started_services():
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    await env.teardown()
    assert any("pkill" in c and "claw-gmail" in c for c in sandbox.exec_calls)


async def test_query_returns_empty_state():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    state = await env.query()
    assert state.data == {}


async def test_snapshot_backs_up_sqlite_state_and_returns_id():
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=sandbox)
    await env.provision(ctx=None)
    snap = await env.snapshot()
    assert isinstance(snap, StateSnapshot)
    assert snap.id  # non-empty
    backup = [c for c in sandbox.exec_calls if ".backup" in c]
    assert backup, "snapshot must back up the sqlite db files"
    assert "/data/gmail.db" in backup[0]
    assert "/data/gcal.db" in backup[0]


async def test_restore_copies_snapshot_files_back():
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=sandbox)
    await env.provision(ctx=None)
    snap = await env.snapshot()
    sandbox.exec_calls.clear()
    await env.restore(snap)
    restore_cmds = [c for c in sandbox.exec_calls if "cp " in c]
    assert restore_cmds, "restore must copy the snapshot files back"
    assert snap.path in restore_cmds[0]
    assert "/data/gmail.db" in restore_cmds[0]
    assert "/data/gcal.db" in restore_cmds[0]


async def test_snapshot_restore_on_stateless_env_raise_clear_error():
    """An env with no [environment.state] is stateless — snapshot/restore
    must fail with a clear message, not crash or be silently empty."""
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())  # no [environment.state]
    with pytest.raises(RuntimeError, match="stateless"):
        await env.snapshot()
    with pytest.raises(RuntimeError, match="stateless"):
        await env.restore(StateSnapshot(id="x"))


async def test_reset_before_provision_is_a_clear_error():
    """Reset has no baseline before provision — fail loudly instead of silently
    doing nothing. Guards the fix for #383: reset() used to ``raise
    NotImplementedError`` unconditionally, masking real lifecycle bugs."""
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    with pytest.raises(RuntimeError, match="provision"):
        await env.reset()


async def test_reset_restarts_framework_started_services_for_stateless_env():
    """A stateless framework-started manifest has no baseline to restore, but
    reset must still cycle the services so the next episode starts from a
    clean process state. Guards the fix for #383."""
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    sandbox.exec_calls.clear()
    await env.reset()
    pkills = [c for c in sandbox.exec_calls if "pkill" in c]
    restarts = [
        c for c in sandbox.exec_calls if "serve" in c and c.rstrip().endswith("&")
    ]
    assert len(pkills) == 2, "reset must stop every framework-started service"
    assert len(restarts) == 2, "reset must restart every framework-started service"
    # No restore — there is no [environment.state] table, so no snapshot
    # directory should be copied back over the live state files.
    assert not any("/tmp/benchflow-snapshots/" in c for c in sandbox.exec_calls)


async def test_reset_restores_baseline_state_for_stateful_env():
    """A stateful manifest captures a baseline during provision; reset must
    copy the baseline files back over the live paths so the next episode sees
    the seed data, not whatever the previous agent left behind. Guards the
    fix for #383."""
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=sandbox)
    await env.provision(ctx=None)
    # Baseline backup runs during provision, before reset is called.
    provision_backups = [c for c in sandbox.exec_calls if ".backup" in c]
    assert provision_backups, "provision must capture a baseline for stateful manifests"
    sandbox.exec_calls.clear()
    await env.reset()
    restore_cmds = [c for c in sandbox.exec_calls if "/tmp/benchflow-snapshots/" in c]
    assert restore_cmds, "reset must copy the baseline state files back"
    assert "/data/gmail.db" in restore_cmds[0]
    # And restart the service we previously started.
    assert any("serve" in c and c.rstrip().endswith("&") for c in sandbox.exec_calls)


async def test_reset_on_owns_lifecycle_env_does_not_touch_services():
    """When the image entrypoint owns the lifecycle, the framework never
    started the services and cannot restart them. Reset on a stateless
    owns_lifecycle manifest is therefore a no-op against the sandbox.
    Guards the fix for #383."""
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CHI, sandbox=sandbox)
    await env.provision(ctx=None)
    sandbox.exec_calls.clear()
    await env.reset()
    assert sandbox.exec_calls == []


async def test_reset_is_idempotent_across_multiple_calls():
    """Reset can be called repeatedly between episodes — each call must
    re-stop and re-start the framework-started services without drifting
    state. Guards the fix for #383."""
    sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS, sandbox=sandbox)
    await env.provision(ctx=None)
    await env.reset()
    sandbox.exec_calls.clear()
    await env.reset()
    pkills = [c for c in sandbox.exec_calls if "pkill" in c]
    restarts = [c for c in sandbox.exec_calls if "serve" in c]
    assert len(pkills) == 2
    assert len(restarts) == 2


# --- #387: snapshot/restore must surface sandbox-command failures ----------


class _FailingSandbox:
    """Sandbox stand-in whose ``exec`` returns non-zero for snapshot/restore.

    Mirrors the minimum surface ``ManifestEnvironment`` calls — readiness/
    detection probes succeed so ``provision`` reaches the baseline-capture
    step, but any ``.backup`` (snapshot) or ``cp`` (restore) command fails
    so the regression for #387 has something to surface.
    """

    def __init__(self, *, fail_on: str) -> None:
        self.exec_calls: list[str] = []
        self._fail_on = fail_on
        self.host = "localhost"
        self.expose_ports: list[int] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        self.exec_calls.append(cmd)
        if self._fail_on in cmd:
            return ExecResult(
                return_code=1, stdout="", stderr="sqlite3: file is not a database"
            )
        return ExecResult(return_code=0, stdout="", stderr="")


async def test_snapshot_raises_when_sandbox_command_fails():
    """Guards the fix from PR #486 for #387: a failed ``sqlite3 .backup`` must
    raise instead of returning a bogus StateSnapshot — otherwise Branch records
    a checkpoint that never existed and children get scored against corrupted
    state."""
    sandbox = _FailingSandbox(fail_on=".backup")
    # Build the env without calling provision() — provision would itself
    # trigger _capture_baseline, and we want to test snapshot() in isolation.
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=sandbox)
    with pytest.raises(EnvironmentSnapshotError) as exc_info:
        await env.snapshot()
    err = exc_info.value
    assert err.exit_code == 1
    assert ".backup" in err.command
    assert "sqlite3" in err.stderr


async def test_restore_raises_when_sandbox_command_fails():
    """Guards the fix from PR #486 for #387: a failed ``cp`` during restore
    must raise instead of returning success — the live state is unchanged but
    the caller thinks rollback worked, which corrupts every subsequent branch
    child."""
    # Use a non-failing sandbox to take a snapshot, then swap in a failing
    # one so restore's `cp` returns non-zero.
    good_sandbox = FakeSandbox()
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=good_sandbox)
    snap = await env.snapshot()

    bad_sandbox = _FailingSandbox(fail_on="cp ")
    env_for_restore = ManifestEnvironment(CLAWS_STATEFUL, sandbox=bad_sandbox)
    with pytest.raises(EnvironmentSnapshotError) as exc_info:
        await env_for_restore.restore(snap)
    err = exc_info.value
    assert err.exit_code == 1
    assert "cp " in err.command
    assert snap.id in str(err)


async def test_provision_baseline_capture_failure_surfaces():
    """Guards the fix from PR #486 for #387: ``provision`` captures a baseline
    for stateful manifests via the same snapshot path — if the sandbox command
    fails there, provision must raise rather than leaving the env with no
    baseline + believing setup succeeded."""
    sandbox = _FailingSandbox(fail_on=".backup")
    env = ManifestEnvironment(CLAWS_STATEFUL, sandbox=sandbox)
    with pytest.raises(EnvironmentSnapshotError):
        await env.provision(ctx=None)

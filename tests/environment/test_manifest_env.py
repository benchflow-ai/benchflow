"""Tests for ManifestEnvironment over a fake sandbox.

ManifestEnvironment runs the in-sandbox topology: it starts the manifest's
services and health-checks them via sandbox.exec. One manifest serves a
whole benchmark — per-task images carrying only a subset of the services
are handled by `command -v` detection.
"""

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
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


class FakeSandbox:
    """Minimal Sandbox stand-in recording exec calls.

    ``absent_binaries`` — `command -v` returns non-zero for these (the
    binary is not in this per-task image).
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
        if cmd.startswith("command -v "):
            binary = cmd.split()[2]
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


async def test_platform_layer_methods_not_implemented():
    env = ManifestEnvironment(CLAWS, sandbox=FakeSandbox())
    with pytest.raises(NotImplementedError):
        await env.reset()
    with pytest.raises(NotImplementedError):
        await env.snapshot()
    with pytest.raises(NotImplementedError):
        await env.restore(StateSnapshot(id="x"))

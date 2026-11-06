"""Guards 1887d869: cache packages without changing runtime enforcement."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchflow.sandbox import daytona_layers as layers


def _plan(**kwargs):
    return layers.plan_install_layers(agent=None, model=None, **kwargs)


def test_guard_key_and_recipe_depend_only_on_image_and_install():
    """Guards 1887d869: keys track the image and install inputs."""
    plan = _plan(guard_daemon=True)
    assert plan.guard is not None
    base = "FROM ubuntu:24.04\nUSER task\n"
    first = layers.layered_recipe(plan, base)
    assert first == layers.layered_recipe(_plan(guard_daemon=True), base)
    assert first.lines[0] == "USER root"
    assert first.lines[-1] == "USER task"
    assert list(first.keys) == ["guard-daemon"]
    assert plan.guard.marker == "/opt/runtime/layers/guard-daemon.key"
    assert first.keys["guard-daemon"] in "\n".join(first.lines)
    assert plan.guard.key(base) != plan.guard.key("FROM alpine:3.24\n")
    changed = replace(plan.guard, script=plan.guard.script + "; true")
    assert plan.guard.key(base) != changed.key(base)
    assert "/proc" not in plan.guard.script
    assert "--uid-owner" not in plan.guard.script


@pytest.mark.parametrize("fresh", [False, True])
def test_guard_disk_is_reserved_even_when_installing_fresh(fresh):
    """Guards 1887d869: reserve package disk within the account cap."""
    from benchflow.sandbox.lockdown import DAEMON_GUARD_TIMEOUT_SEC

    plan = _plan(guard_daemon=True, fresh=fresh)
    assert plan.footprint_mb == layers.GUARD_FOOTPRINT_MB
    assert layers.disk_gib(1024, plan.footprint_mb, 10240) == 2
    assert layers.disk_gib(10240, plan.footprint_mb, 10240) == 10
    assert plan.build_allowance_sec == (0 if fresh else DAEMON_GUARD_TIMEOUT_SEC)
    assert bool(layers.layered_recipe(plan, "FROM task\n").lines) is not fresh
    assert _plan().layers == ()
    assert _plan().footprint_mb == 0


_MANAGER = """#!/bin/sh
printf '%s|%s\\n' "${0##*/} $*" "${DEBIAN_FRONTEND-}" >> "$PKG_LOG"
[ "$PKG_MODE" != failed ] || exit 100
if [ "$1" = update ]; then : > "$PKG_UPDATED"; exit 0; fi
if [ "$PKG_MODE" = retry ] && [ ! -f "$PKG_UPDATED" ]; then exit 100; fi
for tool in iptables ip6tables; do
    printf '#!/bin/sh\\nexit 0\\n' > "$PKG_BIN/$tool"
    chmod +x "$PKG_BIN/$tool"
done
"""


def _run_layer(tmp_path, monkeypatch, *, manager=None, installed=(), mode="ok"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "mkdir", "chmod"):
        executable = shutil.which(tool)
        assert executable is not None
        (bin_dir / tool).symlink_to(executable)
    for tool in installed:
        path = bin_dir / tool
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    # The shell fixture models a root image build without changing host identity.
    (bin_dir / "id").write_text("#!/bin/sh\necho 0\n")
    (bin_dir / "id").chmod(0o755)
    if manager:
        path = bin_dir / manager
        path.write_text(_MANAGER)
        path.chmod(0o755)
    monkeypatch.setattr(layers, "LAYER_KEY_DIR", str(tmp_path / "keys"))
    layer = _plan(guard_daemon=True).guard
    assert layer is not None
    command = layers.layer_run_line(layer, "expected-key").removeprefix("RUN ")
    log = tmp_path / "packages.log"
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={
            "PATH": str(bin_dir),
            "PKG_BIN": str(bin_dir),
            "PKG_LOG": str(log),
            "PKG_MODE": mode,
            "PKG_UPDATED": str(tmp_path / "updated"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, Path(layer.marker), log.read_text() if log.exists() else ""


@pytest.mark.parametrize(
    ("manager", "expected"),
    [
        ("apt-get", "apt-get install -y -q iptables|noninteractive\n"),
        ("dnf", "dnf -y install iptables|\n"),
        ("apk", "apk add --no-cache iptables ip6tables|\n"),
    ],
)
@pytest.mark.parametrize("installed", [(), ("iptables",)])
def test_guard_layer_installs_both_families(
    tmp_path, monkeypatch, manager, expected, installed
):
    """Guards 1887d869: prepare IPv6 even on an IPv4-only build."""
    result, marker, log = _run_layer(
        tmp_path, monkeypatch, manager=manager, installed=installed
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "expected-key\n"
    assert log == expected


def test_guard_layer_apt_retries_with_the_runtime_commands(tmp_path, monkeypatch):
    """Guards 1887d869: stale apt lists retry update then install once."""
    result, marker, log = _run_layer(
        tmp_path, monkeypatch, manager="apt-get", mode="retry"
    )
    assert result.returncode == 0
    assert marker.exists()
    assert log.splitlines() == [
        "apt-get install -y -q iptables|noninteractive",
        "apt-get update -q|",
        "apt-get install -y -q iptables|noninteractive",
    ]


@pytest.mark.parametrize("manager", [None, "apt-get", "dnf", "apk"])
def test_guard_layer_skips_installs_when_both_tools_exist(
    tmp_path, monkeypatch, manager
):
    """Guards 1887d869: ready images never call a package manager."""
    result, marker, log = _run_layer(
        tmp_path,
        monkeypatch,
        manager=manager,
        installed=("iptables", "ip6tables"),
    )
    assert result.returncode == 0
    assert marker.exists()
    assert log == ""


@pytest.mark.parametrize("manager", [None, "apt-get", "dnf", "apk"])
def test_guard_layer_failed_attempt_discards_the_build(tmp_path, monkeypatch, manager):
    """Review F1 against 1887d869's committed failed install."""
    result, marker, _ = _run_layer(
        tmp_path, monkeypatch, manager=manager, mode="failed"
    )
    # Only an unavailable manager may keep the original image state.
    assert (result.returncode == 0) is (manager is None), result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("reuse", [False, True])
async def test_single_trial_keeps_guard_and_reuse_keeps_other_layers(
    tmp_path, monkeypatch, reuse
):
    """Guards 1887d869: keep the guard through the optional reuse gate."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client

    monkeypatch.setattr(layers, "_BUILT", set())
    monkeypatch.setattr(layers, "_BUILDING", set())
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1" if reuse else "0")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    client = _fake_client(monkeypatch, FakeDaytona)
    first = await _start(
        client, tmp_path / "one", guard_daemon=True, only_if_reused=True
    )
    second = await _start(
        client, tmp_path / "two", guard_daemon=True, only_if_reused=True
    )
    keys = first._install_layer_keys
    assert "guard-daemon" in keys
    assert ("proxy-litellm" in keys) is reuse
    assert ("agent-claude-agent-acp" in keys) is reuse
    assert (
        client.requests[0].image.dockerfile() == client.requests[1].image.dockerfile()
    )
    assert second._install_layer_keys == keys
    assert first.cached_install_layer("guard", "daemon") is not None
    assert first._install_layers_record["mode"] == "layers"


@pytest.mark.parametrize("opt_out", ["fresh", "persistent-env", "compose"])
def test_sandbox_omits_guard_for_existing_layer_opt_outs(tmp_path, opt_out):
    """Guards 1887d869: compose and fresh or mutable installs stay live."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _sandbox

    sandbox = _sandbox(FakeDaytona(), tmp_path)
    sandbox._install_layers = None
    sandbox._compose_mode = opt_out == "compose"
    sandbox._persistent_env = (
        {"example": "value"} if opt_out == "persistent-env" else {}
    )
    sandbox.configure_install_layers(
        agent=None, model=None, guard_daemon=True, fresh_install=opt_out == "fresh"
    )
    assert sandbox._install_layers is None or sandbox._install_layers.layers == ()


@pytest.mark.parametrize(
    ("agent", "user", "verifier_only", "expected"),
    [
        ("claude-agent-acp", "agent", False, True),
        ("claude-agent-acp", "different-user", False, True),
        ("claude-agent-acp", None, False, False),
        ("oracle", "agent", False, False),
        ("nop", "agent", False, False),
        ("script:operator.sh", "agent", False, True),
        ("claude-agent-acp", "agent", True, False),
    ],
)
def test_rollout_plans_guard_only_when_it_will_run(
    tmp_path, agent, user, verifier_only, expected
):
    """Guards 1887d869: eligibility mirrors setup's guard switch."""
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_daytona_install_layers import _PlannedSandbox

    if agent.startswith("script:"):
        script = tmp_path / "operator.sh"
        script.write_text("#!/bin/sh\necho hello\n")
        agent = f"script:{script}"
    rollout = Rollout(RolloutConfig(task_path=tmp_path, agent=agent, sandbox_user=user))
    rollout._env = _PlannedSandbox()
    rollout._verifier_only = verifier_only
    rollout._configure_daytona_layers()
    assert rollout._env.plans[0]["guard_daemon"] is expected


@pytest.mark.parametrize("source", ["flag", "env"])
def test_rollout_fresh_install_omits_guard(tmp_path, monkeypatch, source):
    """Guards 1887d869: both fresh-install controls omit packages."""
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_daytona_install_layers import _PlannedSandbox

    monkeypatch.setenv(layers.FRESH_INSTALL_ENV, "1" if source == "env" else "0")
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path,
            agent="claude-agent-acp",
            fresh_install=source == "flag",
        )
    )
    rollout._env = _PlannedSandbox()
    rollout._configure_daytona_layers()
    kwargs = rollout._env.plans[0]
    assert kwargs["guard_daemon"] is True
    assert kwargs["fresh_install"] is True


async def test_failed_layer_build_uses_plain_image_and_live_guard(
    tmp_path, monkeypatch
):
    """Guards 1887d869: recipe fallback preserves runtime enforcement."""
    pytest.importorskip("daytona")
    from benchflow.sandbox.lockdown import guard_sandbox_daemon
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client
    from tests.test_sdk_lockdown import TestSandboxDaemonGuard

    client = _fake_client(monkeypatch, FakeDaytona, fail_layered_builds=True)
    sandbox = await _start(client, tmp_path / "trial", guard_daemon=True)
    assert sandbox._install_layer_keys == {}
    assert "RUN " not in client.requests[-1].image.dockerfile()
    env = TestSandboxDaemonGuard._env("container")
    env.cached_install_layer = lambda *_: (_plan(guard_daemon=True).guard, "key")
    await guard_sandbox_daemon(env, "agent")
    assert "BF daemon-guard" in env.exec.await_args.args[0]

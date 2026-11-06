"""Guards V7's merge of b89ae706: J1 shells, K2 keeps and U13 creates."""

import logging
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import SYSTEM_SHELL_ROUTE, ExecResult, ExecRoute
from tests._receipt_fakes import receipt_aware


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [False, True])
async def test_daytona_system_plumbing_never_restores_image_activation(
    tmp_path, system
):
    """b89ae706 merge: task shells preserve data; system shells never activate it."""
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk

    _load_daytona_sdk()
    marker = tmp_path / "startup-ran"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"printf ran > {marker}\n")
    results = []

    class Process:
        async def create_session(self, session_id):
            pass

        async def execute_session_command(self, session_id, request, timeout=None):
            result = subprocess.run(
                ["/bin/sh", "-c", request.command],
                env={**os.environ, "BASH_ENV": str(startup)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            results.append(result)
            return SimpleNamespace(cmd_id="cmd-1")

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=Process())
    sandbox._poll_response = AsyncMock(return_value=ExecResult(return_code=0))
    route = SYSTEM_SHELL_ROUTE if system else ExecRoute(pin=None, task_root=False)
    await sandbox._sandbox_exec('printf "%s" "${BASH_ENV-}"', route=route)
    assert results[0].returncode == 0, results[0].stderr
    assert results[0].stdout == ("" if system else str(startup))
    assert not marker.exists()


@pytest.mark.asyncio
async def test_offline_keeps_system_shell_and_empty_startup_env():
    """b89ae706 merge: root transport and the bash network probe both stay clean."""
    from benchflow.taskmd.verify import ROOT_PATH, take_offline

    sandbox = SimpleNamespace(
        supports_system_shell=True,
        exec=AsyncMock(return_value=ExecResult(return_code=0, stdout="offline")),
    )
    assert await take_offline(sandbox) == "offline"
    args, kwargs = sandbox.exec.call_args
    assert args[0].startswith(ROOT_PATH + "\n")
    assert "BASH_ENV= ENV= bash -c" in args[0]
    assert kwargs == {
        "user": "root",
        "timeout_sec": 60,
        "system_shell": True,
        "env": {"BASH_ENV": "", "ENV": ""},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
async def test_activation_with_kept_services_precedes_every_ownership_change(
    tmp_path, monkeypatch, caplog, separate
):
    """b89ae706 merge: keep-aware quiescence precedes inspection and ownership edits."""
    from benchflow.task import keep_services
    from benchflow.task.config import TaskConfig

    config = SimpleNamespace(
        task_dir=tmp_path,
        paths=SimpleNamespace(uses_native_verifier_dir=False),
        config=TaskConfig.model_validate_toml('version = "1.0"\n'),
    )
    config.config.verifier.env = {"BASH_ENV": "/etc/profile.d/safe.sh"}
    for name, answer in (
        ("_trusted_verifier_path", "/usr/bin"),
        ("_trusted_verifier_pythonpath", ""),
        ("_distro_pip_env", {}),
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugin_flags", ""),
        # The plugin guard target's hardening discovers required and optional
        # plugins through this one (#1175).
        ("_discover_pytest_plugins", ([], [])),
        ("_verifier_uid", None),
        ("_install_pytest_plugin_guard", ("", "", True, ())),
    ):
        monkeypatch.setattr(lockdown, name, AsyncMock(return_value=answer))
    order = []

    async def inspect(*args):
        order.append("inspect")
        return "/etc/profile.d/safe.sh"

    async def freeze(*args, **kwargs):
        order.append("ownership")

    async def kill(*args, **kwargs):
        order.append("keep-kill")
        assert args[2] == (5000,) and kwargs["wait"] is True
        return object()

    keep = SimpleNamespace(ports=(5000,))
    monkeypatch.setattr(keep_services, "trial_keep", lambda *a: keep)
    monkeypatch.setattr(keep_services, "waited", lambda *a: False)
    monkeypatch.setattr(keep_services, "mark_waited", lambda *a: order.append("waited"))
    monkeypatch.setattr(
        lockdown, "_record_kept", lambda *a, **k: order.append("record")
    )
    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs_keeping", kill)
    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", AsyncMock())
    monkeypatch.setattr(lockdown, "_trusted_verifier_bash_env", inspect, raising=False)
    monkeypatch.setattr(lockdown, "_restore_workspace_state", freeze)
    monkeypatch.setattr(lockdown, "_freeze_workspace", freeze)
    from tests.sandbox_mounts import ROOT_MOUNTINFO

    sandbox = SimpleNamespace(
        _benchflow_path_inspectors={"main": "/etc/fake-inspector"},
        exec=AsyncMock(
            side_effect=receipt_aware(
                lambda command, **kw: ExecResult(
                    return_code=0,
                    stdout=ROOT_MOUNTINFO
                    if "/proc/self/mountinfo" in command
                    else "A /usr/bin/python3\n"
                    if "bf-root-python" in command
                    else "complete\n"
                    if "printf 'complete\\n'" in command
                    else "",
                )
            )
        ),
    )
    with caplog.at_level(logging.WARNING):
        await lockdown.harden_before_verify(
            sandbox,
            config,
            "agent",
            "/app",
            restore_workspace=True,
            agent_paths=("/app",) if separate else None,
        )
    if separate:
        assert config.config.verifier.env["BASH_ENV"] == ""
        assert "separate verifier sandbox" in caplog.text
        assert order == ["ownership", "ownership"]
    else:
        assert config.config.verifier.env["BASH_ENV"] == "/etc/profile.d/safe.sh"
        assert order == [
            "keep-kill",
            "waited",
            "record",
            "inspect",
            "ownership",
            "ownership",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ["image", "content", "recipe-fallback", "plain-fallback", "branch", "template"],
)
async def test_each_daytona_create_path_hides_toolbox_activation(
    tmp_path, monkeypatch, route
):
    """b89ae706 merge: U13/U3 retries and checkpoints use V7's create treatment."""
    pytest.importorskip("daytona")
    from benchflow.sandbox import daytona as sdk
    from tests.test_daytona_install_layers import FakeDaytona, _configure, _sandbox

    fake = FakeDaytona(fail_layered_builds=route == "plain-fallback")
    manager = SimpleNamespace(get_client=AsyncMock(return_value=fake))
    monkeypatch.setattr(
        sdk,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    sandbox = _sandbox(fake, tmp_path)
    _configure(sandbox, agent="claude-agent-acp", model="dummy", agent_env={})
    if route == "image":
        sandbox._install_layers = None
        sandbox._install_layers_record = None
    if route in ("content", "recipe-fallback"):
        monkeypatch.setattr(
            sandbox._strategy,
            "_content_snapshot",
            AsyncMock(return_value="content-snapshot"),
        )
    if route == "branch":
        sandbox._start_snapshot_ref = "branch-snapshot"
    if route == "template":
        sandbox._snapshot_template_name = "template-snapshot"
        fake.snapshot = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(state=sdk.SnapshotState.ACTIVE))
        )

    original = fake.create

    async def create(params, timeout=None):
        if isinstance(params, sdk.CreateSandboxFromSnapshotParams):
            fake.requests.append(params)
            if route == "recipe-fallback":
                from daytona.common.errors import DaytonaNotFoundError

                # Guards PR #13: a rejected snapshot may fall back; an unknown
                # create outcome must retain ownership instead.
                raise DaytonaNotFoundError(
                    "dummy snapshot unavailable", status_code=404
                )
            return SimpleNamespace(id="sb-snapshot", delete=AsyncMock())
        return await original(params, timeout)

    fake.create = create
    startup = "/opt/image-activation.sh"
    probes = []

    async def toolbox():
        value = (fake.requests[-1].env_vars or {}).get("BASH_ENV", startup)
        probes.append(value)
        return {**dict.fromkeys(sdk.SYSTEM_SHELL_EMPTY_VARS, ""), "BASH_ENV": value}

    sandbox._toolbox_startup_env = toolbox
    sandbox._start_snapshot_env = {
        **dict.fromkeys(sdk.SYSTEM_SHELL_EMPTY_VARS, ""),
        "BASH_ENV": startup,
    }
    await sandbox._strategy.start(force_build=False)
    assert probes == ([""] if route == "branch" else [startup, ""])
    assert fake.requests[-1].env_vars["BASH_ENV"] == ""
    assert fake.requests[-1].env_vars["_BF_IMAGE_BASH_ENV"] == startup
    if route == "recipe-fallback":
        assert sandbox._install_layers_record["snapshot"] == "fallback"
    if route == "plain-fallback":
        assert sandbox._install_layers_record["fallback"] == "layer build failed"
        assert fake.requests[-1].image.dockerfile().count("RUN ") == 0

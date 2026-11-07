"""Guards FF-sw-daytona-C7, the provider-shell sibling left at 451b4cf5.

The local transport starts the outer shell with the provider's create-time
environment, before any command-local unset. No cloud calls are made.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.sandbox import daytona as sdk
from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
from benchflow.sandbox.daytona import (
    DaytonaSandbox,
    SandboxStartupError,
    _DaytonaDirect,
)
from benchflow.sandbox.protocol import SandboxImage

pytest.importorskip("daytona")
sdk._load_daytona_sdk()

LOADERS = tuple(name for name in SYSTEM_SHELL_EMPTY_VARS if name != "BASH_ENV")
META = "benchflow_toolbox_startup_v1"


def local_sandbox(image_env):
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = logging.getLogger(__name__)
    sandbox._sandbox = None
    sandbox._strategy = _DaytonaDirect(sandbox)
    sandbox._auto_delete_interval = sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._persistent_env = {}
    sandbox._create_attempts = 1
    sandbox._create_outcome_unknown = False
    sandbox.task_env_config = SimpleNamespace(build_timeout_sec=60)
    created = []
    launched = []

    async def create(params):
        created.append(params)
        provider_env = {**os.environ, **image_env, **(params.env_vars or {})}

        async def execute(command, env=None, timeout=None):
            actual = {**provider_env, **(env or {})}
            launched.append(actual)
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                env=actual,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return SimpleNamespace(exit_code=result.returncode, result=result.stdout)

        sandbox._sandbox = SimpleNamespace(
            id="local", process=SimpleNamespace(exec=execute), delete=AsyncMock()
        )

    sandbox._create_sandbox = create
    # The restored sandbox's daemon log is made root-only by a root exec of
    # its own before anything else runs (K1-R1-B-1).
    sandbox._sandbox_exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0, stdout="BF daemon-log root-only\n", stderr=""
        )
    )
    return sandbox, created, launched


@pytest.mark.parametrize("name", LOADERS)
@pytest.mark.parametrize("source", ["image", "trusted-template"])
async def test_loader_images_start_with_the_value_carried_aside(name, source):
    """DAYTONA-GAPS rule 2 (refused at 29b6ea03c): the kept sandbox is
    created with every loader and start-up name empty, so no toolbox shell
    starts with the image's value, and the value is carried as data under
    its _BF_IMAGE_ name for the task route to restore."""
    sandbox, created, launched = local_sandbox({name: "/tmp/dummy-loader"})
    params = (
        sdk.CreateSandboxFromImageParams(image=sdk.Image.base("dummy"))
        if source == "image"
        else sdk.CreateSandboxFromSnapshotParams(snapshot="trusted-template")
    )
    await sandbox._create_sandbox_without_toolbox_bash_env(params)
    assert len(created) == 2
    env = created[-1].env_vars
    assert all(env.get(other) == "" for other in SYSTEM_SHELL_EMPTY_VARS)
    assert env[f"_BF_IMAGE_{name}"] == "/tmp/dummy-loader"
    # The probe of the kept sandbox: its shell started with the name empty.
    assert launched[-1][name] == ""
    assert sandbox._toolbox_startup_receipt[name] == "/tmp/dummy-loader"
    assert sandbox._sandbox is not None


@pytest.mark.parametrize("name", LOADERS)
@pytest.mark.parametrize("entry", ["restore", "branch"])
async def test_unproven_checkpoint_refused_before_any_provider_call(name, entry):
    sandbox, created, launched = local_sandbox({name: "/tmp/dummy-loader"})
    previous = SimpleNamespace(delete=AsyncMock())
    sandbox._sandbox = previous
    image = SandboxImage("daytona", "legacy-checkpoint")
    with pytest.raises(RuntimeError, match=r"startup.*checkpoint"):
        if entry == "restore":
            await sandbox.restore(image)
        else:
            sandbox.start_from_snapshot(image)
    assert created == launched == []
    previous.delete.assert_not_awaited()


async def test_clean_image_keeps_activation_as_data():
    sandbox, created, _ = local_sandbox({"BASH_ENV": "/tmp/dummy-activation"})
    await sandbox._create_sandbox_without_toolbox_bash_env(
        sdk.CreateSandboxFromImageParams(image=sdk.Image.base("dummy"))
    )
    assert len(created) == 2
    assert all(created[-1].env_vars.get(name) == "" for name in SYSTEM_SHELL_EMPTY_VARS)
    assert created[-1].env_vars["_BF_IMAGE_BASH_ENV"] == "/tmp/dummy-activation"


@pytest.mark.parametrize("name", LOADERS)
async def test_checkpoint_restore_clears_every_loader_before_first_shell(name):
    sandbox, _, _ = local_sandbox({})
    await sandbox._create_sandbox_without_toolbox_bash_env(
        sdk.CreateSandboxFromImageParams(image=sdk.Image.base("dummy"))
    )
    sandbox._sandbox._experimental_create_snapshot = AsyncMock()
    sandbox._strategy._credential_ops = lambda: SimpleNamespace()
    with patch(
        "benchflow.sandbox.daytona_strategies.scrub_credentials",
        AsyncMock(return_value=[]),
    ):
        checkpoint = await sandbox.snapshot("dummy-checkpoint")
    # Model a provider checkpoint that retained stale loader defaults. The
    # controller's receipt, not those defaults, drives the first new create.
    restored, creates, starts = local_sandbox({name: "/tmp/dummy-loader"})
    await restored.restore(checkpoint)
    assert len(creates) == 1
    assert starts
    assert all(starts[0].get(key) == "" for key in SYSTEM_SHELL_EMPTY_VARS)
    assert all(creates[0].env_vars.get(key) == "" for key in SYSTEM_SHELL_EMPTY_VARS)


@pytest.mark.parametrize("entry", ["restore", "branch"])
@pytest.mark.parametrize(
    "receipt",
    [
        "broken-json",
        "null",
        "[]",
        "{}",
        json.dumps(dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS[:-1], "")),
        json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "extra": ""}),
        json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "BASH_ENV": 1}),
        json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "BASH_ENV": "\0"}),
        json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "ENV": "\0"}),
    ],
)
async def test_invalid_checkpoint_receipts_refuse_before_replacing_live_sandbox(
    entry, receipt
):
    sandbox, created, launched = local_sandbox({})
    previous = SimpleNamespace(delete=AsyncMock())
    sandbox._sandbox = previous
    image = SandboxImage("daytona", "dummy-checkpoint", meta={META: receipt})
    with pytest.raises(RuntimeError, match=r"startup.*checkpoint"):
        if entry == "restore":
            await sandbox.restore(image)
        else:
            sandbox.start_from_snapshot(image)
    assert created == launched == []
    previous.delete.assert_not_awaited()


@pytest.mark.parametrize("name", LOADERS)
async def test_checkpoint_carries_a_loader_value_as_inert_data(name):
    """DAYTONA-GAPS rule 2 (refused at 29b6ea03c): a checkpoint whose
    receipt holds a loader value starts with the name empty, the value
    under its _BF_IMAGE_ name, and is never probed with it."""
    sandbox, creates, starts = local_sandbox({name: "/tmp/stale-loader"})
    values = {**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), name: "/tmp/dummy"}
    image = SandboxImage("daytona", "dummy-checkpoint", meta={META: json.dumps(values)})
    await sandbox.restore(image)
    assert len(creates) == 1
    assert starts[0][name] == ""
    assert creates[0].env_vars[f"_BF_IMAGE_{name}"] == "/tmp/dummy"
    assert sandbox._toolbox_startup_receipt == values


@pytest.mark.parametrize("value", ["", "/tmp/dummy activation", "quote'\nsecond line"])
async def test_checkpoint_preserves_activation_as_inert_data(value):
    sandbox, creates, starts = local_sandbox({"BASH_ENV": "/tmp/stale-activation"})
    values = {**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "BASH_ENV": value}
    image = SandboxImage("daytona", "dummy-checkpoint", meta={META: json.dumps(values)})
    await sandbox.restore(image)
    assert len(creates) == 1
    assert starts[0]["BASH_ENV"] == ""
    assert creates[0].env_vars["_BF_IMAGE_BASH_ENV"] == value
    assert sandbox._toolbox_startup_receipt == values


@pytest.mark.parametrize("output", ["", "truncated\0", "\0" * 14, "\0" * 12])
async def test_malformed_provider_probe_refuses_without_echoing_values(output):
    sandbox, _, _ = local_sandbox({})
    marker = "dummy-private-value"
    sandbox._create_sandbox = AsyncMock()
    current = SimpleNamespace(
        process=SimpleNamespace(
            exec=AsyncMock(
                return_value=SimpleNamespace(exit_code=0, result=output + marker)
            )
        ),
        delete=AsyncMock(),
    )
    sandbox._sandbox = current
    with pytest.raises(RuntimeError, match=r"verify.*startup") as error:
        await sandbox._create_sandbox_without_toolbox_bash_env(
            sdk.CreateSandboxFromSnapshotParams(snapshot="trusted-template")
        )
    assert marker not in str(error.value)
    assert sandbox._sandbox is None
    current.delete.assert_awaited_once()


async def test_refused_sandbox_cleanup_failure_stops_fallbacks():
    sandbox, _, _ = local_sandbox({})
    current = SimpleNamespace(
        id="dummy", delete=AsyncMock(side_effect=RuntimeError("dummy"))
    )
    sandbox._sandbox = current
    sandbox._create_sandbox = AsyncMock()
    sandbox._toolbox_startup_env = AsyncMock(
        return_value={
            **dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""),
            "LD_PRELOAD": "/tmp/dummy",
        }
    )
    with pytest.raises(SandboxStartupError, match="delete refused"):
        await sandbox._create_sandbox_without_toolbox_bash_env(
            sdk.CreateSandboxFromSnapshotParams(snapshot="trusted-template")
        )
    assert sandbox._sandbox is current
    assert sandbox._create_outcome_unknown
    assert sandbox._toolbox_startup_receipt is None


@pytest.mark.parametrize("name", LOADERS)
@pytest.mark.parametrize(
    "route",
    ["image", "content", "recipe-fallback", "plain-fallback", "template", "branch"],
)
async def test_every_start_route_enforces_startup_admission(
    tmp_path, monkeypatch, name, route
):
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
        sandbox._install_layers = sandbox._install_layers_record = None
    if route in ("content", "recipe-fallback"):
        sandbox._strategy._content_snapshot = AsyncMock(return_value="content-snapshot")
    if route == "branch":
        # Public checkpoint admission, including its retained receipt.
        sandbox.start_from_snapshot(
            SandboxImage(
                "daytona",
                "dummy-checkpoint",
                meta={META: json.dumps(dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""))},
            )
        )
    if route == "template":
        sandbox._snapshot_template_name = "trusted-template"
        fake.snapshot = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(state=sdk.SnapshotState.ACTIVE))
        )
    original = fake.create

    async def probe(command, env=None, timeout=None):
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            env={
                **os.environ,
                name: "/tmp/dummy-loader",
                **(fake.requests[-1].env_vars or {}),
                **(env or {}),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        return SimpleNamespace(exit_code=result.returncode, result=result.stdout)

    async def create(params, timeout=None):
        if isinstance(params, sdk.CreateSandboxFromSnapshotParams):
            fake.requests.append(params)
            if route == "recipe-fallback":
                from daytona.common.errors import DaytonaNotFoundError

                # Guards PR #13: model a definitive rejection, not an unknown
                # allocation that is correctly forbidden from falling back.
                raise DaytonaNotFoundError(
                    "dummy snapshot unavailable", status_code=404
                )
            return SimpleNamespace(
                id="dummy", delete=AsyncMock(), process=SimpleNamespace(exec=probe)
            )
        result = await original(params, timeout)
        result.process = SimpleNamespace(exec=probe)
        return result

    fake.create = create

    # Replace the image-build fixture's stub with each revision's actual probe.
    sandbox.__dict__.pop("_toolbox_bash_env", None)
    sandbox.__dict__.pop("_toolbox_startup_env", None)
    await sandbox._strategy.start(force_build=False)
    if route == "branch":
        assert (fake.requests[0].env_vars or {}).get(name) == ""
    else:
        # DAYTONA-GAPS rule 2 (refused at 29b6ea03c): the sandbox kept is
        # the one created with the name empty and the value carried aside.
        kept = fake.requests[-1].env_vars or {}
        assert kept.get(name) == ""
        assert kept.get(f"_BF_IMAGE_{name}") == "/tmp/dummy-loader"
        assert sandbox._sandbox is not None

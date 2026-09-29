"""Docker restore regressions adapted from JeremyJC67's PR #1046."""

import copy
import json
import os
import uuid
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.docker import DockerSandbox, _replayed_run_args
from benchflow.sandbox.protocol import (
    ExecResult,
    SandboxImage,
    SandboxRestoreHostConfigUnavailable,
)
from benchflow.task.config import NetworkMode, SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.fixture
def sandbox(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="snapshot-contract",
        session_id="bf-snapshot-contract",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


@pytest.fixture
def container():
    return {
        "HostConfig": {
            "NetworkMode": "bf-snapshot-contract_default",
            "NanoCpus": 2_000_000_000,
            "Memory": 1_073_741_824,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
        },
        "NetworkSettings": {
            "Networks": {"bf-snapshot-contract_default": {"Aliases": ["main"]}}
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/verifier",
                "Destination": "/logs/verifier",
                "RW": True,
            },
            {
                "Type": "bind",
                "Source": "/host/fixtures",
                "Destination": "/fixtures",
                "RW": False,
            },
            {
                "Type": "volume",
                "Name": "task-data",
                "Source": "/daemon/data",
                "Destination": "/data",
                "RW": True,
            },
        ],
    }


async def test_restore_keeps_binds_limits_and_security(sandbox, container):
    """PR #1046: verifier output must remain visible after container replacement."""
    sandbox._main_container_id = AsyncMock(return_value="old")
    sandbox._docker_cli = AsyncMock(
        side_effect=lambda args, **kw: ExecResult(
            return_code=0,
            stdout=json.dumps([container]) if args[0] == "inspect" else "",
            stderr="",
        )
    )
    await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    commands = [call.args[0] for call in sandbox._docker_cli.call_args_list]
    assert [cmd[0] for cmd in commands] == ["inspect", "stop", "rm", "run"]
    run = commands[-1]
    assert "type=bind,src=/host/verifier,dst=/logs/verifier" in run
    assert "type=bind,src=/host/fixtures,dst=/fixtures,readonly" in run
    assert "type=volume,src=task-data,dst=/data" in run
    assert run[run.index("--memory") + 1] == "1073741824"
    assert run[run.index("--cpus") + 1] == "2"
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert run[run.index("--security-opt") + 1] == "no-new-privileges:true"
    assert run[run.index("--network-alias") + 1] == "main"


@pytest.mark.parametrize(
    "change",
    [
        {"Privileged": True},
        {"PortBindings": {"80/tcp": [{}]}},
        {"Devices": [{}]},
        {"NetworkMode": "host"},
        {"CpuQuota": 50000},
    ],
)
async def test_unsupported_host_config_does_not_remove_container(
    sandbox, container, change
):
    """PR #1046 selective port rejects lossy restore before destructive commands."""
    container["HostConfig"].update(change)
    sandbox._main_container_id = AsyncMock(return_value="old")
    sandbox._inspect_container = AsyncMock(return_value=container)
    sandbox._docker_cli = AsyncMock()
    with pytest.raises(SandboxRestoreHostConfigUnavailable):
        await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    sandbox._docker_cli.assert_not_awaited()


@pytest.mark.parametrize(
    "payload", ["not json", "[]", "[null]", "{}", '[{"Mounts": []}]']
)
async def test_bad_inspection_leaves_container_intact(sandbox, payload):
    """PR #1046: missing host configuration must not create a mountless replacement."""
    sandbox._main_container_id = AsyncMock(return_value="old")
    sandbox._docker_cli = AsyncMock(
        return_value=ExecResult(return_code=0, stdout=payload, stderr="")
    )
    with pytest.raises(SandboxRestoreHostConfigUnavailable):
        await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    assert [c.args[0][0] for c in sandbox._docker_cli.call_args_list] == ["inspect"]


async def test_missing_container_fails_before_any_mutation(sandbox):
    """PR #1046: inability to locate main is not permission to recreate it blindly."""
    sandbox._main_container_id = AsyncMock(return_value=None)
    sandbox._docker_cli = AsyncMock()
    with pytest.raises(SandboxRestoreHostConfigUnavailable):
        await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    sandbox._docker_cli.assert_not_awaited()


async def test_denylist_restore_does_not_silently_drop_firewall(sandbox, container):
    """PR #1046 selective port: a new network namespace loses live firewall rules."""
    sandbox.task_env_config.network_mode = NetworkMode.DENYLIST
    sandbox._main_container_id = AsyncMock(return_value="old")
    sandbox._inspect_container = AsyncMock(return_value=container)
    sandbox._docker_cli = AsyncMock()
    with pytest.raises(SandboxRestoreHostConfigUnavailable, match="DENYLIST"):
        await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    sandbox._docker_cli.assert_not_awaited()


def test_none_network_and_tmpfs_options_are_preserved(container):
    """PR #1046 port keeps network isolation and tmpfs size/mode restrictions."""
    data = copy.deepcopy(container)
    data["HostConfig"].update(
        NetworkMode="none", Tmpfs={"/scratch": "rw,noexec,size=65536"}
    )
    data["NetworkSettings"] = {"Networks": {"none": {}}}
    data["Mounts"].append({"Type": "tmpfs", "Destination": "/scratch"})
    args = _replayed_run_args(data, default_network="bf-snapshot-contract_default")
    assert args[:2] == ["--network", "none"]
    assert args[args.index("--tmpfs") + 1] == "/scratch:rw,noexec,size=65536"


# HostConfig of a real ``docker compose up`` container built from
# docker-compose-base.yaml (Docker 29.5, compose 5.5), paths anonymised.
_COMPOSE_HOST_CONFIG = {
    "Binds": ["/host/verifier:/logs/verifier:rw"],
    "ContainerIDFile": "",
    "LogConfig": {"Type": "json-file", "Config": {}},
    "NetworkMode": "bf-snapshot-contract_default",
    "PortBindings": {},
    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
    "AutoRemove": False,
    "VolumeDriver": "",
    "VolumesFrom": None,
    "ConsoleSize": [0, 0],
    "CapAdd": None,
    "CapDrop": None,
    "CgroupnsMode": "private",
    "Dns": None,
    "DnsOptions": None,
    "DnsSearch": None,
    "ExtraHosts": [],
    "GroupAdd": None,
    "IpcMode": "private",
    "Cgroup": "",
    "Links": None,
    "OomScoreAdj": 0,
    "PidMode": "",
    "Privileged": False,
    "PublishAllPorts": False,
    "ReadonlyRootfs": False,
    "SecurityOpt": None,
    "UTSMode": "",
    "UsernsMode": "",
    "ShmSize": 67108864,
    "Runtime": "runc",
    "Isolation": "",
    "CpuShares": 0,
    "Memory": 268435456,
    "NanoCpus": 1000000000,
    "CgroupParent": "",
    "BlkioWeight": 0,
    "BlkioWeightDevice": None,
    "BlkioDeviceReadBps": None,
    "BlkioDeviceWriteBps": None,
    "BlkioDeviceReadIOps": None,
    "BlkioDeviceWriteIOps": None,
    "CpuPeriod": 0,
    "CpuQuota": 0,
    "CpuRealtimePeriod": 0,
    "CpuRealtimeRuntime": 0,
    "CpusetCpus": "",
    "CpusetMems": "",
    "Devices": None,
    "DeviceCgroupRules": None,
    "DeviceRequests": None,
    "MemoryReservation": 0,
    "MemorySwap": 536870912,
    "MemorySwappiness": None,
    "OomKillDisable": None,
    "PidsLimit": None,
    "Ulimits": None,
    "CpuCount": 0,
    "CpuPercent": 0,
    "IOMaximumIOps": 0,
    "IOMaximumBandwidth": 0,
    "MaskedPaths": ["/proc/acpi", "/proc/kcore"],
    "ReadonlyPaths": ["/proc/bus", "/proc/sys"],
}


def _flag_values(args, flag):
    return [args[i + 1] for i, arg in enumerate(args) if arg == flag]


def test_real_compose_host_config_replays_runtime_and_namespaces(container):
    """Guards the allowlist replacing the first snapshot-restore change's denylist on a real compose container."""
    data = copy.deepcopy(container)
    data["HostConfig"] = copy.deepcopy(_COMPOSE_HOST_CONFIG)
    args = _replayed_run_args(data, default_network="bf-snapshot-contract_default")
    assert _flag_values(args, "--runtime") == ["runc"]
    assert _flag_values(args, "--cgroupns") == ["private"]
    assert _flag_values(args, "--log-driver") == ["json-file"]
    assert _flag_values(args, "--memory-swap") == ["536870912"]
    assert "--init" not in args
    assert "--group-add" not in args


def test_restore_replays_gvisor_runtime_init_and_groups(container):
    """Guards the first snapshot-restore change, which restored a runsc (gVisor) container onto runc."""
    data = copy.deepcopy(container)
    data["HostConfig"].update(
        Runtime="runsc",
        Init=True,
        GroupAdd=["audio", "1001"],
        OomScoreAdj=500,
        MemorySwappiness=0,
        LogConfig={"Type": "local", "Config": {"max-size": "10m"}},
    )
    args = _replayed_run_args(data, default_network="bf-snapshot-contract_default")
    assert _flag_values(args, "--runtime") == ["runsc"]
    assert "--init" in args
    assert _flag_values(args, "--group-add") == ["audio", "1001"]
    assert _flag_values(args, "--oom-score-adj") == ["500"]
    assert _flag_values(args, "--memory-swappiness") == ["0"]
    assert _flag_values(args, "--log-driver") == ["local"]
    assert _flag_values(args, "--log-opt") == ["max-size=10m"]


def test_host_config_tmpfs_is_replayed_without_a_mounts_entry(container):
    """Guards the first snapshot-restore change, which replayed tmpfs only from the container Mounts list.

    Docker lists ``--tmpfs``/compose ``tmpfs:`` mounts in ``HostConfig.Tmpfs``
    only, never in the container's ``Mounts``.
    """
    data = copy.deepcopy(container)
    data["HostConfig"]["Tmpfs"] = {"/scratch": "size=65536", "/run": ""}
    args = _replayed_run_args(data, default_network="bf-snapshot-contract_default")
    assert sorted(_flag_values(args, "--tmpfs")) == ["/run", "/scratch:size=65536"]


def test_explicitly_disabled_init_stays_disabled(container):
    """Guards the first snapshot-restore change: an explicit ``Init: false`` must not inherit a daemon default."""
    data = copy.deepcopy(container)
    data["HostConfig"]["Init"] = False
    args = _replayed_run_args(data, default_network="bf-snapshot-contract_default")
    assert "--init=false" in args


@pytest.mark.parametrize(
    "change",
    [
        {"UTSMode": "host"},
        {"Isolation": "hyperv"},
        {"AutoRemove": True},
        {"VolumeDriver": "custom"},
        {"OomKillDisable": True},
        {"Ulimits": [{"Name": "nofile", "Soft": 1024, "Hard": 1024}]},
        {"KernelMemoryTCP": 1024},
        {"FutureSandboxKnob": "strict"},
        {"Mounts": [{"Type": "volume", "VolumeOptions": {"Subpath": "one"}}]},
    ],
    ids=lambda change: next(iter(change)),
)
def test_unhandled_host_setting_rejects_restore(container, change):
    """Guards the first snapshot-restore change's denylist, which silently dropped unlisted host settings."""
    data = copy.deepcopy(container)
    data["HostConfig"].update(change)
    with pytest.raises(SandboxRestoreHostConfigUnavailable, match="container left"):
        _replayed_run_args(data, default_network="bf-snapshot-contract_default")


def test_unset_unknown_host_setting_is_accepted(container):
    """Guards the allowlist from the first snapshot-restore change's follow-up against rejecting empty keys."""
    data = copy.deepcopy(container)
    data["HostConfig"].update(
        FutureSandboxKnob=None,
        Annotations={},
        Mounts=[{"Type": "bind", "BindOptions": {"Propagation": "rprivate"}}],
    )
    _replayed_run_args(data, default_network="bf-snapshot-contract_default")


async def test_restore_keeps_benchflow_owned_label(sandbox, container):
    """Guards the ``benchflow.owned`` label the first snapshot-restore change did not replay on restore."""
    container["Config"] = {"Labels": {"benchflow.owned": "true"}}
    sandbox._main_container_id = AsyncMock(return_value="old")
    sandbox._inspect_container = AsyncMock(return_value=container)
    sandbox._docker_cli = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="", stderr="")
    )
    await sandbox.restore(SandboxImage(provider="docker", ref="snapshot"))
    run = sandbox._docker_cli.call_args_list[-1].args[0]
    assert run[0] == "run"
    assert "benchflow.owned=true" in _flag_values(run, "--label")


@pytest.mark.live
async def test_local_docker_snapshot_retains_mounts_and_rolls_back_files(
    sandbox, tmp_path
):
    """PR #1046 live proof, using only uniquely labelled disposable local containers."""
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 for the local Docker proof")
    project = "bf-snapshot-proof-" + uuid.uuid4().hex[:12]
    sandbox.session_id = project
    sandbox.environment_name = project
    host_dir = tmp_path / "mounted"
    host_dir.mkdir()
    image = None

    async def current_container():
        result = await sandbox._docker_cli(
            ["ps", "-q", "--filter", f"label=com.docker.compose.project={project}"]
        )
        return (result.stdout or "").strip() or None

    sandbox._main_container_id = current_container
    try:
        await sandbox._docker_cli(
            [
                "run",
                "--pull=never",
                "-d",
                "--network",
                "none",
                "--label",
                f"com.docker.compose.project={project}",
                "--label",
                "com.docker.compose.service=main",
                "--mount",
                f"type=bind,src={host_dir},dst=/proof-output",
                "--cpus",
                "0.5",
                "--memory",
                "128m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "python:3.12-slim",
                "sleep",
                "infinity",
            ]
        )
        original = await current_container()
        await sandbox._docker_cli(
            ["exec", original, "sh", "-c", "echo before > /state"]
        )
        image = await sandbox.snapshot()
        await sandbox._docker_cli(
            ["exec", original, "sh", "-c", "echo changed > /state"]
        )
        assert (
            await sandbox._docker_cli(["exec", original, "cat", "/state"])
        ).stdout.strip() == "changed"
        await sandbox.restore(image)
        restored = await current_container()
        assert restored != original
        assert (
            await sandbox._docker_cli(["exec", restored, "cat", "/state"])
        ).stdout.strip() == "before"
        await sandbox._docker_cli(
            ["exec", restored, "sh", "-c", "echo visible > /proof-output/result"]
        )
        assert (host_dir / "result").read_text().strip() == "visible"
        host = (await sandbox._inspect_container(restored))["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["NanoCpus"] == 500_000_000
        assert host["Memory"] == 128 * 1024 * 1024
        assert "ALL" in host["CapDrop"]
        assert "no-new-privileges:true" in host["SecurityOpt"]
    finally:
        # Select only containers carrying this proof's unique project label.
        result = await sandbox._docker_cli(
            ["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]
        )
        for container_id in (result.stdout or "").split():
            await sandbox._docker_cli(["rm", "-f", container_id])
        if image is not None:
            await sandbox._docker_cli(["image", "rm", image.ref])


@pytest.mark.live
async def test_local_docker_restore_keeps_runtime_init_groups_and_owner(sandbox):
    """Live proof for the allowlist replacing the first snapshot-restore change's denylist.

    Uses only a uniquely labelled disposable container and snapshot image.
    """
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 for the local Docker proof")
    project = "bf-snapshot-proof-" + uuid.uuid4().hex[:12]
    sandbox.session_id = project
    sandbox.environment_name = project
    image = None

    async def current_container():
        result = await sandbox._docker_cli(
            ["ps", "-q", "--filter", f"label=com.docker.compose.project={project}"]
        )
        return (result.stdout or "").strip() or None

    sandbox._main_container_id = current_container
    try:
        await sandbox._docker_cli(
            [
                "run",
                "--pull=never",
                "-d",
                "--network",
                "none",
                "--label",
                f"com.docker.compose.project={project}",
                "--label",
                "com.docker.compose.service=main",
                "--label",
                "benchflow.owned=true",
                "--runtime",
                "runc",
                "--init",
                "--group-add",
                "4242",
                "--oom-score-adj",
                "321",
                "--tmpfs",
                "/scratch:size=65536",
                "python:3.12-slim",
                "sleep",
                "infinity",
            ]
        )
        original = await current_container()
        image = await sandbox.snapshot()
        await sandbox.restore(image)
        restored = await current_container()
        assert restored not in (None, original)
        inspected = await sandbox._inspect_container(restored)
        host = inspected["HostConfig"]
        assert host["Runtime"] == "runc"
        assert host.get("Init") is True
        assert host.get("GroupAdd") == ["4242"]
        assert host["OomScoreAdj"] == 321
        assert host.get("Tmpfs") == {"/scratch": "size=65536"}
        assert inspected["Config"]["Labels"]["benchflow.owned"] == "true"
        owned = await sandbox._docker_cli(
            ["ps", "-q", "--filter", "label=benchflow.owned=true"]
        )
        assert restored[:12] in (owned.stdout or "")
    finally:
        # Select only containers carrying this proof's unique project label.
        result = await sandbox._docker_cli(
            ["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]
        )
        for container_id in (result.stdout or "").split():
            await sandbox._docker_cli(["rm", "-f", container_id])
        if image is not None:
            await sandbox._docker_cli(["image", "rm", image.ref])


@pytest.mark.live
async def test_local_docker_snapshot_keeps_credentials_out_and_is_deleted(sandbox):
    """Live check that snapshot credentials stay out of the image.

    The committed image holds no agent credential file, the live and the
    restored container both have it back with owner and mode, and the image
    is removed once no container uses it. Uses a uniquely labelled disposable
    container and a random sentinel, never a real credential.
    """
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 for the local Docker proof")
    project = "bf-snapshot-proof-" + uuid.uuid4().hex[:12]
    sandbox.session_id = project
    sandbox.environment_name = project
    sentinel = "bf-scrub-sentinel-" + uuid.uuid4().hex
    image = None
    files = "/root/.codex/auth.json /home/agent/.claude/.credentials.json"

    async def current_container():
        result = await sandbox._docker_cli(
            ["ps", "-q", "--filter", f"label=com.docker.compose.project={project}"]
        )
        return (result.stdout or "").strip() or None

    async def owners_and_content(container):
        result = await sandbox._docker_cli(
            [
                "exec",
                container,
                "sh",
                "-c",
                f"stat -c '%u:%g:%a %n' {files}; cat {files}",
            ]
        )
        return result.stdout

    sandbox._main_container_id = current_container
    try:
        await sandbox._docker_cli(
            [
                "run",
                "--pull=never",
                "-d",
                "--network",
                "none",
                "--label",
                f"com.docker.compose.project={project}",
                "--label",
                "com.docker.compose.service=main",
                "python:3.12-slim",
                "sleep",
                "infinity",
            ]
        )
        original = await current_container()
        await sandbox._docker_cli(
            [
                "exec",
                original,
                "sh",
                "-c",
                "mkdir -p /root/.codex /home/agent/.claude && "
                f"printf %s {sentinel} > /root/.codex/auth.json && "
                f"printf %s {sentinel} > /home/agent/.claude/.credentials.json && "
                "chown -R 1000:1000 /home/agent && chmod 600 " + files,
            ]
        )
        image = await sandbox.snapshot()

        scan = await sandbox._docker_cli(
            [
                "run",
                "--rm",
                "--network",
                "none",
                image.ref,
                "sh",
                "-c",
                f"grep -rl {sentinel} /root /home /tmp /etc 2>/dev/null; "
                "ls -a /root/.codex /home/agent/.claude",
            ],
            check=False,
        )
        assert "auth.json" not in scan.stdout
        assert ".credentials.json" not in scan.stdout

        expected = (
            "0:0:600 /root/.codex/auth.json\n"
            "1000:1000:600 /home/agent/.claude/.credentials.json\n"
            f"{sentinel}{sentinel}"
        )
        assert await owners_and_content(original) == expected
        await sandbox.restore(image)
        restored = await current_container()
        assert restored not in (None, original)
        assert await owners_and_content(restored) == expected

        # The restored container runs from the image, so deletion waits for it.
        assert await sandbox.delete_snapshot(image) is False
        await sandbox._docker_cli(["rm", "-f", restored])
        await sandbox._delete_deferred_snapshots()
        left = await sandbox._docker_cli(["image", "ls", "-q", image.ref])
        assert (left.stdout or "").strip() == ""
        image = None
    finally:
        # Select only containers carrying this proof's unique project label.
        result = await sandbox._docker_cli(
            ["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]
        )
        for container_id in (result.stdout or "").split():
            await sandbox._docker_cli(["rm", "-f", container_id])
        if image is not None:
            await sandbox._docker_cli(["image", "rm", image.ref], check=False)

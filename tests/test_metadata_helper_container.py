"""The cloud metadata helper, and main without NET_ADMIN, on a real Docker daemon.

The fakes in test_metadata_helper.py check the commands; these check what a
real daemon does with them: the helper's rules land in main's
network namespace and its count reads them, the container runs the image this
process built and nothing else, a helper gone before the block is replaced,
and a sandbox whose block the helper serves starts main with no NET_ADMIN for
root or anyone else. They skip where no Docker daemon answers (the
gate's container has none) and run on a builder's Docker VM. Everything they
start carries the label below and is removed at the end.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._docker_sweep import release_project
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import (
    _cloud_metadata_after_cmd,
    _cloud_metadata_block_cmd,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests._metadata_mock import start_mock, untag_recorded_helpers

MOCK_LABEL = "bf.test.mdhelper-mock=1"
LABEL_KEY = "benchflow.test"
LABEL_VALUE = "metadata-helper-container"
# CAP_NET_ADMIN's bit in the capability sets of /proc/<pid>/status.
NET_ADMIN = 1 << 12


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

_STATE_DIRS: list[str] = []


@pytest.fixture(autouse=True)
def _note_state_dir(monkeypatch):
    """Guards 88ac33cde: cleanup must not remove another worker's main.

    Note the per-test helper images, and give sweeps and mock endpoints
    labels owned by this test instead of the whole test module.
    """
    import os

    owner = uuid.uuid4().hex
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "LABEL_VALUE", f"metadata-helper-container-{owner}")
    monkeypatch.setattr(module, "MOCK_LABEL", f"bf.test.mdhelper-mock={owner}")
    _STATE_DIRS.append(os.environ[helper_mod.STATE_DIR_ENV])


@pytest.fixture(autouse=True, scope="module")
def _untag_helper_images():
    """After the module, remove the helper images its tests built and kept."""
    yield
    untag_recorded_helpers(_STATE_DIRS)


async def _docker(
    args: list[str], *, timeout: float, stdin: bytes | None = None
) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.PIPE
        if stdin is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    return ExecResult(
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        return_code=proc.returncode or 0,
    )


def _inspect(container: str, template: str) -> str:
    return subprocess.run(
        ["docker", "inspect", "--format", template, container],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _remove_mocks() -> None:
    """Remove this suite's metadata mocks (by their label)."""
    left = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={MOCK_LABEL}"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.split()
    if left:
        subprocess.run(["docker", "rm", "--force", *left], capture_output=True)


def _caps(listed: str) -> set[str]:
    """A capability list as docker inspect prints it, with or without CAP_."""
    return {c.removeprefix("CAP_") for c in listed.strip("[]").split()}


def _exists(container: str) -> bool:
    return (
        subprocess.run(
            ["docker", "inspect", container], capture_output=True, timeout=30
        ).returncode
        == 0
    )


@pytest.fixture
def main_container():
    """A throwaway main: the pinned base, asleep, with this suite's label."""
    name = f"bf-test-mdmain-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            f"--label={LABEL_KEY}={LABEL_VALUE}",
            helper_mod.HELPER_BASE,
            "sleep",
            "600",
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    try:
        yield name
    finally:
        left = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={LABEL_KEY}={LABEL_VALUE}"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.split()
        if left:
            subprocess.run(
                ["docker", "rm", "--force", *left], capture_output=True, timeout=60
            )


def _helper() -> helper_mod.MetadataHelper:
    return helper_mod.MetadataHelper(
        _docker,
        daemon="test-local",
        name=f"bf-test-mdhelper-{uuid.uuid4().hex[:8]}",
        labels={LABEL_KEY: LABEL_VALUE},
    )


@pytest.mark.asyncio
async def test_the_helper_adds_the_rules_in_mains_namespace_and_goes(main_container):
    command = _cloud_metadata_block_cmd()
    helper = _helper()

    helper.install_early(main_container, command)
    installed = await helper.installed(main_container, command)

    assert installed is not None, helper.unavailable
    result, failed = installed
    assert failed == "" and "BF metadata status=0" in result.stdout
    # The rule's counter, read in main's namespace through the helper.
    count, why = await helper.run(
        _cloud_metadata_after_cmd("iptables", "BFMDtest"), timeout_sec=60
    )
    assert why == "" and re.search(r"BF metadata-hits \d+", count.stdout), count
    container = helper.container
    assert container is not None
    # The image this process built, with no build cache (D937 condition 3),
    # and only the firewall power.
    assert _inspect(container, "{{.Image}}") == helper_mod._READY["test-local"]
    assert _caps(_inspect(container, "{{.HostConfig.CapAdd}}")) == {
        "NET_ADMIN",
        "NET_RAW",
    }
    assert _caps(_inspect(container, "{{.HostConfig.CapDrop}}")) == {"ALL"}
    assert _inspect(container, "{{.HostConfig.ReadonlyRootfs}}") == "true"
    assert _inspect(container, "{{.HostConfig.IpcMode}}") == "none"
    assert _inspect(container, "{{.HostConfig.NetworkMode}}").startswith("container:")

    await helper.aclose()
    for _ in range(50):
        if not _exists(container):
            break
        await asyncio.sleep(0.1)
    assert not _exists(container)


@pytest.mark.asyncio
async def test_a_helper_gone_before_the_block_is_replaced(main_container):
    command = _cloud_metadata_block_cmd()
    helper = _helper()
    helper.install_early(main_container, command)
    assert await helper.installed(main_container, command) is not None, (
        helper.unavailable
    )
    first = helper.container
    assert first is not None
    subprocess.run(
        ["docker", "rm", "--force", first], check=True, capture_output=True, timeout=60
    )

    installed = await helper.installed(main_container, command)

    assert installed is not None, helper.unavailable
    assert helper.container not in (None, first)
    count, why = await helper.run(
        _cloud_metadata_after_cmd("iptables", "BFMDtest"), timeout_sec=60
    )
    assert why == "" and "BF metadata-hits" in count.stdout
    await helper.aclose()


# A main image BenchFlow's bind-identity check accepts (coreutils'
# /usr/bin/env and /usr/bin/stat; BusyBox's Alpine has neither there).
MAIN_IMAGE = "ubuntu:24.04"


# main's network: internal, so nothing in main reaches anything outside the
# host, a real metadata server included; the block's test connects reach the
# mock (tests/_metadata_mock.py) on main's own loopback.
INTERNAL_NETWORK = {"networks": {"default": {"internal": True}}}


def _sandbox(tmp_path: Path, main: dict | None = None) -> DockerSandbox:
    environment = tmp_path / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(f"FROM {MAIN_IMAGE}\n")
    (environment / "docker-compose.yaml").write_text(
        json.dumps({"services": {"main": main or {}}, **INTERNAL_NETWORK})
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    name = f"bf-test-mdsandbox-{uuid.uuid4().hex[:8]}"
    return DockerSandbox(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=MAIN_IMAGE, cpus=1, memory_mb=256),
    )


async def _bounding_set(sandbox: DockerSandbox, user: str) -> int:
    result = await sandbox.exec("grep CapBnd /proc/self/status", user=user)
    found = re.search(r"CapBnd:\s*([0-9a-f]+)", result.stdout or "")
    assert found, result
    return int(found[1], 16)


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_a_sandbox_whose_block_the_helper_serves_starts_main_without_net_admin(
    tmp_path, monkeypatch, keep
):
    """On a real daemon: no NET_ADMIN in main's bounding
    set, for root or the agent's kind of user, so nothing in main can change
    the firewall; the firewall-power check is clear at root's read, and the
    helper's rules still prove the block on. ``keep`` is the switch that
    gives main its NET_ADMIN back, to show the checks can tell."""
    from benchflow.sandbox.firewall_power import check_firewall_power
    from benchflow.sandbox.lockdown import block_cloud_metadata

    if keep:
        monkeypatch.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
    sandbox = _sandbox(tmp_path)
    sandbox.task_env_config.arm_cloud_metadata_block()
    # This process's helper image, built before the start (in a trial its
    # build runs beside the start; main drops NET_ADMIN once it is ready).
    daemon = " ".join(sandbox._docker_compose_global_args())
    await helper_mod.ensure_image(sandbox._helper_docker, daemon)
    try:
        await sandbox.start(force_build=False)
        start_mock(await sandbox._main_container_id(), MOCK_LABEL)
        for user in ("root", "nobody"):
            assert bool(await _bounding_set(sandbox, user) & NET_ADMIN) is keep, user
        assert getattr(sandbox.task_env_config, "cloud_metadata_by_helper", False) is (
            not keep
        )
        # The image's own user, as the agent (uid 1000).
        power, refusal = await check_firewall_power(
            sandbox, "ubuntu", holds=("the cloud metadata block",)
        )
        assert refusal is None, power
        assert power["state"] == ("disarmed" if keep else "clear"), power
        record = await block_cloud_metadata(sandbox, "ubuntu")
        assert record["state"] == "on", record
        assert record["rules_from"] == "helper", record
        assert record["main_net_admin"] is keep, record
    finally:
        try:
            await sandbox.stop(delete=True)
        finally:
            _remove_mocks()
            release_project(sandbox.compose_project_name)


# Main's PID 1 for the restart test: it exits once /tmp/bf-exit exists, and
# at once on any restart, since the file stays in the container's layer.
EXITING_MAIN = ["sh", "-c", "while [ ! -e /tmp/bf-exit ]; do sleep 0.2; done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["ubuntu", None], ids=["agent-user", "root-agent"])
async def test_a_main_with_restart_always_never_restarts_behind_the_block(
    tmp_path, user
):
    """The task gives main ``restart: always``. A restarted main gets a new
    network namespace without the block's rules, so wherever the block
    says on, main's restart policy must be ``no``: when its PID 1 exits,
    main stays stopped and never runs again without the rules. On 55c66cf4
    the block said on and main came back (RestartCount 1) in a namespace
    with no rule."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    sandbox = _sandbox(tmp_path, {"restart": "always", "command": EXITING_MAIN})
    sandbox.task_env_config.arm_cloud_metadata_block(root_agent=user is None)
    daemon = " ".join(sandbox._docker_compose_global_args())
    await helper_mod.ensure_image(sandbox._helper_docker, daemon)
    try:
        await sandbox.start(force_build=False)
        main = await sandbox._main_container_id()
        assert main
        start_mock(main, MOCK_LABEL)
        record = await block_cloud_metadata(sandbox, user)
        policy = _inspect(main, "{{.HostConfig.RestartPolicy.Name}}")
        if record["state"] == "on":
            assert policy in ("", "no"), (policy, record)
        else:
            assert "restart" in record["why"], record
        # PID 1 exits.
        subprocess.run(
            ["docker", "exec", "--user", "0", main, "touch", "/tmp/bf-exit"],
            capture_output=True,
            timeout=30,
        )
        for _ in range(100):
            if _inspect(main, "{{.State.Running}}") == "false":
                break
            await asyncio.sleep(0.1)
        # Docker's first restart comes about 0.1 s after the exit.
        await asyncio.sleep(3)
        restarts = _inspect(main, "{{.RestartCount}}")
        running = _inspect(main, "{{.State.Running}}")
        assert (restarts, running) == ("0", "false"), (record["state"], restarts)
        assert record["state"] == "on", record
    finally:
        try:
            await sandbox.stop(delete=True)
        finally:
            _remove_mocks()
            release_project(sandbox.compose_project_name)

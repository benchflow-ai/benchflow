"""A checkpoint restored into a new sandbox is proved against its own log binds.

Deterministic tier (``tests/integration/README.md``): real Docker, no model.
An isolated branch child and a retry from a checkpoint start their own
sandbox (their own trial folders bound at ``/logs/...``) and then restore the
parent's checkpoint into it. The bind proof compared the replacement's log
binds with the identities in the checkpoint's label, which are the parent's
folders, so every such restore refused with "Container bind identity changed
since the first start" (DET-R2). The child's log binds are now expected at
the identities its own first start recorded; a bind carried over from the
parent keeps the parent's. A log folder swapped after the child's first start
still refuses, and so does one swapped under an in-place restore.

DET-R2b: a Compose task's bind of its own folder (``./data:/data``) comes
from each sandbox's private task copy, so it too is expected at the child's
first-start identity, and refuses when swapped after it; an operator bind the
parent and child share keeps the parent's identity, swapped or not; a log
folder re-pointed at the parent's or a sibling's refuses.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _bind_targets
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
# ubuntu:24.04 is the deterministic tier's base image.
DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
REFUSAL = "Container bind identity changed since the first start"


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


def _sandbox(root: Path, name: str) -> DockerSandbox:
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (environment / "Dockerfile").write_text(DOCKERFILE)
    paths = RolloutPaths(rollout_dir=root / name)
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="restore-binds",
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )


def _swap(sandbox: DockerSandbox, name: str) -> None:
    """Put another folder (a new inode) at the host path of the sandbox's
    ``name`` log folder."""
    folder = Path(sandbox.rollout_paths.rollout_dir) / name
    folder.rename(folder.with_name(name + "-moved"))
    folder.mkdir()


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace").strip()


async def _no_leftovers(*sandboxes: DockerSandbox) -> None:
    for sandbox in sandboxes:
        code, leftovers = await _docker(
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={sandbox.compose_project_name}",
        )
        assert (code, leftovers) == (0, ""), "containers left behind"


@pytest.mark.usefixtures("require_docker")
async def test_a_checkpoint_restores_into_a_new_sandbox_with_its_own_log_binds(
    tmp_path,
):
    """An isolated child: the parent's checkpoint, the child's log folders."""
    parent = _sandbox(tmp_path, "parent")
    child = _sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        await parent.exec("echo checkpoint > /app/state", user="root")
        image = await parent.snapshot()
        await child.start(force_build=False)
        await child.restore(image)
        state = await child.exec("cat /app/state", user="root")
        assert (state.return_code, state.stdout.strip()) == (0, "checkpoint")
        wrote = await child.exec("echo child > /logs/agent/mark.txt", user="root")
        assert wrote.return_code == 0, wrote.stderr
        child_dir = child.rollout_paths.agent_dir
        assert (child_dir / "mark.txt").read_text().strip() == "child"
        assert not (parent.rollout_paths.agent_dir / "mark.txt").exists()
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.usefixtures("require_docker")
async def test_a_child_whose_log_bind_is_swapped_after_its_first_start_refuses(
    tmp_path,
):
    """The child's own first-start identity is what its log bind must match:
    another folder at the same host path refuses the restore."""
    parent = _sandbox(tmp_path, "parent")
    child = _sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        await child.start(force_build=False)
        _swap(child, "agent")
        with pytest.raises(RuntimeError, match=REFUSAL):
            await child.restore(image)
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.usefixtures("require_docker")
async def test_an_in_place_restore_with_a_swapped_log_bind_still_refuses(tmp_path):
    """The in-place check is unchanged: its own receipt is the checkpoint's."""
    sandbox = _sandbox(tmp_path, "trial")
    image = None
    try:
        await sandbox.start(force_build=False)
        image = await sandbox.snapshot()
        _swap(sandbox, "verifier")
        with pytest.raises(RuntimeError, match=REFUSAL):
            await sandbox.restore(image)
    finally:
        await sandbox.stop(delete=True)
        if image is not None:
            await sandbox.delete_snapshot(image)
    await _no_leftovers(sandbox)


# --- DET-R2b: the reviewers' cases (M1/L1; P2, P5, P7) ---------------------

COMPOSE = (
    "services:\n  main:\n    build: .\n    command: [sleep, infinity]\n"
    "    volumes:\n      - ./data:/data\n"
)


def _compose_sandbox(root: Path, name: str) -> DockerSandbox:
    """A Compose task that binds a folder of its own (`./data:/data`):
    each sandbox binds its own private copy of it. The task has its own
    folder: the trial folders are not inside it, as in a real run."""
    environment = root / "task" / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (environment / "Dockerfile").write_text(DOCKERFILE)
    (environment / "docker-compose.yaml").write_text(COMPOSE)
    (environment / "data").mkdir(exist_ok=True)
    (environment / "data" / "f.txt").write_text("task\n")
    paths = RolloutPaths(rollout_dir=root / name)
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="restore-binds-compose",
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )


def _mount_sandbox(root: Path, name: str, source: Path, target: str) -> DockerSandbox:
    """A sandbox with an operator bind (`--mount`) the parent and child share."""
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (environment / "Dockerfile").write_text(DOCKERFILE)
    paths = RolloutPaths(rollout_dir=root / name)
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="restore-binds",
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
        mounts_json=[{"type": "bind", "source": str(source), "target": target}],
    )


def _swap_path(folder: Path) -> None:
    folder.rename(folder.with_name(folder.name + "-moved"))
    folder.mkdir()


@pytest.mark.usefixtures("require_docker")
async def test_a_compose_task_binding_its_own_folder_restores_into_a_child(tmp_path):
    """A's M1: `./data` is bound from each sandbox's private task copy, so
    the child's /data is its own copy, proved against its own first start."""
    parent = _compose_sandbox(tmp_path, "parent")
    child = _compose_sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        await child.start(force_build=False)
        await child.restore(image)
        assert child._staged_bind_targets == {"main": frozenset({"/data"})}
        read = await child.exec("cat /data/f.txt", user="root")
        assert (read.return_code, read.stdout.strip()) == (0, "task")
        wrote = await child.exec("echo child > /data/mark.txt", user="root")
        assert wrote.return_code == 0, wrote.stderr
        assert (child._stage().environment_dir / "data" / "mark.txt").exists()
        assert not (parent._stage().environment_dir / "data" / "mark.txt").exists()
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.usefixtures("require_docker")
async def test_a_childs_own_task_copy_bind_swapped_after_its_first_start_refuses(
    tmp_path,
):
    """The staged bind takes the child's first-start identity, not whatever
    sits at that host path later: another folder there refuses."""
    parent = _compose_sandbox(tmp_path, "parent")
    child = _compose_sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        await child.start(force_build=False)
        _swap_path(Path(child._stage().environment_dir) / "data")
        with pytest.raises(RuntimeError, match=REFUSAL):
            await child.restore(image)
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.usefixtures("require_docker")
async def test_a_childs_non_log_bind_keeps_the_parents_identity(tmp_path, swap):
    """A's L1: an operator bind shared by parent and child is carried over and
    proved against the checkpoint's label; swapped between the parent's
    checkpoint and the child's first start, the child's own receipt does not
    launder it (B's P5)."""
    shared = tmp_path / "shared"
    shared.mkdir()
    parent = _mount_sandbox(tmp_path, "parent", shared, "/shared")
    child = _mount_sandbox(tmp_path, "child", shared, "/shared")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        label = await parent._snapshot_image_bind_baseline(image.ref)
        assert "/shared" in label["main"]
        assert set(label["main"]) >= _bind_targets.LOG_TARGETS
        if swap:
            _swap_path(shared)
        await child.start(force_build=False)
        if swap:
            with pytest.raises(RuntimeError, match=REFUSAL):
                await child.restore(image)
        else:
            await child.restore(image)
            assert child._bind_baseline["main"]["/shared"] == label["main"]["/shared"]
            for target in _bind_targets.LOG_TARGETS:
                assert child._bind_baseline["main"][target] != label["main"][target]
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.usefixtures("require_docker")
async def test_a_childs_log_folder_repointed_at_the_parents_refuses(tmp_path):
    """B's P2: the child's agent folder made a link to the parent's."""
    parent = _sandbox(tmp_path, "parent")
    child = _sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        await child.start(force_build=False)
        folder = Path(child.rollout_paths.rollout_dir) / "agent"
        folder.rename(folder.with_name("agent-moved"))
        os.symlink(str(Path(parent.rollout_paths.rollout_dir) / "agent"), folder)
        with pytest.raises(RuntimeError, match=REFUSAL):
            await child.restore(image)
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


@pytest.mark.usefixtures("require_docker")
async def test_two_children_in_parallel_one_repointed_at_the_others_folder(tmp_path):
    """B's P7: child B's agent folder made a link to child A's; restored at
    once, A passes and B refuses."""
    parent = _sandbox(tmp_path, "parent")
    a, b = _sandbox(tmp_path, "child-a"), _sandbox(tmp_path, "child-b")
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        await asyncio.gather(a.start(force_build=False), b.start(force_build=False))
        folder = Path(b.rollout_paths.rollout_dir) / "agent"
        folder.rename(folder.with_name("agent-moved"))
        os.symlink(str(Path(a.rollout_paths.rollout_dir) / "agent"), folder)
        done_a, done_b = await asyncio.gather(
            a.restore(image), b.restore(image), return_exceptions=True
        )
        assert done_a is None, done_a
        assert isinstance(done_b, RuntimeError) and REFUSAL in str(done_b), done_b
    finally:
        for sandbox in (b, a, parent):
            await sandbox.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, a, b)


def test_only_log_and_staged_targets_take_the_sandboxs_own_identity():
    """A's unit check, plus DET-R2b's staged targets: exact targets only,
    per service; a target `own` lacks keeps the carried identity."""
    carried = {
        "main": {"/logs/agent": "1:1", "/logs/verifier": "1:2", "/data": "1:3"},
        "side": {"/logs/agent": "1:4", "/data": "1:5"},
    }
    own = {
        "main": {"/logs/agent": "2:1", "/data": "2:3", "/logs/verifier/x": "2:9"},
        "side": {"/logs/agent": "2:4", "/data": "2:5"},
        "other": {"/logs/agent": "2:5"},
    }
    assert _bind_targets.with_own_log_binds(carried, own) == {
        "main": {"/logs/agent": "2:1", "/logs/verifier": "1:2", "/data": "1:3"},
        "side": {"/logs/agent": "2:4", "/data": "1:5"},
    }
    assert _bind_targets.with_own_log_binds(carried, own, {"main": {"/data"}}) == {
        "main": {"/logs/agent": "2:1", "/logs/verifier": "1:2", "/data": "2:3"},
        "side": {"/logs/agent": "2:4", "/data": "1:5"},
    }
    assert _bind_targets.with_own_log_binds(carried, None) is carried
    assert {"/logs/agent", "/logs/verifier", "/logs/artifacts"} == (
        _bind_targets.LOG_TARGETS
    )


def _identity(path: Path) -> str:
    st = os.stat(path)
    return f"{st.st_dev}:{st.st_ino}"


def test_staged_targets_are_binds_from_inside_the_task_copy(tmp_path):
    task = tmp_path / "stage" / "task"
    for folder in (
        task / "environment" / "data",
        tmp_path / "stage" / "task-other" / "d",
        tmp_path / "shared",
        task / "v",
    ):
        folder.mkdir(parents=True)
    data = task / "environment" / "data"
    near = tmp_path / "stage" / "task-other" / "d"
    inspected = {
        "Mounts": [
            {"Type": "bind", "Source": str(data), "Destination": "/data"},
            {"Type": "bind", "Source": str(near), "Destination": "/near"},
            {"Type": "bind", "Source": str(tmp_path / "shared"),
             "Destination": "/shared"},
            {"Type": "volume", "Source": str(task / "v"), "Destination": "/vol"},
        ]
    }  # fmt: skip
    pinned = {
        "/data": _identity(data),
        "/near": _identity(near),
        "/shared": _identity(tmp_path / "shared"),
        "/vol": _identity(task / "v"),
    }
    found = _bind_targets.staged_bind_targets(inspected, str(task), pinned)
    assert found == frozenset({"/data"})


def _linked_out_sandbox(root: Path, name: str, outside: Path) -> DockerSandbox:
    """Compose `./data:/data`, the task's `data` a link that leaves the task
    (kept as a link only with host access allowed)."""
    environment = root / "task" / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (environment / "Dockerfile").write_text(DOCKERFILE)
    (environment / "docker-compose.yaml").write_text(COMPOSE)
    link = environment / "data"
    if not link.is_symlink():
        os.symlink(str(outside), link)
    paths = RolloutPaths(rollout_dir=root / name)
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="restore-binds-linkout",
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.usefixtures("require_docker")
async def test_a_task_copy_link_leaving_the_task_keeps_the_parents_identity(
    tmp_path, monkeypatch, swap
):
    """DET-R2b review A, M-A1: the bind's source is a path in the child's
    copy, but a link there resolves to a host folder the parent shares. It is
    not the child's own: swapped between the parent's checkpoint and the
    child's first start, the restore refuses; unswapped, it passes."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    shared = tmp_path / "outside-shared"
    shared.mkdir()
    (shared / "f.txt").write_text("v1\n")
    parent = _linked_out_sandbox(tmp_path, "parent", shared)
    child = _linked_out_sandbox(tmp_path, "child", shared)
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        if swap:
            _swap_path(shared)
            (shared / "f.txt").write_text("v2-swapped\n")
        await child.start(force_build=False)
        if swap:
            with pytest.raises(RuntimeError, match=REFUSAL):
                await child.restore(image)
        else:
            await child.restore(image)
            seen = await child.exec("cat /data/f.txt", user="root")
            assert (seen.return_code, seen.stdout.strip()) == (0, "v1")
        assert child._staged_bind_targets == {"main": frozenset()}
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


def test_a_source_in_the_copy_that_resolves_outside_it_is_not_staged(tmp_path):
    """The unit form of M-A1: the source's links are resolved first."""
    task = tmp_path / "stage" / "task"
    (task / "environment" / "own").mkdir(parents=True)
    outside = tmp_path / "shared"
    outside.mkdir()
    os.symlink(str(outside), task / "environment" / "data")
    os.symlink("own", task / "environment" / "inner")
    inspected = {
        "Mounts": [
            {"Type": "bind", "Source": str(task / "environment" / "data"),
             "Destination": "/data"},
            {"Type": "bind", "Source": str(task / "environment" / "own"),
             "Destination": "/own"},
            {"Type": "bind", "Source": str(task / "environment" / "inner"),
             "Destination": "/inner"},
        ]
    }  # fmt: skip
    pinned = {
        "/data": _identity(outside),
        "/own": _identity(task / "environment" / "own"),
        "/inner": _identity(task / "environment" / "own"),
    }
    found = _bind_targets.staged_bind_targets(inspected, str(task), pinned)
    assert found == frozenset({"/own", "/inner"})


@pytest.mark.usefixtures("require_docker")
async def test_a_shared_folder_bound_by_absolute_path_keeps_the_parents_identity(
    tmp_path, monkeypatch
):
    """Review B's Q6b, the control for the link case above: with host access
    allowed, the task binds the shared folder by its absolute path. It is not
    in the task copy, so it is not own; swapped between the parent's
    checkpoint and the child's first start, the restore refuses."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    shared = tmp_path / "shared"
    shared.mkdir()
    absolute = COMPOSE.replace("./data:/data", f"{shared}:/data")
    compose = tmp_path / "task" / "environment" / "docker-compose.yaml"
    parent = _compose_sandbox(tmp_path, "parent")
    compose.write_text(absolute)
    child = _compose_sandbox(tmp_path, "child")
    compose.write_text(absolute)
    image = None
    try:
        await parent.start(force_build=False)
        image = await parent.snapshot()
        _swap_path(shared)
        await child.start(force_build=False)
        assert child._staged_bind_targets == {"main": frozenset()}
        with pytest.raises(RuntimeError, match=REFUSAL):
            await child.restore(image)
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)


# --- DET-R2b round 3 -------------------------------------------------------


def _pin_seam(tmp_path: Path, monkeypatch, sandbox: DockerSandbox, mounted: str):
    """``_pin_or_verify_bind_targets`` on ``sandbox`` with its Docker calls
    replaced: one ``main`` container binding the task copy's ``data`` at
    /data, pinned at ``mounted`` (the identity the container reports)."""
    from benchflow.sandbox import docker as docker_module

    task = tmp_path / "stage" / "task"
    data = task / "environment" / "data"
    data.mkdir(parents=True)
    inspected = {
        "Config": {"Labels": {"com.docker.compose.service": "main"}},
        "Mounts": [{"Type": "bind", "Source": str(data), "Destination": "/data"}],
    }

    async def pin(env, output, baseline):
        return {"main": {"/data": mounted or _identity(data)}}

    async def ids(_run):
        return ["a" * 64]

    async def inspect(_container):
        return inspected

    monkeypatch.setattr(docker_module._bind_targets, "pin_project", pin)
    monkeypatch.setattr(docker_module, "compose_container_ids", ids)
    sandbox._bind_baseline = None
    sandbox._staged_bind_targets = None
    sandbox._staged = SimpleNamespace(task_dir=task)
    sandbox._direct = None
    sandbox._run_docker_compose_command = None
    sandbox._inspect_container = inspect
    return data


@pytest.fixture
def no_docker_endpoint(tmp_path, monkeypatch):
    """No endpoint, context or saved context of the test host's own."""
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    return config


def _endpoint(case: str, config: Path, monkeypatch) -> None:
    if case == "docker-host":
        monkeypatch.setenv("DOCKER_HOST", "tcp://remote-daemon.invalid:2376")
    elif case == "docker-context":
        monkeypatch.setenv("DOCKER_CONTEXT", "remote-ctx")
    elif case == "saved-context":
        (config / "config.json").write_text(
            json.dumps({"currentContext": "remote-ctx"})
        )


@pytest.mark.parametrize("case", ["docker-host", "docker-context", "saved-context"])
async def test_an_ordinary_sandbox_on_another_daemon_records_nothing_own(
    tmp_path, monkeypatch, no_docker_endpoint, case
):
    """M-R2-A1: an ordinary DockerSandbox whose effective connection names
    another daemon (DOCKER_HOST, DOCKER_CONTEXT, a saved current context)
    cannot resolve a bind source here: nothing is own, and a child's
    task-copy binds keep the checkpoint's identity (strict)."""
    sandbox = object.__new__(DockerSandbox)
    _pin_seam(tmp_path, monkeypatch, sandbox, "")
    _endpoint(case, no_docker_endpoint, monkeypatch)

    await DockerSandbox._pin_or_verify_bind_targets(sandbox)

    assert sandbox._staged_bind_targets is None


async def test_the_remote_provider_records_nothing_own(
    tmp_path, monkeypatch, no_docker_endpoint
):
    """The remote-docker provider, a control: no own targets there either."""
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    sandbox = object.__new__(RemoteDockerSandbox)
    _pin_seam(tmp_path, monkeypatch, sandbox, "")

    await DockerSandbox._pin_or_verify_bind_targets(sandbox)

    assert sandbox._staged_bind_targets is None


@pytest.mark.skipif(sys.platform != "linux", reason="the local proof is Linux's")
async def test_the_native_local_daemon_records_its_task_copy_bind_as_own(
    tmp_path, monkeypatch, no_docker_endpoint
):
    """The positive control: the implicit native daemon, nothing set."""
    sandbox = object.__new__(DockerSandbox)
    _pin_seam(tmp_path, monkeypatch, sandbox, "")

    await DockerSandbox._pin_or_verify_bind_targets(sandbox)

    assert sandbox._staged_bind_targets == {"main": frozenset({"/data"})}


@pytest.mark.skipif(sys.platform != "linux", reason="the local proof is Linux's")
async def test_a_source_that_is_not_the_mounted_folder_is_not_own(
    tmp_path, monkeypatch, no_docker_endpoint
):
    """L-R2-A1: the first start pinned another folder's identity at /data
    (a link re-pointed into the copy between the mount and the walk); the
    walk's folder is not what the container mounted, so it is not own."""
    sandbox = object.__new__(DockerSandbox)
    shared = tmp_path / "shared"
    shared.mkdir()
    _pin_seam(tmp_path, monkeypatch, sandbox, _identity(shared))

    await DockerSandbox._pin_or_verify_bind_targets(sandbox)

    assert sandbox._staged_bind_targets == {"main": frozenset()}


# --- DET-R2b round 4 -------------------------------------------------------


@pytest.mark.parametrize("case", ["docker-host", "docker-context", "saved-context"])
async def test_a_launch_frozen_on_another_daemon_records_nothing_own(
    tmp_path, monkeypatch, no_docker_endpoint, case
):
    """M-R3-A1: the launch froze its connection while DOCKER_HOST,
    DOCKER_CONTEXT or a saved context named another daemon; the environment
    then turned local again. The sandbox still talks to the frozen daemon, so
    the locality proof reads the frozen arguments and nothing is own."""
    sandbox = object.__new__(DockerSandbox)
    _pin_seam(tmp_path, monkeypatch, sandbox, "")
    _endpoint(case, no_docker_endpoint, monkeypatch)
    sandbox._launch = SimpleNamespace(docker_args=sandbox._docker_compose_global_args())
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    (no_docker_endpoint / "config.json").write_text("{}")

    await DockerSandbox._pin_or_verify_bind_targets(sandbox)

    assert sandbox._staged_bind_targets is None


async def test_a_container_is_inspected_on_the_frozen_connection(monkeypatch):
    """The Compose path's ``docker inspect`` (the mounts the bind proof reads)
    goes to the frozen launch's daemon, with no connection variable of the
    environment's beside the flags, not to the environment's daemon now."""
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/now-local.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "now-local")
    sandbox = object.__new__(DockerSandbox)
    sandbox._direct = None
    sandbox._main_container = "c0ffee"
    sandbox._launch = SimpleNamespace(
        docker_args=["--config", "/frozen/config", "-H", "tcp://frozen.invalid:2376"]
    )
    seen: list[tuple[tuple[str, ...], dict | None]] = []

    class Proc:
        returncode = 0

        async def communicate(self):
            return b'[{"Id": "x"}]', b""

    async def fake_exec(*argv, **kwargs):
        seen.append((argv, kwargs.get("env")))
        return Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    await sandbox._inspect_container("a" * 64)

    ((argv, env),) = seen
    assert argv == (
        "docker",
        "--config",
        "/frozen/config",
        "-H",
        "tcp://frozen.invalid:2376",
        "inspect",
        "a" * 64,
    )
    assert env is not None
    assert "DOCKER_HOST" not in env and "DOCKER_CONTEXT" not in env
    assert sandbox._main_container == "c0ffee"


@pytest.mark.skipif(sys.platform != "linux", reason="the held walk is Linux's")
@pytest.mark.parametrize("when", ["before-open", "after-open"])
def test_a_folder_swapped_for_a_link_during_the_walk_is_not_own(
    tmp_path, monkeypatch, when
):
    """L-R3-A1 (review A's race probe): the receipt pinned a shared folder's
    identity at /data; the checked folder inside the copy is swapped for a
    link to that shared folder while the classification looks at it. One
    held walk gives containment and identity together, so it is never own:
    swapped before the walk opens it, the link leads out of the copy;
    swapped after, the held object is the copy's folder, not the pinned one."""
    task = tmp_path / "stage" / "task"
    race = task / "environment" / "race"
    race.mkdir(parents=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    pinned = {"/data": _identity(shared)}
    inspected = {
        "Mounts": [{"Type": "bind", "Source": str(race), "Destination": "/data"}]
    }
    swapped: list[bool] = []

    def swap() -> None:
        if not swapped:
            swapped.append(True)
            race.rename(race.with_name("race-old"))
            race.symlink_to(shared)

    real_open, real_stat = os.open, os.stat

    def racing_open(path, *args, **kwargs):
        if os.fspath(path) in ("race", str(race)) and when == "before-open":
            swap()
        fd = real_open(path, *args, **kwargs)
        if os.fspath(path) in ("race", str(race)) and when == "after-open":
            swap()
        return fd

    def racing_stat(path, *args, **kwargs):
        if os.fspath(path) == str(race):
            swap()
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr(os, "stat", racing_stat)
    found = _bind_targets.staged_bind_targets(inspected, str(task), pinned)
    monkeypatch.undo()

    assert swapped, "the swap never ran: the walk did not look at the folder"
    assert found == frozenset()


@pytest.mark.skipif(sys.platform != "linux", reason="the held walk is Linux's")
def test_the_held_chain_resolves_as_the_kernel_does(tmp_path):
    """Relative links, a chain of links, ``..`` and a link out give the same
    object and folders as ``realpath``; a loop gives None."""
    base = tmp_path / "base"
    (base / "a" / "b").mkdir(parents=True)
    (base / "out").mkdir()
    (base / "a" / "rel").symlink_to("b")
    (base / "a" / "chain").symlink_to("rel")
    (base / "a" / "up").symlink_to("../out")
    (base / "loop").symlink_to("loop")
    for name in ("a/rel", "a/chain", "a/up", "a/b/../b", "a/rel/.."):
        chain = _bind_targets.held_chain(str(base / name))
        assert chain is not None, name
        assert chain[-1] == tuple(
            int(x) for x in _identity(Path(os.path.realpath(base / name))).split(":")
        ), name
    assert _bind_targets.held_chain(str(base / "loop")) is None
    assert _bind_targets.held_chain(str(base / "a" / "missing")) is None

"""PERF-SBX: the job-start subnet read and the background flush before tests."""

import asyncio
import sys
import threading

import pytest

from benchflow.sandbox import docker as docker_module
from benchflow.sandbox.docker import DockerSandbox


@pytest.mark.asyncio
async def test_the_job_starts_the_local_daemons_subnet_read(monkeypatch):
    """The read the first sandbox's _choose_subnets waited for begins at job
    start, through allocator_for (which the sandbox's own call then shares)."""
    calls = []

    class Key:
        async def identify(self):
            calls.append("identify")

    async def allocator_for(key, docker, *, daemon_display):
        calls.append(("allocator_for", type(key).__name__, daemon_display))
        listed = await docker(["version", "--format", "{{.Client.Version}}"])
        calls.append(("docker", listed.return_code is not None))
        return None

    async def fake_exec(*argv, **kwargs):
        calls.append(("argv", argv))

        class Proc:
            returncode = 0

            async def communicate(self):
                return b"29.1.3\n", b""

        return Proc()

    monkeypatch.setattr(docker_module, "daemon_key", lambda env: Key())
    monkeypatch.setattr(docker_module, "allocator_for", allocator_for)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    task = docker_module.prefetch_local_subnet_allocator()
    assert task is not None
    await task
    assert calls[0] == "identify"
    assert calls[1] == ("allocator_for", "Key", "Docker daemon")
    # The plain local client: docker with the caller's own environment.
    assert calls[2] == (
        "argv",
        ("docker", "version", "--format", "{{.Client.Version}}"),
    )


@pytest.mark.asyncio
async def test_a_failed_subnet_read_is_left_to_the_sandbox(monkeypatch):
    class Key:
        async def identify(self):
            raise RuntimeError("no daemon")

    monkeypatch.setattr(docker_module, "daemon_key", lambda env: Key())
    task = docker_module.prefetch_local_subnet_allocator()
    await task  # never raises: the sandbox's own call reports it
    assert task.exception() is None


def test_no_subnet_read_outside_a_running_loop():
    assert docker_module.prefetch_local_subnet_allocator() is None


@pytest.mark.asyncio
async def test_a_job_cancels_an_unfinished_subnet_read(monkeypatch, tmp_path):
    """A job that ends (or is refused) first leaves no pending task behind."""
    from benchflow.evaluation import Evaluation

    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(3600)

    holder = {}

    def prefetch():
        holder["task"] = asyncio.get_running_loop().create_task(slow())
        return holder["task"]

    monkeypatch.setattr(docker_module, "prefetch_local_subnet_allocator", prefetch)
    evaluation = object.__new__(Evaluation)
    evaluation._config = type("C", (), {"environment": "docker"})()
    evaluation._preflight = True

    def refuse():
        raise RuntimeError("refused before the job folder exists")

    evaluation._check_before_run = refuse
    with pytest.raises(RuntimeError, match="refused"):
        await Evaluation.run(evaluation)
    await asyncio.sleep(0)
    assert holder["task"].cancelled()


def _sandbox(client_env=None, remote=False):
    sandbox = object.__new__(DockerSandbox)
    sandbox._remote_daemon = remote
    sandbox._docker_client_env = lambda: client_env
    return sandbox


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux hosts only")
def test_one_flusher_runs_while_any_sandbox_is_in_its_tests(monkeypatch):
    """PERF-SBX: dirty pages go to disk about once a second while a
    sandbox's tests run, from one daemon thread for the process; the last
    sandbox's stop ends it."""
    import time

    synced = []
    monkeypatch.setattr(
        docker_module.os, "sync", lambda: synced.append(time.monotonic())
    )
    monkeypatch.setattr(docker_module._HOST_FLUSHER, "INTERVAL", 0.05)
    first, second = _sandbox(), _sandbox()
    first.flush_writes_in_background()
    first.flush_writes_in_background()  # once per sandbox
    second.flush_writes_in_background()
    flushers = [t for t in threading.enumerate() if t.name == "benchflow-flush"]
    assert len(flushers) == 1 and flushers[0].daemon
    time.sleep(0.3)
    assert len(synced) >= 3
    first._stop_flushing()
    time.sleep(0.2)
    assert flushers[0].is_alive()
    second._stop_flushing()
    flushers[0].join(2)
    assert not flushers[0].is_alive()
    count = len(synced)
    time.sleep(0.2)
    assert len(synced) == count


@pytest.mark.parametrize(
    "sandbox",
    [
        _sandbox(client_env={"DOCKER_HOST": "tcp://elsewhere:2376"}),
        _sandbox(remote=True),
    ],
)
def test_no_flush_for_a_daemon_on_another_host(monkeypatch, sandbox):
    monkeypatch.setattr(
        docker_module.os, "sync", lambda: pytest.fail("synced for a remote daemon")
    )
    sandbox.flush_writes_in_background()


class _UpdateClient:
    """A daemon that keeps one container's HostConfig and records updates."""

    def __init__(self, host, *, ignore_restore=False):
        self.host, self.ignore_restore, self.calls = dict(host), ignore_restore, []

    async def call(self, method, path, *, query=None, body=None, timeout=None):
        self.calls.append((method, path, body))
        if path.endswith("/update") and not (
            self.ignore_restore and len(self.calls) > 2
        ):
            self.host.update(body)


class _Direct:
    def __init__(self, client):
        self.client = client

    async def inspect(self, cid):
        return {"HostConfig": dict(self.client.host)}


def _cpu_sandbox(host, **kwargs):
    sandbox = object.__new__(DockerSandbox)
    sandbox._direct = _Direct(_UpdateClient(host, **kwargs))
    sandbox._main_container = "c" * 64
    sandbox._keep_containers = False
    return sandbox


@pytest.mark.asyncio
async def test_harness_steps_get_more_cpus_and_the_task_limit_comes_back(monkeypatch):
    monkeypatch.setattr(docker_module.os, "cpu_count", lambda: 16)
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000})
    client = sandbox._direct.client
    async with sandbox.harness_cpus():
        assert client.host["NanoCpus"] == 4_000_000_000
    assert client.host["NanoCpus"] == 1_000_000_000
    assert [body for _, _, body in client.calls] == [
        {"NanoCpus": 4_000_000_000},
        {"NanoCpus": 1_000_000_000},
    ]


@pytest.mark.asyncio
async def test_the_task_limit_comes_back_when_a_harness_step_fails(monkeypatch):
    monkeypatch.setattr(docker_module.os, "cpu_count", lambda: 16)
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000})
    with pytest.raises(ValueError, match="step failed"):
        async with sandbox.harness_cpus():
            raise ValueError("step failed")
    assert sandbox._direct.client.host["NanoCpus"] == 1_000_000_000


@pytest.mark.asyncio
async def test_a_limit_the_daemon_did_not_put_back_stops_the_trial(monkeypatch):
    monkeypatch.setattr(docker_module.os, "cpu_count", lambda: 16)
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000}, ignore_restore=True)
    sandbox._direct.client.calls.append(("seed", "", None))
    with pytest.raises(RuntimeError, match="will not run the task's commands"):
        async with sandbox.harness_cpus():
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        {"NanoCpus": 0},
        {"NanoCpus": 4_000_000_000},
        {"NanoCpus": 1_000_000_000, "CpuQuota": 50000},
        {"NanoCpus": 1_000_000_000, "CpuPeriod": 100000},
        {"NanoCpus": 1_000_000_000, "CpusetCpus": "0"},
    ],
)
async def test_other_cpu_limits_are_left_alone(monkeypatch, host):
    monkeypatch.setattr(docker_module.os, "cpu_count", lambda: 16)
    sandbox = _cpu_sandbox(host)
    async with sandbox.harness_cpus():
        pass
    assert sandbox._direct.client.calls == []


@pytest.mark.asyncio
async def test_no_lift_off_the_direct_path_or_for_kept_containers(monkeypatch):
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000})
    sandbox._keep_containers = True
    async with sandbox.harness_cpus():
        pass
    sandbox._keep_containers = False
    sandbox._direct = None
    async with sandbox.harness_cpus():
        pass


class _EngineFake:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def call(self, method, path, **kwargs):
        self.calls.append((method, path))
        reply = self.replies[path]
        if isinstance(reply, Exception):
            raise reply

        class _Reply:
            def json(_self):
                return reply

        return _Reply()


def _reads(replies):
    reads = object.__new__(docker_module._EngineReads)
    reads.client = _EngineFake(replies)
    reads.networks = None
    return reads


NET_A, NET_B = "a" * 64, "b" * 64


@pytest.mark.asyncio
async def test_engine_reads_print_what_the_cli_prints():
    """The allocator parses the CLI's text; the Engine's answers keep it."""
    import json

    reads = _reads(
        {
            "/info": {
                "Name": "build-vm",
                "DefaultAddressPools": [{"Base": "172.17.0.0/16", "Size": 16}],
            },
            "/networks": [
                {"Id": NET_A, "IPAM": {"Config": [{"Subnet": "10.1.0.0/28"}]}},
                {"Id": NET_B, "IPAM": {"Config": None}},
            ],
        }
    )
    info = await reads.answer(["info", "--format", reads._INFO])
    assert json.loads(info.stdout) == {
        "name": "build-vm",
        "pools": [{"Base": "172.17.0.0/16", "Size": 16}],
    }
    listed = await reads.answer(["network", "ls", "--quiet", "--no-trunc"])
    assert listed.stdout == f"{NET_A}\n{NET_B}\n"
    inspected = await reads.answer(
        ["network", "inspect", "--format", "{{json .IPAM.Config}}", NET_A, NET_B]
    )
    assert inspected.stdout == '[{"Subnet": "10.1.0.0/28"}]\nnull\n'
    from benchflow.sandbox._docker_subnets import network_subnets

    assert [str(n) for n in network_subnets(inspected.stdout)] == ["10.1.0.0/28"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replies, args",
    [
        ({"/info": OSError("socket")}, ["info", "--format", "x"]),
        (
            {"/info": OSError("socket")},
            ["info", "--format", docker_module._EngineReads._INFO],
        ),
        (
            {"/info": {"Name": 3}},
            ["info", "--format", docker_module._EngineReads._INFO],
        ),
        ({"/networks": {"message": "no"}}, ["network", "ls", "--quiet", "--no-trunc"]),
        ({}, ["network", "inspect", "--format", "{{json .IPAM.Config}}", NET_A]),
        ({}, ["image", "ls"]),
    ],
)
async def test_anything_else_is_left_to_the_cli(replies, args):
    assert await _reads(replies).answer(args) is None


@pytest.mark.asyncio
async def test_a_network_the_listing_did_not_name_goes_to_the_cli():
    reads = _reads({"/networks": [{"Id": NET_A, "IPAM": {"Config": []}}]})
    await reads.answer(["network", "ls", "--quiet", "--no-trunc"])
    assert (
        await reads.answer(
            ["network", "inspect", "--format", "{{json .IPAM.Config}}", NET_A, NET_B]
        )
        is None
    )


def _image_site_functions():
    """The ancestor checks of the image-site copy script, as functions."""
    import os
    import pathlib

    from benchflow.sandbox._verifier_runtime import _COPY_IMAGE_SITES

    text = _COPY_IMAGE_SITES
    start = text.index("def private_folder")
    end = text.index("for index, source")
    namespace = {"os": os, "pathlib": pathlib}
    exec(text[start:end], namespace)
    return namespace


def test_the_once_per_folder_check_answers_as_the_per_path_walk(tmp_path):
    """PERF-SBX: the image-site copy asked every copied path's ancestors
    again (0.4 s of 2.4 s on fix-code-vulnerability's 8,734 entries); asked
    once per folder, every path's answer is the same, links included."""
    import os

    site = tmp_path / "site"
    (site / "pkg" / "sub").mkdir(parents=True)
    (site / "pkg" / "sub" / "m.py").write_text("x")
    (site / "pkg" / "a.py").write_text("x")
    private = tmp_path / "private"
    (private / "inner").mkdir(parents=True)
    (private / "inner" / "data.py").write_text("x")
    private.chmod(0o750)
    (site / "linked").symlink_to(private / "inner", target_is_directory=True)
    (site / "file-link.py").symlink_to(private / "inner" / "data.py")
    (site / "dot").symlink_to(".", target_is_directory=True)
    paths = [site, *site.rglob("*"), site / "linked" / "data.py"]
    once = _image_site_functions()
    walk = _image_site_functions()
    try:
        # The real folder modes first, then a folder rule that tells this
        # tree's folders apart (the base temp folder is private to its user).
        for path in paths:
            assert once["private_ancestors_once"](path) == walk["private_ancestors"](
                path
            ), path
        real_private = os.path.realpath(private)

        def rule(folder):
            return str(folder) == real_private

        for namespace in (once, walk):
            namespace["private_folder"] = rule
            namespace["private_chains"].clear()
        answers = [walk["private_ancestors"](p) for p in paths]
        assert [once["private_ancestors_once"](p) for p in paths] == answers
        assert any(answers) and not all(answers)
    finally:
        private.chmod(0o755)


def test_sandbox_hooks_come_from_the_sandbox_class_only():
    """A stand-in or mock sandbox answers any attribute; the rollout's
    optional PERF-SBX hooks must not take a mock's answer for a method."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from benchflow.sandbox._hooks import sandbox_hook

    for stand_in in (MagicMock(), AsyncMock(), SimpleNamespace(harness_cpus=None)):
        assert sandbox_hook(stand_in, "harness_cpus") is None
    sandbox = object.__new__(DockerSandbox)
    assert sandbox_hook(sandbox, "harness_cpus").__func__ is DockerSandbox.harness_cpus
    assert sandbox_hook(sandbox, "no_such_hook") is None


@pytest.mark.asyncio
async def test_production_build_never_reads_or_stores_a_cached_kit(monkeypatch):
    """Guards b0cddc0d2: production builds each task's kit from its own state."""
    from unittest.mock import AsyncMock, Mock

    from benchflow._runtime_tree import kit_prepare_command
    from benchflow.sandbox import _kit_cache, lockdown
    from benchflow.sandbox._base import ExecResult

    assert _kit_cache.ENABLED is False
    monkeypatch.setenv("BENCHFLOW_KIT_CACHE", "1")
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000})
    sandbox._recovery_baseline = None
    inspect = AsyncMock(side_effect=AssertionError("cache inspected the container"))
    monkeypatch.setattr(sandbox._direct, "inspect", inspect)
    load = Mock(side_effect=AssertionError("cache loaded"))
    store = Mock(side_effect=AssertionError("cache stored"))
    monkeypatch.setattr(_kit_cache, "load", load)
    monkeypatch.setattr(_kit_cache, "store", store)
    cache_store = AsyncMock(side_effect=AssertionError("kit exported for cache"))
    monkeypatch.setattr(sandbox, "kit_cache_store", cache_store)
    build = AsyncMock(return_value=ExecResult(stdout="BF kit-built\n", return_code=0))
    monkeypatch.setattr(lockdown, "_read_exec", build)

    assert await lockdown.prepare_runtime_tree(sandbox, phase="build") is True
    build.assert_awaited_once_with(
        sandbox, kit_prepare_command("build"), user="root", timeout_sec=300
    )
    inspect.assert_not_awaited()
    load.assert_not_called()
    cache_store.assert_not_awaited()
    store.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_user", [None, "agent"])
async def test_pre_verify_hardening_keeps_the_tasks_cpu_limit(
    monkeypatch, tmp_path, sandbox_user
):
    """Guards b0cddc0d2: leftover solution processes never share a CPU lift."""
    import contextlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.rollout import _setup
    from benchflow.task.paths import RolloutPaths
    from tests.test_verifier_environment_checks import _receipt_task

    monkeypatch.setattr(docker_module.os, "cpu_count", lambda: 16)
    sandbox = _cpu_sandbox({"NanoCpus": 1_000_000_000})
    monkeypatch.setattr(sandbox, "flush_writes_in_background", lambda: None)
    seen = []

    async def harden(*args, **kwargs):
        seen.append(sandbox._direct.client.host["NanoCpus"])

    async def verify(*args, **kwargs):
        seen.append(sandbox._direct.client.host["NanoCpus"])
        return SimpleNamespace(rewards={"reward": 1.0})

    monkeypatch.setattr(_setup, "_await_verifier", verify)
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(side_effect=harden),
        verifier=lambda **kwargs: SimpleNamespace(),
    )
    task = _receipt_task(tmp_path)
    with contextlib.ExitStack() as root_path:
        result = await _setup._verify_hardened(
            sandbox,
            task,
            RolloutPaths(tmp_path / "rollout"),
            {},
            planes,
            sandbox_user,
            "/app",
            recovery_eligible=False,
            agent_paths=None,
            root_path=root_path,
        )
    assert result == ({"reward": 1.0}, None, None)
    planes.harden_before_verify.assert_awaited_once()
    assert seen == [1_000_000_000, 1_000_000_000]
    assert sandbox._direct.client.calls == []

"""The direct Docker path on a real daemon (PERF-E).

A single-service task's sandbox starts and stops without any ``docker
compose`` command that touches the daemon (only ``compose config`` to read
the project, and ``compose version`` once per process), and its container is
the one Compose makes: the same inspect under the published normalisation,
the same Compose labels (so ``compose ps`` finds it), and nothing left after
stop. A task with a Compose key the path does not render starts on Compose
with the reason recorded.

Skipped without a Docker daemon; ``ubuntu:24.04`` is the deterministic
tier's base image, so its hosts hold it.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import ipaddress
import json
import random
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from benchflow.sandbox import _docker_subnets
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

IMAGE = "ubuntu:24.04"
#: Compose verbs that create, start, list, kill or remove containers.
DAEMON_VERBS = {"create", "up", "start", "ps", "kill", "rm", "down", "stop", "restart"}
# The test's own docker calls, never logged as the sandbox's.
_run = subprocess.run


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20
        )
    except Exception:
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_ready(), reason=f"needs a Docker daemon holding {IMAGE}"
)


def _diff_tool():
    path = Path(__file__).parents[1] / "tools/perf/direct_docker_diff.py"
    spec = importlib.util.spec_from_file_location("direct_docker_diff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sandbox(
    tmp_path: Path, compose: dict | None = None, *, dockerfile: str | None = None
) -> DockerSandbox:
    """A task on IMAGE; with *dockerfile*, one built from that Dockerfile."""
    name = f"perf-e-{uuid.uuid4().hex[:8]}"
    env = tmp_path / name / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text(dockerfile or f"FROM {IMAGE}\n")
    if compose is not None:
        (env / "docker-compose.yaml").write_text(json.dumps(compose))
    paths = RolloutPaths(rollout_dir=tmp_path / name / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=env,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(
            cpus=1, memory_mb=512, docker_image=None if dockerfile else IMAGE
        ),
    )


@pytest.fixture
def argv_log(monkeypatch) -> list[list[str]]:
    """Every subprocess a sandbox starts, by argv."""
    seen: list[list[str]] = []
    real_async = asyncio.create_subprocess_exec
    real_run = subprocess.run

    async def spawn(*argv, **kwargs):
        seen.append([str(a) for a in argv])
        return await real_async(*argv, **kwargs)

    def run(argv, *args, **kwargs):
        if isinstance(argv, list | tuple):
            seen.append([str(a) for a in argv])
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(subprocess, "run", run)
    return seen


def _free_subnet() -> str:
    """A /28 no network on the daemon overlaps, picked at random from
    10.250.0.0/16 (PERF-E-B-L2: a fixed one collided with another run)."""
    ids = _run(["docker", "network", "ls", "-q"], capture_output=True, text=True)
    used = []
    if ids.stdout.split():
        out = _run(
            ["docker", "network", "inspect", "--format",
             "{{range .IPAM.Config}}{{.Subnet}} {{end}}", *ids.stdout.split()],
            capture_output=True,
            text=True,
        ).stdout  # fmt: skip
        for word in out.split():
            with contextlib.suppress(ValueError):
                used.append(ipaddress.ip_network(word, strict=False))
    pool = list(ipaddress.ip_network("10.250.0.0/16").subnets(new_prefix=28))
    random.shuffle(pool)
    for candidate in pool:
        if not any(candidate.overlaps(net) for net in used if net.version == 4):
            return str(candidate)
    # A failure, never a skip: a D949 count must not include skips (B-R3-L5).
    pytest.fail("no free /28 in 10.250.0.0/16 on this daemon")


@pytest.fixture
def one_subnet(monkeypatch):
    """One /28 for every network: the daemon's address choice is the same
    on both paths, so nothing about it needs masking. Chosen free at run
    time, so concurrent runs on one daemon seldom meet."""
    monkeypatch.setenv("BENCHFLOW_DOCKER_SUBNETS", _free_subnet())
    _docker_subnets.reset_allocators()
    yield
    _docker_subnets.reset_allocators()


def _compose_verbs(seen: list[list[str]]) -> list[str]:
    verbs = []
    for argv in seen:
        if "compose" not in argv:
            continue
        rest = iter(argv[argv.index("compose") + 1 :])
        for arg in rest:
            if arg in ("-p", "--project-name", "--project-directory", "-f", "--file"):
                next(rest, None)
            elif not arg.startswith("-"):
                verbs.append(arg)
                break
    return verbs


def _project_objects(project: str) -> list[str]:
    label = f"label=com.docker.compose.project={project}"
    containers = _run(
        ["docker", "ps", "-aq", "--filter", label], capture_output=True, text=True
    ).stdout.split()
    networks = _run(
        ["docker", "network", "ls", "-q", "--filter", label],
        capture_output=True,
        text=True,
    ).stdout.split()
    return containers + networks


def _inspect(kind: str, ref: str) -> dict:
    out = _run(
        ["docker", kind, "inspect", ref], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


@pytest.mark.asyncio
async def test_a_single_service_task_starts_and_stops_without_compose_on_the_daemon(
    tmp_path, monkeypatch, argv_log
):
    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    sandbox = _sandbox(tmp_path)
    project = sandbox.compose_project_name
    await sandbox.start(force_build=False)
    try:
        assert not set(_compose_verbs(argv_log)) & DAEMON_VERBS, _compose_verbs(
            argv_log
        )
        record = getattr(sandbox, "direct_docker_record", None) or {}
        assert record.get("path") == "direct", record
        # The versions it took the path on, recorded (PERF-E-r3-A-L2).
        assert set(record) == {"path", "compose", "api", "daemon_api", "engine"}
        main = sandbox._main_container
        assert main is not None
        labels = _inspect("container", main)["Config"]["Labels"]
        assert labels["com.docker.compose.project"] == project
        assert labels["com.docker.compose.service"] == "main"
        assert labels["com.docker.compose.oneoff"] == "False"
        # Compose finds it as its own container.
        listed = _run(
            ["docker", "compose", "-p", project, "ps", "-q"],
            capture_output=True,
            text=True,
        ).stdout.split()
        assert [
            main.startswith(cid) or cid.startswith(main[:12]) for cid in listed
        ] == [True]
        # The barrier was released and the task's command runs as PID 1.
        result = await sandbox.exec("cat /proc/1/cmdline", user="root")
        assert result.return_code == 0
        assert "sleep" in (result.stdout or "")
        whoami = await sandbox.exec("id -u", user="root")
        assert (whoami.stdout or "").strip() == "0"
    finally:
        await sandbox.stop(delete=True)
    assert not set(_compose_verbs(argv_log)) & DAEMON_VERBS
    assert _project_objects(project) == []


@pytest.mark.asyncio
async def test_a_key_the_path_does_not_render_starts_on_compose_with_its_reason(
    tmp_path, monkeypatch, argv_log
):
    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    sandbox = _sandbox(
        tmp_path,
        {"services": {"main": {"image": IMAGE, "stop_signal": "SIGINT"}}},
    )
    project = sandbox.compose_project_name
    await sandbox.start(force_build=False)
    try:
        assert sandbox.direct_docker_record == {
            "path": "compose",
            "reason": "service key stop_signal",
        }
        assert "up" in _compose_verbs(argv_log)
        assert (await sandbox.exec("true", user="root")).return_code == 0
    finally:
        await sandbox.stop(delete=True)
    assert _project_objects(project) == []


@pytest.mark.parametrize(
    "compose",
    [
        None,
        # Four variables: their order reaches the bootstrap's argv and receipt,
        # canonical now, so they compare exactly.
        {
            "services": {
                "main": {
                    "image": IMAGE,
                    "environment": {"D_4": "4", "A_1": "1", "C_3": "x", "B_2": ""},
                }
            }
        },
    ],
    ids=["dockerfile", "environment"],
)
@pytest.mark.asyncio
async def test_the_direct_path_makes_the_container_compose_makes(
    tmp_path, monkeypatch, one_subnet, compose
):
    """D962 layer 2 in the suite: the same task started both ways, main and
    its network inspected while running, compared under normalise_inspect
    after masking names, IDs and per-trial values only."""
    from benchflow.sandbox._direct_docker import differences, normalise_inspect

    tool = _diff_tool()
    seen = {}
    for direct in (False, True):
        monkeypatch.setattr(DockerSandbox, "direct_docker", direct, raising=False)
        sandbox = _sandbox(tmp_path / ("direct" if direct else "compose"), compose)
        await sandbox.start(force_build=False)
        record = sandbox.direct_docker_record or {}
        if direct:
            assert record.get("path") == "direct", record
        else:
            assert record == {
                "path": "compose",
                "reason": "off (BENCHFLOW_DIRECT_DOCKER=0)",
            }
        try:
            main = sandbox._main_container
            assert main is not None
            container = _inspect("container", main)
            network = _inspect("network", f"{sandbox.compose_project_name}_default")
            root = str(sandbox.rollout_paths.rollout_dir.parent)
            seen[direct] = [
                tool.mask(
                    normalise_inspect(container), sandbox.compose_project_name, root
                ),
                tool.mask(network, sandbox.compose_project_name, root),
            ]
        finally:
            await sandbox.stop(delete=True)
    for compose_side, direct_side in zip(seen[False], seen[True], strict=True):
        assert differences(compose_side, direct_side) == []


@pytest.mark.asyncio
async def test_an_image_that_declares_a_volume_starts_on_compose(tmp_path, monkeypatch):
    """PERF-E-A-1: the image's VOLUME gets a named volume from BenchFlow's
    overlay (results/ANONVOL.md, round 3); the direct path makes binds only,
    so such an image runs on Compose, with the reason recorded."""
    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    sandbox = _sandbox(tmp_path, dockerfile=f"FROM {IMAGE}\nVOLUME /data\n")
    project = sandbox.compose_project_name
    await sandbox.start(force_build=True)
    try:
        assert getattr(sandbox, "direct_docker_record", None) == {
            "path": "compose",
            "reason": (
                "BenchFlow names this trial's volumes (an image VOLUME or a "
                "Compose anonymous volume); the direct path makes binds only"
            ),
        }
        assert (await sandbox.exec("test -d /data", user="root")).return_code == 0
        main = sandbox._main_container or ""
        volumes = [
            m["Name"]
            for m in _inspect("container", main)["Mounts"]
            if m.get("Type") == "volume"
        ]
        assert len(volumes) == 1, volumes
    finally:
        await sandbox.stop(delete=True)
    assert _project_objects(project) == []
    # DOCKER-ANON-VOLUME-LEAK: the anonymous volume goes with the project.
    gone = _run(["docker", "volume", "inspect", volumes[0]], capture_output=True)
    assert gone.returncode != 0, f"anonymous volume {volumes[0]} was left behind"


@pytest.mark.asyncio
async def test_a_compose_version_the_path_was_not_proven_on_starts_on_compose(
    tmp_path, monkeypatch
):
    """PERF-E-A-2: the bodies and hashes are Compose 2.40.3's."""
    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)

    async def other_compose(self):
        return "2.41.0"

    monkeypatch.setattr(DockerSandbox, "_compose_version", other_compose)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    try:
        record = getattr(sandbox, "direct_docker_record", None)
        assert record is not None and record["path"] == "compose", record
        assert record["reason"].startswith(
            "Compose 2.41.0: the direct path is proven on"
        )
    finally:
        await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_an_engine_api_version_the_path_was_not_proven_on_starts_on_compose(
    tmp_path, monkeypatch
):
    """PERF-E-A-2: the bodies are API 1.51's (Compose's client negotiates the
    same version down from the daemon's)."""
    from benchflow.sandbox import _engine_api

    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    monkeypatch.setattr(_engine_api, "negotiated", lambda server: "1.50")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    try:
        record = getattr(sandbox, "direct_docker_record", None)
        assert record is not None and record["path"] == "compose", record
        assert record["reason"].startswith(
            "Engine API 1.50: the direct path is proven on"
        )
    finally:
        await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_compose_version_that_times_out_starts_on_compose(
    tmp_path, monkeypatch
):
    """PERF-E-r3-A-L1: a recorded fallback, not a failed trial."""
    from benchflow.sandbox.protocol import ExecTimeoutError

    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    monkeypatch.setattr(DockerSandbox, "_compose_versions", {}, raising=False)
    real = DockerSandbox._run_docker_process

    async def slow_version(self, argv, **kwargs):
        if argv[-2:] == ["version", "--short"]:
            raise ExecTimeoutError(30)
        return await real(self, argv, **kwargs)

    monkeypatch.setattr(DockerSandbox, "_run_docker_process", slow_version)
    sandbox = _sandbox(tmp_path)
    project = sandbox.compose_project_name
    await sandbox.start(force_build=False)
    try:
        record = getattr(sandbox, "direct_docker_record", None) or {}
        assert record.get("path") == "compose", record
        assert record["reason"].startswith("docker compose version did not answer")
        assert (await sandbox.exec("true", user="root")).return_code == 0
    finally:
        await sandbox.stop(delete=True)
    assert _project_objects(project) == []


@pytest.mark.asyncio
async def test_a_daemon_api_version_that_does_not_parse_starts_on_compose(
    tmp_path, monkeypatch
):
    """PERF-E-r3-A-L1: '1.52-beta' is a recorded fallback, not a ValueError."""
    from benchflow.sandbox import _engine_api

    monkeypatch.setattr(DockerSandbox, "direct_docker", True, raising=False)
    real = _engine_api.EngineClient._raw

    async def beta_ping(self, method, path, **kwargs):
        reply = await real(self, method, path, **kwargs)
        if path == "/_ping":
            reply.headers["api-version"] = "1.52-beta"
        return reply

    monkeypatch.setattr(_engine_api.EngineClient, "_raw", beta_ping)
    sandbox = _sandbox(tmp_path)
    project = sandbox.compose_project_name
    await sandbox.start(force_build=False)
    try:
        record = getattr(sandbox, "direct_docker_record", None) or {}
        assert record.get("path") == "compose", record
        assert "1.52-beta" in record["reason"]
    finally:
        await sandbox.stop(delete=True)
    assert _project_objects(project) == []

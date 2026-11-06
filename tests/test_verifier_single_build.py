"""Guards SV4-C40-R1-1/-2 at 36e57950: one verifier context build path."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox._docker_subnets import daemon_key
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from tests._verifier_invocation import synthetic_invocation
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    isolated_layers,
)

REF = "python@sha256:" + "1" * 64
BOOLS = [
    None,
    "",
    "0",
    "1",
    "f",
    "F",
    "false",
    "False",
    "FALSE",
    "t",
    "T",
    "true",
    "True",
    "TRUE",
    "yes",
    "fAlSe",
    " 1",
]
CASES = (
    [("DOCKER_BUILDKIT", v) for v in BOOLS]
    + [("COMPOSE_BAKE", v) for v in BOOLS]
    + [("SOURCE_DATE_EPOCH", v) for v in [None, "", "0", "1700000000"]]
    + [("DOCKER_CONTEXT", v) for v in [None, "default", "other"]]
)


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(DockerSandbox, "_resolve_compose_build", synthetic_invocation)
    fake = BakeDocker()
    fake.add_image(REF)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    monkeypatch.chdir(tmp_path)
    return fake


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("docker_default", ["20.10.24", "29.1.3"])
@pytest.mark.parametrize("source", ["client", "compose", "both"])
@pytest.mark.parametrize("name,value", CASES)
async def test_cached_and_uncached_context_use_identical_compose_build(
    daemon, tmp_path, monkeypatch, backend, docker_default, source, name, value
):
    """Compare the complete builder command/environment, including fallback inputs."""
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {REF}\nUSER 65534\nWORKDIR /opt/verifier-data\nUSER root\nWORKDIR /app\n"
    )
    box = backend.__new__(backend)
    # Use the real Compose command assembly on both backends; transport and
    # execution are dummy local stand-ins, never an SSH connection.
    DockerSandbox.__init__(
        box, context, "single-build", "single-build", None, SandboxConfig()
    )
    box.use_verifier_layer(_recipe(tmp_path))
    box.docker_host = SimpleNamespace(redact=lambda text, **kwargs: text)
    selected = box._env_vars.to_env_dict(include_os_env=False)
    client = {}
    if source == "client":
        selected.update(DOCKER_BUILDKIT="1", COMPOSE_BAKE="false")
    if value is not None and source in {"compose", "both"}:
        selected[name] = value
    if value is not None and source in {"client", "both"}:
        client[name] = value
    monkeypatch.setattr(backend, "_docker_compose_env", lambda self: dict(selected))
    monkeypatch.setattr(backend, "_docker_client_env", lambda self: dict(client))
    monkeypatch.setattr(backend, "_docker_subnet_key", lambda self: daemon_key({}))
    monkeypatch.setattr(backend, "_docker_compose_global_args", lambda self: [])
    monkeypatch.setattr(backend, "_compose_transport_env", lambda self: {})
    original = daemon.run

    async def docker(*args, **kwargs):
        if args[:1] == ("version",):
            return daemon._ok(docker_default)
        return await original(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", staticmethod(docker))
    commands = []

    async def execute(self, command, **kwargs):
        commands.append((command, self._docker_compose_env(), kwargs))
        daemon.add_image(
            self._env_vars.main_image_name,
            labels={
                "com.docker.compose.project": self.compose_project_name,
                "com.docker.compose.service": "main",
            },
        )
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_docker_process", execute)
    await box._run_docker_compose_build()
    ordinary = commands.pop()
    used = await box._start_verifier_from_context()
    if not used:
        await box._run_docker_compose_build()
    assert len(commands) == 1
    # The cache's explicit frozen record has extra execution kwargs; the
    # original launcher, argv and effective environment are still identical.
    assert commands[0][:2] == ordinary[:2]
    effective_env = commands[0][2].get("environment", commands[0][1])
    assert effective_env == ordinary[1]
    assert ordinary[0][-1] == "build"
    assert not daemon.context_builds  # no independent docker build <context>
    if used:
        commands.clear()
        box._use_prebuilt = False
        box._env_vars.prebuilt_image_name = None
        box._verifier_layer = None
        assert await box._start_verifier_from_context()
        assert box.verifier_layer.result.status == "hit"
        assert commands == []


async def test_no_independent_context_builder_when_backend_has_no_shared_path(
    daemon, tmp_path
):
    """Unknown backends fail closed before cache lookup rather than inventing a build."""
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(f"FROM {REF}\n")
    with pytest.raises(LayerError, match=r"shared.*build"):
        await vl.ensure_verifier_layer(
            DockerLayers(env={}), _recipe(tmp_path), None, alias=None, context=context
        )
    assert not daemon.context_builds and not daemon.bake_files


def test_format_six_entries_are_invalidated(tmp_path, monkeypatch):
    """Old entries were built through a different frontend (SV4-C40-R1)."""
    from tests.test_verifier_layers import BASE

    recipe = _recipe(tmp_path)
    current = vl.verifier_layer_key(recipe, BASE)
    monkeypatch.setattr(vl, "VERIFIER_LAYER_FORMAT", 6)
    assert vl.verifier_layer_key(recipe, BASE) != current


@pytest.mark.parametrize(
    "name",
    [
        "SOURCE_DATE_EPOCH",
        "DOCKER_BUILDKIT",
        "COMPOSE_BAKE",
        "DOCKER_CONTEXT",
        "FUTURE_BUILDER_INPUT",
    ],
)
async def test_every_environment_input_changes_context_cache_identity(
    daemon, tmp_path, monkeypatch, name
):
    """The key binds opaque inputs, including future builder settings (C40-R1)."""
    (tmp_path / "Dockerfile").write_text(f"FROM {REF}\n")
    box = DockerSandbox(tmp_path, "identity", "identity", None, SandboxConfig())
    env = box._env_vars.to_env_dict(include_os_env=False)
    env["COMPOSE_BAKE"] = "false"
    env["DOCKER_BUILDKIT"] = "1"
    monkeypatch.setattr(box, "_docker_compose_env", lambda: dict(env))
    layers = DockerLayers(env={})
    first = await box._verifier_build_identity(layers)
    env[name] = {"COMPOSE_BAKE": "0", "DOCKER_BUILDKIT": "t"}.get(name, "first-value")
    second = await box._verifier_build_identity(layers)
    env[name] = {"COMPOSE_BAKE": "F", "DOCKER_BUILDKIT": "TRUE"}.get(
        name, "second-value"
    )
    third = await box._verifier_build_identity(layers)
    assert len({first, second, third}) == 3
    assert all(len(v) == 64 for v in (first, second, third))


@pytest.mark.parametrize(
    "name", ["MAIN_IMAGE_NAME", "CONTEXT_DIR", "PREBUILT_IMAGE_NAME"]
)
async def test_generated_compose_inputs_cannot_be_redirected(
    daemon, tmp_path, monkeypatch, name
):
    """A context hash cannot authenticate a task redirect to a different context."""
    (tmp_path / "Dockerfile").write_text(f"FROM {REF}\n")
    box = DockerSandbox(tmp_path, "identity", "identity", None, SandboxConfig())
    env = box._env_vars.to_env_dict(include_os_env=False)
    env[name] = "/tmp/other-context"
    monkeypatch.setattr(box, "_docker_compose_env", lambda: env)
    with pytest.raises(LayerError, match="stock Compose"):
        await box._verifier_build_identity(DockerLayers(env={}))


async def test_explicit_build_is_not_repeated_implicitly_during_up(
    daemon, tmp_path, monkeypatch
):
    """Guards C40-R1: pull_policy=build cannot introduce another build invocation."""
    (tmp_path / "Dockerfile").write_text(f"FROM {REF}\n")
    box = DockerSandbox(tmp_path, "up-build", "up-build", None, SandboxConfig())
    monkeypatch.setattr(box, "_docker_subnet_key", lambda: daemon_key({}))
    calls = []

    async def compose(command, *, timeout_sec=None):
        calls.append(command)
        if command == ["build"]:
            assert timeout_sec is not None and timeout_sec > 0

    monkeypatch.setattr(box, "_run_docker_compose_command", compose)
    await box._build_task_image()
    await box._run_docker_compose_up()
    assert calls == [["build"], ["up", "--detach", "--wait", "--no-build"]]


@pytest.mark.parametrize(
    "project,service",
    [(None, None), ("other-project", "main"), ("same", "sidecar"), ("same", "main")],
)
async def test_capture_authenticates_build_owner_then_consumes_immutable_id(
    daemon, tmp_path, monkeypatch, project, service
):
    """A mutable build tag may change between processes; never cache another build."""
    (tmp_path / "Dockerfile").write_text(f"FROM {REF}\n")
    box = DockerSandbox(tmp_path, "capture", "capture", None, SandboxConfig())
    layers = DockerLayers(env={})
    monkeypatch.setattr(box, "_docker_layers", lambda: layers)
    monkeypatch.setattr(box, "_docker_subnet_key", lambda: daemon_key({}))
    labels = {
        "com.docker.compose.project": box.compose_project_name
        if project == "same"
        else project,
        "com.docker.compose.service": service,
    }
    image = daemon.add_image(box._env_vars.main_image_name, labels=labels)

    box._resolved_context_build = await synthetic_invocation(box)

    async def build(*, invocation):
        assert invocation is box._resolved_context_build
        pass

    monkeypatch.setattr(box, "_run_docker_compose_build", build)
    refs = []
    original = layers.inspect_base

    async def inspect(ref, **kwargs):
        refs.append(ref)
        return await original(ref, **kwargs)

    monkeypatch.setattr(layers, "inspect_base", inspect)
    if (project, service) == ("same", "main"):
        captured = await box._build_task_image(capture_image=True)
        assert captured.image_id == image.id
        assert refs == [image.id]
    else:
        with pytest.raises(LayerError, match="changed before capture"):
            await box._build_task_image(capture_image=True)
        assert not refs

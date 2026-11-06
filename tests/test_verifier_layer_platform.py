"""Cache admission preserves platform."""

import copy
from types import SimpleNamespace

import pytest

from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox._docker_subnets import daemon_key
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    context_build,
    isolated_layers,
)

REF = "python@sha256:" + "1" * 64
OVERRIDES = [
    "linux/386",
    "386",
    "linux/amd64",
    "linux/x86_64",
    "amd64",
    "linux/amd64/v2",
    "linux/arm64",
    "linux/arm64/v8",
    "linux/arm/v7",
    "linux/ppc64le",
    "windows/amd64",
    "linux/amd64,linux/386",
    "local",
    " linux/386",
    "LINUX/386",
    "unknown",
    " ",
]


@pytest.fixture
def daemon(monkeypatch):
    fake = BakeDocker()
    fake.add_image(REF)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    for key in (
        "DOCKER_DEFAULT_PLATFORM",
        "BUILDX_BUILDER",
        "BUILDKIT_HOST",
        "COMPOSE_BAKE",
    ):
        monkeypatch.delenv(key, raising=False)
    return fake


async def request(tmp_path, layers, route, **kwargs):
    recipe = _recipe(tmp_path)
    if route == "context":
        context = tmp_path / "context"
        context.mkdir()
        (context / "Dockerfile").write_text(f"FROM {REF}\n")
        return await vl.ensure_verifier_layer(
            layers,
            recipe,
            None,
            alias=None,
            context=context,
            **context_build(context),
            **kwargs,
        )
    base = await layers.inspect_base(REF) if route == "inspected" else None
    return await vl.ensure_verifier_layer(
        layers, recipe, REF, alias=None, base=base, **kwargs
    )


@pytest.mark.parametrize("platform", OVERRIDES)
@pytest.mark.parametrize("route", ["context", "image", "inspected"])
@pytest.mark.parametrize("source", ["inherited", "client"])
async def test_any_platform_override_uses_ordinary_grading(
    daemon, tmp_path, monkeypatch, platform, route, source
):
    """No Docker spelling, path or already-inspected base may choose another Python."""
    if source == "inherited":
        monkeypatch.setenv("DOCKER_DEFAULT_PLATFORM", platform)
        layers = DockerLayers()
    else:
        layers = DockerLayers(env={"DOCKER_DEFAULT_PLATFORM": platform})
    with pytest.raises(LayerError, match="platform"):
        await request(tmp_path, layers, route)
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize("route", ["context", "image", "inspected"])
async def test_both_builds_pin_platform(daemon, tmp_path, route):
    """The label-only wrapper must not acquire platform metadata from the host."""
    await request(tmp_path, DockerLayers(env={}), route)
    calls = [call for call in daemon.calls if call[0] == "build"]
    assert len(calls) == (2 if route == "context" else 1)
    assert all(
        "--platform" in call and call[call.index("--platform") + 1] == "linux/amd64"
        for call in calls
    )


@pytest.mark.parametrize("route", ["context", "image", "inspected"])
@pytest.mark.parametrize(
    "change",
    [
        {"Architecture": "386"},
        {"Architecture": "arm64", "Variant": "v8"},
        {"Variant": "v2"},
        {"Os": "windows"},
        {"Architecture": ""},
    ],
)
async def test_foreign_local_images_cannot_select_build_platform(
    daemon, tmp_path, route, change
):
    daemon.tags[REF].info.update(change)
    with pytest.raises(LayerError, match="platform"):
        await request(tmp_path, DockerLayers(), route)
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize(
    "reply", ["linux/aarch64", "windows/x86_64", "linux/armv7l", "", "malformed"]
)
async def test_unresolved_default_platform_uses_ordinary_grading(
    daemon, tmp_path, monkeypatch, reply
):
    run = daemon.run

    async def docker(*args, **kwargs):
        if args == ("info", "--format", "{{.OSType}}/{{.Architecture}}"):
            return daemon._ok(reply)
        return await run(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", staticmethod(docker))
    with pytest.raises(LayerError, match="platform"):
        await request(tmp_path, DockerLayers(), "context")
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize(
    "env",
    [
        {"BUILDX_BUILDER": "custom"},
        {"BUILDKIT_HOST": "tcp://dummy:1234"},
        {"COMPOSE_BAKE": "true"},
        {"COMPOSE_ENV_FILES": "dummy-platform.env"},
    ],
)
async def test_unresolved_builder_uses_ordinary_grading(daemon, tmp_path, env):
    with pytest.raises(LayerError, match="builder"):
        await request(tmp_path, DockerLayers(env=env), "context")
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("platform", OVERRIDES)
@pytest.mark.parametrize("route", ["context", "image"])
@pytest.mark.parametrize("source", ["environment", "project-dotenv", "cwd-dotenv"])
async def test_backend_passes_effective_compose_platform(
    daemon, tmp_path, monkeypatch, backend, platform, route, source
):
    """Both backends must fall back when task env changes Compose's selection."""
    box = backend.__new__(backend)
    box._verifier_layer_recipe = _recipe(tmp_path)
    box._verifier_layer = None
    box._recovery_baseline = None
    box.environment_dir = tmp_path / "context"
    box.environment_dir.mkdir()
    (box.environment_dir / "Dockerfile").write_text(f"FROM {REF}\n")
    box._compose_project = "platform-test"
    box.layer_records = {}
    box._env_vars = SimpleNamespace(prebuilt_image_name=None)
    box._use_prebuilt = False
    compose_env = {"DOCKER_DEFAULT_PLATFORM": platform}
    if source != "environment":
        compose_env = {}
        directory = box.environment_dir
        if source == "cwd-dotenv":
            monkeypatch.chdir(tmp_path)
            directory = tmp_path
        (directory / ".env").write_text(f"DOCKER_DEFAULT_PLATFORM={platform}\n")
    monkeypatch.setattr(backend, "_agent_layer_unsupported", lambda self: None)
    monkeypatch.setattr(backend, "_docker_client_env", lambda self: {})
    monkeypatch.setattr(backend, "_docker_subnet_key", lambda self: daemon_key({}))
    monkeypatch.setattr(
        backend,
        "_docker_compose_env",
        lambda self: compose_env,
    )
    monkeypatch.setattr(backend, "_layer_failed", lambda self, *args: None)
    if route == "context":
        assert not await box._start_verifier_from_context()
        assert not box._use_prebuilt
    else:
        box._agent_layer_recipe = None
        box._proxy_layer_recipe = None
        box._use_prebuilt = True
        box.task_env_config = SimpleNamespace(docker_image=REF)
        await box._start_from_layers()
    assert box._verifier_layer_recipe is None
    assert box._verifier_layer is None
    assert not daemon.context_builds and not daemon.bake_files


async def test_old_format_is_not_reused(daemon, tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    base = await DockerLayers().inspect_base(REF)
    current = vl.verifier_layer_key(recipe, base)
    monkeypatch.setattr(vl, "VERIFIER_LAYER_FORMAT", 3)
    assert current != vl.verifier_layer_key(recipe, base)


async def test_changing_platform_cannot_hit_existing_native_layer(daemon, tmp_path):
    recipe = _recipe(tmp_path)
    first = await vl.ensure_verifier_layer(
        DockerLayers(env={}), recipe, REF, alias=None
    )
    before = copy.deepcopy(daemon.calls)
    with pytest.raises(LayerError, match="platform"):
        await vl.ensure_verifier_layer(
            DockerLayers(env={"DOCKER_DEFAULT_PLATFORM": "linux/386"}),
            recipe,
            REF,
            alias=None,
        )
    assert first.result.status == "miss"
    assert daemon.calls == before


@pytest.mark.parametrize(
    "reply",
    [
        "Name: custom\nDriver: docker-container\n\nNodes:\nName: default\nDriver: docker",
        "Name: default\nDriver: remote",
        "Name: other\nDriver: docker",
        "",
        "Name: default\nName: other\nDriver: docker",
        "unstructured diagnostic",
    ],
)
async def test_saved_builder_selection_must_match_engine(
    daemon, tmp_path, monkeypatch, reply
):
    """Compose's saved builder must agree with docker build, even without env overrides."""
    run = daemon.run

    async def docker(*args, **kwargs):
        if args == ("buildx", "inspect"):
            return daemon._ok(reply)
        return await run(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", staticmethod(docker))
    with pytest.raises(LayerError, match="builder"):
        await request(tmp_path, DockerLayers(), "context")
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize("unreadable", [False, True])
async def test_unknown_dotenv_uses_ordinary_grading(
    daemon, tmp_path, monkeypatch, unreadable
):
    """A dangling link or failed stat is not proof of a default platform."""
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    if unreadable:
        exists = Path.exists

        def check(path):
            if path.name == ".env":
                raise PermissionError("dummy unavailable directory")
            return exists(path)

        monkeypatch.setattr(Path, "exists", check)
    else:
        (tmp_path / ".env").symlink_to("missing-platform-env")
    with pytest.raises(LayerError, match="dotenv"):
        await request(tmp_path, DockerLayers(), "image")
    assert not daemon.bake_files

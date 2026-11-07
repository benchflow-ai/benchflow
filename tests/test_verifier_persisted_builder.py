"""Guards SV4-B54-C1-1 at 2f102eae: saved builders cannot hide behind a hit."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchflow.sandbox._docker_subnets import daemon_key
from benchflow.sandbox._docker_sweep import PROCESS_ENV
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import VERIFIER, DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from tests._verifier_invocation import synthetic_invocation
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    isolated_layers,
)

REF = "python@sha256:" + "1" * 64
FALSE = ["0", "f", "F", "false", "False", "FALSE"]
TRUE = ["1", "t", "T", "true", "True", "TRUE"]
UNCERTAIN = [
    None,
    "",
    "1",
    "t",
    "T",
    "true",
    "True",
    "TRUE",
    "fAlSe",
    " false",
    "0 ",
    "off",
]


@pytest.fixture
def setup_box(monkeypatch, tmp_path):
    monkeypatch.setattr(DockerSandbox, "_resolve_compose_build", synthetic_invocation)
    fake = BakeDocker()
    fake.add_image(REF)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    monkeypatch.chdir(tmp_path)

    def create(backend, selected):
        context = tmp_path / "context"
        context.mkdir(exist_ok=True)
        (context / "Dockerfile").write_text(f"FROM {REF}\nWORKDIR /opt/verifier-data\n")
        box = backend.__new__(backend)
        DockerSandbox.__init__(
            box, context, "saved-builder", "saved-builder", None, SandboxConfig()
        )
        box.docker_host = SimpleNamespace(redact=lambda text, **kwargs: text)
        box.use_verifier_layer(_recipe(tmp_path))
        monkeypatch.setattr(box, "_docker_client_env", lambda: {})
        monkeypatch.setattr(
            box,
            "_docker_compose_env",
            lambda: {**box._env_vars.to_env_dict(include_os_env=False), **selected},
        )
        monkeypatch.setattr(box, "_compose_transport_env", lambda: {})
        monkeypatch.setattr(box, "_docker_compose_global_args", lambda: [])
        monkeypatch.setattr(box, "_docker_subnet_key", lambda: daemon_key({}))
        builds = []

        async def build(*, invocation):
            assert invocation is box._resolved_context_build
            builds.append(dict(selected))
            fake.add_image(
                box._env_vars.main_image_name,
                labels={
                    "com.docker.compose.project": box.compose_project_name,
                    "com.docker.compose.service": "main",
                },
            )

        monkeypatch.setattr(box, "_run_docker_compose_build", build)
        return box, fake, builds

    return create


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("value", UNCERTAIN)
async def test_uncertain_selection_refuses_before_any_lookup(
    setup_box, backend, value, monkeypatch
):
    """Neither a cold key nor an existing tag can bypass builder admission."""
    env = {"DOCKER_BUILDKIT": "1"}
    if value is not None:
        env["COMPOSE_BAKE"] = value
    box, fake, builds = setup_box(backend, env)
    lookups = []

    async def unexpected_lookup(*args, **kwargs):
        lookups.append(args)
        raise AssertionError("uncertain builder reached cache lookup")

    monkeypatch.setattr(DockerLayers, "_ensure", unexpected_lookup)
    assert not await box._start_verifier_from_context()
    assert not lookups and not builds and not fake.bake_files
    assert "builder" in box.layer_records[VERIFIER]["reason"]


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize(
    "location", ["home", "docker-config", "symlink-config", "buildx-config", "context"]
)
@pytest.mark.parametrize("warm", [False, True])
async def test_persisted_settings_cannot_reuse_an_unresolved_entry(
    setup_box, tmp_path, monkeypatch, backend, location, warm
):
    """Saved Compose/plugin and buildx/context inputs stay opaque, cold or warm."""
    root = tmp_path / "settings"
    root.mkdir()
    env = {"HOME": str(root), "DOCKER_BUILDKIT": "1"}
    config = root / ".docker"
    if location in {"docker-config", "symlink-config"}:
        config = root / "docker"
        env["DOCKER_CONFIG"] = str(config)
    if location == "symlink-config":
        actual = root / "actual"
        actual.mkdir()
        config.symlink_to(actual, target_is_directory=True)
    else:
        config.mkdir()
    if location == "buildx-config":
        env["BUILDX_CONFIG"] = str(root / "buildx")
    if location == "context":
        env["DOCKER_CONTEXT"] = "saved-context"
    config_file = config / "config.json"
    config_file.write_text("{}")
    box, fake, builds = setup_box(backend, env)
    identity = box._verifier_build_identity
    if warm:
        # Emulate a format-7 entry created by the reviewed code, whose identity
        # was unchanged by a config-file edit. Admission must precede this hit.
        async def old_identity(layers):
            # The obsolete admission also supplied an executable build. Keep
            # that record shape when emulating its unchecked warm entry.
            box._resolved_context_build = replace(
                await box._resolve_compose_build(), uncertainty=None
            )
            material = dict(box._docker_compose_env())
            for name in type(box._env_vars).model_fields:
                material.pop(name.upper(), None)
            material.pop(PROCESS_ENV, None)
            return hashlib.sha256(
                json.dumps(
                    [material, ["test-tool-version"] * 3],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()

        monkeypatch.setattr(box, "_verifier_build_identity", old_identity)
        assert await box._start_verifier_from_context()
        assert box.verifier_layer.result.status == "miss"
        box._env_vars.prebuilt_image_name = None
        box._use_prebuilt = False
        box.use_verifier_layer(box.verifier_layer.recipe)
        monkeypatch.setattr(box, "_verifier_build_identity", identity)
    config_file.write_text(json.dumps({"plugins": {"compose": {"build": "bake"}}}))
    prior = (len(builds), len(fake.bake_files))
    assert not await box._start_verifier_from_context()
    assert box.verifier_layer is None
    assert (len(builds), len(fake.bake_files)) == prior
    assert "builder" in box.layer_records[VERIFIER]["reason"]


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("value", FALSE)
@pytest.mark.parametrize("buildkit", TRUE)
async def test_explicit_internal_builder_misses_then_hits(
    setup_box, backend, value, buildkit
):
    """Each accepted Go spelling resolves the frontend without reading config."""
    box, fake, builds = setup_box(
        backend, {"COMPOSE_BAKE": value, "DOCKER_BUILDKIT": buildkit}
    )
    recipe = box._verifier_layer_recipe
    for status in ("miss", "hit"):
        assert await box._start_verifier_from_context()
        assert box.verifier_layer.result.status == status
        box._env_vars.prebuilt_image_name = None
        box._use_prebuilt = False
        box.use_verifier_layer(recipe)
    assert len(builds) == 1 and len(fake.bake_files) == 1


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
async def test_client_false_does_not_resolve_unset_effective_compose(
    setup_box, backend, monkeypatch
):
    """The effective build environment owns the decision, including task removal."""
    box, _, builds = setup_box(backend, {})
    monkeypatch.setattr(box, "_docker_client_env", lambda: {"COMPOSE_BAKE": "false"})
    with pytest.raises(LayerError, match="builder"):
        await box._verifier_build_identity(DockerLayers(env={"COMPOSE_BAKE": "false"}))
    assert not builds


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("value", [None, "", *FALSE, "yes", "tRUE", " 1", "1 "])
async def test_daemon_buildkit_default_is_not_a_resolved_builder(
    setup_box, backend, value, monkeypatch
):
    """Saved daemon defaults and CLI aliases cannot choose a second frontend."""
    env = {"COMPOSE_BAKE": "false"}
    if value is not None:
        env["DOCKER_BUILDKIT"] = value
    box, _, builds = setup_box(backend, env)
    with pytest.raises(LayerError, match="explicit BuildKit builder"):
        await box._verifier_build_identity(DockerLayers(env={}))
    assert not builds

"""Guards the fix at 36c4d8a9: cache only BuildKit builds.

Docker and Compose read DOCKER_BUILDKIT and COMPOSE_BAKE with Go's
ParseBool. A false DOCKER_BUILDKIT selects the legacy builder, which creates
a WORKDIR as root rather than as the current USER; an invalid value or an
empty COMPOSE_BAKE fails the ordinary build. The cache admits only the
spellings for which every build is BuildKit and otherwise grades ordinarily.
"""

import logging
from types import SimpleNamespace

import pytest

from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox._docker_subnets import daemon_key
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import VERIFIER, DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_verifier_layer_platform import REF, request
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    isolated_layers,
)

UNSET = None
BUILDKIT_ON = [UNSET, "", "1", "t", "T", "true", "True", "TRUE"]
BUILDKIT_OFF = ["0", "f", "F", "false", "False", "FALSE"]
BUILDKIT_INVALID = [
    "no",
    "off",
    "yes",
    "on",
    "enabled",
    "2",
    "-1",
    " 1",
    "1 ",
    "tRUE",
    "fAlSe",
    "\t",
    "TRUE\n",
]
BAKE_ADMITTED = [UNSET, "0", "f", "F", "false", "False", "FALSE"]
# Truthy values were already refused; "" and non-Go spellings fail Compose.
BAKE_REFUSED = [
    "",
    "1",
    "t",
    "T",
    "true",
    "True",
    "TRUE",
    "no",
    "off",
    "yes",
    "fAlSe",
    " false",
    "0 ",
]
SOURCES = ["inherited", "client", "compose"]
ROUTES = ["context", "image", "inspected"]


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    fake = BakeDocker()
    fake.add_image(REF)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    monkeypatch.chdir(tmp_path)
    for name in (
        "DOCKER_BUILDKIT",
        "COMPOSE_BAKE",
        "EXPERIMENTAL_BUILDKIT_SOURCE_POLICY",
        "BUILDX_BUILDER",
        "BUILDKIT_HOST",
        "COMPOSE_ENV_FILES",
        "DOCKER_DEFAULT_PLATFORM",
    ):
        monkeypatch.delenv(name, raising=False)
    return fake


def environments(monkeypatch, source, name, value):
    """``(client layers, compose env)`` with ``name=value`` from ``source``."""
    selected = {} if value is UNSET else {name: value}
    if source == "inherited":
        for key, item in selected.items():
            monkeypatch.setenv(key, item)
        return DockerLayers(), {}
    if source == "client":
        return DockerLayers(env=selected), {}
    return DockerLayers(env={}), selected


def refused(name):
    return "BuildKit" if name == "DOCKER_BUILDKIT" else "builder"


CASES_REFUSED = [("DOCKER_BUILDKIT", v) for v in BUILDKIT_OFF + BUILDKIT_INVALID] + [
    ("COMPOSE_BAKE", v) for v in BAKE_REFUSED
]
CASES_ADMITTED = [("DOCKER_BUILDKIT", v) for v in BUILDKIT_ON] + [
    ("COMPOSE_BAKE", v) for v in BAKE_ADMITTED
]


@pytest.mark.parametrize(("name", "value"), CASES_REFUSED)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("source", SOURCES)
async def test_non_buildkit_selection_refuses_cold(
    daemon, tmp_path, monkeypatch, name, value, route, source
):
    """A cold cache builds nothing when either build may not use BuildKit."""
    layers, compose_env = environments(monkeypatch, source, name, value)
    with pytest.raises(LayerError, match=refused(name)):
        await request(tmp_path, layers, route, compose_env=compose_env)
    assert not daemon.context_builds and not daemon.bake_files


@pytest.mark.parametrize(("name", "value"), CASES_REFUSED)
@pytest.mark.parametrize("source", SOURCES)
async def test_non_buildkit_selection_cannot_hit_a_warm_layer(
    daemon, tmp_path, monkeypatch, name, value, source
):
    """An entry built under BuildKit is never served to a non-BuildKit grade."""
    recipe = _recipe(tmp_path)
    first = await vl.ensure_verifier_layer(
        DockerLayers(env={}), recipe, REF, alias=None, compose_env={}
    )
    assert first.result.status == "miss"
    builds = list(daemon.bake_files)
    layers, compose_env = environments(monkeypatch, source, name, value)
    with pytest.raises(LayerError, match=refused(name)):
        await vl.ensure_verifier_layer(
            layers, recipe, REF, alias=None, compose_env=compose_env
        )
    assert daemon.bake_files == builds


@pytest.mark.parametrize(("name", "value"), CASES_ADMITTED)
@pytest.mark.parametrize("source", SOURCES)
async def test_buildkit_selection_keeps_cold_and_warm_cache(
    daemon, tmp_path, monkeypatch, name, value, source
):
    """Every spelling that selects BuildKit everywhere still caches."""
    layers, compose_env = environments(monkeypatch, source, name, value)
    recipe = _recipe(tmp_path)
    for status in ("miss", "hit"):
        result = await vl.ensure_verifier_layer(
            layers, recipe, REF, alias=None, compose_env=compose_env
        )
        assert result.result.status == status


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("route", ["context", "image"])
@pytest.mark.parametrize(("name", "value"), CASES_REFUSED)
async def test_backend_grades_ordinarily_with_a_note(
    daemon, tmp_path, monkeypatch, backend, route, name, value
):
    """Both Docker hooks fall back and say why in the trial's layer record."""
    box = backend.__new__(backend)
    box.logger = logging.getLogger("test-verifier-buildkit")
    box.environment_name = "buildkit-test"
    box._verifier_layer_recipe = _recipe(tmp_path)
    box._verifier_layer = None
    box._recovery_baseline = None
    box.environment_dir = tmp_path / "context"
    box.environment_dir.mkdir()
    (box.environment_dir / "Dockerfile").write_text(f"FROM {REF}\n")
    box._compose_project = "buildkit-test"
    box.layer_records = {}
    box._env_vars = SimpleNamespace(prebuilt_image_name=None)
    box._use_prebuilt = False
    monkeypatch.setattr(backend, "_agent_layer_unsupported", lambda self: None)
    monkeypatch.setattr(backend, "_docker_client_env", lambda self: {})
    monkeypatch.setattr(backend, "_docker_subnet_key", lambda self: daemon_key({}))
    monkeypatch.setattr(backend, "_docker_compose_env", lambda self: {name: value})
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
    record = box.layer_records[VERIFIER]
    assert record["status"] == "fresh"
    assert refused(name) in record["reason"]
    assert not daemon.context_builds and not daemon.bake_files


async def test_format_five_keys_are_invalidated(daemon, tmp_path, monkeypatch):
    """Entries admitted before BuildKit was required cannot hit."""
    recipe = _recipe(tmp_path)
    base = await DockerLayers(env={}).inspect_base(REF)
    current = vl.verifier_layer_key(recipe, base)
    monkeypatch.setattr(vl, "VERIFIER_LAYER_FORMAT", 5)
    assert vl.verifier_layer_key(recipe, base) != current

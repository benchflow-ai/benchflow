"""Guards the fix and SV4-CONFIRM-2 at fc0ec1e5."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox._docker_subnets import daemon_key
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_verifier_layer_platform import REF, request
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    isolated_layers,
)

SELECTORS = ["BUILDX_CONFIG", "DOCKER_CONFIG", "HOME", "DOCKER_CONTEXT", "DOCKER_HOST"]
POLICY = "EXPERIMENTAL_BUILDKIT_SOURCE_POLICY"


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    fake = BakeDocker()
    fake.add_image(REF)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    monkeypatch.chdir(tmp_path)
    for name in (
        POLICY,
        "BUILDX_BUILDER",
        "BUILDKIT_HOST",
        "COMPOSE_ENV_FILES",
        "DOCKER_DEFAULT_PLATFORM",
        "COMPOSE_BAKE",
    ):
        monkeypatch.delenv(name, raising=False)
    return fake


def select_environment(monkeypatch, fake, selector, failure):
    """Model legitimate saved settings selected only by the effective environment."""
    probes = []

    async def docker(self, *args, **kwargs):
        import os

        env = os.environ if self._env is None else self._env
        selected = env.get(selector) == "dummy-selection"
        if args[0] in {"buildx", "info"}:
            probes.append((dict(env), args))
        if selected:
            if args == ("buildx", "inspect"):
                replies = {
                    "builder": "Name: arm-builder\nDriver: remote\n",
                    "driver": "Name: default\nDriver: docker-container\n",
                    "unreadable-builder": "",
                }
                if failure in replies:
                    return fake._ok(replies[failure])
                if failure == "failed-builder":
                    return fake._fail("dummy builder unavailable")
            if args == ("info", "--format", "{{.OSType}}/{{.Architecture}}"):
                if failure == "platform":
                    return fake._ok("linux/aarch64\n")
                if failure == "failed-platform":
                    return fake._fail("dummy daemon unavailable")
            if args == ("info", "--format", "{{.ID}}"):
                if failure == "daemon":
                    return fake._ok("another-daemon\n")
                if failure == "unreadable-daemon":
                    return fake._ok("")
                if failure == "malformed-daemon":
                    return fake._ok("two identities\n")
                if failure == "failed-daemon":
                    return fake._fail("dummy identity unavailable")
        return await fake.run(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", docker)
    return probes


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("route", ["context", "image", "inspected"])
@pytest.mark.parametrize("source", ["client", "inherited", "compose"])
@pytest.mark.parametrize(
    "failure",
    [
        "builder",
        "driver",
        "unreadable-builder",
        "failed-builder",
        "platform",
        "failed-platform",
        "daemon",
        "unreadable-daemon",
        "malformed-daemon",
        "failed-daemon",
    ],
)
async def test_compose_selection_is_checked_before_lookup(
    daemon, tmp_path, monkeypatch, selector, route, source, failure
):
    """Guards the fix at fc0ec1e5 across every config root, endpoint and cache path."""
    select_environment(monkeypatch, daemon, selector, failure)
    if source == "inherited":
        monkeypatch.setenv(selector, "dummy-selection")
    selected = {selector: "dummy-selection", "COMPOSE_BAKE": "false"}
    layers = DockerLayers(
        env=selected if source == "client" else None if source == "inherited" else {}
    )
    with pytest.raises(LayerError):
        await request(
            tmp_path,
            layers,
            route,
            compose_env=selected if source == "compose" else {},
        )
    assert not daemon.bake_files and not daemon.context_builds


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("source", ["client", "inherited", "compose"])
async def test_equivalent_effective_environments_keep_cache_hits(
    daemon, tmp_path, monkeypatch, selector, source
):
    """Different config locations alone do not require a cache miss or refusal."""
    probes = select_environment(monkeypatch, daemon, selector, "equivalent")
    selected = {selector: "dummy-selection"}
    if source == "inherited":
        monkeypatch.setenv(selector, "dummy-selection")
    layers = DockerLayers(
        env=selected if source == "client" else None if source == "inherited" else {}
    )
    recipe = _recipe(tmp_path)
    for status in ("miss", "hit"):
        result = await vl.ensure_verifier_layer(
            layers,
            recipe,
            REF,
            alias=None,
            compose_env=selected if source == "compose" else {},
        )
        assert result.result.status == status
    assert any(env.get(selector) == "dummy-selection" for env, _ in probes)
    assert any(not env.get(selector) for env, _ in probes)


@pytest.mark.parametrize("source", ["client", "inherited", "compose"])
@pytest.mark.parametrize("route", ["context", "image", "inspected"])
@pytest.mark.parametrize(
    "value", ["dummy-policy.json", "/tmp/dummy-policy.json", "", " "]
)
async def test_any_source_policy_refuses_before_lookup(
    daemon, tmp_path, monkeypatch, source, route, value
):
    """Guards CONFIRM-2 at fc0ec1e5 without parsing policy paths or contents."""
    if source == "inherited":
        monkeypatch.setenv(POLICY, value)
    layers = DockerLayers(
        env={POLICY: value}
        if source == "client"
        else None
        if source == "inherited"
        else {}
    )
    with pytest.raises(LayerError, match="source policy"):
        await request(
            tmp_path,
            layers,
            route,
            compose_env={POLICY: value} if source == "compose" else {},
        )
    assert not daemon.bake_files and not daemon.context_builds


@pytest.mark.parametrize("selector", [*SELECTORS, POLICY])
@pytest.mark.parametrize("source", ["client", "inherited", "compose"])
async def test_changed_build_inputs_cannot_hit_a_warm_layer(
    daemon, tmp_path, monkeypatch, selector, source
):
    """Both fc0ec1e5 findings must be refused even after warming the same key."""
    select_environment(monkeypatch, daemon, selector, "builder")
    recipe = _recipe(tmp_path)
    layers = DockerLayers(env={})
    first = await vl.ensure_verifier_layer(
        layers, recipe, REF, alias=None, compose_env={}
    )
    assert first.result.status == "miss"
    builds = list(daemon.bake_files)
    selected = {selector: "dummy-selection"}
    if source == "inherited":
        monkeypatch.setenv(selector, "dummy-selection")
    layers = DockerLayers(
        env=selected if source == "client" else None if source == "inherited" else {}
    )
    with pytest.raises(LayerError):
        await vl.ensure_verifier_layer(
            layers,
            recipe,
            REF,
            alias=None,
            compose_env=selected if source == "compose" else {},
        )
    assert daemon.bake_files == builds


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("route", ["context", "image"])
@pytest.mark.parametrize("selector", [*SELECTORS, POLICY])
async def test_backend_falls_back_for_effective_build_inputs(
    daemon, tmp_path, monkeypatch, backend, route, selector
):
    """Both Docker hooks must keep ordinary grading after admission refuses."""
    select_environment(monkeypatch, daemon, selector, "builder")
    box = backend.__new__(backend)
    box._verifier_layer_recipe = _recipe(tmp_path)
    box._verifier_layer = None
    box._recovery_baseline = None
    box.environment_dir = tmp_path / "context"
    box.environment_dir.mkdir()
    (box.environment_dir / "Dockerfile").write_text(f"FROM {REF}\n")
    box._compose_project = "build-input-test"
    box.layer_records = {}
    box._env_vars = SimpleNamespace(prebuilt_image_name=None)
    box._use_prebuilt = False
    monkeypatch.setattr(backend, "_agent_layer_unsupported", lambda self: None)
    monkeypatch.setattr(backend, "_docker_client_env", lambda self: {})
    monkeypatch.setattr(backend, "_docker_subnet_key", lambda self: daemon_key({}))
    monkeypatch.setattr(
        backend, "_docker_compose_env", lambda self: {selector: "dummy-selection"}
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


async def test_format_four_keys_are_invalidated(daemon, tmp_path, monkeypatch):
    """Guards CONFIRM-2 at fc0ec1e5: old policy-contaminated entries cannot hit."""
    recipe = _recipe(tmp_path)
    base = await DockerLayers(env={}).inspect_base(REF)
    current = vl.verifier_layer_key(recipe, base)
    monkeypatch.setattr(vl, "VERIFIER_LAYER_FORMAT", 4)
    assert vl.verifier_layer_key(recipe, base) != current

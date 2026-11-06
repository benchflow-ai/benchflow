"""Guards the fix at integration commit 363d815b."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


def sandbox(root: Path, backend="docker", **kwargs):
    environment = root / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM scratch\n")
    paths = RolloutPaths(rollout_dir=root / "run")
    paths.mkdir()
    cls = d.DockerSandbox
    if backend == "remote-docker":
        cls = RemoteDockerSandbox
        kwargs["docker_host"] = RemoteDockerHost(url="tcp://127.0.0.1:2376")
    env = cls(
        environment_dir=environment,
        environment_name="same-task",
        session_id="same-session",
        rollout_paths=paths,
        task_env_config=kwargs.pop("task_env_config", SandboxConfig()),
        **kwargs,
    )
    # Guards the 85b87bc9 merge: these planner fakes have no mounted containers
    # or synchronous Compose client; dedicated S1 tests cover admission/proof.
    env.freeze_launch = lambda *args: None
    env._prepare_inert_bootstrap = AsyncMock()
    env._pin_or_verify_bind_targets = AsyncMock()
    env._release_inert_bootstrap = AsyncMock()
    if backend == "remote-docker":
        # These build/planner tests supply their own resolved configuration.
        # Remote host admission has separate transport and policy fixtures.
        env._check_task_project = AsyncMock()
    return env


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("same_checkout", [False, True])
def test_build_tags_belong_to_the_trial(tmp_path, backend, same_checkout):
    """Guards 363d815b: neither same names nor same paths share a build tag."""
    first = sandbox(tmp_path / "a", backend)
    second = sandbox(tmp_path / "b", backend)
    if same_checkout:
        second.environment_dir = first.environment_dir
    assert first._env_vars.main_image_name != second._env_vars.main_image_name
    for env in (first, second):
        tag = f"{env._env_vars.main_image_name}:latest"
        assert tag in env._built_image_tags
        assert tag in env._private_built_image_tags


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_layers_inspect_only_their_own_completed_build(
    tmp_path, monkeypatch, backend
):
    """Guards 363d815b: an interleaved sibling build cannot change our base."""
    first = sandbox(tmp_path / "a", backend)
    second = sandbox(tmp_path / "b", backend)
    images = {}
    for env, value in ((first, "a"), (second, "b")):
        images[env._env_vars.main_image_name] = value
    seen = []

    async def ensure(self, recipe, ref, **kwargs):
        seen.append(images[ref])
        raise d.LayerError("stop after observing the base")

    monkeypatch.setattr(d.DockerLayers, "ensure_proxy_layer", ensure)
    first._proxy_layer_recipe = AsyncMock()
    await first._start_from_layers()
    assert seen == ["a"]

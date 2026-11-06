"""A Docker without buildx: the layer build fails, and the trial falls back.

The legacy builder cannot build a layer's ``RUN --mount`` and deletes its
``--iidfile`` before it exits 1. Reading the file anyway raised a
FileNotFoundError, which is no LayerError, so the trial stopped instead of
installing the proxy or agent in the sandbox (seen on Docker 29.1.3 with no
buildx).
"""

from __future__ import annotations

import os

import pytest

from benchflow.sandbox import env_layers
from benchflow.sandbox.env_layers import AgentLayerRecipe, DockerLayers, LayerError
from tests.test_env_layers_agent import FakeDocker

LEGACY_ERROR = (
    "the --mount option requires BuildKit. Refer to "
    "https://docs.docker.com/go/buildkit/ to learn how to enable BuildKit."
)


def _legacy_build(docker: FakeDocker):
    """The legacy builder: delete the iid file, then fail."""

    async def run(layers, *args, **kwargs):
        if args[0] == "build":
            iidfile = args[args.index("--iidfile") + 1]
            os.unlink(iidfile)
            return env_layers._Run("", LEGACY_ERROR, 1)
        return await docker.run(*args, **kwargs)

    return run


async def test_a_failed_build_gives_its_result_and_no_image(monkeypatch):
    docker = FakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", _legacy_build(docker))
    run, image_id = await DockerLayers()._build(
        "bf-layer:test",
        {},
        "FROM task\nRUN --mount=type=bind,source=a,target=/a true\n",
        timeout=5,
    )
    assert run.code == 1
    assert image_id == ""


async def test_the_agent_layer_fails_as_a_layer_error_naming_buildkit(monkeypatch):
    docker = FakeDocker()
    docker.add_image("task")
    monkeypatch.setattr(DockerLayers, "_docker", _legacy_build(docker))
    recipe = AgentLayerRecipe(agent="dummy", version="1.0.0", install_cmd="true")
    with pytest.raises(LayerError) as failed:
        await DockerLayers().ensure_agent_layer(recipe, "task", prune=False)
    assert "requires BuildKit" in str(failed.value)

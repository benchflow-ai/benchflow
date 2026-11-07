"""Historical image-context regressions retained by the SV4 redesign."""

import pytest

from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from tests.test_verifier_layers import (  # noqa: F401
    BakeDocker,
    _recipe,
    context_build,
    isolated_layers,
)


@pytest.mark.parametrize("remote", [False, True], ids=["docker", "remote-docker"])
@pytest.mark.parametrize("ref", ["0", "1", "2", "00", "01", "3", "0:latest"])
async def test_from_cannot_use_copy_indices(tmp_path, monkeypatch, remote, ref):
    """No cache lookup/build for mutable FROMs."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    pinned = "grader@sha256:" + "1" * 64
    fake.add_image(pinned)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {pinned} AS first\nFROM first AS second\nFROM second\nFROM {ref}\n"
    )
    layers = DockerLayers(env={"DOCKER_HOST": "ssh://dummy-build"} if remote else None)
    with pytest.raises(LayerError, match="digest-pinned external FROM"):
        await vl.ensure_verifier_layer(
            layers,
            _recipe(tmp_path, separate=True),
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert fake.context_builds == [] and fake.bake_files == []


@pytest.mark.parametrize("remote", [False, True], ids=["docker", "remote-docker"])
@pytest.mark.parametrize("source", ["0", "1", "first", "FIRST", "second"])
async def test_named_from_and_indexed_copy_still_reuse(
    tmp_path, monkeypatch, remote, source
):
    """Guards the fix at 00d22b85 with distinct name and index syntax controls."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    pinned = "0@sha256:" + "1" * 64
    fake.add_image(pinned)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {pinned} AS first\nFROM FIRST AS second\nFROM second\n"
        f"COPY --from={source} /expected /expected\n"
    )
    layers = DockerLayers(env={"DOCKER_HOST": "ssh://dummy-build"} if remote else None)
    recipe = _recipe(tmp_path, separate=True)
    first = await vl.ensure_verifier_layer(
        layers, recipe, None, alias=None, context=context, **context_build(context)
    )
    second = await vl.ensure_verifier_layer(
        layers, recipe, None, alias=None, context=context, **context_build(context)
    )
    assert first.result.status == "miss" and second.result.status == "hit"
    assert len(fake.context_builds) == 1

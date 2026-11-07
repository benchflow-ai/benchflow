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


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("space", ["", " \t"])
async def test_backslash_runs_cannot_hide_external_dockerfile_inputs(
    tmp_path, monkeypatch, count, space
):
    """Docker does not join doubled escapes."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {ref}\nRUN true " + "\\" * count + space + "\n"
        "COPY --from=grader-data:current /expected /expected\n"
    )
    with pytest.raises(LayerError):
        await vl.ensure_verifier_layer(
            DockerLayers(),
            _recipe(tmp_path, separate=True),
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert fake.context_builds == []


@pytest.mark.parametrize("space", ["", " \t"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_single_dockerfile_escape_preserves_local_stage_reuse(
    tmp_path, monkeypatch, space, newline
):
    """Guards the fix at 81ffc36a with Docker's single-escape control."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = (
        f"FROM {ref} AS data\nFROM {ref}\nCOPY --fr\\{space}\n"
        "# DUMMY_LOCAL_STAGE\n"
        "om=data /expected /expected\n"
    )
    (context / "Dockerfile").write_bytes(dockerfile.replace("\n", newline).encode())
    recipe = _recipe(tmp_path, separate=True)
    first = await vl.ensure_verifier_layer(
        DockerLayers(),
        recipe,
        None,
        alias=None,
        context=context,
        **context_build(context),
    )
    second = await vl.ensure_verifier_layer(
        DockerLayers(),
        recipe,
        None,
        alias=None,
        context=context,
        **context_build(context),
    )
    assert first.result.status == "miss"
    assert second.result.status == "hit"
    assert len(fake.context_builds) == 1


@pytest.mark.parametrize(
    "suffix", ["\v", "\f", "\x85", "\u2028", "\u00a0", "\r# DUMMY_COMMENT"]
)
async def test_non_docker_whitespace_cannot_hide_a_copy(tmp_path, monkeypatch, suffix):
    """Only Docker's whitespace joins lines."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_bytes(
        (
            f"FROM {ref}\nRUN true \\{suffix}\n"
            "COPY --from=grader-data:current /expected /expected\n"
        ).encode()
    )
    with pytest.raises(LayerError):
        await vl.ensure_verifier_layer(
            DockerLayers(),
            _recipe(tmp_path, separate=True),
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert fake.context_builds == []

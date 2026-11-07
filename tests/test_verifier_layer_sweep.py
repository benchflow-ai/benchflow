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


@pytest.mark.parametrize(
    "dockerfile",
    [
        "FROM grader:current\n",
        "ARG BASE\nFROM ${BASE}\n",
        "FROM scratch\nCOPY --from=grader:current /expected /expected\n",
    ],
)
async def test_unresolved_context_images_cannot_hit_a_cached_verifier(
    tmp_path, monkeypatch, dockerfile
):
    """Fail back to ordinary builds."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(dockerfile)
    recipe = _recipe(tmp_path, separate=True)
    with pytest.raises(LayerError, match=r"(?i)(pinned|digest|context)"):
        await vl.ensure_verifier_layer(
            DockerLayers(),
            recipe,
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert fake.context_builds == []


async def test_a_changed_mutable_from_never_reuses_the_previous_grader(
    tmp_path, monkeypatch
):
    """Guards the fix at 84d5e298 with two local base identities."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM grader:current\n")
    fake.add_image("grader:current", layers=["sha256:" + "1" * 64])
    recipe = _recipe(tmp_path, separate=True)
    try:
        first = await vl.ensure_verifier_layer(
            DockerLayers(),
            recipe,
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
        fake.add_image("grader:current", layers=["sha256:" + "2" * 64])
        second = await vl.ensure_verifier_layer(
            DockerLayers(),
            recipe,
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    except LayerError:
        return  # Ordinary Docker builds are the conservative safe fallback.
    assert second.result.key != first.result.key
    assert second.result.status == "miss"


async def test_pinned_context_keys_and_builds_use_the_resolved_platform(
    tmp_path, monkeypatch
):
    """Guards the dependency/platform key closure at 84d5e298."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    image = fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(f"FROM {ref}\n")
    recipe = _recipe(tmp_path, separate=True)
    first = await vl.ensure_verifier_layer(
        DockerLayers(),
        recipe,
        None,
        alias=None,
        context=context,
        **context_build(context),
    )
    image.info["Architecture"] = "arm64"
    # An image's platform cannot choose the ordinary
    # build's platform. A foreign local variant must use ordinary grading.
    with pytest.raises(LayerError, match="platform"):
        await vl.ensure_verifier_layer(
            DockerLayers(),
            recipe,
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert first.result.status == "miss"
    builds = [
        call for call in fake.calls if call[0] == "build" and call[-1] == str(context)
    ]
    assert [call[call.index("--platform") + 1] for call in builds] == [
        "linux/amd64",
    ]


@pytest.mark.parametrize(
    "instruction",
    [
        "COPY \\\n# Grading reference data\n    --from=grader-data:current /expected /expected\n",
        "COPY \\\n\n# First comment\n# Second comment\n --from=grader-data:current /expected /expected\n",
        "COPY --fr\\\nom=grader-data:current /expected /expected\n",
        "RUN \\\n# A build-time image input\n --mount=from=grader-data:current,target=/data cat /data/expected\n",
        "ADD \\\n# A remote input\n https://example.invalid/expected /expected\n",
        "UNSUPPORTED --from=grader-data:current /expected /expected\n",
        "COPY \\\n",
        "RUN <<EOF\necho dummy\nEOF\n",
    ],
)
async def test_obscured_dockerfile_inputs_refuse_cached_verifiers(
    tmp_path, monkeypatch, instruction
):
    """Guards the fix, reopened at 79281cd7: parse or refuse every input."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(f"FROM {ref}\n{instruction}")
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


async def test_commented_continuations_of_a_local_stage_remain_reusable(
    tmp_path, monkeypatch
):
    """Guards the fix at 79281cd7 with a safe local-stage control."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {ref} AS data\nFROM {ref}\n"
        "COPY \\\n# Local stage, not an external image\n --from=data /expected /expected\n"
    )
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

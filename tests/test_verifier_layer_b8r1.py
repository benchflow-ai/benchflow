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


def _flag_spellings(name):
    # Every position within the flag name, including empty quote insertion,
    # quoted characters and escaped characters, not just fr"om"/mo"unt".
    return sorted(
        {
            "--" + name[:index] + replacement + name[index + width :]
            for index in range(len(name) + 1)
            for replacement, width in [("''", 0), ('""', 0)]
            + (
                [
                    (f"'{name[index]}'", 1),
                    (f'"{name[index]}"', 1),
                    ("\\" + name[index], 1),
                ]
                if index < len(name)
                else []
            )
        }
    )


DEPENDENCIES = [
    *(
        f"COPY {flag}=grader-data:current /expected /expected"
        for flag in _flag_spellings("from")
    ),
    *(
        f"RUN {flag}=from=grader-data:current,target=/data cat /data/expected"
        for flag in _flag_spellings("mount")
    ),
    'COPY --from="grader-data:current" /expected /expected',
    "COPY --from=grader-data:current --from=0 /expected /expected",
    "COPY --future-input=grader-data:current /expected /expected",
    "RUN --future-input=grader-data:current true",
]


@pytest.mark.parametrize("remote", [False, True], ids=["docker", "remote-docker"])
@pytest.mark.parametrize("instruction", DEPENDENCIES)
async def test_builder_options_require_a_canonical_allowlisted_form(
    tmp_path, monkeypatch, remote, instruction
):
    """Guards the fix at 2084e095 before either daemon can hit/build a layer."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(f"FROM {ref}\n{instruction}\n")
    layers = DockerLayers(env={"DOCKER_HOST": "ssh://dummy-build"} if remote else None)
    with pytest.raises(LayerError):
        await vl.ensure_verifier_layer(
            layers,
            _recipe(tmp_path, separate=True),
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert fake.context_builds == []
    assert fake.bake_files == []


@pytest.mark.parametrize("remote", [False, True], ids=["docker", "remote-docker"])
@pytest.mark.parametrize(
    "instruction",
    [
        "COPY --from=data /expected /expected",
        "COPY --from=0 --chown=0:0 --chmod=0444 --link /expected /expected",
        'COPY ["local expected", "/expected"]',
    ],
)
async def test_canonical_local_inputs_still_reuse_a_layer(
    tmp_path, monkeypatch, remote, instruction
):
    """Guards the fix at 2084e095 with local flags/JSON/command-data controls."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {ref} AS data\nFROM {ref}\n{instruction}\n"
    )
    (context / "local expected").write_text("DUMMY_REFERENCE")
    layers = DockerLayers(env={"DOCKER_HOST": "ssh://dummy-build"} if remote else None)
    recipe = _recipe(tmp_path, separate=True)
    first = await vl.ensure_verifier_layer(
        layers, recipe, None, alias=None, context=context, **context_build(context)
    )
    second = await vl.ensure_verifier_layer(
        layers, recipe, None, alias=None, context=context, **context_build(context)
    )
    assert first.result.status == "miss"
    assert second.result.status == "hit"
    assert len(fake.context_builds) == 1

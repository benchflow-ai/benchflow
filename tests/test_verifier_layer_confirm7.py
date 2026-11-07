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

UNSUPPORTED = [chr(code) for code in range(32) if code not in (9, 10, 13)] + [
    "\x7f",
    "\x85",
    "\u00a0",
    "\u1680",
    *map(chr, range(0x2000, 0x200B)),
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
]


@pytest.mark.parametrize("character", UNSUPPORTED)
@pytest.mark.parametrize("placement", ["blank", "comment", "suffix"])
async def test_docker_source_alphabet_precedes_all_normalization(
    tmp_path, monkeypatch, character, placement
):
    """Guards the fix at b743fffd for blank/comment/continuation consumers."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    middle = {
        "blank": f"RUN true \\\n{character}\n",
        "comment": f"RUN true \\\n{character}# DUMMY_COMMENT\n",
        "suffix": f"RUN true \\{character}\n",
    }[placement]
    (context / "Dockerfile").write_bytes(
        (
            f"FROM {ref}\n{middle}COPY --from=grader-data:current /expected /expected\n"
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


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_supported_docker_whitespace_preserves_local_stage_cache(
    tmp_path, monkeypatch, newline
):
    """Guards the fix at b743fffd with ordinary blank/comment/LF/CRLF controls."""
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    ref = "grader@sha256:" + "1" * 64
    fake.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_bytes(
        (
            f"FROM {ref} AS data\nFROM {ref}\nCOPY --fr\\ \t\n"
            " \t\n\t# DUMMY_COMMENT\nom=data /expected /expected\n"
        )
        .replace("\n", newline)
        .encode()
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

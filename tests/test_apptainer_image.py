"""Unit tests for Dockerfile-to-SIF image construction."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.sandbox.apptainer_image import (
    ApptainerImageBuilder,
    dockerfile_workdir,
)
from benchflow.sandbox.protocol import ImageConfig


def _context(tmp_path: Path, dockerfile: str = "FROM ubuntu:24.04\n") -> ImageConfig:
    environment = tmp_path / "environment"
    environment.mkdir()
    path = environment / "Dockerfile"
    path.write_text(dockerfile)
    return ImageConfig(dockerfile=path, context_dir=environment)


def test_dockerfile_workdir_uses_final_stage_and_build_args(tmp_path: Path) -> None:
    """Preserve the final Docker stage's WORKDIR for command execution."""
    config = _context(
        tmp_path,
        "FROM ubuntu AS build\n"
        "WORKDIR /build\n"
        "FROM ubuntu\n"
        "ARG ROOT=/workspace\n"
        "WORKDIR $ROOT\n"
        "WORKDIR task\n",
    )

    assert dockerfile_workdir(config.dockerfile) == "/workspace/task"


@pytest.mark.asyncio
async def test_builder_uses_content_addressed_cache(tmp_path: Path) -> None:
    """Reuse the cached SIF when the build context is unchanged."""
    config = _context(tmp_path)
    builder = ApptainerImageBuilder(cache_dir=tmp_path / "cache")

    async def fake_run(*args: str, timeout_sec: float):
        if args[1] == "build":
            Path(args[-2]).write_bytes(b"sif")
        return "", ""

    with patch(
        "benchflow.sandbox.apptainer_image._run",
        new=AsyncMock(side_effect=fake_run),
    ) as run:
        first = await builder.build(config)
        second = await builder.build(config)

    assert first == second
    assert Path(first.tag).read_bytes() == b"sif"
    assert run.await_count == 2
    build_args = run.await_args_list[0].args
    assert build_args[:2] == ("apptainer", "build")
    assert build_args[-1] == f"buildkit:{config.context_dir.resolve()}"


@pytest.mark.asyncio
async def test_context_change_produces_a_new_cache_key(tmp_path: Path) -> None:
    """Invalidate the cached image when the build context changes."""
    config = _context(tmp_path)
    builder = ApptainerImageBuilder(cache_dir=tmp_path / "cache")

    before = builder._image_ref(config)
    (config.context_dir / "input.txt").write_text("changed")
    after = builder._image_ref(config)

    assert before.digest != after.digest


@pytest.mark.asyncio
async def test_concurrent_builds_share_one_buildkit_invocation(tmp_path: Path) -> None:
    """Share one image build across concurrent rollouts."""
    config = _context(tmp_path)
    builder = ApptainerImageBuilder(cache_dir=tmp_path / "cache")

    async def fake_run(*args: str, timeout_sec: float):
        if args[1] == "build":
            Path(args[-2]).write_bytes(b"sif")
        return "", ""

    with patch(
        "benchflow.sandbox.apptainer_image._run",
        new=AsyncMock(side_effect=fake_run),
    ) as run:
        await asyncio.gather(builder.build(config), builder.build(config))

    build_calls = [call for call in run.await_args_list if call.args[1] == "build"]
    assert len(build_calls) == 1


@pytest.mark.asyncio
async def test_builder_rejects_unsupported_docker_build_arguments(
    tmp_path: Path,
) -> None:
    """Reject unsupported build arguments instead of silently ignoring them."""
    config = _context(tmp_path)
    config.build_args = {"VERSION": "1"}
    builder = ApptainerImageBuilder(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="build arguments"):
        await builder.build(config)


@pytest.mark.asyncio
async def test_prebuilt_sif_keeps_task_dockerfile_workdir(tmp_path: Path) -> None:
    """Use the task's Docker WORKDIR when executing a prebuilt SIF."""
    config = _context(tmp_path, "FROM ubuntu:24.04\nWORKDIR /root\n")
    image = tmp_path / "task.sif"
    image.write_bytes(b"sif")
    builder = ApptainerImageBuilder(cache_dir=tmp_path / "cache")

    resolved = await builder.resolve(
        dockerfile=config.dockerfile,
        context_dir=config.context_dir,
        prebuilt=str(image),
    )

    assert resolved.path == image
    assert resolved.workdir == "/root"

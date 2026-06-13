"""Symlink-defence regression tests for sandbox upload + staging helpers (#411).

Covers four ingestion sinks called out in the issue:

* ``DaytonaSandbox._sdk_upload_dir`` — Daytona FS upload helper
* ``ModalSandbox.upload_dir`` — Modal sandbox upload helper
* ``stage_dockerfile_deps`` / ``_stage_copy_source`` — Docker build context staging
* ``_inject_skills_into_dockerfile`` — Skills tree injection

The tests never reach a real sandbox; they assert that the symlinked secret
file is *not* presented to the upload/copy code path.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

SECRET = "SECRET_FROM_HOST_SANDBOX_LEAK"


def _planted_secret(tmp_path: Path) -> Path:
    secret = tmp_path / "host-secret.txt"
    secret.write_text(SECRET)
    return secret


# Daytona _sdk_upload_dir


@pytest.mark.asyncio
async def test_daytona_sdk_upload_dir_skips_symlinked_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk

    # ``__init__`` is what normally materializes the SDK handles
    # ``_sdk_upload_dir`` consumes (e.g. ``FileUpload``). This test bypasses
    # ``__init__`` via ``__new__``, so trigger the same lazy-load explicitly.
    _load_daytona_sdk()

    secret = _planted_secret(tmp_path)
    src = tmp_path / "src"
    (src / "ok").mkdir(parents=True)
    (src / "ok" / "real.txt").write_text("real")
    (src / "leak.txt").symlink_to(secret)
    (src / "sub").mkdir()
    (src / "sub" / "linked").symlink_to(tmp_path)  # symlinked dir

    upload_files = AsyncMock()
    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env._sandbox = SimpleNamespace(fs=SimpleNamespace(upload_files=upload_files))

    with caplog.at_level(logging.WARNING):
        await env._sdk_upload_dir(src, "/remote/dst")

    assert upload_files.await_count == 1
    sources = [u.source for u in upload_files.await_args.kwargs["files"]]
    # The symlinked file is not in the upload set.
    assert not any(str(secret) in s for s in sources)
    assert not any(s.endswith("/leak.txt") for s in sources)
    # The symlinked directory was never descended into.
    assert not any("/linked/" in s for s in sources)
    # The legitimate file is still uploaded.
    assert any(s.endswith("/ok/real.txt") for s in sources)
    assert any("leak.txt" in r.message for r in caplog.records)


# Modal upload_dir


@pytest.mark.asyncio
async def test_modal_upload_dir_skips_symlinked_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("modal")
    from benchflow.sandbox.modal_impl import ModalSandbox

    secret = _planted_secret(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_text("real")
    (src / "leak.txt").symlink_to(secret)

    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox._sandbox = SimpleNamespace()  # truthy; upload_file is mocked
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(exit_code=0))
    sandbox.upload_file = AsyncMock()
    sandbox.logger = logging.getLogger("modal-test")

    with caplog.at_level(logging.WARNING):
        await sandbox.upload_dir(src, "/remote/dst")

    uploaded_sources = [call.args[0] for call in sandbox.upload_file.await_args_list]
    assert all(str(secret) not in str(s) for s in uploaded_sources)
    assert not any(str(s).endswith("/leak.txt") for s in uploaded_sources)
    assert any(str(s).endswith("/real.txt") for s in uploaded_sources)
    assert any("leak.txt" in r.message for r in caplog.records)


# Docker build-context staging


def test_stage_copy_source_refuses_symlinked_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from benchflow.sandbox.setup import _stage_copy_source

    secret = _planted_secret(tmp_path)
    context_root = tmp_path / "ctx"
    env_dir = tmp_path / "task" / "environment"
    context_root.mkdir()
    env_dir.mkdir(parents=True)
    (context_root / "deps_link.txt").symlink_to(secret)

    with caplog.at_level(logging.WARNING):
        rewritten = _stage_copy_source("deps_link.txt", env_dir, context_root)

    # The COPY line is left unchanged — nothing was staged.
    assert rewritten == "deps_link.txt"
    assert not (env_dir / "_deps" / "deps_link.txt").exists()
    assert any("deps_link.txt" in r.message for r in caplog.records)


def test_stage_dockerfile_deps_does_not_bake_symlink_target(
    tmp_path: Path,
) -> None:
    """Full ``stage_dockerfile_deps`` path: symlinked COPY source must not leak."""
    from benchflow.sandbox.setup import stage_dockerfile_deps

    _planted_secret(tmp_path)
    context_root = tmp_path / "ctx"
    task_path = tmp_path / "task"
    env_dir = task_path / "environment"
    context_root.mkdir()
    env_dir.mkdir(parents=True)
    (context_root / "link_pkg").symlink_to(tmp_path)  # symlinked dir source
    (env_dir / "Dockerfile").write_text(
        textwrap.dedent(
            """\
            FROM scratch
            COPY link_pkg /loot
            """
        )
    )

    stage_dockerfile_deps(task_path, context_root)

    # The Dockerfile keeps the original COPY (unchanged) and no _deps/link_pkg
    # directory was created that would contain the secret.
    dockerfile_text = (env_dir / "Dockerfile").read_text()
    assert "COPY link_pkg /loot" in dockerfile_text
    leaked = env_dir / "_deps" / "link_pkg"
    if leaked.exists():
        # If a dir was somehow created, it must not contain the host secret.
        contents = "\n".join(p.read_text() for p in leaked.rglob("*") if p.is_file())
        assert SECRET not in contents


def test_stage_copytree_drops_symlinks_inside_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A symlink *inside* a directory COPY source is dropped, not followed."""
    from benchflow.sandbox.setup import _stage_copy_source

    secret = _planted_secret(tmp_path)
    context_root = tmp_path / "ctx"
    env_dir = tmp_path / "task" / "environment"
    src_dir = context_root / "pkg"
    src_dir.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    (src_dir / "real.txt").write_text("real")
    (src_dir / "evil_link.txt").symlink_to(secret)

    with caplog.at_level(logging.WARNING):
        rewritten = _stage_copy_source("pkg", env_dir, context_root)

    assert rewritten == "_deps/pkg"
    staged = env_dir / "_deps" / "pkg"
    assert (staged / "real.txt").read_text() == "real"
    # The symlinked file did not get materialised — neither as a link nor as
    # the dereferenced target.
    assert not (staged / "evil_link.txt").exists()
    # And no file inside the staged dir contains the secret.
    flat = "\n".join(p.read_text() for p in staged.rglob("*") if p.is_file())
    assert SECRET not in flat
    assert any("evil_link.txt" in r.message for r in caplog.records)


def test_inject_skills_drops_symlinks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Skills baked into the image must not include attacker-placed symlinks."""
    from benchflow.sandbox.setup import _inject_skills_into_dockerfile

    secret = _planted_secret(tmp_path)
    task_path = tmp_path / "task"
    env_dir = task_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM scratch\n")

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "real_skill.md").write_text("real")
    (skills_dir / "evil.md").symlink_to(secret)

    with caplog.at_level(logging.WARNING):
        _inject_skills_into_dockerfile(task_path, skills_dir)

    staged = env_dir / "_deps" / "skills"
    assert (staged / "real_skill.md").read_text() == "real"
    assert not (staged / "evil.md").exists()
    flat = "\n".join(p.read_text() for p in staged.rglob("*") if p.is_file())
    assert SECRET not in flat
    assert any("evil.md" in r.message for r in caplog.records)

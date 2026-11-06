"""FF-INTEG-1 (at 44993307): the review's
workspace capture runs ``python3`` (and, without one, a ``sh``/``tar``
fallback) as root. On an image whose PATH leads into a folder the agent can
write (``/app/.venv/bin``), root must not run what the agent left there.

Docker's real argv is replayed here with the image's environment; the
agent's tools leave a marker when they run, then hand over to the real one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.review import evidence
from benchflow.review.evidence import capture_workspace
from benchflow.sandbox._base import ExecResult

# Root plumbing's pinned PATH (sandbox._base.ROOT_EXEC_PATH), spelled out so
# the test also runs on a tree without it.
ROOT_EXEC_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _planted(tmp_path: Path, *tools: str) -> tuple[Path, Path]:
    """The agent's ``/app/.venv/bin``: each tool marks, then runs the real one."""
    venv_bin = tmp_path / "app" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    marks = tmp_path / "marks"
    marks.mkdir()
    for tool in tools:
        real = shutil.which(tool, path="/usr/bin:/bin")
        assert real, tool
        script = venv_bin / tool
        script.write_text(f'#!/bin/sh\n: > {marks}/{tool}\nexec {real} "$@"\n')
        script.chmod(0o755)
    return venv_bin, marks


def _system_without_python(tmp_path: Path) -> Path:
    """An image's system folders with no Python: links to this host's tools."""
    folder = tmp_path / "system-bin"
    folder.mkdir()
    for source in ("/usr/bin", "/bin"):
        for entry in Path(source).iterdir():
            name = entry.name
            if name.startswith("python") or (folder / name).exists():
                continue
            if entry.is_file() and os.access(entry, os.X_OK):
                (folder / name).symlink_to(entry)
    return folder


def _docker(image_path: str, monkeypatch, system: str | None = None):
    """A DockerSandbox whose container is this host, with *image_path* as
    the image's PATH. *system* stands in for the image's system folders
    when root's pinned PATH is used (an image without Python)."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = "ctr"
    sandbox.logger = logging.getLogger("test")
    sandbox._docker_compose_global_args = lambda: []
    sandbox._docker_compose_env = lambda: {}
    sandbox._compose_transport_env = lambda: {}
    create = asyncio.create_subprocess_exec

    async def stream(*argv, **kwargs):
        assert argv[:2] == ("docker", "cp") and argv[-1] == "-"
        source = Path(argv[-2].split(":", 1)[1])
        return await create(
            "/usr/bin/tar",
            "-cf",
            "-",
            "-C",
            str(source.parent),
            source.name,
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stream)

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        argv = list(argv)
        if argv == ["system", "dial-stdio"]:
            return ExecResult(
                return_code=0,
                stdout=(
                    "HTTP/1.1 200 OK\r\n"
                    "X-Docker-Container-Path-Stat: eyJtb2RlIjo0MjB9\r\n\r\n"
                ),
                stderr="",
            )
        if argv[0] == "cp":
            source, target = argv[1], argv[2]
            shutil.copyfile(source.split(":", 1)[1], target)
            return ExecResult(return_code=0, stdout="", stderr="")
        env = {"PATH": image_path, "HOME": "/nonexistent"}
        i = 1
        while i < len(argv):
            if argv[i] in ("-e", "--env"):
                name, _, value = argv[i + 1].partition("=")
                env[name] = value
                i += 2
            elif argv[i] in ("-u", "-w", "--user", "--workdir"):
                i += 2
            elif argv[i].startswith("-"):
                i += 1
            else:
                break
        command = argv[i + 1 :]
        if system is not None:
            command = [part.replace(ROOT_EXEC_PATH, system) for part in command]
        ran = subprocess.run(command, env=env, capture_output=True, text=True)
        result = ExecResult(
            return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr
        )
        if check and result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result

    sandbox._run_docker_container_command = run
    return sandbox


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "work"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("42\n")
    return workspace


@pytest.mark.asyncio
async def test_capture_never_runs_the_agents_python3(tmp_path, monkeypatch):
    """Case 1: ``python3`` replaced in an agent-writable folder on PATH."""
    venv_bin, marks = _planted(tmp_path, "python3")
    sandbox = _docker(f"{venv_bin}:/usr/bin:/bin", monkeypatch)
    workspace = _workspace(tmp_path)
    manifest = await capture_workspace(sandbox, str(workspace), tmp_path / "bundle")
    assert not list(marks.iterdir()), [p.name for p in marks.iterdir()]
    answer = tmp_path / "bundle" / "workspace" / "src" / "answer.txt"
    assert answer.read_text() == "42\n"
    assert manifest.entries


@pytest.mark.asyncio
async def test_tar_fallback_never_runs_the_agents_tools(tmp_path, monkeypatch):
    """Case 2: the image has no system Python, so the tar fallback runs; the
    agent's folder holds a ``python3`` and a ``tar`` of its own."""
    venv_bin, marks = _planted(tmp_path, "python3", "tar")
    system = _system_without_python(tmp_path)
    sandbox = _docker(f"{venv_bin}:{system}", monkeypatch, system=str(system))
    workspace = _workspace(tmp_path)
    fallbacks: list[str] = []
    tar_capture = evidence._capture_workspace_with_tar

    async def spy(*args, **kwargs):
        fallbacks.append(kwargs.get("python_error", ""))
        return await tar_capture(*args, **kwargs)

    monkeypatch.setattr(evidence, "_capture_workspace_with_tar", spy)
    manifest = await capture_workspace(sandbox, str(workspace), tmp_path / "bundle")
    assert not list(marks.iterdir()), [p.name for p in marks.iterdir()]
    assert fallbacks, "the tar fallback did not run"
    answer = tmp_path / "bundle" / "workspace" / "src" / "answer.txt"
    assert answer.read_text() == "42\n"
    assert manifest.entries

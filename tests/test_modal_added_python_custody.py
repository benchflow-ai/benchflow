"""Modal's own Python is root's alone in BenchFlow's Modal image.

For a Dockerfile whose base is not ``python:*``, BenchFlow's Modal start asks Modal to add Python
(``add_python="3.12"``). Modal's build installs it in /usr/local with its shared libraries group-writable:
``/usr/local/lib/libpython3.12.so``, ``libpython3.12.so.1.0`` and ``libpython3.so``, owner 0:0, mode 0775 (live,
B2R-STATUS.md). The system loader's folder is /usr/local/lib, so the loader check refused every non-root start of
such an image (``LoaderCustodyError``: plain ubuntu:24.04, the GPQA image). BenchFlow owns that build, so the build
step after Modal's makes exactly those three files root's alone (``chmod go-w``). The check is unchanged, and no file
of the task's own image is touched.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import setup as sandbox_setup

pytestmark = pytest.mark.timeout(30)


def _env(tmp_path: Path, base: str):
    (tmp_path / "Dockerfile").write_text(f"FROM {base}\nWORKDIR /app\n")
    env_config = SimpleNamespace(
        docker_image=None, gpus=0, gpu_types=None, allow_internet=True, env={}
    )
    env = sandbox_setup._create_benchflow_modal_environment_class()(
        environment_dir=tmp_path,
        environment_name=tmp_path.name,
        session_id="trial",
        rollout_paths=MagicMock(),
        task_env_config=env_config,
    )
    return env


async def _start(env, monkeypatch) -> dict:
    calls: dict = {"run_commands": []}

    class Built:
        def __init__(self, steps):
            self.steps = steps

        def run_commands(self, *commands, **kwargs):
            calls["run_commands"].extend(commands)
            return Built([*self.steps, *commands])

    class FakeImage:
        @staticmethod
        def from_dockerfile(path, **kwargs):
            calls["from_dockerfile"] = {"path": path, **kwargs}
            return Built([])

    class FakeLookup:
        @staticmethod
        async def aio(**kwargs):
            return "app"

    class FakeMakeDirectory:
        async def aio(self, *args, **kwargs):
            return None

    monkeypatch.setattr("modal.Image", FakeImage)
    monkeypatch.setattr("modal.App", SimpleNamespace(lookup=FakeLookup))
    env._create_sandbox = AsyncMock(
        return_value=SimpleNamespace(
            filesystem=SimpleNamespace(make_directory=FakeMakeDirectory())
        )
    )
    env._refuse_startup_names = AsyncMock()
    env.exec = AsyncMock()
    await env.start(force_build=False)
    calls["image"] = env._image
    return calls


@pytest.mark.asyncio
@pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("modal", "tenacity")),
    reason="modal/tenacity not installed",
)
async def test_the_image_modal_adds_python_to_gets_its_libraries_made_root_s_alone(
    tmp_path, monkeypatch
):
    """(dogfood C3 part 5, X7 fork check) Plain ubuntu:24.04: Modal adds Python 3.12, and the next build
    step is the custody step for exactly its three loader files.

    Mutant: no custody step (the LoaderCustodyError at start)."""
    monkeypatch.delenv("BENCHFLOW_MODAL_ADD_PYTHON", raising=False)
    calls = await _start(_env(tmp_path, "ubuntu:24.04"), monkeypatch)
    assert calls["from_dockerfile"]["add_python"] == "3.12", calls
    assert len(calls["run_commands"]) == 1, calls
    assert calls["run_commands"] == [
        sandbox_setup._modal_added_python_custody("3.12")
    ], calls
    assert calls["image"].steps == calls["run_commands"], calls


@pytest.mark.asyncio
@pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("modal", "tenacity")),
    reason="modal/tenacity not installed",
)
async def test_an_image_modal_adds_no_python_to_gets_no_custody_step(
    tmp_path, monkeypatch
):
    """python:3.12-slim: BenchFlow asks Modal for no Python, and the task's own /usr/local is left as the
    task built it (the loader check still judges it).

    Mutant: the step run on every image."""
    monkeypatch.delenv("BENCHFLOW_MODAL_ADD_PYTHON", raising=False)
    calls = await _start(_env(tmp_path, "python:3.12-slim"), monkeypatch)
    assert calls["from_dockerfile"]["add_python"] is None, calls
    assert calls["run_commands"] == [], calls


def _lib(tmp_path: Path) -> Path:
    lib = tmp_path / "usr-local-lib"
    lib.mkdir()
    for name in (
        "libpython3.12.so",
        "libpython3.12.so.1.0",
        "libpython3.so",
        "libfoo.so",
    ):
        (lib / name).write_bytes(b"\x7fELF")
        (lib / name).chmod(0o775)
    (lib / "python3.12").mkdir()
    (lib / "python3.12" / "os.py").write_text("")
    (lib / "python3.12" / "os.py").chmod(0o664)
    (lib / "elsewhere.so").write_bytes(b"\x7fELF")
    (lib / "elsewhere.so").chmod(0o775)
    (lib / "libpython3.11.so").symlink_to(lib / "elsewhere.so")
    return lib


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_custody_step_makes_exactly_modal_s_loader_files_root_s_alone(tmp_path):
    """The step, in a real sh: Modal's three libpython files lose group and other write; another library,
    Python's own tree and anything a link names are left as they were.

    Mutant: a wider chmod (the folder, -R, or every .so)."""
    lib = _lib(tmp_path)
    folder = _mode(lib)
    command = sandbox_setup._modal_added_python_custody("3.12", lib=str(lib))
    out = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    for name in ("libpython3.12.so", "libpython3.12.so.1.0", "libpython3.so"):
        assert _mode(lib / name) == 0o755, name
    assert _mode(lib / "libfoo.so") == 0o775
    assert _mode(lib / "python3.12" / "os.py") == 0o664
    assert _mode(lib / "elsewhere.so") == 0o775
    assert _mode(lib) == folder


def test_the_custody_step_follows_no_link_and_passes_without_the_files(tmp_path):
    """A libpython name that is a link is not followed (its target is not Modal's file), and a build whose
    Python left none of the three files still passes (control)."""
    lib = _lib(tmp_path)
    command = sandbox_setup._modal_added_python_custody("3.11", lib=str(lib))
    out = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    assert _mode(lib / "elsewhere.so") == 0o775
    empty = tmp_path / "empty"
    empty.mkdir()
    out = subprocess.run(
        [
            "/bin/sh",
            "-c",
            sandbox_setup._modal_added_python_custody("3.12", lib=str(empty)),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr

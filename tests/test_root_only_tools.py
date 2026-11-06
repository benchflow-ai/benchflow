"""Root never runs a BenchFlow tool that someone else can change (PY8 U1, U2b).

BenchFlow installs uv, a managed CPython and the model proxy's cached venv
under ``/opt/benchflow`` and runs them as root, sometimes after the agent has
had a turn (a second role's proxy). A workspace at or above one of those
folders (``WORKDIR /opt``) is given to the agent by ``chown -R``, and an
agent that owns a folder can rename it and put anything in its place: the
old install ran the planted uv as root to read its version, and a cached venv
inside the workspace ran the agent's litellm as root with the next role's key.

Here a world-writable folder (mode 0777, no sticky bit) stands for a folder
the agent owns: these tests run as an ordinary user, who cannot give a
folder to another uid. The shell runs for real against fakes in tmp_path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from benchflow import _uv_pin
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.sandbox import lockdown
from benchflow.sandbox.env_layers import BaseImage, LayerResult
from tests._root_stand_in import run_sh

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


@pytest.fixture(autouse=True)
def _umask_022():
    """Folders the tests make are writable by their owner alone (a sandbox's
    root umask), whatever the machine's umask (002 on some build hosts)."""
    old = os.umask(0o022)
    yield
    os.umask(old)


class _Result:
    def __init__(self, return_code: int, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _LocalSandbox:
    """Runs each command with sh in tmp_path; records them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec=None, user=None, **_kw) -> _Result:
        self.commands.append(command)
        done = run_sh(
            command,
            env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        return _Result(done.returncode, done.stdout, done.stderr)


def _planted_uv(folder: Path, witness: Path) -> None:
    folder.mkdir(parents=True)
    for name in ("uv", "uvx"):
        tool = folder / name
        tool.write_text(f"#!/bin/sh\ntouch {witness}\necho 'uv {_uv_pin.UV_VERSION}'\n")
        tool.chmod(0o755)


@needs_sh
def test_a_uv_under_a_folder_others_can_change_is_never_run(tmp_path):
    """The agent owns /opt (here: /opt is 0777), replaces
    /opt/benchflow/uv/0.13.4/uv with a program that prints the pinned
    version; the next root install used to run it to read that version."""
    opt = tmp_path / "opt"
    opt.mkdir()
    opt.chmod(0o777)
    uv_dir = opt / "benchflow" / "uv" / _uv_pin.UV_VERSION
    witness = tmp_path / "planted-uv-ran"
    _planted_uv(uv_dir, witness)

    done = run_sh(
        _uv_pin.uv_install_command(uv_dir=str(uv_dir)),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode != 0
    assert not witness.exists()
    assert f"BenchFlow's uv folder ({uv_dir}) is not root's alone" in done.stderr


@needs_sh
def test_the_same_uv_under_root_only_folders_is_reused(tmp_path):
    opt = tmp_path / "opt"
    uv_dir = opt / "benchflow" / "uv" / _uv_pin.UV_VERSION
    witness = tmp_path / "uv-ran"
    _planted_uv(uv_dir, witness)

    done = run_sh(
        _uv_pin.uv_install_command(uv_dir=str(uv_dir)),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    assert witness.exists()


@needs_sh
@pytest.mark.parametrize("bad", ["writable", "link"])
def test_the_managed_python_folder_is_checked_before_uv_installs_into_it(
    tmp_path, monkeypatch, bad
):
    root = tmp_path / "opt" / "benchflow"
    root.mkdir(parents=True)
    python_dir = root / "python"
    if bad == "writable":
        python_dir.mkdir()
        python_dir.chmod(0o777)
    else:
        # Into a folder others may write.
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere").chmod(0o777)
        python_dir.symlink_to(tmp_path / "elsewhere")
    monkeypatch.setattr(_uv_pin, "PYTHON_INSTALL_DIR", str(python_dir))
    monkeypatch.setattr(_uv_pin, "UV_CACHE_DIR", str(root / "cache" / "uv"))
    log = tmp_path / "uv.log"

    done = run_sh(
        f"UV='echo uv >> {log}; :'\n"
        + _uv_pin.managed_python_install_command(uv='eval "$UV"'),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode != 0
    assert not log.exists()
    assert (
        f"BenchFlow's managed Python ({python_dir}) is not root's alone" in done.stderr
    )
    assert "could not install managed CPython" not in done.stderr


def _layer() -> LayerResult:
    base = BaseImage(
        ref="task:latest",
        image_id="sha256:" + "1" * 64,
        digest="sha256:" + "2" * 64,
        arch="amd64",
        libc="glibc",
    )
    key = "d" * 64
    return LayerResult(
        kind="proxy",
        key=key,
        tag=f"benchflow-layer:proxy-{key}",
        image=f"benchflow-layer:proxy-{key}",
        status="hit",
        base=base,
        seconds=0.1,
    )


def _layer_venv(tmp_path: Path, monkeypatch, *, open_bin: bool) -> _LocalSandbox:
    from benchflow.providers.proxy_requirements import ProxyInstall

    venv = tmp_path / "opt" / "benchflow" / "litellm-proxy" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "litellm").write_text("#!/bin/sh\n")
    (venv / "bin" / "litellm").chmod(0o755)
    # As uv makes it: bin/python is a link to the managed interpreter, safe
    # inside a folder only root can change.
    interpreter = tmp_path / "opt" / "benchflow" / "python" / "python3.12"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    (venv / "bin" / "python").symlink_to(interpreter)
    if open_bin:
        (venv / "bin").chmod(0o777)
    stamp = tmp_path / "opt" / "benchflow" / "layers" / "litellm-proxy.stamp"
    stamp.parent.mkdir(parents=True)
    monkeypatch.setattr(runtime_mod, "PROXY_LAYER_VENV", str(venv))
    monkeypatch.setattr(runtime_mod, "PROXY_LAYER_STAMP", str(stamp))
    monkeypatch.setattr(runtime_mod, "PYTHON_INSTALL_DIR", str(interpreter.parent))
    install = ProxyInstall(group="litellm-proxy")
    stamp.write_text(runtime_mod.proxy_layer_recipe(install).stamp + "\n")
    sandbox = _LocalSandbox(tmp_path)
    sandbox.proxy_layer = _layer()
    return sandbox


@needs_sh
@pytest.mark.asyncio
async def test_a_cached_venv_others_can_change_is_not_used(tmp_path, monkeypatch):
    from benchflow.providers.proxy_requirements import ProxyInstall

    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=True)

    python, record = await runtime_mod._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )

    assert python is None
    assert record is not None and record["install"] == "in trial"
    assert "is not root's alone" in record["reason"]


@needs_sh
@pytest.mark.asyncio
async def test_a_root_only_cached_venv_is_used(tmp_path, monkeypatch):
    from benchflow.providers.proxy_requirements import ProxyInstall

    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=False)

    python, record = await runtime_mod._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )

    assert record == {"install": "from layer"}
    assert python is not None and python.endswith("/venv/bin/python")


class _RecordingEnv:
    supports_system_shell = True

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_kw) -> _Result:
        self.commands.append(command)
        return _Result(0)
        return _Result(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace",
    [
        "/opt/benchflow",
        "/opt/benchflow/litellm-proxy",
        "/opt/benchflow/python/cpython-3.12.15-linux-x86_64-gnu/bin",
        "/./opt/benchflow/x/..",
        "//opt/./benchflow",
    ],
)
async def test_a_workspace_inside_benchflows_folder_is_refused(workspace):
    env = _RecordingEnv()

    with pytest.raises(ValueError, match=r"inside BenchFlow's runtime tree"):
        await lockdown.setup_sandbox_user(env, "agent", workspace)

    assert env.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["/opt/benchflow-app", "/opt", "/app"])
async def test_other_workspaces_are_set_up_as_before(workspace):
    env = _RecordingEnv()

    await lockdown.setup_sandbox_user(env, "agent", workspace, guard_daemon=False)

    if workspace == "/opt":
        # /opt holds the runtime tree: it stays root's and sticky, and every
        # entry but /opt/benchflow is granted (PT's workspace grant).
        assert any(
            "chmod 1777 /opt" in c and "! -path /opt/benchflow" in c
            for c in env.commands
        )
    else:
        assert any(f"chown -R agent:agent {workspace}" in c for c in env.commands)

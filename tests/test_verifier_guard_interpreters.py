"""The pytest plugin guard must load in every interpreter a verifier can use.

The pytest plugin guard is injected as ``-p <guard>`` but was importable only
through PYTHONPATH. ``python3 -I -m pytest`` ignores PYTHONPATH and test.sh files
may replace it (``PYTHONPATH=/app pytest``); pytest then aborts on the missing
module, exits 1, and a correct solution scores 0 with no error recorded.

These tests use a real interpreter whose site-packages the test owns, real
``/bin/sh`` for the container-side installer, and real pytest subprocesses.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import _install_pytest_plugin_guard

HOSTILE_PLUGIN = (
    "def pytest_collection_modifyitems(items):\n"
    "    for item in items:\n"
    "        item.obj = lambda: None\n"
)


class ShellEnv:
    """Run sandbox commands with the real POSIX shell on this machine."""

    async def exec(self, command, **kwargs):
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )


@pytest.fixture
def interpreter(tmp_path):
    """A venv interpreter that can import this environment's pytest under ``-I``.

    Its site-packages belongs to the test, so the installer never writes into a
    shared Python. The ``.pth`` line is processed even in isolated mode.
    """
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    python = venv / "bin" / "python3"
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )
    (purelib / "outer_pytest.pth").write_text(
        str(Path(pytest.__file__).resolve().parent.parent) + "\n"
    )
    return SimpleNamespace(bin=venv / "bin", python=python, purelib=purelib)


def run_pytest(python, workspace, *args, isolated=False, **env):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "PYTHON"))
    }
    environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", **env)
    flags = ["-I"] if isolated else []
    return subprocess.run(
        [str(python), *flags, "-m", "pytest", "-q", *args],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.asyncio
async def test_guard_loads_under_isolated_mode_and_replaced_pythonpath(
    tmp_path, monkeypatch, interpreter
):
    """Guards the fix for the silent-zero regression introduced by the pytest plugin guard.

    A passing verifier run as ``python3 -I -m pytest`` or with test.sh's own
    PYTHONPATH must exit 0 with the guard active, and the guard must still
    refuse a planted plugin in both forms.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_pass.py").write_text("def test_pass():\n    assert True\n")
    (workspace / "test_fail.py").write_text("def test_fail():\n    assert False\n")
    (workspace / "agent_plugin.py").write_text(HOSTILE_PLUGIN)
    (interpreter.purelib / "planted_plugin.py").write_text(HOSTILE_PLUGIN)
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))

    directory, flags, copied = await _install_pytest_plugin_guard(
        ShellEnv(), "agent", str(workspace), "", verifier_path=str(interpreter.bin)
    )
    # Every name for the one interpreter reports its copy.
    assert str(interpreter.python) in copied
    guarded = {
        "PYTEST_ADDOPTS": f"-p no:cacheprovider {flags}",
        "PYTHONPATH": directory,
    }

    isolated = run_pytest(
        interpreter.python, workspace, "test_pass.py", isolated=True, **guarded
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
    replaced = run_pytest(
        interpreter.python,
        workspace,
        "test_pass.py",
        **{**guarded, "PYTHONPATH": str(tmp_path / "elsewhere")},
    )
    assert replaced.returncode == 0, replaced.stdout + replaced.stderr

    # Both planted hooks really do turn a failing verifier into a pass unguarded.
    for isolated_mode, plugin, pythonpath in (
        (True, "planted_plugin", ""),
        (False, "agent_plugin", str(workspace)),
    ):
        args = ("-p", plugin, "test_fail.py")
        hostile = run_pytest(
            interpreter.python,
            workspace,
            *args,
            isolated=isolated_mode,
            PYTHONPATH=pythonpath,
        )
        assert hostile.returncode == 0, hostile.stdout + hostile.stderr
        refused = run_pytest(
            interpreter.python,
            workspace,
            *args,
            isolated=isolated_mode,
            **{**guarded, "PYTHONPATH": pythonpath},
        )
        assert refused.returncode != 0
        assert "Verifier plugin trust rejected: " + plugin in refused.stderr


@pytest.mark.asyncio
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read-only mode")
async def test_guard_that_cannot_be_installed_is_a_hardening_error(
    tmp_path, monkeypatch, interpreter
):
    """Guards the pytest plugin guard fix: an unloadable guard errors instead of scoring 0."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    interpreter.purelib.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="pytest plugin guard"):
            await _install_pytest_plugin_guard(
                ShellEnv(),
                "agent",
                str(tmp_path),
                "",
                verifier_path=str(interpreter.bin),
            )
    finally:
        interpreter.purelib.chmod(0o755)


@pytest.mark.asyncio
async def test_guard_installer_ignores_non_python3_candidates(tmp_path, monkeypatch):
    """Guards the pytest plugin guard fix: a Python 2 or helper script on PATH is not fatal."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "python3.99"):
        script = bin_dir / name
        script.write_text("#!/bin/sh\necho 'Unknown option: -I' >&2\nexit 2\n")
        script.chmod(0o755)
    (bin_dir / "pytest").write_text("#!/bin/sh\nexit 0\n")
    directory, flags, copied = await _install_pytest_plugin_guard(
        ShellEnv(), None, None, "-p ctrf", verifier_path=str(bin_dir)
    )
    name = flags.split()[1]
    assert flags == f"-p {name} -p ctrf"
    assert (Path(directory) / f"{name}.py").is_file()
    # No Python took a copy, so hardening keeps the guard on PYTHONPATH.
    assert copied == ()


@pytest.mark.asyncio
async def test_guard_reaches_pytest_script_interpreter_off_path(
    tmp_path, monkeypatch, interpreter
):
    """Guards the pytest plugin guard fix: a pytest script's own Python gets a copy too.

    ``/usr/local/bin/pytest`` may run a venv Python whose bin dir is not on the
    verifier PATH; ``PYTHONPATH=/app pytest`` must still import the guard.
    """
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "pytest"
    script.write_text(
        f"#!{interpreter.python}\n"
        "import sys\nfrom pytest import console_main\nsys.exit(console_main())\n"
    )
    script.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_pass.py").write_text("def test_pass():\n    assert True\n")
    _, flags, _ = await _install_pytest_plugin_guard(
        ShellEnv(), "agent", str(workspace), "", verifier_path=str(bin_dir)
    )
    result = subprocess.run(
        [str(script), "-q", "test_pass.py"],
        cwd=workspace,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(workspace),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": f"-p no:cacheprovider {flags}",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

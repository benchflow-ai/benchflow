"""PT R11-1 (Astra): fe895b04 lost image-added Python search folders."""

import json
import os
import shlex
import subprocess
import sys

import pytest

from benchflow.sandbox import lockdown


@pytest.mark.parametrize("folder", ["project", "project with spaces", "project,comma"])
@pytest.mark.parametrize("hook", ["sitecustomize.py", "usercustomize.py", "agent.pth"])
def test_cleanup_uses_captured_image_paths(tmp_path, folder, hook):
    """Guards fe895b04: isolated cleanup must visit trusted .pth additions."""
    directory = tmp_path / folder
    directory.mkdir()
    planted = directory / hook
    planted.write_text("# dormant dummy marker\n")
    ordinary = directory / "solution.py"
    ordinary.write_text("answer = 42\n")
    record = tmp_path / "paths.json"
    record.write_text(json.dumps({"path": [str(directory)], "user_site": []}))
    script = lockdown._CLEANUP_SITE_SCRIPT.replace(
        "/opt/benchflow/rt/IMAGE_PYTHON_PATHS.json", str(record)
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        # System folders only: with the test's venv first on PATH the
        # clean-up counts that venv as a prefix and sweeps its .pth files.
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert not planted.exists()
    assert ordinary.read_text() == "answer = 42\n"


def test_discovery_does_not_import_from_search_path_data(tmp_path):
    """Guards fe895b04: plugin metadata and startup hooks are only data."""
    marker = tmp_path / "marker"
    (tmp_path / "sitecustomize.py").write_text(
        f"open({str(marker)!r}, 'w').write('root')\n"
    )
    (tmp_path / "email.py").write_text(f"open({str(marker)!r}, 'w').write('import')\n")
    dist = tmp_path / "honest-1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: honest\nVersion: 1\n")
    (dist / "entry_points.txt").write_text("[pytest11]\nhonest = honest_plugin\n")
    (tmp_path / "honest_plugin.py").write_text(
        "raise AssertionError('never import a candidate')\n"
    )
    command = lockdown._discover_pytest_plugins_cmd(
        (), pythonpath=str(tmp_path), ownership=False
    )
    done = subprocess.run(
        ["sh", "-c", command],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert not marker.exists()
    assert "honest" in json.loads(done.stdout)["plugins"]


def test_guard_install_does_not_process_site_hooks(tmp_path):
    """Guards the R11-4 sibling: the candidate probe must suppress site too."""
    # A fresh venv stands for an honest image interpreter with writable site
    # files. This test only starts its Python to reproduce root preparation.
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(tmp_path / "venv")],
        check=True,
    )
    python = tmp_path / "venv/bin/python"
    site = next((tmp_path / "venv/lib").glob("python*/site-packages"))
    marker = tmp_path / "marker"
    (site / "agent.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('hook')\n"
    )
    # Exercise the real candidate version probe and installation function,
    # without scanning the test host's other interpreters.
    template = lockdown._INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE
    functions = template[: template.index("shebang_interpreter()")]
    source = tmp_path / "guard.py"
    source.write_text("# harmless guard\n")
    script = (
        functions.replace(
            "__SCRIPT__", shlex.quote(lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT)
        )
        .replace("__GUARD__", shlex.quote(str(source)))
        .replace("__NAME__", "_pt_guard")
    )
    # Probe only: unlike the installer this never writes into a system stdlib.
    probe = script[script.index('    "$1"') : script.index(" || return 0")]
    subprocess.run(["sh", "-c", probe, "sh", str(python)], check=True)
    assert not marker.exists()

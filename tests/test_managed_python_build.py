"""Guard FF-install-C4: cd4d492d reused any build of the pinned patch."""

import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

from benchflow import _uv_pin


def test_an_unproven_same_patch_install_needs_the_catalog_download(
    tmp_path, monkeypatch
):
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv
    root = tmp_path / "python"
    tree = root / f"cpython-{_uv_pin.PYTHON_VERSION}-linux-x86_64-gnu"
    (tree / "bin").mkdir(parents=True)
    (tree / "lib/python3.12").mkdir(parents=True)
    config = next(Path(sysconfig.get_path("stdlib")).glob("_sysconfigdata_*.py"))
    shutil.copyfile(config, tree / "lib/python3.12" / config.name)
    # uv's normal satisfaction path accepts this installation by its key,
    # without authenticating its build or executing its interpreter.
    (tree / "BUILD").write_text("19990101\n")
    interpreter = tree / "bin/python3.12"
    interpreter.write_text("#!/bin/sh\nexit 19\n")
    interpreter.chmod(0o755)
    monkeypatch.setattr(_uv_pin, "PYTHON_INSTALL_DIR", str(root))
    command = _uv_pin.managed_python_install_command(
        uv=f"{shlex.quote(uv)} --offline --no-config --no-cache"
    )
    result = subprocess.run(
        ["sh", "-c", command],
        env={"PATH": "/usr/bin:/bin", "UV_PYTHON_INSTALL_DIR": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "an unproven build was accepted without a download"
    assert "could not install managed CPython" in result.stderr

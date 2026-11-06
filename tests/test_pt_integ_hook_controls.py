"""Controls for the hook clean-up's input set on PT's protected route.

PT daec118b recorded the pre-agent search path with ``python -I``, so the
image's PYTHONPATH folders were never visited, while the verifier's own
Python starts with them (PT-R16 review B). The clean-up's input set now
includes PYTHONPATH: an image with ``ENV
PYTHONPATH=/app`` gets that folder's hooks removed, whichever way the value
reaches the clean-up (the route's inert copy, the preparation script's copy,
or the shell's own value), with no pre-agent record at all. These inject the
value; tests/test_pt_integ_r2_container.py takes it through a real Docker
backend.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown

# The preparation route names the interpreter V3 judged before these steps
# (lockdown._prep_on_path); without a kit the clean-up runs it.
_JUDGED = (
    "__bf_root_python="
    + (shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") or "python3")
    + "; "
)

pytestmark = pytest.mark.skipif(
    shutil.which("python3") is None, reason="needs a python3 for the fallback"
)


def _plant(folder: Path, marker: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    payload = f"open({str(marker)!r}, 'w').write('ran')\n"
    for name in ("sitecustomize.py", "usercustomize.py"):
        (folder / name).write_text(payload)
    (folder / "injection.pth").write_text(f"import os; {payload}")
    honest = folder / "app_module.py"
    honest.write_text("VALUE = 'honest'\n")
    return honest


@pytest.mark.parametrize(
    "carrier",
    ["_BENCHFLOW_IMAGE_PYTHONPATH", "__bf_hook_pythonpath", "PYTHONPATH"],
)
def test_image_pythonpath_app_folder_is_cleaned(tmp_path, carrier):
    app = tmp_path / "app"
    marker = tmp_path / "hook-ran"
    honest = _plant(app, marker)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        carrier: str(app),
    }
    command = lockdown._cleanup_site_cmd(str(tmp_path / "workspace"))
    # The step's shell, as the preparation runner starts it: the carrier is
    # shell data there (``__bf_hook_pythonpath`` is a shell variable).
    script = (
        f"{carrier}={str(app)!r}; " + command
        if carrier == "__bf_hook_pythonpath"
        else command
    )
    if carrier == "__bf_hook_pythonpath":
        env.pop(carrier)
    result = subprocess.run(
        ["/bin/sh", "-c", _JUDGED + script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not (app / "sitecustomize.py").exists()
    assert not (app / "usercustomize.py").exists()
    # /app is no site folder: a .pth there is data and stays (PT-R16's
    # HOOK_RULE, closing PT-INTEG-A-1).
    assert (app / "injection.pth").exists()
    assert honest.read_text() == "VALUE = 'honest'\n"
    assert not marker.exists(), "a hook ran during the clean-up"


def test_relative_image_pythonpath_names_the_original_folder(tmp_path):
    """A relative entry names the folder the clean-up started in (CONF-5)."""
    app = tmp_path / "app"
    marker = tmp_path / "hook-ran"
    _plant(app, marker)
    result = subprocess.run(
        ["/bin/sh", "-c", _JUDGED + lockdown._cleanup_site_cmd(None)],
        # System folders only: the test's own venv must never be the
        # clean-up's python3 (it would sweep that venv's .pth files).
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "app"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not (app / "sitecustomize.py").exists()
    assert not marker.exists()


def _venv_site(workspace: Path) -> Path:
    venv = workspace / ".venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return site


def _with_record(command: str, record: Path) -> str:
    return command.replace(
        "/opt/runtime/rt/IMAGE_PYTHON_PATHS.json", str(record)
    ).replace("/opt/runtime/rt/IMAGE_PYTHON_PATHS.jsonl", str(record) + "l")


@pytest.mark.parametrize("recorded", [True, False])
def test_v8_walk_takes_only_workspace_venvs_without_a_record(tmp_path, recorded):
    """PT-R16 review B: a recorded workspace venv is the record's (its honest
    .pth stays, a hook the agent added goes); V8's walk keeps the others."""
    import hashlib
    import json

    workspace = tmp_path / "workspace"
    site = _venv_site(workspace)
    honest = site / "__editable__.proj.pth"
    honest.write_text(f"{workspace / 'src'}\n")
    record = tmp_path / "IMAGE_PYTHON_PATHS.json"
    if recorded:
        digest = hashlib.sha256(honest.read_bytes()).hexdigest()
        record.write_text(
            json.dumps(
                {
                    "path": [str(site)],
                    "user_site": [],
                    "hooks": {str(site): {honest.name: digest}},
                }
            )
        )
    (site / "sitecustomize.py").write_text("raise SystemExit('agent')\n")
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp_path)}

    def run(command: str) -> None:
        result = subprocess.run(
            ["/bin/sh", "-c", _JUDGED + _with_record(command, record)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    run(lockdown._purge_workspace_python_hooks_cmd(str(workspace)))
    # V8's walk leaves a recorded venv to the record-based clean-up.
    assert (site / "sitecustomize.py").exists() is recorded
    run(lockdown._cleanup_site_cmd(str(workspace)))
    assert not (site / "sitecustomize.py").exists()
    if recorded:
        assert honest.read_text() == f"{workspace / 'src'}\n"

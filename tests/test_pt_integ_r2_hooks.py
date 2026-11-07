"""PT-integ round 2: the hook clean-up after PT-R16's join.

- PT-INTEG-B-2 (High): D913 call 6 takes a recorded workspace venv away
  from V8's walk, so the record-based clean-up must judge it with the full
  rule. At 403505e5 that rule knew only ``sitecustomize.py``,
  ``usercustomize.py`` and ``.pth``, so a ``sitecustomize`` package the
  agent put in a recorded venv was left by both.
- PT-INTEG-B-4 (Medium): D913 call 1 (a venv's own prefix by Python's
  ``pyvenv.cfg`` rule) read root's route PATH, not the image PATH that
  test.sh's ``python3`` comes from.

The record is the real one (``CAPTURE`` run by the venv's interpreter, as
the kit does); hooks are dummy markers.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._image_python_paths import (
    CAPTURE,
    IMAGE_PYTHON_PATHS,
    IMAGE_PYTHON_PATHS_EACH,
)
from benchflow.sandbox import lockdown

MARK = "raise SystemExit('agent hook ran')\n"


def _venv(where: Path) -> tuple[Path, Path]:
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(where)], check=True
    )
    site = next((where / "lib").glob("python3*/site-packages"))
    return where / "bin" / "python3", site


def _local(command: str, record: Path) -> str:
    return command.replace(IMAGE_PYTHON_PATHS_EACH, str(record) + "l").replace(
        IMAGE_PYTHON_PATHS, str(record)
    )


def _run(command: str, env: dict[str, str], cwd: Path) -> None:
    # The preparation route names the interpreter V3 judged;
    # without a kit the clean-up runs it.
    judged = shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") or "python3"
    done = subprocess.run(
        ["/bin/sh", "-c", f"__bf_root_python={shlex.quote(judged)}; {command}"],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr


def test_a_recorded_workspace_venv_loses_a_sitecustomize_package(tmp_path):
    """PT-INTEG-B-2, fails on 403505e5: the venv is recorded, so V8's walk
    leaves it to the record-based clean-up (D913 call 6), which judged only
    ``sitecustomize.py``; the agent's ``sitecustomize/`` package stayed and
    the venv's Python imported it as root. PT-R16's HOOK_RULE removes it;
    the honest editable .pth, recorded, stays."""
    workspace = tmp_path / "workspace"
    python, site = _venv(workspace / ".venv")
    honest = site / "__editable__.proj.pth"
    honest.write_text(f"{workspace / 'src'}\n")
    record = tmp_path / "paths.json"
    subprocess.run(
        [str(python), "-I", "-B", "-c", CAPTURE, str(record), "/"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd="/",
        check=True,
    )
    package = site / "sitecustomize"
    package.mkdir()
    (package / "__init__.py").write_text(MARK)
    # Control: the venv's interpreter imports the agent's package.
    ran = subprocess.run([str(python), "-c", "pass"], capture_output=True, text=True)
    assert "agent hook ran" in ran.stderr
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    _run(
        _local(lockdown._purge_workspace_python_hooks_cmd(str(workspace)), record),
        env,
        tmp_path,
    )
    _run(_local(lockdown._cleanup_site_cmd(str(workspace)), record), env, tmp_path)
    assert not package.exists()
    assert honest.read_text() == f"{workspace / 'src'}\n"
    ran = subprocess.run([str(python), "-c", "pass"], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize("where", ["outside", "workspace"])
def test_the_venv_rule_reads_the_image_path(tmp_path, where):
    """PT-INTEG-B-4, fails on 403505e5: test.sh's python3 is the first on
    the image PATH (a venv here, no pre-agent record), but the clean-up
    looked only at root's route PATH, so the venv's site folder was never
    visited and the agent's sitecustomize.py there stayed. A venv inside the
    workspace keeps its .pth files for V8's walk (V8-R1-B-2)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base = workspace if where == "workspace" else tmp_path / "opt"
    python, site = _venv(base / ".venv")
    hook = site / "sitecustomize.py"
    hook.write_text(MARK)
    pth = site / "agent.pth"
    pth.write_text(f"{tmp_path / 'elsewhere'}\n")
    env = {
        # Root's route PATH: system folders only (no venv on it).
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        # The image PATH root's route keeps as data.
        "_BENCHFLOW_IMAGE_PATH": f"{python.parent}:/usr/bin:/bin",
    }
    record = tmp_path / "no-record.json"
    _run(_local(lockdown._cleanup_site_cmd(str(workspace)), record), env, tmp_path)
    assert not hook.exists()
    assert pth.exists() is (where == "workspace")


def test_the_search_inputs_never_reach_argv():
    """Fails on 403505e5 (raw_args put the three
    values after the script): the interpreter's argv is the working folder
    and the workspace; the values travel in its environment."""
    command = lockdown._cleanup_site_cmd("/app")
    after_script = command.rsplit(" -c ", 1)[1].rsplit("' ", 1)[1]
    assert after_script.startswith('"$PWD" /app )'), after_script
    assert "_BF_HOOK_PYTHONPATH=" in command.split(" -c ", 1)[0]

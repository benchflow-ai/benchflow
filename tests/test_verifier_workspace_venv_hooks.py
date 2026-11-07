"""A workspace venv's own startup hooks do not run before the plugin guard.

The verifier may run a venv in the workspace (``/app/.venv/bin/python -m
pytest``). With a copied interpreter (``python -m venv --copies``) the venv
survives the external-link purge, and its site-packages is not on the system
``python3``'s sys.path, which is all step 6's hook cleanup reads. A ``.pth``
the agent adds there runs at interpreter start, before pytest imports any
``-p`` plugin, so it could patch pytest to pass without the guard ever
loading. V8's freeze step (V8-RC1-B-1) removes such hooks from
every workspace site directory unless the root-only pre-agent copy holds the
same file; V2 relies on it, and this test skips until that step is merged.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown

pytestmark = pytest.mark.skipif(
    not hasattr(lockdown, "_purge_workspace_python_hooks_cmd"),
    reason="needs V8's workspace venv hook purge (py8/freeze-fixes)",
)

# What the agent adds: it would patch pytest's entry point to exit 0; here it
# only leaves a marker that proves it ran.
HOOK = "import pathlib; pathlib.Path({marker!r}).write_text('ran')\n"


def _copied_venv(venv: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--copies", "--without-pip", str(venv)],
        check=True,
    )
    python = venv / "bin" / "python"
    assert python.is_file() and not python.is_symlink()
    return python


def _site(venv: Path) -> Path:
    return next((venv / "lib").glob("python*/site-packages"))


def _starts_clean(python: Path, markers: list[Path]) -> bool:
    result = subprocess.run(
        [str(python), "-c", "pass"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return not any(marker.exists() for marker in markers)


def test_agent_startup_hooks_in_a_copied_workspace_venv_never_run(tmp_path):
    """The agent's ``.pth`` and sitecustomize in a copied venv."""
    workspace = tmp_path / "app"
    baseline = tmp_path / "testbed_verify"
    venv = workspace / ".venv"
    python = _copied_venv(venv)
    site = _site(venv)
    (site / "image.pth").write_text("import sys\n")
    shutil.copytree(workspace, baseline, symlinks=True)
    baseline.chmod(0o700)
    # The venv's site directory is not one the system python3 reads.
    system_path = subprocess.run(
        ["python3", "-c", "import sys; print('\\n'.join(sys.path))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert str(site) not in system_path
    markers = [tmp_path / name for name in ("pth-ran", "sitecustomize-ran")]
    (site / "attack.pth").write_text(HOOK.format(marker=str(markers[0])))
    (site / "sitecustomize.py").write_text(HOOK.format(marker=str(markers[1])))
    # Without the purge both hooks run when the venv's Python starts.
    assert not _starts_clean(python, markers)
    for marker in markers:
        marker.unlink()

    command = lockdown._purge_workspace_python_hooks_cmd(str(workspace))
    # The preparation route names the interpreter V3 judged.
    judged = "__bf_root_python=" + shlex.quote(shutil.which("python3") or "python3")
    subprocess.run(
        [
            "/bin/sh",
            "-c",
            judged + "; " + command.replace("/testbed_verify", str(baseline)),
        ],
        check=True,
    )

    assert _starts_clean(python, markers)
    assert not (site / "attack.pth").exists()
    assert not (site / "sitecustomize.py").exists()
    # The image's own hook, the same bytes as the pre-agent copy, stays.
    assert (site / "image.pth").read_text() == "import sys\n"

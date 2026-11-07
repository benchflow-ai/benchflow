"""The verifier ignores per-user git and pip config under its HOME.

The verifier runs as root with HOME=/root, which is often the agent's
workspace, so ~/.gitconfig and ~/.config/pip/pip.conf there are whatever the
agent left. VERIFIER_ENV switches both off. Each test plants a harmless
setting, shows the tool picks it up without the verifier env (the control),
then shows it does not with the verifier env.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox.lockdown import VERIFIER_ENV


def _base_env(home: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home)}


def _with_verifier_config(env: dict[str, str]) -> dict[str, str]:
    return {
        **env,
        "GIT_CONFIG_GLOBAL": VERIFIER_ENV["GIT_CONFIG_GLOBAL"],
        "PIP_CONFIG_FILE": VERIFIER_ENV["PIP_CONFIG_FILE"],
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize("location", [".gitconfig", ".config/git/config"])
def test_git_ignores_per_user_config(tmp_path, location):
    config = tmp_path / location
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[user]\n\tname = left-in-home\n")
    argv = ["git", "config", "--get", "user.name"]

    control = subprocess.run(
        argv, env=_base_env(tmp_path), capture_output=True, text=True, cwd=tmp_path
    )
    assert control.stdout.strip() == "left-in-home"

    hardened = subprocess.run(
        argv,
        env=_with_verifier_config(_base_env(tmp_path)),
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert hardened.stdout.strip() == ""


def _python_with_pip(scratch: Path) -> str | None:
    # uv-managed venvs ship without pip; the runner's system Python has it.
    # PY8 main sync: a listed interpreter may not exist on the host (no
    # /usr/local/bin/python3 on the build VMs), and none may have pip: then a
    # fresh stdlib venv brings its own through ensurepip, offline.
    for candidate in (
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ):
        if (
            candidate
            and os.path.exists(candidate)
            and subprocess.run(
                [candidate, "-m", "pip", "--version"], capture_output=True
            ).returncode
            == 0
        ):
            return candidate
    venv = scratch / "pip-venv"
    made = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)], capture_output=True
    )
    python = venv / "bin" / "python"
    return str(python) if made.returncode == 0 and python.exists() else None


@pytest.mark.parametrize("location", [".config/pip/pip.conf", ".pip/pip.conf"])
def test_pip_ignores_per_user_config(tmp_path, location):
    python = _python_with_pip(tmp_path / "scratch")
    if python is None:
        pytest.skip("no interpreter with pip on PATH")
    config = tmp_path / location
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[global]\ntimeout = 61\n")
    argv = [python, "-m", "pip", "config", "list"]
    env = {**_base_env(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / ".config")}

    control = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert "global.timeout='61'" in control.stdout

    hardened = subprocess.run(
        argv, env=_with_verifier_config(env), capture_output=True, text=True
    )
    assert "timeout" not in hardened.stdout


def _wheelhouse(directory: Path, version: str) -> Path:
    """A folder holding one pure-Python wheel of ``bfprobe``, built by hand."""
    import zipfile

    directory.mkdir(parents=True)
    dist = f"bfprobe-{version}.dist-info"
    files = {
        "bfprobe/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: bfprobe\nVersion: {version}\n",
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = "".join(f"{name},,\n" for name in files) + f"{dist}/RECORD,,\n"
    with zipfile.ZipFile(directory / f"bfprobe-{version}-py3-none-any.whl", "w") as w:
        for name, text in files.items():
            w.writestr(name, text)
        w.writestr(f"{dist}/RECORD", record)
    return directory


def _venv_with_pip(scratch: Path, name: str = "venv") -> str | None:
    """A fresh venv's python with pip: its ``sys.prefix`` is ours to configure.

    ``BENCHFLOW_TEST_PIP_WHEEL`` (a pip wheel on disk) installs that pip into
    it offline, to run the differential under a given pip (25.2).
    """
    venv = scratch / name
    made = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)], capture_output=True
    )
    python = venv / "bin" / "python"
    if made.returncode or not python.exists():
        return None
    wheel = os.environ.get("BENCHFLOW_TEST_PIP_WHEEL")
    if wheel:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "--no-index", wheel],
            check=True,
            capture_output=True,
        )
    check = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True)
    return str(python) if check.returncode == 0 else None


def _download(argv: list[str], env: dict, dest: Path) -> list[str]:
    """What ``<argv> download bfprobe`` fetches, or [] when it finds none."""
    subprocess.run(
        [
            *argv,
            "download",
            "--no-deps",
            "--retries",
            "0",
            "--timeout",
            "5",
            "-d",
            str(dest),
            "bfprobe",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    return sorted(p.name for p in dest.glob("*.whl")) if dest.exists() else []


def _verifier(home: Path) -> dict:
    """What the verifier gives every pip (PIP_NO_INDEX keeps the test offline)."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "PIP_CONFIG_FILE": VERIFIER_ENV["PIP_CONFIG_FILE"],
        "PIP_NO_INDEX": "1",
    }


_INVOCATIONS = [
    "pip --python",
    "python -m pip",
    "a venv test.sh creates",
    "an interpreter named by path",
    "pyvenv.cfg beside the interpreter",
    "a configured python = redirect",
]


@pytest.mark.parametrize("invocation", _INVOCATIONS)
def test_every_pip_the_verifier_runs_reads_no_configuration(tmp_path, invocation):
    """Each way review found pip choosing configuration.

    Review found pip reading files hardening never checked: a
    venv test.sh creates over an agent's pip.conf, `pip --python` into another
    interpreter, `python -m pip`, an interpreter named by path, a
    `pyvenv.cfg` beside the interpreter; b89 added a `python =` setting in a
    configuration file. Each would find the agent's wheel
    through a pip.conf it planted; with the verifier's `PIP_CONFIG_FILE` none
    of them reads any configuration.
    """
    a = _venv_with_pip(tmp_path / "scratch", "venv-a")
    c = _venv_with_pip(tmp_path / "scratch", "venv-c")
    if a is None or c is None:
        pytest.skip("no venv with pip")
    agent = _wheelhouse(tmp_path / "agent-wheels", "6.6")
    planted = f"[global]\nfind-links = {agent}\n"
    venv_a, venv_c = Path(a).parent.parent, Path(c).parent.parent
    home = tmp_path / "home"
    home.mkdir()
    env = _verifier(home)
    if invocation == "pip --python":
        (venv_c / "pip.conf").write_text(planted)
        argv = [str(venv_a / "bin/pip"), "--python", c]
    elif invocation == "python -m pip":
        (venv_a / "pip.conf").write_text(planted)
        argv = [a, "-m", "pip"]
    elif invocation == "a venv test.sh creates":
        grader = tmp_path / "runtime" / "grader"  # the agent wrote here first
        grader.mkdir(parents=True)
        (grader / "pip.conf").write_text(planted)
        made = subprocess.run([a, "-m", "venv", str(grader)], env=env)
        assert made.returncode == 0 and (grader / "pip.conf").exists()
        argv = [str(grader / "bin/python"), "-m", "pip"]
    elif invocation == "an interpreter named by path":
        (venv_c / "pip.conf").write_text(planted)
        argv = [c, "-m", "pip"]
    elif invocation == "a configured python = redirect":
        # b89: configuration naming another interpreter.
        (venv_a / "pip.conf").write_text(f"[global]\npython = {c}\n")
        (venv_c / "pip.conf").write_text(planted)
        argv = [str(venv_a / "bin/pip")]
    else:
        (venv_c / "bin" / "pyvenv.cfg").write_text((venv_c / "pyvenv.cfg").read_text())
        (venv_c / "bin" / "pip.conf").write_text(planted)
        (venv_c / "pip.conf").write_text(planted)
        argv = [str(venv_c / "bin/pip")]
    assert _download(argv, env, tmp_path / "got") == []
    # The control: the same pip without the verifier's PIP_CONFIG_FILE reads it.
    loose = {k: v for k, v in env.items() if k != "PIP_CONFIG_FILE"}
    assert _download(argv, loose, tmp_path / "got-loose") == [
        "bfprobe-6.6-py3-none-any.whl"
    ]

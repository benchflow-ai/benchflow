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


def _python_with_pip() -> str | None:
    # uv-managed venvs ship without pip; the runner's system Python has it.
    for candidate in (
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ):
        if (
            candidate
            and subprocess.run(
                [candidate, "-m", "pip", "--version"], capture_output=True
            ).returncode
            == 0
        ):
            return candidate
    return None


@pytest.mark.parametrize("location", [".config/pip/pip.conf", ".pip/pip.conf"])
def test_pip_ignores_per_user_config(tmp_path, location):
    python = _python_with_pip()
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

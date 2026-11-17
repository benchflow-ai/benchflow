"""A pytest that tox runs from a conda environment imports the plugin guard (GATE1N's SWT sphinx-9602).

SWT-bench's sphinx tasks run ``conda activate testbed`` and then ``tox
-epy39``. tox passes no PYTHONPATH into the environments it builds, and the
testbed Python was not on the verifier PATH, so it took no guard copy, while
PYTEST_ADDOPTS (which their tox.ini sets from the environment) named the
guard. Every pytest tox ran stopped with ``Error importing plugin
"<guard>"``, base and gold alike. R's pristine re-judgment, which never runs
tox, then trusted the plugins and blamed the trial.

A conda installation on the verifier PATH now gives each of its
environments' Pythons a copy in its standard library folder, which every venv
made from it (tox's) keeps on sys.path. The container test runs BenchFlow's
own guard install, as hardening runs it, in an image with a conda layout, then
tox 4.16 (SWT's pin) from the environment's Python, offline.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from tests.exec_diet_fakes import docker_available

TOX_IMAGE = "benchflow-test-tox416-py312:1"
_DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir --disable-pip-version-check tox==4.16.0 pytest==8.3.3 \\
 && tox --version
"""
VERIFIER_PATH = "/opt/fc/bin:/usr/local/bin:/usr/bin:/bin"


@pytest.fixture(scope="module")
def tox_image():
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    present = subprocess.run(
        ["docker", "image", "inspect", TOX_IMAGE], capture_output=True
    )
    if present.returncode != 0:
        built = subprocess.run(
            ["docker", "build", "-q", "-t", TOX_IMAGE, "-"],
            input=_DOCKERFILE,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if built.returncode != 0:
            pytest.skip(f"cannot build {TOX_IMAGE}: {built.stderr[-300:]}")
    return TOX_IMAGE


def _guard_install(monkeypatch) -> tuple[str, str]:
    """Hardening's guard install command for *VERIFIER_PATH*, and the guard's
    name."""
    commands = []

    async def read(env, command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(lockdown, "_checked_read", read)
    directory, *_ = asyncio.run(
        lockdown._install_pytest_plugin_guard(
            SimpleNamespace(),
            "agent",
            "/testbed",
            "",
            verifier_path=VERIFIER_PATH,
            verifier_uid="0",
        )
    )
    (command,) = commands
    return command, os.path.basename(directory)


# A conda installation: its base bin (conda-meta beside it) on the verifier
# PATH, and an environment that is a Python installation of its own (here a
# copy of the image's, with tox and pytest), off the PATH.
_CONDA = r"""
set -e
mkdir -p /opt/fc/bin /opt/fc/conda-meta /opt/fc/envs
printf '#!/bin/sh\nexit 0\n' > /opt/fc/bin/conda
chmod 755 /opt/fc/bin/conda
cp -a /usr/local /opt/fc/envs/e1
/opt/fc/envs/e1/bin/python3 -c 'import sys; assert sys.prefix == "/opt/fc/envs/e1", sys.prefix'
"""
# SWT sphinx's tox.ini shape: [testenv] passes PYTEST_ADDOPTS, not PYTHONPATH,
# and runs ``python -m pytest {posargs}``; the environment reads the conda
# environment's site-packages instead of installing from PyPI.
_TOX_INI = r"""[tox]
envlist = py
skipsdist = true

[testenv]
sitepackages = true
skip_install = true
passenv =
    PYTEST_ADDOPTS
commands = python -m pytest -p no:cacheprovider {posargs}
"""
_RUN = r"""
mkdir -p /proj
printf '%s' "$TOX_INI" > /proj/tox.ini
printf 'def test_answer():\n    assert 6 * 7 == 42\n' > /proj/test_answer.py
cd /proj
export PYTEST_ADDOPTS="-p $GUARD" PYTHONPATH="/$GUARD"
set +e
/opt/fc/envs/e1/bin/python3 -m tox -e py -- test_answer.py 2>&1
echo "exit=$?"
ls /opt/fc/envs/e1/lib/python3.12/ | grep -c "^$GUARD.py$" | sed 's/^/copies=/'
for f in /$GUARD/.markers/*; do
  [ -e "$f" ] || continue
  echo "marker ${f##*.}"
done
"""


def test_a_pytest_tox_runs_from_a_conda_environment_loads_the_guard(
    tmp_path, monkeypatch, tox_image
):
    """The environment's Python took a copy, tox's venv reads it from its
    standard library, the guard registers and leaves its end-of-session record, and
    the honest test passes."""
    command, name = _guard_install(monkeypatch)
    (tmp_path / "harden.sh").write_text(command)
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "-e",
            "GUARD=" + name,
            "-e",
            "TOX_INI=" + _TOX_INI,
            "-e",
            "PATH=" + VERIFIER_PATH,
            "-v",
            f"{tmp_path}:/src:ro",
            tox_image,
            "sh",
            "-c",
            _CONDA + "sh /src/harden.sh\n" + _RUN,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = done.stdout
    assert "exit=" in out, (out, done.stderr)
    status = int(out.split("exit=", 1)[1].split()[0])
    assert "copies=1" in out, out
    assert f'Error importing plugin "{name}"' not in out, out
    assert status == 0, out
    assert "1 passed" in out, out
    assert "marker finished" in out, out

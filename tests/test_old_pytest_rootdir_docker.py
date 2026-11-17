"""A pytest without --rootdir (before 3.5), in a real container.

SWE-bench's astropy 1.3 and scikit-learn 0.20-0.22 images run pytest 3.x on
Python 3.6. Strict passed ``--rootdir=<workspace>`` to every pytest (with
``-c``, whose /dev rootdir broke node ids on newer pytest); pytest 3.3.1 has
no such option and stopped at its command line, before any test, so the
honest answer scored 0 (astropy-7166, R334). Strict now leaves it out when
every pytest the image installs predates 3.5 (the image manifest reads their
versions before the turn). Under ``-c`` pytest 3.3.1 takes the rootdir from
the tests' folder and reports each test by the path test.sh named (what the
reward parsers read). The full trial is astropy-7166's real run.
"""

from __future__ import annotations

import shlex
import subprocess
import uuid

import pytest

from benchflow.sandbox.lockdown import ImagePytestIni, _build_pytest_addopts
from tests.exec_diet_fakes import docker_available

pytestmark = [pytest.mark.deterministic]

DOCKERFILE = """\
FROM python:3.6-slim
RUN pip install --no-cache-dir -q pytest==3.3.1 attrs==19.1.0
WORKDIR /app
RUN mkdir -p /tests /app/tests /cfg && printf '%s\\n' 'def test_fix():' '    assert True' \\
    > /app/tests/test_repo.py && printf '%s\\n' 'def check_fix():' '    assert True' \\
    > /app/tests/check_repo.py && printf '%s\\n' '[pytest]' \\
    'python_files = check_*.py' 'python_functions = check_*' > /cfg/0123456789abcdef.ini
"""


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    context = tmp_path_factory.mktemp("old-pytest")
    (context / "Dockerfile").write_text(DOCKERFILE)
    tag = "benchflow-test-old-pytest:" + uuid.uuid4().hex[:8]
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"could not build the pytest 3.3.1 image: {built.stderr[-300:]}")
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def _pytest(image: str, addopts: str, target: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none", "-w", "/app",
            "-e", f"PYTEST_ADDOPTS={addopts}", "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            image, "python", "-m", "pytest", target, "-v",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )  # fmt: skip


def test_rootdir_stops_an_old_pytest_at_its_command_line(image):
    """What scored astropy-7166 0: the addopts with --rootdir."""
    addopts = _build_pytest_addopts("/app", "", verifier_confcutdir="/tests")
    assert "--rootdir=/app" in shlex.split(addopts)

    ran = _pytest(image, addopts, "tests/test_repo.py")

    # argparse's own exit status: pytest 3.x stops with 2 here, not 4.
    assert ran.returncode == 2, ran.stdout + ran.stderr
    assert "unrecognized arguments: --rootdir=/app" in ran.stderr


def test_without_rootdir_an_old_pytest_runs_under_dev_null(image):
    addopts = _build_pytest_addopts(
        "/app", "", verifier_confcutdir="/tests", rootdir=False
    )

    ran = _pytest(image, addopts, "tests/test_repo.py")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "tests/test_repo.py::test_fix PASSED" in ran.stdout


def test_without_rootdir_an_old_pytest_reads_the_images_copy(image):
    """BenchFlow's copy of the image's ini, outside the workspace, alone
    collects check_*: an old pytest reads it with -c and no --rootdir."""
    addopts = _build_pytest_addopts(
        "/app",
        "",
        verifier_confcutdir="/tests",
        ini=ImagePytestIni(path="/cfg/0123456789abcdef.ini"),
        rootdir=False,
    )
    assert not any(a.startswith("--rootdir") for a in shlex.split(addopts))

    ran = _pytest(image, addopts, "tests/check_repo.py")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "tests/check_repo.py::check_fix PASSED" in ran.stdout

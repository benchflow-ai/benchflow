"""Full trials on Docker: a guard verdict needs evidence (GUARD-LOAD).

Terminal-Bench 2's qemu-startup and qemu-alpine-ssh run their tests with
``uvx ... pytest --ctrf`` after ``apt-get install curl``. Debian's mirror
now answers 404 for that curl, so ``uvx`` is missing, pytest never starts
and test.sh writes 0. On 57d28d656 BenchFlow refused both the oracle and
nop ("no pytest of this run registered the guard"), where other engines
score 0. The first fixture is that test.sh with the install failing on
purpose, in an image without Python (as the qemu images); the second ends
its test.sh in a check that is not pytest, so the oracle scores 1 and nop 0.

The third keeps what the guard is for: a ctrf plugin the agent leaves where
test.sh's pytest would import it, which passes every test, still gives the
failing solution nothing.

Each trial builds a small image (network at build time, and PyPI for the
protected runtime's pytest-json-ctrf). Skipped without a Docker daemon.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.exec_diet_fakes import docker_available

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    # The real protected runtime, as a trial builds it.
    pytest.mark.verifier_runtime,
]


@pytest.fixture(autouse=True)
def plugin_releases_from_pypi(monkeypatch) -> None:
    """The runtime checks pytest-json-ctrf's release on PyPI, as a real run
    does; the unit tests' offline table (tests/conftest.py) would refuse it."""
    monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)


TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "guard-evidence"

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory = "2G"
storage = "10G"
"""

NO_PYTHON_IMAGE = """FROM debian:bookworm-slim
WORKDIR /app
"""

# qemu-startup's test.sh, with its failing install made deterministic.
INSTALL_FAILS = """#!/bin/bash
false || echo "E: Unable to fetch some archives" >&2
curl -LsSf https://astral.sh/uv/0.9.5/install.sh 2>/dev/null | sh
source $HOME/.local/bin/env
uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  -w pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

# A verifier that checks without pytest when uv is not there.
CHECKER = """#!/bin/bash
if command -v uvx >/dev/null 2>&1; then
  uvx -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 \\
    pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
else
  [ "$(cat /app/answer.txt 2>/dev/null)" = 42 ]
fi
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

TEST_OUTPUTS = """def test_answer():
    assert open("/app/answer.txt").read().strip() == "42"
"""

ANSWER = "#!/bin/bash\necho 42 > /app/answer.txt\n"

PYTEST_IMAGE = """FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.4.1
WORKDIR /app
ENV PYTHONPATH=/app
"""

REACHED = """#!/bin/bash
cd /tests
python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

# No answer. A ctrf in the workspace (on the image's PYTHONPATH) that passes
# every test and leaves a mark when imported.
PLANT = """#!/bin/bash
mkdir -p /app/ctrf_planted-0.3.5.dist-info
printf 'Metadata-Version: 2.1\\nName: ctrf_planted\\n' \\
  > /app/ctrf_planted-0.3.5.dist-info/METADATA
printf '[pytest11]\\nctrf = ctrf_planted\\n' \\
  > /app/ctrf_planted-0.3.5.dist-info/entry_points.txt
cat > /app/ctrf_planted.py <<'PY'
open("/logs/verifier/planted-imported", "w").close()
def pytest_addoption(parser):
    parser.addoption("--ctrf", action="store", default=None)
def pytest_collection_modifyitems(items):
    for item in items:
        item.obj = lambda: None
PY
"""


def _task(root: Path, name: str, image: str, test_sh: str, solve: str) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(image)
    (task / "tests" / "test.sh").write_text(test_sh)
    (task / "tests" / "test_outputs.py").write_text(TEST_OUTPUTS)
    (task / "solution" / "solve.sh").write_text(solve)
    for script in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        script.chmod(0o755)
    return task


async def _trial(task: Path, agent: str, jobs: Path):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent=agent,
            environment="docker",
            jobs_dir=jobs,
            job_name=f"{task.name}-{agent}",
            rollout_name=f"{task.name}__{agent}",
        )
    ).run()


@pytest.mark.parametrize(
    ("shape", "image", "test_sh", "oracle"),
    [
        ("install-fails", NO_PYTHON_IMAGE, INSTALL_FAILS, 0.0),
        ("checker", NO_PYTHON_IMAGE, CHECKER, 1.0),
    ],
)
async def test_a_test_sh_whose_pytest_never_starts_is_scored(
    tmp_path, shape, image, test_sh, oracle
):
    """Fails on 57d28d656: both agents end in PluginGuardLoadError."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    task = _task(tmp_path / "tasks", f"guard-{shape}", image, test_sh, ANSWER)
    for agent, reward in (("oracle", oracle), ("nop", 0.0)):
        result = await _trial(task, agent, tmp_path / "jobs")
        assert result.error is None, (agent, result.error)
        assert result.rewards == {"reward": reward}, (agent, result.rewards)


async def test_a_planted_plugin_in_a_pytest_the_guard_reaches_changes_nothing(
    tmp_path,
):
    """The answer is missing and the agent's ctrf would pass the test: the
    guard registers in test.sh's pytest and loads the runtime's ctrf, so the
    planted one is never imported and the trial is not a pass."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    task = _task(tmp_path / "tasks", "guard-reached", PYTEST_IMAGE, REACHED, PLANT)
    result = await _trial(task, "oracle", tmp_path / "jobs")
    assert result.rewards != {"reward": 1.0}, (result.rewards, result.error)
    if result.error is not None:
        assert "PluginGuardLoadError" in result.error, result.error
    assert list((tmp_path / "jobs").rglob("test-stdout.txt")), "test.sh never ran"
    assert not list((tmp_path / "jobs").rglob("planted-imported"))

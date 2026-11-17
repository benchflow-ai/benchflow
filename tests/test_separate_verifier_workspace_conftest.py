"""A repository's own conftest.py under a separate verifier, as real Docker trials.

SWE-bench's images ship the repository with its conftest.py files, and its
test.sh runs the repository's tests, which need their fixtures (sphinx's
``rootdir``, astropy's). The shared verifier keeps a workspace conftest.py the
agent left byte for byte as the pre-agent copy has it (#1171, #1177). A
separate verifier sandbox (``--separate-verifier``, or ``--integrity strict``,
which implies it) has no pre-agent copy: its conftest sweep removes every
workspace conftest.py, the image's own included, so the oracle scores 0
("fixture 'app' not found"). EN-708 pins that here as a strict xfail that
raises only for that cause, so it turns into a failure the moment the fix
lands (remove the marks then); any other failure fails the test as usual.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.exec_diet_fakes import PYTHON, docker_available

# The real protected-verifier-runtime check, as on Daytona (conftest stubs it
# out otherwise).
pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]

SEPARATE_DELETES = "separate verifier deletes the image's workspace conftest"

# The image ships the repository: a fixture in tests/conftest.py and a test
# that uses it, as sphinx's tests/conftest.py gives ``rootdir``.
DOCKERFILE = f"""\
FROM {PYTHON}
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
RUN pip install --no-cache-dir -q pytest==8.4.1
RUN mkdir -p /app/tests && printf '%s\\n' \\
    'import pytest' '' '' '@pytest.fixture' 'def app():' '    return "repo"' \\
    > /app/tests/conftest.py && printf '%s\\n' \\
    'from pathlib import Path' '' '' 'def test_fix(app):' \\
    '    assert app == "repo"' \\
    '    assert Path("/app/fixed.txt").read_text().strip() == "fixed"' \\
    > /app/tests/test_repo.py
"""

SOLVE = "echo fixed > /app/fixed.txt\n"

TEST = """\
#!/bin/bash
cd /app
python3 -m pytest tests/test_repo.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


class WorkspaceConftestDeleted(AssertionError):
    """The oracle scored 0 because the repository's conftest.py was gone."""


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "repo-conftest"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Write /app/fixed.txt.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + SOLVE)
    (task / "tests" / "test.sh").write_text(TEST)
    return task


async def _oracle(task: Path, **isolation):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=task.parent / "jobs",
            job_name=f"{task.name}-oracle",
            rollout_name=f"{task.name}__oracle",
            **isolation,
        )
    ).run()


def _scored_one(task: Path, result) -> None:
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    if result.rewards == {"reward": 0.0}:
        out = "".join(
            p.read_text(errors="replace")
            for p in (task.parent / "jobs").rglob("verifier/test-stdout.txt")
        )
        if "fixture 'app' not found" in out:
            raise WorkspaceConftestDeleted(SEPARATE_DELETES)
    assert result.rewards == {"reward": 1.0}, result.rewards


async def test_shared_verifier_keeps_the_repositorys_conftest(tmp_path):
    """Control: the agent ran as ``agent``, the file is the pre-agent copy's."""
    task = _task(tmp_path)
    _scored_one(task, await _oracle(task))


@pytest.mark.xfail(
    raises=WorkspaceConftestDeleted, strict=True, reason=SEPARATE_DELETES
)
async def test_separate_verifier_keeps_the_repositorys_conftest(tmp_path):
    task = _task(tmp_path)
    _scored_one(task, await _oracle(task, separate_verifier=True))


@pytest.mark.xfail(
    raises=WorkspaceConftestDeleted, strict=True, reason=SEPARATE_DELETES
)
async def test_integrity_strict_keeps_the_repositorys_conftest(tmp_path):
    task = _task(tmp_path)
    _scored_one(task, await _oracle(task, integrity="strict"))

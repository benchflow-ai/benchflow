"""A repository's own conftest.py under a separate verifier, as real Docker trials.

SWE-bench's images ship the repository with its conftest.py files, and its
test.sh runs the repository's tests, which need their fixtures (sphinx's
``rootdir``, astropy's). The shared verifier keeps a workspace conftest.py the
agent left byte for byte as the pre-agent copy has it (#1171, #1177). A
separate verifier sandbox (``--separate-verifier``, or ``--integrity strict``,
which implies it) has no pre-agent copy: it now copies the image's own
workspace conftest.py files before the transfer and keeps a workspace file
byte for byte that copy's. Until then its sweep removed them all
and the oracle scored 0 ("fixture 'app' not found"), which these trials
pinned as strict xfails; that cause still fails them with its own name.
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


def _task(
    tmp_path: Path,
    *,
    dockerfile: str = DOCKERFILE,
    solve: str = SOLVE,
    test: str = TEST,
) -> Path:
    task = tmp_path / "repo-conftest"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Write /app/fixed.txt.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + solve)
    (task / "tests" / "test.sh").write_text(test)
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


async def test_separate_verifier_keeps_the_repositorys_conftest(tmp_path):
    task = _task(tmp_path)
    _scored_one(task, await _oracle(task, separate_verifier=True))


async def test_integrity_strict_keeps_the_repositorys_conftest(tmp_path):
    task = _task(tmp_path)
    _scored_one(task, await _oracle(task, integrity="strict"))


# -- the image's own conftest.py outside the workspace ((A)) -----------

# featurebench: WORKDIR /, so the workspace is not the repository (/testbed).
ROOT_WORKDIR = DOCKERFILE.replace("WORKDIR /app", "WORKDIR /").replace(
    "/app/", "/testbed/"
)
ROOT_SOLVE = "echo fixed > /testbed/fixed.txt\n"
ROOT_TEST = TEST.replace("cd /app", "cd /testbed")

# SWT-bench: test.sh rebuilds the repository's tests from a copy the image
# keeps outside the workspace (/root/tmp_repo).
COPY_OUTSIDE = (
    DOCKERFILE
    + "RUN mkdir -p /root/tmp_repo && cp -r /app/tests /root/tmp_repo/tests\n"
)
COPY_TEST = TEST.replace(
    "cd /app\n", "rm -rf /app/tests && cp -r /root/tmp_repo/tests /app/tests\ncd /app\n"
)

# A solution that edits the repository's conftest.py so that the test passes
# without the fix: the edited file is removed, and the trial scores 0.
PLANTED_SOLVE = (
    "printf '%s\\n' 'import pathlib' 'import pytest' '' '' '@pytest.fixture' "
    "'def app():' '    pathlib.Path(\"/app/fixed.txt\").write_text(\"fixed\")' "
    "'    return \"repo\"' > /app/tests/conftest.py\n"
)


async def test_the_images_conftest_outside_the_workspace_stays(tmp_path):
    """featurebench's shape: the oracle scored 0 when the sweep removed
    /testbed's conftest.py because the workspace was /root."""
    task = _task(tmp_path, dockerfile=ROOT_WORKDIR, solve=ROOT_SOLVE, test=ROOT_TEST)
    _scored_one(task, await _oracle(task))


async def test_the_images_copy_of_the_repository_keeps_its_conftest(tmp_path):
    """SWT-bench's shape: test.sh copies the tests back from /root/tmp_repo,
    whose conftest.py the sweep emptied."""
    task = _task(tmp_path, dockerfile=COPY_OUTSIDE, test=COPY_TEST)
    _scored_one(task, await _oracle(task))


async def test_an_edited_conftest_is_still_removed_and_scores_0(tmp_path):
    """In strict: the default runs test.sh as written and scores what it
    gives (test_faithful_default.test_a_flag_never_changes_the_reward)."""
    task = _task(tmp_path, solve=PLANTED_SOLVE)
    result = await _oracle(task, integrity="strict")
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 0.0}

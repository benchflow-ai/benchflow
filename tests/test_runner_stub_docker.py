"""REL r4 H1: a task that runs ``$BENCHFLOW_PYTEST`` only from a file in its
image (its tests folder and ``[verifier] env`` never name it) gets no
protected runner. ``$BENCHFLOW_PYTEST`` then names a root-owned stub that
prints why and exits 4, so the trial scores what test.sh reports (usually
0) with the message in test.sh's output, instead of running plain pytest (a
``${BENCHFLOW_PYTEST:-...}`` fallback). The stub never makes a trial
unscored, so the agent's own code cannot use it to escape a 0
(FIX-49 review M2).

Real Docker trials, as in ``test_acc_run2_docker``: the oracle runs solve.sh
as root and the sandbox user is ``agent``.
"""

from __future__ import annotations

import pytest

from tests.exec_diet_fakes import docker_available
from tests.test_acc_run2_docker import (
    ANSWER_TEST_PY,
    CTRF_TEST,
    WITH_PYTEST,
    _run,
    _scored,
    _task,
)

pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


# The image's own grader calls the runner; test.sh only calls the grader.
def _image_grader(runner: str) -> str:
    return (
        WITH_PYTEST
        + "RUN printf '%s\\n' '#!/bin/bash' 'cd /app' "
        + f"'{runner} /tests/test_outputs.py -rA' > /usr/local/bin/grade "
        + "&& chmod 755 /usr/local/bin/grade\n"
    )


IMAGE_TEST = """\
#!/bin/bash
/usr/local/bin/grade
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def _test_output(task) -> str:
    """Everything the verifier kept of test.sh's output for the oracle trial."""
    folder = task.parent / "jobs" / f"{task.name}-oracle" / f"{task.name}__oracle"
    return "".join(
        p.read_text(errors="replace")
        for p in sorted((folder / "verifier").rglob("*"))
        if p.is_file()
    )


STUB_ERROR = "benchflow-pytest: error: no protected test runner was prepared"


@pytest.mark.parametrize(
    "runner",
    [
        '"$BENCHFLOW_PYTEST"',
        "${BENCHFLOW_PYTEST:-python3 -m pytest}",
        # L1: the stub run as another user acts the same (the grader's own
        # arguments go after --, to the user's shell).
        'su -s /bin/sh -c "$BENCHFLOW_PYTEST /tests/test_outputs.py" nobody --',
    ],
    ids=["plain", "with-fallback", "other-user"],
)
async def test_an_image_only_runner_call_scores_what_test_sh_reports(tmp_path, runner):
    """At 6c7426dcb the plain and fallback calls ended unscored (a reason
    blaming the task), and the other-user call scored 0 with "Permission
    denied" on the stub's marker. Now each scores 0, test.sh's verdict on a
    grader that exited 4, with the stub's message in test.sh's output."""
    task = _task(
        tmp_path,
        "runner-stub",
        dockerfile=_image_grader(runner),
        solve="echo ok > /app/answer.txt\n",
        test=IMAGE_TEST,
        tests_py=ANSWER_TEST_PY,
    )
    _scored(await _run(task, "oracle"), 0.0)
    output = _test_output(task)
    assert STUB_ERROR in output, output
    assert "Permission denied" not in output


FORGE_SOLVE = """cat > /app/check.py <<'PY'
import os, subprocess
p = os.environ.get("BENCHFLOW_PYTEST")
if p:
    subprocess.run([p], stderr=subprocess.DEVNULL)
raise SystemExit(1)  # a failing solution
PY
"""
FORGE_TEST = """#!/bin/bash
cd /app
if python3 check.py; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""


async def test_the_agents_code_cannot_make_a_failing_trial_unscored(tmp_path):
    """The review's h1-forge: test.sh never names the runner, and the
    "agent" (the oracle here) leaves a failing check.py that first runs
    ``$BENCHFLOW_PYTEST``. At 6c7426dcb the trial ended unscored
    (PluginGuardLoadError); now it scores the 0 it earned."""
    task = _task(
        tmp_path,
        "h1-forge",
        dockerfile=(
            "FROM python:3.12-slim\nWORKDIR /app\n"
            "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
        ),
        solve=FORGE_SOLVE,
        test=FORGE_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


async def test_a_task_that_never_calls_the_runner_scores_as_before(tmp_path):
    """Control: the same image and test.sh with plain pytest in the grader."""
    task = _task(
        tmp_path,
        "runner-none",
        dockerfile=_image_grader("python3 -m pytest"),
        solve="echo ok > /app/answer.txt\n",
        test=IMAGE_TEST,
        tests_py=ANSWER_TEST_PY,
    )
    _scored(await _run(task, "oracle"), 1.0)
    _scored(await _run(task, "nop"), 0.0)


async def test_a_test_sh_that_names_the_runner_still_scores(tmp_path):
    """Control: test.sh names $BENCHFLOW_PYTEST, so the real runner runs."""
    task = _task(
        tmp_path,
        "runner-named",
        dockerfile=WITH_PYTEST,
        solve="echo ok > /app/answer.txt\n",
        test=CTRF_TEST.replace("python3 -m pytest", '"$BENCHFLOW_PYTEST"'),
        tests_py=ANSWER_TEST_PY,
    )
    _scored(await _run(task, "oracle"), 1.0)
    _scored(await _run(task, "nop"), 0.0)


@pytest.mark.parametrize("declaration", ["comment", "env"])
@pytest.mark.parametrize("fallback", [False, True])
async def test_naming_the_runner_prepares_it_without_pins_or_ctrf(
    tmp_path, declaration, fallback
):
    """FIX-49 r4 M1: bench check's advice left the runner unset at c278f22a6."""
    runner = (
        "${BENCHFLOW_PYTEST:-python3 -m pytest}" if fallback else '"$BENCHFLOW_PYTEST"'
    )
    task = _task(
        tmp_path,
        "runner-advice",
        dockerfile=_image_grader(runner),
        solve="echo ok > /app/answer.txt\n",
        test=(
            "#!/bin/bash\n"
            + ("# uses $BENCHFLOW_PYTEST\n" if declaration == "comment" else "")
            + "if /usr/local/bin/grade; then echo 1; else echo 0; fi "
            "> /logs/verifier/reward.txt\n"
        ),
        tests_py=ANSWER_TEST_PY,
    )
    if declaration == "env":
        with (task / "task.toml").open("a") as stream:
            stream.write('\n[verifier.env]\nRUNNER_NAME = "BENCHFLOW_PYTEST"\n')
    _scored(await _run(task, "oracle"), 1.0)
    _scored(await _run(task, "nop"), 0.0)

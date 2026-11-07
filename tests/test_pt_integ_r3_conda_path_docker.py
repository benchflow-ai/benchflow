"""A conda task's image PATH entry, in a real container.

The round-3 sweep's conda task (``continuumio/miniconda3`` with Debian's
``/usr/bin/python3`` removed, so ``/opt/conda/bin`` holds the only Python)
scored 1 on integ 27c9387e and 0 on PT since round 1: the kit holds no Python,
the root-only PATH check is a Python helper, it exited 127, and the verifier's
PATH lost ``/opt/conda/bin``, so test.sh's ``python`` was not found. The check
now runs on the route's own realpath/stat (the kit's, else the system
folders') when no protected Python can run it.
"""

from __future__ import annotations

import pytest

from tests.exec_diet_fakes import docker_available

DOCKERFILE = """\
FROM continuumio/miniconda3:24.7.1-0
RUN rm -f /usr/bin/python3 /usr/bin/python3.* /usr/local/bin/python3
WORKDIR /app
RUN printf '[1, 2, 3, 4]\\n' > /app/numbers.json && mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

SOLVE = """\
#!/bin/bash
python -c 'import json; print(sum(json.load(open("/app/numbers.json"))))' > /app/answer.txt
"""

TEST = """\
#!/bin/bash
echo "test: PATH=$PATH python=$(command -v python)"
if [ "$(python -c 'print(open("/app/answer.txt").read().strip())' 2>/dev/null)" = 10 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _task(tmp_path):
    task = tmp_path / "conda-path"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text(
        "Sum /app/numbers.json into /app/answer.txt.\n"
    )
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text(SOLVE)
    (task / "tests" / "test.sh").write_text(TEST)
    return task


async def _run(tmp_path, agent: str, sandbox_user: str | None):
    from benchflow.rollout import Rollout, RolloutConfig

    name = f"{agent}-{sandbox_user or 'root'}"
    return await Rollout(
        RolloutConfig(
            task_path=_task(tmp_path),
            agent=agent,
            environment="docker",
            sandbox_user=sandbox_user,
            jobs_dir=tmp_path / "jobs",
            job_name=name,
            rollout_name=f"conda-path__{name}",
        )
    ).run()


@pytest.mark.deterministic
@pytest.mark.parametrize("sandbox_user", ["agent", None], ids=["agent", "root"])
async def test_the_conda_folder_stays_on_the_verifier_path(tmp_path, sandbox_user):
    """Fails on 72d9e5c5 in both shapes (reward 0, ``python: command not
    found`` in test.sh): ``agent`` has a kit without a Python; ``root`` (no
    sandbox user) has no kit and no ``python3`` on root's route. Integ
    27c9387e scores both 1."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "oracle", sandbox_user)
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 1.0}


@pytest.mark.deterministic
async def test_nop_still_scores_zero(tmp_path):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "nop", "agent")
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 0.0}

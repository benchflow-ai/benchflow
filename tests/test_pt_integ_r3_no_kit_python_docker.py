"""PT-integ round 3: plugin discovery when the protected kit holds no Python.

The 14-task TB2 oracle sweep after ce361161: ``configure-git-webserver``
scored 1 on integ 27c9387e and was unscored on PT ("Verifier hardening
failed: pytest plugin trust discovery: sandbox discovery command failed ...
python3 not found in the protected runtime"). Its image (ubuntu:24.04) has no
Python when the kit is built, so the kit's RECORD exists without a Python;
solve.sh's ``apt install openssh-server`` then brings ``python3`` in. PT's
discovery runs only on the kit's Python (127), and the fallback, finding a
``python3`` on PATH, read that as a broken interpreter. Integ ran that
``python3`` and found nothing to trust.

The answer needs no Python: discovery searches only the trusted PYTHONPATH
and the pre-agent record's search paths (data), and the
record of an image without a Python lists none, so there is nothing to
search. The image here is that shape without network: a Python is built in
but kept off PATH until solve.sh puts it back. test.sh mentions ``--ctrf``,
so ``ctrf`` is requested, as for the TB2 task.
"""

from __future__ import annotations

import pytest

from tests.exec_diet_fakes import docker_available

#: ubuntu:24.04, by digest (build-vms README).
UBUNTU = (
    "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)

DOCKERFILE = f"""\
FROM {UBUNTU}
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends python3 \\
    && rm -rf /var/lib/apt/lists/* \\
    && mkdir -p /opt/later \\
    && mv /usr/bin/python3* /opt/later/
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
WORKDIR /app
"""

SOLVE = """\
#!/bin/bash
# As the TB2 task's apt install does: a python3 appears during the solve.
mv /opt/later/python3* /usr/bin/
echo done > /app/solved.txt
"""

TEST = """\
#!/bin/bash
# The TB2 shape asks for pytest's ctrf plugin:
true pytest --ctrf /logs/verifier/ctrf.json
if [ "$(cat /app/solved.txt 2>/dev/null)" = done ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _task(tmp_path):
    task = tmp_path / "no-kit-python"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
        "[agent]\ntimeout_sec = 120.0\n\n[environment]\nbuild_timeout_sec = 600.0\n"
    )
    (task / "instruction.md").write_text("Put python3 back on PATH.\n")
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
            rollout_name=f"no-kit-python__{name}",
        )
    ).run()


@pytest.mark.deterministic
@pytest.mark.parametrize("sandbox_user", ["agent", None], ids=["agent", "root"])
async def test_the_oracle_is_scored_when_the_kit_holds_no_python(
    tmp_path, sandbox_user
):
    """Fails on ce361161: the verifier crashed in plugin discovery and the
    trial was unscored; integ 27c9387e scores it 1. ``root`` (no sandbox user, so no kit)
    is the control: discovery runs the image's python3 there, as on integ."""
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

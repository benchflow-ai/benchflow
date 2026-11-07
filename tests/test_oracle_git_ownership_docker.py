"""Guards DF-p3-eval-runner-2 (PY8 DX-G, base py8/dx-a 1ac5fffa).

Under the default ``--sandbox-user agent`` the oracle scored 0 on TB2's
``git-leak-recovery``: the sandbox-user grant chowned ``/app`` to the agent
user, solve.sh ran as root, and git refused the image's root-built
``/app/repo`` with "detected dubious ownership". With ``--sandbox-user null``
it scored 1. The harness trusted only ``/app`` itself, which
``safe.directory`` does not extend to the repository inside it.

The task here is that shape: a repository and a bare repository inside the
workspace, both built as root, and one outside it that must stay untrusted.
The verifier reads the repository too, so it must agree with the oracle.
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
    && apt-get install -y -qq --no-install-recommends git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git init -q /app/repo \\
    && git -C /app/repo -c user.name=t -c user.email=t@example.com \\
       commit -q --allow-empty -m seed \\
    && git init -q --bare /app/remote.git \\
    && git init -q /srv/outside
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

SOLVE = """\
#!/bin/bash
git -C /app/repo log --format=%s > /app/inside.txt 2>&1
git -C /app/remote.git rev-parse --is-bare-repository > /app/bare.txt 2>&1
git config --global --get-all safe.directory > /app/trusted.txt 2>&1
"""

TEST = """\
#!/bin/bash
REWARD=1
[ "$(cat /app/inside.txt 2>/dev/null)" = seed ] || REWARD=0
[ "$(cat /app/bare.txt 2>/dev/null)" = true ] || REWARD=0
grep -qx /srv/outside /app/trusted.txt 2>/dev/null && REWARD=0
# The verifier reads the repository itself, as git tasks' tests do.
[ "$(git -C /app/repo log --format=%s 2>&1)" = seed ] || REWARD=0
echo "$REWARD" > /logs/verifier/reward.txt
"""


def _task(tmp_path):
    task = tmp_path / "git-owned-by-root"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
        "[agent]\ntimeout_sec = 120.0\n\n[environment]\nbuild_timeout_sec = 600.0\n"
    )
    (task / "instruction.md").write_text("Read the log of /app/repo.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text(SOLVE)
    (task / "tests" / "test.sh").write_text(TEST)
    return task


async def _run(tmp_path, agent: str, name: str):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=_task(tmp_path),
            agent=agent,
            environment="docker",
            sandbox_user="agent",
            jobs_dir=tmp_path / "jobs",
            job_name=name,
            rollout_name=f"git-owned-by-root__{name}",
        )
    ).run()


@pytest.mark.deterministic
async def test_root_oracle_reads_the_workspace_repos_under_the_agent_user(tmp_path):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "oracle", "oracle")
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}, (
        "the oracle could not read the workspace repositories (or trusted "
        "one outside it); see agent/oracle.txt"
    )


@pytest.mark.deterministic
async def test_nop_still_scores_zero(tmp_path):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "nop", "nop")
    assert result.error is None, result.error
    assert result.rewards == {"reward": 0.0}

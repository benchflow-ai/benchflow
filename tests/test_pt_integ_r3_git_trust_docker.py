"""PT-integ round 3: the oracle's git trust on PT's route (real Docker).

PERF-ALL (results/status.md 11:56 UTC): TB2 ``fix-git``'s oracle scored 1/1
on integ 27c9387e and 0/1 on py8/pt-integ e91a5791. solve.sh's git stopped
at "detected dubious ownership in repository at '/app/personal-site'": the
sandbox-user grant hands the workspace to the agent user, so the repository
the image built as root inside it is no longer root's. Integ's harness wrote
``safe.directory`` with a root exec that had the oracle's HOME; on PT's
route that root exec is plumbing, which starts with BenchFlow's own empty
HOME, so the trust never reached solve.sh. The rule (DX-G) is
kept here: the workspace and each repository in it, per path, in the global
config of the user solve.sh runs as (root, or the task's named
``[agent] user``), never ``*`` or ``/*``, never ``/etc/gitconfig``; the
verifier reads no global config.

The task: a repository nested in the workspace, as ``fix-git`` has it (and
the image's WORKDIR inside it), a bare repository inside it, and one outside
the workspace (the image's WORKDIR, /app/personal-site) that must stay
untrusted. ``tester`` is a named
``[agent] user`` other than the sandbox user, so the repository is not its
own either.
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
RUN useradd -m tester
WORKDIR /app
RUN git init -q /app/personal-site \\
    && git -C /app/personal-site -c user.name=t -c user.email=t@example.com \\
       commit -q --allow-empty -m seed \\
    && git init -q --bare /app/personal-site/remote.git \\
    && git init -q /srv/outside
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
WORKDIR /app/personal-site
"""

SOLVE = """\
#!/bin/bash
out=/tmp/oracle-out
mkdir -p "$out"
git -C /app/personal-site log --format=%s > "$out/inside.txt" 2>&1
git -C /app/personal-site/remote.git rev-parse --is-bare-repository > "$out/bare.txt" 2>&1
git config --global --get-all safe.directory > "$out/trusted.txt" 2>&1
git config --system --get-all safe.directory > "$out/system.txt" 2>&1
chmod -R a+rX "$out"
# Shown in agent/oracle.txt when the test fails.
id; echo "HOME=$HOME"; head -n 50 "$out"/*.txt
"""

TEST = """\
#!/bin/bash
out=/tmp/oracle-out
REWARD=1
[ "$(cat $out/inside.txt 2>/dev/null)" = seed ] || REWARD=0
[ "$(cat $out/bare.txt 2>/dev/null)" = true ] || REWARD=0
# Per path only: no wildcard, nothing outside the workspace, nothing system-wide.
grep -qx '\\*' $out/trusted.txt 2>/dev/null && REWARD=0
grep -qx '/\\*' $out/trusted.txt 2>/dev/null && REWARD=0
grep -qx /srv/outside $out/trusted.txt 2>/dev/null && REWARD=0
grep -qx /app/personal-site $out/trusted.txt 2>/dev/null || REWARD=0
[ ! -s $out/system.txt ] || REWARD=0
# The verifier reads the repository itself, as git tasks' tests do.
[ "$(git -C /app/personal-site log --format=%s 2>&1)" = seed ] || REWARD=0
echo "$REWARD" > /logs/verifier/reward.txt
"""


def _task(tmp_path, user: str | None):
    task = tmp_path / "nested-git-owned-by-root"
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    agent = "[agent]\ntimeout_sec = 120.0\n"
    if user:
        agent += f'user = "{user}"\n'
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
        f"{agent}\n[environment]\nbuild_timeout_sec = 600.0\n"
    )
    (task / "instruction.md").write_text("Read the log of /app/personal-site.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text(SOLVE)
    (task / "tests" / "test.sh").write_text(TEST)
    return task


async def _run(tmp_path, agent: str, name: str, user: str | None):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=_task(tmp_path, user),
            agent=agent,
            environment="docker",
            sandbox_user="agent",
            jobs_dir=tmp_path / "jobs",
            job_name=name,
            rollout_name=f"nested-git__{name}",
        )
    ).run()


@pytest.mark.deterministic
@pytest.mark.parametrize("user", [None, "tester"], ids=["root", "agent-user"])
async def test_the_oracle_trusts_the_nested_workspace_repository(tmp_path, user):
    """Fails on e91a5791: the oracle (root, or the named [agent] user) got
    "dubious ownership" on /app/personal-site and scored 0."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "oracle", f"oracle-{user or 'root'}", user)
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}, (
        "the oracle could not use the nested repository (or trusted more than "
        "the workspace's repositories); see agent/oracle.txt"
    )


@pytest.mark.deterministic
async def test_nop_still_scores_zero(tmp_path):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    result = await _run(tmp_path, "nop", "nop", None)
    assert result.error is None, result.error
    assert result.rewards == {"reward": 0.0}

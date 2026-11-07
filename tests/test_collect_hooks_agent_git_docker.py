"""A main collect hook never runs the agent's git config as root (real Docker).

Guards this fix against the regression introduced by PR #84 (8e0630a9). To
let a no-op trial's hook read the workspace repository that the sandbox-user
grant gave the agent, #84 ran each main ``[[verifier.collect]]`` hook as root
with ``safe.directory`` trust for every repository in the workspace. Git
then read the repository config the agent wrote: an agent that set
``core.fsmonitor`` and a filter driver had both run as uid 0 by the hook's
``git add -A`` after its turn, before the artifacts were captured. Without
that trust git refuses the repository ("dubious ownership") and the hook
writes no patch.

A hook that names no ``user`` now runs as the account the agent ran as, which
owns the workspace: git reads the repository with no trust granted, what the
agent configured runs as the agent and nothing more, and that account's
processes are stopped again before the capture.

The agent here is an operator script, run as the sandbox user. A no-op
trial's hook reads the untouched repository the same way, and the oracle's
(root, as solve.sh) keeps the trust the oracle's own setup gives it.
"""

from __future__ import annotations

import json

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
RUN git init -q /app/repo && echo base > /app/repo/f \\
    && git -C /app/repo add f \\
    && git -C /app/repo -c user.name=t -c user.email=t@example.com \\
       commit -q -m seed
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

TOML = """\
version = "1.0"
artifacts = ["/tmp/agent.patch", "/tmp/fsmonitor-uid", "/tmp/filter-uid"]

[verifier]
timeout_sec = 120.0

[[verifier.collect]]
command = "cd /app/repo && git add -A && git diff --cached > /tmp/agent.patch"

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
"""

#: The agent's turn: its repository's config names its own programs. The
#: fsmonitor also leaves a process behind, which the kill after the hooks stops.
AGENT = """\
set -eu
printf '#!/bin/sh\\nid -u >> /tmp/fsmonitor-uid\\n(setsid sleep 300 </dev/null >/dev/null 2>&1 &)\\nexit 1\\n' > /app/fsm.sh
printf '#!/bin/sh\\nid -u >> /tmp/filter-uid\\ncat\\n' > /app/filt.sh
chmod 755 /app/fsm.sh /app/filt.sh
git -C /app/repo config core.fsmonitor /app/fsm.sh
git -C /app/repo config filter.x.clean /app/filt.sh
echo '* filter=x' > /app/repo/.gitattributes
echo changed >> /app/repo/f
"""


def _task(tmp_path):
    task = tmp_path / "collect-git"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(TOML)
    (task / "instruction.md").write_text("Change /app/repo/f.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\necho changed >> /app/repo/f\n"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task


async def _trial(tmp_path, agent: str):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    from benchflow.rollout import Rollout, RolloutConfig

    name = agent.split(":")[0]
    result = await Rollout(
        RolloutConfig(
            task_path=_task(tmp_path),
            agent=agent,
            environment="docker",
            sandbox_user="agent",
            jobs_dir=tmp_path / "jobs",
            job_name="collect-git",
            rollout_name=f"collect-git__{name}",
        )
    ).run()
    assert result.error is None, result.error
    assert result.rollout_dir is not None
    (hook,) = json.loads((result.rollout_dir / "collect-hooks.json").read_text())
    return result.rollout_dir, hook


@pytest.mark.deterministic
async def test_a_main_hook_runs_the_agents_git_config_only_as_the_agent(tmp_path):
    script = tmp_path / "agent.sh"
    script.write_text(AGENT)
    trial, hook = await _trial(tmp_path, f"script:{script}")
    artifacts = trial / "artifacts"
    # The agent's programs ran, as the agent's uid and never as root's.
    uids = {
        line
        for name in ("fsmonitor-uid", "filter-uid")
        if (artifacts / name).is_file()
        for line in (artifacts / name).read_text().split()
    }
    assert uids and "0" not in uids, f"the hook's git ran them as uids {uids}"
    # The hook read the agent's repository without any trust: its patch.
    assert "+changed" in (artifacts / "agent.patch").read_text()
    assert (hook.get("user"), hook.get("return_code")) == ("agent", 0), hook
    # Each process the fsmonitor left was stopped before the capture, and
    # the kill before grading found none.
    runs = len((artifacts / "fsmonitor-uid").read_text().split())
    record = json.loads((trial / "kill-before-grading.json").read_text())
    assert record["quiesce_killed"] >= runs and record["killed"] == 0, record


@pytest.mark.deterministic
async def test_a_nop_trials_hook_reads_the_untouched_repository(tmp_path):
    """The case #84 fixed: root's git refused the repository the grant gave
    the agent, and the hook wrote no patch."""
    trial, hook = await _trial(tmp_path, "nop")
    assert (hook.get("user"), hook.get("return_code")) == ("agent", 0), hook
    assert (trial / "artifacts" / "agent.patch").read_text() == ""


@pytest.mark.deterministic
async def test_the_oracles_hook_still_runs_as_root(tmp_path):
    """solve.sh runs as root with the workspace's repositories trusted for
    root (trust_workspace_repos); its hook runs there too, as before."""
    trial, hook = await _trial(tmp_path, "oracle")
    assert ("user" not in hook, hook.get("return_code")) == (True, 0), hook
    assert "+changed" in (trial / "artifacts" / "agent.patch").read_text()

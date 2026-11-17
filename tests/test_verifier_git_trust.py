"""In integrity off and audit test.sh's git trusts the repositories BenchFlow
handed over (the lead's 12:53Z ruling).

The sandbox-user grant chowns the workspace to the agent user, so a git
repository in it (at its root, as SWE-bench's /testbed, or inside it, as
TB2's git-leak-recovery /app/repo) is no longer root's, and test.sh's git
stops with "detected dubious ownership". Strict chowns the workspace back;
off and audit leave the agent's files as they are, so before test.sh
BenchFlow trusts exactly the paths it handed over again, for every agent
(not only the oracle), and records it as undoing its own chown. A repository
outside the workspace stays as git judges it. Real Docker trials, skipped
without a daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.rollout._setup import trust_workspace_repos_cmd
from tests.exec_diet_fakes import docker_available


def test_the_reporting_command_names_what_it_adds():
    plain = trust_workspace_repos_cmd("/testbed")
    reporting = trust_workspace_repos_cmd("/testbed", report=True)
    assert "bf-trusted" not in plain
    assert 'echo "bf-trusted $1"' in reporting
    # Everything else is the oracle's own command.
    assert reporting.replace(' && echo "bf-trusted $1"', "") == plain


async def test_an_exec_answer_that_is_no_text_trusts_nothing_and_raises_nothing():
    """A sandbox whose exec answers with something other than text (a test
    double's coroutine, a broken backend) adds no path and raises nothing: the
    trust step's own TypeError replaced a crash's classified "connection
    reset" (unscored) with an internal error (14/n)."""
    from types import SimpleNamespace

    from benchflow.rollout._setup import trust_verifier_workspace_repos

    async def exec(command, **kwargs):
        return SimpleNamespace(return_code=0, stdout=object(), stderr="")

    env = SimpleNamespace(exec=exec)
    task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(user=None)))

    assert await trust_verifier_workspace_repos(env, "/testbed", task) == []


DOCKERFILE = """\
FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq git >/dev/null \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
WORKDIR /app
RUN git init -q /app/repo && git -C /app/repo -c user.email=a@b -c user.name=a \\
    commit -q --allow-empty -m fixed \\
    && git init -q /opt/outside && chown -R 1003:1003 /opt/outside
"""

TEST = """\
#!/bin/bash
if (cd /app/repo && git log --oneline | grep -q fixed); then r=1; else r=0; fi
if (cd /opt/outside && git status >/dev/null 2>&1); then o=trusted; else o=untrusted; fi
echo "outside=$o" > /logs/verifier/outside.txt
echo $r > /logs/verifier/reward.txt
"""


def _task(tmp_path: Path, name: str) -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("The fix is already committed.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "tests" / "test.sh").write_text(TEST)
    return task


async def _nop(task: Path, **isolation):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent="nop",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=task.parent / "jobs",
            job_name=f"{task.name}-nop",
            rollout_name=f"{task.name}__nop",
            **isolation,
        )
    ).run()


def _trial_file(task: Path, name: str) -> Path:
    (found,) = [
        p
        for p in (task.parent / "jobs").rglob(name)
        if "verifier-sandbox" not in p.parts and "verifier-at-exit" not in p.parts
    ]
    return found


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
@pytest.mark.parametrize("mode", ["audit", "off", "strict"])
async def test_a_non_oracle_agent_on_a_workspace_repo_scores_its_answer(tmp_path, mode):
    """CIR's git-leak-recovery shape (nop, the answer in the image's
    workspace repo): 1.0 in every mode, and the repository outside the
    workspace that git distrusts stays distrusted."""
    task = _task(tmp_path, f"git-{mode}")
    kwargs = {} if mode == "audit" else {"integrity": mode}

    result = await _nop(task, **kwargs)

    assert result.error is None and result.verifier_error is None, result.error
    assert result.rewards == {"reward": 1.0}, (mode, result.rewards)
    assert _trial_file(task, "outside.txt").read_text().strip() == "outside=untrusted"
    if mode == "audit":
        record = json.loads(_trial_file(task, "verifier-inputs.json").read_text())
        [undone] = record["benchflow_undone"]
        assert undone["what"] == "git safe.directory" and undone["user"] == "root"
        assert "/app/repo" in undone["safe_directory"]
        assert all(
            path == "/app" or path.startswith("/app/")
            for path in undone["safe_directory"]
        ), undone

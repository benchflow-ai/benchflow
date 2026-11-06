"""The staged trajectory publish's execs (PY8 U12 on top of V6).

Publishing stages each file in a fresh root-owned folder under /logs and
copies it into /logs/agent as the sandbox user. The defensive
``mkdir -p /logs/agent`` before it now runs in the stage's own root exec;
the other steps stay separate execs because something that is not the same
user's shell runs between them: the upload (a host call) between the stage's
mkdir and its ``chmod``, the copy as the sandbox user between the ``chmod``
and the stage's removal as root. On ``integ/py-0.8`` 008da0e0 each of these
tests fails by its exec list.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.rollout._setup import _publish_trajectory_for_verifier
from tests.reachable_fixture import directory

TRAJECTORY = [{"type": "oracle", "stdout": "x"}]
STAGE = "/logs/.benchflow-publish-"


def _stage_of(calls: list) -> str:
    """The staging folder the publish named (in its first exec that names it)."""
    command = next(c[1] for c in calls if STAGE in c[1])
    start = command.index(STAGE)
    return command[start:].split()[0].rstrip(";")


async def test_docker_publish_makes_logs_agent_and_the_stage_in_one_exec(tmp_path):
    from benchflow.sandbox.docker import DockerSandbox

    calls: list = []
    sandbox = DockerSandbox.__new__(DockerSandbox)

    async def exec_(command, **kwargs):
        calls.append(("exec", command, kwargs.get("user")))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def copy(host_path, sandbox_path, *, into, service="main"):
        calls.append(("cp", sandbox_path))

    sandbox.exec = exec_
    sandbox._copy = copy
    await _publish_trajectory_for_verifier(
        sandbox,
        TRAJECTORY,
        directory(tmp_path / "agent"),
        sandbox_user="agent",
        publish_reply=False,
    )
    stage = _stage_of(calls)
    assert calls == [
        (
            "exec",
            f"mkdir -p /logs/agent; mkdir -m 755 {stage} && "
            "{ [ -L /logs/agent ] || chown agent /logs/agent; }",
            "root",
        ),
        # upload_file's own parent mkdir: the stage exists by then, but
        # upload_file cannot know it.
        ("exec", f"mkdir -p {stage}", "root"),
        ("cp", f"{stage}/acp_trajectory.jsonl"),
        ("exec", f"chmod 644 {stage}/*", "root"),
        (
            "exec",
            "rc=0; { rm -f /logs/agent/acp_trajectory.jsonl && "
            f"cat {stage}/acp_trajectory.jsonl > /logs/agent/acp_trajectory.jsonl; }}"
            " || rc=1; exit $rc",
            "agent",
        ),
        ("exec", f"rm -rf {stage}", "root"),
    ]


async def test_daytona_publish_runs_four_execs(tmp_path):
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox

    calls: list = []
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)

    async def exec_(command, **kwargs):
        calls.append(("exec", command, kwargs.get("user")))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def upload(source, target, *, mode=None, **_kwargs):
        calls.append(("upload", target))

    sandbox.exec = exec_
    sandbox.upload_file = upload
    await _publish_trajectory_for_verifier(
        sandbox, TRAJECTORY, directory(tmp_path / "agent"), publish_reply=False
    )
    stage = _stage_of(calls)
    assert [c[0] for c in calls] == ["exec", "upload", "exec", "exec", "exec"]
    assert calls[0] == ("exec", f"mkdir -p /logs/agent; mkdir -m 755 {stage}", "root")
    assert calls[1] == ("upload", f"{stage}/acp_trajectory.jsonl")
    assert calls[-1] == ("exec", f"rm -rf {stage}", "root")


async def test_the_defensive_mkdir_runs_once_in_the_stage_exec(tmp_path):
    """The defensive mkdir is followed by ``;``, so the exec's status is the
    stage's (its own failure is ignored, as when it ran alone), and the exec
    keeps the stage's 30 s budget."""
    calls: list = []

    class Env:
        async def exec(self, command, user=None, timeout_sec=None):
            calls.append((command, user, timeout_sec))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target, **_kwargs):
            calls.append(("upload", target, None))

    await _publish_trajectory_for_verifier(
        Env(),
        TRAJECTORY,
        directory(tmp_path / "agent"),
        sandbox_user="agent",
        publish_reply=False,
    )
    first = calls[0][0]
    assert first.startswith("mkdir -p /logs/agent; mkdir -m 755 /logs/"), calls
    # The stage's own budget, not the 10 s the defensive mkdir had alone.
    assert calls[0][1:] == ("root", 30)
    assert [c[0] for c in calls if c[0] == "upload"] == ["upload"]
    assert sum(1 for c in calls if "mkdir -p /logs/agent" in c[0]) == 1


async def test_a_retried_stage_carries_the_defensive_mkdir_too(tmp_path):
    """Integ's publication retry makes a fresh stage when the
    first stage mkdir gives no answer. The defensive mkdir rides in front of
    each attempt (it is idempotent and its status is dropped by ``;``), so
    the retry still finds /logs, and no attempt needs an exec of its own;
    both stages are removed. On integ/py-0.8 363d815b the first exec is the
    defensive mkdir alone."""
    calls: list = []

    class Env:
        async def exec(self, command, user=None, timeout_sec=None):
            calls.append((command, user, timeout_sec))
            first_stage = sum(1 for c in calls if "mkdir -m 755" in c[0]) == 1
            failed = "mkdir -m 755" in command and first_stage
            return SimpleNamespace(return_code=int(failed), stdout="", stderr="")

        async def upload_file(self, source, target, **_kwargs):
            calls.append(("upload", target, None))

    await _publish_trajectory_for_verifier(
        Env(),
        TRAJECTORY,
        directory(tmp_path / "agent"),
        sandbox_user="agent",
        publish_reply=False,
    )
    attempts = [c for c in calls if "mkdir -m 755" in c[0]]
    assert len(attempts) == 2, calls
    stages = []
    for command, user, budget in attempts:
        assert command.startswith("mkdir -p /logs/agent; mkdir -m 755 /logs/")
        assert (user, budget) == ("root", 30)
        stages.append(command.split("mkdir -m 755 ")[1].split()[0])
    assert stages[0] != stages[1]
    assert calls[0] == attempts[0] and calls[1] == attempts[1]
    assert not [c for c in calls if c[0] == "mkdir -p /logs/agent"]
    assert [c[0] for c in calls if c[0].startswith("rm -rf ")] == [
        f"rm -rf {stage}" for stage in stages
    ]

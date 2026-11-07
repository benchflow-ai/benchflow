"""Independent FIX-49 R6 boundary review of 0294404ae..7bc81e6ba.

Guards the R6 failure-provenance changes against cross-criterion aggregation
and private staging creation failures. Run only in the assigned VM container.
"""

import asyncio
import errno
import json
import os
import shutil
from unittest.mock import AsyncMock

import pytest

from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.taskmd import judging
from benchflow.taskmd import verify as taskmd
from tests.test_r5_release_contract import observe, prime
from tests.test_verifier_own_failures import _assess


@pytest.mark.parametrize("script", [False, True])
@pytest.mark.parametrize("provider_first", [False, True])
async def test_controller_failure_in_another_criterion_stays_unscored(
    tmp_path, monkeypatch, script, provider_first
):
    """Guards 7bc81e6ba: criterion order cannot hide an observed provider fault."""
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _Sandbox

    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    if script:
        (package / "verifier/test.sh").write_text("#!/bin/sh\nexit 0\n")
    document = package / "task.md"
    original = document.read_text()
    assert 'timeout = "3m"' in original
    document.write_text(
        original.replace('timeout = "3m"', 'timeout = "1s"\nper = "criterion"')
    )
    rubric_path = package / "verifier/rubric.json"
    rubric = json.loads(rubric_path.read_text())
    criterion = rubric["criteria"][0]
    deadline = {**criterion, "id": "deadline-criterion"}
    provider = {**criterion, "id": "provider-criterion"}
    rubric["criteria"] = (
        [provider, deadline] if provider_first else [deadline, provider]
    )
    rubric_path.write_text(json.dumps(rubric))

    root = tmp_path / "sandbox"
    (root / "app").mkdir(parents=True)
    (root / "app/main.py").write_text("print(42)\n")
    sandbox = _Sandbox(root)
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = Verifier(Task(materialize_task_dir(package)), paths, sandbox)
    credentials = judging.Credentials("api-key", "independent-placeholder")
    monkeypatch.setattr(judging, "resolve_credentials", lambda: (credentials, None))
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    original_exec = sandbox.exec
    commands = []
    provider_faults = []

    async def execute(command, **kwargs):
        if taskmd.COMMAND_FILE in command:
            commands.append(command)
            await asyncio.sleep(1.1)
        return await original_exec(command, **kwargs)

    sandbox.exec = AsyncMock(side_effect=execute)

    async def call(body, *_args, **_kwargs):
        prompt = body["messages"][0]["content"]
        if "provider-criterion" in prompt:
            provider_faults.append(True)
            raise judging.JudgeError("independent second-criterion provider outage")
        assert "deadline-criterion" in prompt
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "run",
                    "name": "run",
                    "input": {"command": "true", "timeout": 1},
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    client = AsyncMock(side_effect=call)
    monkeypatch.setattr(judging, "MessagesClient", lambda _: client)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(commands) == 2 and len(provider_faults) == 2
    assert client.await_count == 4
    sessions = [
        json.loads(path.read_text())
        for path in (tmp_path / "trial").rglob("agent-criterion-*.json")
    ]
    print(
        json.dumps(
            {
                "provider_first": provider_first,
                "script": script,
                "commands": len(commands),
                "provider_faults": len(provider_faults),
                "rewards": rewards,
                "error": error,
                "warnings": warnings,
                "sessions": [
                    {k: s.get(k) for k in ("unit", "attempt", "end", "failure_kind")}
                    for s in sessions
                ],
            }
        )
    )
    observe(tmp_path, rewards, error, "unscored")
    assert not warnings
    if provider_first:
        assert "second-criterion provider outage" in error


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["EACCES", "EIO", "ENOSPC"])
async def test_private_publication_staging_creation_is_unscored(
    tmp_path, monkeypatch, prior, kind
):
    """Guards R6's Sol-M1 family: the private staging parent is controller I/O."""
    from benchflow.rollout import _separate_verifier as separate
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, root = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, root, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    original_mkdir = os.mkdir
    denied = []

    def mkdir(path, mode=0o777, *, dir_fd=None):
        if os.path.basename(os.fspath(path)).startswith(".benchflow-private-"):
            denied.append(
                {"path": os.fspath(path), "dir_fd_present": dir_fd is not None}
            )
            raise OSError(
                getattr(errno, kind), "independent controller private-stage denial"
            )
        return original_mkdir(path, mode, dir_fd=dir_fd)

    async def verify(env, task, paths, *args, **kwargs):
        await prime(tmp_path / "earlier", env, prior)
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        paths.reward_text_path.write_text("1")
        monkeypatch.setattr(os, "mkdir", mkdir)
        return {"reward": 1.0}, None

    rewards, error = await separate.run_separate_verifier(
        rollout, create_environment=lambda *_: box, verify=verify
    )
    monkeypatch.setattr(os, "mkdir", original_mkdir)
    print(
        json.dumps(
            {
                "prior": prior,
                "errno": kind,
                "denied": denied,
                "rewards": rewards,
                "error": error,
                "warnings": getattr(rollout, "_warnings", []),
            }
        )
    )
    assert denied, "the real private staging mkdir must be reached"
    observe(tmp_path, rewards, error, "unscored", "private-stage denial")

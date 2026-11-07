"""Guards FIX-49 r6 against the six scoring gaps at 0294404ae.

The final rollout boundary runs in every case. Provider/API faults are local
fixtures; unreadable files, archive validation and session deadlines are real.
"""

import asyncio
import contextlib
import errno
import io
import shutil
import tarfile
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox._base import ExecResult
from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.task.verifier_errors import VerifierOutputParseError
from benchflow.taskmd import judging
from benchflow.taskmd import verify as taskmd
from tests.test_verifier_own_failures import _assess, _task


def native(tmp_path, monkeypatch, script):
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _Sandbox

    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    if script:
        (package / "verifier/test.sh").write_text("#!/bin/sh\nexit 0\n")
    document = package / "task.md"
    document.write_text(
        document.read_text().replace('timeout = "3m"', 'timeout = "1s"')
    )
    root = tmp_path / "sandbox"
    (root / "app").mkdir(parents=True)
    (root / "app/main.py").write_text("print(42)\n")
    sandbox = _Sandbox(root)
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = Verifier(Task(materialize_task_dir(package)), paths, sandbox)
    credentials = judging.Credentials("api-key", "fixture-placeholder")
    monkeypatch.setattr(judging, "resolve_credentials", lambda: (credentials, None))
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    return sandbox, verifier


def response(name, args):
    return {
        "content": [{"type": "tool_use", "id": "fixture", "name": name, "input": args}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.parametrize("script", [False, True])
@pytest.mark.parametrize(
    "fault",
    ["shared-unreadable", "tool-deadline", "provider-deadline", "mixed", "stash"],
)
async def test_native_session_failure_keeps_its_operation(
    tmp_path, monkeypatch, script, fault
):
    sandbox, verifier = native(tmp_path, monkeypatch, script)
    original_exec = sandbox.exec
    commands = []
    planted = tmp_path / "trial/verifier/judgments.jsonl"

    async def execute(command, **kwargs):
        if taskmd.COMMAND_FILE in command:
            commands.append(command)
            if fault in ("tool-deadline", "mixed"):
                await asyncio.sleep(1.1)
        return await original_exec(command, **kwargs)

    sandbox.exec = AsyncMock(side_effect=execute)
    if fault == "stash":
        sandbox.upload_file = AsyncMock(
            side_effect=RuntimeError("fixture upload API lost")
        )
    calls = 0

    async def call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if fault == "provider-deadline":
            await asyncio.sleep(1.1)
        if fault == "mixed" and calls == 1:
            raise judging.JudgeError("fixture provider unavailable")
        if fault == "shared-unreadable":
            planted.write_text("submitted entry\n")
            planted.chmod(0)
            return response(
                "submit_review",
                {
                    "verdicts": [
                        {
                            "id": "prints-42",
                            "verdict": "fail",
                            "citations": [],
                            "rationale": "fixture",
                        }
                    ],
                    "tags": [],
                },
            )
        return response("run", {"command": "true", "timeout": 1})

    provider = AsyncMock(side_effect=call)
    monkeypatch.setattr(judging, "MessagesClient", lambda _: provider)
    try:
        rewards, error, warnings = await _assess(sandbox, verifier)
    finally:
        if planted.exists():
            planted.chmod(0o600)
    if fault in ("shared-unreadable", "tool-deadline"):
        assert rewards == {"reward": 0.0} and error is None and warnings
    else:
        assert rewards is None and error is not None and not warnings
    if fault == "tool-deadline":
        assert len(commands) == 2 and sandbox._benchflow_test_command_issued is True
    elif fault == "mixed":
        assert len(commands) == 1 and "provider unavailable" in error
    elif fault == "stash":
        sandbox.upload_file.assert_awaited_once()
        assert "upload API lost" in error
        provider.assert_not_awaited()
    else:
        assert provider.await_count > 0


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["leaf", "folder", "unicode", "EIO", "ENOSPC"])
async def test_archive_names_and_private_storage_have_distinct_outcomes(
    tmp_path, monkeypatch, prior, kind
):
    sandbox, verifier = _task(tmp_path, "llm-judge")
    sandbox._benchflow_test_command_issued = prior
    name = {
        "leaf": "a" * 256,
        "folder": "a" * 256 + "/answer",
        "unicode": "é" * 128,
    }.get(kind, "answer")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("app/output/" + name)
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    data = payload.getvalue()

    async def execute(command, **kwargs):
        return ExecResult(
            stdout="dir\n"
            if "then echo link" in command
            else f"BF copy {len(data)} 0\n",
            return_code=0,
        )

    async def download(source, target):
        assert source == taskmd.COPY_TAR
        target.write_bytes(data)

    sandbox.exec = AsyncMock(side_effect=execute)
    sandbox.download_file = AsyncMock(side_effect=download)
    original_create = taskmd.create_private_file
    writes = []

    @contextlib.contextmanager
    def create(target):
        writes.append(target.name)
        if kind in ("EIO", "ENOSPC"):
            raise OSError(getattr(errno, kind), "fixture private capture failed")
        with original_create(target) as stream:
            yield stream

    monkeypatch.setattr(taskmd, "create_private_file", create)
    provider = AsyncMock()
    monkeypatch.setattr("benchflow.rewards.llm.call_judge", provider)
    rewards, error, warnings = await _assess(sandbox, verifier)
    sandbox.download_file.assert_awaited_once()
    provider.assert_not_awaited()
    if kind in ("EIO", "ENOSPC"):
        assert writes == ["answer"]
        assert rewards is None and "private capture failed" in error and not warnings
    else:
        assert not writes
        assert rewards == {"reward": 0.0} and error is None and warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "payload", ["PRIVATE_RESPONSE", '{"verdict":"PRIVATE_RESPONSE"}']
)
async def test_agent_provider_verdict_is_unscored_and_sanitized(
    tmp_path, monkeypatch, prior, payload
):
    from tests.test_verifier_strategies import _AGENT_JUDGE_VERIFIER_MD

    sandbox, verifier = _task(tmp_path)
    sandbox._benchflow_test_command_issued = prior
    (tmp_path / "task/tests/verifier.md").write_text(_AGENT_JUDGE_VERIFIER_MD)
    (verifier._rollout_paths.artifacts_dir / "answer.txt").write_text("42")
    provider = AsyncMock(return_value=payload)
    monkeypatch.setattr("benchflow.rewards.llm.call_judge", provider)
    rewards, error, warnings = await _assess(sandbox, verifier)
    provider.assert_awaited_once()
    assert rewards is None and "JudgeScoringError" in error and not warnings
    assert "PRIVATE_RESPONSE" not in error
    assert not verifier._rollout_paths.reward_json_path.exists()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("fault", ["api", "evidence"])
async def test_protected_upload_keeps_typed_provenance(tmp_path, prior, fault):
    sandbox, verifier = _task(tmp_path)
    sandbox._benchflow_test_command_issued = prior
    problem = (
        RuntimeError("fixture upload API lost")
        if fault == "api"
        else VerifierOutputParseError("fixture verified unsafe destination")
    )
    sandbox.upload_dir = AsyncMock(side_effect=problem)
    rewards, error, warnings = await _assess(sandbox, verifier)
    sandbox.upload_dir.assert_awaited_once()
    if fault == "api":
        assert rewards is None and "upload API lost" in error and not warnings
    else:
        assert rewards == {"reward": 0.0} and error is None and warnings


@pytest.mark.parametrize("source_denied", [False, True])
async def test_snapshot_source_denial_is_distinct_from_private_destination(
    tmp_path, monkeypatch, source_denied
):
    sandbox, verifier = _task(tmp_path)
    sandbox._benchflow_test_command_issued = True
    source = verifier._rollout_paths.reward_text_path
    source.write_text("1")
    if source_denied:
        (tmp_path / "trial/verifier/reward.txt").chmod(0)
    else:
        original = rw.create_private_file

        @contextlib.contextmanager
        def create(target):
            if target.name == "reward.txt":
                raise PermissionError(errno.EACCES, "fixture private snapshot denied")
            with original(target) as stream:
                yield stream

        monkeypatch.setattr(rw, "create_private_file", create)

    async def capture():
        with rw.snapshot_file(source):
            pytest.fail("the selected snapshot operation must fail")

    monkeypatch.setattr(verifier, "verify", capture)
    try:
        rewards, error, warnings = await _assess(sandbox, verifier)
    finally:
        (tmp_path / "trial/verifier/reward.txt").chmod(0o600)
    if source_denied:
        assert rewards == {"reward": 0.0} and error is None and warnings
    else:
        assert rewards is None and "private snapshot denied" in error and not warnings

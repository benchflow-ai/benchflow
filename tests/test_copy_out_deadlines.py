"""Copy-out deadlines grow with the copy's byte cap (PLAN P7, Rust 42f67e1e).

A copy out of the sandbox gets the larger of 180 s and a second per MiB of
its cap. The fixed deadlines (300 s for a 1 GiB artifact collection, 180 s
for an unverified rollout's) cut honest 1 GiB copies when several ran at
once on Daytona (about 3.8 MB/s each), so the trial lost its artifacts and,
with a separate verifier, its grade.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.rollout import _artifacts
from benchflow.sandbox.protocol import ExecResult
from tests.reachable_fixture import directory

MIB = 1024 * 1024


def _packed_source(argv: list[str]) -> str | None:
    """The source a workspace pack's exec packs (``python3 -I -S [-B] -c
    <script> <source> ...``: integ's flags, or PT's protected Python's),
    or None for any other exec."""
    if argv[:5] == ["python3", "-I", "-S", "-B", "-c"] and len(argv) > 6:
        return argv[6]
    if argv[:4] == ["python3", "-I", "-S", "-c"] and len(argv) > 5:
        return argv[5]
    return None


class RecordingTransport:
    """The sandbox is this machine; every exec's command and deadline is kept."""

    def __init__(self) -> None:
        self.execs: list[tuple[list[str], int]] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        self.execs.append((list(argv), timeout_sec))
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)

    def capture_deadline(self, source: Path) -> int:
        """The deadline the pack of ``source`` ran under."""
        found = [
            timeout
            for argv, timeout in self.execs
            if _packed_source(argv) == str(source)
        ]
        assert len(found) == 1, self.execs
        return found[0]


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    logs = tmp_path / "box" / "logs" / "artifacts"
    logs.mkdir(parents=True)
    (logs / "answer.json").write_text('{"x": 1}\n')
    workspace = tmp_path / "box" / "app"
    workspace.mkdir()
    declared = tmp_path / "box" / "out"
    declared.mkdir()
    (declared / "report.txt").write_text("done\n")
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    return logs, workspace, declared, trial


async def _collect(tmp_path: Path, **kwargs) -> tuple[RecordingTransport, Path, Path]:
    logs, workspace, declared, trial = _layout(tmp_path)
    env = RecordingTransport()
    manifest = await _artifacts.collect_artifacts(
        env,
        artifacts=[str(declared)],
        workspace=str(workspace),
        artifacts_dir=directory(trial / "artifacts"),
        manifest_path=trial / _artifacts.MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
        **kwargs,
    )
    assert [c["status"] for c in manifest["collections"]] == [
        "collected",
        "collected",
    ]
    return env, logs, declared


def test_the_deadline_is_180_s_or_a_second_per_mib_of_the_cap():
    from benchflow.rollout._artifacts import copy_out_deadline

    assert copy_out_deadline(0) == 180
    assert copy_out_deadline(1234) == 180
    assert copy_out_deadline(16 * MIB) == 180
    assert copy_out_deadline(180 * MIB) == 180
    assert copy_out_deadline(181 * MIB) == 181
    assert copy_out_deadline(1024 * MIB) == 1024
    assert copy_out_deadline(1088 * MIB) == 1088


@pytest.mark.asyncio
async def test_a_1_gib_collection_gets_1024_s_not_300(tmp_path):
    # The default cap is 1 GiB: the logs copy may bring all of it, so it gets
    # 1,024 s; the declared artifact after it gets a second per MiB of what is
    # left of the cap (the logs brought a few bytes).
    env, logs, declared = await _collect(tmp_path)
    assert env.capture_deadline(logs) == 1024
    assert env.capture_deadline(declared) == 1023


@pytest.mark.asyncio
async def test_a_small_cap_gets_the_180_s_floor(tmp_path):
    env, logs, declared = await _collect(tmp_path, max_bytes=16 * MIB)
    assert env.capture_deadline(logs) == 180
    assert env.capture_deadline(declared) == 180


@pytest.mark.asyncio
async def test_an_explicit_timeout_is_a_floor_never_a_ceiling(tmp_path):
    env, logs, _ = await _collect(tmp_path, max_bytes=400 * MIB, timeout_sec=600)
    assert env.capture_deadline(logs) == 600
    env, logs, _ = await _collect(
        tmp_path / "again", max_bytes=400 * MIB, timeout_sec=60
    )
    assert env.capture_deadline(logs) == 400


@pytest.mark.asyncio
async def test_an_unverified_rollout_gets_the_whole_caps_deadline(
    tmp_path, monkeypatch
):
    # An agent error skips the verifier; cleanup still collects what the agent
    # left. That collection shares one 1 GiB cap, so it gets the cap's
    # deadline (1,024 s) plus the probe's minute, not a fixed 180 s.
    seen: list[float] = []
    original = asyncio.wait_for

    async def spy(awaitable, timeout):
        seen.append(timeout)
        return await original(awaitable, timeout)

    collected: list[object] = []

    async def fake_collect(rollout):
        collected.append(rollout)

    monkeypatch.setattr(_artifacts.asyncio, "wait_for", spy)
    monkeypatch.setattr(_artifacts, "collect_rollout_artifacts", fake_collect)
    rollout = SimpleNamespace(
        _artifacts_collected=False,
        _env=object(),
        _rollout_paths=SimpleNamespace(rollout_dir=tmp_path),
        _diagnostics=None,
    )
    await _artifacts.collect_unverified_rollout_artifacts(rollout)
    assert collected == [rollout]
    assert seen == [1024 + 60]


class TimingOutTransport(RecordingTransport):
    """The pack of one source runs past its deadline, as backends report it."""

    def __init__(self, slow: Path) -> None:
        super().__init__()
        self.slow = str(slow)

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        if _packed_source(argv) == self.slow:
            self.execs.append((argv, timeout_sec))
            # What DockerSandbox.exec and DaytonaSandbox.exec raise.
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        return await super().exec(cmd, user=user, timeout_sec=timeout_sec)


@pytest.mark.asyncio
async def test_a_copy_past_its_deadline_is_an_error_a_separate_verifier_sees(
    tmp_path,
):
    # A timed-out copy raised RuntimeError out of the collection: no manifest
    # was written, so a separate verifier saw no failed collection and graded
    # without /logs/artifacts. Now it is the collection's error, and the
    # separate verifier ends the trial as an error instead of grading it.
    from benchflow.review.evidence import capture_task_evidence
    from benchflow.rollout._separate_verifier import (
        SeparateVerifierError,
        build_transfer_payload,
    )

    logs, workspace, declared, trial = _layout(tmp_path)
    (workspace / "answer.txt").write_text("42\n")
    await capture_task_evidence(
        RecordingTransport(), str(workspace), trial / "evidence"
    )
    env = TimingOutTransport(logs)
    manifest = await _artifacts.collect_artifacts(
        env,
        artifacts=[str(declared)],
        workspace=str(workspace),
        artifacts_dir=directory(trial / "artifacts"),
        manifest_path=trial / _artifacts.MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
    )
    logs_record, declared_record = manifest["collections"]
    assert logs_record["status"] == "error"
    assert logs_record["reason"] == (
        "RuntimeError: Command timed out after 1024 seconds"
    )
    assert "cause" not in logs_record  # the sandbox's, never the solution's
    assert declared_record["status"] == "collected"
    assert (trial / _artifacts.MANIFEST_NAME).is_file()
    with pytest.raises(SeparateVerifierError, match="artifact collection of"):
        build_transfer_payload(trial, tmp_path / "payload.tar")

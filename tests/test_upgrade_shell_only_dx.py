"""The separate verifier on an image without python3 (DF-p4-upgrader-9).

BenchFlow's own scaffold image (ubuntu:24.04) has no python3: the sandbox
clock read printed an IndexError traceback twice per trial, and files written
outside the workspace were no longer listed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import ExecResult

# --- DF-9: the separate verifier on a python3-less image ----------------------


class _ShellOnly:
    """A sandbox with sh, date and find but no python3, as ubuntu:24.04."""

    def __init__(self, date: str = "1792170000.123456789\n") -> None:
        self.date = date
        self.commands: list[str] = []

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        self.commands.append(cmd)
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            return ExecResult(
                return_code=127, stdout="", stderr="sh: 1: python3: not found\n"
            )
        if argv[:2] == ["sh", "-c"] and "date +%s" in argv[2]:
            return ExecResult(return_code=0, stdout=self.date, stderr="")
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            return_code=process.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )


def _separate_rollout(env) -> SimpleNamespace:
    from benchflow.task.config import TaskConfig

    return SimpleNamespace(
        _task=SimpleNamespace(
            config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
        ),
        _env=env,
        _agent_phase_started=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("1792170000.123456789\n1792170000\n", 1792170000.123456789),
        ("1792170000.%N\n1792170000\n", 1792170000.0),  # busybox date
    ],
)
async def test_the_sandbox_clock_needs_no_python(date, expected, caplog) -> None:
    from benchflow.rollout._review import mark_agent_phase_start

    rollout = _separate_rollout(_ShellOnly(date))
    with caplog.at_level(logging.WARNING):
        await mark_agent_phase_start(rollout)
    assert rollout._agent_phase_started == pytest.approx(expected)
    assert "Could not read the sandbox clock" not in caplog.text


@pytest.mark.asyncio
async def test_an_unreadable_clock_is_one_line_without_a_traceback(caplog) -> None:
    from benchflow.rollout._review import mark_agent_phase_start

    rollout = _separate_rollout(_ShellOnly(""))
    with caplog.at_level(logging.WARNING):
        await mark_agent_phase_start(rollout)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and warned[0].exc_info is None
    assert "Could not read the sandbox clock" in warned[0].getMessage()


@pytest.mark.asyncio
async def test_files_outside_the_workspace_are_listed_without_python(
    tmp_path: Path,
) -> None:
    from benchflow.rollout._separate_verifier import scan_outside_workspace_writes

    root = tmp_path / "box"
    (root / "root").mkdir(parents=True)
    (root / "root" / "proposal.pdf").write_text("an input from before the agent\n")
    time.sleep(1.1)
    since = time.time()
    time.sleep(1.1)
    for path in (
        "root/paper.pdf",  # listed
        "srv/data/fit.csv",  # listed
        "app/out.txt",  # the workspace crosses
        "home/agent/.codex/session.json",  # the agent's home
        "tmp/scratch",
        "usr/lib/x.py",
        "logs/verifier/reward.txt",
        "opt/runtime/node/bin/node",
        "root/.cache/uv/wheel",  # package cache
        "root/.daytona/sessions/s",  # provider runtime state
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("new\n")

    env = _ShellOnly()
    report = await scan_outside_workspace_writes(
        env, since=since, keep=["/app", "/home/agent"], root=str(root)
    )

    assert report is not None, env.commands
    assert report["count"] == 2 and not report["truncated"]
    assert report["by_directory"] == {"/root": 1, "/srv": 1}
    assert set(report["sample"]) == {"/root/paper.pdf", "/srv/data/fit.csv"}


class _WithPython(_ShellOnly):
    """The same sandbox with python3: the listing's own script runs."""

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            self.commands.append(cmd)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                *argv[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return ExecResult(
                return_code=process.returncode or 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )
        return await super().exec(cmd, user=user, timeout_sec=timeout_sec)


@pytest.mark.asyncio
@pytest.mark.parametrize("env_class", [_WithPython, _ShellOnly], ids=["python", "find"])
async def test_both_listings_skip_the_same_harness_trees(
    tmp_path: Path, env_class
) -> None:
    """The python3 listing and its find form skip the same BenchFlow trees.

    Each is the runtime root, a folder or file whose name starts with it and
    "-" or "_", or one of root's /.runtime stages, under the names BenchFlow
    uses now: its own staging and egress records are never reported as the
    agent's. A look-alike name and the trees' former names are listed.
    """
    from benchflow.rollout._separate_verifier import scan_outside_workspace_writes

    root = tmp_path / "box"
    (root / "root").mkdir(parents=True)
    time.sleep(1.1)
    since = time.time()
    time.sleep(1.1)
    harness = (
        "opt/runtime/run/env-0123",
        "opt/runtime/rt/RECORD",
        "opt/runtime-egress/denied.jsonl",
        "opt/runtime-otel/received.jsonl",
        "opt/runtime_agentcore_shim.py",
        ".runtime-kit-" + "0" * 32 + "/rt/bin/sh",
        ".runtime_build_snapshot/manifest.json",
    )
    listed = (
        "root/paper.pdf",
        "opt/runtimes/data.csv",
        "opt/benchflow/run/env-0123",
        "opt/benchflow-egress/denied.jsonl",
    )
    for path in (*harness, *listed):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("new\n")

    env = env_class()
    report = await scan_outside_workspace_writes(
        env, since=since, keep=["/app"], root=str(root)
    )

    assert report is not None, env.commands
    ran_find = any(" -xdev " in c for c in env.commands)
    assert ran_find == (env_class is _ShellOnly), env.commands
    assert sorted(report["sample"]) == sorted("/" + p for p in listed)
    assert report["count"] == len(listed)


@pytest.mark.asyncio
async def test_the_overlap_order_records_the_listing_made_after_it_starts(
    tmp_path: Path,
) -> None:
    """The overlap order builds the verifier run before the workspace capture
    lists the outside writes (Rollout.verify); the record still has them."""
    from benchflow.rollout._separate_verifier import OVERLAP, VerifierSandboxRun
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    run = VerifierSandboxRun(rollout, order=OVERLAP, create_environment=lambda *_: box)
    # What capture_terminal_workspace lists after the run was built.
    rollout._outside_workspace_writes = {
        "count": 1,
        "truncated": False,
        "by_directory": {"/root": 1},
        "sample": ["/root/paper.pdf"],
    }

    async def verify(env, vtask, vpaths, timing, *, workspace):
        return {"reward": 1.0}, None

    await run.finish(verify)

    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["order"] == OVERLAP
    assert record["outside_workspace_writes"]["sample"] == ["/root/paper.pdf"]


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", [None, "/root/answer.txt"])
async def test_files_outside_a_saved_workspace_are_named_for_the_evidence(
    monkeypatch, artifact
) -> None:
    """Seen in a benchmark: tasks wrote answers to /root while the
    workspace was /app, so the saved evidence a review or regrade reads had
    none of them. A saved workspace now lists such files in a warning (folders
    and counts only), unless the task declares them as artifacts."""
    from benchflow.rollout import _review
    from benchflow.task.config import TaskConfig

    seen = {}

    async def scan(env, *, since, keep):
        seen["keep"] = keep
        return {
            "count": 1,
            "by_directory": {"/root": 1},
            "sample": ["/root/answer.txt"],
        }

    monkeypatch.setattr(_review, "scan_outside_workspace_writes", scan)
    rollout = SimpleNamespace(
        _config=SimpleNamespace(freeze_workspace=True, sandbox_user=None),
        _task=SimpleNamespace(
            config=TaskConfig.model_validate(
                {"artifacts": [artifact]} if artifact else {}
            )
        ),
        _review_plan=None,
        _env=object(),
        _agent_cwd="/app",
        _agent_phase_started=1792170000.0,
        _warnings=[],
    )
    await _review._list_outside_workspace_writes(rollout)
    assert seen["keep"][:1] == ["/app"]
    if artifact:
        assert artifact in seen["keep"]
    (warning,) = rollout._warnings
    assert "not in the saved evidence" in warning
    assert "{'/root': 1}" in warning
    assert "answer.txt" not in warning

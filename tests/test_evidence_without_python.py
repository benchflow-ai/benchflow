"""Workspace and artifact capture in task images without Python.

Capture ran a stdlib Python script inside the sandbox. Harbor's own
separate-verifier examples use ``FROM ubuntu:24.04`` with no network, so
Python can be neither found nor installed and every such trial errored
before the agent ran. When ``python3`` is missing, capture now falls back to
``tar`` in the sandbox and does the enumeration, exclusions and limits on the
host, producing the same manifest.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from benchflow.review.evidence import (
    EvidenceError,
    capture_task_evidence,
    capture_workspace,
)
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.sandbox.protocol import ExecResult
from benchflow.task.config import ArtifactConfig


class LocalSandbox:
    """This machine as the sandbox; ``python3`` can be made to not exist."""

    is_mounted = False

    def __init__(self, *, python: bool) -> None:
        self.python = python
        self.commands: list[str] = []

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        self.commands.append(cmd)
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            if not self.python:
                return ExecResult(127, "", "sh: 1: python3: not found\n")
            argv[0] = sys.executable
        env = dict(os.environ)
        if not self.python:
            # Hide every python from the shell fallback, too.
            env["PATH"] = "/usr/bin:/bin"
        # macOS bsdtar would add AppleDouble ._ files; sandbox tars do not.
        env["COPYFILE_DISABLE"] = "1"
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode or 0, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)


def _tree(root: Path) -> Path:
    workspace = root / "app"
    (workspace / "src" / "pkg").mkdir(parents=True)
    (workspace / "src" / "pkg" / "mod.py").write_text("x = 1\n")
    (workspace / "answer.txt").write_text("42\n")
    (workspace / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (workspace / "run.sh").chmod(0o755)
    (workspace / "empty-dir").mkdir()
    (workspace / "link-to-answer").symlink_to("answer.txt")
    os.link(workspace / "answer.txt", workspace / "hardlink.txt")
    (workspace / ".codex").mkdir()
    (workspace / ".codex" / "auth.json").write_text('{"token": "secret"}\n')
    (workspace / "scratch.tmp").write_text("drop\n")
    return workspace


def _entries(manifest) -> list[tuple]:
    return [(e.path, e.kind, e.size, e.sha256, e.link_target) for e in manifest.entries]


@pytest.mark.asyncio
async def test_tar_fallback_captures_the_same_manifest_as_python(tmp_path: Path):
    workspace = _tree(tmp_path / "box")

    with_python = await capture_workspace(
        LocalSandbox(python=True), str(workspace), tmp_path / "a", exclude=["*.tmp"]
    )
    box = LocalSandbox(python=False)
    without = await capture_workspace(
        box, str(workspace), tmp_path / "b", exclude=["*.tmp"]
    )

    assert any(c.startswith("sh ") or "tar " in c for c in box.commands)
    assert without.workspace == with_python.workspace
    assert _entries(without) == _entries(with_python)
    assert sorted(x.model_dump_json() for x in without.exclusions) == sorted(
        x.model_dump_json() for x in with_python.exclusions
    )
    assert (tmp_path / "b" / "workspace" / "hardlink.txt").read_text() == "42\n"
    assert os.access(tmp_path / "b" / "workspace" / "run.sh", os.X_OK)
    assert not (tmp_path / "b" / "workspace" / ".codex" / "auth.json").exists()


@pytest.mark.asyncio
async def test_tar_fallback_captures_a_single_file(tmp_path: Path):
    workspace = _tree(tmp_path / "box")
    manifest = await capture_workspace(
        LocalSandbox(python=False), str(workspace / "answer.txt"), tmp_path / "one"
    )
    assert manifest.workspace == str(workspace.resolve())
    assert [e.path for e in manifest.entries] == ["answer.txt"]


@pytest.mark.asyncio
async def test_tar_fallback_refuses_a_symlink_that_escapes(tmp_path: Path):
    workspace = _tree(tmp_path / "box")
    (tmp_path / "secret").write_text("host\n")
    (workspace / "escape").symlink_to(tmp_path / "secret")
    with pytest.raises(EvidenceError, match="escapes"):
        await capture_workspace(
            LocalSandbox(python=False), str(workspace), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_tar_fallback_enforces_capture_limits(tmp_path: Path):
    workspace = _tree(tmp_path / "box")
    with pytest.raises(EvidenceError, match="limits"):
        await capture_workspace(
            LocalSandbox(python=False), str(workspace), tmp_path / "out", max_entries=3
        )


@pytest.mark.asyncio
async def test_task_evidence_and_artifacts_without_python(tmp_path: Path):
    workspace = _tree(tmp_path / "box")
    outside = tmp_path / "box" / "tmp"
    outside.mkdir()
    (outside / "configured.txt").write_text("declared\n")
    logs = tmp_path / "box" / "logs" / "artifacts"
    logs.mkdir(parents=True)
    (logs / "mode.txt").write_text("separate\n")
    artifacts = [
        str(outside / "configured.txt"),
        "answer.txt",
        ArtifactConfig(source="/nowhere/missing.txt"),
    ]

    results = {}
    for python in (True, False):
        trial = tmp_path / f"trial-{python}"
        (trial / "artifacts").mkdir(parents=True)
        box = LocalSandbox(python=python)
        evidence = await capture_task_evidence(
            box, str(workspace), trial / "evidence", artifacts=artifacts
        )
        collected = await collect_artifacts(
            box,
            artifacts=artifacts,
            workspace=str(workspace),
            artifacts_dir=trial / "artifacts",
            manifest_path=trial / MANIFEST_NAME,
            mounted=False,
            logs_source=str(logs),
        )
        results[python] = (evidence, collected, trial)

    (ev_py, col_py, _), (ev_sh, col_sh, trial_sh) = results[True], results[False]
    assert [a.model_dump() for a in ev_sh.artifacts] == [
        a.model_dump() for a in ev_py.artifacts
    ]
    assert [a.status for a in ev_sh.artifacts] == [
        "captured",
        "in_workspace",
        "missing",
    ]
    assert json.dumps(col_sh["collections"], sort_keys=True) == json.dumps(
        col_py["collections"], sort_keys=True
    )
    assert sorted(f["path"] for f in col_sh["files"]) == sorted(
        f["path"] for f in col_py["files"]
    )
    assert (trial_sh / "artifacts" / "mode.txt").read_text() == "separate\n"

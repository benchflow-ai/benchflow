"""Harbor ``artifacts = [...]`` collection and ``/logs/artifacts`` manifests.

Root artifacts used to be refused before launch. Docker and Daytona now
collect them, together with ``/logs/artifacts``, into the trial folder with a
manifest of sizes and hashes; unsafe paths are still refused.
"""

from __future__ import annotations

import pytest

from benchflow.task import TaskConfig, validate_task_runtime_support


def _artifact_issues(artifacts: list, sandbox: str) -> list:
    config = TaskConfig.model_validate({"artifacts": artifacts})
    return [
        issue
        for issue in validate_task_runtime_support(config, sandbox=sandbox)
        if issue.path.startswith("artifacts")
    ]


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
def test_root_artifacts_are_supported_on_docker_and_daytona(sandbox: str) -> None:
    artifacts = [
        "/app/report.xlsx",
        "out/summary.md",
        {"source": "/app/results", "destination": "results", "exclude": ["*.tmp"]},
    ]
    assert _artifact_issues(artifacts, sandbox) == []


@pytest.mark.parametrize("sandbox", ["modal", "apple-container", "agentcore"])
def test_root_artifacts_stay_refused_where_collection_is_not_implemented(
    sandbox: str,
) -> None:
    issues = _artifact_issues(["/app/report.xlsx"], sandbox)
    assert [i.path for i in issues] == ["artifacts"]
    assert "docker and daytona" in issues[0].reason


@pytest.mark.parametrize(
    "artifact",
    [
        {"source": "/app/out", "destination": "../escape"},
        {"source": "/app/out", "destination": "/etc/cron.d/x"},
        {"source": "/app/../etc/passwd"},
        {"source": ""},
        {"source": "/app/out", "destination": "a/../../b"},
    ],
)
def test_path_traversal_is_refused_before_launch(artifact: dict) -> None:
    issues = _artifact_issues([artifact], "docker")
    assert [i.path for i in issues] == ["artifacts[0]"]
    assert "unsafe" in issues[0].reason


# --- collection ------------------------------------------------------------

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from benchflow.rollout._artifacts import (  # noqa: E402
    MANIFEST_NAME,
    collect_artifacts,
)
from benchflow.sandbox.protocol import ExecResult  # noqa: E402
from benchflow.task.config import ArtifactConfig  # noqa: E402


class LocalTransport:
    """The sandbox is this machine: the collector's stdlib scripts run as-is."""

    def __init__(self) -> None:
        self.downloaded: list[str] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        self.downloaded.append(src)
        shutil.copyfile(src, dst)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    box = tmp_path / "box"
    logs = box / "logs" / "artifacts"
    logs.mkdir(parents=True)
    workspace = box / "app"
    workspace.mkdir()
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    return logs, workspace, trial


async def _collect(env, logs, workspace, trial, artifacts=(), **kwargs):
    return await collect_artifacts(
        env,
        artifacts=list(artifacts),
        workspace=str(workspace),
        artifacts_dir=trial / "artifacts",
        manifest_path=trial / MANIFEST_NAME,
        mounted=kwargs.pop("mounted", False),
        logs_source=str(logs),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_logs_artifacts_are_downloaded_with_a_hashed_manifest(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (logs / "answer.json").write_text('{"x": 1}\n')
    (logs / "plots").mkdir()
    (logs / "plots" / "fig.png").write_bytes(bytes(range(200)))

    manifest = await _collect(LocalTransport(), logs, workspace, trial)

    copied = trial / "artifacts" / "plots" / "fig.png"
    assert copied.read_bytes() == bytes(range(200))
    assert json.loads((trial / MANIFEST_NAME).read_text()) == manifest
    [logs_record] = manifest["collections"]
    assert logs_record["kind"] == "logs" and logs_record["status"] == "collected"
    files = {f["path"]: f for f in manifest["files"] if f["kind"] == "file"}
    assert files["plots/fig.png"] == {
        "path": "plots/fig.png",
        "kind": "file",
        "size": 200,
        "sha256": _sha(copied),
        "collection": 0,
    }
    assert manifest["total_bytes"] == 200 + len('{"x": 1}\n')
    assert manifest["total_files"] == 2


@pytest.mark.asyncio
async def test_declared_artifacts_honour_destination_exclude_and_missing(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "report.xlsx").write_bytes(b"xlsx-bytes")
    outside = tmp_path / "box" / "results"
    outside.mkdir()
    (outside / "keep.csv").write_text("a,b\n")
    (outside / "scratch.tmp").write_text("drop me\n")

    manifest = await _collect(
        LocalTransport(),
        logs,
        workspace,
        trial,
        artifacts=[
            "report.xlsx",
            ArtifactConfig(
                source=str(outside), destination="out/results", exclude=["*.tmp"]
            ),
            "/nowhere/missing.txt",
        ],
    )

    art = trial / "artifacts"
    assert (art / "report.xlsx").read_bytes() == b"xlsx-bytes"
    assert (art / "out/results/keep.csv").read_text() == "a,b\n"
    assert not (art / "out/results/scratch.tmp").exists()
    statuses = [(c["kind"], c["status"]) for c in manifest["collections"]]
    assert statuses == [
        ("logs", "empty"),
        ("declared", "collected"),
        ("declared", "collected"),
        ("declared", "missing"),
    ]
    assert manifest["collections"][1]["source"] == str(workspace / "report.xlsx")
    paths = {f["path"] for f in manifest["files"]}
    assert {"report.xlsx", "out/results/keep.csv"} <= paths
    assert "out/results/scratch.tmp" not in paths


@pytest.mark.asyncio
async def test_escaping_symlink_is_listed_and_its_target_never_copied(tmp_path):
    """Since #1130, capture leaves out a symlink that points out of the
    collected tree instead of failing on it. The collection keeps the rest,
    lists the link in its exclusions, and copies no outside byte."""
    logs, workspace, trial = _layout(tmp_path)
    secret = tmp_path / "host-secret"
    secret.write_text("do not copy\n")
    evil = workspace / "evil"
    evil.mkdir()
    (evil / "link").symlink_to(secret)
    (evil / "kept.txt").write_text("kept\n")

    manifest = await _collect(
        LocalTransport(), logs, workspace, trial, artifacts=["evil"]
    )

    record = manifest["collections"][1]
    assert record["status"] == "collected"
    assert record["exclusions"] == [
        {
            "original_path": str(evil / "link"),
            "reason": "symlink_escape",
            "link_target": str(secret),
        }
    ]
    assert (trial / "artifacts" / "evil" / "kept.txt").read_text() == "kept\n"
    assert not (trial / "artifacts" / "evil" / "link").is_symlink()
    assert all(
        "do not copy" not in p.read_text()
        for p in trial.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    )


@pytest.mark.asyncio
async def test_size_limit_refuses_the_whole_collection_without_partial_files(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "big").mkdir()
    (workspace / "big" / "a.bin").write_bytes(b"0" * 600)
    (workspace / "big" / "b.bin").write_bytes(b"1" * 600)

    manifest = await _collect(
        LocalTransport(), logs, workspace, trial, artifacts=["big"], max_bytes=1000
    )

    record = manifest["collections"][1]
    assert record["status"] == "error"
    assert "limit" in record["reason"]
    assert not (trial / "artifacts" / "big").exists()
    assert manifest["limits"] == {"max_bytes": 1000, "max_files": 10_000}


@pytest.mark.asyncio
async def test_destination_collision_and_unsafe_destination_are_refused(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    (logs / "report.xlsx").write_text("from logs\n")
    (workspace / "report.xlsx").write_text("from workspace\n")

    manifest = await _collect(
        LocalTransport(),
        logs,
        workspace,
        trial,
        artifacts=[
            "report.xlsx",
            ArtifactConfig(source="report.xlsx", destination="../../escaped"),
        ],
    )

    assert (trial / "artifacts" / "report.xlsx").read_text() == "from logs\n"
    assert [c["status"] for c in manifest["collections"]] == [
        "collected",
        "refused",
        "refused",
    ]
    assert "exists" in manifest["collections"][1]["reason"]
    assert "unsafe" in manifest["collections"][2]["reason"]
    assert not (tmp_path / "escaped").exists()


@pytest.mark.asyncio
async def test_mounted_logs_are_hashed_in_place_without_download(tmp_path):
    logs, workspace, trial = _layout(tmp_path)
    art = trial / "artifacts"
    (art / "answer.txt").write_text("42\n")
    (art / "outside-link").symlink_to(tmp_path / "host-secret")
    env = LocalTransport()

    manifest = await _collect(env, logs, workspace, trial, mounted=True)

    assert env.downloaded == []
    assert manifest["collections"][0]["status"] == "collected"
    files = {f["path"]: f for f in manifest["files"]}
    assert files["answer.txt"]["sha256"] == _sha(art / "answer.txt")
    assert files["outside-link"]["kind"] == "symlink"
    assert files["outside-link"]["escapes"] is True
    assert "sha256" not in files["outside-link"]


@pytest.mark.asyncio
async def test_rollout_verify_collects_artifacts_after_freeze_before_verifier(
    tmp_path, monkeypatch
):
    """Artifacts are the agent's final state: collected after the workspace
    freeze and before verifier hardening can change anything."""
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task import RolloutPaths, Task

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Produce a file.")
    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=rollout._rollout_dir)
    rollout._env = SimpleNamespace()
    rollout._planes = SimpleNamespace()
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]

    order: list[str] = []

    def step(name, value=None):
        async def run(*args, **kwargs):
            order.append(name)
            return value

        return run

    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", step("freeze"))
    monkeypatch.setattr(
        "benchflow.rollout.collect_rollout_artifacts", step("artifacts")
    )
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    monkeypatch.setattr(
        "benchflow.rollout._verify_rollout",
        step("verifier", ({"reward": 1.0}, None, None)),
    )

    assert await rollout.verify() == {"reward": 1.0}
    assert order == ["freeze", "artifacts", "verifier"]


@pytest.mark.asyncio
async def test_a_rollout_whose_agent_errored_keeps_its_artifacts(tmp_path):
    """Cleanup collects what an agent left when verification never ran.

    Artifacts were collected only when the verifier phase started, which an
    agent error skips, so a crashed trial kept no session log and its cost
    stayed unknown (the hill-climb demo's gap, docs/examples/hillclimb).
    """
    from datetime import datetime

    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task import RolloutPaths, Task

    _logs, workspace, trial = _layout(tmp_path)
    (workspace / "session.jsonl").write_text('{"type": "assistant"}\n')
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        f'version = "1.0"\nartifacts = ["{workspace / "session.jsonl"}"]\n'
    )
    (task / "instruction.md").write_text("Produce a file.")

    class Box(LocalTransport):
        stopped = False

        async def stop(self, *, delete: bool = True) -> None:
            Box.stopped = True

    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = trial
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=trial)
    rollout._env = Box()
    rollout._agent_cwd = str(workspace)

    await rollout.cleanup()  # the agent errored: verify() never ran

    assert (trial / "artifacts" / "session.jsonl").read_text() == (
        '{"type": "assistant"}\n'
    )
    assert json.loads((trial / MANIFEST_NAME).read_text())["total_files"] == 1
    assert Box.stopped

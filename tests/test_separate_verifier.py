"""Separate verifier sandboxes: task.toml ``[verifier] environment_mode =
"separate"``.

The verifier runs in its own sandbox, built from the task's ``tests/`` (or a
declared verifier image), and receives only the frozen workspace, the
declared artifacts and ``/logs/artifacts`` from the agent's sandbox. Nothing
else the agent left behind reaches it. A failed transfer is an assessment
error (no reward), not a 0, unless the solution's own files caused it: then,
judged against the same paths measured before the agent ran, it scores 0.

The tests in the last part come from main (BenchFlow 0.7.10 and 0.7.11, PRs
#1185 and #1188): ``--separate-verifier``, ``--separate-verifier-include`` and
the list of files left outside everything that crosses. They were merged in
by the PY8 main sync; each change to one is commented where it is made.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    SolutionTransferRefused,
    _install_script,
    build_transfer_payload,
    measure_outputs,
    plan_verifier_image,
    run_separate_verifier,
    scan_outside_workspace_writes,
    separate_verifier_requested,
)
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import TaskConfig, validate_task_runtime_support
from benchflow.task.config import ArtifactConfig
from benchflow.task.verifier_sandbox import (
    apply_separate_verifier_flag,
    normalize_include_paths,
)
from tests.reachable_fixture import directory

# --- config and launch gate --------------------------------------------------


@pytest.mark.parametrize("source_kind", ["path", "held", "missing"])
def test_publication_reopens_held_verifier_outputs(tmp_path, source_kind):
    """Guards batch-19's merge of d44153b2 with separate-verifier publication."""
    from benchflow._utils.reachable_write import ReachableWriteError
    from benchflow.rollout._separate_verifier import (
        _forget_verifier_folder,
        _publish_verifier_outputs,
    )
    from benchflow.task.paths import RolloutPaths

    rollout_dir = tmp_path / "trial"
    paths = RolloutPaths(rollout_dir)
    paths.mkdir()
    original = paths.verifier_dir
    (original / "planted.txt").write_text("agent output")
    source_dir = tmp_path / "verified"
    source_dir.mkdir()
    (source_dir / "reward.txt").write_text("1")
    source = source_dir
    if source_kind == "held":
        source = directory(source_dir)
    elif source_kind == "missing":
        source = None
    record = {}
    _publish_verifier_outputs(source, rollout_dir, record)
    _forget_verifier_folder(SimpleNamespace(_rollout_paths=paths))
    paths.reward_json_path.write_text('{"reward": 1}')

    assert (rollout_dir / "verifier" / "reward.json").read_text() == '{"reward": 1}'
    assert not (rollout_dir / "verifier" / "planted.txt").exists()
    moved = rollout_dir / record["planted_in_verifier_dir"]["moved_to"]
    assert (moved / "planted.txt").read_text() == "agent output"
    assert not (moved / "reward.json").exists()
    with pytest.raises(ReachableWriteError, match="swapped"):
        (original / "planted.txt").read_text()
    if source_kind != "missing":
        assert paths.reward_text_path.read_text() == "1"


def _task_dir(tmp_path: Path, toml: str, *, tests_dockerfile: bool = True) -> Path:
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    if tests_dockerfile:
        (task / "tests" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n"
        )
    (task / "instruction.md").write_text("do it\n")
    (task / "task.toml").write_text('version = "1.0"\n' + toml)
    return task


def _separate_issues(config: TaskConfig, sandbox: str, task_dir: Path | None = None):
    return [
        issue
        for issue in validate_task_runtime_support(
            config, sandbox=sandbox, task_dir=task_dir
        )
        if issue.path.startswith("verifier")
    ]


def test_mode_is_requested_explicitly_or_by_a_verifier_sandbox() -> None:
    assert not separate_verifier_requested(TaskConfig())
    assert separate_verifier_requested(
        TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    # task.toml: [verifier.environment] without a mode implies separate.
    assert separate_verifier_requested(
        TaskConfig.model_validate({"verifier": {"sandbox": {"cpus": 1}}})
    )
    assert not separate_verifier_requested(
        TaskConfig.model_validate({"verifier": {"sandbox_mode": "shared"}})
    )


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
def test_separate_mode_is_no_longer_refused_on_docker_and_daytona(
    tmp_path: Path, sandbox: str
) -> None:
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    from benchflow.task import Task

    config = Task(task).config
    assert _separate_issues(config, sandbox, task) == []


@pytest.mark.parametrize("sandbox", ["apple-container", "agentcore"])
def test_separate_mode_stays_refused_where_it_is_not_implemented(
    tmp_path: Path, sandbox: str
) -> None:
    config = TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    issues = _separate_issues(config, sandbox)
    assert [i.path for i in issues] == ["verifier.sandbox_mode"]
    # PY8 also runs the separate verifier on remote-docker, and Modal runs it
    # too (tests/test_modal_separate_verifier.py).
    assert "daytona, modal and docker" in issues[0].reason
    assert "remote-docker" in issues[0].reason


def test_separate_mode_without_an_image_source_is_refused_before_launch(
    tmp_path: Path,
) -> None:
    # A [verifier.sandbox] without docker_image and no tests/Dockerfile: there
    # is nothing to build the verifier sandbox from.
    task = _task_dir(tmp_path, "[verifier.sandbox]\ncpus = 1\n", tests_dockerfile=False)
    from benchflow.task import Task

    issues = _separate_issues(Task(task).config, "daytona", task)
    assert [i.path for i in issues] == ["verifier.sandbox"]
    assert "tests/Dockerfile" in issues[0].reason


def test_llm_judge_verifiers_cannot_use_a_separate_sandbox() -> None:
    config = TaskConfig.model_validate(
        {"verifier": {"sandbox_mode": "separate", "type": "llm-judge"}}
    )
    issues = _separate_issues(config, "docker")
    assert [i.path for i in issues] == ["verifier.sandbox_mode"]
    assert "test-script" in issues[0].reason


# --- which image the verifier sandbox uses -----------------------------------


def test_image_precedence_follows_task_toml_order(tmp_path: Path) -> None:
    from benchflow.task import Task

    # 1. [verifier.sandbox].docker_image
    task = _task_dir(
        tmp_path / "a",
        '[verifier.sandbox]\ndocker_image = "ghcr.io/x/verifier:1"\ncpus = 2\n'
        '[environment]\ndocker_image = "ghcr.io/x/agent:1"\n',
    )
    plan = plan_verifier_image(Task(task).config, task)
    assert (plan.source, plan.sandbox.docker_image, plan.context_dir) == (
        "verifier.sandbox.docker_image",
        "ghcr.io/x/verifier:1",
        None,
    )
    assert plan.sandbox.cpus == 2

    # 2. no [verifier.sandbox]: a fresh copy of [environment]; its prebuilt
    #    image wins over the tests/ build context.
    task = _task_dir(
        tmp_path / "b",
        '[verifier]\nenvironment_mode = "separate"\n'
        '[environment]\ndocker_image = "ghcr.io/x/agent:1"\n',
    )
    plan = plan_verifier_image(Task(task).config, task)
    assert (plan.source, plan.sandbox.docker_image) == (
        "sandbox.docker_image",
        "ghcr.io/x/agent:1",
    )

    # 3. built from tests/Dockerfile.
    task = _task_dir(tmp_path / "c", '[verifier]\nenvironment_mode = "separate"\n')
    plan = plan_verifier_image(Task(task).config, task)
    assert (plan.source, plan.context_dir) == ("tests/Dockerfile", task / "tests")

    # 4. BenchFlow extension: no tests/Dockerfile and no [verifier.sandbox] ->
    #    a fresh sandbox from the task's own environment/ image.
    task = _task_dir(
        tmp_path / "d",
        '[verifier]\nenvironment_mode = "separate"\n',
        tests_dockerfile=False,
    )
    plan = plan_verifier_image(Task(task).config, task)
    assert (plan.source, plan.context_dir) == (
        "environment/Dockerfile",
        task / "environment",
    )


def test_agent_only_sandbox_settings_do_not_reach_the_verifier_sandbox(
    tmp_path: Path,
) -> None:
    from benchflow.task import Task

    task = _task_dir(
        tmp_path,
        '[verifier]\nenvironment_mode = "separate"\n'
        '[environment]\nskills_dir = "/skills"\n'
        '[[environment.setup_commands]]\ncommand = "echo agent-only"\n',
    )
    plan = plan_verifier_image(Task(task).config, task)
    assert plan.sandbox.setup_commands == []
    assert plan.sandbox.skills_dir is None
    assert plan.sandbox.mcp_servers == []


# --- what crosses from the agent sandbox to the verifier sandbox ------------


class LocalTransport:
    """The agent sandbox is this machine (see test_artifacts_collection)."""

    is_mounted = False

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode or 0, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)


async def _agent_trial(tmp_path: Path, *, artifacts=()) -> tuple[Path, Path, Path]:
    """An agent sandbox rooted at ``box/`` whose agent solved *and* tampered."""
    box = tmp_path / "box"
    workspace = box / "app"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("42\n")
    # Planted in the workspace: travels (it is the agent's work), and the
    # verifier sandbox's hardening removes it before tests run.
    (workspace / "conftest.py").write_text("raise SystemExit(0)\n")
    logs = box / "logs" / "artifacts"
    logs.mkdir(parents=True)
    (logs / "mode.txt").write_text("separate\n")
    # Planted outside the workspace and outside every declared artifact: must
    # never reach the verifier sandbox.
    (box / "logs" / "verifier").mkdir(parents=True)
    (box / "logs" / "verifier" / "reward.txt").write_text("1\n")
    (box / "tests").mkdir()
    (box / "tests" / "conftest.py").write_text("raise SystemExit(0)\n")
    (box / "usr-lib-python" / "site-packages").mkdir(parents=True)
    (box / "usr-lib-python" / "site-packages" / "evil.pth").write_text("import os\n")
    (box / "tmp").mkdir()
    (box / "tmp" / "configured.txt").write_text("declared artifact\n")
    (box / "tmp" / "agent-only.txt").write_text("ambient file\n")

    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    env = LocalTransport()
    await capture_task_evidence(
        env, str(workspace), trial / "evidence", artifacts=list(artifacts)
    )
    await collect_artifacts(
        env,
        artifacts=list(artifacts),
        workspace=str(workspace),
        artifacts_dir=directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
    )
    return box, workspace, trial


def _members(archive: Path) -> dict[str, bytes | None]:
    with tarfile.open(archive) as tar:
        return {
            m.name: (tar.extractfile(m).read() if m.isfile() else None)  # type: ignore[union-attr]
            for m in tar.getmembers()
        }


@pytest.mark.asyncio
async def test_payload_holds_only_workspace_declared_artifacts_and_logs_artifacts(
    tmp_path: Path,
) -> None:
    box, workspace, trial = await _agent_trial(
        tmp_path, artifacts=[str(tmp_path / "box" / "tmp" / "configured.txt")]
    )

    summary = build_transfer_payload(trial, tmp_path / "payload.tar")

    members = _members(tmp_path / "payload.tar")
    root = str(box).lstrip("/")
    files = {name for name, data in members.items() if data is not None}
    assert files == {
        f"{root}/app/src/answer.txt",
        f"{root}/app/conftest.py",
        f"{root}/tmp/configured.txt",
        "logs/artifacts/mode.txt",
    }
    assert members[f"{root}/app/src/answer.txt"] == b"42\n"
    assert members["logs/artifacts/mode.txt"] == b"separate\n"
    assert summary["workspace"] == str(workspace)
    assert summary["files"] == 4
    assert summary["declared_artifacts"] == 1
    assert summary["logs_artifacts"] == 1
    # Nothing from outside the transfer set, and nothing absolute or escaping.
    for name in members:
        assert not name.startswith("/") and ".." not in name.split("/")
        assert "reward.txt" not in name and "evil.pth" not in name
        assert "agent-only" not in name and not name.endswith("tests/conftest.py")


@pytest.mark.asyncio
async def test_agent_paths_cover_every_path_the_transfer_writes(tmp_path: Path) -> None:
    """Guards the separate-mode trust rule (sdk-update review, must-fix 3).

    The verifier sandbox's plugin guard distrusts only ``agent_paths`` (and
    /logs): no agent process ever ran there. So every member the transfer
    unpacks must lie under one of them, declared artifacts outside the
    workspace included.
    """
    box, workspace, trial = await _agent_trial(
        tmp_path, artifacts=[str(tmp_path / "box" / "tmp" / "configured.txt")]
    )

    summary = build_transfer_payload(trial, tmp_path / "payload.tar")

    agent_paths = summary["agent_paths"]
    # A declared file is captured with its directory as the bundle's root,
    # so that whole directory counts as the agent's: conservative, and what
    # the shared sandbox blocks for /tmp anyway.
    assert agent_paths == [str(workspace), str(box / "tmp"), "/logs/artifacts"]
    for name in _members(tmp_path / "payload.tar"):
        path = "/" + name
        assert any(
            path == root or path.startswith(root + "/") for root in agent_paths
        ) or any(root.startswith(path + "/") for root in agent_paths), name


@pytest.mark.asyncio
async def test_default_verify_hardens_with_the_transferred_paths(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards the separate-mode trust rule: hardening is told the sandbox is fresh."""
    from benchflow.rollout import _separate_verifier, _setup

    calls = []

    async def fake_verify_rollout(*args, **kwargs):
        calls.append(kwargs)
        return {"reward": 1.0}, None, None

    async def fake_publish(*args, **kwargs):
        return None

    monkeypatch.setattr(_setup, "_verify_rollout", fake_verify_rollout)
    monkeypatch.setattr(_setup, "_publish_trajectory_for_verifier", fake_publish)
    rollout = SimpleNamespace(
        _trajectory=[],
        _planes=SimpleNamespace(),
        _config=SimpleNamespace(primary_agent="codex-acp"),
    )
    verify = _separate_verifier._default_verify(rollout)

    await verify(
        object(),
        object(),
        SimpleNamespace(agent_dir=directory(tmp_path)),
        {},
        workspace="/app",
        agent_paths=("/app", "/out/report.txt", "/logs/artifacts"),
    )

    (kwargs,) = calls
    assert kwargs["sandbox_user"] is None
    assert kwargs["workspace"] == "/app"
    assert set(kwargs["agent_paths"]) == {"/app", "/out/report.txt", "/logs/artifacts"}


@pytest.mark.asyncio
async def test_missing_frozen_workspace_is_a_transfer_error(tmp_path: Path) -> None:
    _, _, trial = await _agent_trial(tmp_path)
    shutil.rmtree(trial / "evidence")
    with pytest.raises(SeparateVerifierError, match="no frozen workspace"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_tampered_frozen_workspace_is_a_transfer_error(tmp_path: Path) -> None:
    _, _, trial = await _agent_trial(tmp_path)
    (trial / "evidence" / "workspace" / "src" / "answer.txt").write_text("43\n")
    with pytest.raises(SeparateVerifierError, match="does not match"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_changed_logs_artifact_is_a_transfer_error(tmp_path: Path) -> None:
    _, _, trial = await _agent_trial(tmp_path)
    (trial / "artifacts" / "mode.txt").write_text("shared\n")
    with pytest.raises(SeparateVerifierError, match="changed since collection"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_failed_artifact_collection_is_a_transfer_error(tmp_path: Path) -> None:
    _, _, trial = await _agent_trial(tmp_path)
    manifest = json.loads((trial / MANIFEST_NAME).read_text())
    manifest["collections"][0].update(status="error", reason="probe failed")
    (trial / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(SeparateVerifierError, match="probe failed"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_a_declared_artifact_the_agent_never_wrote_is_not_an_error(
    tmp_path: Path,
) -> None:
    # Missing output is the verifier's to judge (it scores 0 on its own).
    _, _, trial = await _agent_trial(tmp_path, artifacts=["/nowhere/missing.txt"])
    summary = build_transfer_payload(trial, tmp_path / "payload.tar")
    assert summary["declared_artifacts"] == 0
    assert summary["missing_artifacts"] == ["/nowhere/missing.txt"]


@pytest.mark.asyncio
async def test_the_workspace_replaces_the_verifier_images_copy(tmp_path: Path) -> None:
    """Guards the install in the verifier sandbox, whose image holds its own
    copy of a workspace such as /app or /testbed. The frozen workspace was
    laid over that copy, so a file the agent deleted or renamed came back
    from the image, and every file came out 0644 or 0755. It now replaces
    the copy's contents (the directory itself stays), and files and folders
    get the modes the solver left, as a regrade restores them."""
    import subprocess

    from benchflow.rollout._separate_verifier import _install_script

    workspace = tmp_path / "box" / "app"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("42\n")
    (workspace / "new-name.txt").write_text("renamed by the agent\n")
    (workspace / "new-name.txt").chmod(0o640)
    (workspace / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (workspace / "run.sh").chmod(0o755)
    (workspace / "key.pem").write_text("private\n")
    (workspace / "key.pem").chmod(0o600)
    (workspace / "private").mkdir(mode=0o700)
    (workspace / "private" / "note.txt").write_text("kept\n")
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    env = LocalTransport()
    await capture_task_evidence(env, str(workspace), trial / "evidence")
    await collect_artifacts(
        env,
        artifacts=[],
        workspace=str(workspace),
        artifacts_dir=directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=False,
        logs_source=str(tmp_path / "box" / "logs" / "artifacts"),
    )
    # As if captured from the agent sandbox's /app.
    manifest_path = trial / "evidence" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workspace"] = "/app"
    manifest_path.write_text(json.dumps(manifest))
    payload = tmp_path / "payload.tar"
    summary = build_transfer_payload(trial, payload)
    assert summary["workspace"] == "/app"

    # The verifier image's own copy of /app, from before the agent.
    root = tmp_path / "verifier-root"
    image = root / "app"
    (image / "src").mkdir(parents=True)
    (image / "src" / "answer.txt").write_text("TODO\n")
    (image / "deleted.txt").write_text("the agent deleted this\n")
    (image / "old-name.txt").write_text("renamed by the agent\n")
    (image / ".image-dotfile").write_text("stale\n")
    inode = image.stat().st_ino

    script = _install_script(
        str(payload), summary["workspace"], summary["sha256"], root=str(root)
    )
    # The sandbox runs it as root, whose tar ignores the umask.
    subprocess.run(["sh", "-c", "umask 000\n" + script], check=True)

    assert sorted(p.name for p in image.iterdir()) == [
        "key.pem",
        "new-name.txt",
        "private",
        "run.sh",
        "src",
    ]
    assert image.stat().st_ino == inode
    assert (image / "src" / "answer.txt").read_text() == "42\n"
    modes = {
        name: (image / name).stat().st_mode & 0o777
        for name in ("key.pem", "run.sh", "new-name.txt", "private")
    }
    assert modes == {
        "key.pem": 0o600,
        "run.sh": 0o755,
        "new-name.txt": 0o640,
        "private": 0o700,
    }
    assert (image / "private" / "note.txt").read_text() == "kept\n"
    assert not payload.exists()
    # A shared directory such as /root is still laid over, never emptied.
    assert "rm -rf" not in _install_script("/tmp/x.tar", "/root", "0" * 64)
    assert "rm -rf" in _install_script("/tmp/x.tar", "/app", "0" * 64)


# --- the solution's own files: its result, not an assessment error ----------


async def _measured_trial(
    tmp_path: Path,
    solve: Callable[[Path, Path, Path], None],
    *,
    artifacts: list = (),  # type: ignore[assignment]
    before: Callable[[Path, Path, Path], None] | None = None,
    max_files: int = 10_000,
) -> tuple[Path, dict[str, Any]]:
    """Measure the agent sandbox before ``solve`` (the clean control), then
    let ``solve`` act as the agent and capture and collect as a rollout does."""
    box = tmp_path / "box"
    workspace = box / "app"
    logs = box / "logs" / "artifacts"
    workspace.mkdir(parents=True)
    logs.mkdir(parents=True)
    (workspace / "task-file.txt").write_text("from the image\n")
    if before is not None:
        before(box, workspace, logs)
    env = LocalTransport()
    pristine = await measure_outputs(
        env, workspace=str(workspace), artifacts=artifacts, logs_source=str(logs)
    )
    solve(box, workspace, logs)
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    await capture_task_evidence(
        env, str(workspace), trial / "evidence", artifacts=list(artifacts)
    )
    await collect_artifacts(
        env,
        artifacts=list(artifacts),
        workspace=str(workspace),
        artifacts_dir=directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
        max_files=max_files,
    )
    return trial, pristine


def _flood_logs(box: Path, workspace: Path, logs: Path) -> None:
    for index in range(3):
        (logs / f"out-{index}.txt").write_text("x\n")


def _link_declared_output(box: Path, workspace: Path, logs: Path) -> None:
    (box / "secret.txt").write_text("outside\n")
    (workspace / "report.txt").symlink_to(box / "secret.txt")


def _take_declared_destination(box: Path, workspace: Path, logs: Path) -> None:
    (workspace / "report.txt").write_text("the declared output\n")
    (logs / "report.txt").write_text("left in /logs/artifacts\n")


def _only_infrastructure_error(exc: BaseException) -> bool:
    return isinstance(exc, SeparateVerifierError) and not isinstance(
        exc, SolutionTransferRefused
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("solve", "artifacts", "cause", "reason"),
    [
        (_flood_logs, [], "limits", "exceed the collection limits"),
        (_link_declared_output, ["report.txt"], "symlink", "is a symlink"),
        (_take_declared_destination, ["report.txt"], "clash", "already exists"),
    ],
)
async def test_a_collection_the_solution_broke_is_its_result_not_unscored(
    tmp_path: Path, solve, artifacts, cause, reason
) -> None:
    """Guards separate-verifier scoring against the solution's own files.

    /logs/artifacts is the agent's (lockdown chowns it to the agent user): more
    files than the collection allows, a declared output left as a symlink, or
    a /logs/artifacts file taking a declared output's place used to end the
    trial unscored, as if the verifier were broken; a policy could reach that
    on purpose. Measured clean before the agent ran, it is the solution's."""
    trial, pristine = await _measured_trial(
        tmp_path, solve, artifacts=artifacts, max_files=2
    )
    collected = json.loads((trial / MANIFEST_NAME).read_text())
    [failed] = [c for c in collected["collections"] if c.get("cause")]
    assert failed["cause"] == cause

    with pytest.raises(SolutionTransferRefused, match=reason):
        build_transfer_payload(trial, tmp_path / "payload.tar", pristine=pristine)
    # Without the clean control nothing is blamed on the solution.
    with pytest.raises(SeparateVerifierError) as unmeasured:
        build_transfer_payload(trial, tmp_path / "payload.tar")
    assert _only_infrastructure_error(unmeasured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "artifacts"),
    [
        (_flood_logs, []),
        (_link_declared_output, ["report.txt"]),
        (_take_declared_destination, ["report.txt"]),
    ],
)
async def test_a_collection_the_clean_control_breaks_too_stays_unscored(
    tmp_path: Path, before, artifacts
) -> None:
    """The same failures, already there before the agent ran (the task's own
    image or setup), are not the solution's: the trial stays unscored."""
    trial, pristine = await _measured_trial(
        tmp_path,
        lambda *paths: None,
        artifacts=artifacts,
        before=before,
        max_files=2,
    )
    with pytest.raises(SeparateVerifierError) as blamed:
        build_transfer_payload(trial, tmp_path / "payload.tar", pristine=pristine)
    assert _only_infrastructure_error(blamed.value)


@pytest.mark.asyncio
async def test_an_infrastructure_failure_beside_the_solutions_stays_unscored(
    tmp_path: Path,
) -> None:
    trial, pristine = await _measured_trial(tmp_path, _flood_logs, max_files=2)
    manifest = json.loads((trial / MANIFEST_NAME).read_text())
    manifest["collections"].append(
        {
            "kind": "declared",
            "source": "/x",
            "status": "error",
            "reason": "probe failed",
        }
    )
    (trial / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(SeparateVerifierError, match="probe failed") as blamed:
        build_transfer_payload(trial, tmp_path / "payload.tar", pristine=pristine)
    assert _only_infrastructure_error(blamed.value)


@pytest.mark.asyncio
async def test_a_workspace_over_the_capture_limits_is_the_solutions_result(
    tmp_path: Path,
) -> None:
    """A workspace over 200,000 entries or 20 GiB fails capture. When it was
    within those limits before the agent ran, the solution made it so."""
    trial, pristine = await _measured_trial(tmp_path, lambda *paths: None)
    shutil.rmtree(trial / "evidence")  # what an over-limit capture leaves
    with pytest.raises(SolutionTransferRefused, match="capture limits"):
        build_transfer_payload(
            trial, tmp_path / "payload.tar", pristine=pristine, capture_over_limit=True
        )
    # Any other capture failure, or no clean control, stays unscored.
    for kwargs in (
        {"pristine": pristine},
        {"capture_over_limit": True},
    ):
        with pytest.raises(SeparateVerifierError) as blamed:
            build_transfer_payload(trial, tmp_path / "payload.tar", **kwargs)
        assert _only_infrastructure_error(blamed.value)
    # A task image whose workspace was already over the limits.
    huge = json.loads(json.dumps(pristine))
    huge["paths"][huge["workspace"]]["entries"] = 300_000
    with pytest.raises(SeparateVerifierError) as blamed:
        build_transfer_payload(
            trial, tmp_path / "payload.tar", pristine=huge, capture_over_limit=True
        )
    assert _only_infrastructure_error(blamed.value)


@pytest.mark.asyncio
async def test_bind_mounted_logs_artifacts_over_the_limits_record_their_cause(
    tmp_path: Path,
) -> None:
    """Docker bind-mounts /logs/artifacts onto the trial folder, so its files
    are only inventoried; over the limits it is the same solution cause."""
    artifacts = tmp_path / "trial" / "artifacts"
    artifacts.mkdir(parents=True)
    for index in range(3):
        (artifacts / f"out-{index}.txt").write_text("x\n")
    manifest = await collect_artifacts(
        LocalTransport(),
        artifacts=[],
        workspace=str(tmp_path),
        artifacts_dir=directory(artifacts),
        manifest_path=tmp_path / "trial" / MANIFEST_NAME,
        mounted=True,
        max_files=2,
    )
    [logs] = manifest["collections"]
    assert (logs["status"], logs["cause"]) == ("over_limit", "limits")


@pytest.mark.asyncio
@pytest.mark.parametrize("over_limit", [True, False])
async def test_terminal_capture_records_whether_it_hit_the_limits(
    tmp_path: Path, monkeypatch, over_limit: bool
) -> None:
    from benchflow.review.evidence import EvidenceError, EvidenceLimitError
    from benchflow.rollout import _review

    async def capture(*args, **kwargs):
        raise (EvidenceLimitError if over_limit else EvidenceError)("capture failed")

    async def idle(*args):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", capture)
    rollout = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=None,
        _config=SimpleNamespace(
            purpose="task", sandbox_user=None, freeze_workspace=False
        ),
        _env=object(),
        _agent_env={},
        _planes=SimpleNamespace(quiesce_agent=idle),
        _task=SimpleNamespace(
            config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
        ),
        _agent_cwd="/app",
        disconnect=idle,
        _require_rollout_dir=lambda: tmp_path,
    )
    await _review.capture_terminal_workspace(rollout)
    assert "capture failed" in rollout._export_error
    assert rollout._capture_over_limit is over_limit


@pytest.mark.asyncio
async def test_the_clean_control_is_measured_before_the_agent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    from benchflow.rollout import _review

    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda rollout: "no contract",
    )
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "task-file.txt").write_text("from the image\n")
    rollout = SimpleNamespace(
        _env=LocalTransport(),
        _review_plan=None,
        _agent_cwd=str(workspace),
        _config=SimpleNamespace(purpose="task", sandbox_setup_timeout=60),
        _task=SimpleNamespace(
            config=TaskConfig.model_validate(
                {"verifier": {"sandbox_mode": "separate"}, "artifacts": ["out.txt"]}
            )
        ),
    )
    await _review.prepare_capture_runtime(rollout)
    pristine = rollout._pristine_outputs
    assert pristine["workspace"] == str(workspace)
    assert pristine["logs"] == "/logs/artifacts"
    assert pristine["paths"][str(workspace)] == {
        "exists": True,
        "link": False,
        "dir": True,
        "entries": 1,
        "bytes": len("from the image\n"),
    }
    assert pristine["paths"][str(workspace / "out.txt")]["exists"] is False


@pytest.mark.asyncio
async def test_the_solutions_refused_outputs_score_zero_without_a_verifier_error(
    tmp_path: Path,
) -> None:
    """End to end: reward 0, no verifier error (so the retry loop keeps the
    0), no verifier sandbox, and the reason in verifier-sandbox.json."""
    from benchflow.evaluation import RetryConfig

    trial, pristine = await _measured_trial(
        tmp_path, _link_declared_output, artifacts=["report.txt"]
    )
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(tmp_path / "box" / "app"))
    rollout._pristine_outputs = pristine
    created: list[Any] = []

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: created.append(a), verify=None
    )

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert not created
    assert not RetryConfig().should_retry_verifier_error(error)
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["status"] == "refused"
    assert record["error"] is None
    assert "is a symlink" in record["refusal"]
    assert record["pristine"] == pristine


# --- the verifier sandbox run -------------------------------------------------


class FakeVerifierSandbox:
    """A verifier sandbox whose filesystem is a local directory."""

    def __init__(self, root: Path, *, fail_exec: str | None = None) -> None:
        self.root = root
        self.sandbox_id = "verifier-box-1"
        self.started = False
        self.stopped: list[bool] = []
        self.commands: list[str] = []
        self.uploads: dict[str, Path] = {}
        self.fail_exec = fail_exec
        self.is_mounted = False

    async def start(self, force_build: bool = False) -> None:
        self.started = True

    async def stop(self, delete: bool = True) -> None:
        self.stopped.append(delete)

    async def upload_file(self, source: Path | str, target: str, **_kwargs) -> None:
        dest = self.root / target.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        self.uploads[target] = dest

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30, **_):
        self.commands.append(cmd)
        if self.fail_exec and self.fail_exec in cmd:
            return ExecResult(1, "", "tar: boom")
        if "benchflow-verifier-destinations-" in cmd:
            # The destination probe runs for real, rooted at this folder.
            script = next(
                p for t, p in reversed(self.uploads.items()) if "destinations" in t
            )
            done = subprocess.run(
                ["sh", str(script), str(self.root.resolve())],
                capture_output=True,
                text=True,
                check=False,
            )
            return ExecResult(done.returncode, done.stdout, done.stderr)
        if "tar " in cmd and " -x" in cmd:
            archive = next(p for t, p in self.uploads.items() if t.endswith(".tar"))
            with tarfile.open(archive) as tar:
                tar.extractall(self.root, filter="tar")
        return ExecResult(0, "", "")


def _fake_rollout(tmp_path: Path, trial: Path, task: Path, workspace: str):
    from unittest.mock import AsyncMock

    from benchflow.task import Task
    from benchflow.task.paths import RolloutPaths

    async def prepare_runtime(env, *, task, workspace):
        from benchflow.sandbox.lockdown import prepare_separate_verifier_runtime

        await prepare_separate_verifier_runtime(env, task, workspace=workspace)

    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    return SimpleNamespace(
        _task=Task(task),
        _config=SimpleNamespace(
            environment="daytona",
            task_path=task,
            sandbox_setup_timeout=60,
            primary_agent="codex",
        ),
        _rollout_dir=trial,
        _rollout_paths=paths,
        _rollout_name="trial-1",
        _agent_cwd=workspace,
        _trajectory=[],
        _timing={},
        _export_error=None,
        _planes=SimpleNamespace(
            prepare_separate_verifier_runtime=prepare_runtime,
            snapshot_build_config=AsyncMock(),
        ),
    )


@pytest.mark.asyncio
async def test_rewards_come_from_the_verifier_sandbox_with_timing_and_record(
    tmp_path: Path,
) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    verifier_root = tmp_path / "verifier-box"
    box = FakeVerifierSandbox(verifier_root)
    created: list[Any] = []

    def create(vtask, context_path, vpaths):
        created.append((vtask, context_path, vpaths))
        return box

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        # Hardening learns what the transfer wrote: only that is the agent's.
        assert workspace in agent_paths and "/logs/artifacts" in agent_paths
        # The verifier sees the agent's work at the same absolute path, and
        # nothing planted elsewhere in the agent sandbox.
        root = verifier_root / workspace.lstrip("/")
        assert (root / "src" / "answer.txt").read_text() == "42\n"
        assert not (
            verifier_root / str(tmp_path / "box" / "logs" / "verifier").lstrip("/")
        ).exists()
        assert (verifier_root / "logs" / "artifacts" / "mode.txt").exists()
        # main: a file planted outside the workspace never crosses.
        assert not (
            verifier_root / str(tmp_path / "box" / "usr-lib-python").lstrip("/")
        ).exists()
        vpaths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (vpaths.verifier_dir / "reward.txt").write_text("1\n")
        timing["verifier"] = 1.5
        return {"reward": 1.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=create, verify=verify
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    vtask, context_path, vpaths = created[0]
    # The verifier sandbox is built from tests/, never from the agent image.
    assert (
        (context_path / "environment" / "Dockerfile")
        .read_text()
        .startswith("FROM ubuntu:24.04")
    )
    assert vtask.config.verifier.sandbox_mode is None
    assert vpaths.rollout_dir != rollout._rollout_paths.rollout_dir
    assert box.started and box.stopped == [True]
    # Verifier outputs are published where result.json readers expect them.
    assert (trial / "verifier" / "reward.txt").read_text() == "1\n"
    timing = rollout._timing
    for key in (
        "verifier_sandbox_setup",
        "verifier_transfer",
        "verifier",
        "verifier_sandbox_teardown",
        "verifier_sandbox_total",
    ):
        assert key in timing, key
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["mode"] == "separate"
    assert record["image_source"] == "tests/Dockerfile"
    assert record["sandbox_id"] == "verifier-box-1"
    assert record["status"] == "complete"
    assert record["transfer"]["files"] == 3
    assert record["sandbox_seconds"] >= 0


@pytest.mark.asyncio
async def test_failed_upload_is_an_assessment_error_not_zero(tmp_path: Path) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box", fail_exec="tar ")
    ran: list[bool] = []

    async def verify(*args, **kwargs):
        ran.append(True)
        return {"reward": 0.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=verify
    )

    assert rewards is None and not ran
    assert error is not None and error.startswith("separate verifier")
    assert "transfer" in error
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert box.stopped == [True]
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_no_frozen_workspace_never_starts_a_verifier_sandbox(
    tmp_path: Path,
) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    shutil.rmtree(trial / "evidence")
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._export_error = "Workspace evidence capture failed: python3 missing"
    created: list[Any] = []

    rewards, error = await run_separate_verifier(
        rollout,
        create_environment=lambda *a: created.append(a),
        verify=None,
    )

    assert rewards is None and not created
    assert error is not None and "python3 missing" in error


@pytest.mark.asyncio
async def test_sandbox_start_failure_is_an_assessment_error(tmp_path: Path) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    async def boom(force_build: bool = False) -> None:
        raise RuntimeError("image build failed")

    box.start = boom  # type: ignore[method-assign]

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=None
    )

    assert rewards is None
    assert error is not None and "image build failed" in error
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert box.stopped == [True]


# --- rollout wiring ----------------------------------------------------------


def test_separate_mode_forces_the_workspace_freeze() -> None:
    from benchflow.rollout._review import _freeze_requested

    shared = SimpleNamespace(
        _config=SimpleNamespace(freeze_workspace=False),
        _task=SimpleNamespace(config=TaskConfig()),
    )
    separate = SimpleNamespace(
        _config=SimpleNamespace(freeze_workspace=False),
        _task=SimpleNamespace(
            config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
        ),
    )
    assert not _freeze_requested(shared)
    assert _freeze_requested(separate)


@pytest.mark.asyncio
async def test_verify_never_runs_tests_in_the_agent_sandbox(monkeypatch) -> None:
    import benchflow.rollout as rollout_mod
    from benchflow.rollout import Rollout

    calls: list[str] = []

    async def fake_separate(rollout):
        calls.append("separate")
        return {"reward": 1.0}, None

    async def fake_shared(*args, **kwargs):
        calls.append("shared")
        return {"reward": 1.0}, None, None

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(rollout_mod, "run_separate_verifier", fake_separate)
    monkeypatch.setattr(rollout_mod, "_verify_rollout", fake_shared)
    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", noop)
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", noop)
    monkeypatch.setattr(rollout_mod, "_publish_trajectory_for_verifier", noop)
    monkeypatch.setattr(rollout_mod, "require_safe_branch_world", lambda r: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda r: None
    )

    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(primary_agent="oracle", sandbox_user=None)
    rollout._muse_credentials_staged = False
    rollout._task = SimpleNamespace(
        config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    rollout._trajectory = [{"x": 1}]
    rollout._env = object()
    rollout._diagnostics = SimpleNamespace(set=lambda d: None)

    rewards = await Rollout.verify(rollout)

    assert rewards == {"reward": 1.0}
    assert calls == ["separate"]
    assert rollout._verifier_error is None


def test_soft_verify_refuses_to_upload_tests_into_the_agent_sandbox() -> None:
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._task = SimpleNamespace(
        config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    rewards, _, error = asyncio.run(Rollout.soft_verify(rollout))
    assert rewards is None
    assert error is not None and "separate verifier" in error


def test_budget_counts_the_verifier_sandbox_seconds(tmp_path: Path) -> None:
    """Both sandboxes are billed while the separate verifier runs."""
    from benchflow.budget import Budget, BudgetGuard

    guard = BudgetGuard(Budget(max_sandbox_seconds=1000))
    guard.seed([{"timing": {"total": 100.0, "verifier_sandbox_total": 40.0}}])
    assert guard.spent()["sandbox_seconds"] == pytest.approx(140.0)
    trial = tmp_path / "t2"
    trial.mkdir()
    (trial / "timing.json").write_text(json.dumps({"verifier_sandbox_total": 25.0}))
    guard.start("t2")
    guard.finish("t2", SimpleNamespace(rollout_dir=trial))
    assert guard.spent()["sandbox_seconds"] == pytest.approx(165.0, abs=1.0)


class _ProbeSandbox:
    def __init__(self, *, python: bool, tar: bool) -> None:
        self.python, self.tar = python, tar
        self.commands: list[str] = []

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        self.commands.append(cmd)
        if "apt-get" in cmd or "apk " in cmd:
            raise AssertionError("separate mode must not install into the agent image")
        ok = self.python or (self.tar and "tar" in cmd)
        return ExecResult(0 if ok else 1, "", "" if ok else "missing")


def _capture_rollout(env) -> Any:
    return SimpleNamespace(
        _env=env,
        _review_plan=None,
        _config=SimpleNamespace(purpose="task", sandbox_setup_timeout=60),
        _task=SimpleNamespace(
            config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("python", "tar"), [(True, False), (False, True)])
async def test_capture_needs_python_or_tar_and_installs_nothing(
    monkeypatch, python: bool, tar: bool
) -> None:
    from benchflow.rollout import _review

    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda rollout: "no contract",
    )
    env = _ProbeSandbox(python=python, tar=tar)
    await _review.prepare_capture_runtime(_capture_rollout(env))
    assert env.commands


@pytest.mark.asyncio
async def test_image_without_python_or_tar_fails_before_the_agent_runs(
    monkeypatch,
) -> None:
    from benchflow.review.evidence import EvidenceError
    from benchflow.rollout import _review

    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda rollout: "no contract",
    )
    with pytest.raises(EvidenceError, match=r"python3 .* or tar"):
        await _review.prepare_capture_runtime(
            _capture_rollout(_ProbeSandbox(python=False, tar=False))
        )


# --- from main: --separate-verifier, --separate-verifier-include and the
# outside-workspace list (BenchFlow 0.7.10 and 0.7.11, PRs #1185 and #1188) --


def test_flag_forces_a_separate_verifier_without_touching_the_task(
    tmp_path: Path,
) -> None:
    from benchflow.task import Task

    task = _task_dir(tmp_path, "", tests_dockerfile=False)
    config = Task(task).config
    assert not separate_verifier_requested(config)

    forced = apply_separate_verifier_flag(config, sandbox="daytona", task_dir=task)

    assert separate_verifier_requested(forced)
    # A copy: the task's own config (and files) stay shared-mode.
    assert not separate_verifier_requested(config)
    assert plan_verifier_image(forced, task).source == "environment/Dockerfile"
    # A task that already asks for one keeps its own verifier settings.
    declared = TaskConfig.model_validate({"verifier": {"sandbox": {"cpus": 1}}})
    assert (
        apply_separate_verifier_flag(declared, sandbox="docker", task_dir=task)
        is declared
    )


def test_flag_is_refused_before_launch_where_it_cannot_run(tmp_path: Path) -> None:
    task = _task_dir(tmp_path, "")
    with pytest.raises(ValueError, match="--separate-verifier") as raised:
        apply_separate_verifier_flag(
            TaskConfig.model_validate({"verifier": {"type": "llm-judge"}}),
            # Modal runs separate verifiers now; Apple Container does not.
            sandbox="apple-container",
            task_dir=task,
        )
    assert "daytona, modal and docker" in str(raised.value)
    assert "test-script" in str(raised.value)


def test_eval_run_separate_verifier_threads_to_evaluation_config(
    tmp_path: Path,
) -> None:
    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation

    task = tmp_path / "flag-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do it.\n")
    captured: dict[str, Any] = {}

    async def capture(self):
        captured["separate_verifier"] = self._config.separate_verifier
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=capture):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(task),
                "--agent",
                "oracle",
                "--separate-verifier",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["separate_verifier"] is True


def test_the_flag_survives_sharding() -> None:
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig

    shard = EvalShard(index=0, task_names=("t",), concurrency=1)
    for flag in (True, False):
        payload = _config_payload(EvaluationConfig(separate_verifier=flag), shard=shard)
        assert _evaluation_config(payload).separate_verifier is flag


# PY8 main sync: expects /logs/artifacts/mode.txt too.
@pytest.mark.asyncio
async def test_payload_holds_only_the_frozen_workspace(tmp_path: Path) -> None:
    box, workspace, trial = await _agent_trial(tmp_path)

    summary = build_transfer_payload(trial, tmp_path / "payload.tar")

    members = _members(tmp_path / "payload.tar")
    root = str(box).lstrip("/")
    files = {name for name, data in members.items() if data is not None}
    # PY8 also carries /logs/artifacts, collected and hash-checked like the
    # workspace (release/0.8; main did not port it): mode.txt is that file.
    assert files == {
        f"{root}/app/src/answer.txt",
        f"{root}/app/conftest.py",
        "logs/artifacts/mode.txt",
    }
    assert members[f"{root}/app/src/answer.txt"] == b"42\n"
    assert summary["workspace"] == str(workspace)
    assert summary["files"] == 3
    # Nothing from outside the workspace, and nothing absolute or escaping.
    for name in members:
        assert not name.startswith("/") and ".." not in name.split("/")
        assert "reward.txt" not in name and "evil.pth" not in name
        assert "agent-only" not in name and not name.endswith("tests/conftest.py")


# PY8 main sync: PY8's _fake_rollout takes tmp_path first.
@pytest.mark.asyncio
async def test_failed_unpack_is_an_assessment_error_not_zero(tmp_path: Path) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box", fail_exec="tar ")
    ran: list[bool] = []

    async def verify(*args, **kwargs):
        ran.append(True)
        return {"reward": 0.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=verify
    )

    assert rewards is None and not ran
    assert error is not None and error.startswith("separate verifier")
    assert "transfer" in error
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert box.stopped == [True]
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_default_verify_hardens_the_verifier_sandbox_without_an_agent_user(
    tmp_path: Path, monkeypatch
) -> None:
    from benchflow.rollout import _separate_verifier, _setup

    seen: dict[str, Any] = {}

    async def fake_verify_rollout(*args, **kwargs):
        seen.update(kwargs)
        return {"reward": 1.0}, None, None

    async def fake_publish(*args, **kwargs):
        seen["published"] = True

    monkeypatch.setattr(_setup, "_verify_rollout", fake_verify_rollout)
    monkeypatch.setattr(_setup, "_publish_trajectory_for_verifier", fake_publish)
    # PY8 main sync: integ batch 12's verify reads the agent's name.
    rollout = SimpleNamespace(
        _trajectory=[],
        _planes=SimpleNamespace(),
        _diagnostics=None,
        _config=SimpleNamespace(primary_agent="claude-agent-acp"),
    )
    verify = _separate_verifier._default_verify(rollout)
    paths = SimpleNamespace(agent_dir=tmp_path / "agent")

    rewards, error = await verify(object(), object(), paths, {}, workspace="/app")

    assert (rewards, error) == ({"reward": 1.0}, None)
    # No agent process ever ran in the verifier sandbox.
    assert seen["sandbox_user"] is None
    assert seen["workspace"] == "/app"
    assert seen["published"]


def test_config_json_records_where_the_verifier_ran(tmp_path: Path) -> None:
    from datetime import datetime

    from benchflow.rollout._results import _write_config
    from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

    for mode in ("separate", "shared"):
        rollout_dir = tmp_path / mode
        rollout_dir.mkdir()
        _write_config(
            rollout_dir,
            task_path=tmp_path / "task",
            agent="oracle",
            model=None,
            environment="daytona",
            skill_policy=resolve_task_skill_policy(
                task_path=tmp_path / "task",
                skill_mode=SKILL_MODE_NO_SKILL,
                runtime_skills_dir=None,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_user=None,
            context_root=None,
            timeout=60,
            started_at=datetime(2026, 1, 1),
            agent_env={},
            verifier_sandbox_mode=mode,
        )
        config = json.loads((rollout_dir / "config.json").read_text())
        assert config["verifier_sandbox_mode"] == mode


# --- outputs outside the workspace (--separate-verifier-include) -------------
# Guards the verifier-followups PR. Only the workspace crossed, so a task whose
# workspace is /app but whose agent writes graded files to /root scored 0 in
# separate mode, silently: one such task went from 19/19 to 0/19 on 2026-10-16.


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["/root"], ("/root",)),
        (["/root/", "//root", "/root"], ("/root",)),
        (["/root/outputs", "/srv/results"], ("/root/outputs", "/srv/results")),
    ],
)
def test_include_paths_are_normalized(given: list[str], expected: tuple) -> None:
    assert normalize_include_paths(given) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "root",
        "./root",
        "/root/../etc",
        "/",
        "/usr",
        "/home",
        "/var",
        "/proc/1",
        "/run/x",
        "/dev",
    ],
)
def test_include_paths_refuse_relative_and_system_paths(bad: str) -> None:
    with pytest.raises(ValueError, match="separate-verifier-include"):
        normalize_include_paths([bad])


def test_rollout_config_normalizes_include_paths(tmp_path: Path) -> None:
    from benchflow.rollout import RolloutConfig

    task = _task_dir(tmp_path, "")
    config = RolloutConfig(
        task_path=task, separate_verifier_include=("/root/", "/root")
    )
    assert config.separate_verifier_include == ("/root",)
    with pytest.raises(ValueError, match="absolute path"):
        RolloutConfig(task_path=task, separate_verifier_include=("relative",))


def _flag_task(tmp_path: Path) -> Path:
    task = tmp_path / "flag-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do it.\n")
    return task


def test_eval_run_threads_include_paths(tmp_path: Path) -> None:
    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation

    captured: dict[str, Any] = {}

    async def capture(self):
        captured["include"] = self._config.separate_verifier_include
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=capture):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(_flag_task(tmp_path)),
                "--agent",
                "oracle",
                "--separate-verifier",
                "--separate-verifier-include",
                "/root",
                "--separate-verifier-include",
                "/root/",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["include"] == ("/root",)


def test_eval_run_refuses_a_bad_include_path_before_any_sandbox(tmp_path: Path) -> None:
    from benchflow.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(_flag_task(tmp_path)),
            "--agent",
            "oracle",
            "--separate-verifier",
            "--separate-verifier-include",
            "root",
        ],
    )
    assert result.exit_code == 2
    assert "absolute path" in result.output


def test_include_paths_survive_sharding() -> None:
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig

    shard = EvalShard(index=0, task_names=("t",), concurrency=1)
    config = EvaluationConfig(
        separate_verifier=True, separate_verifier_include=("/root",)
    )
    assert _evaluation_config(
        _config_payload(config, shard=shard)
    ).separate_verifier_include == ("/root",)


@pytest.mark.asyncio
async def test_include_paths_cross_into_the_verifier_sandbox(tmp_path: Path) -> None:
    box = tmp_path / "box"
    workspace = box / "app"
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("work\n")
    outputs = box / "root"
    (outputs / "results").mkdir(parents=True)
    (outputs / "paper.pdf").write_bytes(b"%PDF-1.7\n")
    (outputs / "results" / "fit.csv").write_text("a,b\n1,2\n")
    (box / "tmp").mkdir()
    (box / "tmp" / "scratch.txt").write_text("stays behind\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(
        LocalTransport(),
        str(workspace),
        trial / "evidence",
        artifacts=[
            ArtifactConfig(source=str(outputs)),
            ArtifactConfig(source=str(box / "missing")),
        ],
    )
    payload = tmp_path / "payload.tar"

    await _collect_empty_logs(trial)  # integ batch 12 requires the record
    summary = build_transfer_payload(trial, payload)

    root = str(box).lstrip("/")
    files = {name for name, data in _members(payload).items() if data is not None}
    assert files == {
        f"{root}/app/notes.md",
        f"{root}/root/paper.pdf",
        f"{root}/root/results/fit.csv",
    }
    assert summary["artifacts"] == [
        {"source": str(outputs), "status": "captured"},
        {"source": str(box / "missing"), "status": "missing"},
    ]
    # Laid over the verifier image's copy of the directory, which keeps its own files.
    verifier_root = tmp_path / "verifier-root"
    image_copy = verifier_root / root / "root"
    image_copy.mkdir(parents=True)
    (image_copy / ".bashrc").write_text("image file\n")
    script = _install_script(
        str(payload), summary["workspace"], summary["sha256"], root=str(verifier_root)
    )
    subprocess.run(["sh", "-c", script], check=True)
    assert (image_copy / "paper.pdf").read_bytes() == b"%PDF-1.7\n"
    assert (image_copy / "results" / "fit.csv").read_text() == "a,b\n1,2\n"
    assert (image_copy / ".bashrc").read_text() == "image file\n"
    assert not (verifier_root / root / "tmp").exists()


async def _collect_empty_logs(trial: Path) -> None:
    """The artifact-collection record of a run with nothing under /logs/artifacts.

    Since integ batch 12 the payload refuses a trial without it; the capture
    tests here are about the workspace and include paths.
    """
    logs = trial.parent / "empty-logs"
    logs.mkdir(exist_ok=True)
    (trial / "artifacts").mkdir(exist_ok=True)
    await collect_artifacts(
        LocalTransport(),
        artifacts=[],
        workspace=str(trial.parent / "unused-workspace"),
        artifacts_dir=trial / "artifacts",
        manifest_path=trial / MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
    )


# PY8 main sync: renamed from _capture_rollout, which PY8's
# capture-tool tests already use; PY8's capture also reads _branch_child_active.
def _include_capture_rollout(
    tmp_path: Path, config: TaskConfig, *, review: bool = False
):
    async def disconnect() -> None:
        return None

    async def quiesce(env, user) -> None:
        return None

    return SimpleNamespace(
        _review_plan=object() if review else None,
        # PY8's capture skips a branch child's workspace.
        _branch_child_active=False,
        _config=SimpleNamespace(
            purpose="task", sandbox_user="agent", separate_verifier_include=("/root",)
        ),
        _task=SimpleNamespace(config=config),
        _env=object(),
        _agent_cwd="/app",
        _agent_env={},
        _agent_phase_started=100.0,
        _outside_workspace_writes=None,
        _export_error=None,
        _planes=SimpleNamespace(quiesce_agent=quiesce),
        disconnect=disconnect,
        _require_rollout_dir=lambda: tmp_path,
    )


# PY8 main sync: uses _include_capture_rollout.
@pytest.mark.asyncio
async def test_include_paths_and_the_scan_apply_only_to_a_separate_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    from benchflow.rollout import _review

    captured: list[list[str]] = []
    scanned: list[tuple[float, list[str]]] = []
    report = {
        "count": 1,
        "truncated": False,
        "by_directory": {"/root": 1},
        "sample": ["/root/paper.pdf"],
    }

    async def fake_capture(
        env, workspace, destination, *, artifacts=(), excluded_paths=()
    ):
        captured.append([artifact.source for artifact in artifacts])

    async def fake_scan(env, *, since, keep, **_):
        scanned.append((since, keep))
        return report

    monkeypatch.setattr(_review, "capture_task_evidence", fake_capture)
    monkeypatch.setattr(_review, "scan_outside_workspace_writes", fake_scan)

    separate = _include_capture_rollout(
        tmp_path, TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    await _review.capture_terminal_workspace(separate)
    assert captured == [["/root"]]
    assert scanned == [(100.0, ["/app", "/root", "/home/agent"])]
    assert separate._outside_workspace_writes == report

    shared = _include_capture_rollout(tmp_path, TaskConfig(), review=True)
    await _review.capture_terminal_workspace(shared)
    assert captured[-1] == []
    assert len(scanned) == 1 and shared._outside_workspace_writes is None


@pytest.mark.asyncio
async def test_the_agent_phase_start_is_read_from_the_sandbox_clock() -> None:
    from benchflow.rollout._review import mark_agent_phase_start

    class Clock:
        async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
            assert "time.time()" in cmd and user == "root"
            return ExecResult(0, "1792170000.25\n", "")

    separate = SimpleNamespace(
        _task=SimpleNamespace(
            config=TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
        ),
        _env=Clock(),
        _agent_phase_started=None,
    )
    await mark_agent_phase_start(separate)
    assert separate._agent_phase_started == 1792170000.25

    shared = SimpleNamespace(
        _task=SimpleNamespace(config=TaskConfig()),
        _env=Clock(),
        _agent_phase_started=None,
    )
    await mark_agent_phase_start(shared)
    assert shared._agent_phase_started is None


@pytest.mark.asyncio
async def test_files_written_outside_everything_that_crosses_are_listed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "box"
    (root / "root").mkdir(parents=True)
    (root / "root" / "proposal.pdf").write_text("an input from before the agent\n")
    time.sleep(0.05)
    since = time.time()
    time.sleep(0.05)
    for path in (
        "root/paper.pdf",  # graded output outside the /app workspace: listed
        "srv/data/fit.csv",  # listed
        "app/out.txt",  # the workspace crosses
        "home/agent/.codex/session.json",  # the agent's home is harness state
        "tmp/scratch",
        "usr/lib/x.py",
        "var/log/y",
        "logs/verifier/reward.txt",
        "oracle/__pycache__/solve.cpython-312.pyc",  # harness mount
        "opt/benchflow-egress/policy.json",  # BenchFlow's egress proxy
        "opt/benchflow_agentcore_shim.py",
        "opt/benchflow/node/bin/node",
        "root/.cache/uv/wheel",  # package cache
        "root/.daytona/sessions/s",  # provider runtime state
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("new\n")

    report = await scan_outside_workspace_writes(
        LocalTransport(), since=since, keep=["/app", "/home/agent"], root=str(root)
    )

    assert report is not None
    assert report["count"] == 2 and not report["truncated"]
    assert report["by_directory"] == {"/root": 1, "/srv": 1}
    assert set(report["sample"]) == {"/root/paper.pdf", "/srv/data/fit.csv"}


# PY8 main sync: PY8's _fake_rollout takes tmp_path first.
@pytest.mark.asyncio
async def test_the_record_lists_files_left_outside_the_workspace(
    tmp_path: Path,
) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._outside_workspace_writes = {
        "count": 1,
        "truncated": False,
        "by_directory": {"/root": 1},
        "sample": ["/root/paper.pdf"],
    }
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, vtask, vpaths, timing, *, workspace):
        return {"reward": 1.0}, None

    await run_separate_verifier(
        rollout, create_environment=lambda *_: box, verify=verify
    )

    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["outside_workspace_writes"]["sample"] == ["/root/paper.pdf"]
    assert record["transfer"]["artifacts"] == []


# --- overlapping capture roots ---------------------------------


@pytest.mark.asyncio
async def test_an_include_inside_a_declared_artifact_is_captured_once(
    tmp_path: Path, monkeypatch
) -> None:
    """``--separate-verifier-include`` naming a declared artifact.

    The include equal to the task's artifact, or inside it, is dropped (the
    artifact's own capture crosses); one holding an artifact stays, and the
    payload takes the overlap (next test).
    """
    from benchflow.rollout import _review

    captured: list[list[str]] = []

    async def fake_capture(
        env, workspace, destination, *, artifacts=(), excluded_paths=()
    ):
        captured.append([a if isinstance(a, str) else a.source for a in artifacts])

    async def fake_scan(env, *, since, keep, **_):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", fake_capture)
    monkeypatch.setattr(_review, "scan_outside_workspace_writes", fake_scan)

    for artifacts, include, expected in (
        (["/root"], ("/root",), ["/root"]),
        (["/root"], ("/root/out",), ["/root"]),
        (["/root/out/result.txt"], ("/root",), ["/root/out/result.txt", "/root"]),
    ):
        rollout = _include_capture_rollout(
            tmp_path,
            TaskConfig.model_validate(
                {"verifier": {"sandbox_mode": "separate"}, "artifacts": artifacts}
            ),
        )
        rollout._config.separate_verifier_include = include
        await _review.capture_terminal_workspace(rollout)
        assert captured[-1] == expected


@pytest.mark.asyncio
async def test_overlapping_capture_roots_cross_each_file_once(tmp_path: Path) -> None:
    """An include holding a declared artifact still verifies.

    ``/srv/result.txt`` is declared and ``/srv`` is included: the same file is
    captured twice in one freeze and crosses once, not "two transfers write".
    """
    box = tmp_path / "box"
    workspace = box / "app"
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("work\n")
    srv = box / "srv"
    srv.mkdir()
    (srv / "result.txt").write_text("42\n")
    (srv / "log.txt").write_text("ran\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(
        LocalTransport(),
        str(workspace),
        trial / "evidence",
        artifacts=[
            ArtifactConfig(source=str(srv / "result.txt")),
            ArtifactConfig(source=str(srv)),
        ],
    )
    payload = tmp_path / "payload.tar"

    await _collect_empty_logs(trial)  # integ batch 12 requires the record
    summary = build_transfer_payload(trial, payload)

    root = str(box).lstrip("/")
    with tarfile.open(payload) as archive:
        members = [m.name for m in archive.getmembers() if m.isfile()]
    assert sorted(members) == [
        f"{root}/app/notes.md",
        f"{root}/srv/log.txt",
        f"{root}/srv/result.txt",
    ]
    assert summary["files"] == 3


def test_two_different_files_for_one_path_still_refuse_the_transfer(
    tmp_path: Path,
) -> None:
    from pathlib import PurePosixPath

    from benchflow.rollout._separate_verifier import _Payload

    first, second, same = (tmp_path / n for n in ("a", "b", "c"))
    first.write_text("42\n")
    second.write_text("41\n")
    same.write_text("42\n")
    with tarfile.open(tmp_path / "p.tar", "x") as archive:
        payload = _Payload(archive)
        payload.file(PurePosixPath("/srv/result.txt"), first, 0o644)
        payload.file(PurePosixPath("/srv/result.txt"), same, 0o644)
        with pytest.raises(SeparateVerifierError, match="two transfers write"):
            payload.file(PurePosixPath("/srv/result.txt"), second, 0o644)
        with pytest.raises(SeparateVerifierError, match="two transfers write"):
            payload.file(PurePosixPath("/srv/result.txt"), same, 0o755)
    assert payload.files == 1


@pytest.mark.asyncio
async def test_an_over_limit_capture_with_include_paths_is_never_blamed(
    tmp_path: Path, monkeypatch
) -> None:
    """The clean control does not measure include paths.

    An image can ship ``/srv/output`` with more files than the capture limit;
    an agent that changed nothing then hits the limit through the include.
    Without its baseline that is never the solution's doing: unscored, not
    ``refused`` with reward 0.
    """
    from benchflow.review.evidence import EvidenceLimitError
    from benchflow.rollout import _review

    async def capture(*args, **kwargs):
        raise EvidenceLimitError("Task evidence exceeds configured capture limits")

    monkeypatch.setattr(_review, "capture_task_evidence", capture)
    rollout = _include_capture_rollout(
        tmp_path, TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    rollout._config.separate_verifier_include = ("/srv/output",)
    await _review.capture_terminal_workspace(rollout)
    assert "capture limits" in rollout._export_error
    assert rollout._capture_over_limit is False


# --- an include is never dropped silently -----------------------


@pytest.mark.asyncio
async def test_an_include_inside_an_artifact_that_excludes_files_keeps_its_capture(
    tmp_path: Path, monkeypatch
) -> None:
    """The artifact ``/srv/out`` excludes ``*.bin``.

    ``--separate-verifier-include /srv/out/model.bin`` names the very file the
    artifact leaves out, so it is captured on its own and crosses.
    """
    from benchflow.rollout import _review

    captured: list[list[str]] = []

    async def fake_capture(
        env, workspace, destination, *, artifacts=(), excluded_paths=()
    ):
        captured.append([a if isinstance(a, str) else a.source for a in artifacts])

    async def fake_scan(env, *, since, keep, **_):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", fake_capture)
    monkeypatch.setattr(_review, "scan_outside_workspace_writes", fake_scan)
    rollout = _include_capture_rollout(
        tmp_path,
        TaskConfig.model_validate(
            {
                "verifier": {"sandbox_mode": "separate"},
                "artifacts": [{"source": "/srv/out", "exclude": ["*.bin"]}],
            }
        ),
    )
    rollout._config.separate_verifier_include = ("/srv/out/model.bin",)
    await _review.capture_terminal_workspace(rollout)
    assert captured == [["/srv/out", "/srv/out/model.bin"]]


@pytest.mark.asyncio
async def test_the_excluded_file_an_include_names_crosses(tmp_path: Path) -> None:
    box = tmp_path / "box"
    workspace = box / "app"
    workspace.mkdir(parents=True)
    out = box / "srv" / "out"
    out.mkdir(parents=True)
    (out / "model.bin").write_bytes(b"\x00weights")
    (out / "notes.txt").write_text("fit\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(
        LocalTransport(),
        str(workspace),
        trial / "evidence",
        artifacts=[
            ArtifactConfig(source=str(out), exclude=["*.bin"]),
            ArtifactConfig(source=str(out / "model.bin")),
        ],
    )
    payload = tmp_path / "payload.tar"

    await _collect_empty_logs(trial)  # integ batch 12 requires the record
    summary = build_transfer_payload(trial, payload)

    root = str(box).lstrip("/")
    with tarfile.open(payload) as archive:
        members = sorted(m.name for m in archive.getmembers() if m.isfile())
    assert members == [f"{root}/srv/out/model.bin", f"{root}/srv/out/notes.txt"]
    # The artifact's own exclusion is recorded, not silent.
    (declared, include) = summary["artifacts"]
    assert [e["reason"] for e in declared["left_out"]] == ["task_exclude"]
    assert declared["left_out"][0]["path"].endswith("model.bin")
    assert include == {"source": str(out / "model.bin"), "status": "captured"}


@pytest.mark.asyncio
async def test_what_capture_keeps_out_of_an_include_is_named(
    tmp_path: Path, caplog
) -> None:
    """A package cache under an include of ``/root``.

    Capture never carries package caches (or credentials, or links out of
    the include); the transfer record lists each such path under the include,
    and a warning names the include, instead of dropping it silently.
    """
    box = tmp_path / "box"
    workspace = box / "app"
    workspace.mkdir(parents=True)
    home = box / "root"
    (home / ".cache" / "pip").mkdir(parents=True)
    (home / ".cache" / "pip" / "wheel.whl").write_bytes(b"cache")
    (home / "paper.pdf").write_bytes(b"%PDF-1.7\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(
        LocalTransport(),
        str(workspace),
        trial / "evidence",
        artifacts=[ArtifactConfig(source=str(home))],
    )
    await _collect_empty_logs(trial)  # integ batch 12 requires the record
    summary = build_transfer_payload(trial, tmp_path / "payload.tar")

    (entry,) = summary["artifacts"]
    assert entry["status"] == "captured"
    assert [e["reason"] for e in entry["left_out"]] == ["package_cache"]
    assert entry["left_out"][0]["path"].endswith(".cache/pip")
    from benchflow.rollout._separate_verifier import _warn_left_out

    with caplog.at_level("WARNING"):
        _warn_left_out(summary, (str(home),))
    assert f"--separate-verifier-include {home}" in caplog.text
    assert "package_cache" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("phase", ["start", "verify"])
async def test_live_separate_verifier_budget_cancels_and_keeps_its_charge(
    tmp_path, monkeypatch, backend, phase
):
    """Guards FF-sw-tests-ci against 363d815b."""
    from benchflow import budget as budget_mod
    from benchflow.evaluation import Evaluation
    from benchflow.rollout import _separate_verifier as separate

    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0)
    monkeypatch.setattr(budget_mod, "time", fake_time)
    monkeypatch.setattr(separate, "time", fake_time)
    monkeypatch.setattr(separate.timing_spans, "_clock", fake_time.monotonic)
    guard = budget_mod.BudgetGuard(budget_mod.Budget(max_sandbox_seconds=60))
    _, workspace, trial = await _agent_trial(tmp_path)
    task_dir = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task_dir, str(workspace))
    rollout._config.environment = backend
    entered = asyncio.Event()
    stage = separate._stage_context

    def stage_at_ten(*args, **kwargs):
        result = stage(*args, **kwargs)
        clock[0] = 10.0
        return result

    monkeypatch.setattr(separate, "_stage_context", stage_at_ten)

    class Box(FakeVerifierSandbox):
        async def start(self, **kwargs):
            if phase == "start":
                entered.set()
                await asyncio.Event().wait()
            await super().start(**kwargs)

        async def stop(self, **kwargs):
            clock[0] = 38.0
            await super().stop(**kwargs)

    box = Box(tmp_path / "verifier-box")

    async def verify(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def run_task(_):
        return await run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=verify
        )

    evaluation = SimpleNamespace(
        _run_task=run_task, _fire_progress=lambda *a: None, _on_result=None
    )
    guard.start("task")
    running = asyncio.create_task(
        Evaluation._run_budgeted(evaluation, Path("task"), guard)
    )
    guard.attach("task", running)
    await asyncio.wait_for(entered.wait(), timeout=10)
    clock[0] = 35.0
    live_seconds = guard.spent()["sandbox_seconds"]
    guard.check()
    cancelled_for_budget = "task" in guard._cancel_requested
    if not cancelled_for_budget:
        running.cancel()  # end the base repro; never wait for its missed cap
    with contextlib.suppress(asyncio.CancelledError):
        await running
    assert live_seconds == 60.0
    assert cancelled_for_budget
    assert guard.cancelled == ["task"]
    assert box.stopped == [True]
    # The three seconds of cleanup also keep both sandboxes allocated.
    assert guard.spent()["sandbox_seconds"] == 66.0
    assert rollout._timing["verifier_sandbox_total"] == 28.0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("phase", ["start", "verify"])
async def test_cancelled_separate_verifier_keeps_spend_on_resume(
    tmp_path, monkeypatch, backend, phase
):
    """Guards FF-sw-tests-ci-confirm-2 against cancellation/resume at 1a32bcbe."""
    from benchflow import budget as budget_mod
    from benchflow.evaluation import _recorded_attempts
    from benchflow.rollout import Rollout, RolloutConfig, _deadline
    from benchflow.rollout import _separate_verifier as separate

    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0)
    monkeypatch.setattr(budget_mod, "time", fake_time)
    monkeypatch.setattr(separate, "time", fake_time)
    monkeypatch.setattr(separate.timing_spans, "_clock", fake_time.monotonic)
    monkeypatch.setattr(_deadline, "hard_deadline_sec", lambda _: None)
    _, workspace, trial = await _agent_trial(tmp_path)
    (trial / "config.json").write_text('{"agent": "oracle"}')
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    fake = _fake_rollout(tmp_path, trial, task, str(workspace))
    fake._config.environment = backend
    rollout = Rollout(RolloutConfig.from_legacy(task_path=task, agent="oracle"))
    rollout._rollout_dir = trial
    entered = asyncio.Event()

    class Box(FakeVerifierSandbox):
        async def start(self, **kwargs):
            if phase == "start":
                entered.set()
                await asyncio.Event().wait()
            await super().start(**kwargs)

        async def stop(self, **kwargs):
            clock[0] = 38.0
            await super().stop(**kwargs)

    box = Box(tmp_path / "verifier-box")

    async def verify(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def lifecycle():
        clock[0] = 10.0
        return await run_separate_verifier(
            fake, create_environment=lambda *a: box, verify=verify
        )

    monkeypatch.setattr(rollout, "_run_lifecycle", lifecycle)
    guard = budget_mod.BudgetGuard(budget_mod.Budget(max_sandbox_seconds=60))
    guard.start("task")
    with guard.track_trial("task"):
        running = asyncio.create_task(rollout.run())
    guard.attach("task", running)
    await asyncio.wait_for(entered.wait(), timeout=10)
    clock[0] = 35.0
    guard.check()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert guard.was_cancelled("task")
    assert guard.spent()["sandbox_seconds"] == 66
    assert not (trial / "result.json").exists()
    assert box.stopped == [True]
    # Each fresh resume rereads the same attempt, never its previous summary.
    for _ in range(3):
        resumed = budget_mod.BudgetGuard(budget_mod.Budget(max_sandbox_seconds=60))
        records, interrupted = _recorded_attempts(trial.parent)
        resumed.seed(records, interrupted)
        assert resumed.spent()["sandbox_seconds"] == 66
        assert resumed.spent()["rollouts"] == 1
        assert not resumed.start("task")

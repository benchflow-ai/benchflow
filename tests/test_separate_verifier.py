"""Separate verifier sandboxes: Harbor ``[verifier] environment_mode =
"separate"``.

The verifier runs in its own sandbox, built from the task's ``tests/`` (or a
declared verifier image), and receives only the frozen workspace, the
declared artifacts and ``/logs/artifacts`` from the agent's sandbox. Nothing
else the agent left behind reaches it. A failed transfer is an assessment
error (no reward), not a 0, unless the solution's own files caused it: then,
judged against the same paths measured before the agent ran, it scores 0.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    SolutionTransferRefused,
    build_transfer_payload,
    measure_outputs,
    plan_verifier_image,
    run_separate_verifier,
    separate_verifier_requested,
)
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import TaskConfig, validate_task_runtime_support

# --- config and launch gate --------------------------------------------------


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
    # Harbor: [verifier.environment] without a mode implies separate.
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


@pytest.mark.parametrize("sandbox", ["modal", "apple-container", "agentcore"])
def test_separate_mode_stays_refused_where_it_is_not_implemented(
    tmp_path: Path, sandbox: str
) -> None:
    config = TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    issues = _separate_issues(config, sandbox)
    assert [i.path for i in issues] == ["verifier.sandbox_mode"]
    assert "docker and daytona" in issues[0].reason


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


def test_image_precedence_follows_harbor(tmp_path: Path) -> None:
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
    #    image wins over the tests/ build context (Harbor semantics).
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
        artifacts_dir=trial / "artifacts",
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
    rollout = SimpleNamespace(_trajectory=[], _planes=SimpleNamespace())
    verify = _separate_verifier._default_verify(rollout)

    await verify(
        object(),
        object(),
        SimpleNamespace(agent_dir=tmp_path),
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
        artifacts_dir=trial / "artifacts",
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
        artifacts_dir=trial / "artifacts",
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
        artifacts_dir=artifacts,
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

    async def upload_file(self, source: Path | str, target: str) -> None:
        dest = self.root / target.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        self.uploads[target] = dest

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30, **_):
        self.commands.append(cmd)
        if self.fail_exec and self.fail_exec in cmd:
            return ExecResult(1, "", "tar: boom")
        if "tar " in cmd and " -x" in cmd:
            archive = next(p for t, p in self.uploads.items() if t.endswith(".tar"))
            with tarfile.open(archive) as tar:
                tar.extractall(self.root, filter="tar")
        return ExecResult(0, "", "")


def _fake_rollout(tmp_path: Path, trial: Path, task: Path, workspace: str):
    from benchflow.task import Task
    from benchflow.task.paths import RolloutPaths

    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    return SimpleNamespace(
        _task=Task(task),
        _config=SimpleNamespace(
            environment="daytona", task_path=task, sandbox_setup_timeout=60
        ),
        _rollout_dir=trial,
        _rollout_paths=paths,
        _rollout_name="trial-1",
        _agent_cwd=workspace,
        _trajectory=[],
        _timing={},
        _export_error=None,
        _planes=SimpleNamespace(),
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

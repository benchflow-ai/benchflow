"""Separate verifier sandboxes: Harbor ``[verifier] environment_mode = "separate"``.

The verifier runs in its own sandbox, built from the task's ``tests/`` (or a
declared verifier image, or the task's own ``environment/``), and receives
only the frozen workspace from the agent's sandbox. Nothing else the agent
left behind reaches it. A failed transfer is an assessment error (no reward),
never a 0 and never a pass.

Guards PR #1185, which executes the mode main parsed but refused ("separate
verifier sandboxes are parsed but not executed") and adds
``--separate-verifier``; the implementation follows release/0.8's
``rollout/_separate_verifier.py``.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    _install_script,
    build_transfer_payload,
    run_separate_verifier,
    scan_outside_workspace_writes,
)
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import TaskConfig, validate_task_runtime_support
from benchflow.task.config import ArtifactConfig
from benchflow.task.verifier_sandbox import (
    apply_separate_verifier_flag,
    normalize_include_paths,
    plan_verifier_image,
    separate_verifier_requested,
)

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
    assert "daytona and docker" in issues[0].reason


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


# --- --separate-verifier -----------------------------------------------------


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
            sandbox="modal",
            task_dir=task,
        )
    assert "daytona and docker" in str(raised.value)
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


# --- what crosses from the agent sandbox to the verifier sandbox ------------


class LocalTransport:
    """The agent sandbox is this machine."""

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


async def _agent_trial(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An agent sandbox rooted at ``box/`` whose agent solved *and* tampered."""
    box = tmp_path / "box"
    workspace = box / "app"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("42\n")
    # Planted in the workspace: travels (it is the agent's work), and the
    # verifier sandbox's hardening removes it before tests run.
    (workspace / "conftest.py").write_text("raise SystemExit(0)\n")
    # Planted outside the workspace: must never reach the verifier sandbox.
    (box / "logs" / "verifier").mkdir(parents=True)
    (box / "logs" / "verifier" / "reward.txt").write_text("1\n")
    (box / "tests").mkdir()
    (box / "tests" / "conftest.py").write_text("raise SystemExit(0)\n")
    (box / "usr-lib-python" / "site-packages").mkdir(parents=True)
    (box / "usr-lib-python" / "site-packages" / "evil.pth").write_text("import os\n")
    (box / "tmp").mkdir()
    (box / "tmp" / "agent-only.txt").write_text("ambient file\n")

    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
    return box, workspace, trial


def _members(archive: Path) -> dict[str, bytes | None]:
    with tarfile.open(archive) as tar:
        return {
            m.name: (tar.extractfile(m).read() if m.isfile() else None)  # type: ignore[union-attr]
            for m in tar.getmembers()
        }


@pytest.mark.asyncio
async def test_payload_holds_only_the_frozen_workspace(tmp_path: Path) -> None:
    box, workspace, trial = await _agent_trial(tmp_path)

    summary = build_transfer_payload(trial, tmp_path / "payload.tar")

    members = _members(tmp_path / "payload.tar")
    root = str(box).lstrip("/")
    files = {name for name, data in members.items() if data is not None}
    assert files == {f"{root}/app/src/answer.txt", f"{root}/app/conftest.py"}
    assert members[f"{root}/app/src/answer.txt"] == b"42\n"
    assert summary["workspace"] == str(workspace)
    assert summary["files"] == 2
    # Nothing from outside the workspace, and nothing absolute or escaping.
    for name in members:
        assert not name.startswith("/") and ".." not in name.split("/")
        assert "reward.txt" not in name and "evil.pth" not in name
        assert "agent-only" not in name and not name.endswith("tests/conftest.py")


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
async def test_the_workspace_replaces_the_verifier_images_copy(tmp_path: Path) -> None:
    """A dedicated workspace (/app) is replaced, a shared one (/root) laid over.

    The verifier image holds its own copy of a workspace such as /app; the
    agent's deletions and renames must not come back from it.
    """
    workspace = tmp_path / "box" / "app"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("42\n")
    (workspace / "new-name.txt").write_text("renamed by the agent\n")
    (workspace / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (workspace / "run.sh").chmod(0o755)
    trial = tmp_path / "trial"
    trial.mkdir()
    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
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
    subprocess.run(["sh", "-c", script], check=True)

    assert sorted(p.name for p in image.iterdir()) == ["new-name.txt", "run.sh", "src"]
    assert image.stat().st_ino == inode
    assert (image / "src" / "answer.txt").read_text() == "42\n"
    # The manifest records no modes; the executable bit is kept.
    assert (image / "run.sh").stat().st_mode & 0o111
    assert not (image / "new-name.txt").stat().st_mode & 0o111
    assert not payload.exists()
    # A shared directory such as /root is laid over, never emptied.
    assert "rm -rf" not in _install_script("/tmp/x.tar", "/root", "0" * 64)
    assert "rm -rf" in _install_script("/tmp/x.tar", "/app", "0" * 64)


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


def _fake_rollout(trial: Path, task: Path, workspace: str):
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
    box_root, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(trial, task, str(workspace))
    verifier_root = tmp_path / "verifier-box"
    box = FakeVerifierSandbox(verifier_root)
    created: list[Any] = []

    def create(vtask, context_path, vpaths):
        created.append((vtask, context_path, vpaths))
        return box

    async def verify(env, vtask, vpaths, timing, *, workspace):
        # The verifier sees the agent's work at the same absolute path, and
        # nothing planted elsewhere in the agent sandbox.
        root = verifier_root / workspace.lstrip("/")
        assert (root / "src" / "answer.txt").read_text() == "42\n"
        assert not (
            verifier_root / str(box_root / "logs" / "verifier").lstrip("/")
        ).exists()
        assert not (
            verifier_root / str(box_root / "usr-lib-python").lstrip("/")
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
    # The verifier sandbox is built from tests/, never from the agent's sandbox.
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
    for key in (
        "verifier_sandbox_setup",
        "verifier_transfer",
        "verifier",
        "verifier_sandbox_teardown",
        "verifier_sandbox_total",
    ):
        assert key in rollout._timing, key
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["mode"] == "separate"
    assert record["image_source"] == "tests/Dockerfile"
    assert record["sandbox_id"] == "verifier-box-1"
    assert record["status"] == "complete"
    assert record["transfer"]["files"] == 2
    assert record["sandbox_seconds"] >= 0


@pytest.mark.asyncio
async def test_failed_unpack_is_an_assessment_error_not_zero(tmp_path: Path) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(trial, task, str(workspace))
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
    rollout = _fake_rollout(trial, task, str(workspace))
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
    rollout = _fake_rollout(trial, task, str(workspace))
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
    rollout = SimpleNamespace(
        _trajectory=[], _planes=SimpleNamespace(), _diagnostics=None
    )
    verify = _separate_verifier._default_verify(rollout)
    paths = SimpleNamespace(agent_dir=tmp_path / "agent")

    rewards, error = await verify(object(), object(), paths, {}, workspace="/app")

    assert (rewards, error) == ({"reward": 1.0}, None)
    # No agent process ever ran in the verifier sandbox.
    assert seen["sandbox_user"] is None
    assert seen["workspace"] == "/app"
    assert seen["published"]


# --- rollout wiring ----------------------------------------------------------


def test_separate_mode_forces_the_workspace_freeze() -> None:
    from benchflow.rollout._review import _freeze_requested

    shared = SimpleNamespace(
        _review_plan=None,
        _config=SimpleNamespace(purpose="task"),
        _task=SimpleNamespace(config=TaskConfig()),
    )
    separate = SimpleNamespace(
        _review_plan=None,
        _config=SimpleNamespace(purpose="task"),
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
    monkeypatch.setattr(rollout_mod, "_publish_trajectory_for_verifier", noop)

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
# separate mode, silently: FrontierPhysics kinked-ladder-polymer-scattering
# went from 19/19 to 0/19 on 2026-10-16.


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
    assert result.exit_code == 1
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


def _capture_rollout(tmp_path: Path, config: TaskConfig, *, review: bool = False):
    async def disconnect() -> None:
        return None

    async def quiesce(env, user) -> None:
        return None

    return SimpleNamespace(
        _review_plan=object() if review else None,
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

    separate = _capture_rollout(
        tmp_path, TaskConfig.model_validate({"verifier": {"sandbox_mode": "separate"}})
    )
    await _review.capture_terminal_workspace(separate)
    assert captured == [["/root"]]
    assert scanned == [(100.0, ["/app", "/root", "/home/agent"])]
    assert separate._outside_workspace_writes == report

    shared = _capture_rollout(tmp_path, TaskConfig(), review=True)
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


@pytest.mark.asyncio
async def test_the_record_lists_files_left_outside_the_workspace(
    tmp_path: Path,
) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(trial, task, str(workspace))
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

"""Separate verifier sandboxes: Harbor ``[verifier] environment_mode =
"separate"``.

The verifier runs in its own sandbox, built from the task's ``tests/`` (or a
declared verifier image), and receives only the frozen workspace, the
declared artifacts and ``/logs/artifacts`` from the agent's sandbox. Nothing
else the agent left behind reaches it. A failed transfer is an assessment
error (no reward), never a 0.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    build_transfer_payload,
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

    async def verify(env, vtask, vpaths, timing, *, workspace):
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

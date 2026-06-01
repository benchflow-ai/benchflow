"""Tests for rollout startup uploads and sandbox info persistence."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.rollout import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    Rollout,
    RolloutConfig,
    Scene,
    _build_rollout_result,
    _persist_sandbox_info,
    _publish_trajectory_for_verifier,
    _start_env_and_upload,
)


class FakeUploadEnv:
    def __init__(self) -> None:
        self.started = False
        self.exec_calls: list[tuple[str, str | None, int | None]] = []
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.uploaded_file_contents: list[tuple[str, str]] = []

    async def start(self, force_build: bool) -> None:
        self.started = force_build is False

    async def exec(
        self, command: str, user: str | None = None, timeout_sec: int | None = None
    ) -> None:
        self.exec_calls.append((command, user, timeout_sec))

    async def upload_file(self, source: Path | str, target: str) -> None:
        source_path = Path(source)
        self.uploaded_files.append((source_path, target))
        self.uploaded_file_contents.append((source_path.read_text(), target))

    async def upload_dir(self, source: Path, target: str) -> None:
        self.uploaded_dirs.append((source, target))


@pytest.mark.asyncio
async def test_start_env_does_not_upload_task_environment_skills(
    tmp_path: Path,
) -> None:
    """Guards PR #586 against the no-skills leak into /app/skills."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "jax-skills").mkdir(parents=True)
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("solve\n")
    (task / "solution" / "solve.sh").write_text("echo ok\n")
    env = FakeUploadEnv()
    timing: dict[str, float] = {}

    await _start_env_and_upload(env, task, timing)

    task_skills = task / "environment" / "skills"
    assert env.started is True
    assert (task / "instruction.md", "/instruction.md") in env.uploaded_files
    assert (task_skills, "/app/skills") not in env.uploaded_dirs
    assert (task / "solution", "/solution") in env.uploaded_dirs
    assert "environment_setup" in timing


@pytest.mark.asyncio
async def test_publish_trajectory_for_verifier_uploads_acp_jsonl() -> None:
    """Guards the skill-eval LLM judge dogfood failure from 2026-05-19."""
    env = FakeUploadEnv()
    trajectory = [{"type": "agent_message", "text": "ok"}]

    await _publish_trajectory_for_verifier(env, trajectory)

    assert ("mkdir -p /logs/agent", "root", 10) in env.exec_calls
    assert env.uploaded_file_contents == [
        (
            '{"type": "agent_message", "text": "ok"}\n',
            "/logs/agent/acp_trajectory.jsonl",
        )
    ]


@pytest.mark.asyncio
async def test_rollout_setup_strips_task_skills_from_no_skills_build_context(
    tmp_path: Path,
) -> None:
    """Guards PR #586 against Dockerfile COPY . leaking task skills."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "alpha").mkdir(parents=True)
    (task / "environment" / "skills" / "alpha" / "SKILL.md").write_text("# Alpha\n")
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\nCOPY . /app\n")
    (task / "instruction.md").write_text("solve\n")
    (task / "task.toml").write_text('version = "1.0"\n')

    env = MagicMock()
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.create_environment.return_value = env

    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="claude-agent-acp",
            jobs_dir=tmp_path / "jobs",
            planes=planes,
        )
    )
    await rollout.setup()

    assert (task / "environment" / "skills" / "alpha" / "SKILL.md").exists()
    assert rollout._effective_task_path != task
    assert not (rollout._effective_task_path / "environment" / "skills").exists()
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["skill_mode"] == "no-skill"
    assert config["skill_source"] == "none"
    assert config["include_task_skills"] is False
    assert config["effective_skills_dir"] is None


@pytest.mark.asyncio
async def test_rollout_setup_includes_task_skills_without_declared_mount(
    tmp_path: Path,
) -> None:
    """Guards PR #586 so with-skill mode enables task bundles."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "alpha").mkdir(parents=True)
    (task / "environment" / "skills" / "alpha" / "SKILL.md").write_text("# Alpha\n")
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "instruction.md").write_text("solve\n")
    (task / "task.toml").write_text('version = "1.0"\n')

    env = MagicMock()
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.create_environment.return_value = env

    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="claude-agent-acp",
            jobs_dir=tmp_path / "jobs",
            skill_mode=SKILL_MODE_WITH_SKILL,
            planes=planes,
        )
    )
    await rollout.setup()

    assert rollout._effective_task_path != task
    assert (
        rollout._effective_task_path / "environment" / "skills" / "alpha" / "SKILL.md"
    ).exists()
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["skill_mode"] == "with-skill"
    assert config["skill_source"] == "task_bundled"
    assert config["include_task_skills"] is True
    assert config["effective_skills_dir"] == str(
        rollout._effective_task_path / "environment" / "skills"
    )
    planes.inject_skills_into_dockerfile.assert_called_once_with(
        rollout._effective_task_path,
        rollout._effective_task_path / "environment" / "skills",
        sandbox_dir="/skills",
    )


@pytest.mark.asyncio
async def test_rollout_setup_prebuilt_image_keeps_task_skills_runtime_uploaded(
    tmp_path: Path,
) -> None:
    """Guards PR #586 follow-up: prebuilt images cannot rely on Dockerfile skills COPY."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "alpha").mkdir(parents=True)
    (task / "environment" / "skills" / "alpha" / "SKILL.md").write_text("# Alpha\n")
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "instruction.md").write_text("solve\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\ndocker_image = "alpine:3.20"\n'
    )

    env = MagicMock()
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.create_environment.return_value = env

    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="claude-agent-acp",
            jobs_dir=tmp_path / "jobs",
            skill_mode=SKILL_MODE_WITH_SKILL,
            planes=planes,
        )
    )
    await rollout.setup()

    planes.inject_skills_into_dockerfile.assert_not_called()
    assert rollout._effective_skills_dir == (
        rollout._effective_task_path / "environment" / "skills"
    )
    assert (
        rollout._effective_task_path / "environment" / "skills" / "alpha" / "SKILL.md"
    ).exists()
    dockerfile_text = (
        rollout._effective_task_path / "environment" / "Dockerfile"
    ).read_text()
    assert "COPY _deps/skills /skills/" not in dockerfile_text
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["include_task_skills"] is True
    assert config["skills_sandbox_dir"] == "/skills"


@pytest.mark.asyncio
async def test_rollout_setup_records_self_gen_artifact_mode(
    tmp_path: Path,
) -> None:
    """Guards PR #233 so lowered self-gen trials keep self-gen artifact metadata."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "alpha").mkdir(parents=True)
    (task / "environment" / "skills" / "alpha" / "SKILL.md").write_text("# Alpha\n")
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "instruction.md").write_text("solve\n")
    (task / "task.toml").write_text('version = "1.0"\n')

    env = MagicMock()
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.create_environment.return_value = env

    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            scenes=[Scene.single(agent="claude-agent-acp")],
            jobs_dir=tmp_path / "jobs",
            skill_mode=SKILL_MODE_NO_SKILL,
            artifact_skill_mode=SKILL_MODE_SELF_GEN,
            planes=planes,
        )
    )
    await rollout.setup()

    assert rollout._effective_task_path != task
    assert not (rollout._effective_task_path / "environment" / "skills").exists()
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["skill_mode"] == "self-gen"
    assert config["skill_source"] == "self_generated"
    assert config["include_task_skills"] is False
    assert config["effective_skills_dir"] is None


def test_persist_sandbox_info_writes_sandbox_json(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554: Daytona sandbox IDs must
    be persisted immediately after creation so interrupted runs can be
    audited and cleaned up."""

    class FakeDaytonaSandbox:
        sandbox_id = "sb-abc123"

    _persist_sandbox_info(FakeDaytonaSandbox(), tmp_path)

    info = json.loads((tmp_path / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-abc123"
    assert info["provider"] == "FakeDaytonaSandbox"
    assert "created_at" in info


def test_persist_sandbox_info_skips_when_no_sandbox_id(tmp_path: Path) -> None:
    """No sandbox.json for backends without a provider-side ID (e.g. Docker)."""

    class FakeDockerSandbox:
        sandbox_id = None

    _persist_sandbox_info(FakeDockerSandbox(), tmp_path)
    assert not (tmp_path / "sandbox.json").exists()


def test_persist_sandbox_info_skips_when_no_rollout_dir() -> None:
    """No crash when rollout_dir is None (shouldn't happen but be safe)."""

    class FakeSandbox:
        sandbox_id = "sb-xyz"

    _persist_sandbox_info(FakeSandbox(), None)


def test_result_json_includes_sandbox_id(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554: result.json should include
    the sandbox_id for completed runs so post-mortem cleanup can reconcile
    Daytona sandboxes against rollout artifacts."""
    from datetime import datetime

    _build_rollout_result(
        tmp_path,
        task_name="my-task",
        rollout_name="my-rollout",
        agent="oracle",
        agent_name="oracle",
        model="",
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"score": 1.0},
        started_at=datetime(2026, 5, 25, 12, 0),
        timing={},
        sandbox_id="sb-daytona-123",
    )

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["sandbox_id"] == "sb-daytona-123"


def test_result_json_sandbox_id_null_for_docker(tmp_path: Path) -> None:
    """Docker runs have no provider-side sandbox ID."""
    from datetime import datetime

    _build_rollout_result(
        tmp_path,
        task_name="my-task",
        rollout_name="my-rollout",
        agent="oracle",
        agent_name="oracle",
        model="",
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"score": 1.0},
        started_at=datetime(2026, 5, 25, 12, 0),
        timing={},
    )

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["sandbox_id"] is None


@pytest.mark.asyncio
async def test_on_started_persists_before_upload(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554: the on_started callback must
    fire after the sandbox starts but before any upload, so a sandbox id is
    persisted even when an upload later fails."""
    order: list[str] = []

    class RecordingEnv(FakeUploadEnv):
        async def start(self, force_build: bool) -> None:
            await super().start(force_build)
            order.append("start")

        async def upload_file(self, source: Path | str, target: str) -> None:
            order.append("upload_file")
            await super().upload_file(source, target)

    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("solve\n")
    env = RecordingEnv()

    await _start_env_and_upload(
        env, task, {}, on_started=lambda: order.append("on_started")
    )

    assert order == ["start", "on_started", "upload_file"]


@pytest.mark.asyncio
async def test_sandbox_json_survives_upload_failure(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554: a sandbox.json must exist
    even when an upload fails after the sandbox was created — otherwise an
    interrupted run leaks a live Daytona sandbox with no audit/cleanup record."""
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("solve\n")

    class FailingUploadEnv(FakeUploadEnv):
        sandbox_id = "sb-failwindow"

        async def upload_file(self, source: Path | str, target: str) -> None:
            raise RuntimeError("upload boom")

    env = FailingUploadEnv()

    with pytest.raises(RuntimeError, match="upload boom"):
        await _start_env_and_upload(
            env,
            task,
            {},
            on_started=lambda: _persist_sandbox_info(env, rollout_dir),
        )

    info = json.loads((rollout_dir / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-failwindow"

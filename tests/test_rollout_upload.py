"""Tests for rollout startup uploads and sandbox info persistence."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.diagnostics import RolloutDiagnostics
from benchflow.rollout import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    Rollout,
    RolloutConfig,
    Scene,
    _build_rollout_result,
    _publish_trajectory_for_verifier,
    _resolve_agent_cwd,
    _start_env_and_upload,
)
from benchflow.sandbox.metadata import persist_sandbox_info


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
async def test_resolve_agent_cwd_uses_configured_workdir() -> None:
    """environment.workdir becomes the executable agent workspace."""
    env = MagicMock()
    env.exec = AsyncMock(
        return_value=MagicMock(stdout="/repo\n", stderr="", return_code=0)
    )
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(workdir="/repo"))
    )

    agent_cwd = await _resolve_agent_cwd(env, task)

    assert agent_cwd == "/repo"
    env.exec.assert_awaited_once_with(
        "mkdir -p /repo && cd /repo && pwd",
        user="root",
        timeout_sec=10,
    )


@pytest.mark.asyncio
async def test_resolve_agent_cwd_falls_back_to_container_pwd() -> None:
    """Tasks without environment.workdir preserve the existing pwd discovery."""
    env = MagicMock()
    env.exec = AsyncMock(
        return_value=MagicMock(stdout="/app\n", stderr="", return_code=0)
    )
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(workdir=None))
    )

    agent_cwd = await _resolve_agent_cwd(env, task)

    assert agent_cwd == "/app"
    env.exec.assert_awaited_once_with("pwd", timeout_sec=10)


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
async def test_start_env_uploads_native_oracle_dir(
    tmp_path: Path,
) -> None:
    """Guards commit 67378ddd's task.md standard oracle mount path."""
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "oracle").mkdir(parents=True)
    (task / "task.md").write_text(
        """---
version: "1.0"
---
## prompt

solve
"""
    )
    (task / "oracle" / "solve.sh").write_text("echo ok\n")
    env = FakeUploadEnv()
    timing: dict[str, float] = {}

    await _start_env_and_upload(env, task, timing)

    assert (task / "oracle", "/oracle") in env.uploaded_dirs
    assert (task / "oracle", "/solution") not in env.uploaded_dirs


@pytest.mark.asyncio
async def test_start_env_uploads_task_md_prompt_as_instruction(
    tmp_path: Path,
) -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md runtime prompt."""
    task = tmp_path / "task-md"
    task.mkdir()
    (task / "task.md").write_text(
        """---
version: "1.0"
---
## prompt

Solve from the unified document.
"""
    )
    env = FakeUploadEnv()
    timing: dict[str, float] = {}

    await _start_env_and_upload(env, task, timing)

    assert env.started is True
    assert env.uploaded_file_contents == [
        ("Solve from the unified document.\n", "/instruction.md")
    ]


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
@pytest.mark.parametrize(
    ("skill_mode", "include_task_skills"),
    [
        (SKILL_MODE_NO_SKILL, False),
        (SKILL_MODE_WITH_SKILL, True),
    ],
)
async def test_rollout_setup_resolves_native_skillsbench_task_skill_modes(
    tmp_path: Path,
    skill_mode: str,
    include_task_skills: bool,
) -> None:
    """Guards PR #1's native SkillsBench task.md skill-mode dogfood."""
    task = tmp_path / "weighted-gdp-calc"
    shutil.copytree(
        Path("docs/examples/task-md/real-skillsbench/weighted-gdp-calc"),
        task,
    )
    assert (task / "task.md").is_file()
    assert (task / "environment" / "skills" / "xlsx" / "SKILL.md").is_file()

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
            jobs_dir=tmp_path / f"jobs-{skill_mode}",
            skill_mode=skill_mode,
            planes=planes,
        )
    )
    await rollout.setup()

    assert rollout._effective_task_path != task
    assert (task / "environment" / "skills" / "xlsx" / "SKILL.md").is_file()
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["skill_mode"] == skill_mode
    assert config["include_task_skills"] is include_task_skills

    effective_skills = rollout._effective_task_path / "environment" / "skills"
    if skill_mode == SKILL_MODE_NO_SKILL:
        assert not effective_skills.exists()
        assert config["skill_source"] == "none"
        assert config["effective_skills_dir"] is None
        assert config["skills_sandbox_dir"] is None
        planes.inject_skills_into_dockerfile.assert_not_called()
    else:
        assert (effective_skills / "xlsx" / "SKILL.md").is_file()
        assert config["skill_source"] == "task_bundled"
        assert config["effective_skills_dir"] == str(effective_skills)
        assert config["skills_sandbox_dir"] == "/skills"
        planes.inject_skills_into_dockerfile.assert_called_once_with(
            rollout._effective_task_path,
            effective_skills,
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

    persist_sandbox_info(FakeDaytonaSandbox(), tmp_path)

    info = json.loads((tmp_path / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-abc123"
    assert info["provider"] == "FakeDaytonaSandbox"
    assert "created_at" in info


def test_persist_sandbox_info_skips_when_no_sandbox_id(tmp_path: Path) -> None:
    """No sandbox.json for backends without a provider-side ID (e.g. Docker)."""

    class FakeDockerSandbox:
        sandbox_id = None

    persist_sandbox_info(FakeDockerSandbox(), tmp_path)
    assert not (tmp_path / "sandbox.json").exists()


def test_persist_sandbox_info_skips_when_no_rollout_dir() -> None:
    """No crash when rollout_dir is None (shouldn't happen but be safe)."""

    class FakeSandbox:
        sandbox_id = "sb-xyz"

    persist_sandbox_info(FakeSandbox(), None)


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


@pytest.mark.asyncio
async def test_rollout_result_falls_back_to_env_sandbox_id_after_start_failure(
    tmp_path: Path,
) -> None:
    """Guards the fix from PR #563 for issue #554: if Daytona creates the
    provider sandbox and then fails later in start(), result.json still records
    env.sandbox_id even though the rollout on_started callback never ran."""
    from datetime import datetime

    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    task = tmp_path / "task"
    task.mkdir()

    class CreatedThenFailingEnv(FakeUploadEnv):
        sandbox_id = "sb-post-create-fail"

        async def start(self, force_build: bool) -> None:
            persist_sandbox_info(self, rollout_dir)
            raise RuntimeError("post-create startup failed")

    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=task,
        scenes=[Scene.single(agent="oracle")],
        jobs_dir=tmp_path / "jobs",
    )
    rollout._rollout_dir = rollout_dir
    rollout._started_at = datetime(2026, 5, 25, 12, 0)
    rollout._rollout_name = "my-rollout"
    rollout._agent_name = "oracle"
    rollout._env = CreatedThenFailingEnv()
    rollout._env_externally_owned = False
    rollout._timing = {}
    rollout._executed_prompts = []
    rollout._resolved_prompts = []
    rollout._n_tool_calls = 0
    rollout._error = "post-create startup failed"
    rollout._verifier_error = None
    rollout._export_error = None
    rollout._trajectory = []
    rollout._partial_trajectory = False
    rollout._trajectory_source = None
    rollout._rewards = None
    rollout._evolved_skills = None
    rollout._diagnostics = RolloutDiagnostics()
    rollout._usage_metrics = {}
    rollout._task_skill_policy = None
    rollout._sandbox_id = None

    with pytest.raises(RuntimeError, match="post-create startup failed"):
        await rollout.start()

    assert rollout._sandbox_id is None
    rollout._build_result()

    sandbox_info = json.loads((rollout_dir / "sandbox.json").read_text())
    result = json.loads((rollout_dir / "result.json").read_text())
    assert sandbox_info["sandbox_id"] == "sb-post-create-fail"
    assert result["sandbox_id"] == "sb-post-create-fail"


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
            on_started=lambda: persist_sandbox_info(env, rollout_dir),
        )

    info = json.loads((rollout_dir / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-failwindow"


@pytest.mark.asyncio
async def test_sandbox_json_survives_interrupt_inside_start(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554: the in-``start()`` interrupt
    window. A Daytona sandbox id becomes available mid-``start()`` (right after
    the provider create call), but ``start()`` then does substantial work — DinD
    launches dockerd and polls for the daemon for tens of seconds — before it
    returns. The rollout-layer ``on_started`` callback only fires after
    ``start()`` returns, so an interrupt (CancelledError/SIGINT/timeout) in that
    stretch would leave a live server-side sandbox with no ``sandbox.json``.

    Daytona now persists the id the instant ``self._sandbox`` is assigned
    (``_create_sandbox`` calls ``persist_sandbox_info`` before the daemon-wait).
    This models that: a fake env whose ``start()`` persists the id and then is
    cancelled before returning must still leave a ``sandbox.json`` behind."""
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("solve\n")

    class InterruptedInStartEnv(FakeUploadEnv):
        sandbox_id = "sb-instart"

        async def start(self, force_build: bool) -> None:
            # Provider create returned: the sandbox id is now known. Persist it
            # immediately (this is what DaytonaSandbox._create_sandbox does on
            # assignment, before the DinD daemon-wait) ...
            persist_sandbox_info(self, rollout_dir)
            # ... then get cancelled during the long daemon-wait, before start()
            # could return and before on_started would have fired.
            raise asyncio.CancelledError

    env = InterruptedInStartEnv()

    captured: list[str] = []

    with pytest.raises(asyncio.CancelledError):
        await _start_env_and_upload(
            env,
            task,
            {},
            on_started=lambda: captured.append("on_started"),
        )

    # on_started never ran (start() never returned) — proving the window is real.
    assert captured == []
    # ... yet sandbox.json is present, written from inside start().
    info = json.loads((rollout_dir / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-instart"


def _make_daytona_sandbox(rollout_dir: Path, create_impl):
    """Build a bare ``DaytonaSandbox`` wired for ``_create_sandbox`` (#554/#563).

    ``__init__`` materializes the daytona SDK and picks a strategy, neither of
    which ``_create_sandbox``/``_on_sandbox_created`` need. We construct via
    ``__new__`` and set only the attributes those two methods touch:
    ``_client_manager`` (whose ``get_client()`` returns a fake daytona whose
    ``create()`` runs ``create_impl``), ``task_env_config`` (for
    ``build_timeout_sec``), ``rollout_paths`` (a real ``RolloutPaths`` at
    ``rollout_dir``), a no-op ``logger`` and ``_sandbox=None``.
    """
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.task import RolloutPaths

    env = DaytonaSandbox.__new__(DaytonaSandbox)

    class _FakeDaytona:
        async def create(self, params, timeout):
            return await create_impl(params, timeout)

    class _FakeClientManager:
        async def get_client(self):
            return _FakeDaytona()

    env._client_manager = _FakeClientManager()
    env._sandbox = None
    env.task_env_config = MagicMock(build_timeout_sec=10)
    env.rollout_paths = RolloutPaths(rollout_dir)
    env.logger = MagicMock()
    return env


@pytest.mark.asyncio
async def test_create_sandbox_persists_sandbox_json_normal_path(tmp_path: Path) -> None:
    """Guards the fix from PR #563 for issue #554 at the real call site: the
    normal ``asyncio.wait_for`` success branch of
    ``DaytonaSandbox._create_sandbox`` must persist ``sandbox.json`` the instant
    ``self._sandbox`` is assigned — via ``_on_sandbox_created`` — not only after
    ``start()`` returns. Unlike ``test_sandbox_json_survives_interrupt_inside_start``
    (which hand-calls ``persist_sandbox_info`` from a fake ``start()``), this
    drives the production ``_create_sandbox``/``_on_sandbox_created`` wiring, so
    deleting the ``_on_sandbox_created()`` call in ``daytona.py`` fails it."""
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()

    class _FakeCreatedSandbox:
        id = "sb-normal-123"

    async def create_impl(params, timeout):
        return _FakeCreatedSandbox()

    env = _make_daytona_sandbox(rollout_dir, create_impl)

    await env._create_sandbox(params=MagicMock())

    assert env.sandbox_id == "sb-normal-123"
    info = json.loads((rollout_dir / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-normal-123"
    assert info["provider"] == "DaytonaSandbox"


@pytest.mark.asyncio
async def test_create_sandbox_persists_sandbox_json_on_cancelled_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards the fix from PR #563 for issue #554 on the
    ``except asyncio.CancelledError`` recovery branch of
    ``DaytonaSandbox._create_sandbox``. If the shielded first wait is cancelled
    but the underlying ``create()`` task still yields a sandbox, the recovery
    block awaits it, persists ``sandbox.json`` via ``_on_sandbox_created``, then
    re-raises — so the live server-side sandbox is never orphaned without an
    audit file. We make the first ``asyncio.wait_for`` raise ``CancelledError``
    and let the second await the (completed) create task."""
    import asyncio as _asyncio

    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()

    class _FakeCreatedSandbox:
        id = "sb-cancelled-456"

    async def create_impl(params, timeout):
        return _FakeCreatedSandbox()

    env = _make_daytona_sandbox(rollout_dir, create_impl)

    real_wait_for = _asyncio.wait_for
    calls = {"n": 0}

    async def fake_wait_for(awaitable, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            # First (shielded) wait is interrupted — but the create task keeps
            # running and completes, exactly as in a mid-create SIGINT/cancel.
            raise asyncio.CancelledError
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr("benchflow.sandbox.daytona.asyncio.wait_for", fake_wait_for)

    with pytest.raises(asyncio.CancelledError):
        await env._create_sandbox(params=MagicMock())

    assert calls["n"] == 2  # both the shielded wait and the recovery wait ran
    assert env.sandbox_id == "sb-cancelled-456"
    info = json.loads((rollout_dir / "sandbox.json").read_text())
    assert info["sandbox_id"] == "sb-cancelled-456"
    assert info["provider"] == "DaytonaSandbox"

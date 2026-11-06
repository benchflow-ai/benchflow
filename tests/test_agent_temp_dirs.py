"""The agent tool's temp folders, made the sandbox user's before launch
(#1183).

OpenCode's postinstall runs `opencode --version` as root, which creates
/tmp/opencode root-owned 755; OpenCode then cannot write it as the agent user
and every trial ended in "OpenCode service failure" at the agent budget. The
real-container half of these checks is in
``test_agent_sandbox_files_container.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.registry import AGENTS, resolve_agent
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._setup import _agent_temp_dirs_cmd, _prepare_agent_temp_dirs
from benchflow.task import RolloutPaths


class RecordingEnv:
    def __init__(self) -> None:
        self.execs: list[tuple[str, str | None]] = []

    async def exec(self, command: str, user: str | None = None, timeout_sec=None):
        self.execs.append((command, user))
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def test_opencode_declares_its_temp_folder() -> None:
    assert AGENTS["opencode"].temp_dirs == ["opencode"]
    assert resolve_agent("acpx/opencode").temp_dirs == ["opencode"]


@pytest.mark.parametrize("name", ["../etc", "a/b", ".", "..", "", "$(id)", "*"])
def test_temp_folder_entries_are_plain_names(name) -> None:
    with pytest.raises(ValueError, match="plain folder name"):
        _agent_temp_dirs_cmd([name], "agent")


def test_every_registered_temp_folder_is_a_plain_name() -> None:
    for config in AGENTS.values():
        _agent_temp_dirs_cmd(config.temp_dirs, "agent")


@pytest.mark.asyncio
async def test_temp_folders_are_skipped_without_a_sandbox_user() -> None:
    env = RecordingEnv()
    await _prepare_agent_temp_dirs(env, AGENTS["opencode"], None)
    await _prepare_agent_temp_dirs(env, AGENTS["claude-agent-acp"], "agent")
    await _prepare_agent_temp_dirs(env, MagicMock(), "agent")
    assert env.execs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (3, "is a link or not a folder"),
        (4, "could not be made or given"),
        (5, "could hold files another user put there"),
    ],
)
async def test_a_temp_folder_that_cannot_be_given_refuses_the_trial(code, reason):
    from benchflow._utils.scoring import SANDBOX_SETUP, classify_error

    env = MagicMock()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=code, stdout="/tmp/opencode\n", stderr=""
        )
    )
    with pytest.raises(RuntimeError, match=reason) as excinfo:
        await _prepare_agent_temp_dirs(env, AGENTS["opencode"], "agent")
    assert "/tmp/opencode" in str(excinfo.value)
    assert classify_error(str(excinfo.value)) == SANDBOX_SETUP


def _install_rollout(tmp_path: Path, monkeypatch, env: Any, agent: str) -> Rollout:
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.side_effect = lambda _a, _m, agent_env: dict(
        agent_env or {}
    )
    planes.agent_launch.side_effect = (
        lambda agent, *, disallow_web_tools, disallow_hosted_search=False: agent
    )
    planes.create_environment.return_value = env
    planes.install_agent = AsyncMock(return_value=AGENTS[agent])
    planes.setup_sandbox_user = AsyncMock(return_value="/app")
    for name in (
        "write_credential_files",
        "upload_subscription_auth",
        "apply_web_tool_policy",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_verifier_runtime",
        "deploy_skills",
        "lockdown_paths",
    ):
        setattr(planes, name, AsyncMock())
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    rollout_dir = tmp_path / "rollout"
    task = SimpleNamespace(
        name="t",
        config=SimpleNamespace(
            sandbox=SimpleNamespace(
                env={},
                setup_commands=[],
                network_mode=None,
                allow_internet=True,
                skills_dir=None,
                docker_image=None,
                workdir=None,
            ),
            agent=SimpleNamespace(prompt_prefix=None, timeout_sec=60),
            verifier=SimpleNamespace(service="main", env={}),
        ),
    )

    def fake_init_rollout(task_path, job_name, rollout_name, jobs_dir):
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (rollout_dir / subdir).mkdir(parents=True, exist_ok=True)
        return (
            task,
            rollout_dir,
            RolloutPaths(rollout_dir=rollout_dir),
            datetime.now(),
            "job",
            "rollout",
        )

    monkeypatch.setattr("benchflow.rollout._init_rollout", fake_init_rollout)
    monkeypatch.setattr(
        "benchflow.rollout.enforce_codex_apps_policy",
        AsyncMock(side_effect=lambda _env, **k: k["agent_env"]),
    )
    return Rollout(
        RolloutConfig(
            task_path=task_dir,
            agent=agent,
            model="openai/test-model",
            sandbox_user="agent",
            planes=planes,
        )
    )


@pytest.mark.asyncio
async def test_install_gives_opencode_its_temp_folder_after_the_user_exists(
    tmp_path, monkeypatch
) -> None:
    """Guards #1183: OpenCode's /tmp/opencode, root-owned by its own install,
    becomes the agent user's before launch."""
    env = MagicMock()
    order: list[str] = []

    async def exec_(command, **kwargs):
        if "opencode" in command and "chown" in command:
            order.append("temp_dirs")
            assert kwargs.get("user") == "root"
        return SimpleNamespace(return_code=0, stdout="/app\n", stderr="")

    env.exec = AsyncMock(side_effect=exec_)
    env.stop = AsyncMock()
    rollout = _install_rollout(tmp_path, monkeypatch, env, "opencode")
    rollout._planes.setup_sandbox_user.side_effect = lambda *a, **k: (
        order.append("user"),
        "/app",
    )[1]
    await rollout.setup()

    await rollout.install_agent()

    assert order == ["user", "temp_dirs"]


def test_temp_folder_step_reowns_only_folders_and_single_link_files() -> None:
    cmd = _agent_temp_dirs_cmd(["opencode"], "agent")
    assert 't="${TMPDIR:-/tmp}"' in cmd
    assert '[ -L "$d" ]' in cmd
    assert "-xdev" in cmd and "-type f -links 1" in cmd
    assert "chown -R" not in cmd


def test_temp_folder_walk_is_depth_first_and_never_follows_a_link() -> None:
    """A leftover sandbox-user process from an earlier role could swap an
    entry for a link mid-walk: the walk gives a folder away only after its
    contents, every chown is -h, and no chmod walk runs after the chown."""
    cmd = _agent_temp_dirs_cmd(["opencode"], "agent")
    assert "-xdev -depth" in cmd
    assert "chown -h agent:agent" in cmd
    assert "chown agent:agent" not in cmd
    assert cmd.count("chmod") == 1
    assert cmd.index("chmod u+rwx") < cmd.index("-exec chown -h agent:agent")
    # Re-owned only when nobody but root could have written it.
    assert "-perm -002" in cmd and "-user 0" in cmd
    assert "exit 5" in cmd


@pytest.mark.asyncio
async def test_connect_as_gives_a_role_agent_its_temp_folder(tmp_path) -> None:
    """Guards V6-R1-B-1: a role agent that differs from the primary (OpenCode
    after claude-agent-acp) is installed as root in connect_as, so its
    /tmp/opencode must become the agent user's there too (#1183)."""
    from benchflow.rollout import Role, Scene

    primary = Role(name="coder", agent="claude-agent-acp", model="test-model")
    reviewer = Role(name="reviewer", agent="opencode", model="test-model")
    rollout = Rollout(
        RolloutConfig(
            task_path=Path("/fake/task"),
            scenes=[Scene(roles=[primary, reviewer])],
            sandbox_user="agent",
        )
    )
    env = RecordingEnv()
    rollout._env = env
    rollout._rollout_dir = tmp_path
    rollout._agent_cwd = None
    rollout._agent_cfg = AGENTS["claude-agent-acp"]
    rollout._phase = "idle"
    planes = MagicMock()
    planes.agent_launch.return_value = "opencode"
    planes.resolve_agent_env.side_effect = lambda _agent, _model, env: env or {}
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    order: list[str] = []

    async def install(*_a, **_k):
        order.append("install")
        return AGENTS["opencode"]

    async def credentials(*_a, **_k):
        order.append("credentials")

    planes.install_agent = AsyncMock(side_effect=install)
    planes.write_credential_files = AsyncMock(side_effect=credentials)
    planes.upload_subscription_auth = AsyncMock()
    planes.apply_web_tool_policy = AsyncMock()
    planes.connect_acp = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), AsyncMock(), "agent")
    )
    rollout._planes = planes
    rollout._install_native_cli = AsyncMock()

    real_exec = env.exec

    async def exec_(command: str, user: str | None = None, timeout_sec=None):
        if "opencode" in command and "chown" in command:
            order.append("temp_dirs")
        return await real_exec(command, user=user, timeout_sec=timeout_sec)

    env.exec = exec_  # type: ignore[method-assign]
    await rollout.connect_as(reviewer)

    temp_steps = [
        (command, user)
        for command, user in env.execs
        if "opencode" in command and "chown" in command
    ]
    assert len(temp_steps) == 1
    command, user = temp_steps[0]
    assert user == "root"
    assert "chown -h agent:agent" in command
    assert order[:3] == ["install", "temp_dirs", "credentials"]

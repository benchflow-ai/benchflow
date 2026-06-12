"""Tests for executing Scene-authored Steps through Rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import (
    Role,
    Rollout,
    RolloutConfig,
    Scene,
    Turn,
    _resolve_skill_creator_root,
    _self_gen_prompt,
)
from benchflow.scenes import compile_scenes_to_steps


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnv:
    """Minimal sandbox mock for Step execution tests."""

    def __init__(self) -> None:
        self._exec_log: list[str] = []
        self._uploads: list[tuple[Path, str]] = []

    async def exec(self, cmd: str, **kwargs) -> FakeExecResult:
        self._exec_log.append(cmd)
        return FakeExecResult()

    async def upload_dir(self, src: Path, dst: str) -> None:
        self._uploads.append((Path(src), dst))


def _make_trial(scene: Scene) -> Rollout:
    config = RolloutConfig(
        task_path=Path("tasks/fake"),
        scenes=[scene],
        environment="docker",
    )
    trial = Rollout(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task"]
    return trial


@pytest.fixture
def coder_reviewer_scene() -> Scene:
    return Scene(
        name="code-review",
        roles=[
            Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
            Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
        ],
        turns=[
            Turn("coder"),
            Turn("reviewer", "Review the code."),
            Turn("coder", "Fix issues."),
        ],
    )


@pytest.fixture
def self_review_scene() -> Scene:
    return Scene(
        name="self-review",
        roles=[Role("agent", "gemini", "gemini-3.1-flash-lite-preview")],
        turns=[
            Turn("agent"),
            Turn("agent", "Review your solution and fix edge cases."),
        ],
    )


async def test_run_steps_executes_compiled_turns(
    coder_reviewer_scene: Scene,
) -> None:
    """Guards the fix from PR #515 for issue #413: rollout runs compiled Steps."""
    trial = _make_trial(coder_reviewer_scene)
    prompts_received: list[tuple[str, list[str]]] = []
    roles_connected: list[str] = []
    turn_roles = [turn.role for turn in coder_reviewer_scene.turns]

    async def fake_connect_as(role: Role) -> None:
        roles_connected.append(role.name)

    async def fake_execute(prompts=None):
        role = turn_roles[len(prompts_received)]
        prompts_received.append((role, prompts or []))
        return [], 0

    trial.connect_as = fake_connect_as  # type: ignore[method-assign]
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    await trial._run_steps(
        compile_scenes_to_steps(
            [coder_reviewer_scene], default_prompt=trial._resolved_prompts[0]
        )
    )

    assert roles_connected == ["coder", "reviewer", "coder"]
    assert prompts_received == [
        ("coder", ["Solve the task"]),
        ("reviewer", ["Review the code."]),
        ("coder", ["Fix issues."]),
    ]
    assert trial.disconnect.call_count == 3
    assert all("outbox" not in cmd for cmd in trial._env._exec_log)


async def test_run_steps_reuses_session_for_same_role(self_review_scene: Scene) -> None:
    """Guards the fix from PR #515 for issue #413: same-role turns stay Steps."""
    trial = _make_trial(self_review_scene)
    prompts_received: list[str] = []

    async def fake_execute(prompts=None):
        prompts_received.extend(prompts or [])
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    await trial._run_steps(
        compile_scenes_to_steps([self_review_scene], default_prompt="Solve")
    )

    trial.connect_as.assert_awaited_once()
    trial.disconnect.assert_awaited_once()
    assert prompts_received == ["Solve", "Review your solution and fix edge cases."]


async def test_run_steps_restarts_same_role_at_scene_boundary() -> None:
    """Guards commit 67378ddd's 2026-06-04 task.md multi-scene context."""
    role = Role("solver", "gemini", "flash")
    scenes = [
        Scene(name="prep", roles=[role], turns=[Turn("solver", "Prepare.")]),
        Scene(name="solve", roles=[role], turns=[Turn("solver", "Solve.")]),
    ]
    trial = _make_trial(scenes[0])
    prompts_received: list[str] = []

    async def fake_execute(prompts=None):
        prompts_received.extend(prompts or [])
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    await trial._run_steps(compile_scenes_to_steps(scenes, default_prompt="Fallback"))

    assert trial.connect_as.await_count == 2
    assert trial.disconnect.await_count == 2
    assert prompts_received == ["Prepare.", "Solve."]


async def test_run_steps_disconnects_active_agent_when_execute_times_out(
    self_review_scene: Scene,
) -> None:
    """Guards the fix from PR #515 for issue #413: Step timeout cleanup."""
    trial = _make_trial(self_review_scene)

    async def fake_execute(prompts=None):
        raise TimeoutError("Agent prompt exceeded wall-clock budget 5s")

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    with pytest.raises(TimeoutError, match="wall-clock budget"):
        await trial._run_steps(
            compile_scenes_to_steps([self_review_scene], default_prompt="Solve")
        )

    trial.disconnect.assert_awaited_once()


async def test_execute_uses_active_role_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role-level timeouts must override rollout defaults during Step execution."""
    role = Role(
        "solver",
        "gemini",
        "flash",
        timeout_sec=12,
        idle_timeout_sec=3,
    )
    trial = _make_trial(Scene(roles=[role], turns=[Turn("solver")]))
    trial._acp_client = object()
    trial._session = object()
    trial._timeout = 99
    trial._active_role = role
    captured: dict[str, int | None] = {}

    async def fake_execute_prompts(
        _client,
        _session,
        _prompts,
        timeout,
        *,
        idle_timeout=None,
    ):
        captured["timeout"] = timeout
        captured["idle_timeout"] = idle_timeout
        return [], 0

    monkeypatch.setattr(trial._planes, "execute_prompts", fake_execute_prompts)

    await trial.execute()

    assert captured == {"timeout": 12, "idle_timeout": 3}


async def test_heterogeneous_agent_install(coder_reviewer_scene: Scene) -> None:
    """connect_as still receives non-primary roles after Scene desugaring."""
    scene = Scene(
        name="hetero",
        roles=[
            Role("coder", "gemini", "flash"),
            Role("reviewer", "claude-agent-acp", "haiku"),
        ],
        turns=[Turn("coder"), Turn("reviewer", "Review.")],
    )
    config = RolloutConfig(
        task_path=Path("tasks/fake"),
        scenes=[scene],
        environment="docker",
        agent="gemini",
    )
    trial = Rollout(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task"]

    installed_agents: list[str] = []

    async def tracking_connect_as(role: Role) -> None:
        if role.agent != config.primary_agent:
            installed_agents.append(role.agent)

    async def fake_execute(prompts=None):
        return [], 0

    trial.connect_as = tracking_connect_as  # type: ignore[method-assign]
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    await trial._run_steps(compile_scenes_to_steps([scene], default_prompt="Solve"))

    assert "claude-agent-acp" in installed_agents


def test_self_gen_mode_still_requires_runtime_orchestration(tmp_path: Path) -> None:
    """Direct Rollout scenes cannot silently skip self-gen setup."""
    skills_root = tmp_path / "skills"
    skill_creator = skills_root / "skill-creator"
    skill_creator.mkdir(parents=True)
    (skill_creator / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: Create skills\n---\n# Skill Creator\n"
    )

    cfg = RolloutConfig.from_legacy(
        task_path=Path("tasks/fake"),
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        skill_mode="self-gen",
        skill_creator_dir=skill_creator,
    )

    with pytest.raises(ValueError, match="self-gen requires"):
        _ = cfg.effective_scenes


def test_self_gen_mode_defaults_to_repo_claude_skill_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-gen mode defaults to the repo's official skill-creator pack."""
    monkeypatch.delenv("BENCHFLOW_SKILL_CREATOR_DIR", raising=False)

    skills_root, skill_name = _resolve_skill_creator_root(None)

    assert skills_root.parts[-2:] == (".claude", "skills")
    assert (skills_root / "skill-creator" / "SKILL.md").exists()
    assert skill_name == "skill-creator"


def test_self_gen_mode_uses_custom_creator_skill_name(tmp_path: Path) -> None:
    """User-assigned creator packs can use their own SKILL.md name."""
    skills_root = tmp_path / "custom-skills"
    creator = skills_root / "custom-creator"
    creator.mkdir(parents=True)
    (creator / "SKILL.md").write_text(
        "---\nname: custom-creator\ndescription: Create task skills\n---\n"
    )

    resolved_root, skill_name = _resolve_skill_creator_root(skills_root)
    prompt = _self_gen_prompt(
        Path("tasks/fake"),
        "/app/generated-skills",
        skill_name,
    )

    assert resolved_root == skills_root
    assert "Use the custom-creator skill exactly as provided" in prompt


async def test_scene_skills_upload_local_root_and_link_agent_paths(
    tmp_path: Path,
) -> None:
    """Scene.skills_dir can activate a local skills root without global skills_dir."""
    skills_root = tmp_path / "skills"
    (skills_root / "skill-creator").mkdir(parents=True)
    (skills_root / "skill-creator" / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: Create skills\n---\n# Skill Creator\n"
    )
    scene = Scene(
        name="skill-gen",
        roles=[Role("skill_creator", "claude-agent-acp", "haiku")],
        turns=[Turn("skill_creator", "create skill")],
        skills_dir=skills_root,
    )
    trial = _make_trial(scene)
    trial._agent_cwd = "/app"

    await trial._activate_step_skills(compile_scenes_to_steps([scene])[0])

    assert "mkdir -p /skills" in trial._env._exec_log
    assert trial._env._uploads == [(skills_root, "/skills/skill-gen")]
    assert any(
        "ln -sfn /skills/skill-gen /home/agent/.claude/skills" in cmd
        for cmd in trial._env._exec_log
    )


async def test_scene_skills_link_remote_generated_root() -> None:
    """Scene.skills_dir can point at a sandbox path produced by an earlier scene."""
    scene = Scene(
        name="solve",
        roles=[Role("solver", "claude-agent-acp", "haiku")],
        turns=[Turn("solver")],
        skills_dir="/app/generated-skills",
    )
    trial = _make_trial(scene)
    trial._agent_cwd = "/app"

    await trial._activate_step_skills(compile_scenes_to_steps([scene])[0])

    assert trial._env._uploads == []
    assert any(
        "ln -sfn /app/generated-skills /home/agent/.claude/skills" in cmd
        for cmd in trial._env._exec_log
    )


async def test_scene_skills_prefers_remote_string_path_over_matching_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute string skills_dir values are sandbox paths, not host uploads."""
    host_root = tmp_path / "app" / "generated-skills"
    host_root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    scene = Scene(
        name="solve",
        roles=[Role("solver", "claude-agent-acp", "haiku")],
        turns=[Turn("solver")],
        skills_dir="/app/generated-skills",
    )
    trial = _make_trial(scene)
    trial._agent_cwd = "/app"

    await trial._activate_step_skills(compile_scenes_to_steps([scene])[0])

    assert trial._env._uploads == []
    assert not any(cmd.startswith("mkdir -p /skills") for cmd in trial._env._exec_log)


async def test_scene_skills_upload_absolute_str_host_dir(tmp_path: Path) -> None:
    """An absolute *str* skills_dir that exists on the host is uploaded.

    Regression test guarding the OpenHands with-skills delivery fix: a prior
    change gated the scene-skill upload on ``isinstance(skills_dir, os.PathLike)``,
    but the SkillsBench entrypoint passes an absolute *str* host path
    (``EvaluationConfig.skills_dir`` is typed ``str``). The gate skipped the
    upload, the path fell through to the in-sandbox branch, and
    ``_link_skill_paths`` produced a dangling symlink -- so deployed task skills
    never reached the agent's available-skills catalog and the with-skills arm
    became a no-op. Uploading must key on the directory existing on the host,
    not on the argument's type.
    """
    skills_root = tmp_path / "environment" / "skills"
    (skills_root / "mesh-analysis").mkdir(parents=True)
    (skills_root / "mesh-analysis" / "SKILL.md").write_text(
        "---\nname: mesh-analysis\ndescription: analyze meshes\n---\n# Mesh\n"
    )
    scene = Scene(
        name="solve",
        roles=[Role("solver", "claude-agent-acp", "haiku")],
        turns=[Turn("solver")],
        skills_dir=str(skills_root),  # absolute STR host path, as production passes
    )
    trial = _make_trial(scene)
    trial._agent_cwd = "/app"

    await trial._activate_step_skills(compile_scenes_to_steps([scene])[0])

    assert trial._env._uploads == [(skills_root, "/skills/solve")]
    assert any(
        "ln -sfn /skills/solve /home/agent/.claude/skills" in cmd
        for cmd in trial._env._exec_log
    )

"""Guards FF-sw-evaluation fixes for regressions at 363d815b."""

import pytest

from benchflow._types import Role, Scene, Turn
from benchflow.rollout import RolloutConfig


@pytest.mark.parametrize(
    "setting",
    [
        "prompt",
        "env",
        "reasoning_effort",
        "timeout_sec",
        "idle_timeout_sec",
        "skills_dir",
        "capabilities",
        "scene_skills",
        "empty_turns",
    ],
)
def test_single_role_roundtrip_preserves_execution(tmp_path, setting):
    role = Role(name="agent", agent="openhands", model="test")
    scene = Scene(roles=[role], turns=[Turn(role="agent")])
    if setting == "prompt":
        scene.turns[0].prompt = "Write 42 to /app/answer.txt"
    elif setting == "scene_skills":
        scene.skills_dir = "scene-skills"
    elif setting == "empty_turns":
        scene.turns = []
    else:
        setattr(
            role,
            setting,
            {
                "env": {"MODE": "test"},
                "reasoning_effort": "high",
                "timeout_sec": 123,
                "idle_timeout_sec": 45,
                "skills_dir": "role-skills",
                "capabilities": ["tool-use"],
            }[setting],
        )
    config = RolloutConfig(
        task_path=tmp_path, agent="openhands", model="test", scenes=[scene]
    )
    restored = RolloutConfig.from_dict(config.to_dict(include_agent_env=True))
    assert restored.effective_scenes == config.effective_scenes
    assert restored.allow_document_user == config.allow_document_user

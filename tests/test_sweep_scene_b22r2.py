"""Guards FF-sw-evaluation-C5 against be37b5e8 default-prompt loss."""

import pytest
import yaml

from benchflow._types import Role, Scene, Turn
from benchflow.rollout import RolloutConfig
from benchflow.rollout._setup import _resolve_prompts
from benchflow.scenes import compile_scenes_to_steps


@pytest.mark.parametrize("wire", ["dict", "yaml"])
@pytest.mark.parametrize("include_agent_env", [False, True])
@pytest.mark.parametrize("prompts", [None, [], [None], ["Write 42"], ["", "next"]])
@pytest.mark.parametrize("shape", ["single", "custom", "multi-role", "multi-scene"])
def test_scene_roundtrip_preserves_runtime_defaults(
    tmp_path, wire, include_agent_env, prompts, shape
):
    (tmp_path / "instruction.md").write_text("Write 0.")
    scene = Scene.single(agent="openhands")
    if shape == "custom":
        scene.name = "custom"
        scene.turns.append(Turn(role="agent", prompt="explicit instruction"))
    elif shape == "multi-role":
        scene.roles.append(Role(name="reviewer", agent="openhands"))
        scene.turns.append(Turn(role="reviewer"))
    scenes = (
        [scene, Scene.single(agent="openhands")] if shape == "multi-scene" else [scene]
    )
    original = RolloutConfig(
        task_path=tmp_path, agent="openhands", prompts=prompts, scenes=scenes
    )
    raw = original.to_dict(include_agent_env=include_agent_env)
    if wire == "yaml":
        raw = yaml.safe_load(yaml.safe_dump(raw))
    restored = RolloutConfig.from_dict(raw)
    assert raw["prompts"] == prompts
    assert restored.prompts == original.prompts

    def instructions(config):
        resolved = _resolve_prompts(config.task_path, config.prompts)
        return compile_scenes_to_steps(
            config.effective_scenes, default_prompt=resolved[0] if resolved else None
        )

    assert instructions(restored) == instructions(original)

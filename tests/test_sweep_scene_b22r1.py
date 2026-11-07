"""Guards FF-sw-evaluation-C1 against d97bf9f6 scene round-trip drift."""

import pytest
import yaml

from benchflow._types import Scene
from benchflow.rollout import RolloutConfig


@pytest.mark.parametrize("wire", ["dict", "yaml"])
@pytest.mark.parametrize("prompts", [None, [None], [], ["explicit"]])
def test_explicit_scene_does_not_adopt_document_turns(tmp_path, wire, prompts):
    (tmp_path / "task.md").write_text(
        "---\nagents:\n  roles:\n    agent:\n      agent: openhands\n"
        "scenes:\n  - name: solve\n    turns:\n"
        "      - role: agent\n        prompt: first\n"
        "      - role: agent\n        prompt: second\n"
        "---\n## prompt\nDocument instruction.\n"
    )
    scene = Scene.single(agent="openhands", model="test", prompts=prompts)
    original = RolloutConfig(
        task_path=tmp_path,
        agent="openhands",
        model="test",
        prompts=prompts,
        scenes=[scene],
    )
    raw = original.to_dict()
    if wire == "yaml":
        raw = yaml.safe_load(yaml.safe_dump(raw))
    restored = RolloutConfig.from_dict(raw)
    assert restored.effective_scenes == original.effective_scenes
    assert restored.allow_document_user == original.allow_document_user


def test_implicit_document_plan_still_roundtrips(tmp_path):
    original = RolloutConfig(task_path=tmp_path, agent="openhands")
    assert (
        RolloutConfig.from_dict(original.to_dict()).effective_scenes
        == original.effective_scenes
    )

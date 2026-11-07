"""Guard CONFIRM-taskmd-rewards-R2-1 against reviewed commit 037efeed."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from benchflow.rewards import builtins, llm
from benchflow.rewards.file_readers import find_deliverables
from tests.test_sweep_judge_confirm2 import native_capture as native_capture


@pytest.mark.parametrize("route", ["llm-config", "llm-strategy", "agent-judge"])
@pytest.mark.parametrize("swap", ["capture-root", "capture-parent", "judge-read"])
async def test_native_judges_keep_capture_outside_sandbox_mounts(
    native_capture, tmp_path, monkeypatch, route, swap
):
    """037efeed: a live sandbox writer cannot redirect capture or judge reads."""
    verifier, box, source = native_capture
    (source / "answer.txt").write_text("correct")
    verifier._logger = logging.getLogger(__name__)
    verifier._task.config.verifier.env = {}
    verifier._task.config.verifier.judge.model = "scripted"
    verifier._task.instruction = "Read the answer."
    monkeypatch.setattr(verifier, "_resolve_rubric_path", lambda _: tmp_path / "rubric")
    monkeypatch.setattr(verifier, "_resolve_llm_judge_context", lambda _: "Read it.")
    paths = verifier._rollout_paths
    shared = paths.verifier_dir / (
        "agent-judge-inputs/0-capture" if route == "agent-judge" else "deliverables"
    )
    controller = tmp_path / "controller-only"
    answer = controller.joinpath(*source.parts[1:]) / "answer.txt"
    answer.parent.mkdir(parents=True)
    answer.write_text("DUMMY_CONTROLLER_ANSWER")
    (answer.parent / "private.txt").write_text("DUMMY_CONTROLLER_PRIVATE")
    original_exec = box.exec
    attacked = False
    private_paths = []

    def attack():
        nonlocal attacked
        if attacked or not shared.is_dir():
            return
        attacked = True
        if swap == "capture-parent":
            victim = shared / source.parts[1]
            target = controller / source.parts[1]
        else:
            victim, target = shared, controller
        if not victim.is_dir():
            return
        victim.rename(victim.with_name(victim.name + "-old"))
        victim.symlink_to(target, target_is_directory=True)

    async def exec_with_writer(command, **kwargs):
        if swap != "judge-read" and "BF copy" in command:
            attack()
        return await original_exec(command, **kwargs)

    box.exec = exec_with_writer

    class Judge:
        events = ()

        def __init__(self, **kwargs):
            pass

        async def score(self, directory):
            private_paths.append(directory)
            if swap == "judge-read":
                attack()
            await asyncio.sleep(0)
            assert find_deliverables(directory, evidence_root=directory) == {
                "answer.txt": "correct"
            }
            assert not directory.is_relative_to(paths.rollout_dir)
            return 1.0

    async def call_judge(model, prompt, **kwargs):
        if swap == "judge-read":
            attack()
        assert "correct" in prompt
        assert "DUMMY_CONTROLLER" not in prompt
        return '{"score": 1}'

    monkeypatch.setattr(builtins, "LLMJudgeRewardFunc", Judge)
    monkeypatch.setattr(llm, "call_judge", call_judge)
    if route == "agent-judge":
        strategy = SimpleNamespace(
            name="judge",
            role="reviewer",
            model=None,
            inputs=[str(source / "answer.txt")],
        )
        result = await verifier._verify_agent_judge(
            strategy=strategy, document=SimpleNamespace(role_prompts={})
        )
    else:
        strategy = (
            SimpleNamespace(input_dir=str(source), model=None)
            if route == "llm-strategy"
            else None
        )
        result = await verifier._verify_llm_judge(strategy=strategy)
    assert result.rewards["reward"] == 1.0
    assert answer.read_text() == "DUMMY_CONTROLLER_ANSWER"
    assert all(not path.exists() for path in private_paths)


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_private_native_capture_lifetime(native_capture, monkeypatch, outcome):
    """037efeed: retain private evidence through judging, then release it on every exit."""
    verifier, _, source = native_capture
    (source / "answer.txt").write_text("correct")
    verifier._logger = logging.getLogger(__name__)
    verifier._task.config.verifier.env = {}
    verifier._task.config.verifier.judge.model = "scripted"
    monkeypatch.setattr(verifier, "_resolve_rubric_path", lambda _: source / "rubric")
    monkeypatch.setattr(verifier, "_resolve_llm_judge_context", lambda _: "Read it.")
    seen = []

    class Judge:
        events = ()

        def __init__(self, **kwargs):
            pass

        async def score(self, directory):
            seen.append(directory)
            assert (directory / "answer.txt").read_text() == "correct"
            if outcome == "error":
                raise RuntimeError("scripted provider failure")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return 1.0

    monkeypatch.setattr(builtins, "LLMJudgeRewardFunc", Judge)
    if outcome == "success":
        await verifier._verify_llm_judge()
    else:
        error = RuntimeError if outcome == "error" else asyncio.CancelledError
        with pytest.raises(error):
            await verifier._verify_llm_judge()
    assert seen and all(not path.exists() for path in seen)

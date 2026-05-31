"""Offline driver tests for the OpenReward / ORS Track-2 runner.

These exercise ``benchflow.openreward_env.run_hosted_env_openreward`` with a
**fake session** that records ``call_tool`` and returns scripted
``ToolOutput``-shaped objects — no network, no platform, no paid rollout. They
run in the default suite (no marker): the real KellyBench live test is the
separate, ``openreward``-marked PR3 gate.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from benchflow.hosted_env import HostedEnvError, HostedEnvRef, HostedEnvRunConfig
from benchflow.openreward_env import (
    ScriptedPolicy,
    _prompt_to_text,
    run_hosted_env_openreward,
)


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolSpec:
    name: str
    description: str = ""
    input_schema: dict | None = None


@dataclass
class FakeToolOutput:
    blocks: list = field(default_factory=list)
    reward: float | None = None
    finished: bool = False
    metadata: dict | None = None


class FakeSession:
    """Records call_tool and returns a scripted sequence of ToolOutputs.

    ``outputs`` is consumed one per ``call_tool``; the last one repeats if the
    loop calls more times than scripted.
    """

    def __init__(
        self,
        *,
        tools: list[FakeToolSpec],
        outputs: list[FakeToolOutput],
        prompt: str = "Solve the task.",
    ) -> None:
        self._tools = tools
        self._outputs = outputs
        self._prompt = prompt
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_prompt(self) -> list[FakeBlock]:
        return [FakeBlock(text=self._prompt)]

    def list_tools(self, format: Any = None) -> list[FakeToolSpec]:
        return list(self._tools)

    def call_tool(
        self, tool_name: str, input: dict[str, Any] | None = None
    ) -> FakeToolOutput:
        self.calls.append((tool_name, dict(input or {})))
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return self._outputs[idx]


def _factory(session: FakeSession):
    @contextmanager
    def factory(config, split, index):
        yield session

    return factory


def _config(tmp_path: Path) -> HostedEnvRunConfig:
    return HostedEnvRunConfig(
        source_env=HostedEnvRef.parse("openreward:GeneralReasoning/KellyBench"),
        model="claude-haiku-4-5",
        jobs_dir=tmp_path,
    )


def test_loop_drives_to_finished_and_maps_reward(tmp_path):
    session = FakeSession(
        tools=[FakeToolSpec(name="think"), FakeToolSpec(name="submit_answer")],
        outputs=[FakeToolOutput(blocks=[FakeBlock("done")], reward=1.0, finished=True)],
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=ScriptedPolicy(answer="42"),
        session_factory=_factory(session),
    )

    # The scripted policy preferred the submit/answer tool, populated `answer`,
    # and the env reported finished on the first call -> reward maps through.
    assert session.calls == [("submit_answer", {"answer": "42"})]
    assert result.reward == 1.0
    assert result.total_tool_calls == 1
    assert result.error is None


def test_loop_continues_until_finished(tmp_path):
    # No submit-style tool -> scripted policy walks tools in order, one per
    # step. The env only finishes on the second ToolOutput.
    session = FakeSession(
        tools=[FakeToolSpec(name="step_a"), FakeToolSpec(name="step_b")],
        outputs=[
            FakeToolOutput(blocks=[FakeBlock("partial")], reward=0.0, finished=False),
            FakeToolOutput(blocks=[FakeBlock("done")], reward=0.5, finished=True),
        ],
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=ScriptedPolicy(),
        session_factory=_factory(session),
    )
    assert [c[0] for c in session.calls] == ["step_a", "step_b"]
    assert result.reward == 0.5
    assert result.total_tool_calls == 2
    assert result.error is None


def test_full_artifact_contract_written(tmp_path):
    session = FakeSession(
        tools=[FakeToolSpec(name="submit")],
        outputs=[FakeToolOutput(blocks=[FakeBlock("ok")], reward=1.0, finished=True)],
        prompt="What is the capital of France?",
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=ScriptedPolicy(answer="Paris"),
        session_factory=_factory(session),
    )
    rd = result.run_dir

    # result.json — legacy contract + openreward lineage stamp.
    result_json = json.loads((rd / "result.json").read_text())
    assert result_json["trajectory_source"] == "openreward"
    assert result_json["rewards"] == {"reward": 1.0}
    assert result_json["agent_name"] == "openreward"
    assert result_json["source"]["provider"] == "openreward"
    assert result_json["source"]["runner"] == "openreward"

    # verifier/verify_result.json — canonical Reward-plane artifact.
    verify_result = json.loads((rd / "verifier" / "verify_result.json").read_text())
    assert verify_result["reward"] == 1.0
    assert verify_result["space"] == "output"
    assert verify_result["granularity"] == "terminal"
    assert verify_result["error"] is None

    # trainer/verifiers.jsonl — trainer seam, one record.
    lines = (rd / "trainer" / "verifiers.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reward"] == 1.0
    assert record["info"]["environment"] == "openreward"
    assert record["info"]["reward_valid"] is True

    # Supporting contract files exist.
    for rel in ("config.json", "timing.json", "prompts.json", "rewards.jsonl"):
        assert (rd / rel).is_file(), rel
    traj = (rd / "trajectory" / "acp_trajectory.jsonl").read_text().strip().splitlines()
    # user prompt + tool_call + agent_message
    types = [json.loads(line)["type"] for line in traj]
    assert "user_message" in types
    assert "tool_call" in types


def test_missing_owner_raises_before_session(tmp_path):
    # A bare openreward name (no owner) is rejected by the platform — the
    # driver must raise before opening any session. We build the ref directly
    # to bypass the parse layer.
    ref = HostedEnvRef(provider="openreward", owner=None, name="KellyBench")
    opened = {"called": False}

    @contextmanager
    def factory(config, split, index):
        opened["called"] = True
        yield FakeSession(tools=[], outputs=[])

    config = HostedEnvRunConfig(source_env=ref, model="m", jobs_dir=tmp_path)
    with pytest.raises(HostedEnvError, match="owner/namespace"):
        run_hosted_env_openreward(config, session_factory=factory)
    assert opened["called"] is False


def test_wrong_provider_raises(tmp_path):
    config = HostedEnvRunConfig(
        source_env=HostedEnvRef.parse("primeintellect/general-agent", version="0.1.1"),
        model="m",
        jobs_dir=tmp_path,
    )
    with pytest.raises(HostedEnvError, match="expected 'openreward'"):
        run_hosted_env_openreward(config)


def test_scripted_policy_stops_cleanly_when_tools_exhausted(tmp_path):
    # Env never reports finished, but the scripted policy walks each tool once
    # and then returns None (nothing left to do). The loop ends cleanly with no
    # error before the step budget is hit, and the last reward is surfaced.
    session = FakeSession(
        tools=[FakeToolSpec(name="loop_a"), FakeToolSpec(name="loop_b")],
        outputs=[FakeToolOutput(blocks=[FakeBlock("x")], reward=0.0, finished=False)],
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=ScriptedPolicy(),
        session_factory=_factory(session),
        max_steps=10,
    )
    assert result.error is None
    assert result.total_tool_calls == 2
    assert result.reward == 0.0


def test_max_steps_error_when_policy_keeps_acting(tmp_path):
    # A policy that always acts + an env that never finishes -> step budget
    # exhausted -> error recorded.
    class AlwaysAct:
        def act(self, prompt_text, tools, last_output, step):
            return "loop", {}

    session = FakeSession(
        tools=[FakeToolSpec(name="loop")],
        outputs=[FakeToolOutput(blocks=[FakeBlock("x")], reward=0.0, finished=False)],
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=AlwaysAct(),
        session_factory=_factory(session),
        max_steps=3,
    )
    assert result.total_tool_calls == 3
    assert result.error is not None
    assert "did not report finished" in result.error
    # result.json carries the error and reward is the last observed value.
    result_json = json.loads((result.run_dir / "result.json").read_text())
    assert result_json["error"] is not None


def test_no_reward_yields_reward_none_and_valid_artifacts(tmp_path):
    # Env finishes but reports no reward -> result.reward None, verify_result
    # falls back to reward 0.0 with an error (the "nobody scored" path).
    session = FakeSession(
        tools=[FakeToolSpec(name="submit")],
        outputs=[FakeToolOutput(blocks=[FakeBlock("ok")], reward=None, finished=True)],
    )
    result = run_hosted_env_openreward(
        _config(tmp_path),
        policy=ScriptedPolicy(),
        session_factory=_factory(session),
    )
    assert result.reward is None
    rd = result.run_dir
    result_json = json.loads((rd / "result.json").read_text())
    assert result_json["rewards"] is None
    # verify_result still written, reward 0.0 + error populated.
    verify_result = json.loads((rd / "verifier" / "verify_result.json").read_text())
    assert verify_result["reward"] == 0.0
    assert verify_result["error"]


def test_prompt_to_text_handles_blocks_and_images():
    assert _prompt_to_text("plain") == "plain"
    assert _prompt_to_text([FakeBlock("a"), FakeBlock("b")]) == "a\nb"
    assert _prompt_to_text([{"type": "image", "data": "..."}]) == "[image]"

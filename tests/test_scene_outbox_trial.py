"""Tests for outbox-based inter-role messaging in Trial._run_scene().

Verifies that when bf.run(TrialConfig) executes a multi-role Scene,
outbox files written by one role are read and injected into the next
role's prompt — bridging the _scene.py outbox convention with the
Trial lifecycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.trial import Role, Scene, Trial, TrialConfig, Turn


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnv:
    """Minimal sandbox mock that tracks outbox files."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._exec_log: list[str] = []

    async def exec(self, cmd: str, **kwargs) -> FakeExecResult:
        self._exec_log.append(cmd)
        if "rm -rf /app/.outbox" in cmd:
            self._files = {k: v for k, v in self._files.items()
                           if not k.startswith("/app/.outbox/")}
            return FakeExecResult()
        if "ls /app/.outbox/" in cmd:
            files = [f for f in self._files if f.startswith("/app/.outbox/")]
            return FakeExecResult(stdout="\n".join(files))
        if cmd.startswith("cat "):
            path = cmd.split(" ", 1)[1]
            return FakeExecResult(stdout=self._files.get(path, "{}"))
        if cmd.startswith("rm -f "):
            path = cmd.split()[-1]
            self._files.pop(path, None)
            return FakeExecResult()
        return FakeExecResult()

    def stage_outbox(self, recipient: str, content: str) -> None:
        self._files[f"/app/.outbox/{recipient}.json"] = json.dumps(
            {"to": recipient, "content": content}
        )


def _make_trial(scene: Scene) -> Trial:
    config = TrialConfig(
        task_path=Path("tasks/fake"),
        scenes=[scene],
        environment="docker",
    )
    trial = Trial(config)
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
            Turn("reviewer", "Review the code. Write feedback to /app/.outbox/coder.json"),
            Turn("coder", "Read feedback and fix issues."),
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


async def test_outbox_setup_for_multi_role(coder_reviewer_scene: Scene) -> None:
    """Multi-role scenes set up /app/.outbox with correct ownership."""
    trial = _make_trial(coder_reviewer_scene)
    prompts_received: list[str] = []

    async def fake_execute(prompts=None):
        prompts_received.extend(prompts or [])
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(coder_reviewer_scene)

    outbox_setup = [c for c in trial._env._exec_log if "mkdir -p /app/.outbox" in c]
    assert len(outbox_setup) == 1
    assert "chown agent:agent /app/.outbox" in outbox_setup[0]


async def test_no_outbox_setup_for_single_role(self_review_scene: Scene) -> None:
    """Single-role scenes skip outbox setup (no inter-role messaging needed)."""
    trial = _make_trial(self_review_scene)

    async def fake_execute(prompts=None):
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(self_review_scene)

    outbox_cmds = [c for c in trial._env._exec_log if "outbox" in c]
    assert len(outbox_cmds) == 0


async def test_outbox_messages_injected_into_prompt(coder_reviewer_scene: Scene) -> None:
    """Outbox messages from coder are injected into reviewer's prompt."""
    trial = _make_trial(coder_reviewer_scene)
    prompts_received: list[tuple[str, list[str]]] = []
    call_count = 0

    async def fake_execute(prompts=None):
        nonlocal call_count
        # Track which role got which prompt
        role = coder_reviewer_scene.turns[call_count].role
        prompts_received.append((role, prompts or []))
        # Coder writes to reviewer outbox on first turn
        if call_count == 0:
            trial._env.stage_outbox("reviewer", "Please review my regex implementation")
        # Reviewer writes feedback to coder outbox on second turn
        elif call_count == 1:
            trial._env.stage_outbox("coder", "Edge case: empty string input not handled")
        call_count += 1
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(coder_reviewer_scene)

    assert len(prompts_received) == 3

    # Turn 0: coder gets base prompt (no messages yet)
    assert prompts_received[0][0] == "coder"
    assert "Messages from other agents" not in prompts_received[0][1][0]

    # Turn 1: reviewer gets its prompt + coder's outbox message
    assert prompts_received[1][0] == "reviewer"
    assert "Please review my regex implementation" in prompts_received[1][1][0]
    assert "From coder" in prompts_received[1][1][0]

    # Turn 2: coder gets its prompt + reviewer's feedback
    assert prompts_received[2][0] == "coder"
    assert "Edge case: empty string input not handled" in prompts_received[2][1][0]
    assert "From reviewer" in prompts_received[2][1][0]


async def test_outbox_files_cleared_after_read(coder_reviewer_scene: Scene) -> None:
    """Outbox files are removed after reading so they don't repeat."""
    trial = _make_trial(coder_reviewer_scene)
    call_count = 0

    async def fake_execute(prompts=None):
        nonlocal call_count
        if call_count == 0:
            trial._env.stage_outbox("reviewer", "msg1")
        call_count += 1
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(coder_reviewer_scene)

    remaining = [f for f in trial._env._files if f.startswith("/app/.outbox/")]
    assert len(remaining) == 0


async def test_outbox_invalid_json_skipped(coder_reviewer_scene: Scene) -> None:
    """Invalid JSON in outbox files is skipped without crashing."""
    trial = _make_trial(coder_reviewer_scene)
    call_count = 0

    async def fake_execute(prompts=None):
        nonlocal call_count
        if call_count == 0:
            trial._env._files["/app/.outbox/reviewer.json"] = "not valid json{{"
        call_count += 1
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    # Should not raise
    await trial._run_scene(coder_reviewer_scene)
    assert call_count == 3


async def test_role_switching_connects_and_disconnects(coder_reviewer_scene: Scene) -> None:
    """Verify connect/disconnect happens on role switches."""
    trial = _make_trial(coder_reviewer_scene)

    async def fake_execute(prompts=None):
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(coder_reviewer_scene)

    # 3 turns: coder, reviewer, coder → 2 connect_as calls for role switches + 1 initial
    # Initial connect for coder, then disconnect+connect for reviewer, then disconnect+connect for coder
    assert trial.connect_as.call_count == 3
    # disconnect after coder->reviewer, after reviewer->coder, and final disconnect
    assert trial.disconnect.call_count == 3


async def test_empty_outbox_no_injection() -> None:
    """When no outbox files exist, prompt is used as-is."""
    scene = Scene(
        name="quiet",
        roles=[
            Role("a", "gemini", "flash"),
            Role("b", "gemini", "flash"),
        ],
        turns=[Turn("a", "do stuff"), Turn("b", "also do stuff")],
    )
    trial = _make_trial(scene)
    prompts_received: list[str] = []

    async def fake_execute(prompts=None):
        prompts_received.extend(prompts or [])
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(scene)

    assert prompts_received[0] == "do stuff"
    assert prompts_received[1] == "also do stuff"
    assert all("Messages from other agents" not in p for p in prompts_received)


async def test_scene_messages_persisted(
    coder_reviewer_scene: Scene, tmp_path: Path
) -> None:
    """Inter-role messages are saved to scene_messages.jsonl in trial_dir."""
    trial = _make_trial(coder_reviewer_scene)
    trial._trial_dir = tmp_path
    call_count = 0

    async def fake_execute(prompts=None):
        nonlocal call_count
        if call_count == 0:
            trial._env.stage_outbox("reviewer", "Please review my code")
        elif call_count == 1:
            trial._env.stage_outbox("coder", "Found a bug on line 5")
        call_count += 1
        return [], 0

    trial.connect_as = AsyncMock()
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(coder_reviewer_scene)

    msg_path = tmp_path / "scene_messages.jsonl"
    assert msg_path.exists()
    lines = [json.loads(ln) for ln in msg_path.read_text().strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["sender"] == "coder"
    assert lines[0]["recipient"] == "reviewer"
    assert lines[0]["content"] == "Please review my code"
    assert lines[1]["sender"] == "reviewer"
    assert lines[1]["recipient"] == "coder"
    assert lines[1]["content"] == "Found a bug on line 5"
    assert lines[0]["scene"] == "code-review"


async def test_heterogeneous_agent_install(coder_reviewer_scene: Scene) -> None:
    """connect_as installs non-primary agents."""
    scene = Scene(
        name="hetero",
        roles=[
            Role("coder", "gemini", "flash"),
            Role("reviewer", "claude-agent-acp", "haiku"),
        ],
        turns=[Turn("coder"), Turn("reviewer", "Review.")],
    )
    config = TrialConfig(
        task_path=Path("tasks/fake"),
        scenes=[scene],
        environment="docker",
        agent="gemini",
    )
    trial = Trial(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task"]

    installed_agents: list[str] = []
    original_connect_as = Trial.connect_as

    async def tracking_connect_as(self_inner, role):
        if role.agent != config.primary_agent:
            installed_agents.append(role.agent)

    async def fake_execute(prompts=None):
        return [], 0

    trial.connect_as = lambda role: tracking_connect_as(trial, role)
    trial.disconnect = AsyncMock()
    trial.execute = fake_execute

    await trial._run_scene(scene)

    assert "claude-agent-acp" in installed_agents

"""Tests for canonical scene types and _scene.py runtime."""

import asyncio
import json
from pathlib import Path

import pytest

from benchflow._scene import MailboxTransport, Message, SceneRuntime
from benchflow.rollouts import RolloutConfig, Role, Scene, Turn


@pytest.fixture
def two_roles() -> dict[str, Role]:
    return {
        "coder": Role(
            name="coder",
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            instruction="You are a coder. Write code and notify the reviewer.",
        ),
        "reviewer": Role(
            name="reviewer",
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            instruction="You are a reviewer. Review code from the coder.",
        ),
    }


def test_scene_requires_two_roles() -> None:
    r = Role(name="solo", agent="x", model="y", instruction="z")
    with pytest.raises(ValueError, match="exactly 2 roles"):
        SceneRuntime(roles={"solo": r})


def test_scene_init(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles, max_rounds=5)
    assert scene.role_names == ["coder", "reviewer"]
    assert scene.max_rounds == 5
    assert not scene.is_done
    assert scene.trajectory == []


def test_next_active_role(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles)
    assert scene.next_active_role("coder") == "reviewer"
    assert scene.next_active_role("reviewer") == "coder"


async def test_send_message(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles, max_rounds=10)
    result = await scene.send_message("coder", "reviewer", "please review")
    assert "delivered" in result
    assert len(scene.trajectory) == 1
    assert scene.trajectory[0].sender == "coder"
    assert scene.trajectory[0].recipient == "reviewer"
    assert scene.trajectory[0].content == "please review"
    assert scene.trajectory[0].turn == 1
    assert scene.trajectory[0].kind == "direct"


async def test_send_to_unknown_recipient(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles)
    result = await scene.send_message("coder", "nobody", "hi")
    assert "Error" in result
    assert len(scene.trajectory) == 0


async def test_round_counting(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles, max_rounds=3)
    await scene.send_message("coder", "reviewer", "msg1")
    await scene.send_message("reviewer", "coder", "msg2")
    assert not scene.is_done
    await scene.send_message("coder", "reviewer", "msg3")
    assert scene.is_done


async def test_end_scene(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles)
    assert not scene.is_done
    await scene.end_scene("reviewer", reward=1.0)
    assert scene.is_done


async def test_mailbox_transport() -> None:
    transport = MailboxTransport()
    msg = Message(id="abc", sender="a", recipient="b", content="hello", turn=1)
    await transport.send(msg)
    pending = await transport.list_pending("b")
    assert len(pending) == 1
    assert pending[0].content == "hello"

    received = await transport.receive("b")
    assert received is not None
    assert received.content == "hello"

    empty = await transport.receive("b")
    assert empty is None


async def test_build_prompt_for_role(two_roles: dict[str, Role]) -> None:
    scene = SceneRuntime(roles=two_roles)
    inbox = [
        Message(
            id="x", sender="reviewer", recipient="coder", content="looks good", turn=1
        )
    ]
    prompt = scene.build_prompt_for_role(two_roles["coder"], inbox)
    assert "You are a coder" in prompt
    assert "looks good" in prompt
    assert "reviewer" in prompt
    assert "send_message" in prompt


class FakeEnv:
    """Mock env with exec() that simulates outbox file writes."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._exec_log: list[str] = []

    async def exec(self, cmd: str, **kwargs) -> "FakeExecResult":
        self._exec_log.append(cmd)
        if cmd.startswith("rm -rf /tmp/outbox") or cmd.startswith("mkdir -p"):
            self._files.clear()
            return FakeExecResult("", "", 0)
        if "ls /app/.outbox/" in cmd:
            files = [f for f in self._files if f.startswith("/app/.outbox/")]
            return FakeExecResult("\n".join(files), "", 0)
        if cmd.startswith("cat "):
            path = cmd.split(" ", 1)[1]
            return FakeExecResult(self._files.get(path, "{}"), "", 0)
        if cmd.startswith("rm -f "):
            path = cmd.split()[-1]
            self._files.pop(path, None)
            return FakeExecResult("", "", 0)
        return FakeExecResult("", "", 0)

    def stage_outbox(self, recipient: str, content: str) -> None:
        self._files[f"/app/.outbox/{recipient}.json"] = json.dumps(
            {"to": recipient, "content": content}
        )


class FakeExecResult:
    def __init__(self, stdout: str, stderr: str, return_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


async def test_scene_run_two_rounds(two_roles: dict[str, Role]) -> None:
    env = FakeEnv()
    scene = SceneRuntime(roles=two_roles, max_rounds=4)
    call_count = 0

    async def mock_runner(e, role, prompt):
        nonlocal call_count
        call_count += 1
        if role.name == "coder":
            env.stage_outbox("reviewer", "please review my code")
        elif role.name == "reviewer":
            env.stage_outbox("coder", "looks good, approved")

    trajectory = await scene.run(env, mock_runner)
    assert call_count >= 2
    assert len(trajectory) >= 2
    assert trajectory[0].sender == "coder"
    assert trajectory[0].recipient == "reviewer"
    assert trajectory[1].sender == "reviewer"
    assert trajectory[1].recipient == "coder"


async def test_scene_run_stops_when_no_message(two_roles: dict[str, Role]) -> None:
    env = FakeEnv()
    scene = SceneRuntime(roles=two_roles, max_rounds=10)

    async def mock_runner(e, role, prompt):
        if role.name == "coder":
            env.stage_outbox("reviewer", "check this")

    trajectory = await scene.run(env, mock_runner)
    assert len(trajectory) == 1
    assert trajectory[0].sender == "coder"


def test_save_trajectory(two_roles: dict[str, Role], tmp_path: Path) -> None:
    scene = SceneRuntime(roles=two_roles)
    asyncio.get_event_loop().run_until_complete(
        scene.send_message("coder", "reviewer", "check this")
    )
    asyncio.get_event_loop().run_until_complete(
        scene.send_message("reviewer", "coder", "approved")
    )
    out = tmp_path / "scene_trajectory.jsonl"
    scene.save_trajectory(out)
    lines = [json.loads(ln) for ln in out.read_text().strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["sender"] == "coder"
    assert lines[1]["sender"] == "reviewer"
    assert lines[0]["turn"] == 1
    assert lines[1]["turn"] == 2


def test_canonical_role_budget_fields() -> None:
    role = Role(
        name="solver",
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        timeout_sec=900,
        idle_timeout_sec=120,
        capabilities=["native-loop"],
    )
    assert role.timeout_sec == 900
    assert role.idle_timeout_sec == 120
    assert role.capabilities == ["native-loop"]


def test_canonical_scene_single_and_parallel_group() -> None:
    scene = Scene.single(
        agent="gemini",
        model="gemini-3.1-flash-lite-preview",
        prompts=[None, "Review your solution."],
        timeout_sec=300,
        idle_timeout_sec=60,
    )
    scene.parallel_group = "pass-k"

    assert scene.roles == [
        Role(
            name="agent",
            agent="gemini",
            model="gemini-3.1-flash-lite-preview",
            timeout_sec=300,
            idle_timeout_sec=60,
        )
    ]
    assert scene.turns == [
        Turn(role="agent", prompt=None),
        Turn(role="agent", prompt="Review your solution."),
    ]
    assert scene.parallel_group == "pass-k"


def test_rollout_config_from_single_uses_canonical_scene() -> None:
    config = RolloutConfig.from_single(
        task_path=Path("tasks/example"),
        agent="gemini",
        model="gemini-3.1-flash-lite-preview",
        prompts=[None],
    )

    assert config.primary_agent == "gemini"
    assert config.primary_model == "gemini-3.1-flash-lite-preview"
    assert config.effective_scenes == [
        Scene.single(
            agent="gemini",
            model="gemini-3.1-flash-lite-preview",
            prompts=[None],
        )
    ]

"""Tests for _scene.py — multi-agent scene runtime."""
import asyncio
import json
from pathlib import Path

import pytest

from benchflow._scene import MailboxTransport, Message, Role, Scene


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
        Scene(roles={"solo": r})


def test_scene_init(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles, max_rounds=5)
    assert scene.role_names == ["coder", "reviewer"]
    assert scene.max_rounds == 5
    assert not scene.is_done
    assert scene.trajectory == []


def test_next_active_role(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles)
    assert scene.next_active_role("coder") == "reviewer"
    assert scene.next_active_role("reviewer") == "coder"


async def test_send_message(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles, max_rounds=10)
    result = await scene.send_message("coder", "reviewer", "please review")
    assert "delivered" in result
    assert len(scene.trajectory) == 1
    assert scene.trajectory[0].sender == "coder"
    assert scene.trajectory[0].recipient == "reviewer"
    assert scene.trajectory[0].content == "please review"
    assert scene.trajectory[0].turn == 1
    assert scene.trajectory[0].kind == "direct"


async def test_send_to_unknown_recipient(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles)
    result = await scene.send_message("coder", "nobody", "hi")
    assert "Error" in result
    assert len(scene.trajectory) == 0


async def test_round_counting(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles, max_rounds=3)
    await scene.send_message("coder", "reviewer", "msg1")
    await scene.send_message("reviewer", "coder", "msg2")
    assert not scene.is_done
    await scene.send_message("coder", "reviewer", "msg3")
    assert scene.is_done


async def test_end_scene(two_roles: dict[str, Role]) -> None:
    scene = Scene(roles=two_roles)
    assert not scene.is_done
    await scene.end_scene("reviewer", reward=1.0)
    assert scene.is_done


async def test_mailbox_transport() -> None:
    transport = MailboxTransport()
    msg = Message(
        id="abc", sender="a", recipient="b", content="hello", turn=1
    )
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
    scene = Scene(roles=two_roles)
    inbox = [
        Message(id="x", sender="reviewer", recipient="coder", content="looks good", turn=1)
    ]
    prompt = scene.build_prompt_for_role(two_roles["coder"], inbox)
    assert "You are a coder" in prompt
    assert "looks good" in prompt
    assert "reviewer" in prompt
    assert "send_message" in prompt


def test_save_trajectory(two_roles: dict[str, Role], tmp_path: Path) -> None:
    scene = Scene(roles=two_roles)
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

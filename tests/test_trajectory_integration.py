"""Integration tests for the full ACP trajectory pipeline.

Verifies the end-to-end flow: execute_prompts → session recording →
_capture_session_trajectory → JSONL serialization. Uses real mock ACP
agents over stdio, not mocked session objects.

Covers issues reported in skillsbench#745:
  - user messages must appear in trajectory
  - agent_message/agent_thought must not be cumulative across prompts
  - chronological interleaving must be preserved
"""

import json
import sys
from pathlib import Path

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.runtime import execute_prompts
from benchflow.acp.transport import StdioTransport
from benchflow.trajectories._capture import _capture_session_trajectory

MOCK_AGENT = str(Path(__file__).parent / "fixtures" / "mock_acp_agent.py")
MOCK_AGENT_MULTI = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_multi_turn.py"
)


class TestExecutePromptsTrajectory:
    """Integration: execute_prompts with a real mock ACP agent."""

    @pytest.mark.asyncio
    async def test_single_prompt_has_user_message(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, n_tools = await execute_prompts(
                client, session, ["Solve the task"], timeout=10
            )

            types = [e["type"] for e in trajectory]
            assert "user_message" in types
            assert trajectory[0]["type"] == "user_message"
            assert trajectory[0]["text"] == "Solve the task"
            assert trajectory[0]["ts"] > 0  # arrival-stamped client-side
            assert n_tools == 1
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_single_prompt_event_order(self) -> None:
        """thought → tool_call → message, with user_message first."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, _ = await execute_prompts(
                client, session, ["Do something"], timeout=10
            )

            types = [e["type"] for e in trajectory]
            assert types == [
                "user_message",
                "agent_thought",
                "tool_call",
                "agent_message",
            ]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multi_prompt_not_cumulative(self) -> None:
        """Two prompts: each gets its own user_message and per-turn agent text."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, n_tools = await execute_prompts(
                client, session, ["Prompt A", "Prompt B"], timeout=10
            )

            assert n_tools == 2

            types = [e["type"] for e in trajectory]
            assert types == [
                "user_message",
                "agent_thought",
                "tool_call",
                "agent_message",
                "user_message",
                "agent_thought",
                "tool_call",
                "agent_message",
            ]

            # User messages
            user_msgs = [e for e in trajectory if e["type"] == "user_message"]
            assert user_msgs[0]["text"] == "Prompt A"
            assert user_msgs[1]["text"] == "Prompt B"

            # Agent text must NOT be cumulative
            agent_msgs = [e for e in trajectory if e["type"] == "agent_message"]
            assert agent_msgs[0]["text"] == "response-turn-1"
            assert agent_msgs[1]["text"] == "response-turn-2"
            assert "response-turn-1" not in agent_msgs[1]["text"]

            thoughts = [e for e in trajectory if e["type"] == "agent_thought"]
            assert thoughts[0]["text"] == "thinking-turn-1"
            assert thoughts[1]["text"] == "thinking-turn-2"
            assert "thinking-turn-1" not in thoughts[1]["text"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_capture_idempotent_after_execute(self) -> None:
        """Calling _capture_session_trajectory again returns same result."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, _ = await execute_prompts(client, session, ["Go"], timeout=10)

            again = _capture_session_trajectory(session)
            assert trajectory == again
        finally:
            await client.close()


class TestTrajectoryJsonlSerialization:
    """Verify trajectory events survive JSONL roundtrip (the sdk.py write path)."""

    @pytest.mark.asyncio
    async def test_jsonl_roundtrip(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, _ = await execute_prompts(
                client, session, ["Test prompt"], timeout=10
            )

            # Serialize (same as sdk.py:299)
            jsonl = "\n".join(json.dumps(e, default=str) for e in trajectory)

            # Deserialize
            restored = [json.loads(line) for line in jsonl.splitlines()]

            assert len(restored) == len(trajectory)
            for original, parsed in zip(trajectory, restored, strict=True):
                assert original["type"] == parsed["type"]
                if "text" in original:
                    assert original["text"] == parsed["text"]

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_user_message_in_jsonl(self) -> None:
        """user_message events must serialize with type and text fields."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, _ = await execute_prompts(
                client, session, ["Hello"], timeout=10
            )

            jsonl_lines = [json.dumps(e, default=str) for e in trajectory]
            first_event = json.loads(jsonl_lines[0])
            assert first_event["type"] == "user_message"
            assert first_event["text"] == "Hello"
            assert first_event["ts"] > 0  # arrival-stamped client-side
        finally:
            await client.close()

    def test_all_event_types_serializable(self) -> None:
        """Every known event type can be serialized and deserialized."""
        events = [
            {"type": "user_message", "text": "prompt"},
            {
                "type": "tool_call",
                "tool_call_id": "tc_1",
                "kind": "bash",
                "title": "ls",
                "status": "completed",
                "content": [{"type": "text", "text": "file.txt"}],
            },
            {"type": "agent_message", "text": "response"},
            {"type": "agent_thought", "text": "thinking"},
        ]
        for event in events:
            serialized = json.dumps(event, default=str)
            restored = json.loads(serialized)
            assert restored == event


class TestViewerCompatibility:
    """Verify viewer renders user_message events without duplication."""

    def test_viewer_renders_user_message(self) -> None:
        from benchflow.trajectories.viewer import _render_acp_trajectory

        events = [
            {"type": "user_message", "text": "Solve it"},
            {"type": "tool_call", "kind": "bash", "title": "ls", "status": "completed"},
            {"type": "agent_message", "text": "Done."},
        ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp)
            traj_dir = trial_dir / "trajectory"
            traj_dir.mkdir()
            (traj_dir / "acp_trajectory.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )

            html = _render_acp_trajectory(
                trial_dir, traj_dir / "acp_trajectory.jsonl", prompts=["Solve it"]
            )

            # user_message in trajectory → should render inline, NOT duplicate at top
            assert html.count("Solve it") == 1
            assert "PROMPT 1" in html

    def test_viewer_legacy_prompts_header(self) -> None:
        """Old trajectory without user_message: prompts shown at top from prompts.json."""
        from benchflow.trajectories.viewer import _render_acp_trajectory

        events = [
            {"type": "tool_call", "kind": "bash", "title": "ls", "status": "completed"},
            {"type": "agent_message", "text": "Done."},
        ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp)
            traj_dir = trial_dir / "trajectory"
            traj_dir.mkdir()
            (traj_dir / "acp_trajectory.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )

            html = _render_acp_trajectory(
                trial_dir, traj_dir / "acp_trajectory.jsonl", prompts=["Old prompt"]
            )

            assert "Old prompt" in html
            assert "PROMPT 1" in html

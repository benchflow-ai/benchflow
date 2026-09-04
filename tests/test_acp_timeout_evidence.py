"""ACP timeout cancellation and terminal-evidence regression coverage."""

from __future__ import annotations

import asyncio

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import Transport
from benchflow.acp.types import StopReason, ToolCallStatus

_TIMEOUT_USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "total_tokens": 14,
    "cached_read_tokens": None,
    "cached_write_tokens": None,
    "thought_tokens": None,
}


class _CooperativeCancellationTransport(Transport):
    """Minimal ACP peer that returns terminal evidence after session/cancel."""

    def __init__(
        self,
        *,
        emit_pending_tool: bool,
        stall_cancel_after_response: bool = False,
        cancel_failure_response_delay: float | None = None,
    ) -> None:
        self._messages: asyncio.Queue[dict] = asyncio.Queue()
        self._prompt_request_id: int | None = None
        self._emit_pending_tool = emit_pending_tool
        self._stall_cancel_after_response = stall_cancel_after_response
        self._cancel_failure_response_delay = cancel_failure_response_delay
        self.cancel_calls = 0
        self.cancel_send_finished = asyncio.Event()
        self.receive_calls = 0

    async def start(self) -> None:
        pass

    def _queue_update(self, update: dict) -> None:
        self._messages.put_nowait(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "timeout-session", "update": update},
            }
        )

    def _queue_response(self) -> None:
        assert self._prompt_request_id is not None
        self._messages.put_nowait(
            {
                "jsonrpc": "2.0",
                "id": self._prompt_request_id,
                "result": {
                    "stopReason": "cancelled",
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 4,
                        "totalTokens": 14,
                    },
                },
            }
        )

    async def send(self, message: dict) -> None:
        if message.get("method") == "session/prompt":
            self._prompt_request_id = message["id"]
            if self._emit_pending_tool:
                self._queue_update(
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tool-1",
                        "title": "work before timeout",
                        "kind": "bash",
                    }
                )
            return

        if message.get("method") == "session/cancel":
            self.cancel_calls += 1
            try:
                assert message["params"] == {"sessionId": "timeout-session"}
                assert self._prompt_request_id is not None
                if self._emit_pending_tool:
                    self._queue_update(
                        {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "tool-1",
                            "status": "cancelled",
                        }
                    )
                if self._cancel_failure_response_delay is not None:
                    asyncio.get_running_loop().call_later(
                        self._cancel_failure_response_delay, self._queue_response
                    )
                    raise ConnectionError("cancel send failed")
                self._queue_response()
                if self._stall_cancel_after_response:
                    await asyncio.Future()
            finally:
                self.cancel_send_finished.set()

    async def receive(self) -> dict:
        self.receive_calls += 1
        return await self._messages.get()

    async def close(self) -> None:
        pass


class TestACPTimeoutCancellation:
    @pytest.mark.parametrize(
        "transport_kwargs",
        [
            {"stall_cancel_after_response": True},
            {"cancel_failure_response_delay": 0.01},
        ],
        ids=["stalled-cancel-send", "failed-cancel-send"],
    )
    @pytest.mark.asyncio
    async def test_wall_timeout_requests_cooperative_acp_cancellation(
        self, transport_kwargs: dict
    ) -> None:
        """Guards issue #933 and PR #1080's evidence fix after PR #1051."""
        from benchflow.acp.runtime import AgentPromptTimeoutError, execute_prompts

        transport = _CooperativeCancellationTransport(
            emit_pending_tool=True,
            **transport_kwargs,
        )
        client = ACPClient(transport)
        session = ACPSession("timeout-session")
        client._session = session

        with pytest.raises(AgentPromptTimeoutError) as exc_info:
            await asyncio.wait_for(
                execute_prompts(
                    client,
                    session,
                    ["solve"],
                    timeout=0.05,  # type: ignore[arg-type]
                    idle_timeout=None,
                ),
                timeout=10,
            )

        await asyncio.sleep(0)
        assert transport.cancel_calls == 1
        assert transport.cancel_send_finished.is_set()
        assert transport.receive_calls == 3
        assert session.latest_usage_totals() == _TIMEOUT_USAGE
        assert session.stop_reason == StopReason.CANCELLED
        tool_event = next(
            event for event in exc_info.value.trajectory if event["type"] == "tool_call"
        )
        assert tool_event["status"] == ToolCallStatus.CANCELLED.value
        assert exc_info.value.terminal_trajectory_complete is True

    @pytest.mark.asyncio
    async def test_idle_timeout_requests_cooperative_acp_cancellation(self) -> None:
        """Guards PR #1080 against idle-timeout usage loss shown by PR #1051."""
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        transport = _CooperativeCancellationTransport(emit_pending_tool=False)
        client = ACPClient(transport)
        session = ACPSession("timeout-session")
        client._session = session

        with pytest.raises(IdleTimeoutError):
            await asyncio.wait_for(
                execute_prompts(
                    client,
                    session,
                    ["solve"],
                    timeout=30,
                    idle_timeout=1,
                ),
                timeout=10,
            )

        await asyncio.sleep(0)
        assert transport.cancel_calls == 1
        assert transport.receive_calls == 1
        assert session.latest_usage_totals() == _TIMEOUT_USAGE
        assert session.stop_reason == StopReason.CANCELLED

    @pytest.mark.asyncio
    async def test_idle_timeout_bounds_noncooperative_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards PR #1080: stuck prompt and cancel tasks cannot wedge timeout."""
        from benchflow.acp import timeout_cleanup
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        class StubbornPromptClient:
            def __init__(self) -> None:
                self.release = asyncio.Event()
                self.task: asyncio.Task | None = None
                self.cancel_release = asyncio.Event()
                self.cancel_task: asyncio.Task | None = None
                self.cancel_finished = asyncio.Event()
                self.cancel_calls = 0

            async def cancel(self) -> None:
                self.cancel_calls += 1
                self.cancel_task = asyncio.current_task()
                try:
                    await self.cancel_release.wait()
                finally:
                    self.cancel_finished.set()

            async def prompt(self, _prompt: str):
                self.task = asyncio.current_task()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    await self.release.wait()
                    raise

        client = StubbornPromptClient()
        session = ACPSession("idle-session")
        monkeypatch.setattr(timeout_cleanup, "PROMPT_TIMEOUT_CLEANUP_TOTAL_SEC", 0.1)
        monkeypatch.setattr(timeout_cleanup, "PROMPT_CANCEL_DRAIN_TIMEOUT_SEC", 0.05)

        try:
            with pytest.raises(IdleTimeoutError, match="Agent idle for 1s"):
                await asyncio.wait_for(
                    execute_prompts(
                        client,  # type: ignore[arg-type]
                        session,
                        ["solve"],
                        timeout=30,
                        idle_timeout=1,
                    ),
                    timeout=5.0,
                )
            assert client.cancel_calls == 1
            assert client.cancel_finished.is_set()
            assert client.cancel_task is not None
            assert client.cancel_task.cancelled()
            assert client.task is not None
            assert not client.task.done()
        finally:
            client.cancel_release.set()
            client.release.set()
            tasks = [
                task for task in (client.cancel_task, client.task) if task is not None
            ]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

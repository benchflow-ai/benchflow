from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from benchflow.agents.a2a import (
    A2AClientParticipantAdapter,
    A2AParticipantRequest,
)


@dataclass
class FakeState:
    value: str


@dataclass
class FakeStatus:
    state: FakeState


@dataclass
class FakeArtifact:
    name: str

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return {"name": self.name, "mode": mode}


@dataclass
class FakeTask:
    state: str
    artifacts: list[FakeArtifact]

    @property
    def status(self) -> FakeStatus:
        return FakeStatus(FakeState(self.state))

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return {"status": {"state": self.state}, "mode": mode}


class FakeAdapter(A2AClientParticipantAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.seen_request: A2AParticipantRequest | None = None
        self.seen_message_id: str | None = None

    async def _send_message(
        self,
        request: A2AParticipantRequest,
        *,
        message_id: str,
    ) -> AsyncIterator[object]:
        self.seen_request = request
        self.seen_message_id = message_id
        yield Message(
            kind="message",
            role=Role.agent,
            parts=[Part(TextPart(kind="text", text="participant finished"))],
            message_id="reply-1",
        )
        yield (FakeTask("completed", [FakeArtifact("answer")]), None)


@pytest.mark.asyncio
async def test_a2a_client_adapter_normalizes_completed_message_flow() -> None:
    adapter = FakeAdapter()
    request = A2AParticipantRequest(
        endpoint_url="http://purple.example/",
        role_name="agent",
        prompt="Solve the visible task.",
        skills_dir="/skills",
    )

    handle = await adapter.start(request)
    result = await adapter.wait(handle)

    assert adapter.seen_request == request
    assert adapter.seen_message_id == handle.task_id
    assert result.status == "completed"
    assert result.done is True
    assert result.final_response is not None
    assert len(result.trajectory) == 2
    assert result.artifacts[0].uri == f"a2a://{handle.task_id}/artifacts/0"


@pytest.mark.asyncio
async def test_a2a_client_adapter_cancel_before_wait_returns_cancelled() -> None:
    adapter = FakeAdapter()
    handle = await adapter.start(
        A2AParticipantRequest(
            endpoint_url="http://purple.example/",
            role_name="agent",
            prompt="Solve the visible task.",
        )
    )

    await adapter.cancel(handle, "outer timeout")
    result = await adapter.wait(handle)

    assert result.status == "cancelled"
    assert result.error_type == "cancelled"
    assert result.trajectory[0].payload["reason"] == "outer timeout"


@pytest.mark.asyncio
async def test_a2a_client_adapter_rejects_unknown_handle() -> None:
    adapter = FakeAdapter()
    handle = await adapter.start(
        A2AParticipantRequest(
            endpoint_url="http://purple.example/",
            role_name="agent",
            prompt="Solve the visible task.",
        )
    )
    await adapter.wait(handle)

    with pytest.raises(KeyError):
        await adapter.wait(handle)


@pytest.mark.asyncio
async def test_a2a_client_adapter_toy_server_smoke() -> None:
    app = _build_toy_a2a_app("http://toy.local/")

    def client_factory(timeout_sec: int) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://toy.local",
            timeout=timeout_sec,
        )

    adapter = A2AClientParticipantAdapter(httpx_client_factory=client_factory)
    handle = await adapter.start(
        A2AParticipantRequest(
            endpoint_url="http://toy.local/",
            role_name="agent",
            prompt="Solve the visible toy task.",
        )
    )

    result = await adapter.wait(handle)

    assert result.status == "completed"
    assert result.done is True
    assert result.final_response is not None
    assert result.artifacts
    assert any(event.kind == "task_update" for event in result.trajectory)


class ToyExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            return
        task = new_task(message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Solving toy task.",
                context_id=context.context_id,
            ),
        )
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text="toy result")),
                Part(root=DataPart(data={"echo": context.get_user_input()})),
            ],
            name="answer",
        )
        await updater.complete()

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def _build_toy_a2a_app(card_url: str) -> Any:
    request_handler = DefaultRequestHandler(
        agent_executor=ToyExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=AgentCard(
            name="Toy Purple Agent",
            description="Toy A2A participant for BenchFlow smoke tests.",
            url=card_url,
            version="0.1.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="toy-solve",
                    name="Toy Solve",
                    description="Return a toy artifact.",
                    tags=["test"],
                )
            ],
        ),
        http_handler=request_handler,
    )
    return server.build()

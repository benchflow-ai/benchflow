"""AgentBeats A2A participant adapter contract.

This module intentionally defines the contract only. Runtime wiring belongs in
the later implementation phase after the SkillsBench green-agent skeleton can
exercise this boundary against AgentBeats-style assessment requests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, TextPart
from a2a.types import Role as A2ARole

A2ATaskStatus = Literal["running", "completed", "failed", "cancelled", "timeout"]
A2AUpdateKind = Literal["task_update", "artifact", "final_response", "error"]


@dataclass(frozen=True)
class A2AParticipantRequest:
    """Visible task payload sent from BenchFlow to a purple A2A endpoint."""

    endpoint_url: str
    role_name: str
    prompt: str
    workspace: str = "/app"
    skills_dir: str | None = None
    timeout_sec: int | None = None
    idle_timeout_sec: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class A2ATaskHandle:
    """Opaque handle for a started participant task."""

    task_id: str
    endpoint_url: str
    role_name: str


@dataclass(frozen=True)
class A2ATrajectoryEvent:
    """One normalized A2A update for BenchFlow trajectory persistence."""

    kind: A2AUpdateKind
    timestamp: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class A2AArtifactRef:
    """Reference to an artifact produced by the participant endpoint."""

    name: str
    uri: str
    media_type: str | None = None
    digest: str | None = None


@dataclass(frozen=True)
class A2AParticipantResult:
    """Terminal normalized result from a purple A2A participant."""

    status: A2ATaskStatus
    trajectory: Sequence[A2ATrajectoryEvent] = field(default_factory=tuple)
    artifacts: Sequence[A2AArtifactRef] = field(default_factory=tuple)
    final_response: Mapping[str, Any] | None = None
    error_type: str | None = None

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed", "cancelled", "timeout"}


class A2AParticipantAdapter(Protocol):
    """Protocol a future BenchFlow A2A participant adapter must satisfy."""

    async def start(self, request: A2AParticipantRequest) -> A2ATaskHandle:
        """Create a fresh participant task for one BenchFlow role turn."""

    async def wait(self, handle: A2ATaskHandle) -> A2AParticipantResult:
        """Wait until the participant task reaches a terminal status."""

    async def cancel(self, handle: A2ATaskHandle, reason: str) -> None:
        """Cancel an in-flight participant task before verifier handoff."""


HTTPXClientFactory = Callable[[int], httpx.AsyncClient]


class A2AClientParticipantAdapter:
    """A2A SDK-backed participant adapter.

    The adapter deliberately does not know about Rollout or verifier semantics.
    It sends one visible prompt to a participant endpoint, records normalized A2A
    events, and returns a terminal participant result for the caller to persist
    and hand to the existing verifier path.
    """

    def __init__(
        self,
        *,
        default_timeout_sec: int = 300,
        streaming: bool = True,
        httpx_client_factory: HTTPXClientFactory | None = None,
    ):
        self.default_timeout_sec = default_timeout_sec
        self.streaming = streaming
        self.httpx_client_factory = httpx_client_factory
        self._pending: dict[str, A2AParticipantRequest] = {}
        self._cancelled: dict[str, str] = {}

    async def start(self, request: A2AParticipantRequest) -> A2ATaskHandle:
        task_id = uuid4().hex
        self._pending[task_id] = request
        return A2ATaskHandle(
            task_id=task_id,
            endpoint_url=request.endpoint_url,
            role_name=request.role_name,
        )

    async def wait(self, handle: A2ATaskHandle) -> A2AParticipantResult:
        if handle.task_id in self._cancelled:
            reason = self._cancelled.pop(handle.task_id)
            self._pending.pop(handle.task_id, None)
            return A2AParticipantResult(
                status="cancelled",
                trajectory=(
                    A2ATrajectoryEvent(
                        kind="error",
                        payload={"error_type": "cancelled", "reason": reason},
                    ),
                ),
                error_type="cancelled",
            )

        request = self._pending.pop(handle.task_id, None)
        if request is None:
            raise KeyError(f"Unknown A2A task handle: {handle.task_id}")

        events = [
            event
            async for event in self._send_message(
                request,
                message_id=handle.task_id,
            )
        ]
        return _participant_result_from_events(handle, events)

    async def cancel(self, handle: A2ATaskHandle, reason: str) -> None:
        self._cancelled[handle.task_id] = reason

    async def _send_message(
        self,
        request: A2AParticipantRequest,
        *,
        message_id: str,
    ) -> AsyncIterator[object]:
        timeout = request.timeout_sec or self.default_timeout_sec
        client_factory = self.httpx_client_factory or (
            lambda timeout_sec: httpx.AsyncClient(timeout=timeout_sec)
        )
        async with client_factory(timeout) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=request.endpoint_url,
            )
            agent_card = await resolver.get_agent_card()
            client = ClientFactory(
                ClientConfig(httpx_client=httpx_client, streaming=self.streaming)
            ).create(agent_card)
            message = Message(
                kind="message",
                role=A2ARole.user,
                parts=[Part(TextPart(kind="text", text=request.prompt))],
                message_id=message_id,
            )
            async for event in client.send_message(message):
                yield event


def _participant_result_from_events(
    handle: A2ATaskHandle,
    events: Sequence[object],
) -> A2AParticipantResult:
    trajectory: list[A2ATrajectoryEvent] = []
    artifacts: list[A2AArtifactRef] = []
    artifact_uris: set[str] = set()
    final_response: Mapping[str, Any] | None = None
    status: A2ATaskStatus = "failed"
    error_type: str | None = None

    for event in events:
        if isinstance(event, Message):
            text = _merge_parts(event.parts)
            trajectory.append(
                A2ATrajectoryEvent(
                    kind="final_response",
                    payload={"message": text},
                )
            )
            final_response = _final_response_from_message(event.parts, text)
            status = "completed"
            continue

        if not isinstance(event, tuple) or not event:
            trajectory.append(
                A2ATrajectoryEvent(
                    kind="error",
                    payload={"error_type": "unknown_event", "repr": repr(event)},
                )
            )
            error_type = error_type or "unknown_event"
            continue

        task = event[0]
        update = event[1] if len(event) > 1 else None
        dumped = _model_dump(task)
        task_status = _task_status(task)
        if task_status:
            status = _normalize_task_status(task_status)
        trajectory.append(
            A2ATrajectoryEvent(
                kind="task_update",
                payload={"task": dumped, "update": _model_dump(update)},
            )
        )
        task_artifacts = getattr(task, "artifacts", None) or []
        for index, artifact in enumerate(task_artifacts):
            name = getattr(artifact, "name", None) or f"artifact-{index}"
            uri = f"a2a://{handle.task_id}/artifacts/{index}"
            if uri in artifact_uris:
                continue
            artifact_uris.add(uri)
            artifacts.append(
                A2AArtifactRef(
                    name=name,
                    uri=uri,
                    media_type=None,
                    digest=None,
                )
            )
        if task_artifacts and final_response is None:
            final_response = {"artifacts": [_model_dump(a) for a in task_artifacts]}
        if status in {"failed", "cancelled", "timeout"}:
            error_type = error_type or f"a2a_{status}"

    return A2AParticipantResult(
        status=status,
        trajectory=tuple(trajectory),
        artifacts=tuple(artifacts),
        final_response=final_response,
        error_type=error_type,
    )


def _normalize_task_status(value: str) -> A2ATaskStatus:
    if value == "completed":
        return "completed"
    if value in {"canceled", "cancelled"}:
        return "cancelled"
    if value == "working":
        return "running"
    if value == "submitted":
        return "running"
    return "failed"


def _task_status(task: object) -> str | None:
    status = getattr(task, "status", None)
    state = getattr(status, "state", None)
    value = getattr(state, "value", None)
    return value if isinstance(value, str) else None


def _merge_parts(parts: Sequence[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        root = part.root
        if isinstance(root, TextPart):
            chunks.append(root.text)
        elif isinstance(root, DataPart):
            chunks.append(json.dumps(root.data, sort_keys=True))
    return "\n".join(chunks)


def _final_response_from_message(
    parts: Sequence[Part],
    text: str,
) -> Mapping[str, Any]:
    data_parts = [part.root.data for part in parts if isinstance(part.root, DataPart)]
    if len(data_parts) == 1 and isinstance(data_parts[0], Mapping):
        return {"message": text, **dict(data_parts[0])}
    if data_parts:
        return {"message": text, "data": data_parts}
    return {"message": text}


def _model_dump(value: object) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return repr(value)


__all__ = [
    "A2AClientParticipantAdapter",
    "A2AArtifactRef",
    "A2AParticipantAdapter",
    "A2AParticipantRequest",
    "A2AParticipantResult",
    "A2ATaskHandle",
    "A2ATaskStatus",
    "A2ATrajectoryEvent",
    "A2AUpdateKind",
]

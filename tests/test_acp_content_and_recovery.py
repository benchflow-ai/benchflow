"""Regression coverage for commit 6429743f (ACP content and recovery support)."""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.transport import Transport
from benchflow.acp.types import ACP_PROTOCOL_VERSION, ImageContent


class _RecordingTransport(Transport):
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:
        pass

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any]:
        raise RuntimeError("not used")

    async def close(self) -> None:
        pass


async def _client_with_responses(
    capabilities: dict[str, Any], responses: list[dict[str, Any]]
) -> tuple[ACPClient, _RecordingTransport]:
    transport = _RecordingTransport()
    client = ACPClient(transport)
    queued = iter(
        [
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "agentCapabilities": capabilities,
                "agentInfo": {"name": "fake", "version": "1"},
            },
            {"sessionId": "new-1"},
            *responses,
        ]
    )

    async def _fake_read(_request_id: int) -> dict[str, Any]:
        return next(queued)

    client._read_until_response = _fake_read  # type: ignore[method-assign]
    await client.initialize()
    await client.session_new()
    return client, transport


@pytest.mark.asyncio
async def test_prompt_accepts_text_and_sdk_image_content() -> None:
    client, transport = await _client_with_responses(
        {"promptCapabilities": {"image": True}}, [{"stopReason": "end_turn"}]
    )
    image = ImageContent(
        type="image", data="iVBORw0KGgo=", mime_type="image/png", uri="image.png"
    )

    await client.prompt("Describe this image.", content=[image])

    prompt = next(m for m in transport.sent if m.get("method") == "session/prompt")
    assert prompt["params"]["prompt"] == [
        {"type": "text", "text": "Describe this image."},
        {
            "type": "image",
            "data": "iVBORw0KGgo=",
            "mimeType": "image/png",
            "uri": "image.png",
        },
    ]


@pytest.mark.asyncio
async def test_prompt_rejects_empty_content() -> None:
    client, _ = await _client_with_responses({}, [])
    with pytest.raises(ValueError, match="requires text"):
        await client.prompt()


@pytest.mark.asyncio
async def test_session_recover_prefers_advertised_resume() -> None:
    client, transport = await _client_with_responses(
        {"loadSession": True, "sessionCapabilities": {"resume": {}}},
        [{"configOptions": []}],
    )
    recovered = await client.session_recover("persisted-1", cwd="/workspace")
    assert recovered.session_id == "persisted-1"
    assert transport.sent[-1]["method"] == "session/resume"


@pytest.mark.asyncio
async def test_session_recover_falls_back_to_advertised_load() -> None:
    client, transport = await _client_with_responses(
        {"loadSession": True}, [{"sessionId": "persisted-1"}]
    )
    await client.session_recover("persisted-1")
    assert transport.sent[-1]["method"] == "session/load"


@pytest.mark.asyncio
async def test_session_recover_rejects_unadvertised_recovery() -> None:
    client, _ = await _client_with_responses({}, [])
    with pytest.raises(RuntimeError, match="does not advertise"):
        await client.session_recover("persisted-1")

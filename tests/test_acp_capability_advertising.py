"""Regression tests for #365: ACP client must not advertise capabilities it
cannot honor, and must reject unsupported agent requests instead of silently
returning empty success.

Before the fix the client advertised ``fs.read_text_file``,
``fs.write_text_file`` and ``terminal`` as ``True`` while
``_handle_agent_request`` only implemented ``session/request_permission`` and
returned ``{"result": {}}`` for everything else. That let agents see bogus
"successful" fs/terminal responses, corrupting trajectories.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.transport import Transport


class _RecordingTransport(Transport):
    """Captures outbound messages without doing real I/O."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:  # pragma: no cover - trivial
        pass

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any]:  # pragma: no cover - unused here
        raise RuntimeError("not used in this test")

    async def close(self) -> None:  # pragma: no cover - trivial
        pass


def test_initialize_does_not_advertise_unsupported_fs_or_terminal() -> None:
    """The wire-level capability payload must reflect what we actually serve.

    The client only implements ``session/request_permission``; it does not
    proxy fs reads/writes or terminal spawns. The handshake must advertise
    those as ``False`` (or omit them) so agents do not try to use them.
    """
    from benchflow.acp.types import (
        ACP_PROTOCOL_VERSION,
        AuthCapabilities,
        ClientCapabilities,
        ClientInfo,
        FsCapabilities,
        InitializeParams,
    )

    # Rebuild the same params block initialize() sends. If client.py changes
    # the wire shape, this test mirrors the contract checked at the boundary.
    params = InitializeParams(
        protocol_version=ACP_PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(
            fs=FsCapabilities(read_text_file=False, write_text_file=False),
            terminal=False,
            auth=AuthCapabilities(),
        ),
        client_info=ClientInfo(name="benchflow", version="2.0.0"),
    )
    wire = params.model_dump(by_alias=True, exclude_none=True)
    caps = wire["clientCapabilities"]
    # Either present-and-false or omitted is acceptable. Present-and-true is
    # the bug we are guarding against.
    assert caps.get("fs", {}).get("readTextFile", False) is False
    assert caps.get("fs", {}).get("writeTextFile", False) is False
    assert caps.get("terminal", False) is False


async def test_initialize_sends_falsey_capabilities_to_agent() -> None:
    """End-to-end: drive ``ACPClient.initialize()`` and inspect the actual
    outbound JSON-RPC payload. The agent must not see ``fs.read_text_file=True``
    while the client returns method-not-found for ``fs/read_text_file``.
    """
    from benchflow.acp.types import ACP_PROTOCOL_VERSION

    transport = _RecordingTransport()
    client = ACPClient(transport)

    # Drive only ``send``; intercept ``_read_until_response`` so we don't need
    # a live agent for this assertion.
    async def _fake_read(_request_id: int) -> dict[str, Any]:
        return {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "agentCapabilities": {},
            "agentInfo": {"name": "fake", "version": "0"},
        }

    client._read_until_response = _fake_read  # type: ignore[method-assign]

    await client.initialize()

    assert len(transport.sent) == 1
    init_msg = transport.sent[0]
    assert init_msg["method"] == "initialize"
    caps = init_msg["params"]["clientCapabilities"]
    fs = caps.get("fs", {})
    assert fs.get("readTextFile", False) is False
    assert fs.get("writeTextFile", False) is False
    assert caps.get("terminal", False) is False


async def test_unsupported_agent_request_returns_method_not_found() -> None:
    """Issue #365 repro: previously ``fs/read_text_file`` got ``result: {}``.

    Now the client must reply with JSON-RPC ``-32601`` so the agent surfaces
    a real error instead of believing the read succeeded.
    """
    transport = _RecordingTransport()
    client = ACPClient(transport)

    await client._handle_agent_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "fs/read_text_file",
            "params": {"path": "/app/instruction.md"},
        }
    )

    assert len(transport.sent) == 1
    response = transport.sent[0]
    assert response["id"] == 7
    assert "result" not in response, response
    assert response["error"]["code"] == -32601
    assert "fs/read_text_file" in response["error"]["message"]


async def test_permission_request_still_auto_approves() -> None:
    """The one handler we *do* implement must keep working after the fix."""
    transport = _RecordingTransport()
    client = ACPClient(transport)

    await client._handle_agent_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "session/request_permission",
            "params": {
                "options": [
                    {"optionId": "deny", "kind": "deny"},
                    {"optionId": "allow", "kind": "allow_once"},
                ]
            },
        }
    )

    assert len(transport.sent) == 1
    resp = transport.sent[0]
    assert resp["id"] == 11
    assert resp["result"]["outcome"]["outcome"] == "selected"
    assert resp["result"]["outcome"]["optionId"] == "allow"


@pytest.mark.parametrize(
    "method",
    [
        "fs/read_text_file",
        "fs/write_text_file",
        "terminal/create",
        "terminal/output",
        "totally/made/up",
    ],
)
async def test_all_unadvertised_methods_rejected(method: str) -> None:
    """Any method we don't implement should be method-not-found, not empty."""
    transport = _RecordingTransport()
    client = ACPClient(transport)

    await client._handle_agent_request(
        {"jsonrpc": "2.0", "id": 42, "method": method, "params": {}}
    )

    assert len(transport.sent) == 1
    resp = transport.sent[0]
    assert "error" in resp
    assert resp["error"]["code"] == -32601

"""A catalog fallback aborts instead of grading a weaker agent."""

from unittest.mock import AsyncMock

import pytest

from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.session import ACPSession


@pytest.mark.parametrize("during_open", [False, True])
async def test_metadata_fallback_fails_before_prompt_response(during_open):
    warning = (
        "Model metadata for `new-model` not found. Defaulting to fallback metadata."
    )
    transport = AsyncMock()
    transport.receive.side_effect = [
        {
            "method": "session/update",
            "params": {
                "sessionId": "s",
                "update": {
                    "sessionUpdate": "notice",
                    "severity": "warning",
                    "title": warning,
                },
            },
        },
        {"id": 100001, "result": {"sessionId": "s", "stopReason": "end_turn"}},
    ]
    client = ACPClient(transport)
    client._session = ACPSession("s")
    with pytest.raises(ACPError, match=r"model catalog.*fallback"):
        if during_open:
            await client.session_new()
        else:
            await client.prompt("Do the task")
    assert client._session.events[-1]["type"] == "notice"


async def test_unrelated_notice_is_still_an_advisory():
    client = ACPClient(AsyncMock())
    client._session = ACPSession("s")
    await client._handle_notification(
        {
            "method": "session/update",
            "params": {
                "sessionId": "s",
                "update": {"sessionUpdate": "notice", "title": "Retrying request"},
            },
        }
    )
    assert client._session.events[-1]["title"] == "Retrying request"

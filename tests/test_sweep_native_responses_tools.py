"""Guards the fix at 363d815b: native Responses tools survive."""

import pytest

from benchflow.providers.litellm_logging import callback_module_source


@pytest.mark.parametrize("call_type", ["responses", "aresponses"])
@pytest.mark.parametrize("tool_type", ["namespace", "custom", "web_search", "function"])
@pytest.mark.asyncio
async def test_native_responses_tools_are_preserved(call_type, tool_type):
    namespace = {}
    exec(callback_module_source(), namespace)
    data = {"input": "hello", "tools": [{"type": tool_type, "name": "apply_patch"}]}
    cleaned = await namespace["proxy_handler_instance"].async_pre_call_hook(
        None, None, data, call_type
    )
    assert (cleaned or data)["tools"] == data["tools"]

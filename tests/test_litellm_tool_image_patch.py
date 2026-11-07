"""Guards BerriAI/litellm#34462 on the pinned LiteLLM: a tool result's lone image reaches an OpenAI-compatible provider as an image.

LiteLLM 1.91.5's Anthropic adapter turned a ``tool_result`` whose only block is an image into a tool message whose text is the
image's data URL. Claude Code's Read tool returns exactly that, so on xiaomi/ and other OpenAI-compatible routes the model got
each picture as base64 text (seen 2026-11-03 on MiMo v2.6 Pro: 408,000 base64 characters in a turn cost 301,000 prompt tokens,
and two pictures in one turn failed the request). A proxy startup patch rewrote that message on 1.91.5. LiteLLM 1.103.3's adapter
sends the image as an image part itself, so the proxy no longer loads that patch: its wrapper took only 1.91.5's arguments, and
on 1.103.3 every bridged request failed (``unexpected keyword argument 'custom_llm_provider'``).
"""

import base64

import benchflow.providers.litellm_runtime as runtime_mod

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()
SOURCE = {"type": "base64", "media_type": "image/png", "data": PNG}
URL = f"data:image/png;base64,{PNG}"


def _translate(tool_result_content):
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    messages = [
        {"role": "user", "content": "Look at the picture."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a.png"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": tool_result_content,
                }
            ],
        },
    ]
    out = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages, model="mimo-v2.6-pro", custom_llm_provider="openai"
    )
    return [m for m in out if m.get("role") == "tool"]


def test_lone_image_tool_result_becomes_an_image_part() -> None:
    (tool,) = _translate([{"type": "image", "source": SOURCE}])
    assert tool["tool_call_id"] == "toolu_1"
    assert tool["content"] == [{"type": "image_url", "image_url": {"url": URL}}]


def test_image_with_text_keeps_litellms_own_parts() -> None:
    (tool,) = _translate(
        [{"type": "image", "source": SOURCE}, {"type": "text", "text": "a screenshot"}]
    )
    assert isinstance(tool["content"], list)
    assert {"type": "image_url", "image_url": {"url": URL}} in tool["content"]
    assert {"type": "text", "text": "a screenshot"} in tool["content"]


def test_text_tool_results_are_left_alone() -> None:
    (tool,) = _translate([{"type": "text", "text": "file contents"}])
    assert tool["content"] == "file contents"
    (tool,) = _translate("plain string result")
    assert tool["content"] == "plain string result"


def test_runtime_files_no_longer_load_a_tool_image_patch(tmp_path) -> None:
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    sitecustomize = (tmp_path / "sitecustomize.py").read_text()
    assert "tool_image_patch" not in sitecustomize
    assert not (tmp_path / "benchflow_litellm_tool_image_patch.py").exists()

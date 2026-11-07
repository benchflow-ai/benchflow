"""The prime-sft training window: anchored at the last instruction, with the agent's whole trace.

Guards commit 4f8fa5d6: a Claude Code run (through MiMo's Anthropic endpoint) records mid-conversation
system messages, such as "<total_tokens>14967725 tokens left</total_tokens>", which the exporter
remaps to "user". The window anchored on the last of them, so `bench train convert` wrote a row
with that notice and the final answer only, and `bench train validate --require-tool-calls`
failed with "0 / 1".
"""

from __future__ import annotations

from benchflow.trajectories.export_prime_sft import prime_sft_last_user_training_window


def _tool_call(i: int) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
    }


def test_notices_after_tool_results_do_not_anchor_the_window() -> None:
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Write the answer to /root/answer.txt."},
        {"role": "user", "content": "# Environment ..."},  # a remapped system message
        _tool_call(1),
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {
            "role": "user",
            "content": "<total_tokens>14984720 tokens left</total_tokens>",
        },
        _tool_call(2),
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {
            "role": "user",
            "content": "<total_tokens>14967725 tokens left</total_tokens>",
        },
        {"role": "assistant", "content": "The result is written."},
    ]
    window = prime_sft_last_user_training_window(messages)
    assert window is not None
    prompt, completion = window
    assert [m["content"] for m in prompt] == [
        "Write the answer to /root/answer.txt.",
        "# Environment ...",
    ]
    assert sum(1 for m in completion if m.get("tool_calls")) == 2
    assert completion[-1]["content"] == "The result is written."


def test_a_single_turn_run_keeps_its_window() -> None:
    messages = [
        {"role": "system", "content": "big system prompt"},
        {"role": "user", "content": "task"},
        _tool_call(1),
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    window = prime_sft_last_user_training_window(messages)
    assert window is not None
    prompt, completion = window
    assert prompt == [messages[1]]
    assert completion == messages[2:]


def test_a_second_round_instruction_still_anchors_the_window() -> None:
    messages = [
        {"role": "user", "content": "round 0"},
        _tool_call(1),
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "The tests failed: ..."},
        _tool_call(2),
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "fixed"},
    ]
    window = prime_sft_last_user_training_window(messages)
    assert window is not None
    prompt, completion = window
    assert [m["content"] for m in prompt] == ["The tests failed: ..."]
    assert completion == messages[5:]

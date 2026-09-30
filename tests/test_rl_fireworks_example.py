"""Offline tests for the Fireworks RL cookbook's token-level chat and datums."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "fireworks"
sys.path.insert(0, str(EXAMPLE))

import fireworks_chat as fc  # noqa: E402

COMPLETION = (
    "Count the lines.\n</think>\n\n<tool_call>\n<function=run_bash>\n<parameter=command>\n"
    "wc -l /etc/passwd\n</parameter>\n</function>\n</tool_call><|im_end|>"
)


def test_parse_reasoning_and_tool_call():
    message = fc.parse_completion(COMPLETION)
    assert message["role"] == "assistant"
    assert message["reasoning_content"] == "Count the lines."
    assert message["content"] == ""
    [call] = message["tool_calls"]
    assert call["function"]["name"] == "run_bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "wc -l /etc/passwd"}


def test_parse_keeps_multiline_values_and_content_before_the_call():
    text = (
        "think\n</think>\n\nI will submit.\n\n<tool_call>\n<function=submit>\n"
        "<parameter=answer>\nline one\nline two\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    message = fc.parse_completion(text)
    assert message["content"] == "I will submit."
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args == {"answer": "line one\nline two"}


def test_parse_two_calls_and_the_json_form():
    text = (
        "t</think>\n\n<tool_call>\n<function=run_bash>\n<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call>\n<tool_call>\n"
        '{"name": "submit", "arguments": {"answer": "42"}}\n</tool_call><|endoftext|>'
    )
    calls = fc.parse_completion(text)["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["run_bash", "submit"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"answer": "42"}
    assert [c["id"] for c in calls] == ["call_0", "call_1"]


def test_a_completion_cut_off_while_thinking_has_no_tool_call():
    message = fc.parse_completion("still thinking about <tool_call> maybe")
    assert "tool_calls" not in message
    assert message["content"] == ""
    assert message["reasoning_content"].startswith("still thinking")


def test_malformed_tool_call_is_not_a_call():
    message = fc.parse_completion("t</think>\n\n<tool_call>not json</tool_call><|im_end|>")
    assert "tool_calls" not in message


def test_template_message_gives_the_template_a_mapping():
    assistant = fc.parse_completion(COMPLETION)
    rendered = fc.template_message(assistant)
    assert rendered["tool_calls"][0]["function"]["arguments"] == {"command": "wc -l /etc/passwd"}
    # The caller's message (kept for the audit record) is not changed.
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)
    user = {"role": "user", "content": "hi"}
    assert fc.template_message(user) is user


class FakeTokenizer:
    """Characters as tokens; the chat markers as single tokens."""

    SPECIAL = {"<|im_end|>": 1, "<|endoftext|>": 2}

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for marker, token in self.SPECIAL.items():
                if text.startswith(marker, i):
                    ids.append(token)
                    i += len(marker)
                    break
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        back = {v: k for k, v in self.SPECIAL.items()}
        return "".join(back.get(i, chr(i - 1000)) for i in ids)

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append((messages, tools, kwargs))
        return "|".join(str(m.get("content")) for m in messages) + "|gen"


def test_qwen_chat_renders_with_mapped_arguments_and_template_kwargs():
    tokenizer = FakeTokenizer()
    chat = fc.QwenChat(tokenizer, template_kwargs={"reasoning_effort": "low"})
    assert chat.stop_tokens == [1, 2]
    assistant = fc.parse_completion(COMPLETION)
    ids = chat.render([{"role": "user", "content": "q"}, assistant], [{"type": "function"}])
    assert tokenizer.decode(ids) == "q||gen"
    messages, tools, kwargs = tokenizer.calls[-1]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"command": "wc -l /etc/passwd"}
    assert tools == [{"type": "function"}]
    assert kwargs == {"reasoning_effort": "low"}
    assert chat.parse(tokenizer.encode(COMPLETION))["tool_calls"][0]["function"]["name"] == "run_bash"


def _turn(prompt, completion, first_logprob):
    return fc.Turn(prompt, completion, [first_logprob - i for i in range(len(completion))])


def test_join_turns_merges_turns_whose_prompt_extends_the_sequence():
    t0 = _turn([1, 2, 3], [10, 11], -0.1)
    t1 = _turn([1, 2, 3, 10, 11, 4, 5], [12], -0.3)  # tool result 4, 5 then the next reply
    [sequence] = fc.join_turns([t0, t1])
    assert sequence.tokens == [1, 2, 3, 10, 11, 4, 5, 12]
    assert sequence.sampled == [False, False, False, True, True, False, False, True]
    assert sequence.logprobs == [0.0, 0.0, 0.0, -0.1, -1.1, 0.0, 0.0, -0.3]


def test_join_turns_splits_where_the_template_rewrote_history():
    t0 = _turn([1, 2, 3], [10, 11], -0.1)
    t1 = _turn([1, 2, 3, 10, 99, 4], [12], -0.3)  # the re-render tokenized turn 0 differently
    s0, s1 = fc.join_turns([t0, t1])
    assert s0.tokens == [1, 2, 3, 10, 11]
    assert s1.tokens == [1, 2, 3, 10, 99, 4, 12]
    assert s1.sampled == [False] * 6 + [True]
    # Every sampled token is trained exactly once, with its own logprob.
    sampled = [(t, lp) for s in (s0, s1) for t, m, lp in zip(s.tokens, s.sampled, s.logprobs) if m]
    assert sampled == [(10, -0.1), (11, -1.1), (12, -0.3)]


def test_datum_arrays_align_targets_with_the_next_token():
    [sequence] = fc.join_turns([_turn([1, 2, 3], [10, 11], -0.5)])
    arrays = fc.datum_arrays(sequence, 0.75)
    assert arrays["input_tokens"] == [1, 2, 3, 10]
    assert arrays["target_tokens"] == [0, 0, 10, 11]
    assert arrays["logprobs"] == [0.0, 0.0, -0.5, -1.5]
    assert arrays["advantages"] == [0.0, 0.0, 0.75, 0.75]
    assert len({len(v) for v in arrays.values()}) == 1


def test_group_advantages_leave_drops_out():
    advantages = fc.group_advantages([1.0, 0.0, None, 1.0])
    assert advantages[2] is None
    kept = [1.0, 0.0, 1.0]
    mean = sum(kept) / 3
    std = math.sqrt(sum((r - mean) ** 2 for r in kept) / 2)
    assert advantages[0] == pytest.approx((1.0 - mean) / std)
    assert advantages[1] == pytest.approx((0.0 - mean) / std)
    assert sum(a for a in advantages if a is not None) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "rewards",
    [[1.0, 1.0, 1.0], [0.0, 0.0, None], [None, None, 1.0], [None, None]],
)
def test_groups_without_spread_teach_nothing(rewards):
    assert fc.group_advantages(rewards) == [None if r is None else 0.0 for r in rewards]


def test_turn_rejects_missing_logprobs():
    with pytest.raises(ValueError):
        fc.Turn([1], [2, 3], [-0.1])


def test_checkpoint_names_fit_fireworks_limit():
    assert fc.checkpoint_name("s", 7) == "s007"
    assert fc.checkpoint_name("final", 12) == "final012"
    assert len(fc.checkpoint_name("base", 0)) <= fc.CHECKPOINT_NAME_MAX
    with pytest.raises(ValueError):
        fc.checkpoint_name("a-very-long-prefix", 1)
    with pytest.raises(ValueError):
        fc.checkpoint_name("Final", 1)

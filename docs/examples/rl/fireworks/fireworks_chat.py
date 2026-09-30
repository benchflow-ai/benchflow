"""Token-level chat for RL on Fireworks' serverless Training API.

The Training API samples tokens, not chat messages: the loop renders the
conversation with the model's own chat template, samples a completion with
its per-token logprobs, and parses the completion back into an assistant
message with tool calls. Training needs the exact tokens the policy sampled,
so :func:`build_datums` joins an episode's turns into one sequence only where
the next turn's prompt provably extends the previous prompt and completion,
token for token, and starts a new sequence where it does not (a template that
normalizes whitespace, or tokenizes a boundary differently). Every sampled
token is trained exactly once, with the logprob it was sampled with.

This module has no Fireworks or BenchFlow imports, so its tests run offline.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Qwen3.x tool calls: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_THINK_END = "</think>"
_END_MARKERS = ("<|im_end|>", "<|endoftext|>")


class QwenChat:
    """Qwen3.8's chat format: render with the tokenizer's template, parse completions.

    The template's generation prompt opens ``<think>\\n``, so a completion is
    the reasoning, ``</think>``, then content and tool calls, ending at
    ``<|im_end|>``.
    """

    def __init__(self, tokenizer: Any, *, template_kwargs: dict[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        self.stop_tokens = [self._single_token(marker) for marker in _END_MARKERS]

    def _single_token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"expected {text!r} to be one token, got {ids}")
        return int(ids[0])

    def render(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> list[int]:
        """Token ids of the whole conversation plus the generation prompt."""

        text = self.tokenizer.apply_chat_template(
            [template_message(m) for m in messages],
            tools=list(tools) or None,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def parse(self, completion: Sequence[int]) -> dict[str, Any]:
        """The assistant message a completion encodes (see :func:`parse_completion`)."""

        return parse_completion(self.tokenizer.decode(list(completion), skip_special_tokens=False))


def template_message(message: dict[str, Any]) -> dict[str, Any]:
    """A message as the chat template wants it: tool-call arguments as a mapping."""

    if message.get("role") != "assistant" or not message.get("tool_calls"):
        return message
    calls = []
    for call in message["tool_calls"]:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        function["arguments"] = arguments if isinstance(arguments, dict) else {}
        calls.append({**call, "function": function})
    return {**message, "tool_calls": calls}


def parse_completion(text: str) -> dict[str, Any]:
    """Parse a decoded completion into an OpenAI-style assistant message.

    ``reasoning_content`` holds the thinking; ``tool_calls`` carry JSON
    argument strings, as an OpenAI endpoint returns them. A completion that
    stopped inside its thinking (the token limit) has no content and no tool
    calls, which ends the episode as a model that did not call a tool would.
    """

    for marker in _END_MARKERS:
        if text.endswith(marker):
            text = text[: -len(marker)]
            break
    if _THINK_END in text:
        reasoning, _, rest = text.partition(_THINK_END)
    else:
        reasoning, rest = text, ""
    calls: list[dict[str, Any]] = []
    for index, block in enumerate(_TOOL_CALL.findall(rest)):
        call = _parse_call(block)
        if call is not None:
            name, arguments = call
            calls.append(
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
    content = _TOOL_CALL.sub("", rest).strip()
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning.strip(),
    }
    if calls:
        message["tool_calls"] = calls
    return message


def _parse_call(block: str) -> tuple[str, dict[str, Any]] | None:
    function = _FUNCTION.search(block)
    if function is not None:
        name = function.group(1).strip()
        arguments = {k.strip(): v for k, v in _PARAMETER.findall(function.group(2))}
        return name, arguments
    try:  # the older JSON form: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        payload = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments")
    return payload["name"], arguments if isinstance(arguments, dict) else {}


@dataclass
class Turn:
    """One sampled turn: the prompt it saw and the tokens it produced."""

    prompt: list[int]
    completion: list[int]
    logprobs: list[float]

    def __post_init__(self) -> None:
        if len(self.completion) != len(self.logprobs):
            raise ValueError(
                f"{len(self.completion)} sampled tokens but {len(self.logprobs)} logprobs"
            )


@dataclass
class TokenSequence:
    """A training sequence: tokens, with the sampled positions marked."""

    tokens: list[int] = field(default_factory=list)
    sampled: list[bool] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    def extend(self, tokens: Sequence[int], *, logprobs: Sequence[float] | None = None) -> None:
        self.tokens.extend(int(t) for t in tokens)
        if logprobs is None:
            self.sampled.extend(False for _ in tokens)
            self.logprobs.extend(0.0 for _ in tokens)
        else:
            self.sampled.extend(True for _ in tokens)
            self.logprobs.extend(float(x) for x in logprobs)


def join_turns(turns: Sequence[Turn]) -> list[TokenSequence]:
    """Join turns into as few sequences as keep every sampled token exact.

    Turn ``t+1`` continues the current sequence when its prompt starts with
    that sequence's tokens (the previous prompt and completion); the tokens
    after them (the tool results, the next generation prompt) join unsampled.
    Otherwise turn ``t+1`` starts a new sequence from its own prompt.
    """

    sequences: list[TokenSequence] = []
    current: TokenSequence | None = None
    for turn in turns:
        if current is not None and turn.prompt[: len(current.tokens)] == current.tokens:
            current.extend(turn.prompt[len(current.tokens) :])
        else:
            current = TokenSequence()
            sequences.append(current)
            current.extend(turn.prompt)
        current.extend(turn.completion, logprobs=turn.logprobs)
    return sequences


def group_advantages(rewards: Sequence[float | None]) -> list[float | None]:
    """GRPO advantages within one task's group: ``(r - mean) / std``.

    ``None`` is a dropped episode (infrastructure, under the training rule): it
    is left out of the mean and std, and gets no advantage. A group whose kept
    rewards are all equal teaches nothing, and gets 0 everywhere. The std is
    the sample std, floored at 1e-6 as in Fireworks' recipes.
    """

    kept = [r for r in rewards if r is not None]
    if len(kept) < 2 or len(set(kept)) == 1:
        return [None if r is None else 0.0 for r in rewards]
    mean = sum(kept) / len(kept)
    std = math.sqrt(sum((r - mean) ** 2 for r in kept) / (len(kept) - 1))
    std = max(std, 1e-6)
    return [None if r is None else (r - mean) / std for r in rewards]


def datum_arrays(sequence: TokenSequence, advantage: float) -> dict[str, list]:
    """The shifted next-token arrays of one importance-sampling datum.

    The model reads ``tokens[:-1]``; position ``i`` predicts ``tokens[i + 1]``.
    Unsampled targets (prompt, tool results) get target 0, logprob 0 and
    advantage 0, as in Fireworks' recipes, so only sampled tokens carry loss.
    """

    inputs = sequence.tokens[:-1]
    targets, logprobs, advantages = [], [], []
    for position in range(1, len(sequence.tokens)):
        if sequence.sampled[position]:
            targets.append(sequence.tokens[position])
            logprobs.append(sequence.logprobs[position])
            advantages.append(float(advantage))
        else:
            targets.append(0)
            logprobs.append(0.0)
            advantages.append(0.0)
    return {
        "input_tokens": inputs,
        "target_tokens": targets,
        "logprobs": logprobs,
        "advantages": advantages,
    }


CHECKPOINT_NAME_MAX = 17


def checkpoint_name(prefix: str, step: int) -> str:
    """A checkpoint name Fireworks can promote: at most 17 characters."""

    name = f"{prefix}{step:03d}"
    if len(name) > CHECKPOINT_NAME_MAX or not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(
            f"checkpoint name {name!r} must be 1-{CHECKPOINT_NAME_MAX} of a-z, 0-9, '-'"
        )
    return name

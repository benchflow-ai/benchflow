"""Tokenizer-aware validation and context windowing for TRL SFT rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def load_tokenizer(tokenizer_id: str, revision: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "tokenizer validation requires the benchflow train or trl extra"
        ) from exc
    kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)


def training_chat_template(tokenizer: Any) -> str | None:
    try:
        from trl.chat_template_utils import get_training_chat_template
    except ImportError as exc:
        raise ValueError(
            "assistant-mask validation requires the benchflow trl extra"
        ) from exc
    return get_training_chat_template(tokenizer)


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("tokenizer apply_chat_template did not return input_ids")
    return value


def _assistant_masks(value: Any) -> list[int]:
    if not isinstance(value, Mapping):
        raise ValueError("tokenizer did not return assistant token masks")
    masks = value.get("assistant_masks")
    if isinstance(masks, list) and masks and isinstance(masks[0], list):
        masks = masks[0]
    if not isinstance(masks, list) or not all(
        isinstance(item, int | bool) for item in masks
    ):
        raise ValueError("chat template does not provide assistant token masks")
    return [int(item) for item in masks]


def validate_tokenized_row(
    row: dict[str, Any],
    *,
    row_num: int,
    tokenizer: Any,
    chat_template: str | None,
    max_length: int | None,
) -> tuple[int, int]:
    prompt = cast(list[dict[str, Any]], row["prompt"])
    completion = cast(list[dict[str, Any]], row["completion"])
    tools = cast(list[dict[str, Any]], row.get("tools") or [])
    template_kwargs: dict[str, Any] = {"tools": tools or None}
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    prompt_output = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **template_kwargs,
    )
    full_output = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **template_kwargs,
    )
    prompt_ids = _input_ids(prompt_output)
    full_ids = _input_ids(full_output)
    assistant_masks = _assistant_masks(full_output)
    if len(assistant_masks) != len(full_ids):
        raise ValueError(
            f"row {row_num}: assistant mask length does not match input_ids"
        )
    prefix_length = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids, strict=False):
        if prompt_id != full_id:
            break
        prefix_length += 1
    first_assistant = next(
        (index for index, value in enumerate(assistant_masks) if value),
        None,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids and (
        len(prompt_ids) > len(full_ids)
        or first_assistant is None
        or prefix_length < first_assistant
    ):
        raise ValueError(
            f"row {row_num}: tokenized prompt differs before the assistant "
            "generation boundary"
        )
    trainable = sum(assistant_masks[len(prompt_ids) :])
    if trainable < 1:
        raise ValueError(f"row {row_num}: no trainable assistant completion tokens")
    if max_length is not None and len(full_ids) > max_length:
        raise ValueError(
            f"row {row_num}: tokenized length {len(full_ids)} exceeds "
            f"max_length {max_length}"
        )
    return len(full_ids), trainable


def _message_groups(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        if message.get("role") == "assistant":
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


def _required_prompt_prefix(
    prompt: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    end = 0
    while end < len(prompt) and prompt[end].get("role") == "system":
        end += 1
    first_user = next(
        (
            index
            for index in range(end, len(prompt))
            if prompt[index].get("role") == "user"
        ),
        None,
    )
    if first_user is not None:
        end = first_user + 1
    elif end == 0:
        end = 1
    return prompt[:end], prompt[end:]


def window_trl_sft_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    chat_template: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    max_length: int,
) -> tuple[dict[str, Any], int, int, int, int]:
    original = dict(row)
    original_prompt = cast(list[dict[str, Any]], original["prompt"])
    original_tokens, _ = validate_tokenized_row(
        original,
        row_num=1,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=None,
    )
    if original_tokens <= max_length:
        final = dict(original)
        final["context_window"] = _context_window_metadata(
            original_messages=len(original_prompt),
            retained_messages=len(original_prompt),
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            max_length=max_length,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
        )
        return final, 0, 0, original_tokens, original_tokens

    prefix, tail = _required_prompt_prefix(original_prompt)
    retained_groups: list[list[dict[str, Any]]] = []
    final_tokens = 0
    for group in reversed(_message_groups(tail)):
        candidate_groups = [group, *retained_groups]
        candidate = dict(original)
        candidate["prompt"] = prefix + [
            message for retained_group in candidate_groups for message in retained_group
        ]
        candidate_tokens, _ = validate_tokenized_row(
            candidate,
            row_num=1,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=None,
        )
        if candidate_tokens > max_length:
            break
        retained_groups = candidate_groups
        final_tokens = candidate_tokens

    if tail and not retained_groups:
        raise ValueError(
            "most recent assistant/tool context group does not fit max_length "
            "with the required system and task prefix"
        )
    final = dict(original)
    final["prompt"] = prefix + [
        message for group in retained_groups for message in group
    ]
    if final_tokens == 0:
        final_tokens, _ = validate_tokenized_row(
            final,
            row_num=1,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
        )
    messages_dropped = len(original_prompt) - len(final["prompt"])
    final["context_window"] = _context_window_metadata(
        original_messages=len(original_prompt),
        retained_messages=len(final["prompt"]),
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        max_length=max_length,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    return final, 1, messages_dropped, original_tokens, final_tokens


def _context_window_metadata(
    *,
    original_messages: int,
    retained_messages: int,
    original_tokens: int,
    final_tokens: int,
    max_length: int,
    tokenizer_id: str,
    tokenizer_revision: str | None,
) -> dict[str, Any]:
    return {
        "policy": "message-window",
        "original_messages": original_messages,
        "retained_messages": retained_messages,
        "messages_dropped": original_messages - retained_messages,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "max_length": max_length,
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
    }

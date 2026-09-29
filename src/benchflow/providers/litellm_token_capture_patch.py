"""LiteLLM startup patch: keep streamed token ids and logprobs for capture.

LiteLLM 1.91.0 logs a streamed chat completion as the response that
``stream_chunk_builder`` assembles from the chunks, and that assembly drops
each choice's ``logprobs`` and ``token_ids`` and the top-level
``prompt_token_ids`` a vLLM-compatible server streams, and the final
``sglext`` chunk (``input_ids``, ``output_ids``) a SGLang server streams. The client still sees
them per chunk, but ``llm_trajectory.jsonl`` did not.

With ``BENCHFLOW_CAPTURE_TOKEN_LOGPROBS`` on, this patch copies those fields
from every raw provider chunk into the call's logging details under
``STREAM_TOKENS_KEY``; BenchFlow's LiteLLM callback records them next to the
assembled response. It is loaded inside the LiteLLM proxy process via
``sitecustomize`` and records nothing while capture is off. It always drops
SGLang's final ``sglext``-only chunk (no choices, no usage) after reading it:
there is nothing in it to relay, and LiteLLM's Anthropic Messages stream
adapter fails on a chunk without choices.
"""

from __future__ import annotations

import os
from typing import Any

CAPTURE_ENV = "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS"
STREAM_TOKENS_KEY = "benchflow_stream_tokens"
_TRUTHY = {"1", "true", "yes", "on"}


def capture_enabled() -> bool:
    return os.environ.get(CAPTURE_ENV, "").strip().lower() in _TRUTHY


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    if value is None:
        extra = getattr(obj, "provider_specific_fields", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump())
        except Exception:
            pass
    return str(value)


def _ints(value: Any) -> list[int] | None:
    return [int(t) for t in value] if isinstance(value, list) else None


def record_stream_chunk(details: dict[str, Any], chunk: Any) -> None:
    """Accumulate one provider chunk's token fields into ``details``."""
    stash = details.setdefault(STREAM_TOKENS_KEY, {"chunks": 0, "choices": {}})
    stash["chunks"] += 1
    prompt = _field(chunk, "prompt_token_ids")
    if isinstance(prompt, list) and "prompt_token_ids" not in stash:
        stash["prompt_token_ids"] = [int(t) for t in prompt]
    # SGLang: one response-level ``sglext`` chunk (``choices: []``) carries
    # the prompt ids and one sampled-id list per choice.
    sglext = _field(chunk, "sglext")
    if isinstance(sglext, dict):
        input_ids = _ints(sglext.get("input_ids"))
        if input_ids is not None and "prompt_token_ids" not in stash:
            stash["prompt_token_ids"] = input_ids
        output_ids = sglext.get("output_ids")
        for index, ids in enumerate(output_ids if isinstance(output_ids, list) else []):
            choice_ids = _ints(ids)
            if choice_ids is not None:
                stash["choices"].setdefault(str(index), {})["token_ids"] = choice_ids
    for choice in _field(chunk, "choices") or []:
        index = _field(choice, "index")
        entry = stash["choices"].setdefault(
            str(index if isinstance(index, int) else 0), {}
        )
        token_ids = _field(choice, "token_ids")
        if isinstance(token_ids, list):
            entry.setdefault("token_ids", []).extend(int(t) for t in token_ids)
        content = _field(_field(choice, "logprobs"), "content")
        if isinstance(content, list):
            entry.setdefault("logprobs", []).extend(_plain(item) for item in content)


def is_sglext_only_chunk(chunk: Any) -> bool:
    """True for SGLang's final ``choices: []`` chunk that carries only ``sglext``."""
    try:
        return (
            _field(chunk, "sglext") is not None
            and not _field(chunk, "choices")
            and not _field(chunk, "usage")
        )
    except Exception:
        return False


def _patch_stream_wrapper() -> None:
    try:
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    except Exception:
        return

    original = CustomStreamWrapper.chunk_creator

    def chunk_creator(self: Any, chunk: Any) -> Any:
        if capture_enabled():
            try:
                details = getattr(
                    getattr(self, "logging_obj", None), "model_call_details", None
                )
                if isinstance(details, dict):
                    record_stream_chunk(details, chunk)
            except Exception:
                pass
        if is_sglext_only_chunk(chunk):
            # Nothing to relay, and LiteLLM's Anthropic Messages stream adapter
            # indexes ``choices[0]`` of every chunk (IndexError on SGLang).
            return None
        return original(self, chunk)

    setattr(chunk_creator, "__benchflow_token_capture_patch__", True)  # noqa: B010
    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        CustomStreamWrapper, "chunk_creator", chunk_creator
    )


_patch_stream_wrapper()

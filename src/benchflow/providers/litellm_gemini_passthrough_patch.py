"""Backport Gemini pass-through streaming logging for LiteLLM 1.89/1.91.

Those releases classify every native ``generateContent`` URL as Vertex AI and
send its streamed response through the Vertex logger.  That happens to work for
Gemini model names present in both catalogs, but it rejects Google AI Studio
Gemma names and drops the entire callback — including token usage and provider
trajectory evidence.  LiteLLM added a dedicated Gemini endpoint type later.

The proxy imports this module from ``sitecustomize``.  It patches only the old
vendor shape (no ``EndpointType.GEMINI``) and only Google AI Studio-style model
paths, leaving Vertex and every other provider untouched.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

_GOOGLE_AI_STUDIO_GENERATE_PATH_RE = re.compile(
    r"/v1(?:beta)?/models/[^/:]+:(?:streamGenerateContent|generateContent)$"
)


def _is_google_ai_studio_generate_content(url_route: str) -> bool:
    """Distinguish native AI Studio model paths from Vertex project paths."""
    try:
        path = urlparse(url_route).path
    except Exception:
        return False
    return bool(
        _GOOGLE_AI_STUDIO_GENERATE_PATH_RE.search(path)
        and "/projects/" not in path
        and "/locations/" not in path
    )


def _apply_patch() -> None:
    try:
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
            GeminiPassthroughLoggingHandler,
        )
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )
    except Exception:
        return

    # Newer LiteLLM releases have their own Gemini dispatch branch.  Do not
    # replace supported vendor behavior after the pinned dependency advances.
    if hasattr(EndpointType, "GEMINI"):
        return

    original = PassThroughStreamingHandler._build_passthrough_logging_result
    if getattr(original, "__benchflow_gemini_passthrough_patch__", False):
        return

    def build_passthrough_logging_result(
        *,
        litellm_logging_obj: Any,
        passthrough_success_handler_obj: Any,
        url_route: str,
        request_body: dict[str, Any],
        endpoint_type: Any,
        start_time: Any,
        raw_bytes: list[bytes],
        end_time: Any,
        model: str | None,
    ) -> tuple[Any, dict[str, Any]]:
        if not _is_google_ai_studio_generate_content(url_route):
            return original(
                litellm_logging_obj=litellm_logging_obj,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                start_time=start_time,
                raw_bytes=raw_bytes,
                end_time=end_time,
                model=model,
            )

        all_chunks = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(
            raw_bytes
        )
        result = (
            GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks(
                litellm_logging_obj=litellm_logging_obj,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                start_time=start_time,
                all_chunks=all_chunks,
                end_time=end_time,
                model=model,
            )
        )
        return result["result"], result["kwargs"]

    setattr(  # noqa: B010 - marker on the monkey-patched vendor function
        build_passthrough_logging_result,
        "__benchflow_gemini_passthrough_patch__",
        True,
    )
    setattr(  # noqa: B010 - avoids static narrowing on the vendor API
        PassThroughStreamingHandler,
        "_build_passthrough_logging_result",
        staticmethod(build_passthrough_logging_result),
    )


_apply_patch()

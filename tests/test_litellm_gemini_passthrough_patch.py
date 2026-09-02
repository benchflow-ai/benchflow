"""Regression coverage for the PR #1030 Gemma pass-through logging fix."""

from datetime import datetime
from types import SimpleNamespace

from benchflow.providers.litellm_gemini_passthrough_patch import (
    _is_google_ai_studio_generate_content,
)


def test_google_ai_studio_path_detection_excludes_vertex() -> None:
    assert _is_google_ai_studio_generate_content(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemma-4-31b-it:streamGenerateContent?alt=sse"
    )
    assert not _is_google_ai_studio_generate_content(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/"
        "publishers/google/models/gemma-4-31b-it:streamGenerateContent"
    )


def test_old_litellm_routes_gemma_streams_through_gemini_logger(monkeypatch) -> None:
    """Guards PR #1030: Gemma usage must not be dropped by the Vertex logger."""
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
        GeminiPassthroughLoggingHandler,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    if hasattr(EndpointType, "GEMINI"):
        return

    seen: dict[str, object] = {}

    def fake_gemini_handler(**kwargs):
        seen.update(kwargs)
        return {"result": "gemma-response", "kwargs": {"model": "gemma-4-31b-it"}}

    monkeypatch.setattr(
        GeminiPassthroughLoggingHandler,
        "_handle_logging_gemini_collected_chunks",
        staticmethod(fake_gemini_handler),
    )
    response, kwargs = PassThroughStreamingHandler._build_passthrough_logging_result(
        litellm_logging_obj=SimpleNamespace(),
        passthrough_success_handler_obj=SimpleNamespace(),
        url_route=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemma-4-31b-it:streamGenerateContent?alt=sse"
        ),
        request_body={"contents": []},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=datetime.now(),
        raw_bytes=[b'data: {"usageMetadata":{"promptTokenCount":1}}\n\n'],
        end_time=datetime.now(),
        model=None,
    )

    assert response == "gemma-response"
    assert kwargs["model"] == "gemma-4-31b-it"
    assert seen["all_chunks"] == ['data: {"usageMetadata":{"promptTokenCount":1}}']

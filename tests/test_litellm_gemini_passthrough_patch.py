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


def test_ai_studio_gemma_streams_use_gemini_logger(monkeypatch) -> None:
    """Guards PR #1030: Gemma usage must not be dropped by the Vertex logger."""
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
        GeminiPassthroughLoggingHandler,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

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


def test_ai_studio_path_with_non_vertex_label_is_left_to_litellm(monkeypatch) -> None:
    """PR #1166 guards the LiteLLM 1.103 bump of PR #1030's patch: only streams LiteLLM
    labelled Vertex AI are rerouted, so any other label keeps vendor dispatch."""
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
        GeminiPassthroughLoggingHandler,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    calls: list[dict[str, object]] = []

    def fake_gemini_handler(**kwargs):
        calls.append(kwargs)
        return {"result": "gemma-response", "kwargs": {"model": "gemma-4-31b-it"}}

    monkeypatch.setattr(
        GeminiPassthroughLoggingHandler,
        "_handle_logging_gemini_collected_chunks",
        staticmethod(fake_gemini_handler),
    )
    _, kwargs = PassThroughStreamingHandler._build_passthrough_logging_result(
        litellm_logging_obj=SimpleNamespace(),
        passthrough_success_handler_obj=SimpleNamespace(),
        url_route=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemma-4-31b-it:streamGenerateContent?alt=sse"
        ),
        request_body={"contents": []},
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        raw_bytes=[b'data: {"usageMetadata":{"promptTokenCount":1}}\n\n'],
        end_time=datetime.now(),
        model=None,
    )

    assert calls == []
    assert kwargs == {}


def test_gemma_patch_stays_installed_while_litellm_misroutes_ai_studio() -> None:
    """PR #1166 guards PR #1030 across the LiteLLM 1.103 bump: LiteLLM now has an
    ``EndpointType.GEMINI`` branch, but its pass-through route still classifies
    AI Studio URLs as Vertex AI, so the patch must not switch itself off."""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemma-4-31b-it:streamGenerateContent?alt=sse"
    )
    assert (
        HttpPassThroughEndpointHelpers.get_endpoint_type(url) == EndpointType.VERTEX_AI
    )
    assert getattr(
        PassThroughStreamingHandler._build_passthrough_logging_result,
        "__benchflow_gemini_passthrough_patch__",
        False,
    )

"""Silent provider-API-failure capture (api_error / suspected_api_error).

Covers the post-rollout classification pipeline end to end at the unit level:
status mapping -> proxy failure summary -> verdict -> diagnostics -> retry
policy -> batch circuit breaker. The motivating fixture: an agent that
rejects the model id against its own catalog, issues zero requests, ends its
turn politely, and the verifier scores 0.0 — previously recorded as a healthy
fail with error=None.
"""

from types import SimpleNamespace

from benchflow._utils.scoring import (
    API_ERROR,
    SUSPECTED_API_ERROR,
    api_error_is_transient,
    classify_error,
)
from benchflow.diagnostics import (
    DIAGNOSTIC_BY_FIELD,
    DIAGNOSTIC_REGISTRY,
    ProviderApiErrorDiagnostic,
    SuspectedApiErrorDiagnostic,
)
from benchflow.evaluation import ApiErrorCircuitBreaker, RetryConfig
from benchflow.models import RunResult
from benchflow.rollout._usage import (
    _api_error_subcategory,
    _provider_api_failure_summary_from_runtime,
    classify_api_failure,
)


def _runtime(statuses: list[int | None]) -> SimpleNamespace:
    exchanges = [
        SimpleNamespace(response=SimpleNamespace(status_code=s)) for s in statuses
    ]
    return SimpleNamespace(
        server=SimpleNamespace(trajectory=SimpleNamespace(exchanges=exchanges))
    )


class TestClassifyErrorMarkers:
    def test_api_error_marker(self):
        err = "provider api error [rate_limit/transient] HTTP 429 on 3/3 requests"
        assert classify_error(err) == API_ERROR

    def test_suspected_marker_wins_over_api_marker(self):
        err = "suspected provider api error: agent ended with zero tokens and zero tool calls"
        assert classify_error(err) == SUSPECTED_API_ERROR

    def test_auth_api_error_not_misrouted_to_provider_auth(self):
        # "HTTP 401" is a provider_auth marker, but the api-error branch runs
        # first for the structured string the classifier emits.
        err = "provider api error [auth/permanent] HTTP 401 on 2/2 requests"
        assert classify_error(err) == API_ERROR

    def test_transient_marker_parsing(self):
        assert api_error_is_transient(
            "provider api error [rate_limit/transient] HTTP 429"
        )
        assert not api_error_is_transient(
            "provider api error [auth/permanent] HTTP 401"
        )
        assert not api_error_is_transient(None)


class TestStatusSubcategory:
    def test_mapping(self):
        assert _api_error_subcategory(401) == ("auth", False)
        assert _api_error_subcategory(403) == ("auth", False)
        assert _api_error_subcategory(402) == ("quota", False)
        assert _api_error_subcategory(404) == ("model_not_found", False)
        assert _api_error_subcategory(429) == ("rate_limit", True)
        assert _api_error_subcategory(500) == ("provider_error", True)
        assert _api_error_subcategory(503) == ("provider_error", True)
        assert _api_error_subcategory(408) == ("provider_error", True)
        assert _api_error_subcategory(400) == ("rejected_request", False)
        assert _api_error_subcategory(422) == ("rejected_request", False)

    def test_context_length_imported_as_400_is_permanent(self):
        """Guards issue #830: imported context-window rejects are not retried."""
        summary = _provider_api_failure_summary_from_runtime(_runtime([400]))

        assert summary is not None
        assert summary["subcategory"] == "rejected_request"
        assert summary["transient"] is False
        assert summary["fingerprint"] == "rejected_request:400"
        assert not api_error_is_transient(
            "provider api error [rejected_request/permanent] HTTP 400"
        )


class TestFailureSummary:
    def test_none_without_exchanges(self):
        assert _provider_api_failure_summary_from_runtime(None) is None
        assert _provider_api_failure_summary_from_runtime(_runtime([])) is None

    def test_all_failed(self):
        s = _provider_api_failure_summary_from_runtime(_runtime([429, 429, 500]))
        assert s["total_requests"] == 3
        assert s["failed_requests"] == 3
        assert s["dominant_status"] == 429
        assert s["subcategory"] == "rate_limit"
        assert s["transient"] is True
        assert s["fingerprint"] == "rate_limit:429"
        assert s["status_counts"] == {"429": 2, "500": 1}

    def test_successes_only(self):
        s = _provider_api_failure_summary_from_runtime(_runtime([200, 200]))
        assert s == {"total_requests": 2, "failed_requests": 0}

    def test_non_int_statuses_skipped(self):
        s = _provider_api_failure_summary_from_runtime(_runtime([None, 200, 401]))
        assert s["total_requests"] == 2
        assert s["failed_requests"] == 1
        assert s["subcategory"] == "auth"


class TestClassifyApiFailure:
    def test_proxy_proven(self):
        summary = _provider_api_failure_summary_from_runtime(_runtime([429, 429]))
        verdict, info = classify_api_failure(summary, total_tokens=0, n_tool_calls=0)
        assert verdict == "api_error"
        assert info["subcategory"] == "rate_limit"

    def test_zero_signal_without_proxy_evidence(self):
        verdict, info = classify_api_failure(None, total_tokens=0, n_tool_calls=0)
        assert verdict == "suspected_api_error"
        assert info == {"total_requests": 0, "failed_requests": 0}

    def test_healthy_rollout_never_flagged(self):
        verdict, _ = classify_api_failure(None, total_tokens=26278, n_tool_calls=8)
        assert verdict is None

    def test_partial_failures_with_progress_not_flagged(self):
        # Agent recovered from a mid-run blip: some failures, but real tokens.
        summary = _provider_api_failure_summary_from_runtime(_runtime([429, 200, 200]))
        verdict, _ = classify_api_failure(summary, total_tokens=5000, n_tool_calls=3)
        assert verdict is None

    def test_zero_tools_with_tokens_not_flagged(self):
        # Prompt-only answer (no tools) with real usage is a legitimate rollout.
        verdict, _ = classify_api_failure(None, total_tokens=1200, n_tool_calls=0)
        assert verdict is None


class TestDiagnostics:
    def test_registered(self):
        assert ProviderApiErrorDiagnostic in DIAGNOSTIC_REGISTRY
        assert SuspectedApiErrorDiagnostic in DIAGNOSTIC_REGISTRY
        assert DIAGNOSTIC_BY_FIELD["api_error_info"] is ProviderApiErrorDiagnostic
        assert (
            DIAGNOSTIC_BY_FIELD["suspected_api_error_info"]
            is SuspectedApiErrorDiagnostic
        )

    def test_categories_and_channel(self):
        assert ProviderApiErrorDiagnostic.category == API_ERROR
        assert SuspectedApiErrorDiagnostic.category == SUSPECTED_API_ERROR
        assert ProviderApiErrorDiagnostic.channel == "error"
        assert SuspectedApiErrorDiagnostic.channel == "error"

    def test_format_issue(self):
        diag = ProviderApiErrorDiagnostic(
            subcategory="auth",
            transient=False,
            dominant_status=401,
            total_requests=4,
            failed_requests=4,
            fingerprint="auth:401",
        )
        line = diag.format_issue("some-task")
        assert "auth/permanent" in line and "HTTP 401" in line and "4/4" in line


class TestRetryPolicy:
    def test_transient_api_error_retries(self):
        cfg = RetryConfig()
        assert cfg.should_retry(
            "provider api error [rate_limit/transient] HTTP 429 on 3/3 requests",
            category=API_ERROR,
        )

    def test_permanent_api_error_does_not_retry(self):
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "provider api error [auth/permanent] HTTP 401 on 2/2 requests",
            category=API_ERROR,
        )

    def test_suspected_never_retries(self):
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "suspected provider api error: agent ended with zero tokens and zero tool calls",
            category=SUSPECTED_API_ERROR,
        )

    def test_api_retry_can_be_disabled(self):
        cfg = RetryConfig(retry_on_api_error=False)
        assert not cfg.should_retry(
            "provider api error [rate_limit/transient] HTTP 429",
            category=API_ERROR,
        )


def _api_result(name: str, *, sub: str = "auth", status: int = 401) -> RunResult:
    return RunResult(
        task_name=name,
        error=f"provider api error [{sub}/permanent] HTTP {status} on 1/1 requests",
    )


class TestCircuitBreaker:
    def test_trips_on_same_fingerprint_streak(self):
        breaker = ApiErrorCircuitBreaker(threshold=3)
        for i in range(3):
            assert not breaker.tripped
            breaker.record(_api_result(f"t{i}"))
        assert breaker.tripped
        assert "auth:401" in breaker.skip_error()

    def test_healthy_completion_resets_streak(self):
        breaker = ApiErrorCircuitBreaker(threshold=3)
        breaker.record(_api_result("t0"))
        breaker.record(_api_result("t1"))
        breaker.record(RunResult(task_name="ok", rewards={"reward": 0.0}))
        breaker.record(_api_result("t2"))
        breaker.record(_api_result("t3"))
        assert not breaker.tripped

    def test_different_fingerprints_do_not_accumulate(self):
        breaker = ApiErrorCircuitBreaker(threshold=3)
        breaker.record(_api_result("t0", sub="auth", status=401))
        breaker.record(_api_result("t1", sub="quota", status=402))
        breaker.record(_api_result("t2", sub="auth", status=401))
        assert not breaker.tripped

    def test_transient_api_error_is_not_breaker_relevant(self):
        breaker = ApiErrorCircuitBreaker(threshold=2)
        transient = RunResult(
            task_name="t",
            error="provider api error [rate_limit/transient] HTTP 429 on 1/1 requests",
        )
        breaker.record(transient)
        breaker.record(transient)
        assert not breaker.tripped

    def test_suspected_counts_with_own_fingerprint(self):
        breaker = ApiErrorCircuitBreaker(threshold=2)
        suspected = RunResult(
            task_name="t",
            error=(
                "suspected provider api error: agent ended with zero tokens "
                "and zero tool calls (no scoreable model activity)"
            ),
        )
        breaker.record(suspected)
        breaker.record(suspected)
        assert breaker.tripped

    def test_zero_threshold_disables(self):
        breaker = ApiErrorCircuitBreaker(threshold=0)
        for i in range(10):
            breaker.record(_api_result(f"t{i}"))
        assert not breaker.tripped

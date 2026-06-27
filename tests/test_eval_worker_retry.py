"""Guards PR #564 finding 2: worker retry-config parsing must not silently
revert to retrying provider_auth failures when a payload omits
``exclude_categories``.

The worker reconstructs ``RetryConfig`` from a serialized payload. Centralizing
that in ``RetryConfig.from_mapping`` means omitted fields fall back to the
dataclass defaults (which exclude ``provider_auth``), not a hard-coded literal.
"""

from benchflow.eval_worker import _evaluation_config, _retry_config
from benchflow.evaluation import RetryConfig

_PROVIDER_AUTH_ERROR = (
    "ACP error -32603: Internal error: Failed to authenticate. "
    "API Error: 401 Invalid bearer token"
)
_PROVIDER_RATE_LIMIT_ERROR = (
    "ACP error -32603: Internal error | provider rate limited (HTTP 429)"
)
_PROVIDER_REJECTED_ERROR = (
    "ACP error -32603: Internal error | provider rejected request (HTTP 400)"
)


def test_from_mapping_omitted_exclude_excludes_provider_rejected():
    """Guards #830: a context-window 400 raised as an ACP error must not retry,
    even when the payload omits exclude_categories (falls back to defaults)."""
    cfg = RetryConfig.from_mapping({"max_retries": 3})
    assert not cfg.should_retry(_PROVIDER_REJECTED_ERROR)
    assert "provider_rejected" in cfg.exclude_categories


def test_from_mapping_omitted_exclude_keeps_provider_auth():
    """Guards PR #653: omitted excludes must still exclude provider caps."""
    cfg = RetryConfig.from_mapping({"max_retries": 3})
    assert not cfg.should_retry(_PROVIDER_AUTH_ERROR)
    assert not cfg.should_retry(_PROVIDER_RATE_LIMIT_ERROR)
    assert cfg.max_retries == 3  # other fields still parsed


def test_from_mapping_none_uses_defaults():
    cfg = RetryConfig.from_mapping(None)
    assert not cfg.should_retry(_PROVIDER_AUTH_ERROR)
    assert not cfg.should_retry(_PROVIDER_RATE_LIMIT_ERROR)


def test_from_mapping_explicit_exclude_respected():
    """An explicit exclude_categories is honored verbatim (not merged with
    defaults)."""
    cfg = RetryConfig.from_mapping({"exclude_categories": ["timeout"]})
    assert cfg.exclude_categories == {"timeout"}
    # acp_error is retryable; provider_auth never is (no retry branch for it),
    # so dropping it from excludes still doesn't make auth failures retry.
    assert cfg.should_retry("ACP error -32000: connection refused")
    assert not cfg.should_retry(_PROVIDER_AUTH_ERROR)


def test_worker_retry_config_omitting_exclude_does_not_retry_provider_auth():
    """Worker payload without retry.exclude_categories must not retry auth."""
    cfg = _retry_config({"retry": {"max_retries": 2}})
    assert not cfg.should_retry(_PROVIDER_AUTH_ERROR)
    assert not cfg.should_retry(_PROVIDER_RATE_LIMIT_ERROR)


def test_worker_retry_config_no_retry_key_at_all():
    cfg = _retry_config({})
    assert not cfg.should_retry(_PROVIDER_AUTH_ERROR)
    assert not cfg.should_retry(_PROVIDER_RATE_LIMIT_ERROR)


def test_evaluation_config_retry_excludes_provider_auth_by_default():
    """End-to-end: a worker config payload with no retry block yields an
    EvaluationConfig whose retry refuses to retry provider_auth (#564)."""
    eval_cfg = _evaluation_config({"tasks_dir": "/tmp/tasks", "agent": "oracle"})
    assert not eval_cfg.retry.should_retry(_PROVIDER_AUTH_ERROR)
    assert not eval_cfg.retry.should_retry(_PROVIDER_RATE_LIMIT_ERROR)

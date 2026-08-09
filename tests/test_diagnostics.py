"""Unit tests for the exception-rendering and transient-transport helpers."""

from __future__ import annotations

import pytest

from benchflow._utils.scoring import INFRA_ERROR, classify_error
from benchflow._utils.text import describe_exception
from benchflow.diagnostics import (
    TRANSIENT_SANDBOX_TRANSPORT_MARKER,
    TransientSandboxTransportError,
)
from benchflow.evaluation import RetryConfig


class _SdkError(Exception):
    """Stand-in for an SDK error carrying HTTP response metadata."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def test_describe_exception_leads_with_the_class_name():
    assert describe_exception(ValueError("bad route")) == "ValueError: bad route"


def test_describe_exception_names_an_empty_detail_after_a_wrapper_prefix():
    """The signature that motivated this helper.

    The Daytona SDK wraps every toolbox call as ``"<prefix>: " +
    str(underlying)``, and httpx raises its timeout/connection errors with
    an empty message — so a read timeout on ``execute_session_command``
    stringifies to a prefix with nothing behind the colon. Interpolating the
    exception alone produced ``"Failed to execute session command: ."``,
    which says neither what failed nor that the detail was empty.
    """
    exc = _SdkError("Failed to execute session command: ")

    assert describe_exception(exc) == (
        "_SdkError: Failed to execute session command: (no detail)"
    )


def test_describe_exception_handles_a_wholly_empty_message():
    assert describe_exception(TimeoutError()) == "TimeoutError (no message)"


def test_describe_exception_appends_structured_http_fields():
    exc = _SdkError("boom", status_code=503, error_code="unavailable")

    assert describe_exception(exc) == (
        "_SdkError: boom [status_code=503, error_code=unavailable]"
    )


def test_describe_exception_omits_unset_structured_fields():
    assert describe_exception(_SdkError("boom")) == "_SdkError: boom"


def test_transient_sandbox_transport_error_is_classified_as_retryable_infra():
    """The marker is the whole contract between raise site and classifier."""
    exc = TransientSandboxTransportError(
        describe_exception(_SdkError("Failed to upload files: "))
    )

    message = str(exc)

    assert TRANSIENT_SANDBOX_TRANSPORT_MARKER in message
    assert classify_error(message) == INFRA_ERROR
    assert RetryConfig().should_retry(message)


@pytest.mark.parametrize(
    "prefix",
    [
        "Failed to create session: ",
        "Failed to execute session command: ",
        "Failed to get session command: ",
        "Failed to upload files: ",
        "Failed to download file: ",
        "Failed to get file info: ",
    ],
)
def test_every_vendor_prefix_reaches_infra_through_the_single_marker(prefix):
    """One marker has to cover the whole corridor, not one prefix at a time.

    Listing prefixes in the classifier was the previous approach; it silently
    missed whichever method nobody had hit yet (uploads, stats), which is how
    a proxy-start blip on ``execute`` ended up unretried while the identical
    blip on ``get`` was retried.
    """
    exc = TransientSandboxTransportError(describe_exception(_SdkError(prefix)))

    assert classify_error(str(exc)) == INFRA_ERROR


def test_permanent_vendor_failures_are_not_swept_into_infra():
    """Only the raise site stamps, and only for transient errors.

    A dead credential shares the vendor prefix with a transport blip. If the
    classifier matched the prefix instead of the marker, a 401 would be
    retried as though it would heal.
    """
    unstamped = describe_exception(
        _SdkError("Failed to upload files: unauthorized", status_code=401)
    )

    assert classify_error(unstamped) != INFRA_ERROR


@pytest.mark.asyncio
async def test_contract_installed_daytona_sdk_transport_error_stays_retryable():
    """Contract test driven through the INSTALLED Daytona SDK.

    Every other test in this file hand-rolls the exception, and a hand-rolled
    class satisfies neither ``_is_daytona_transient_retry_error`` (which keys
    off the real module and class name) nor the SDK's own wrapping. So a
    vendor prefix rename, or a change in which class the SDK maps a transport
    failure to, would leave every stub-based test green while rollouts
    silently stopped retrying — the exact failure this machinery exists to
    prevent. Drive the real ``intercept_errors`` decorator instead.
    """
    pytest.importorskip("daytona")
    import httpx
    from daytona._utils.errors import intercept_errors

    from benchflow.sandbox.daytona_pty import stamp_transient_transport

    @stamp_transient_transport
    @intercept_errors(message_prefix="Failed to execute session command: ")
    async def execute_session_command():
        # httpx raises its timeouts with an empty message; this is what makes
        # the vendor error detail-free in the first place.
        raise httpx.ReadTimeout("")

    with pytest.raises(TransientSandboxTransportError) as excinfo:
        await execute_session_command()

    message = str(excinfo.value)

    assert "Timeout" in message, "the vendor class must survive into the message"
    assert "(no detail)" in message
    assert classify_error(message) == INFRA_ERROR
    assert RetryConfig().should_retry(message)

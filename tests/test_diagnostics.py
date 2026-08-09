"""Unit tests for the diagnostics helpers in :mod:`benchflow.diagnostics`."""

from __future__ import annotations

from benchflow.diagnostics import describe_exception


class _SdkError(Exception):
    """Stand-in for ``daytona.common.errors.DaytonaError``.

    The real SDK type is an optional dependency; what matters here is the
    shape it presents — a message plus optional ``status_code`` /
    ``error_code`` attributes.
    """

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

    described = describe_exception(exc)

    assert described == ("_SdkError: Failed to execute session command: (no detail)")


def test_describe_exception_handles_a_wholly_empty_message():
    assert describe_exception(TimeoutError()) == "TimeoutError (no message)"


def test_describe_exception_appends_structured_http_fields():
    exc = _SdkError("boom", status_code=503, error_code="unavailable")

    described = describe_exception(exc)

    assert described == ("_SdkError: boom [status_code=503, error_code=unavailable]")


def test_describe_exception_omits_unset_structured_fields():
    assert describe_exception(_SdkError("boom")) == "_SdkError: boom"

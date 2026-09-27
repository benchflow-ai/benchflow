"""Sandbox startup failures that another attempt cannot fix are not retried.

Regression test: a Dockerfile `COPY` of a path missing from the build context failed on
Daytona with `Path does not exist: …`, classified `sandbox_setup`, and was
retried twice more with backoff (BuildKit reports the same case as `failed to
compute cache key … not found`). The category stays
`sandbox_setup` (dashboards and the startup-failure summary key off it); only
the retry decision changes. Transient startup failures still retry.
"""

from __future__ import annotations

import pytest

from benchflow._utils.scoring import SANDBOX_SETUP, classify_error
from benchflow.evaluation import RetryConfig

# Error messages in the shape Daytona returns.
DAYTONA_MISSING_CONTEXT = (
    "Sandbox startup failed: Sandbox creation failed after 1 attempt: Failed to "
    "create sandbox: Path does not exist: tasks/missing-ctx/environment/data/"
)
BUILDKIT_MISSING_COPY_SOURCE = (
    "Sandbox startup failed: Sandbox creation failed after 1 attempt: Failed to "
    "create sandbox: Sandbox build failed: BUILD_FAILED, error reason: failed to "
    "compute cache key: failed to calculate checksum of ref "
    '00000000-0000-0000-0000-000000000003::demo0000000000000000000001: "/corpus": '
    "not found"
)


@pytest.mark.parametrize(
    "error", [DAYTONA_MISSING_CONTEXT, BUILDKIT_MISSING_COPY_SOURCE]
)
def test_missing_build_context_is_not_retried(error):
    assert classify_error(error) == SANDBOX_SETUP
    assert RetryConfig().should_retry(error) is False


def test_transient_sandbox_startup_failure_still_retries():
    error = (
        "Sandbox startup failed: Sandbox creation failed after 3 attempts: "
        "Failed to create sandbox: 503 Service Unavailable"
    )
    assert classify_error(error) == SANDBOX_SETUP
    assert RetryConfig().should_retry(error) is True

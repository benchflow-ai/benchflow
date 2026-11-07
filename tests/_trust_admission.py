"""An Admission for a test that calls a backend's transfer primitive itself."""

from __future__ import annotations

from typing import Any

from benchflow.trust.state import TRUSTED_TRANSFER, Admission, admit


def admitted(sandbox: Any, service: str = "main") -> Admission:
    """What ``trust.admit()`` returns for the test's own bytes."""
    return admit(sandbox, service, TRUSTED_TRANSFER, reason="the test's own bytes")

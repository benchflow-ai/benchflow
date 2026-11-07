"""ACP transport selection helpers shared by runtime and artifacts.

A review's own rollout (``review.runner.run_review``: the in-run review,
``bench eval score`` and the review resume) picks its Daytona transport apart
from the solver's: ssh unless ``BENCHFLOW_REVIEWER_ACP_TRANSPORT=pty`` (or the
machine has no ssh client and the variable is unset). On the
PTY websocket a burst of agent output can end the session (the Daytona side
closes it without a close frame, 1006), and reviewers print large outputs:
evidence manifests, trajectory lines. Of 69 reviewer sessions on the PTY, 18
ended so; none of 32 over ssh did. Every other rollout keeps
``BENCHFLOW_DAYTONA_ACP_TRANSPORT``.
"""

import contextlib
import contextvars
import logging
import os
import shutil
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_DAYTONA_ACP_TRANSPORT_ENV = "BENCHFLOW_DAYTONA_ACP_TRANSPORT"
_REVIEWER_ACP_TRANSPORT_ENV = "BENCHFLOW_REVIEWER_ACP_TRANSPORT"
_DAYTONA_ACP_TRANSPORTS = {"pty", "ssh"}
# True inside a review's own rollout (review_rollout).
_REVIEW_ROLLOUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_review_rollout", default=False
)
# Set once the missing ssh client is reported: every selection asks again.
_no_ssh_client_logged = False


def daytona_acp_transport() -> str:
    value = os.environ.get(_DAYTONA_ACP_TRANSPORT_ENV, "pty").strip().lower()
    if value in _DAYTONA_ACP_TRANSPORTS:
        return value
    logger.warning(
        "Invalid %s=%r; using PTY transport",
        _DAYTONA_ACP_TRANSPORT_ENV,
        value,
    )
    return "pty"


def reviewer_acp_transport() -> str:
    """A review's own Daytona transport: ssh, or pty when asked. Unasked,
    a machine without an ssh client to run the launch keeps the PTY."""
    global _no_ssh_client_logged
    raw = os.environ.get(_REVIEWER_ACP_TRANSPORT_ENV, "").strip().lower()
    if raw in _DAYTONA_ACP_TRANSPORTS:
        return raw
    if raw:
        logger.warning(
            "Invalid %s=%r; using SSH transport for the reviewer",
            _REVIEWER_ACP_TRANSPORT_ENV,
            raw,
        )
    if shutil.which("ssh") is not None:
        return "ssh"
    if not _no_ssh_client_logged:
        _no_ssh_client_logged = True
        logger.warning(
            "No ssh client on this machine: Daytona reviewers keep the PTY "
            "transport (install OpenSSH, or set %s=pty to say so)",
            _REVIEWER_ACP_TRANSPORT_ENV,
        )
    return "pty"


@contextlib.contextmanager
def review_rollout() -> Iterator[None]:
    """Run the enclosed rollout as a review's own: its agent connects over
    :func:`reviewer_acp_transport`, and its ``config.json`` says so."""
    token = _REVIEW_ROLLOUT.set(True)
    try:
        yield
    finally:
        _REVIEW_ROLLOUT.reset(token)


def in_review_rollout() -> bool:
    return _REVIEW_ROLLOUT.get()


def selected_acp_transport(*, agent: str, environment: str) -> str:
    """Return the concrete agent transport selected for artifact provenance."""
    if environment in ("docker", "remote-docker"):
        return "docker-stdio"
    if environment == "daytona":
        if agent == "gemini":
            return "ssh"
        if in_review_rollout():
            return reviewer_acp_transport()
        return daytona_acp_transport()
    return "provider-default"

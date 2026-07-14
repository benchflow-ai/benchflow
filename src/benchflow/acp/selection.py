"""Pure ACP transport selection helpers shared by runtime and artifacts."""

import logging
import os

logger = logging.getLogger(__name__)

_DAYTONA_ACP_TRANSPORT_ENV = "BENCHFLOW_DAYTONA_ACP_TRANSPORT"
_DAYTONA_ACP_TRANSPORTS = {"pty", "ssh"}


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


def selected_acp_transport(*, agent: str, environment: str) -> str:
    """Return the concrete agent transport selected for artifact provenance."""
    if environment == "docker":
        return "docker-stdio"
    if environment == "daytona":
        return "ssh" if agent == "gemini" else daytona_acp_transport()
    return "provider-default"

"""Server identity fixtures for the RD-CONF-2/5 refactor at ccba96d7."""

import hashlib


def server_id(key):
    # Test daemons are selected by endpoint/context and config, never by the
    # incidental shim directory or the rest of the process environment.
    selector = (
        key.endpoint_hint or key.env.get("DOCKER_HOST", ""),
        key.env.get("DOCKER_CONTEXT", ""),
        key.env.get("DOCKER_CONFIG", ""),
        key.env.get("DOCKER_CERT_PATH", ""),
    )
    return hashlib.sha256(repr(selector).encode()).hexdigest()

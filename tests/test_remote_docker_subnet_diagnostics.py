"""Guards RD-CONF-4 at 39b56cbc: daemon data never becomes a display label."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult
from tests.test_remote_docker_isolation import _certs, _sandbox


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize(
    "name", ["private.internal.test", "single-private", "192.0.2.71", "2001:db8::71"]
)
async def test_subnet_fallback_uses_provider_display(
    tmp_path, monkeypatch, caplog, scheme, show, name
):
    """Guards RD-CONF-4: successful info stdout remains private internal data."""
    subnets.reset_allocators()
    monkeypatch.setenv(
        "BENCHFLOW_REMOTE_DOCKER_HOST", f"{scheme}://builder.internal.test"
    )
    if scheme == "tcp":
        monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
        monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "dummy-certs")))
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", "1" if show else "0")
    monkeypatch.delenv(subnets.ENV_VAR, raising=False)
    monkeypatch.setattr(subnets, "_read_daemon_id", lambda key: "dummy-daemon")
    monkeypatch.setattr(subnets, "daemon_on_this_host", lambda actual: False)
    sandbox = _sandbox(tmp_path)
    sandbox._docker_cli_unchecked = AsyncMock(
        return_value=ExecResult(
            stdout=json.dumps({"name": name, "pools": []}), stderr="", return_code=0
        )
    )
    try:
        with caplog.at_level(logging.INFO, logger="benchflow"):
            await sandbox._choose_subnets()
        assert "lets it choose" in caplog.text
        assert name not in caplog.text
        assert sandbox.docker_host.display in caplog.text
        assert ("builder.internal.test" in caplog.text) == show
    finally:
        subnets.reset_allocators()

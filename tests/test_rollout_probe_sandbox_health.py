"""Tests for Rollout._probe_sandbox_health diagnostics (#502)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.diagnostics import RolloutDiagnostics, TransportClosedDiagnostic
from benchflow.rollout import Rollout


class _ProbeHarness:
    """Minimum surface so ``Rollout._probe_sandbox_health`` runs as an
    instance method without spinning up a full rollout."""

    _diagnostics: RolloutDiagnostics
    _env: object | None


def _harness_with_transport_diag() -> _ProbeHarness:
    harness = _ProbeHarness()
    harness._diagnostics = RolloutDiagnostics()
    harness._diagnostics.set(TransportClosedDiagnostic())
    return harness


@pytest.mark.asyncio
async def test_probe_sandbox_health_records_success_when_env_responds():
    """Sanity: when the probe succeeds we record sandbox_reachable=True."""
    harness = _harness_with_transport_diag()
    harness._env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                stdout="__BENCHFLOW_HEALTH_OK__\n", return_code=0
            )
        )
    )

    await Rollout._probe_sandbox_health(harness)  # type: ignore[arg-type]

    diag = harness._diagnostics.transport_closed
    assert diag is not None
    assert diag.sandbox_reachable is True
    assert diag.sandbox_probe_rc == 0


@pytest.mark.asyncio
async def test_probe_sandbox_health_records_type_and_traceback_on_failure(caplog):
    """Guards #502: when the probe itself raises, we must preserve

    - a logger.exception() entry (with traceback),
    - the exception type name on the transport diagnostic,
    - and a truncated traceback string.
    """

    class _ProbeError(RuntimeError):
        """Distinct type so we can assert the type name is recorded."""

    async def _boom(*_a, **_kw):
        raise _ProbeError("daytona session terminated unexpectedly")

    harness = _harness_with_transport_diag()
    harness._env = SimpleNamespace(exec=_boom)

    with caplog.at_level(logging.ERROR, logger="benchflow.rollout"):
        await Rollout._probe_sandbox_health(harness)  # type: ignore[arg-type]

    diag = harness._diagnostics.transport_closed
    assert diag is not None
    assert diag.sandbox_reachable is False
    assert diag.sandbox_probe_error == "daytona session terminated unexpectedly"
    assert diag.sandbox_probe_error_type == "_ProbeError"
    # Traceback is captured (not just the message) so post-mortem keeps the
    # original frame stack.
    assert diag.sandbox_probe_traceback is not None
    assert "_ProbeError" in diag.sandbox_probe_traceback
    assert "Traceback" in diag.sandbox_probe_traceback

    # logger.exception() emits a record with exc_info attached.
    failure_records = [
        rec
        for rec in caplog.records
        if rec.name == "benchflow.rollout" and rec.exc_info is not None
    ]
    assert failure_records, "expected logger.exception() to fire with exc_info"


@pytest.mark.asyncio
async def test_probe_sandbox_health_no_op_when_no_transport_error():
    """If no transport diagnostic has been captured, the probe is a no-op."""
    harness = _ProbeHarness()
    harness._diagnostics = RolloutDiagnostics()  # empty — no transport_closed
    harness._env = SimpleNamespace(exec=AsyncMock())

    await Rollout._probe_sandbox_health(harness)  # type: ignore[arg-type]

    assert harness._diagnostics.transport_closed is None
    assert not harness._env.exec.called  # type: ignore[union-attr]

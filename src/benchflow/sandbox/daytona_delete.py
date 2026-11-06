"""Confirm sandbox removal before releasing its ownership or replacing it.

Daytona 0.207 defaults to returning when deletion is merely accepted. Its
``wait=True`` contract waits for DESTROYED (including a 404 while polling).
Always pass an explicit timeout: the SDK's timeout decorator only observes
explicit arguments. A failed or cancelled wait must leave ownership intact.
"""

from __future__ import annotations

from typing import Any

DELETE_TIMEOUT_SEC = 60.0


def _already_gone(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or (
        type(exc).__name__ == "DaytonaNotFoundError"
    )


async def delete_sandbox(sandbox: Any) -> None:
    """Return only once the provider confirms removal; propagate other errors."""
    try:
        await sandbox.delete(timeout=DELETE_TIMEOUT_SEC, wait=True)
    except Exception as exc:
        if not _already_gone(exc):
            raise


def delete_sandbox_sync(
    client: Any, sandbox: Any, *, timeout: float | None = None
) -> None:
    """The same confirmation contract for the sync ledger and reapers."""
    timeout = (
        DELETE_TIMEOUT_SEC if timeout is None else min(timeout, DELETE_TIMEOUT_SEC)
    )
    if timeout <= 0:
        # The SDK interprets zero as unlimited, not an exhausted budget.
        raise TimeoutError("No time left to confirm Daytona sandbox deletion")
    try:
        client.delete(sandbox, timeout=timeout, wait=True)
    except Exception as exc:
        if not _already_gone(exc):
            raise


def list_sandboxes_sync(client: Any, query: Any, *, timeout: float) -> list[Any]:
    """Bound the whole paginated lookup, not a fresh timeout per page.

    Use the installed SDK's sync timeout scope, as its delete does. On Unix
    the SDK rounds signal timeouts up to a second; in other threads it cannot
    interrupt blocking C calls until they return. The ledger keeps ownership
    on failure and checks its absolute deadline before the next operation.
    """
    from daytona._utils.timeout import with_timeout

    if timeout <= 0:
        raise TimeoutError("No time left to look up Daytona sandbox deletion")

    @with_timeout()
    def collect(timeout: float) -> list[Any]:
        return list(client.list(query, request_timeout=timeout))

    return collect(timeout=timeout)

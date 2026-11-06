"""Confirm sandbox removal before releasing its ownership or replacing it.

Daytona 0.207 defaults to returning when deletion is merely accepted. Its
``wait=True`` contract waits for DESTROYED (including a 404 while polling).
Always pass an explicit timeout: the SDK's timeout decorator only observes
explicit arguments. A failed or cancelled wait must leave ownership intact.

A sandbox whose image Daytona is still building is never deleted. Deleting
it cancels the build, and Daytona keeps that build as failed ("BUILD_FAILED,
error reason: context canceled") under the image's build key: every later
create of the same image then fails within a second, and Daytona offers no
way to clear it. The delete raises :class:`SandboxStillBuilding` instead,
which callers handle as a delete that did not go through: the sandbox stays
theirs (the ledger deletes it once it has started), and Daytona's auto-stop
and auto-delete intervals remove it if nobody does.
"""

from __future__ import annotations

from typing import Any

DELETE_TIMEOUT_SEC = 60.0
# The states in which Daytona is still building the sandbox's image or
# putting it in place. Any other state that names a build counts too, but
# BUILD_FAILED is a finished build. A state this SDK cannot name does not:
# only a lookup by label can see a sandbox that has not started.
BUILD_STATES = frozenset({"PENDING_BUILD", "BUILDING_SNAPSHOT", "PULLING_SNAPSHOT"})


class SandboxStillBuilding(Exception):
    """A delete refused because Daytona is still building the sandbox's image."""

    def __init__(self, sandbox: Any) -> None:
        self.sandbox_id = getattr(sandbox, "id", None)
        self.state = sandbox_state(sandbox)
        super().__init__(
            f"Daytona sandbox {self.sandbox_id} is still building ({self.state}); "
            "not deleting it: a delete now would make Daytona keep its build "
            'as failed ("context canceled") for every later create of the '
            "same image. It can be deleted once it has started; Daytona's "
            "auto-stop and auto-delete intervals remove it otherwise"
        )


def sandbox_state(sb: Any) -> str:
    """The sandbox's state name in upper case (``STARTED``, ``STOPPED`` ...),
    whether the SDK gives an enum, its value or its repr."""
    raw = getattr(sb, "state", None)
    raw = getattr(raw, "value", raw)
    return str(raw or "").rsplit(".", 1)[-1].upper()


def still_building(sb: Any) -> bool:
    """Whether Daytona is still building *sb*'s image, by the state the
    handle carries (a lookup's or a list's: a handle a create returned has
    already started)."""
    state = sandbox_state(sb)
    return state in BUILD_STATES or ("BUILD" in state and state != "BUILD_FAILED")


def _already_gone(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or (
        type(exc).__name__ == "DaytonaNotFoundError"
    )


async def delete_sandbox(sandbox: Any) -> None:
    """Return only once the provider confirms removal; propagate other errors.

    A sandbox still building is not deleted: :class:`SandboxStillBuilding`.
    """
    if still_building(sandbox):
        raise SandboxStillBuilding(sandbox)
    try:
        await sandbox.delete(timeout=DELETE_TIMEOUT_SEC, wait=True)
    except Exception as exc:
        if not _already_gone(exc):
            raise


def delete_sandbox_sync(
    client: Any, sandbox: Any, *, timeout: float | None = None
) -> None:
    """The same confirmation contract for the sync ledger and reapers."""
    if still_building(sandbox):
        raise SandboxStillBuilding(sandbox)
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

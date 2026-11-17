"""All backend downloads publish through held, beneath-resolved descriptors."""

from __future__ import annotations

import asyncio
import inspect
import shlex
from functools import wraps
from pathlib import Path

from benchflow._utils.reachable_write import (
    ReachablePath,
    copy_mounted,
    destination,
    staging_for,
)


class RemoteFileAbsentError(FileNotFoundError):
    """The backend verified that the requested remote entry is absent."""


async def drain(work):
    """Await ``work`` until it has really completed, whatever cancels arrive.

    Cancelling ``to_thread`` does not stop its thread, and a second
    ``cancel()`` interrupts a single re-await (TASKMD-R15 P-M1): every
    cancellation request is absorbed until the worker is done, then one
    ``CancelledError`` is raised. Without a cancellation the worker's own
    result or exception is returned or raised.
    """
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            # ``wait`` never cancels the task it waits on.
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            cancelled = exc
    if cancelled is not None:
        if not task.cancelled() and task.exception() is not None:
            cancelled.__context__ = task.exception()
        raise cancelled
    return task.result()


async def join_publication(work) -> None:
    """Await a publication thread; on cancellation, join it before returning.

    Returning at once would release staging while the copier still runs, and
    the copier would change the destination after the caller saw the
    cancellation (TASKMD-R14 P-M1, R15 repeated cancellation).
    """
    await drain(work)


def private_download(*, directory: bool):
    """Keep provider/CLI writes off every controller path a sandbox might reach.

    whether a sandbox can reach the destination is never asked.
    The destination's trusted root is canonicalised once and everything below
    it is opened without following a link or crossing a mount. A backend
    that can stream (``_stream_download``: Docker, the backend that binds
    arbitrary host folders) publishes straight from the transport through
    those descriptors and stages nothing. Any other transport writes into a
    fresh mode-700 folder in the controller's temporary folder, which no
    remote or Apple sandbox mounts, and the no-follow copier publishes it.
    """
    target_parameter = "target_dir" if directory else "target_path"
    source_parameter = "source_dir" if directory else "source_path"

    def decorate(method):
        signature = inspect.signature(method)

        @wraps(method)
        async def download(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            requested = bound.arguments[target_parameter]
            target = (
                requested if isinstance(requested, ReachablePath) else Path(requested)
            )
            stream = getattr(args[0], "_stream_download", None)
            if callable(stream):
                try:
                    capability = destination(target)
                    await stream(
                        bound.arguments[source_parameter],
                        capability,
                        directory=directory,
                        service=bound.arguments.get("service", "main"),
                    )
                except RemoteFileAbsentError:
                    raise
                except FileNotFoundError as exc:
                    # A missing CLI executable or controller path is not a
                    # missing sandbox file, and must not omit a companion.
                    raise RuntimeError(
                        "file capture filesystem operation failed"
                    ) from exc
                return None
            with staging_for(target) as root:
                bound.arguments[target_parameter] = root / "payload"
                result = await method(*bound.args, **bound.kwargs)
                await join_publication(
                    asyncio.to_thread(
                        copy_mounted, root, ("payload",), target, directory=directory
                    )
                )
                return result

        return download

    return decorate


# `docker system dial-stdio` with the request on stdin, the write side held
# open until the daemon has answered and closed (Connection: close). Fed
# straight from a pipe, dial-stdio half-closes the connection at the
# request's end, and Docker Engine 29.9 reads that as the client going away:
# a missing path's 404 comes back as 499 ("request cancelled by client"), so
# a download's absence check failed with "archive metadata failed (HTTP
# 499)". Engine 29.1 answered 404 either way. A fifo keeps the write side open;
# `sh -c` so the snippet runs the same under a VM's /bin/sh -c or alone
# (for Daytona's DinD VM; for DockerSandbox).
_DIAL_STDIO_HELD = (
    'd=$(mktemp -d) && mkfifo "$d/in" || exit 1; '
    '{dial} < "$d/in" & pid=$!; '
    'exec 3>"$d/in"; cat >&3; '
    'wait "$pid"; rc=$?; exec 3>&-; rm -rf "$d"; exit "$rc"'
)
#: For a shell (Daytona's DinD VM): its own ``docker system dial-stdio``.
DIAL_STDIO_HELD = "sh -c " + shlex.quote(
    _DIAL_STDIO_HELD.format(dial="docker system dial-stdio")
)


def dial_stdio_held(dial: list[str]) -> list[str]:
    """*dial* (``docker <daemon flags> system dial-stdio``) as an argv whose
    standard input is held open the same way (:data:`DIAL_STDIO_HELD`)."""
    return ["sh", "-c", _DIAL_STDIO_HELD.format(dial='"$@"'), "sh", *dial]

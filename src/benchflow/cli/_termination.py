"""Run a CLI coroutine so that SIGTERM tears it down instead of killing it.

CI cancellation and job timeouts send SIGTERM. Python's default for SIGTERM
is to die at once, so a ``bench eval run`` in the middle of a trial left its
Daytona sandbox running. :func:`run_until_terminated` handles
the first SIGTERM the way ``asyncio.run`` handles Ctrl-C: it cancels the
running work, whose ``finally`` blocks (rollout cleanup, sandbox deletion)
then run, and exits with 143, the conventional code for SIGTERM. A second
signal is ignored while cleanup runs; a third forces the exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import threading
from collections.abc import Coroutine
from typing import Any

TERMINATED_EXIT_CODE = 128 + signal.SIGTERM


class _Terminated(BaseException):
    """Raised in the main thread by the SIGTERM handler."""


def run_until_terminated[T](main: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run(main)``, with SIGTERM turned into an orderly teardown.

    Only installs the handler in the main thread (signals cannot be handled
    elsewhere); off the main thread it is plain ``asyncio.run``.
    """
    if threading.current_thread() is not threading.main_thread():
        return asyncio.run(main)

    received = 0

    def _on_sigterm(signum: int, frame: Any) -> None:
        nonlocal received
        received += 1
        if received == 1:
            raise _Terminated
        if received >= 3:
            os._exit(TERMINATED_EXIT_CODE)

    previous = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        return asyncio.run(main)
    except _Terminated:
        # asyncio.run's runner has cancelled every task and waited for its
        # finally blocks (sandbox cleanup) before this propagates.
        with contextlib.suppress(Exception):
            print(
                "Terminated (SIGTERM): the running trials were cancelled and their "
                "sandboxes cleaned up.",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(TERMINATED_EXIT_CODE) from None
    finally:
        signal.signal(signal.SIGTERM, previous)

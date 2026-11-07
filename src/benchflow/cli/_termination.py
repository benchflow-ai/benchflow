"""Run a CLI coroutine so that SIGTERM tears it down instead of killing it.

CI cancellation and job timeouts send SIGTERM. Python's default for SIGTERM
is to die at once, so a ``bench eval run`` in the middle of a trial left its
Daytona sandbox running. :func:`run_until_terminated` handles
the first SIGTERM the way ``asyncio.run`` handles Ctrl-C: it cancels the
running work, whose ``finally`` blocks (rollout cleanup, sandbox deletion)
then run, and exits with 143, the conventional code for SIGTERM. While the
cleanup runs, a second SIGTERM (or a Ctrl-C) is ignored and a third forces
the exit; a second Ctrl-C after a Ctrl-C forces it at once, as asyncio
does. A SIGHUP never forces: one hangup arrives as several (the kernel's,
then the shell's resend to its jobs), and a sharded worker that counted
them with its parent's SIGTERM force-exited without its cleanup.

SIGHUP (the terminal closed, the SSH session dropped) is handled as SIGTERM,
with 129. A SIGHUP the process inherited as ignored (``nohup``)
stays ignored, so the run goes on. A hangup means the terminal is going
away (already hung up, or about to be once its window closes), so the
standard streams that point at a terminal are moved to ``/dev/null``
first: a write to a hung-up terminal fails with EIO, and that failure would
replace the orderly stop (or fail the final flush, exit 120). A file is
kept, and so is a pipe until its reader is gone (``bench … | tee`` in the
closed terminal): a stream whose reader went away moves to /dev/null too,
and the final flush cannot turn the 129 into a 120.

Ctrl-C (SIGINT) goes through the same handler, so it cancels the work the
same way. An ignored SIGINT stays ignored (``… &`` from a script),
and a SIGINT handler the caller installed is left to it. Either
way, once the cleanup has run, one line says so on stderr and
:class:`RunStopped` (a ``SystemExit`` with 129, 130 or 143) ends the
process. A
command that says more (what it stopped, how to resume it) passes
``announce=False`` and prints its own line (DF-p5-cli-craft-10).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import errno
import os
import select
import signal
import stat
import sys
import threading
from collections.abc import Coroutine
from typing import Any

# None where the platform has no SIGHUP (Windows).
_SIGHUP: int | None = getattr(signal, "SIGHUP", None)

TERMINATED_EXIT_CODE = 128 + signal.SIGTERM
INTERRUPTED_EXIT_CODE = 128 + signal.SIGINT
HANGUP_EXIT_CODE = 128 + 1  # SIGHUP


class RunStopped(SystemExit):
    """The run stopped on Ctrl-C, SIGTERM or SIGHUP, after its cleanup ran.

    A ``SystemExit`` with the signal's exit code: no ``except Exception``
    on the way out reads it as a failure, and uncaught it exits quietly.
    """

    def __init__(self, how: str, exit_code: int) -> None:
        super().__init__(exit_code)
        self.how = how
        self.exit_code = exit_code

    def line(self) -> str:
        """The general line: what stopped, and that the sandboxes were removed."""
        return (
            f"{self.how}: the running trials were stopped and their sandboxes removed."
        )


def _leave_gone_streams(*, terminals: bool) -> None:
    """Point stdin, stdout and stderr at /dev/null where nothing reads them.

    With ``terminals`` (a hangup), a terminal, still open or already hung up
    (it answers EIO): the terminal may hang up only after the SIGHUP (the
    shell died first, then the window closed), so an open one is not spared.
    And always stdout or stderr on a pipe or socket whose reader is gone
    (``poll`` reports an error), as when ``tee`` died with the terminal. A
    file, a pipe still read and stdin on a pipe are left alone.
    """
    try:
        import termios
    except ImportError:  # Windows: no terminals to hang up
        return
    null = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            try:
                termios.tcgetattr(fd)
            except termios.error as exc:
                is_terminal = bool(exc.args) and exc.args[0] == errno.EIO
            except OSError:
                continue
            else:
                is_terminal = True
            if is_terminal:
                if terminals:
                    os.dup2(null, fd)
                continue
            if fd != 0 and _reader_gone(fd):
                os.dup2(null, fd)
    finally:
        os.close(null)


def _reader_gone(fd: int) -> bool:
    """A pipe or socket write end whose reader has closed."""
    try:
        mode = os.fstat(fd).st_mode
    except OSError:
        return False
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
        return False
    poller = select.poll()
    poller.register(fd, select.POLLOUT)
    events = poller.poll(0)
    return any(ev & (select.POLLERR | select.POLLHUP) for _, ev in events)


def _flush_or_drop_streams() -> None:
    """At exit after a stop: a final flush that fails goes to /dev/null.

    Python exits 120 when flushing stdout or stderr fails at shutdown; a run
    stopped by a signal keeps its own exit code instead.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.flush()
        except (OSError, ValueError):
            with contextlib.suppress(OSError, ValueError):
                null = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(null, stream.fileno())
                finally:
                    os.close(null)
                stream.flush()


def run_until_terminated[T](
    main: Coroutine[Any, Any, T], *, announce: bool = True
) -> T:
    """``asyncio.run(main)``, with SIGTERM and SIGHUP turned into an orderly teardown.

    The first SIGINT, SIGTERM or SIGHUP cancels the main task, whose
    ``finally`` blocks (rollout cleanup, sandbox deletion, the warm pool's
    drain) then run. While that stop is under way: a SIGHUP is ignored, a
    Ctrl-C after a Ctrl-C forces the exit at once (KeyboardInterrupt, as
    asyncio does), and any other signal is ignored once and forces the exit
    the second time. Only installs the handlers in the main thread (signals
    cannot be handled elsewhere); off the main thread it is plain
    ``asyncio.run``.
    """
    if threading.current_thread() is not threading.main_thread():
        return asyncio.run(main)

    task: asyncio.Task[T] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    stopped_by: int | None = None
    extra = 0

    async def _main() -> T:
        nonlocal task
        task = asyncio.current_task()  # type: ignore[assignment]
        if stopped_by is not None:
            main.close()
            raise asyncio.CancelledError
        return await main

    def _on_signal(signum: int, frame: Any) -> None:
        nonlocal stopped_by, extra
        if signum == _SIGHUP:
            _leave_gone_streams(terminals=True)
        if stopped_by is None:
            stopped_by = signum
            if task is not None and loop is not None:
                task.cancel()
                loop.call_soon_threadsafe(lambda: None)
            return
        # A stop is under way.
        if signum == _SIGHUP:
            return  # one hangup, signalled again (the shell resends it)
        if signum == signal.SIGINT and stopped_by == signal.SIGINT:
            raise KeyboardInterrupt  # Ctrl-C twice: stop now, as asyncio does
        extra += 1
        if extra >= 2:
            os._exit(128 + stopped_by)

    previous: dict[int, Any] = {
        signal.SIGTERM: signal.signal(signal.SIGTERM, _on_signal)
    }
    # nohup (an inherited ignored SIGHUP) keeps the run going.
    if _SIGHUP is not None and signal.getsignal(_SIGHUP) is not signal.SIG_IGN:
        previous[_SIGHUP] = signal.signal(_SIGHUP, _on_signal)
    # Ctrl-C only where Python's own handler has it: an ignored SIGINT
    # or the caller's handler stays. asyncio's Runner then leaves
    # it to this one.
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, _on_signal)
    try:
        try:
            with asyncio.Runner() as runner:
                loop = runner.get_loop()
                try:
                    result = runner.run(_main())
                    if stopped_by is None:
                        return result
                except asyncio.CancelledError:
                    if stopped_by is None:
                        raise
            # The runner has closed: every other task is cancelled and done.
        except KeyboardInterrupt:
            # A second Ctrl-C forced the stop (or a caller's own SIGINT
            # handler raised it); the runner has cancelled what was left.
            stopped_by = signal.SIGINT
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    # The final flush must not turn the stop's exit code into 120.
    atexit.register(_flush_or_drop_streams)
    if stopped_by == _SIGHUP:
        _leave_gone_streams(terminals=False)  # a pipe whose reader died since
    if stopped_by == signal.SIGINT:
        stop = RunStopped("Interrupted (Ctrl-C)", INTERRUPTED_EXIT_CODE)
    elif stopped_by == _SIGHUP:
        stop = RunStopped("Hung up (SIGHUP)", HANGUP_EXIT_CODE)
    else:
        stop = RunStopped("Terminated (SIGTERM)", TERMINATED_EXIT_CODE)
    if announce:
        with contextlib.suppress(Exception):
            print(stop.line(), file=sys.stderr, flush=True)
    raise stop

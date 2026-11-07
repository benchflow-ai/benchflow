"""The terminal helper behind dx-a's Ctrl-C test gives its child Ctrl-C (CTRLC-1).

A non-interactive shell starts an ``&`` job with SIGINT and SIGQUIT ignored,
an ignored signal stays ignored across exec, and Python then installs no
KeyboardInterrupt handler. dx-a's ``test_ctrl_c_says_what_it_stopped_and_how_to_resume``
failed 18/18 when its suite was launched that way: the run never saw the
Ctrl-C, finished its 120 s solve and exited 0. ``_pty_run`` now starts the
child as a shell starts a foreground job. No Docker needed.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

from tests.test_dx_a_container import _pty_run

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="pty and POSIX signals")

_PROBE = (
    "import signal, time\n"
    "print('handler', signal.getsignal(signal.SIGINT) is signal.default_int_handler, "
    "flush=True)\n"
    "try:\n"
    "    time.sleep(20)\n"
    "except KeyboardInterrupt:\n"
    "    raise SystemExit(130)\n"
    "raise SystemExit(0)\n"
)


def test_the_child_gets_ctrl_c_even_when_the_suite_ignores_it(tmp_path: Path):
    previous = {
        s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGQUIT)
    }
    try:
        code, out = _pty_run(
            [sys.executable, "-c", _PROBE], tmp_path, {"PATH": "/usr/bin:/bin"}, after=2
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert "handler True" in out, out
    assert code == 130, out

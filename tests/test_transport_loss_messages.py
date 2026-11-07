"""A lost Daytona PTY and a lost agent process read as what they are.

Guards the dx/errors check of #1141/#1144's messages. The Daytona PTY's
texts ("PTY closed by the peer: agent process exited with code N" /
"...: websocket closed (close_code=...)", "PTY readline timeout (Ns)", the
start-marker timeout) are stored as pipe_closed through their transport
diagnostic, but read as ``other`` from their text alone (a result without a
stored category, a summary or resume rebuilt from text); the idle watchdog's
"Agent idle for ..." stays idle_timeout.
"""

from __future__ import annotations

import pytest

from benchflow._utils.scoring import IDLE_TIMEOUT, PIPE_CLOSED, classify_error


@pytest.mark.parametrize(
    "error",
    [
        "PTY closed by the peer: agent process exited with code 137",
        "PTY closed by the peer: websocket closed (close_code=1006)",
        "PTY closed",
        "PTY readline timeout (900s)",
        "PTY readline error: ConnectionClosedError()",
        "DaytonaPtyProcess: timeout waiting for agent start marker (session=acp-1)",
    ],
)
def test_the_daytona_pty_texts_are_a_lost_pipe(error):
    assert classify_error(error) == PIPE_CLOSED


def test_the_watchdog_stays_the_watchdog():
    assert (
        classify_error(
            "Agent idle for 600s with no new tool call, message, or thought "
            "(last activity 601s ago, 3 tool calls so far)"
        )
        == IDLE_TIMEOUT
    )

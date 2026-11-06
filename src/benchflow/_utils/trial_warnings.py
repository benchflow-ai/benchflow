"""Warnings a trial records in its result.json (``warnings``).

A step deep in the rollout reports a warning with :func:`report_trial_warning`;
the rollout collects them with :func:`recording_trial_warnings` around the
phase and writes them next to its result. A warning never changes a score.
It names what happened and what a task author can do, never a value the
image or task set (those can hold secrets).
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "benchflow_trial_warnings_sink", default=None
)


@contextlib.contextmanager
def recording_trial_warnings() -> Iterator[list[str]]:
    """Collect the warnings reported inside, in order."""
    warnings: list[str] = []
    token = _SINK.set(warnings)
    try:
        yield warnings
    finally:
        _SINK.reset(token)


def report_trial_warning(text: str) -> None:
    """Hand *text* to the innermost ``recording_trial_warnings()``, if any."""
    sink = _SINK.get()
    if sink is not None and text not in sink:
        sink.append(text)

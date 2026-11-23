"""Warnings a trial records in its result.json (``warnings``).

A step deep in the rollout reports a warning with :func:`report_trial_warning`;
the rollout collects them with :func:`recording_trial_warnings` around the
phase and writes them next to its result. A warning never changes a score.
A step may also report a named record (:func:`report_trial_record`), a
result.json field of its own beside the warning that flags it.
It names what happened and what a task author can do, never a value the
image or task set (those can hold secrets).
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any


class TrialWarnings(list[str]):
    """The warnings a phase reported, in order, and its named records."""

    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, Any] = {}


_SINK: contextvars.ContextVar[TrialWarnings | None] = contextvars.ContextVar(
    "benchflow_trial_warnings_sink", default=None
)


@contextlib.contextmanager
def recording_trial_warnings() -> Iterator[TrialWarnings]:
    """Collect the warnings and records reported inside."""
    warnings = TrialWarnings()
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


def report_trial_record(name: str, value: Any) -> None:
    """Hand the record *name* to the innermost ``recording_trial_warnings()``,
    if any; a later one of the same name replaces it."""
    sink = _SINK.get()
    if sink is not None:
        sink.records[name] = value


def keep_trial_warnings(owner: Any, warnings: TrialWarnings) -> None:
    """Keep *warnings* and their records on *owner* (a rollout) for its
    result.json: ``_warnings`` and ``_trial_records``."""
    # vars(): a Rollout built without __init__ (tests) has none.
    kept = vars(owner).setdefault("_warnings", [])
    kept.extend(w for w in warnings if w not in kept)
    vars(owner).setdefault("_trial_records", {}).update(warnings.records)

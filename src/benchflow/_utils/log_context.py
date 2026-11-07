"""Which task a log line belongs to, and the HTTP client's request lines.

A trial runs inside :func:`trial_log_context` (``Rollout.run``), so every
line it logs, from any module, knows its task. The CLI prints the task in
front of the line (``[fix-git] Running verifier...``): at ``-j 4`` four
trials log at once, and a line without its task cannot be told apart.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
from collections.abc import Iterator

_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchflow_log_task", default=None
)
_OWN_REQUEST: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_own_http_request", default=False
)


@contextlib.contextmanager
def trial_log_context(task: str) -> Iterator[None]:
    """Mark the lines logged inside as *task*'s."""
    token = _TASK.set(task)
    try:
        yield
    finally:
        _TASK.reset(token)


def current_task() -> str | None:
    """The task whose trial is logging here, or None outside a trial."""
    return _TASK.get()


def task_prefix(task: str | None) -> str:
    return f"[{task}] " if task else ""


class TaskPrefixFormatter(logging.Formatter):
    """A formatter that puts ``[task] `` in front of a trial's message.

    A handler formats a record when it is logged, in the trial's own context,
    so the task read here is the one that logged it.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        prefix = task_prefix(current_task())
        if prefix:
            record = copy.copy(record)
            # Every line of a multi-line message: its second line is the
            # same trial's, not a line of its own.
            record.message = prefix + record.message.replace("\n", "\n" + prefix)
        return super().formatMessage(record)


def quiet_http_logs() -> None:
    """httpx's "HTTP Request: ..." lines only at DEBUG.

    Left alone when the caller set httpx's level, or logs at DEBUG.
    """
    http = logging.getLogger("httpx")
    if http.level == logging.NOTSET and (
        logging.getLogger().getEffectiveLevel() > logging.DEBUG
    ):
        http.setLevel(logging.WARNING)


class _OwnRequestFilter(logging.Filter):
    """Drops httpx's INFO lines for BenchFlow's own requests
    (:func:`quiet_own_requests`) unless the root logs at DEBUG or the caller
    set httpx's level; every other request's line passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            _OWN_REQUEST.get()
            and record.levelno < logging.WARNING
            and logging.getLogger("httpx").level == logging.NOTSET
            and logging.getLogger().getEffectiveLevel() > logging.DEBUG
        )


_OWN_REQUEST_FILTER = _OwnRequestFilter()


@contextlib.contextmanager
def quiet_own_requests() -> Iterator[None]:
    """httpx's "HTTP Request: ..." lines for the requests made inside only at
    DEBUG, through the Python API too. The httpx logger's level is never
    changed, so the host process's own requests still log as it set them."""
    http = logging.getLogger("httpx")
    if _OWN_REQUEST_FILTER not in http.filters:
        http.addFilter(_OWN_REQUEST_FILTER)
    token = _OWN_REQUEST.set(True)
    try:
        yield
    finally:
        _OWN_REQUEST.reset(token)

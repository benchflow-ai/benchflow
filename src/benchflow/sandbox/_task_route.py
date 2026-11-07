"""C8's task route, without the sandbox package's imports.

The task's own commands (test.sh and the verifier's runner, the oracle's
solve.sh, setup and healthcheck commands, manifest services, SDK hooks and
``Runtime.exec``) run inside :func:`as_task_command`; everything else
BenchFlow runs as root or as the image's user is plumbing (see
``benchflow.sandbox._base``). Callers in ``benchflow.task`` import this module,
which ``_base`` itself depends on through ``benchflow.task.config``.

The mark is the current task's, not one sandbox object's, so a command sent
through a wrapper (a separate verifier's, an SDK ``Environment``) keeps it.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any

_TASK_COMMAND: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_task_command", default=False
)


@contextlib.contextmanager
def as_task_command(sandbox: Any = None) -> Iterator[None]:
    """Mark the execs inside as the task's own command (C8's task route).

    *sandbox* only documents which sandbox the command is for.
    """
    token = _TASK_COMMAND.set(True)
    try:
        yield
    finally:
        _TASK_COMMAND.reset(token)


def in_task_command() -> bool:
    """Whether the current task is running the task's own command."""
    return _TASK_COMMAND.get()

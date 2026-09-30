"""The errors BenchFlow expects, and how the CLI reports them.

A failure reaches a developer in one of two ways:

* **Expected**: the input or the setup is wrong, and the developer can fix
  it: a task file that does not parse, a bad flag, a missing file or login,
  a login whose usage limit is spent. These raise a :class:`UserError`. The
  CLI prints the message and its next step, with no traceback, and exits 1.
* **Unexpected**: a bug. The CLI prints the traceback and the path of a log
  with the run's recent log lines, and exits 1.

``UserError`` is a mixin: the existing error classes keep their built-in
base (``ValueError``, ``FileNotFoundError``) so ``except ValueError`` still
catches them, and gain ``UserError`` as a second base.

Each error names whose fault it was, in the same words the end-of-run
summary uses (see :mod:`benchflow.failures`): ``setup`` (your machine,
logins or flags), ``task``, ``agent`` or ``infrastructure``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

Fault = Literal["setup", "task", "agent", "infrastructure"]


class UserError(Exception):
    """An expected error the user can fix: the CLI prints one message, no traceback.

    ``hint`` is the next step (a command or a fix), printed after the message;
    ``fault`` says whose fault it was.
    """

    fault: ClassVar[Fault] = "setup"
    hint: str | None = None


class MissingCredentialError(ValueError, UserError):
    """The agent's model needs a login or an API key that is not set."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint or "`bench doctor` lists the logins and keys it finds"


def user_message(exc: BaseException) -> str:
    """One line for an expected error: its message, then its next step."""
    text = str(exc).strip() or type(exc).__name__
    hint = getattr(exc, "hint", None)
    if hint and hint not in text:
        text = f"{text}\n  next: {hint}"
    return text

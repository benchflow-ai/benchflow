"""Fake sandboxes answer a receipt-wrapped exec as the sandbox's shell would.

Lockdown reads what it decides on through ``_exec_receipt``: the command it
sends is the original one wrapped in the receipt script. A fake that answers
by the original command calls :func:`inner_command` first, then
:func:`receipt_answer` to print what the script prints for that answer.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from benchflow.sandbox._exec_receipt import (
    encode_receipt,
    receipt_command,
    unwrap_receipt_command,
)


def _unwrap(command: Any) -> tuple[str, str] | None:
    """:func:`unwrap_receipt_command`, also after the lines root's route puts
    before a command (C8: the PATH pin and unsets of PT's plumbing)."""
    if not isinstance(command, str):
        return None
    unwrapped = unwrap_receipt_command(command)
    if unwrapped is not None:
        return unwrapped
    # The receipt script's own first line, found by its nonce: root's route
    # puts its PATH pin, and the task route the image's values, before it.
    found = re.search(r"BF-RECEIPT-%s %s %s\\n' ([0-9a-f]{24}) ", command)
    if found is None:
        return None
    head = receipt_command("", found[1]).split("\n", 1)[0]
    start = command.find(head)
    return None if start < 0 else unwrap_receipt_command(command[start:])


_ROUTE_PREFIX = '_BENCHFLOW_IMAGE_PATH="${_BENCHFLOW_IMAGE_PATH-$PATH}"; '


def _root_route_line() -> str:
    """The line root's route puts before a probe since batch 32b round 3
    (lockdown._keep_route_path, 405f5e739): keep the kit's PATH, or pin the
    system one."""
    from benchflow.sandbox.lockdown import _keep_route_path

    return _keep_route_path("")


def inner_command(command: Any) -> Any:
    """The command a receipt script runs, or *command* when it is not one;
    either way without the line root's route pins PATH with first (C8)."""
    unwrapped = _unwrap(command)
    if unwrapped is not None:
        command = unwrapped[0]
    if isinstance(command, str) and command.startswith(_root_route_line()):
        return command[len(_root_route_line()) :]
    if unwrapped is not None:
        return command
    if isinstance(command, str) and command.startswith(_ROUTE_PREFIX):
        return command.split("\n", 1)[1] if "\n" in command else command
    return command


def _code(result: Any) -> int:
    for name in ("return_code", "exit_code"):
        value = getattr(result, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def receipt_answer(command: Any, result: Any) -> Any:
    """What the exec of *command* returns when its command answered *result*."""
    unwrapped = _unwrap(command)
    if unwrapped is None:
        return result
    if any(
        isinstance(getattr(result, n, None), bool) for n in ("return_code", "exit_code")
    ):
        # A backend that broke the exec contract: no receipt was printed.
        return result
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    # Mutable, as most fakes' own answers are: a test may still change the
    # exec's status after the fact (a timeout, a lost connection).
    return SimpleNamespace(
        stdout=encode_receipt(
            (stdout if isinstance(stdout, str) else "").encode(),
            _code(result),
            unwrapped[1],
        ),
        stderr=stderr if isinstance(stderr, str) else "",
        return_code=0,
    )


def receipt_aware(exec_fn: Any) -> Any:
    """Wrap a fake's ``exec`` (a method, an async or a plain function) so it
    answers the command a receipt wraps and prints that answer's receipt."""
    import functools
    import inspect

    params = list(inspect.signature(exec_fn).parameters)
    method = bool(params) and params[0] == "self"

    @functools.wraps(exec_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        args = list(args)
        at = 1 if method else 0
        if len(args) > at:
            wrapped = args[at]
            args[at] = inner_command(wrapped)
        else:
            wrapped = kwargs.get("command")
            kwargs["command"] = inner_command(wrapped)
        result = exec_fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return receipt_answer(wrapped, result)

    return wrapper


def receipt_mock(answer: Any) -> Any:
    """``AsyncMock(return_value=answer)`` that answers a receipt as one."""
    from unittest.mock import AsyncMock

    return AsyncMock(
        side_effect=lambda command, **kwargs: receipt_answer(command, answer)
    )


def receipt_mock_sequence(answers: list[Any]) -> Any:
    """``AsyncMock(side_effect=answers)`` that answers each receipt as one."""
    from unittest.mock import AsyncMock

    pending = iter(answers)
    return AsyncMock(
        side_effect=lambda command, **kwargs: receipt_answer(command, next(pending))
    )

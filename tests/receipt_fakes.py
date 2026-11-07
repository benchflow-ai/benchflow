"""A fake exec's answer to a command, as the receipt the harness reads.

Structured reads cross as a receipt (``benchflow.sandbox._exec_receipt``):
the command a fake sees is the receipt script around the read. A fake that
answers reads by their text answers the read inside, and this turns its
answer into the receipt line the script would have printed.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command


def receipt_answer(command: str, answer: Callable[[str], Any]) -> Any:
    """``answer(command)``; when *command* is a receipt script,
    ``answer(read)`` for the read inside it, printed as that receipt."""
    unwrapped = unwrap_receipt_command(command)
    if unwrapped is None:
        return answer(command)
    read, nonce = unwrapped
    result = answer(read)
    if inspect.isawaitable(result):
        pending = result

        async def receipt() -> Any:
            return _receipt(await pending, nonce)

        return receipt()
    return _receipt(result, nonce)


def _receipt(result: Any, nonce: str) -> Any:
    code = getattr(result, "return_code", None)
    if code is None:
        code = getattr(result, "exit_code", 0)
    stdout = getattr(result, "stdout", "") or ""
    return SimpleNamespace(
        stdout=encode_receipt(stdout.encode(), int(code), nonce),
        stderr=getattr(result, "stderr", "") or "",
        return_code=0,
    )

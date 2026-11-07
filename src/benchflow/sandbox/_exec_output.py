"""Keep machine output intact, and redact only the copy used for diagnostics.

Harness readers enter through ``exec_raw`` even when a read is batched, retried,
or passed through a protocol adapter. The task-local request reaches the actual
transport without adding provider-specific keywords to those adapters. There is
no retry through ordinary (possibly redacted) exec on an unsupported signature.
SDK transports already return raw streams; Docker also uses this request to
keep stderr out of stdout. Resetting the token preserves nested/concurrent calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_RAW_OUTPUT: ContextVar[bool] = ContextVar("benchflow_raw_exec_output", default=False)


def raw_output_requested() -> bool:
    return _RAW_OUTPUT.get()


async def exec_raw[Result](
    run: Callable[..., Awaitable[Result]], *args: Any, **kwargs: Any
) -> Result:
    """Execute once and return unmodified machine data on every backend.

    Use ``diagnostic_result`` before formatting, clipping, logging or showing
    this result. Transport exceptions remain diagnostics, including in raw mode.
    """
    token = _RAW_OUTPUT.set(True)
    try:
        return await run(*args, **kwargs)
    finally:
        _RAW_OUTPUT.reset(token)


@dataclass(frozen=True)
class DiagnosticText:
    """Presentation only: deliberately not a string or a parser input.

    Rendering is explicit and restricted to presentation sinks by the source
    contract. No string operations or implicit conversion can turn this copy
    back into machine data.
    """

    _text: str = field(repr=False)

    def render(self) -> str:
        return self._text

    def __str__(self) -> str:
        raise TypeError("Diagnostic text must be explicitly rendered for display")


@dataclass(frozen=True)
class DiagnosticResult:
    stdout: DiagnosticText
    stderr: DiagnosticText
    return_code: int


def diagnostic_result(result: Any) -> DiagnosticResult:
    """A separate, typed display copy; never input to a harness parser."""
    from benchflow._diagnostics import render_text
    from benchflow.sandbox._base import ExecResult

    if isinstance(result, DiagnosticResult):
        return result
    if isinstance(result, ExecResult) and result._diagnostic_result is not None:
        return result._diagnostic_result
    code = getattr(result, "return_code", -1)
    return DiagnosticResult(
        stdout=DiagnosticText(render_text(str(getattr(result, "stdout", "") or ""))),
        stderr=DiagnosticText(render_text(str(getattr(result, "stderr", "") or ""))),
        return_code=code if isinstance(code, int) else -1,
    )

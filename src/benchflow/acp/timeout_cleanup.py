"""Bounded cooperative cleanup for timed-out ACP prompts.

The original prompt task remains the sole ACP response reader. A best-effort
``session/cancel`` request runs independently while the prompt gets a short
grace period to receive peer-authored terminal usage and tool updates. At the
shared deadline, every surviving task is cancelled and boundedly drained.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

PROMPT_TIMEOUT_CLEANUP_TOTAL_SEC = 5.0
PROMPT_CANCEL_DRAIN_TIMEOUT_SEC = 0.25
#: How long the stop waits for its first step, the model gate (at most about
#: a second a proxy) or, on a route without one, the kill of the agent's
#: processes, before it goes on.
GATE_CLOSE_TIMEOUT_SEC = 20.0


def _consume_task_result(task: asyncio.Task) -> None:
    with contextlib.suppress(BaseException):
        task.result()


async def _cancel_and_drain_tasks(tasks: set[asyncio.Task], timeout: float) -> None:
    pending = {task for task in tasks if not task.done()}
    for task in pending:
        task.cancel()
    _done, pending = (
        await asyncio.wait(pending, timeout=timeout) if pending else (set(), set())
    )
    for task in tasks - pending:
        _consume_task_result(task)
    for task in pending:
        task.add_done_callback(_consume_task_result)


async def cancel_and_drain_prompt_task(prompt_task: asyncio.Task) -> bool:
    """Hard-cancel a prompt and return whether it completed within the drain."""
    if prompt_task.done():
        _consume_task_result(prompt_task)
        return True
    await _cancel_and_drain_tasks({prompt_task}, PROMPT_CANCEL_DRAIN_TIMEOUT_SEC)
    if prompt_task.done():
        return True

    logger.warning(
        "ACP prompt task did not finish within %.2fs after cancellation",
        PROMPT_CANCEL_DRAIN_TIMEOUT_SEC,
    )
    return False


async def close_gate_first(before_cancel: Callable[[], Awaitable[Any]] | None) -> None:
    """The stop's first step: close the trial's model gate, so no
    call the agent makes from here on, in answer to the stop or not, reaches
    a provider. Bounded, and never raises: the agent is stopped whatever the
    gate does."""
    if before_cancel is None:
        return
    try:
        await asyncio.wait_for(before_cancel(), GATE_CLOSE_TIMEOUT_SEC)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("The model gate did not close before the stop", exc_info=True)


async def cancel_prompt_after_timeout(
    acp_client: Any,
    prompt_task: asyncio.Task,
    *,
    before_cancel: Callable[[], Awaitable[Any]] | None = None,
) -> bool:
    """Close the model gate (*before_cancel*), request peer cancellation, then
    bound cleanup without adding a reader."""
    await close_gate_first(before_cancel)
    cancel = getattr(acp_client, "cancel", None)
    if not callable(cancel):
        return await cancel_and_drain_prompt_task(prompt_task)
    if prompt_task.done():
        _consume_task_result(prompt_task)
        return True

    async def _request_cancel() -> None:
        try:
            await cancel()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Failed to request ACP prompt cancellation", exc_info=True)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + PROMPT_TIMEOUT_CLEANUP_TOTAL_SEC
    cancel_task = asyncio.create_task(_request_cancel())
    try:
        cooperative_timeout = max(
            0.0,
            deadline - loop.time() - PROMPT_CANCEL_DRAIN_TIMEOUT_SEC,
        )
        await asyncio.wait({prompt_task}, timeout=cooperative_timeout)
    finally:
        remaining = max(0.0, deadline - loop.time())
        await _cancel_and_drain_tasks(
            {prompt_task, cancel_task},
            min(PROMPT_CANCEL_DRAIN_TIMEOUT_SEC, remaining),
        )
    return prompt_task.done()

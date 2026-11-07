"""Independent steps of one sandbox at once, failing as one after another would.

Before the verifier starts, BenchFlow reads several facts from the agent's
sandbox that need nothing from each other and change nothing there (the
image's BASH_ENV, the mount aliases of the blocked folders, the trusted PATH
and PYTHONPATH entries). Each read is its own round trip; run one after
another they add up on the path from the agent's end to the verifier's start
(PERF-G).

:func:`in_order` awaits such steps at once on a sandbox that can run several
execs side by side (``concurrent_steps``: Docker and remote Docker, whose
execs are separate ``docker exec`` clients or requests on the exec channel)
and one after another everywhere else, as before. Either way the caller gets
the results in the order given, and when steps fail it gets the first failure
in that order: the one running them one after another would have raised.
So pass only reads, or steps neither of which reads what another writes (the
tool-state move beside plugin discovery). When a step fails, a
change an earlier step made is still made, as one after another; a change a
later step made would not have been, so a step that changes something goes
before the steps that may fail beside it.

Each step is a callable that makes its awaitable, so a step that one after
another would never have started is never made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


def runs_steps_at_once(sandbox: Any) -> bool:
    """Whether *sandbox* takes several execs side by side."""
    return getattr(type(sandbox), "concurrent_steps", False) is True


async def in_order(sandbox: Any, *steps: Callable[[], Awaitable[Any]]) -> list[Any]:
    """Await *steps* (see the module docstring) and return their results."""
    if not runs_steps_at_once(sandbox):
        return [await step() for step in steps]
    results = await asyncio.gather(*(step() for step in steps), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return list(results)

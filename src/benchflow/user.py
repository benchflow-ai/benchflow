"""User abstraction for progressive-disclosure trial loops.

A User is a Python callback that participates in the trial loop alongside the
agent. Each round: user.run() produces a prompt → agent executes → verifier
checks → user sees the result and decides what to say next (or stops).

This is distinct from multi-role scenes (PR #179) where multiple ACP agents
collaborate via outbox files. The User runs in the scheduler process, not in
the sandbox.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class RoundResult:
    """Outcome of one agent round, passed to User.run() for the next decision."""

    round: int
    trajectory: list[dict] = field(default_factory=list)
    rewards: dict[str, Any] | None = None
    verifier_output: str | None = None
    verifier_error: str | None = None
    n_tool_calls: int = 0


class BaseUser:
    """Abstract user that drives a progressive-disclosure trial loop.

    Subclass and implement ``run()`` to control what prompt the agent sees
    each round. Return ``None`` from ``run()`` to stop the loop early.
    """

    async def setup(self, instruction: str, solution: str | None = None) -> None:
        """Called once before the first round.

        ``instruction`` is the original task instruction (from instruction.md).
        ``solution`` is the gold answer if oracle access is enabled, else None.
        """

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        """Produce the next prompt for the agent, or None to stop.

        ``round`` starts at 0. ``round_result`` is None on the first call
        and contains the previous round's outcome on subsequent calls.
        """
        raise NotImplementedError


class PassthroughUser(BaseUser):
    """Sends the original instruction unchanged. Single-round, backward compatible."""

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction
        return None


class FunctionUser(BaseUser):
    """Wraps a plain function as a User — for lightweight one-off use.

    The function signature matches ``BaseUser.run()``:
        fn(round, instruction, round_result) -> str | None

    Both sync and async functions are supported.
    """

    def __init__(
        self,
        fn: Callable[
            [int, str, RoundResult | None],
            str | None | Awaitable[str | None],
        ],
    ) -> None:
        self._fn = fn

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        result = self._fn(round, instruction, round_result)
        if inspect.isawaitable(result):
            return cast(str | None, await result)
        return cast(str | None, result)

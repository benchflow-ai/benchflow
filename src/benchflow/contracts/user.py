"""User-loop contracts for progressive-disclosure rollouts."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class RoundResult:
    """Outcome of one agent round, passed to ``BaseUser.run()``."""

    round: int
    trajectory: list[dict] = field(default_factory=list)
    rewards: dict[str, Any] | None = None
    verifier_output: str | None = None
    verifier_error: str | None = None
    n_tool_calls: int = 0


class BaseUser:
    """Abstract user that drives a progressive-disclosure rollout loop."""

    async def setup(self, instruction: str, solution: str | None = None) -> None:
        """Called once before the first round."""

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        """Produce the next prompt for the agent, or ``None`` to stop."""
        raise NotImplementedError


class PassthroughUser(BaseUser):
    """Sends the original instruction unchanged for one round."""

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
    """Wrap a sync or async function as a ``BaseUser``."""

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

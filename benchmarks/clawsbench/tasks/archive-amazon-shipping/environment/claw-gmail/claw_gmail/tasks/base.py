"""Base task class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Task(ABC):
    """Base class for all evaluation tasks."""

    name: str
    description: str
    instruction: str  # Natural language instruction for the agent
    category: str  # "capability" or "safety"
    scenario: str = "default"  # Which seed scenario to use
    points: float = 1.0
    tags: list[str] = field(default_factory=list)

    @abstractmethod
    def evaluate(
        self,
        final_state: dict,
        diff: dict,
        action_log: list[dict],
    ) -> tuple[float, bool]:
        """Evaluate task completion.

        Args:
            final_state: Full state dump after agent actions
            diff: Diff between initial and final state
            action_log: List of all API calls made

        Returns:
            (reward, done) — reward in [-1, 1], done flag
        """
        ...

    def get_initial_setup(self) -> dict | None:
        """Optional: return extra setup to apply before task starts."""
        return None

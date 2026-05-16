"""Adapter to expose BenchFlow tasks as Inspect AI tasks.

This module provides thin format converters — no ``inspect-ai`` dependency
is required.  The helpers translate BenchFlow's declarative ``Scene`` and
composable ``Rubric`` types into plain dicts that follow Inspect AI's
expected shapes so downstream code can feed them into the Inspect runtime
directly.

Extend by subclassing ``InspectAdapter`` and overriding the ``to_inspect_task``
method for custom field mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchflow._types import Scene
    from benchflow.rewards.rubric import Rubric


@dataclass
class InspectAdapter:
    """Wraps a BenchFlow Scene + Rubric as an Inspect AI-compatible task."""

    scene: Scene
    rubric: Rubric | None = None

    def to_inspect_task(self) -> dict:
        """Convert to Inspect AI task format.

        Returns a dict with:
        - ``name``:    scene name
        - ``dataset``: list of samples derived from scene turns
        - ``scorer``:  rubric metadata (present only when a rubric is set)
        """
        samples = []
        for turn in self.scene.turns:
            samples.append(
                {
                    "input": turn.prompt or "",
                    "role": turn.role,
                }
            )

        result: dict = {
            "name": self.scene.name,
            "dataset": samples,
        }

        if self.rubric:
            result["scorer"] = {
                "type": "benchflow_rubric",
                "reward_funcs": len(self.rubric.reward_funcs),
                "weights": self.rubric.weights,
            }

        return result


def to_inspect_task(scene: Scene, rubric: Rubric | None = None) -> dict:
    """Convenience function to convert a Scene to Inspect AI format."""
    return InspectAdapter(scene=scene, rubric=rubric).to_inspect_task()

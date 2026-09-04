"""Benchmark tasks over the Metal simulator (registered under ``inspect_robots.tasks``)."""

from __future__ import annotations

from inspect_robots.scene import Scene, Target
from inspect_robots.task import Task

SCORERS = ("success_at_end", "min_distance_to_goal", "episode_length")


def metal_sim_reach(num_scenes: int = 4, max_steps: int = 120) -> Task:
    """Reach a seeded red cube with the fingertips."""
    n, steps = int(num_scenes), int(max_steps)
    return Task(
        name="metal-sim-reach",
        scenes=[
            Scene(
                id=f"reach-{i}",
                instruction="Reach the red cube: bring the fingertips to the cube centre.",
                target=Target(kind="cube", spec={"goal": "reach"}),
                init_seed=i,
                metadata={"goal": "reach"},
            )
            for i in range(n)
        ],
        scorer=list(SCORERS),
        max_steps=steps,
    )


def metal_sim_pick(num_scenes: int = 4, max_steps: int = 200) -> Task:
    """Grasp the seeded cube and lift it off the table."""
    n, steps = int(num_scenes), int(max_steps)
    return Task(
        name="metal-sim-pick",
        scenes=[
            Scene(
                id=f"pick-{i}",
                instruction="Pick up the red cube: open the gripper, grasp the cube, and lift it at least 8 cm.",
                target=Target(kind="cube", spec={"goal": "pick"}),
                init_seed=i,
                metadata={"goal": "pick"},
            )
            for i in range(n)
        ],
        scorer=list(SCORERS),
        max_steps=steps,
    )

"""Kinematic simulator of the MakerMods Metal arm for Inspect Robots.

Registered components (see ``pyproject.toml``):

* embodiment ``metal_sim`` — the simulator, same 7-D joint-degree action space as the real
  ``metal_arm`` plugin, plus privileged success signals and a rendered ``front`` camera.
* policies ``metal_ik`` (damped-least-squares IK oracle) and ``metal_pyfile`` (loads a user
  ``act()`` from a Python file; the code-as-policy contract used by the BenchFlow tasks).
* tasks ``metal-sim-reach`` and ``metal-sim-pick``.
"""

from inspect_robots_metal_sim.embodiment import MetalSimEmbodiment
from inspect_robots_metal_sim.kinematics import MetalKinematics

__all__ = ["MetalKinematics", "MetalSimEmbodiment"]

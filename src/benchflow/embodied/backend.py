"""The simulator backend contract the episode server drives.

A benchmark implements `SimBackend` for each simulator family. Only the first six methods are required; the
server probes the optional ones with ``getattr``. Standard library only (numpy arrays pass through untyped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .spec import Embodiment


@dataclass
class StepResult:
    """Outcome of one simulator step.

    success: the task's success signal right after this step (backends that only judge at the end may return
    False here; the server then probes ``success()`` for the dense trace when the embodiment's reward is sparse).
    reward: the backend's shaped reward, or 0.0 when it has none.
    """

    success: bool
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SimBackend(Protocol):
    def embodiment(self) -> Embodiment: ...

    def reset(self, seed: int) -> None: ...

    def step(self, action: list[float]) -> StepResult: ...

    def observe(self) -> dict[str, Any]:
        """JSON-serialisable state (proprioception + object state). Visibility filtering is the server's job."""
        ...

    def render(self, camera: str | None = None) -> Any:
        """HxWx3 uint8 image from `camera` (None: the recording camera)."""
        ...

    def success(self) -> bool: ...


# Optional methods (duck-typed; not part of the runtime-checkable protocol):
#   ee_position(arm: str) -> sequence of 3 floats        built-in arm skills
#   ee_command(arm: str, delta: list[float], grip: float | None) -> list[float]
#                                                         flat action for an end-effector servo step; default:
#                                                         pack {arm.ee_delta: delta, gripper: grip} by the skill's binds
#   hold_action() -> list[float]                          the "hold still" action; default from the group hold policies
#   run_skill(name: str, args: list[str] | dict) -> dict  backend skills (impl: backend)
#   pop_frames() -> list[image]                           frames rendered while a backend skill ran
#   judge(outcome: str, text: str) -> bool                verdict that depends on how the episode ended
#   camera_calibration(name: str, width: int, height: int) -> dict | None
#   workspace_images() -> dict[str, image]                images given to the agent at the start
#   privileged_state() -> dict                            extra state only the oracle sees (merged into observe)
#   close() -> None

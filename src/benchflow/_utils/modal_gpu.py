"""Modal GPU request validation shared by task admission and both launchers.

Keep this module independent of the optional provider SDK: checking a task
must not import Modal or contact its control plane.
"""

from __future__ import annotations

import re

_GPU_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*[!+]?")


class ModalGPURequestError(ValueError):
    """An accelerator request cannot be represented by a Modal Sandbox."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        super().__init__(reason)


def modal_gpu_request(gpus: int, gpu_types: list[str] | None) -> str | None:
    """Return Modal's single ``TYPE:COUNT`` request, or ``None`` for CPU.

    Sandbox.create accepts one type, unlike Modal Functions' fallback lists.
    Provider availability and supported hardware/count combinations are still
    checked by Modal; this validates the task's representation without guessing
    a changing hardware catalog or silently dropping acceptable types.
    """
    if type(gpus) is not int or gpus < 0:
        raise ModalGPURequestError(
            "sandbox.gpus", "GPU count must be a non-negative integer"
        )
    if gpu_types is not None and not isinstance(gpu_types, list):
        raise ModalGPURequestError(
            "sandbox.gpu_types", "GPU types must be a list of type names"
        )
    if not gpus:
        if gpu_types:
            raise ModalGPURequestError(
                "sandbox.gpu_types",
                "GPU types require sandbox.gpus greater than zero",
            )
        return None
    if not gpu_types:
        return f"any:{gpus}"
    if len(gpu_types) != 1:
        raise ModalGPURequestError(
            "sandbox.gpu_types",
            "Modal Sandboxes accept one GPU type; choose a single gpu_types entry",
        )
    gpu_type = gpu_types[0]
    if not isinstance(gpu_type, str) or not _GPU_TYPE.fullmatch(gpu_type):
        raise ModalGPURequestError(
            "sandbox.gpu_types",
            "Use one GPU type name without a count or whitespace; "
            "set the count in sandbox.gpus",
        )
    return f"{gpu_type}:{gpus}"

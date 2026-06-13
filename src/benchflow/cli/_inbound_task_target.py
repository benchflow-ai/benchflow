"""Compatibility imports for CLI native task targets."""

from __future__ import annotations

from benchflow.adapters.task_target import (
    InboundTaskMaterializer,
    NativeTaskTarget,
    native_task_target,
)

__all__ = ["InboundTaskMaterializer", "NativeTaskTarget", "native_task_target"]

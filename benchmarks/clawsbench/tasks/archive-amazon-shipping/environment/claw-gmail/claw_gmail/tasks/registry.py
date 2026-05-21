"""Task registry."""

from __future__ import annotations

from .base import Task

_REGISTRY: dict[str, Task] = {}


def register_task(task: Task):
    """Register a task instance."""
    _REGISTRY[task.name] = task


def get_task(name: str) -> Task | None:
    """Get a task by name."""
    return _REGISTRY.get(name)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(_REGISTRY.keys())

"""Task and evaluation framework."""

from __future__ import annotations

from .base import Task
from .registry import get_task, list_tasks, register_task

# Import demo tasks to trigger registration
from . import demo as _demo_tasks  # noqa: F401

__all__ = ["Task", "get_task", "list_tasks", "register_task"]

"""Small helpers shared by benchflow.taskmd's modules."""

from __future__ import annotations

from typing import Any, cast


def table(value: object) -> dict[str, Any]:
    """``value`` when it is a table (a dict), else an empty one."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def listed(value: object) -> list[Any]:
    """``value`` when it is a list, else an empty one."""
    return cast("list[Any]", value) if isinstance(value, list) else []

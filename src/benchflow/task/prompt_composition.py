"""Prompt composition for task-standard ``task.md`` documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

PromptPart = Literal["base", "role", "scene", "turn"]
CompositionMode = Literal["append", "replace"]

DEFAULT_PROMPT_ORDER: tuple[PromptPart, ...] = ("base", "role", "scene", "turn")
_LEGACY_FALLBACK_ORDER: tuple[PromptPart, ...] = ("turn", "scene", "role", "base")
_VALID_PARTS = frozenset(DEFAULT_PROMPT_ORDER)


@dataclass(frozen=True)
class PromptCompositionSettings:
    """Resolved ``benchflow.prompt`` settings for a task document."""

    composition: CompositionMode | None = None
    order: tuple[PromptPart, ...] = DEFAULT_PROMPT_ORDER


def prompt_composition_settings(benchflow: dict[str, Any]) -> PromptCompositionSettings:
    """Read ``benchflow.prompt`` composition settings from a benchflow block."""

    raw = benchflow.get("prompt")
    if raw is None:
        return PromptCompositionSettings()
    if not isinstance(raw, dict):
        raise ValueError("benchflow.prompt must be a mapping")

    composition = raw.get("composition")
    if composition is not None and composition not in ("append", "replace"):
        raise ValueError(
            "benchflow.prompt.composition must be 'append' or 'replace'"
        )

    order_raw = raw.get("order")
    if order_raw is None:
        order = DEFAULT_PROMPT_ORDER
    else:
        if not isinstance(order_raw, list) or not order_raw:
            raise ValueError("benchflow.prompt.order must be a non-empty list")
        order_parts: list[PromptPart] = []
        for index, item in enumerate(order_raw):
            if not isinstance(item, str) or item not in _VALID_PARTS:
                raise ValueError(
                    f"benchflow.prompt.order[{index}] must be one of "
                    f"base, role, scene, turn"
                )
            order_parts.append(item)
        order = tuple(order_parts)

    return PromptCompositionSettings(
        composition=composition,
        order=order,
    )


def _normalize_optional_part(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _parts_by_name(
    base: str | None,
    role: str | None,
    scene: str | None,
    turn: str | None,
    *,
    explicit_turn: bool,
) -> dict[PromptPart, str | None]:
    if explicit_turn:
        turn_value = turn.strip() if isinstance(turn, str) else ""
        return {
            "base": _normalize_optional_part(base),
            "role": _normalize_optional_part(role),
            "scene": _normalize_optional_part(scene),
            "turn": turn_value if turn_value else None,
        }
    return {
        "base": _normalize_optional_part(base),
        "role": _normalize_optional_part(role),
        "scene": _normalize_optional_part(scene),
        "turn": _normalize_optional_part(turn),
    }


def compose_task_prompt(
    base: str | None,
    role: str | None,
    scene: str | None,
    turn: str | None,
    *,
    composition: CompositionMode | None = None,
    order: Sequence[PromptPart] | None = None,
    explicit_turn: bool = False,
) -> str:
    """Compose prompt parts using legacy fallback or explicit composition rules.

    When *composition* is ``None``, use legacy fallback precedence:
    turn > scene > role > base.

    ``append`` joins non-empty parts in *order* with double newlines.

    ``replace`` returns the highest-priority non-empty part using reverse
    *order* (turn wins by default). When *explicit_turn* is true, only the
    inline turn prompt is used.
    """

    resolved_order = tuple(order) if order is not None else DEFAULT_PROMPT_ORDER
    parts = _parts_by_name(base, role, scene, turn, explicit_turn=explicit_turn)

    if composition is None:
        for part_name in _LEGACY_FALLBACK_ORDER:
            value = parts[part_name]
            if value:
                return value
        return ""

    if composition == "append":
        chunks = [parts[part_name] for part_name in resolved_order if parts[part_name]]
        return "\n\n".join(chunks)

    if explicit_turn:
        return parts["turn"] or ""

    for part_name in reversed(resolved_order):
        value = parts[part_name]
        if value:
            return value
    return ""


__all__ = [
    "CompositionMode",
    "DEFAULT_PROMPT_ORDER",
    "PromptCompositionSettings",
    "PromptPart",
    "compose_task_prompt",
    "prompt_composition_settings",
]

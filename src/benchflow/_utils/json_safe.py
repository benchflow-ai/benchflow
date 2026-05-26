"""Shared helpers for emitting strictly-valid JSON.

Plain ``json.dumps`` emits non-finite floats as the bare tokens ``NaN``,
``Infinity``, ``-Infinity`` — valid Python but rejected by strict JSON
parsers (jq, serde, Node ``JSON.parse``). Both the trajectory exporter
(``benchflow/trajectories/export.py``) and the skill-eval GEPA exporter
(``benchflow/skill_eval.py``) need to scrub these values before writing
artifacts that downstream trainers or review tools consume.

This module is the single canonical home for that scrub + dump logic so
the two callers stay in sync. See issue #426.
"""

from __future__ import annotations

import json
import math
from typing import Any


def scrub_non_finite(value: Any) -> Any:
    """Replace ``NaN`` / ``±Infinity`` floats with ``None`` recursively.

    Walks dicts, lists, and tuples; leaves all other values untouched.
    Returned tuples become lists so the output is JSON-shape-stable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: scrub_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_non_finite(v) for v in value]
    return value


def dumps_finite(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` with NaN/Infinity scrubbed AND ``allow_nan=False``.

    The scrub pass turns non-finite floats into ``null`` so downstream
    parsers succeed; ``allow_nan=False`` is defense-in-depth that turns
    any future regression into a loud ``ValueError`` at write time rather
    than silently producing invalid JSON.
    """
    return json.dumps(scrub_non_finite(obj), allow_nan=False, **kwargs)

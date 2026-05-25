"""JSON serialization helpers shared by artifact exporters."""

from __future__ import annotations

import json
import math
from typing import Any


def scrub_non_finite(value: Any) -> Any:
    """Replace ``NaN`` / ``Infinity`` floats with ``None`` recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: scrub_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_non_finite(v) for v in value]
    return value


def json_safe_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize as strict JSON after normalizing non-finite floats to ``null``."""
    kwargs["allow_nan"] = False
    return json.dumps(scrub_non_finite(obj), **kwargs)

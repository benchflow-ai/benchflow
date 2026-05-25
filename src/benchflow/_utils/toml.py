"""Small TOML serialization helpers."""

from __future__ import annotations

import tomli_w


def toml_quote(value: str) -> str:
    """Return ``value`` as a TOML string literal using the canonical writer."""
    if not isinstance(value, str):
        raise TypeError(f"toml_quote expected str, got {type(value).__name__}")
    return tomli_w.dumps({"value": value}).split(" = ", 1)[1].strip()

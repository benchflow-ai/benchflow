"""Read back what the shared env wrappers set (``benchflow.sandbox._base``)."""

from __future__ import annotations

import re

# The inline form (Daytona): the shell's own printf of octal escapes, eval'd.
INLINE_START = "set -- benchflow-env-pending; eval \"$(printf '"
_PRINTF = re.compile(r"eval \"\$\(printf '((?:\\[0-7]{3})*)'\)\"")


def wrapped_env_body(wrapped: str) -> str:
    """The ``export`` lines the inline wrapper's ``printf`` produces in the sandbox."""
    match = _PRINTF.search(wrapped)
    assert match is not None, wrapped[:200]
    octal = match.group(1).split("\\")[1:]
    return bytes(int(code, 8) for code in octal).decode()

"""Environment variable utilities — internalized from Harbor."""

from __future__ import annotations

import os
import re

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Resolve environment variable templates in a dictionary.

    Templates like ``${VAR_NAME}`` are replaced with values from ``os.environ``.
    Use ``${VAR_NAME:-default}`` to provide a default when the variable is unset.
    Literal values are passed through unchanged.

    Raises:
        ValueError: If a required environment variable is not found and no default.
    """
    resolved = {}
    for key, value in env_dict.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                resolved[key] = os.environ[var_name]
            elif default is not None:
                resolved[key] = default
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not found in host environment"
                )
        else:
            resolved[key] = value
    return resolved

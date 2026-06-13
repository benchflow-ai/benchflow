"""Environment variable utilities."""

from __future__ import annotations

import os
import re

from benchflow._dotenv import load_dotenv_env

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Resolve environment variable templates in a dictionary.

    Templates like ``${VAR_NAME}`` are replaced with values from ``os.environ``
    or from BenchFlow's local `.env` fallback.
    Use ``${VAR_NAME:-default}`` to provide a default when the variable is unset.
    Literal values are passed through unchanged.

    Raises:
        ValueError: If a required environment variable is not found and no default.
    """
    source_env = {**load_dotenv_env(), **os.environ}
    resolved = {}
    for key, value in env_dict.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in source_env:
                resolved[key] = source_env[var_name]
            elif default is not None:
                resolved[key] = default
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not found in host environment or .env"
                )
        else:
            resolved[key] = value
    return resolved

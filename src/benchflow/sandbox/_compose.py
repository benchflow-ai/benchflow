"""Shared docker-compose assets used by both the docker and daytona adapters.

Daytona's DinD mode wraps docker compose inside a Daytona sandbox, so both
adapters legitimately need the same compose file paths and the same
env-var template resolver. Keeping these in a neutral module (rather than
one adapter importing from the other) satisfies the sibling-independence
rule enforced by Item 7's import-linter.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Resolve ${VAR} / ${VAR:-default} template substitutions against os.environ."""
    resolved: dict[str, str] = {}
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

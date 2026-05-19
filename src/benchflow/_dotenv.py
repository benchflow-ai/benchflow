"""Small `.env` reader shared by CLI/runtime env resolution."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DOTENV_PATH = Path(".env")
_DOTENV_PATH_ENV = "BENCHFLOW_DOTENV_PATH"


def load_dotenv_env(path: str | Path | None = None) -> dict[str, str]:
    """Read a local `.env` file into a plain dict.

    Missing files are treated as empty input. `BENCHFLOW_DOTENV_PATH` lets tests
    or callers override the implicit `.env` lookup without changing cwd.
    """
    if path is not None:
        dotenv_path = Path(path)
    else:
        dotenv_path = Path(os.environ.get(_DOTENV_PATH_ENV, _DEFAULT_DOTENV_PATH))
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        parsed[key] = value
    return parsed

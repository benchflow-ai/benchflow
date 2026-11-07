"""Input validation for the keyless operator-script runner."""

from pathlib import Path

from benchflow.agents.registry import is_operator_script

MAX_SCRIPT_BYTES = 16 * 1024 * 1024


def read_script(spec: str) -> bytes:
    """Read a regular file with Rust's 16 MiB cap; never open a FIFO."""
    path = Path(spec.removeprefix("script:")).expanduser()
    if not spec.removeprefix("script:") or not path.is_file():
        raise ValueError("script: requires a regular file; use --agent script:wrong.sh")
    with path.open("rb") as source:
        data = source.read(MAX_SCRIPT_BYTES + 1)
    if len(data) > MAX_SCRIPT_BYTES:
        raise ValueError("script: file is larger than 16 MiB; use a smaller bash file")
    return data


def normalize_script(spec: str) -> str:
    """Keep the absolute host path in configs, including paths with slashes."""
    read_script(spec)
    return f"script:{Path(spec.removeprefix('script:')).expanduser().absolute()}"


def script_label(spec: str) -> str:
    """Trial records identify the script without publishing its host path."""
    if is_operator_script(spec):
        return f"script:{Path(spec.removeprefix('script:')).name}"
    return spec


def validate_script_user(user: str | None) -> None:
    """A script is an agent, never the root solution runner."""
    if (
        not user
        or not user.strip()
        or user.strip().lower() in {"root", "0", "none", "null"}
    ):
        raise ValueError("script: requires a non-root --sandbox-user; use agent")

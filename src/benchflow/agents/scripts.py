"""Input validation for the keyless operator-script runner."""

from pathlib import Path

from benchflow.agents.registry import is_operator_script
from benchflow.errors import UserError

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


class ScriptBackendError(ValueError, UserError):
    """A ``script:`` agent on a backend without the trusted root runner its
    script needs. Nothing was started."""

    exit_code = 2
    hint = (
        "use --sandbox docker, or run the logic as an ACP agent "
        "(docs/agents/custom-agent.md)"
    )


def refuse_script_backend(agent: str | None, environment: str) -> None:
    """Refuse an operator script on a built-in backend without the runner
    (``runs_operator_scripts``), before any sandbox or provider call.

    Every rollout's setup calls it first (``validate_agent_transport``), and so
    do the plans that make a provider call before any setup: ``bench run``'s
    flags, a ``--config`` file and ``bench branch``. Their dry runs and runs
    give the same text, exit 2. A backend of the caller's own planes is left
    to its sandbox's own runner check."""
    if not is_operator_script(agent):
        return
    from benchflow._utils.sandbox_providers import (
        OPERATOR_SCRIPT_PROVIDERS,
        SANDBOX_PROVIDERS,
    )

    if (
        environment in SANDBOX_PROVIDERS
        and environment not in OPERATOR_SCRIPT_PROVIDERS
    ):
        raise ScriptBackendError(
            "script: runs only on "
            + ", ".join(n for n in SANDBOX_PROVIDERS if n in OPERATOR_SCRIPT_PROVIDERS)
            + f": its runner needs a trusted root transport the {environment} "
            "backend does not have"
        )

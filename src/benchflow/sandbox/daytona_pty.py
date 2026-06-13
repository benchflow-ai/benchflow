"""Command-wrapping and exec helpers for the Daytona backend.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam: the small,
SDK-free helpers that shape what gets run on a Daytona sandbox — the secret-safe
env-file command wrapper, exec-failure output formatting, the single-container
service guard, and the API-key preflight. The names here are re-exported from
``benchflow.sandbox.daytona`` so existing imports such as
``from benchflow.sandbox.daytona import _wrap_daytona_command_with_env_file``
keep working unchanged.
"""

from __future__ import annotations

import os

from benchflow.sandbox._base import ExecResult, wrap_command_with_env_file

# Prefix for the decoded env file inside the Daytona sandbox. A unique 16-hex
# suffix is appended by the shared wrapper so concurrent exec() calls can't
# clobber each other's env file.
_DAYTONA_ENV_FILE_PREFIX = "/tmp/.benchflow_daytona_env_"


def _wrap_daytona_command_with_env_file(env: dict[str, str], command: str) -> str:
    """Return *command* prefixed to materialize *env* from a file.

    Thin wrapper over the canonical
    :func:`benchflow.sandbox._base.wrap_command_with_env_file` so the
    secret-redaction logic lives in exactly one place (shared with the Docker
    backend). See that function for the full contract: secrets never reach the
    remote process argv (visible via ``ps``, Daytona audit logs, or any
    provider-side command logging) — they are base64-encoded into the command
    string, decoded to a mode-0600 file inside the sandbox, sourced, and
    unconditionally removed via ``trap ... EXIT``.

    Issue #412: previously this used ``env K=V ...`` argv, which placed raw
    secret values into the remote command line.
    """
    return wrap_command_with_env_file(
        env, command, env_path_prefix=_DAYTONA_ENV_FILE_PREFIX
    )


def _exec_failure_output(result: ExecResult) -> str:
    output = " ".join(
        text.strip()
        for text in (result.stdout or "", result.stderr or "")
        if text and text.strip()
    )
    return output[:4000]


def _reject_non_main_service(service: str) -> None:
    """Raise ``ValueError`` for a non-``main`` service on the direct strategy.

    The direct (single-container) Daytona sandbox cannot target additional
    compose services; multi-container (vulhub-style) tasks require a
    ``docker-compose.yaml`` (#248). Centralizes the identical guard that
    ``_DaytonaDirect.exec``/``upload_dir``/``download_dir`` each raised inline.
    """
    if service != "main":
        raise ValueError(
            f"Direct (non-compose) Daytona sandbox is single-container "
            f"and cannot target service {service!r}. Multi-container "
            "(vulhub-style) tasks require a docker-compose.yaml (#248)."
        )


def _daytona_preflight() -> None:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit(
            "Daytona requires DAYTONA_API_KEY to be set. "
            "Please set this environment variable and try again."
        )
